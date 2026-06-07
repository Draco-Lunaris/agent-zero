from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import warnings
from typing import Any

import aiohttp

from helpers import files, plugins
from helpers.notification import (
    NotificationManager,
    NotificationPriority,
    NotificationType,
)
from helpers.print_style import PrintStyle
from plugins._whisper_stt.helpers import migration


warnings.filterwarnings("ignore", category=FutureWarning)


PLUGIN_NAME = "_whisper_stt"
DEFAULT_CONFIG = {
    "model_size": "base",
    "language": "en",
    "message_mode": "send",
    "silence_threshold": 0.3,
    "silence_duration": 1000,
    "waiting_timeout": 2000,
    "remote_url": "",
}
VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large", "turbo"}
VALID_MESSAGE_MODES = {"send", "draft"}

_model = None
_model_name = ""
is_updating_model = False


def normalize_config(config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(DEFAULT_CONFIG)
    if not isinstance(config, dict):
        return normalized

    model_size = str(config.get("model_size", normalized["model_size"]) or "").strip()
    if model_size in VALID_MODEL_SIZES:
        normalized["model_size"] = model_size

    language = str(config.get("language", normalized["language"]) or "").strip()
    if language:
        normalized["language"] = language

    message_mode = (
        str(config.get("message_mode", normalized["message_mode"]) or "")
        .strip()
        .lower()
    )
    if message_mode in VALID_MESSAGE_MODES:
        normalized["message_mode"] = message_mode

    try:
        silence_threshold = float(
            config.get("silence_threshold", normalized["silence_threshold"])
    )
        normalized["silence_threshold"] = min(max(silence_threshold, 0.0), 1.0)
    except (TypeError, ValueError):
        pass

    try:
        silence_duration = int(
            config.get("silence_duration", normalized["silence_duration"])
        )
        if silence_duration > 0:
            normalized["silence_duration"] = silence_duration
    except (TypeError, ValueError):
        pass

    try:
        waiting_timeout = int(config.get("waiting_timeout", normalized["waiting_timeout"]))
        if waiting_timeout > 0:
            normalized["waiting_timeout"] = waiting_timeout
    except (TypeError, ValueError):
        pass

    remote_url = str(config.get("remote_url", normalized["remote_url"]) or "").strip()
    if remote_url:
        normalized["remote_url"] = remote_url.rstrip("/")

    return normalized


def get_config() -> dict[str, Any]:
    migration.ensure_config_seeded()
    config = plugins.get_plugin_config(PLUGIN_NAME) or {}
    return normalize_config(config)


def get_loaded_model_name() -> str:
    return _model_name


def is_globally_enabled() -> bool:
    return plugins.determined_toggle_from_paths(
        True, reversed(plugins.get_plugin_roots(PLUGIN_NAME))
    )


async def preload(model_name: str | None = None):
    cfg = get_config()
    resolved_model = str(model_name or cfg["model_size"])
    return await _preload(resolved_model)


async def _preload(model_name: str):
    global _model, _model_name, is_updating_model

    while is_updating_model:
        await asyncio.sleep(0.1)

    try:
        is_updating_model = True
        if not _model or _model_name != model_name:
            NotificationManager.send_notification(
                NotificationType.INFO,
                NotificationPriority.NORMAL,
                "Loading Whisper model...",
                display_time=99,
                group="whisper-preload",
            )
            PrintStyle.standard(f"Loading Whisper model: {model_name}")
            _model = whisper.load_model(
                name=model_name,
                download_root=files.get_abs_path("/tmp/models/whisper"),
            )
            _model_name = model_name
            NotificationManager.send_notification(
                NotificationType.INFO,
                NotificationPriority.NORMAL,
                "Whisper model loaded.",
                display_time=2,
                group="whisper-preload",
            )
    finally:
        is_updating_model = False


async def is_downloading() -> bool:
    return is_updating_model


async def is_downloaded() -> bool:
    return _model is not None


async def is_remote_healthy() -> tuple[bool, str]:
    """Check if the remote Whisper API server is reachable.

    Returns (healthy, error_message). If no remote_url is configured,
    returns (False, "Not configured").
    """
    cfg = get_config()
    remote_url = cfg.get("remote_url", "")
    if not remote_url:
        return False, "Not configured"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{remote_url}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    return True, ""
                return False, f"HTTP {resp.status}"
    except Exception as e:
        return False, str(e)


async def transcribe(
    audio_bytes_b64: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    cfg = normalize_config(config or get_config())
    remote_url = str(cfg.get("remote_url", ""))

    if remote_url:
        return await _transcribe_remote(
            audio_bytes_b64,
            remote_url=remote_url,
            language=_resolve_language(str(cfg["language"])),
        )

    return await _transcribe_local(
        str(cfg["model_size"]),
        audio_bytes_b64,
        language=_resolve_language(str(cfg["language"])),
    )


async def _transcribe_remote(
    audio_bytes_b64: str,
    *,
    remote_url: str,
    language: str | None = None,
) -> dict[str, Any]:
    """Transcribe audio using a remote Whisper-compatible API server.

    Uses the OpenAI-compatible /v1/audio/transcriptions endpoint.
    """
    audio_bytes = base64.b64decode(audio_bytes_b64)

    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field(
                "file",
                audio_bytes,
                filename="audio.wav",
                content_type="audio/wav",
            )
            if language:
                data.add_field("language", language)

            async with session.post(
                f"{remote_url}/v1/audio/transcriptions",
                data=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result if isinstance(result, dict) else {"text": str(result)}
    except Exception as e:
        PrintStyle.error(f"Error in remote Whisper STT transcription: {e}")
        raise


async def _transcribe_local(
    model_name: str, audio_bytes_b64: str, *, language: str | None = None
) -> dict[str, Any]:
    await _preload(model_name)

    audio_bytes = base64.b64decode(audio_bytes_b64)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(audio_bytes)
        temp_path = audio_file.name

    try:
        kwargs: dict[str, Any] = {"fp16": False}
        if language:
            kwargs["language"] = language

        result = _model.transcribe(temp_path, **kwargs)  # type: ignore[union-attr]
        return result if isinstance(result, dict) else {}
    except Exception as e:
        PrintStyle.error(f"Error in local Whisper STT transcription: {e}")
        raise
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def _resolve_language(language: str) -> str | None:
    value = language.strip().lower()
    if not value or value == "auto":
        return None
    return value
