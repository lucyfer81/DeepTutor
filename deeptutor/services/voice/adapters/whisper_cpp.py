"""Local whisper.cpp CLI speech-to-text adapter.

Runs the machine-local ``whisper-cli`` binary (ggml Whisper) as a subprocess;
audio never leaves the host and no API key is required. The catalog ``model``
field names a whisper.cpp ggml ``*.bin`` file — a full path, or a bare name
resolved inside the whisper.cpp models directory. ``DEEPTUTOR_WHISPER_CLI``
and ``DEEPTUTOR_WHISPER_CPP_MODELS`` override the binary and models directory.

whisper-cli prints transcripts on stdout and diagnostics on stderr, so the
adapter captures pipes and parses stdout: ``-nt`` yields plain text lines,
while the default layout yields ``[HH:MM:SS.mmm --> HH:MM:SS.mmm]  text``
segments that become timed :class:`TranscriptCue` rows.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile

from deeptutor.services.voice.audio import normalize_wav
from deeptutor.services.voice.base import (
    BaseSTTAdapter,
    TranscriptCue,
    VoiceProviderError,
)
from deeptutor.services.voice.config import STTConfig

logger = logging.getLogger(__name__)

ENV_WHISPER_CLI = "DEEPTUTOR_WHISPER_CLI"
ENV_WHISPER_MODELS_DIR = "DEEPTUTOR_WHISPER_CPP_MODELS"

_DEFAULT_BINARY = "whisper-cli"
_FALLBACK_BINARIES = (
    "~/.local/bin/whisper-cli",
    "~/tools/whisper.cpp/build/bin/whisper-cli",
)
_FALLBACK_MODEL_DIRS = (
    "~/models/whisper.cpp",
    "~/models/whisper-cpp",
    "~/models/whisper",
    "~/tools/whisper.cpp/models",
)

# whisper-cli segment line: "[00:00:00.000 --> 00:00:08.000]   text".
# Hours are optional so builds printing "MM:SS.mmm" still parse.
_CUE_LINE = re.compile(
    r"^\[(?:(\d+):)?(\d{1,2}):(\d{2})\.(\d{3}) --> (?:(\d+):)?(\d{1,2}):(\d{2})\.(\d{3})\]\s*(.*)$"
)
# Decoded special tokens such as "[BLANK_AUDIO]" or "[MUSIC]" are not speech.
_BRACKET_ONLY = re.compile(r"^\[[A-Z][A-Z_ ]*\]$")

# whisper.cpp inference saturates every core; serialize concurrent clips so
# parallel voice messages queue instead of thrashing the CPU.
_WHISPER_LOCK = asyncio.Lock()


def resolve_cli_binary() -> str:
    """Locate the whisper.cpp CLI: env override, PATH, then known build paths."""
    override = (os.environ.get(ENV_WHISPER_CLI) or "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise VoiceProviderError(
            f"{ENV_WHISPER_CLI} points to a missing binary: {override}"
        )
    found = shutil.which(_DEFAULT_BINARY)
    if found:
        return found
    for raw in _FALLBACK_BINARIES:
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return str(candidate)
    raise VoiceProviderError(
        "Local Whisper transcription needs the whisper.cpp 'whisper-cli' binary. "
        f"Build whisper.cpp and keep whisper-cli on PATH, or set {ENV_WHISPER_CLI} "
        "to the binary path."
    )


def model_search_dirs() -> list[Path]:
    """Directories scanned for bare model names, env override first."""
    dirs: list[Path] = []
    override = (os.environ.get(ENV_WHISPER_MODELS_DIR) or "").strip()
    if override:
        dirs.append(Path(override).expanduser())
    dirs.extend(Path(raw).expanduser() for raw in _FALLBACK_MODEL_DIRS)
    unique: list[Path] = []
    for directory in dirs:
        if directory not in unique:
            unique.append(directory)
    return unique


def resolve_model_path(model: str) -> str:
    """Map the catalog model field onto a local ggml ``*.bin`` file.

    Accepts a full path, or a bare name resolved as ``<name>``,
    ``<name>.bin`` and ``ggml-<name>.bin`` inside the models directories.
    """
    value = (model or "").strip()
    if not value:
        raise VoiceProviderError(
            "Local Whisper needs a model. Set the model field to the full path of "
            "a whisper.cpp ggml *.bin file, or a model name such as 'small'."
        )
    direct = Path(value).expanduser()
    if direct.is_file():
        return str(direct)
    if direct.name != value:
        # A path with a directory part was given but does not exist.
        raise VoiceProviderError(f"Local Whisper model file not found: {value}")
    checked: list[str] = []
    for directory in model_search_dirs():
        if not directory.is_dir():
            continue
        for name in (value, f"{value}.bin", f"ggml-{value}.bin"):
            candidate = directory / name
            checked.append(str(candidate))
            if candidate.is_file():
                return str(candidate)
    hint = ", ".join(str(directory) for directory in model_search_dirs())
    raise VoiceProviderError(
        f"Local Whisper model '{value}' was not found. Point the model field at a "
        f"whisper.cpp ggml *.bin file, or place one under: {hint} "
        f"(set {ENV_WHISPER_MODELS_DIR} to override). Download models from "
        "https://huggingface.co/ggml-org or the whisper.cpp models script."
    )


def _resolve_language(config: STTConfig) -> str:
    """whisper-cli defaults to English; empty config means auto-detect."""
    language = (config.language or "").strip().lower()
    return language or "auto"


class WhisperCppSTTAdapter(BaseSTTAdapter):
    """Transcribe with the local whisper.cpp CLI (offline, no API key)."""

    async def transcribe(
        self,
        audio: bytes,
        config: STTConfig,
        *,
        filename: str = "audio.webm",
        content_type: str = "application/octet-stream",
    ) -> str:
        stdout = await self._run_whisper(
            audio, config, timestamps=False, filename=filename, content_type=content_type
        )
        return _plain_text(stdout)

    async def transcribe_cues(
        self,
        audio: bytes,
        config: STTConfig,
        *,
        filename: str = "audio.webm",
        content_type: str = "application/octet-stream",
    ) -> list[TranscriptCue]:
        stdout = await self._run_whisper(
            audio, config, timestamps=True, filename=filename, content_type=content_type
        )
        cues = _parse_cli_cues(stdout)
        if cues:
            return cues
        # Non-speech clips can come back without segments; degrade to the
        # base-class contract instead of losing a plain transcript.
        text = _plain_text(stdout)
        return [TranscriptCue(0.0, 0.0, text, timed=False)] if text else []

    async def _run_whisper(
        self,
        audio: bytes,
        config: STTConfig,
        *,
        timestamps: bool,
        filename: str,
        content_type: str,
    ) -> str:
        if not audio:
            raise VoiceProviderError("No audio data to transcribe.")
        wav_audio = await normalize_wav(audio)
        if not wav_audio:
            raise VoiceProviderError("Audio conversion returned an empty file.")
        binary = resolve_cli_binary()
        model_path = resolve_model_path(config.model)
        language = _resolve_language(config)

        with tempfile.TemporaryDirectory(prefix="deeptutor-whisper-stt-") as directory:
            source = Path(directory) / "clip.wav"
            source.write_bytes(wav_audio)
            argv = [
                binary,
                "-m",
                model_path,
                "-f",
                str(source),
                "-l",
                language,
                # Results only on stdout; diagnostics stay on stderr.
                "-np",
            ]
            if not timestamps:
                argv.append("-nt")

            async with _WHISPER_LOCK:
                try:
                    process = await asyncio.create_subprocess_exec(
                        *argv,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except OSError as exc:
                    raise VoiceProviderError(
                        f"Could not start the whisper.cpp CLI at {binary}: {exc}"
                    ) from exc
                try:
                    async with asyncio.timeout(config.request_timeout):
                        stdout, stderr = await process.communicate()
                except TimeoutError:
                    if process.returncode is None:
                        process.kill()
                    await process.wait()
                    raise VoiceProviderError(
                        "Local Whisper transcription timed out after "
                        f"{config.request_timeout} seconds. Increase Request timeout "
                        "in Settings > Voice, trim the clip, or use a smaller model."
                    ) from None
                except asyncio.CancelledError:
                    if process.returncode is None:
                        process.kill()
                    await process.wait()
                    raise

            if process.returncode != 0:
                detail = stderr.decode("utf-8", errors="replace").strip()[:400]
                raise VoiceProviderError(
                    "Local Whisper transcription failed"
                    + (f": {detail}" if detail else f" (exit code {process.returncode}).")
                )
            text = stdout.decode("utf-8", errors="replace")
            logger.debug(
                "whisper.cpp transcribe model=%s lang=%s bytes=%d timestamps=%s chars=%d",
                Path(model_path).name,
                language,
                len(audio),
                timestamps,
                len(text.strip()),
            )
            return text


def _parse_cli_cues(stdout: str) -> list[TranscriptCue]:
    """Turn whisper-cli timestamped stdout lines into timed cues."""
    cues: list[TranscriptCue] = []
    for line in (stdout or "").splitlines():
        match = _CUE_LINE.match(line.strip())
        if not match:
            continue
        h0, m0, s0, ms0, h1, m1, s1, ms1, text = match.groups()
        start = (int(h0 or 0) * 3600) + int(m0) * 60 + int(s0) + int(ms0) / 1000
        end = (int(h1 or 0) * 3600) + int(m1) * 60 + int(s1) + int(ms1) / 1000
        text = (text or "").strip()
        if not text or _BRACKET_ONLY.match(text):
            continue
        if 0 <= start <= end:
            cues.append(TranscriptCue(start, end, text))
    return cues


def _plain_text(stdout: str) -> str:
    """Join whisper-cli stdout into one transcript, dropping special rows."""
    lines: list[str] = []
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped or _CUE_LINE.match(stripped) or _BRACKET_ONLY.match(stripped):
            continue
        lines.append(stripped)
    return " ".join(lines).strip()


__all__ = [
    "ENV_WHISPER_CLI",
    "ENV_WHISPER_MODELS_DIR",
    "WhisperCppSTTAdapter",
    "model_search_dirs",
    "resolve_cli_binary",
    "resolve_model_path",
]
