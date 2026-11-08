"""Local whisper.cpp CLI STT adapter: resolution, CLI wiring and parsing."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from deeptutor.services.config.provider_runtime import STT_PROVIDERS
from deeptutor.services.voice.adapters import get_stt_adapter
from deeptutor.services.voice.adapters import whisper_cpp as adapter_module
from deeptutor.services.voice.adapters.whisper_cpp import (
    ENV_WHISPER_CLI,
    ENV_WHISPER_MODELS_DIR,
    WhisperCppSTTAdapter,
    _parse_cli_cues,
    _plain_text,
    resolve_cli_binary,
    resolve_model_path,
)
from deeptutor.services.voice.audio import pcm_to_wav
from deeptutor.services.voice.base import VoiceProviderError
from deeptutor.services.voice.config import STTConfig
from deeptutor.services.voice.options import voice_options

pytestmark = pytest.mark.asyncio

_ADAPTER_REF = "deeptutor.services.voice.adapters.whisper_cpp"


class _FakeProcess:
    """Mimic the subset of asyncio.subprocess.Process the adapter uses."""

    def __init__(
        self,
        argv: list[str],
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int = 0,
        communicate_delay: float = 0.0,
    ) -> None:
        self.argv = argv
        self._stdout = stdout
        self._stderr = stderr
        self._returncode = returncode
        self._communicate_delay = communicate_delay
        self.returncode: int | None = None
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._communicate_delay:
            await asyncio.sleep(self._communicate_delay)
        self.returncode = self._returncode
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int | None:
        return self.returncode


def _capture_spawn(
    monkeypatch: pytest.MonkeyPatch, process: _FakeProcess
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_exec(*args: object, **_kwargs: object) -> _FakeProcess:
        captured["argv"] = list(args)
        return process

    monkeypatch.setattr(f"{_ADAPTER_REF}.asyncio.create_subprocess_exec", fake_exec)
    return captured


def _canonical_wav() -> bytes:
    # Already 16 kHz mono PCM, so normalize_wav returns it without ffmpeg.
    return pcm_to_wav(b"\x00\x00", sample_rate=16000)


def _stt_config(**overrides: Any) -> STTConfig:
    values: dict[str, Any] = {
        "model": "ggml-small-q5_1",
        "provider_name": "whisper_cpp",
        "adapter": "whisper_cpp",
        "request_timeout": 10,
    }
    values.update(overrides)
    return STTConfig(**values)


# --------------------------------------------------------------------------
# Registration and provider metadata


def test_whisper_cpp_adapter_is_registered() -> None:
    assert isinstance(get_stt_adapter("whisper_cpp"), WhisperCppSTTAdapter)


def test_whisper_cpp_provider_spec_is_local_and_keyless() -> None:
    spec = STT_PROVIDERS["whisper_cpp"]
    assert spec.adapter == "whisper_cpp"
    assert spec.is_local
    assert spec.default_api_base == ""
    assert spec.default_model


def test_voice_options_advertise_local_note_and_auto_language() -> None:
    options = voice_options("whisper_cpp", "stt")
    note = options["fallback"]["configuration_note"]
    assert "whisper.cpp" in note
    assert ENV_WHISPER_MODELS_DIR in note
    language_ids = [item["id"] for item in options["fallback"]["languages"]]
    assert "auto" in language_ids
    assert options["docs_url"] == "https://github.com/ggml-org/whisper.cpp"


# --------------------------------------------------------------------------
# Binary and model resolution


def test_cli_binary_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = tmp_path / "whisper-cli"
    binary.write_text("#!/bin/sh\n")
    monkeypatch.setenv(ENV_WHISPER_CLI, str(binary))
    assert resolve_cli_binary() == str(binary)


def test_cli_binary_env_override_errors_on_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_WHISPER_CLI, str(tmp_path / "absent-cli"))
    with pytest.raises(VoiceProviderError, match="DEEPTUTOR_WHISPER_CLI"):
        resolve_cli_binary()


def test_cli_binary_missing_everywhere_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_WHISPER_CLI, raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(adapter_module, "_FALLBACK_BINARIES", ())
    with pytest.raises(VoiceProviderError, match="whisper-cli"):
        resolve_cli_binary()


def test_model_full_path_is_used_verbatim(tmp_path: Path) -> None:
    model = tmp_path / "custom-model.bin"
    model.write_bytes(b"ggml")
    assert resolve_model_path(str(model)) == str(model)


def test_model_name_resolves_in_models_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(ENV_WHISPER_MODELS_DIR, str(tmp_path))
    (tmp_path / "ggml-small-q5_1.bin").write_bytes(b"ggml")
    assert resolve_model_path("small-q5_1") == str(tmp_path / "ggml-small-q5_1.bin")
    assert resolve_model_path("ggml-small-q5_1.bin") == str(
        tmp_path / "ggml-small-q5_1.bin"
    )


def test_model_missing_name_lists_search_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_dir = tmp_path / "models"
    empty_dir.mkdir()
    monkeypatch.setenv(ENV_WHISPER_MODELS_DIR, str(empty_dir))
    with pytest.raises(VoiceProviderError) as excinfo:
        resolve_model_path("tiny")
    assert "tiny" in str(excinfo.value)
    assert str(empty_dir) in str(excinfo.value)
    assert ENV_WHISPER_MODELS_DIR in str(excinfo.value)


def test_model_path_with_missing_directory_errors(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "ggml-tiny.bin"
    with pytest.raises(VoiceProviderError, match="not found"):
        resolve_model_path(str(missing))


def test_model_empty_value_errors() -> None:
    with pytest.raises(VoiceProviderError, match="Set the model field"):
        resolve_model_path("")


# --------------------------------------------------------------------------
# Transcription wiring


async def test_transcribe_runs_cli_with_expected_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = _FakeProcess([], stdout=b"\n And so my fellow Americans \n")
    captured = _capture_spawn(monkeypatch, process)
    monkeypatch.setenv(ENV_WHISPER_MODELS_DIR, str(tmp_path))
    (tmp_path / "ggml-small-q5_1.bin").write_bytes(b"ggml")

    adapter = WhisperCppSTTAdapter()
    text = await adapter.transcribe(_canonical_wav(), _stt_config())

    assert text == "And so my fellow Americans"
    argv = captured["argv"]
    assert argv[0] == resolve_cli_binary()
    assert argv[argv.index("-m") + 1] == str(tmp_path / "ggml-small-q5_1.bin")
    assert argv[argv.index("-l") + 1] == "auto"
    assert "-np" in argv and "-nt" in argv
    assert str(captured["argv"][argv.index("-f") + 1]).endswith(".wav")


async def test_transcribe_passes_configured_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = _FakeProcess([], stdout=b"hi")
    captured = _capture_spawn(monkeypatch, process)
    monkeypatch.setenv(ENV_WHISPER_MODELS_DIR, str(tmp_path))
    (tmp_path / "ggml-small-q5_1.bin").write_bytes(b"ggml")

    await WhisperCppSTTAdapter().transcribe(
        _canonical_wav(), _stt_config(language="zh")
    )
    argv = captured["argv"]
    assert argv[argv.index("-l") + 1] == "zh"


async def test_transcribe_surfaces_cli_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = _FakeProcess([], stderr=b"error: failed to initialize whisper context\n", returncode=3)
    _capture_spawn(monkeypatch, process)
    monkeypatch.setenv(ENV_WHISPER_MODELS_DIR, str(tmp_path))
    (tmp_path / "ggml-small-q5_1.bin").write_bytes(b"ggml")

    with pytest.raises(VoiceProviderError, match="failed to initialize whisper context"):
        await WhisperCppSTTAdapter().transcribe(_canonical_wav(), _stt_config())


async def test_transcribe_timeout_kills_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = _FakeProcess([], communicate_delay=30.0)
    _capture_spawn(monkeypatch, process)
    monkeypatch.setenv(ENV_WHISPER_MODELS_DIR, str(tmp_path))
    (tmp_path / "ggml-small-q5_1.bin").write_bytes(b"ggml")
    real_timeout = asyncio.timeout
    monkeypatch.setattr(asyncio, "timeout", lambda _seconds: real_timeout(0.05))

    with pytest.raises(VoiceProviderError, match="timed out"):
        await WhisperCppSTTAdapter().transcribe(
            _canonical_wav(), _stt_config(request_timeout=30)
        )
    assert process.killed


async def test_transcribe_rejects_empty_audio() -> None:
    with pytest.raises(VoiceProviderError, match="No audio data"):
        await WhisperCppSTTAdapter().transcribe(b"", _stt_config())


async def test_transcribe_cues_return_timestamped_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stdout = (
        b"\n"
        b"[00:00:00.000 --> 00:00:02.500]   Hello world.\n"
        b"[00:00:02.500 --> 00:00:04.000]   Second line.\n"
    )
    process = _FakeProcess([], stdout=stdout)
    captured = _capture_spawn(monkeypatch, process)
    monkeypatch.setenv(ENV_WHISPER_MODELS_DIR, str(tmp_path))
    (tmp_path / "ggml-small-q5_1.bin").write_bytes(b"ggml")

    cues = await WhisperCppSTTAdapter().transcribe_cues(_canonical_wav(), _stt_config())

    assert [(cue.start_seconds, cue.end_seconds, cue.text) for cue in cues] == [
        (0.0, 2.5, "Hello world."),
        (2.5, 4.0, "Second line."),
    ]
    assert all(cue.timed for cue in cues)
    assert "-nt" not in captured["argv"]


async def test_cues_fall_back_to_untimed_plain_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    process = _FakeProcess([], stdout=b"\n plain transcript \n")
    _capture_spawn(monkeypatch, process)
    monkeypatch.setenv(ENV_WHISPER_MODELS_DIR, str(tmp_path))
    (tmp_path / "ggml-small-q5_1.bin").write_bytes(b"ggml")

    cues = await WhisperCppSTTAdapter().transcribe_cues(_canonical_wav(), _stt_config())

    assert len(cues) == 1
    assert cues[0].text == "plain transcript"
    assert not cues[0].timed


# --------------------------------------------------------------------------
# Parsing helpers


def test_parse_cli_cues_filters_special_rows_and_keeps_order() -> None:
    stdout = (
        "\n"
        "[00:00:00.000 --> 00:00:01.000]   [BLANK_AUDIO]\n"
        "[00:00:01.000 --> 00:00:03.250]  First line.\n"
        "[00:04:05.010 --> 01:00:06.020]   Later line.\n"
    )
    cues = _parse_cli_cues(stdout)
    assert [(cue.start_seconds, cue.end_seconds, cue.text) for cue in cues] == [
        (1.0, 3.25, "First line."),
        (245.01, 3606.02, "Later line."),
    ]


def test_plain_text_drops_timestamped_and_special_lines() -> None:
    stdout = (
        "\n"
        "[00:00:00.000 --> 00:00:01.000]   timed row stays out\n"
        "[MUSIC]\n"
        " real words \n"
    )
    assert _plain_text(stdout) == "real words"
    assert _plain_text("") == ""
