"""pi coding agent backend — the ``--mode json`` wire contract.

Pins the command line (session id, system prompt, model, images, extra args)
and the JSONL event mapping against the stream shapes pi actually emits (its
own ``json.md`` reference, verified against a live 1.5.0 run): session header,
assistant message_update fragments, tool_execution frames, and the bookkeeping
brackets around them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from deeptutor.services.subagent.config import BackendConfig
from deeptutor.services.subagent.pi import PiBackend


def _stream(*items: Any, exit_code: str = "0"):
    """Replay canned stdout/stderr lines in stream_process_lines' shape."""

    async def _fake(cmd, cwd=None, **kwargs):  # noqa: ARG001
        for item in items:
            if isinstance(item, tuple):
                yield item
            else:
                yield "stdout", json.dumps(item)
        yield "exit", exit_code

    return _fake


async def _consult(monkeypatch, *items: Any, exit_code: str = "0", **kwargs):
    monkeypatch.setattr(
        "deeptutor.services.subagent.pi.stream_process_lines",
        _stream(*items, exit_code=exit_code),
    )
    seen: list[Any] = []

    async def on_event(event):
        seen.append(event)

    result = await PiBackend().consult("question", on_event=on_event, **kwargs)
    return result, seen


def _fragment(ftype: str, **extra) -> dict[str, Any]:
    return {"type": "message_update", "assistantMessageEvent": {"type": ftype, **extra}}


# ---- command building --------------------------------------------------------


def test_command_carries_json_mode_session_prompt_model_and_extras() -> None:
    config = BackendConfig(
        system_prompt="be concise",
        model="openai/gpt-5.2",
        effort="high",
        extra_args=["--no-mcp"],
    )
    cmd = PiBackend()._build_command("--literal prompt\n你好", session_id="sid-1", config=config)
    assert cmd[:5] == ["pi", "--mode", "json", "--session-id", "sid-1"]
    assert cmd[cmd.index("--append-system-prompt") + 1] == "be concise"
    assert cmd[cmd.index("--model") + 1] == "openai/gpt-5.2"
    assert cmd[cmd.index("--thinking") + 1] == "high"
    assert "--no-mcp" in cmd
    assert cmd[-1] == "--literal prompt\n你好"  # the prompt stays the trailing positional


def test_command_omits_empty_system_prompt_model_and_effort() -> None:
    cmd = PiBackend()._build_command("hi", session_id="sid-1", config=BackendConfig())
    assert "--append-system-prompt" not in cmd
    assert "--model" not in cmd
    assert "--thinking" not in cmd
    assert cmd[-1] == "hi"


def test_unknown_thinking_level_is_dropped_rather_than_passed_through() -> None:
    """`--thinking` accepts a fixed scale; anything else exits non-zero."""
    cmd = PiBackend()._build_command("hi", session_id="sid-1", config=BackendConfig(effort="ultra"))
    assert "--thinking" not in cmd


def test_resumed_sessions_still_carry_the_system_prompt() -> None:
    """pi's flag is idempotent, so every consult passes it — resumed included."""
    cmd = PiBackend()._build_command(
        "again", session_id="sid-1", config=BackendConfig(system_prompt="be concise")
    )
    assert cmd[cmd.index("--append-system-prompt") + 1] == "be concise"


def test_images_append_prompt_path_references() -> None:
    cmd = PiBackend()._build_command(
        "look",
        session_id="sid-1",
        config=BackendConfig(),
        images=["/tmp/dt-pi/00_a.png", "/tmp/dt-pi/01_b.png"],
    )
    assert cmd[-1] == "look\n\n@/tmp/dt-pi/00_a.png\n@/tmp/dt-pi/01_b.png"


@pytest.mark.asyncio
async def test_consult_mints_a_uuid_session_id_upfront(monkeypatch) -> None:
    """A fresh consult pre-generates the session id (never parsed from output)."""
    import uuid as uuid_mod

    captured: dict = {}

    def fake_build(question, *, session_id, config, images=None):  # noqa: ARG001
        captured["sid"] = session_id
        return ["true"]  # a no-op command that exits 0

    backend = PiBackend()
    backend._build_command = fake_build  # type: ignore[method-assign]

    async def on_event(event):
        pass

    result = await backend.consult("q", on_event=on_event)
    assert result.session_id == captured["sid"]
    assert uuid_mod.UUID(result.session_id)  # a well-formed UUID


# ---- event stream ------------------------------------------------------------


@pytest.mark.asyncio
async def test_assistant_deltas_stream_and_message_end_finalizes(monkeypatch) -> None:
    result, seen = await _consult(
        monkeypatch,
        {"type": "session", "version": 3, "id": "sid-1", "cwd": "/tmp"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        # the giant compiled system prompt echo — ignored wholesale
        {
            "type": "message_start",
            "message": {"role": "system", "content": "", "sections": {"preamble": "…"}},
        },
        {"type": "message_start", "message": {"role": "user", "content": "question"}},
        {"type": "message_end", "message": {"role": "user", "content": "question"}},
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        _fragment("text_start", contentIndex=0),
        _fragment("text_delta", contentIndex=0, delta="OK"),
        _fragment("text_delta", contentIndex=0, delta=", yes"),
        _fragment("text_end", contentIndex=0, content="OK, yes"),
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "OK, yes"}]},
        },
        {"type": "turn_end", "message": {"role": "assistant"}, "toolResults": []},
        {"type": "agent_end", "messages": [], "willRetry": False},
        {"type": "agent_settled", "aborted": False},
    )
    assert result.success is True
    assert result.final_text == "OK, yes"
    assert result.session_id  # the minted id, threaded back
    texts = [e for e in seen if e.kind == "text"]
    assert [e.text for e in texts] == ["OK", ", yes", "OK, yes"]
    # deltas and the completed block collapse into one evolving row
    assert {e.meta["merge_id"] for e in texts} == {"txt:1:0"}
    assert all(e.meta.get("partial") for e in texts[:2])


@pytest.mark.asyncio
async def test_message_end_surfaces_text_when_no_deltas_streamed(monkeypatch) -> None:
    result, seen = await _consult(
        monkeypatch,
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "All done."}]},
        },
    )
    assert result.final_text == "All done."
    assert [e.text for e in seen if e.kind == "text"] == ["All done."]


@pytest.mark.asyncio
async def test_thinking_maps_to_reasoning_channel(monkeypatch) -> None:
    result, seen = await _consult(
        monkeypatch,
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        _fragment("thinking_start", contentIndex=0),
        _fragment("thinking_delta", contentIndex=0, delta="planning"),
        _fragment("thinking_end", contentIndex=0, content="planning the run"),
        _fragment("text_end", contentIndex=1, content="Answer."),
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "planning the run"},
                    {"type": "text", "text": "Answer."},
                ],
            },
        },
    )
    assert result.final_text == "Answer."  # thinking blocks never leak into the answer
    assert [(e.kind, e.text) for e in seen if e.kind == "reasoning"] == [
        ("reasoning", "planning"),
        ("reasoning", "planning the run"),
    ]
    assert {e.meta["merge_id"] for e in seen if e.kind == "reasoning"} == {"rsn:1:0"}


@pytest.mark.asyncio
async def test_tool_calls_and_results_render_with_headers(monkeypatch) -> None:
    tool_call = {
        "type": "toolCall",
        "id": "call_1",
        "name": "bash",
        "arguments": {"command": "ls -la", "timeout": 15},
    }
    result, seen = await _consult(
        monkeypatch,
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        _fragment("toolcall_start", contentIndex=1, id="call_1", toolName="bash"),
        _fragment("toolcall_delta", contentIndex=1, delta='{"comm'),
        _fragment("toolcall_end", contentIndex=1, toolCall=tool_call),
        {
            "type": "tool_execution_start",
            "toolCallId": "call_1",
            "toolName": "bash",
            "args": {"command": "ls -la"},
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "call_1",
            "toolName": "bash",
            "result": {"content": [{"type": "text", "text": "a.py\nb.py"}]},
            "isError": False,
        },
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [tool_call, {"type": "text", "text": "Found two."}],
            },
        },
    )
    assert result.final_text == "Found two."  # toolCall blocks are not prose
    tools = [e for e in seen if e.kind == "tool"]
    results = [e for e in seen if e.kind == "tool_result"]
    assert [e.text for e in tools] == ["bash(ls -la)"]  # start/delta stay silent
    assert [e.text for e in results] == ["a.py\nb.py"]
    # the lifecycle's start/update frames are equally silent — the tool row
    # above already shows the call and the result row carries the output
    assert not any(e.kind == "log" for e in seen)


@pytest.mark.asyncio
async def test_tool_result_error_prefix_and_empty_result(monkeypatch) -> None:
    _, seen = await _consult(
        monkeypatch,
        {
            "type": "tool_execution_end",
            "toolCallId": "call_1",
            "toolName": "bash",
            "result": {"content": [{"type": "text", "text": "denied"}]},
            "isError": True,
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "call_2",
            "toolName": "read",
            "result": {"content": []},
            "isError": False,
        },
    )
    assert [e.text for e in seen if e.kind == "tool_result"] == [
        "error · denied",
        "(empty result)",
    ]


@pytest.mark.asyncio
async def test_tool_without_a_primary_arg_renders_the_bounded_args(monkeypatch) -> None:
    _, seen = await _consult(
        monkeypatch,
        _fragment(
            "toolcall_end",
            contentIndex=0,
            toolCall={"type": "toolCall", "id": "c", "name": "web_search", "arguments": {"q": "x"}},
        ),
    )
    assert seen[0].kind == "tool"
    assert seen[0].text.startswith("web_search(")


@pytest.mark.asyncio
async def test_service_and_unknown_frames_log_one_line_each(monkeypatch) -> None:
    _, seen = await _consult(
        monkeypatch,
        {"type": "queue_update", "steering": [], "followUp": []},
        {"type": "compaction_start", "reason": "threshold"},
        {"type": "auto_retry_start", "attempt": 1, "maxAttempts": 3, "errorMessage": "529"},
        {"type": "future_event", "data": "whatever"},
        ("stdout", "warning: plain text on stdout"),
        ("stderr", "Warning: no project session found with id 'x'; creating"),
    )
    logs = [e.text for e in seen if e.kind == "log"]
    assert len(logs) == 6
    assert any("queue_update" in t for t in logs)
    assert any("compaction_start" in t for t in logs)
    assert any(t.startswith("pi: future_event") for t in logs)
    assert any(t == "warning: plain text on stdout" for t in logs)
    assert any("no project session found" in t for t in logs)


@pytest.mark.asyncio
async def test_nonzero_exit_without_an_answer_fails(monkeypatch) -> None:
    result, seen = await _consult(
        monkeypatch,
        {"type": "message_start", "message": {"role": "assistant", "content": []}},
        exit_code="1",
    )
    assert result.success is False
    assert "code 1" in result.error
    assert seen[-1].kind == "error"


@pytest.mark.asyncio
async def test_nonzero_exit_after_an_answer_keeps_it(monkeypatch) -> None:
    """A late crash after the answer still surfaces the answer (kimi parity)."""
    result, _ = await _consult(
        monkeypatch,
        _fragment("text_end", contentIndex=0, content="partial answer"),
        exit_code="1",
    )
    assert result.success is True
    assert result.final_text == "partial answer"


@pytest.mark.asyncio
async def test_empty_stream_is_not_a_success_with_an_answer(monkeypatch) -> None:
    result, seen = await _consult(monkeypatch)
    assert result.success is True
    assert result.final_text == ""
    assert seen == []  # the consult tool layer reports the missing answer


@pytest.mark.asyncio
async def test_consult_passes_cwd_and_resumes_the_given_session(monkeypatch) -> None:
    captured: dict = {}

    async def stream(cmd, cwd=None):  # noqa: ARG001
        captured.update(cmd=cmd, cwd=cwd)
        yield (
            "stdout",
            json.dumps(
                {
                    "type": "message_end",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                }
            ),
        )
        yield "exit", "0"

    monkeypatch.setattr("deeptutor.services.subagent.pi.stream_process_lines", stream)

    async def on_event(event):
        pass

    result = await PiBackend().consult(
        "follow up", on_event=on_event, cwd="/project", session_id="saved-session"
    )
    assert result.success and result.final_text == "ok"
    assert result.session_id == "saved-session"
    assert captured["cwd"] == "/project"
    assert captured["cmd"][captured["cmd"].index("--session-id") + 1] == "saved-session"


# ---- detection and registry --------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("installed", [True, False])
async def test_detect_reports_availability_from_the_version_probe(monkeypatch, installed):
    from deeptutor.services.subagent.registry import get_backend, list_backend_kinds

    async def probe(cmd):
        assert cmd[:2] == ["pi", "--version"]
        return installed, "1.5.0" if installed else "not installed"

    monkeypatch.setattr("deeptutor.services.subagent.pi.probe_version", probe)
    assert "pi" in list_backend_kinds()
    backend = get_backend("pi")
    assert backend is not None and backend.local_cli is True
    detected = await backend.detect()
    assert detected.available is installed
    assert detected.version == ("1.5.0" if installed else "")
    assert (detected.detail == "") is installed


@pytest.mark.asyncio
async def test_options_parse_the_list_models_table(monkeypatch) -> None:
    from deeptutor.services.subagent import models as models_mod
    from deeptutor.services.subagent.models import sync_backend_options

    table = (
        "provider       model                     context  max-out  thinking  images\n"
        "openai         gpt-5.2                   400K     128K     yes       yes   \n"
        "anthropic      claude-sonnet-4.6         200K     64K      yes       yes   \n"
        "not-a-model-row\n"
    )

    async def fake_probe(cmd):
        return True, "pi 1.5.0"

    class FakeProcess:
        async def communicate(self):
            return table.encode(), b""

    async def fake_create_subprocess_exec(*cmd, **kwargs):  # noqa: ARG001
        assert "--list-models" in cmd
        return FakeProcess()

    monkeypatch.setattr(models_mod, "probe_version", fake_probe)
    monkeypatch.setattr(models_mod.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    options = await sync_backend_options("pi")
    assert options.kind == "pi"
    assert options.display_name == "Pi Coding Agent"
    assert options.available is True
    assert options.allow_custom_model is True
    assert options.synced_at  # a live read reports itself as the sync event
    assert options.efforts == ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
    assert [m.slug for m in options.models] == [
        "openai/gpt-5.2",
        "anthropic/claude-sonnet-4.6",
    ]
