"""pi coding agent backend — drive the local ``pi`` CLI in headless JSON mode.

Uses ``pi --mode json <prompt>``: a strict JSONL stream of session events, one
object per line, flushed live (stdout is JSONL only; diagnostics go to stderr).
The vocabulary is pi's own session protocol — ``message_update`` frames carry an
``assistantMessageEvent`` (``text_delta`` / ``thinking_delta`` / ``toolcall_end``
/ …), tool runs report through top-level ``tool_execution_*`` events, and each
assistant message is framed by ``message_start`` / ``message_end`` records whose
``content`` part list is the authoritative text. The stream opens with a
``session`` header and echoes the whole compiled system prompt as a giant
``role=system`` message — both are noise here and ignored, as are the echoed
``role=user`` records.

Session continuity uses the same trick as the Kimi backend: ``--session-id
<uuid>`` with a nonexistent id *creates* the session under that exact id, so we
mint the id ourselves and never parse one from output. Threading the same id
across consults resumes the same agent session. The delegate instruction rides
``--append-system-prompt`` (a flag pi accepts text or a file path for, and one
that is idempotent across a resumed session, so every consult carries it — no
first-message prefixing). The model is pinned with ``--model`` and the thinking
level with ``--thinking`` when configured.

Approvals never stall: in ``--mode json`` pi executes its tools fully
autonomously with no interactive prompts, so there is no ``--yolo`` equivalent
to pass and ``auto_approve`` is a no-op here. Images have no flag either — pi
attaches files (images included) named in the prompt as ``@<absolute path>``,
one reference per line, when ``forward_images`` opts in.

Auth and config are inherited automatically from the user's ``~/.pi`` — no
token is ever handled here.
"""

from __future__ import annotations

import json
import logging
from typing import Any
import uuid

from deeptutor.services.subagent.base import OnEvent, SubagentBackend
from deeptutor.services.subagent.config import BackendConfig
from deeptutor.services.subagent.process import (
    compact_field as _compact,
)
from deeptutor.services.subagent.process import (
    probe_version,
    stream_process_lines,
    truncate_field,
)
from deeptutor.services.subagent.types import (
    EVENT_ERROR,
    EVENT_LOG,
    EVENT_REASONING,
    EVENT_TEXT,
    EVENT_TOOL,
    EVENT_TOOL_RESULT,
    ConsultResult,
    DetectResult,
    SubagentEvent,
)

logger = logging.getLogger(__name__)

_TOOL_HEADER_CHARS = 160

# `--thinking` accepts exactly these levels; anything else is dropped rather
# than passed through to a non-zero exit on an unknown value (the same
# arrangement as the Antigravity backend's `--effort` allowlist).
_THINKING_LEVELS = frozenset({"off", "minimal", "low", "medium", "high", "xhigh", "max"})

# The salient argument to put in a tool header — `bash(ls -la)` rather than raw
# JSON, matching how the other CLI backends render their tool rows.
_TOOL_PRIMARY_ARGS = (
    "command",
    "file_path",
    "path",
    "pattern",
    "query",
    "url",
    "prompt",
    "description",
)

# Bookkeeping frames with nothing to surface: the session header, the run/turn
# brackets (turn_end's message already arrived as its own message_end), the
# agent_end message dump (which repeats everything, minus nothing), and the
# tool_execution start/update frames — the toolcall_end row already shows the
# call and tool_execution_end below carries the result.
_IGNORED_TYPES = frozenset(
    {
        "session",
        "agent_start",
        "turn_start",
        "turn_end",
        "agent_end",
        "agent_settled",
        "tool_execution_start",
        "tool_execution_update",
    }
)

# Block-level fragments that only make sense once complete (or not at all):
# toolcall_start/delta stream a half-built call whose authoritative form is the
# toolcall_end toolCall, and text/thinking blocks have no content at start.
_IGNORED_FRAGMENT_TYPES = frozenset(
    {"text_start", "thinking_start", "toolcall_start", "toolcall_delta", "done", "start"}
)

# Service frames worth one bounded log line: queue and retry bookkeeping plus
# compaction progress (compaction_end carries the whole summary otherwise).
_SERVICE_LOG_TYPES = frozenset(
    {
        "queue_update",
        "entry_appended",
        "session_info_changed",
        "thinking_level_changed",
        "auto_retry_start",
        "auto_retry_end",
        "summarization_retry_scheduled",
        "summarization_retry_attempt_start",
        "summarization_retry_finished",
    }
)


class PiBackend(SubagentBackend):
    """Consult the local pi coding agent (``pi``) as a subagent."""

    kind = "pi"
    display_name = "Pi Coding Agent"
    cli_command = "pi"

    async def detect(self) -> DetectResult:
        ok, text = await probe_version([self.cli_command, "--version"])
        return DetectResult(
            kind=self.kind,
            display_name=self.display_name,
            available=ok,
            version=text if ok else "",
            detail="" if ok else (text or "pi CLI not found on PATH"),
        )

    def _build_command(
        self,
        question: str,
        *,
        session_id: str,
        config: BackendConfig,
        images: list[str] | None = None,
    ) -> list[str]:
        prompt = question
        # No attachment flag: pi reads files named in the prompt with
        # @<absolute path> (images included) — one reference per line.
        if images:
            prompt = f"{prompt}\n\n" + "\n".join(f"@{path}" for path in images)
        cmd = [
            self.cli_command,
            "--mode",
            "json",
            "--session-id",
            session_id,
        ]
        if config.system_prompt.strip():
            # Idempotent flag: every consult carries the delegate instruction,
            # resumed sessions included — no first-message prefixing needed.
            cmd += ["--append-system-prompt", config.system_prompt.strip()]
        if config.model:
            cmd += ["--model", config.model]
        if config.effort in _THINKING_LEVELS:
            cmd += ["--thinking", config.effort]
        cmd += list(config.extra_args)
        cmd += [prompt]
        return cmd

    async def consult(
        self,
        question: str,
        *,
        on_event: OnEvent,
        cwd: str | None = None,
        session_id: str | None = None,
        config: BackendConfig | None = None,
        images: list[str] | None = None,
        partner_id: str | None = None,  # noqa: ARG002 — partner-only; ignored here
    ) -> ConsultResult:
        config = config or BackendConfig()
        # Mint the session id ourselves (a nonexistent --session-id id creates
        # the session under that id), so continuity never depends on parsing
        # output (pi echoes it only in the session header anyway).
        sid = session_id or str(uuid.uuid4())
        cmd = self._build_command(question, session_id=sid, config=config, images=images)
        result = ConsultResult(session_id=sid)
        # message_update deltas carry no message id, so count assistant messages
        # to keep each block's merge id stable and unique across a multi-step run.
        stream: dict[str, Any] = {"message": 0, "saw_text": False}

        async def emit(
            kind: str, text: str, raw: dict[str, Any], meta: dict[str, Any] | None = None
        ) -> None:
            result.event_count += 1
            await on_event(SubagentEvent(kind=kind, text=text, raw=raw, meta=meta or {}))

        try:
            async for channel, line in stream_process_lines(cmd, cwd=cwd):
                if channel == "exit":
                    if line != "0" and result.success and not result.final_text:
                        result.success = False
                        result.error = f"pi exited with code {line}"
                        await emit(EVENT_ERROR, result.error, {"returncode": line})
                    continue
                if channel == "stderr":
                    if line.strip():
                        await emit(EVENT_LOG, line, {"stream": "stderr"})
                    continue
                event = _parse_json(line)
                if event is None:
                    if line.strip():
                        await emit(EVENT_LOG, line, {"stream": "stdout"})
                    continue
                await self._handle_event(event, result, stream, emit)
        except Exception as exc:  # pragma: no cover - defensive: surface, don't crash the turn
            logger.warning("pi consult failed: %s", exc, exc_info=True)
            result.success = False
            result.error = str(exc)
            await emit(EVENT_ERROR, str(exc), {})

        return result

    async def _handle_event(
        self,
        event: dict[str, Any],
        result: ConsultResult,
        stream: dict[str, Any],
        emit: Any,
    ) -> None:
        etype = str(event.get("type") or "")

        if etype in _IGNORED_TYPES:
            return

        if etype in ("message_start", "message_end"):
            message = event.get("message")
            message = message if isinstance(message, dict) else {}
            if message.get("role") != "assistant":
                return  # the compiled system-prompt echo or the user's question
            if etype == "message_start":
                stream["message"] += 1
                stream["saw_text"] = False
                return
            # message_end: the authoritative part list finalizes the answer.
            text = _text_from_parts(message.get("content"))
            if not text:
                return  # a tool-call-only message — nothing new to answer with
            if not stream["saw_text"]:
                stream["saw_text"] = True
                await emit(EVENT_TEXT, text, event)  # no deltas streamed — surface it once
            result.final_text = text
            return

        if etype == "message_update":
            fragment = event.get("assistantMessageEvent")
            if isinstance(fragment, dict):
                await self._handle_fragment(fragment, result, stream, emit)
            return

        if etype == "tool_execution_end":
            text = _result_text(event.get("result"))
            if event.get("isError") and text:
                text = f"error · {text}"
            await emit(EVENT_TOOL_RESULT, truncate_field(text) or "(empty result)", event)
            return

        if etype in _SERVICE_LOG_TYPES or etype.startswith("compaction_"):
            await emit(EVENT_LOG, truncate_field(_compact(event)), event)
            return

        # Unknown future frames: log the type only (payloads may be large).
        await emit(EVENT_LOG, f"pi: {etype or 'unknown'}", event)

    async def _handle_fragment(
        self,
        fragment: dict[str, Any],
        result: ConsultResult,
        stream: dict[str, Any],
        emit: Any,
    ) -> None:
        ftype = str(fragment.get("type") or "")
        if ftype in _IGNORED_FRAGMENT_TYPES:
            return
        index = fragment.get("contentIndex", 0)
        message = stream["message"]

        if ftype == "text_delta":
            delta = str(fragment.get("delta") or "")
            if delta:
                stream["saw_text"] = True
                await emit(
                    EVENT_TEXT,
                    delta,
                    fragment,
                    {"partial": True, "merge_id": f"txt:{message}:{index}"},
                )
            return
        if ftype == "text_end":
            content = str(fragment.get("content") or "").strip()
            if content:
                stream["saw_text"] = True
                await emit(EVENT_TEXT, content, fragment, {"merge_id": f"txt:{message}:{index}"})
                result.final_text = content
            return
        if ftype == "thinking_delta":
            delta = str(fragment.get("delta") or "")
            if delta:
                await emit(
                    EVENT_REASONING,
                    delta,
                    fragment,
                    {"partial": True, "merge_id": f"rsn:{message}:{index}"},
                )
            return
        if ftype == "thinking_end":
            content = str(fragment.get("content") or "").strip()
            if content:
                await emit(
                    EVENT_REASONING, content, fragment, {"merge_id": f"rsn:{message}:{index}"}
                )
            return
        if ftype == "toolcall_end":
            call = fragment.get("toolCall")
            if isinstance(call, dict):
                await emit(EVENT_TOOL, _render_tool_call(call), call)
            return
        if ftype == "error":
            message_text = str(
                fragment.get("error") or fragment.get("reason") or "provider stream error"
            )
            await emit(EVENT_ERROR, truncate_field(message_text), fragment)
            return
        await emit(EVENT_LOG, f"pi: {ftype or 'unknown'}", fragment)


def _parse_json(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line or line[0] not in "{[":
        return None
    try:
        parsed = json.loads(line)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _text_from_parts(content: Any) -> str:
    """Join the text parts of a pi message/result ``content``.

    ``content`` is polymorphic: a plain string or a block list whose text
    blocks (``{"type": "text", "text": …}``) carry the prose — thinking and
    toolCall blocks are not prose and are skipped.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = [
        str(block.get("text") or "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part.strip()).strip()


def _result_text(result: Any) -> str:
    """Render a tool_execution_end ``result`` — text parts, else the structured blob."""
    if not isinstance(result, dict):
        return str(result or "")
    text = _text_from_parts(result.get("content"))
    if text:
        return text
    structured = result.get("structuredContent")
    return _compact(structured) if structured is not None else ""


def _render_tool_call(call: dict[str, Any]) -> str:
    """``name(primary arg)`` for the sidebar row; arguments arrive as a dict."""
    name = str(call.get("name") or "tool")
    args = call.get("arguments")
    args = args if isinstance(args, dict) else {}
    if args:
        for key in _TOOL_PRIMARY_ARGS:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return f"{name}({_inline(value)})"
        return f"{name}({_inline(_compact(args))})"
    return name


def _inline(text: str) -> str:
    one_line = " ".join(text.split())
    if len(one_line) > _TOOL_HEADER_CHARS:
        return one_line[:_TOOL_HEADER_CHARS].rstrip() + " …"
    return one_line


__all__ = ["PiBackend"]
