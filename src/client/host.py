"""MCP host: connects to the RunDoctor server over stdio and runs a tool-calling loop
against an OpenAI-compatible chat model (Ollama by default).

The LLM sits behind the small ``ChatModel`` protocol so tests can script it.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TextIO

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field

from config import get_settings
from log import get_logger

log = get_logger(__name__)

DEFAULT_MAX_ITERATIONS = 8
# Stop after this many consecutive iterations in which every tool call failed.
MAX_CONSECUTIVE_ERROR_ITERATIONS = 3
LLM_TIMEOUT_S = 300.0

SYSTEM_PROMPT = (
    "You are RunDoctor, an assistant that helps a machine learning engineer inspect and fix "
    "training runs. Use the provided tools to look up real data instead of guessing; never "
    "invent run ids or metrics. Never stop or launch runs unless the user clearly asks. "
    "When you have enough information, answer concisely and name the specific runs, "
    "problems, and fixes."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

Message = dict[str, Any]
StopReason = Literal["answer", "max_iterations", "repeated_errors", "llm_error"]
ErrorKind = Literal["malformed_json", "unknown_tool", "tool_error"]


# --- LLM interface -------------------------------------------------------------------


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    truncated: bool = False  # the model hit max_tokens


class ChatModel(Protocol):
    name: str

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[Message]
    ) -> AssistantTurn: ...


def make_client(base_url: str | None = None) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=base_url or get_settings().ollama.base_url,
        api_key="ollama",  # Ollama ignores the key, the client requires one
        timeout=LLM_TIMEOUT_S,
    )


class OpenAIChatModel:
    """Chat model served over an OpenAI-compatible API (Ollama)."""

    def __init__(
        self,
        name: str,
        base_url: str | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.name = name
        self.temperature = temperature
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        # Pass a shared ``client`` when creating many models, so HTTP connections are reused.
        self._client = client or make_client(base_url)

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[Message]
    ) -> AssistantTurn:
        kwargs: dict[str, Any] = {}
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": self.reasoning_effort}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        resp = await self._client.chat.completions.create(
            model=self.name,
            messages=list(messages),  # type: ignore[arg-type]
            tools=list(tools),  # type: ignore[arg-type]
            temperature=self.temperature,
            **kwargs,
        )
        choice = resp.choices[0]
        msg = choice.message
        calls = [
            ToolCallRequest(
                id=tc.id or f"call_{uuid.uuid4().hex[:8]}",
                name=tc.function.name,
                arguments=tc.function.arguments or "",
            )
            for tc in (msg.tool_calls or [])
            if tc.type == "function"
        ]
        return AssistantTurn(
            content=msg.content,
            tool_calls=calls,
            truncated=getattr(choice, "finish_reason", None) == "length",
        )


# --- trajectory ----------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    iteration: int
    name: str
    raw_arguments: str
    arguments: dict[str, Any] | None
    result: str
    is_error: bool
    error_kind: ErrorKind | None = None
    latency_s: float
    from_text: bool = False  # recovered from JSON written in the message text


class Trajectory(BaseModel):
    model: str
    prompt: str
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    final_answer: str | None = None
    stop_reason: StopReason = "answer"
    iterations: int = 0
    malformed_calls: int = 0
    text_tool_calls: int = 0
    truncated_turns: int = 0  # responses cut off by max_tokens
    llm_latencies_s: list[float] = Field(default_factory=list)
    total_latency_s: float = 0.0
    error: str | None = None

    @property
    def tools_called(self) -> list[str]:
        return [c.name for c in self.tool_calls]


# --- MCP plumbing --------------------------------------------------------------------


def server_parameters(
    extra_args: Sequence[str] = (), env: dict[str, str] | None = None
) -> StdioServerParameters:
    """Parameters to spawn the RunDoctor server with the current interpreter."""
    server_env = {"RUNDOCTOR_LOG_LEVEL": os.environ.get("RUNDOCTOR_LOG_LEVEL", "WARNING")}
    for key in ("RUNDOCTOR_DB", "RUNDOCTOR_CONFIG", "RUNDOCTOR_LOG_DIR"):
        if key in os.environ:
            server_env[key] = os.environ[key]
    server_env.update(env or {})
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "server", *extra_args],
        env=server_env,
    )


@asynccontextmanager
async def connect(
    params: StdioServerParameters | None = None, errlog: TextIO = sys.stderr
) -> AsyncIterator[ClientSession]:
    async with (
        stdio_client(params or server_parameters(), errlog=errlog) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


def mcp_tools_to_openai(tools: Sequence[Any]) -> list[Message]:
    """Convert MCP ``Tool`` objects to OpenAI ``tools`` definitions."""
    converted = []
    for t in tools:
        params = dict(t.input_schema or {})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": params,
                },
            }
        )
    return converted


def _result_text(result: Any) -> str:
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        parts.append(text if isinstance(text, str) else f"[{block.type} content]")
    return "\n".join(parts)


def strip_thinking(text: str | None) -> str:
    return _THINK_RE.sub("", text or "").strip()


def extract_text_tool_calls(text: str) -> list[ToolCallRequest]:
    """Recover tool calls a model wrote as JSON text instead of structured tool_calls.

    Some models (notably llama3.1) sometimes emit ``{"name": "list_runs", "parameters": {...}}``
    in the message body. Any JSON object with a string ``name`` and an object
    ``parameters``/``arguments`` counts; surrounding prose and code fences are ignored.
    """
    decoder = json.JSONDecoder()
    calls: list[ToolCallRequest] = []
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            params = obj.get("parameters", obj.get("arguments"))
            if isinstance(params, dict):
                calls.append(
                    ToolCallRequest(
                        id=f"text_call_{uuid.uuid4().hex[:8]}",
                        name=obj["name"],
                        arguments=json.dumps(params),
                    )
                )
        i = text.find("{", end)
    return calls


# --- agent ---------------------------------------------------------------------------


class Agent:
    """Runs the tool-calling loop for one MCP session and one chat model."""

    def __init__(
        self,
        session: ClientSession,
        model: ChatModel,
        tools: Sequence[Message],
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.session = session
        self.model = model
        self.tools = list(tools)
        self.tool_names = {t["function"]["name"] for t in self.tools}
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

    @classmethod
    async def create(cls, session: ClientSession, model: ChatModel, **kwargs: Any) -> Agent:
        listed = await session.list_tools()
        return cls(session, model, mcp_tools_to_openai(listed.tools), **kwargs)

    def new_history(self) -> list[Message]:
        return [{"role": "system", "content": self.system_prompt}]

    async def _execute(self, call: ToolCallRequest, iteration: int) -> ToolCallRecord:
        start = time.perf_counter()
        raw = call.arguments.strip()
        args: dict[str, Any] | None = None
        error_kind: ErrorKind | None = None
        result = ""
        try:
            parsed = json.loads(raw) if raw else {}
            if not isinstance(parsed, dict):
                raise ValueError("arguments must be a JSON object")
            args = parsed
        except ValueError as exc:
            error_kind = "malformed_json"
            result = (
                f"Error: arguments for {call.name} are not a valid JSON object ({exc}). "
                "Call the tool again with a JSON object of arguments."
            )

        if error_kind is None and call.name not in self.tool_names:
            error_kind = "unknown_tool"
            result = (
                f"Error: unknown tool '{call.name}'. "
                f"Available tools: {', '.join(sorted(self.tool_names))}."
            )

        if error_kind is None:
            assert args is not None
            try:
                res = await self.session.call_tool(call.name, args)
                result = _result_text(res)
                if res.is_error:
                    error_kind = "tool_error"
            except Exception as exc:  # transport/protocol failure; report to the model
                log.warning("tool %s failed: %r", call.name, exc)
                error_kind = "tool_error"
                result = f"Error calling {call.name}: {exc}"

        return ToolCallRecord(
            iteration=iteration,
            name=call.name,
            raw_arguments=call.arguments,
            arguments=args,
            result=result,
            is_error=error_kind is not None,
            error_kind=error_kind,
            latency_s=time.perf_counter() - start,
        )

    async def run(self, prompt: str, history: list[Message] | None = None) -> Trajectory:
        """Answer ``prompt``. ``history`` (if given) is extended in place for multi-turn chat."""
        messages = history if history is not None else self.new_history()
        messages.append({"role": "user", "content": prompt})
        traj = Trajectory(model=self.model.name, prompt=prompt)
        start = time.perf_counter()
        error_streak = 0
        traj.stop_reason = "max_iterations"

        for iteration in range(1, self.max_iterations + 1):
            traj.iterations = iteration
            t0 = time.perf_counter()
            try:
                turn = await self.model.complete(messages, self.tools)
            except (OpenAIError, OSError) as exc:
                traj.stop_reason = "llm_error"
                traj.error = f"{type(exc).__name__}: {exc}"
                break
            finally:
                traj.llm_latencies_s.append(time.perf_counter() - t0)

            if turn.truncated:
                traj.truncated_turns += 1
            from_text = False
            if not turn.tool_calls:
                recovered = extract_text_tool_calls(strip_thinking(turn.content))
                if recovered:
                    turn = AssistantTurn(content=turn.content, tool_calls=recovered)
                    from_text = True

            assistant: Message = {"role": "assistant", "content": turn.content or ""}
            if turn.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": c.arguments},
                    }
                    for c in turn.tool_calls
                ]
            messages.append(assistant)

            if not turn.tool_calls:
                traj.final_answer = strip_thinking(turn.content)
                traj.stop_reason = "answer"
                break

            records = [await self._execute(c, iteration) for c in turn.tool_calls]
            for call, rec in zip(turn.tool_calls, records, strict=True):
                if from_text:
                    rec.from_text = True
                    traj.text_tool_calls += 1
                traj.tool_calls.append(rec)
                if rec.error_kind in ("malformed_json", "unknown_tool"):
                    traj.malformed_calls += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": rec.result,
                    }
                )

            error_streak = error_streak + 1 if all(r.is_error for r in records) else 0
            if error_streak >= MAX_CONSECUTIVE_ERROR_ITERATIONS:
                traj.stop_reason = "repeated_errors"
                break

        traj.messages = list(messages)
        traj.total_latency_s = time.perf_counter() - start
        return traj
