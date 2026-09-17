"""Agent loop tests: a scripted fake LLM drives the real MCP server over stdio."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import db
from client.host import (
    Agent,
    AssistantTurn,
    Message,
    ToolCallRequest,
    Trajectory,
    connect,
    server_parameters,
    strip_thinking,
)
from models import Epoch, RunConfig


class ScriptedModel:
    """Returns pre-baked turns; repeats the last one when the script runs out."""

    name = "scripted"

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = turns
        self.calls = 0
        self.seen_tools: list[str] = []
        self.seen_messages: list[list[Message]] = []

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[Message]
    ) -> AssistantTurn:
        self.seen_tools = [t["function"]["name"] for t in tools]
        self.seen_messages.append(list(messages))
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        return turn


def call(name: str, args: str, cid: str = "c1") -> AssistantTurn:
    return AssistantTurn(content=None, tool_calls=[ToolCallRequest(cid, name, args)])


def answer(text: str) -> AssistantTurn:
    return AssistantTurn(content=text)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "host.db"
    with db.connection(path) as conn:
        run_id = db.create_run(conn, "flat", "signal1d", RunConfig(lr=1e-6))
        for i in range(1, 9):
            db.insert_epoch(
                conn,
                Epoch(
                    run_id=run_id,
                    epoch=i,
                    train_loss=1.1,
                    val_loss=1.1,
                    val_acc=0.33,
                    grad_norm=0.2,
                ),
            )
        db.set_status(conn, run_id, "completed")
    return path


def run_agent(db_path: Path, model: ScriptedModel, prompt: str, **kwargs: Any) -> Trajectory:
    async def go() -> Trajectory:
        params = server_parameters(env={"RUNDOCTOR_DB": str(db_path)})
        async with connect(params) as session:
            agent = await Agent.create(session, model, **kwargs)
            return await agent.run(prompt)

    return asyncio.run(go())


def test_happy_path(db_path: Path) -> None:
    model = ScriptedModel(
        [
            call("list_runs", '{"status": "completed"}'),
            call("diagnose_run", '{"run_id": 1}', "c2"),
            answer("<think>hmm</think>Run 1 (flat) is on a plateau; raise the lr."),
        ]
    )
    traj = run_agent(db_path, model, "What's wrong with my runs?")
    assert traj.stop_reason == "answer"
    assert traj.tools_called == ["list_runs", "diagnose_run"]
    assert traj.tool_calls[0].arguments == {"status": "completed"}
    assert "plateau" in traj.tool_calls[1].result
    assert traj.final_answer == "Run 1 (flat) is on a plateau; raise the lr."
    assert traj.iterations == 3
    assert traj.malformed_calls == 0
    assert len(traj.llm_latencies_s) == 3
    assert set(model.seen_tools) == {
        "list_runs",
        "get_training_curve",
        "compare_runs",
        "diagnose_run",
        "launch_run",
        "kill_run",
    }
    # Tool results are fed back with the matching tool_call_id.
    last_msgs = model.seen_messages[-1]
    tool_msgs = [m for m in last_msgs if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    roles = [m["role"] for m in traj.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]


def test_malformed_json_is_reported_not_raised(db_path: Path) -> None:
    model = ScriptedModel(
        [call("diagnose_run", '{"run_id": 1'), call("diagnose_run", "[1]"), answer("done")]
    )
    traj = run_agent(db_path, model, "diagnose run 1")
    assert traj.stop_reason == "answer"
    assert traj.malformed_calls == 2
    assert all(c.error_kind == "malformed_json" for c in traj.tool_calls)
    assert "not a valid JSON object" in traj.tool_calls[0].result


def test_empty_arguments_mean_no_args(db_path: Path) -> None:
    traj = run_agent(db_path, ScriptedModel([call("list_runs", ""), answer("ok")]), "list")
    assert not traj.tool_calls[0].is_error
    assert traj.tool_calls[0].arguments == {}


def test_unknown_tool(db_path: Path) -> None:
    model = ScriptedModel([call("delete_everything", "{}"), answer("sorry")])
    traj = run_agent(db_path, model, "wipe it")
    rec = traj.tool_calls[0]
    assert rec.error_kind == "unknown_tool"
    assert "Available tools:" in rec.result and "list_runs" in rec.result
    assert traj.malformed_calls == 1


def test_tool_error_is_passed_to_model(db_path: Path) -> None:
    model = ScriptedModel([call("diagnose_run", '{"run_id": 42}'), answer("no such run")])
    traj = run_agent(db_path, model, "diagnose run 42")
    rec = traj.tool_calls[0]
    assert rec.error_kind == "tool_error"
    assert "Valid ids: 1-1" in rec.result
    assert traj.malformed_calls == 0
    assert traj.stop_reason == "answer"


def test_max_iterations(db_path: Path) -> None:
    model = ScriptedModel([call("list_runs", "{}")])  # never answers
    traj = run_agent(db_path, model, "loop forever", max_iterations=4)
    assert traj.stop_reason == "max_iterations"
    assert traj.iterations == 4
    assert model.calls == 4
    assert traj.final_answer is None


def test_repeated_errors_stop_early(db_path: Path) -> None:
    model = ScriptedModel([call("nope", "{}")])
    traj = run_agent(db_path, model, "break", max_iterations=8)
    assert traj.stop_reason == "repeated_errors"
    assert traj.iterations == 3


def test_error_streak_resets_after_success(db_path: Path) -> None:
    model = ScriptedModel(
        [
            call("nope", "{}"),
            call("nope", "{}"),
            call("list_runs", "{}"),
            call("nope", "{}"),
            call("nope", "{}"),
            answer("fine"),
        ]
    )
    traj = run_agent(db_path, model, "mixed")
    assert traj.stop_reason == "answer"


def test_llm_error_stops_cleanly(db_path: Path) -> None:
    class Broken(ScriptedModel):
        async def complete(
            self, messages: Sequence[Message], tools: Sequence[Message]
        ) -> AssistantTurn:
            raise ConnectionRefusedError("ollama is down")

    traj = run_agent(db_path, Broken([]), "hello")
    assert traj.stop_reason == "llm_error"
    assert "ollama is down" in (traj.error or "")


def test_strip_thinking() -> None:
    assert strip_thinking("<think>\na\nb\n</think>\n\nAnswer") == "Answer"
    assert strip_thinking(None) == ""


def test_openai_model_passes_reasoning_effort() -> None:
    from types import SimpleNamespace

    from client.host import OpenAIChatModel

    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        msg = SimpleNamespace(content="hi", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")])

    def make(**kw: Any) -> OpenAIChatModel:
        m = OpenAIChatModel("qwen3:8b", base_url="http://127.0.0.1:9/v1", **kw)
        m._client = SimpleNamespace(  # type: ignore[assignment]
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )
        return m

    turn = asyncio.run(make(seed=1, reasoning_effort="none", max_tokens=64).complete([], []))
    assert turn.content == "hi" and turn.tool_calls == []
    assert turn.truncated
    assert captured["max_tokens"] == 64
    assert captured["extra_body"] == {"reasoning_effort": "none"}
    assert captured["seed"] == 1

    captured.clear()
    asyncio.run(make().complete([], []))
    assert "extra_body" not in captured and "seed" not in captured
    assert "max_tokens" not in captured


def test_text_tool_calls_are_recovered(db_path: Path) -> None:
    model = ScriptedModel(
        [
            answer('Let me check.\n{"name": "diagnose_run", "parameters": {"run_id": 1}}'),
            answer('```json\n{"name": "get_runs", "arguments": {}}\n```'),
            answer("Run 1 has plateaued."),
        ]
    )
    traj = run_agent(db_path, model, "diagnose run 1")
    assert traj.stop_reason == "answer"
    assert traj.tools_called == ["diagnose_run", "get_runs"]
    assert all(c.from_text for c in traj.tool_calls)
    assert traj.text_tool_calls == 2
    assert "plateau" in traj.tool_calls[0].result
    assert traj.tool_calls[1].error_kind == "unknown_tool"
    assert traj.malformed_calls == 1
    # Recovered calls are replayed as structured calls so the tool results pair up.
    assistant = [m for m in traj.messages if m["role"] == "assistant"]
    assert assistant[0]["tool_calls"][0]["function"]["name"] == "diagnose_run"


def test_extract_text_tool_calls_ignores_prose() -> None:
    from client.host import extract_text_tool_calls

    assert extract_text_tool_calls("") == []
    assert extract_text_tool_calls('Config: {"lr": 1, "name": "x"}') == []
    assert extract_text_tool_calls("{broken") == []
    calls = extract_text_tool_calls(
        '{"name": "a", "parameters": {}} and {"name": "b", "arguments": {"x": 1}}'
    )
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[1].arguments == '{"x": 1}'


def test_truncated_turns_are_counted(db_path: Path) -> None:
    model = ScriptedModel([AssistantTurn(content="Run 1 is ...", truncated=True)])
    traj = run_agent(db_path, model, "diagnose run 1")
    assert traj.truncated_turns == 1
    assert traj.stop_reason == "answer"
