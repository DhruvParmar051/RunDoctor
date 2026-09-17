"""Interactive chat with RunDoctor: ``uv run rundoctor-chat --model qwen3:8b``."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from client.host import Agent, OpenAIChatModel, Trajectory, connect
from config import get_settings

app = typer.Typer(add_completion=False)

EXIT_WORDS = {"exit", "quit", ":q"}


def _short(text: str, limit: int = 160) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= limit else one_line[: limit - 3] + "..."


def _show(traj: Trajectory, verbose: bool) -> None:
    for call in traj.tool_calls:
        args = json.dumps(call.arguments) if call.arguments is not None else call.raw_arguments
        status = typer.style("error", fg="red") if call.is_error else "ok"
        typer.secho(f"  → {call.name}({args}) [{status}, {call.latency_s:.2f}s]", dim=True)
        if verbose or call.is_error:
            typer.secho(f"    {_short(call.result, 400 if verbose else 160)}", dim=True)
    if traj.stop_reason != "answer":
        typer.secho(
            f"  (stopped: {traj.stop_reason}{': ' + traj.error if traj.error else ''})", fg="yellow"
        )
    if traj.final_answer:
        typer.echo(f"\n{traj.final_answer}\n")
    typer.secho(
        f"  [{traj.iterations} iteration(s), {len(traj.tool_calls)} tool call(s), "
        f"{traj.total_latency_s:.1f}s]",
        dim=True,
    )


async def _chat(
    model: str, verbose: bool, max_iterations: int, once: str | None, db: Path | None
) -> None:
    if db is not None:
        os.environ["RUNDOCTOR_DB"] = str(db.resolve())
    settings = get_settings()
    llm = OpenAIChatModel(
        model,
        temperature=settings.eval.temperature,
        reasoning_effort=settings.models.reasoning_effort.get(model),
        max_tokens=settings.eval.max_tokens,
    )
    async with connect() as session:
        agent = await Agent.create(session, llm, max_iterations=max_iterations)
        history = agent.new_history()
        typer.secho(
            f"RunDoctor chat · model={model} · {len(agent.tools)} tools · "
            "type 'exit' to quit, '/reset' to clear history",
            bold=True,
        )
        while True:
            if once is not None:
                prompt = once
            else:
                try:
                    prompt = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    typer.echo()
                    return
            if not prompt:
                continue
            if prompt.lower() in EXIT_WORDS:
                return
            if prompt == "/reset":
                history = agent.new_history()
                typer.secho("  (history cleared)", dim=True)
                continue
            traj = await agent.run(prompt, history)
            _show(traj, verbose)
            if once is not None:
                return


@app.command()
def main(
    model: Annotated[str | None, typer.Option(help="Ollama model name")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show tool results")] = False,
    max_iterations: Annotated[int, typer.Option(help="Max tool-calling rounds per turn")] = 8,
    once: Annotated[str | None, typer.Option(help="Ask a single question and exit")] = None,
    db: Annotated[Path | None, typer.Option(help="SQLite DB path")] = None,
) -> None:
    """Chat with a local model that can inspect and diagnose your training runs."""
    asyncio.run(
        _chat(model or get_settings().models.chat_default, verbose, max_iterations, once, db)
    )


def run() -> None:
    app()


if __name__ == "__main__":
    run()
