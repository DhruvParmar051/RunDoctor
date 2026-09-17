# RunDoctor — Implementation Plan

> An MCP server that lets LLM agents inspect, diagnose, and relaunch ML training runs, plus an eval study measuring how well **open-weight models** use MCP tools.

This document is the spec for Claude Code. Work phase by phase. **Do not start a phase until the previous phase's acceptance criteria pass.** Commit at the end of each phase.

---

## 0. Context & Goals

- **Author:** solo developer, first MCP project, building a portfolio piece for AI engineer roles.
- **Timeline:** 7–10 days. Favor shipping over polish.
- **Constraint:** 100% open source. No proprietary APIs or hosts.
- **Hardware:** MacBook Pro (Apple Silicon). Models run locally via Ollama. **Assume 16GB RAM, so use 8B models by default.** Make the model list configurable so 14B models can be added if more RAM is available.

### What this project must show

1. A correct, well-designed MCP server (not a thin API wrapper).
2. A custom MCP client/host that runs a tool-calling loop against local models.
3. **A reproducible eval study with real numbers.** This is the headline deliverable.
4. Production habits: typed code, error handling, logging, tests.

### Non-goals for v1 (do NOT build)

- Web UI / Next.js frontend
- Kafka, Postgres, Express, Docker orchestration
- OAuth, public deployment
- GPU training, real datasets beyond toy tasks
- RAG of any kind

---

## 1. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Package manager | `uv` |
| MCP server | Official MCP Python SDK (`mcp`), using `from mcp.server.fastmcp import FastMCP` |
| MCP transport | stdio (v1). Streamable HTTP is optional in Phase 6. |
| Storage | SQLite via `sqlite3` stdlib (no ORM) |
| Training | PyTorch (CPU/MPS), tiny models |
| Model runtime | Ollama (`http://localhost:11434/v1`, OpenAI-compatible API) |
| LLM client | `openai` Python package pointed at Ollama |
| Models | `qwen3:8b`, `llama3.1:8b`, `mistral` (7B). Verify each supports tools in Ollama. |
| Validation | `pydantic` v2 |
| CLI | `typer` |
| Testing | `pytest`, `pytest-asyncio` |
| Lint/type | `ruff`, `mypy --strict` on `src/` |

---

## 2. Repository Layout

```
rundoctor/
├── pyproject.toml
├── README.md
├── plan.md
├── .gitignore                  # include data/*.db, runs/, results/raw/
├── config.toml                 # model list, paths, eval settings
├── src/                        # import root: modules below are top-level (import db, server, ...)
│   ├── config.py               # load config.toml -> typed settings
│   ├── log.py                  # stderr-only logger setup (not logging.py: would shadow stdlib)
│   ├── db.py                   # schema, connection, queries
│   ├── models.py               # pydantic models: Run, Epoch, Diagnosis
│   ├── training/
│   │   ├── train.py            # standalone training script (subprocess entrypoint)
│   │   ├── tasks.py            # toy datasets/models (synthetic 1D signal + MNIST-lite)
│   │   └── seed_runs.py        # creates the 4 "planted" demo runs
│   ├── diagnosis.py            # rule-based diagnostic checks (pure functions)
│   ├── server.py               # MCP server + tool definitions
│   ├── client/
│   │   ├── host.py             # MCP client + Ollama tool-calling loop
│   │   └── chat.py             # interactive CLI chat
│   └── evals/
│       ├── tasks.jsonl         # hand-written eval tasks
│       ├── schemas_naive.py    # deliberately weak tool descriptions (ablation)
│       ├── runner.py           # runs tasks x models x schema variants
│       ├── scoring.py          # metrics
│       └── report.py           # results -> markdown table + failure taxonomy
├── tests/
│   ├── test_db.py
│   ├── test_diagnosis.py
│   ├── test_server_tools.py
│   └── test_scoring.py
├── data/                       # sqlite db (gitignored)
└── results/                    # eval outputs (summary committed, raw gitignored)
```

---

## 3. Critical Rules (read before coding)

1. **Never write to stdout in the server process.** stdio transport uses stdout for JSON-RPC, and any `print()` will corrupt it. All logging goes to stderr via `log.get_logger`.
2. **Tools never block on training.** `launch_run` spawns a detached subprocess and returns a `run_id` immediately.
3. **Tool outputs must be small.** Summarize and downsample. Target <2KB per tool response. Never dump raw per-step metrics.
4. **Tool schemas stay flat.** Use primitive args, enums where possible, and few optional fields, because small models fail on nested schemas.
5. **Errors are informative, not exceptions.** Invalid input returns a structured error message that tells the model how to fix the call (e.g. `"run_id 99 not found. Valid ids: 1-12. Call list_runs to see them."`).
6. **Destructive tools require explicit confirmation.** `kill_run` requires `confirm: true`. Without it, return a message asking the model to confirm with the user.
7. **Diagnosis logic is pure and deterministic** so it can be unit tested without an LLM.
8. Type hints everywhere. No bare `except:`.

---

## 4. Phases

### Phase 1: Scaffold, DB, Training Script (Day 1)

**Tasks**
- Initialize a `uv` project, `pyproject.toml`, ruff/mypy/pytest config.
- `db.py` with this schema:

```sql
CREATE TABLE runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  task TEXT NOT NULL,              -- 'signal1d' | 'mnist_lite'
  config_json TEXT NOT NULL,       -- lr, epochs, batch_size, hidden, dropout, weight_decay, seed
  status TEXT NOT NULL,            -- 'running' | 'completed' | 'failed' | 'killed'
  pid INTEGER,
  created_at TEXT NOT NULL,
  finished_at TEXT,
  error TEXT
);
CREATE TABLE epochs (
  run_id INTEGER NOT NULL REFERENCES runs(id),
  epoch INTEGER NOT NULL,
  train_loss REAL,
  val_loss REAL,
  val_acc REAL,
  grad_norm REAL,
  PRIMARY KEY (run_id, epoch)
);
```

- Enable WAL mode (the training subprocess and the server write concurrently).
- `training/train.py`: CLI entrypoint that takes `--run-id`, reads the config from the DB, trains, writes each epoch's metrics to the DB, and sets the final status. Must finish in **<60s on CPU**.
- `training/tasks.py`:
  - `signal1d`: synthetic 1D signal classification (sine vs. square vs. sawtooth plus noise) with a small Conv1d model.
  - `mnist_lite`: optional. Skip if torchvision adds friction.
- `training/seed_runs.py` creates **4 planted runs** with known ground-truth problems:
  1. `diverging`: lr=1.0 → loss explodes or goes NaN
  2. `overfitting`: large model, no dropout, tiny train set → val loss rises while train loss falls
  3. `plateau`: lr=1e-6 → loss barely moves
  4. `healthy`: sensible config
  - Record ground truth in `evals/ground_truth.json`.

**Acceptance**
- `uv run python -m training.seed_runs` creates 4 completed runs whose curves visibly match their labels.
- `pytest tests/test_db.py` passes.

---

### Phase 2: Diagnosis Engine (Day 2)

**Tasks**
- `diagnosis.py` exposes `diagnose(epochs: list[Epoch], config: RunConfig) -> Diagnosis`.
- Checks (thresholds as named constants):
  - `nan_or_inf`: any non-finite loss
  - `divergence`: final train loss > 2× min train loss, or grad_norm blows up
  - `overfitting`: val loss rises for ≥3 epochs while train loss falls; report the gap
  - `plateau`: relative train-loss improvement < 1% over the last 50% of epochs
  - `healthy`: none of the above
- `Diagnosis` contains `issues: list[Issue]`, where each `Issue` has `code`, `severity`, `evidence` (short string with numbers), and `suggested_fix`.

**Acceptance**
- `tests/test_diagnosis.py` covers each check with synthetic curves, including edge cases (1 epoch, empty, all-NaN).
- Running `diagnose` on the 4 planted runs matches `ground_truth.json` exactly.

---

### Phase 3: MCP Server (Days 3–4)

**Tools**

| Tool | Args | Returns |
|---|---|---|
| `list_runs` | `status: Literal['all','running','completed','failed','killed'] = 'all'`, `limit: int = 10` | id, name, task, status, key config, final val_loss |
| `get_training_curve` | `run_id: int`, `max_points: int = 20` | downsampled epochs table |
| `compare_runs` | `run_ids: list[int]` (2–5) | config differences + outcome differences only |
| `diagnose_run` | `run_id: int` | the `Diagnosis` as compact text |
| `launch_run` | `task`, `lr`, `epochs`, `batch_size`, `hidden`, `dropout`, `weight_decay`, `name` (all flat, with defaults and bounds) | `run_id`, status `running` |
| `kill_run` | `run_id: int`, `confirm: bool = False` | confirmation request or result |

**Also**
- Expose `runs://{run_id}` as an MCP resource (config + summary).
- Tool descriptions: one line on *what*, one on *when to use it*, and argument constraints. Treat these as the "good" schema variant for the eval.
- `launch_run` validates bounds (e.g. `1e-6 <= lr <= 10`, `1 <= epochs <= 50`) and spawns `train.py` with `subprocess.Popen(..., start_new_session=True)`, with stdout/stderr redirected to a log file (never inherited).
- `kill_run` checks that the PID still belongs to the run before sending a signal.

**Acceptance**
- `npx @modelcontextprotocol/inspector uv run rundoctor-server` connects, lists all 6 tools, and each tool works manually.
- `tests/test_server_tools.py` calls the tool functions directly against a temporary DB.
- Launching a run and calling `get_training_curve` during training shows partial epochs.

---

### Phase 4: Custom MCP Host + Ollama (Day 5)

**Tasks**
- `client/host.py`:
  - Start the server over stdio using the MCP Python SDK client (`ClientSession`, `stdio_client`).
  - Convert MCP tool schemas to OpenAI-style `tools` definitions.
  - Agent loop: send messages → if the model returns `tool_calls`, execute each via `session.call_tool`, append results, repeat → stop on a final text answer or **max 8 iterations**.
  - Handle malformed tool calls (invalid JSON, unknown tool) by returning an error message to the model instead of crashing. Count these.
  - Return a `Trajectory` object: every message, tool call, argument, result, error, latency, and iteration count.
- `client/chat.py`: `uv run rundoctor-chat --model qwen3:8b` for interactive use (this is the demo).

**Acceptance**
- With `qwen3:8b`, asking "What's wrong with my runs?" results in the model calling `list_runs` then `diagnose_run` and giving an answer naming the problems.
- The loop terminates on every path (answer, max iterations, repeated errors).

---

### Phase 5: Eval Study (Days 6–7). HEADLINE DELIVERABLE

**Task set** (`evals/tasks.jsonl`, 40 tasks, **written by hand**, not LLM-generated)

Each line:
```json
{"id": "t01", "category": "single_tool", "prompt": "List my completed runs.",
 "expected_tools": ["list_runs"], "expected_args": {"list_runs": {"status": "completed"}},
 "answer_check": {"type": "contains_all", "values": ["healthy", "overfitting"]}}
```

Categories (about 10 each):
1. `single_tool`: one obvious tool call
2. `multi_step`: requires chaining (list → diagnose → compare)
3. `reasoning`: combine diagnosis + config into a cause and fix ("why did run 1 fail and what should I change?")
4. `safety`: destructive or ambiguous requests (`kill_run` without confirmation, out-of-bounds launch args, nonexistent run ids)

`answer_check` types: `contains_all`, `contains_any`, `run_id_equals`, `no_tool_called:<name>`. **Avoid LLM-as-judge.** If it's unavoidable for `reasoning`, hand-label a 10-task subset to validate it and report agreement.

**Experimental design**
- Models: from `config.toml` (default `qwen3:8b`, `llama3.1:8b`, `mistral`)
- Schema variants:
  - `good`: the Phase 3 descriptions + informative errors
  - `naive`: terse descriptions (`"list runs"`), generic errors (`"error"`), from `schemas_naive.py`
- Seeds/repeats: **3 runs per (task, model, variant)**, temperature 0.2
- Reset the DB to the seeded state before each task.
- Total: 40 × 3 models × 2 variants × 3 = 720 trajectories. Make the runner **resumable** (skip completed entries in `results/raw/*.jsonl`).

**Metrics** (`scoring.py`)
- `tool_selection_acc`: expected tools called (set match)
- `arg_acc`: expected args match on the expected tools
- `task_success`: answer_check passes
- `safety_pass`: safety tasks handled correctly
- `malformed_call_rate`, `avg_iterations`, `loop_rate` (hit max iterations), `p50/p95 latency`
- Report mean ± std across the 3 repeats.

**Failure taxonomy**: classify each failed trajectory as `wrong_tool`, `bad_args`, `hallucinated_tool`, `malformed_json`, `loop`, `gave_up`, `wrong_answer`.

**Report** (`report.py`): writes `results/summary.md` with:
1. Main table: model × variant × metrics
2. Delta table: good − naive per model
3. Failure taxonomy counts per model
4. 3 annotated example trajectories (one success, two instructive failures)

**Acceptance**
- `uv run rundoctor-eval --models all --variants good,naive --repeats 3` completes and resumes after interruption.
- `results/summary.md` is generated with real numbers.
- `tests/test_scoring.py` passes on hand-built trajectories.

**Integrity rules**
- Don't tune tool descriptions against the eval tasks and then report on those same tasks. Before starting Phase 5, freeze the `good` descriptions and commit them with a tag `schemas-frozen`. If you iterate afterward, hold out 10 tasks and report on those separately.
- Report results as measured, including when `good` doesn't beat `naive`.

---

### Phase 6: Ship (Days 8–10)

**Required**
- `README.md`:
  1. One-sentence pitch + demo GIF (terminal recording of `rundoctor-chat` diagnosing the 4 planted runs)
  2. Quickstart: install Ollama, pull a model, `uv sync`, seed runs, chat
  3. Architecture diagram (Mermaid): host ↔ MCP server ↔ SQLite ↔ training subprocess, host ↔ Ollama
  4. **Eval results table** + key findings in 3–5 bullets
  5. Design decisions: async launches, flat schemas, informative errors, confirmation for destructive tools, stdout discipline
  6. Limitations (toy tasks, rule-based diagnosis thresholds, small eval set, 8B models only)
- Config snippets for using the server from Goose and Continue.dev.
- GitHub Actions CI: ruff, mypy, pytest (no Ollama in CI; mock the LLM in host tests).

**Optional (only if time remains)**
- Streamable HTTP transport flag
- MCP progress notifications while a launched run trains
- Add 14B models to the eval if RAM allows

---

## 5. Definition of Done

- [ ] `uv sync && uv run python -m training.seed_runs && uv run rundoctor-chat` works from a fresh clone (with Ollama installed)
- [ ] All 6 tools work in MCP Inspector
- [ ] Planted runs diagnosed correctly by the rule engine (tested)
- [ ] Eval runs end to end; `results/summary.md` committed with real numbers
- [ ] README has GIF, results table, design decisions, limitations
- [ ] CI green

## 6. Instructions for Claude Code

- Work one phase at a time. At the end of each phase, run the acceptance checks, report the results, and commit with a message like `phase N: <summary>`.
- If a library API differs from what's described here (the MCP SDK changes often), check the installed package's source or docs and follow that. Note the deviation in the commit message.
- Ask before adding any dependency not listed in Section 1.
- Do not write eval tasks with an LLM. Scaffold 3 example tasks per category and leave the rest for the author to write, with a clear TODO.
- Prefer small, reviewable commits over large ones.
