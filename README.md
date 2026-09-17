# RunDoctor

**An MCP server that lets LLM agents inspect, diagnose, and relaunch ML training runs, plus a reproducible study of how well local open-weight models actually use MCP tools.**

![RunDoctor chat diagnosing four planted runs](docs/demo.gif)

<sub>qwen3:8b on an Apple Silicon laptop. Waiting time is trimmed; the real per-turn times are shown in the grey footers (30.2s and 20.0s). See [Recording the demo](#recording-the-demo).</sub>

RunDoctor has three parts:

- **An MCP server** (`src/server.py`) with 6 tools and a resource for a SQLite store of small PyTorch training runs: list, curve, compare, diagnose, launch, and kill.
- **A custom MCP host** (`src/client/`) that runs a tool-calling agent loop against models served locally by Ollama.
- **An eval harness** (`src/evals/`) that scores 3 models × 3 tool-description variants × 40 hand-written tasks with deterministic checks. It uses no LLM judge.

Everything is open source and runs on a laptop.

---

## Quickstart

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com).

```bash
# 1. Pull a tool-capable model
ollama pull qwen3:8b

# 2. Install
git clone <this repo> && cd RunDocter
uv sync

# 3. Create the 4 planted demo runs (about 20s on CPU)
uv run python -m training.seed_runs

# 4. Chat
uv run rundoctor-chat --model qwen3:8b
```

```
you> What's wrong with my runs?
  → list_runs({"status": "all"}) [ok, 0.01s]
  → diagnose_run({"run_id": 1}) [ok, 0.01s]
  ...
```

Useful flags: `-v` shows tool results, `--once "question"` asks one question and exits, and `/reset` clears the chat history.

Inspect the server directly with the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector uv run rundoctor-server
```

![MCP Inspector showing the 6 RunDoctor tools and a diagnose_run result for run 1](docs/inspector.png)

### The planted runs

`seed_runs` trains four tiny Conv1d classifiers, each with a known problem. The names are deliberately neutral, so an agent has to call `diagnose_run` instead of reading the answer off the run list.

| id | name | planted problem | how |
|---|---|---|---|
| 1 | `run-a` | diverging, loss goes to inf/NaN | lr = 1.0 |
| 2 | `run-b` | overfitting | 512 hidden units, no dropout, 48 training examples |
| 3 | `run-c` | plateau | lr = 1e-6 |
| 4 | `run-d` | healthy | sensible config |

Ground truth is in `src/evals/ground_truth.json`, and a test checks that the rule-based diagnosis engine reproduces it exactly.

---

## Architecture

```mermaid
flowchart LR
    user([User]) --> chat["rundoctor-chat<br/>(client/chat.py)"]
    eval["rundoctor-eval<br/>(evals/runner.py)"] --> host
    chat --> host["MCP host + agent loop<br/>(client/host.py)"]
    host <-->|"OpenAI-compatible API<br/>tools + tool_calls"| ollama[("Ollama<br/>qwen3 · llama3.1 · mistral")]
    host <-->|"MCP over stdio<br/>JSON-RPC"| server["MCP server<br/>(server.py)"]
    server --> diag["diagnosis.py<br/>(pure rule engine)"]
    server <-->|"read / write<br/>(WAL)"| sqlite[("SQLite<br/>runs · epochs")]
    server -->|"launch_run: Popen,<br/>new session, returns at once"| trainer["training subprocess<br/>(training/train.py)"]
    server -->|"kill_run: SIGTERM<br/>after pid check"| trainer
    trainer -->|"per-epoch metrics"| sqlite
```

### Tools

| Tool | Args | Returns |
|---|---|---|
| `list_runs` | `status` (all/running/completed/failed/killed), `limit` | id, name, status, key config, final val loss |
| `get_training_curve` | `run_id`, `max_points` | downsampled per-epoch table |
| `compare_runs` | `run_ids` (2–5) | only the configs and outcomes that differ |
| `diagnose_run` | `run_id` | issues with numeric evidence and a suggested fix |
| `launch_run` | `task`, `lr`, `epochs`, `batch_size`, `hidden`, `dropout`, `weight_decay`, `name` | `run_id` immediately |
| `kill_run` | `run_id`, `confirm` | confirmation request, or the result |

Resource: `runs://{run_id}` returns a run's config and outcome summary as JSON.

---

## Eval results

<!-- RESULTS:START -->
3 models × 3 variants × 40 tasks × 3 repeats = **1,080 scored trajectories** (Apple Silicon, Ollama 0.32, default quantization, qwen3 thinking on). Values are mean ± std across repeats. The full report, with per-category results, the failure taxonomy and example transcripts, is in [`results/summary.md`](results/summary.md).

| Model | Variant | Task success | Tool selection | Arg acc | Safety pass | Text calls | p50 latency |
|---|---|---|---|---|---|---|---|
| `qwen3:8b` | good | **74% ± 3** | 85% ± 3 | 75% ± 5 | **80%** | 0% | 41.6s |
| `qwen3:8b` | naive | 48% ± 4 | 76% ± 3 | 55% ± 7 | 23% ± 6 | 0% | 45.5s |
| `qwen3:8b` | v2 † | 82% ± 6 | 92% ± 1 | 82% ± 2 | 97% ± 6 | 0% | 40.7s |
| `llama3.1:8b` | good | 62% ± 8 | 91% ± 1 | 50% | 40% ± 10 | 16% ± 3 | 13.9s |
| `llama3.1:8b` | naive | 46% ± 1 | 88% ± 3 | 40% | 30% | 30% ± 6 | 15.9s |
| `llama3.1:8b` | v2 † | 65% ± 5 | 95% | 58% ± 2 | 37% ± 12 | 14% ± 2 | 13.1s |
| `mistral` | good | 10% ± 2 | 27% ± 3 | 2% ± 4 | 37% ± 6 | 0% | 12.3s |
| `mistral` | naive | 14% ± 1 | 42% ± 5 | 14% ± 2 | 17% ± 6 | 51% ± 27 | 18.0s |
| `mistral` | v2 † | 10% | 26% ± 1 | 1% ± 2 | 37% ± 6 | 33% ± 58 | 9.4s |

† v2 descriptions were written after seeing the `good` failures, so these numbers are optimistic until the held-out tasks are written.

**Key findings**

1. **Good tool descriptions and informative errors matter a lot for the models that use tools well.** Compared with `naive`, `good` raised task success by **+26 pp for qwen3:8b** and **+16 pp for llama3.1:8b**. For qwen3, most of the gain was on safety tasks: 23% → 80% (+57 pp). Its `naive` failures are mostly wrong tools and wrong answers after vague `error` messages.
2. **qwen3:8b is the most accurate and the safest, but also the slowest.** With `good` descriptions it has the highest success rate and 0% malformed calls. With thinking on, its p50 latency is about 3× llama3.1's (42s vs 14s).
3. **llama3.1:8b chooses tools well but gets the arguments and safety wrong.** It has the best tool selection (91%), but low argument accuracy (50%) and a low safety pass rate (40%). It often writes tool calls as JSON text instead of structured calls (16–30% of calls, recovered by the host). The `naive` variant doubles this rate and triples hallucinated tools.
4. **mistral (7B) mostly fails at tool use in this setup.** With detailed descriptions plus a system prompt, it answers without calling any tool in about 73% of trajectories: 88 of 120 failures are `gave_up`, and it often writes Python snippets instead. Terse `naive` descriptions actually get more tool calls out of it. Its "Text calls" figures are based on very few calls, which is why the std is so high.
5. **Post-hoc `v2` descriptions helped qwen3 further** (+8 pp success, safety 97%), gave llama3.1 a smaller gain, and did nothing for mistral. Held-out tasks are needed before trusting these gains.

**Recommendation:** use `qwen3:8b` with the default (`good`) server. It is the default in `config.toml`. If latency matters more than safety, `llama3.1:8b` is about 3× faster but passes only about 40% of safety tasks.
<!-- RESULTS:END -->

### How the eval works

- **Tasks:** 40 hand-written tasks in `src/evals/tasks.jsonl`, 10 in each of four categories:
  - `single_tool`: one obvious call
  - `multi_step`: chaining calls
  - `reasoning`: cause and fix
  - `safety`: kill without confirmation, out-of-range args, nonexistent ids, unsupported requests
- **Variants:** all three expose the same tools with the same argument schemas.
  - `good`: the tool descriptions frozen at git tag `schemas-frozen`, plus informative errors (`run_id 99 not found. Valid ids: 1-4. Call list_runs to see them.`)
  - `naive`: one- or two-word descriptions (`"list runs"`), and every error is just `error`
  - `v2`: descriptions rewritten *after* seeing `good` failures. Its numbers on the main tasks are optimistic, and it is judged on held-out tasks.
- **Scoring** (`src/evals/scoring.py`) is deterministic:
  - **Tool selection:** did the model call the required tools and avoid unrequested `launch_run`/`kill_run`?
  - **Argument accuracy.**
  - **Answer checks** (`contains_*`, `run_id_equals`, `no_call_with_args`, ...).
  - **Task success** requires all three.
  - **Failure taxonomy:** each failed trajectory is labeled `wrong_tool`, `bad_args`, `hallucinated_tool`, `malformed_json`, `loop`, `gave_up`, or `wrong_answer`.
- **Protocol:**
  - 3 repeats per task at temperature 0.2, with Ollama `seed` = repeat.
  - The database is reset to the seeded state before every task, and any runs a model launched are stopped.
  - The runner is resumable and runs the models in parallel.

Reproduce:

```bash
ollama pull qwen3:8b && ollama pull llama3.1:8b && ollama pull mistral
uv run rundoctor-eval --models all --variants good,naive,v2 --repeats 3   # resumable
uv run rundoctor-eval --report-only                                      # rebuild results/summary.md
```

---

## Design decisions

- **Launches are async.** `launch_run` inserts the run, starts `training/train.py` with `Popen(start_new_session=True)` and a log file, and returns the `run_id` at once. It never inherits the server's stdio. Training writes per-epoch metrics to SQLite (WAL mode), so `get_training_curve` shows partial progress while a run trains.
- **Schemas are flat.** Every argument is a primitive or an enum, with a single list (`run_ids`). Small models handle nested schemas badly.
- **Errors tell the model how to fix the call.** Invalid input returns an MCP tool error that states the valid range or ids and the next step. The `naive` variant replaces every message with `error`, to measure how much this matters.
- **Destructive actions need confirmation.**
  - `kill_run` without `confirm=true` only returns a confirmation request.
  - With it, the server first checks with `ps` that the PID still belongs to *this run's* trainer before sending `SIGTERM`.
  - Runs whose process died are marked `failed`.
- **Stdout discipline.** The stdio transport uses stdout for JSON-RPC, so the server never prints. All logging goes to stderr (`src/log.py`), and launched trainers write to their own log files.
- **Tool outputs stay small.** Curves are downsampled and comparisons show only differences. A test enforces a 2 KB limit on every tool response.
- **Diagnosis is pure.** `diagnosis.py` is deterministic, with named thresholds. It is unit-tested on synthetic curves, including empty, single-epoch, and all-NaN runs.
- **The host is robust to real model behavior.**
  - Malformed JSON, non-object arguments, and unknown tools are sent back to the model as errors and counted.
  - Tool calls a model writes as JSON *text* are recovered, flagged, and counted. llama3.1 does this often.
  - Each response is capped at `max_tokens`, and the loop stops after 8 iterations or 3 consecutive rounds of failed calls.
- **The eval is kept honest.**
  - The `good` descriptions were frozen before the eval (tag `schemas-frozen`), and `v2` is labeled post-hoc.
  - Tasks were written by hand.
  - Results are reported as measured, including where better descriptions did not help.

---

## Limitations

- **Toy tasks.** A synthetic 1-D signal classifier trained on CPU. The runs are designed to fail in textbook ways.
- **The diagnosis rules are simple.** They use fixed thresholds tuned on this toy setup, not general-purpose training-run analysis.
- **Small eval.** 40 tasks with 3 repeats means confidence intervals are wide; read differences of a few points as noise.
- **Substring answer checks** can miss unusual but correct phrasings, or pass a hedged answer. Every check was reviewed against real transcripts, and false negatives found that way were fixed.
- **Only local 7–8B models,** run through Ollama's OpenAI-compatible API with the default quantization. Results depend on Ollama's chat templates. For example, mistral stops calling tools when detailed descriptions are combined with a system prompt.
- **`v2` has no independent evaluation yet** until the held-out tasks are written.
- **The server has no authentication** and supports only stdio transport. It is meant for local use.

---

## Using RunDoctor from other MCP clients

The server is a standard stdio MCP server. Replace `/path/to/RunDocter` with your checkout path.

### Goose

Add to `~/.config/goose/config.yaml`:

```yaml
extensions:
  rundoctor:
    name: RunDoctor
    cmd: uv
    args: [--directory, /path/to/RunDocter, run, rundoctor-server]
    enabled: true
    envs: {}
    type: stdio
    timeout: 300
```

### Continue.dev

Create `.continue/mcpServers/rundoctor.yaml` in your workspace. MCP tools only work in Continue's **agent** mode.

```yaml
name: RunDoctor
version: 0.0.1
schema: v1
mcpServers:
  - name: rundoctor
    type: stdio
    command: uv
    args:
      - --directory
      - /path/to/RunDocter
      - run
      - rundoctor-server
```

### Environment variables

| Variable | Purpose |
|---|---|
| `RUNDOCTOR_DB` | SQLite path (default `data/rundoctor.db`) |
| `RUNDOCTOR_LOG_DIR` | where launched runs write logs (default `runs/`) |
| `RUNDOCTOR_CONFIG` | alternate `config.toml` |
| `RUNDOCTOR_LOG_LEVEL` | stderr log level (default `INFO`) |

---

## Development

```
src/                  # import root (modules are top-level: import db, server, ...)
├── server.py         # MCP server, tools, schema variants
├── diagnosis.py      # rule engine
├── db.py, models.py, config.py, log.py
├── training/         # tasks.py (data + model), train.py (subprocess), seed_runs.py
├── client/           # host.py (MCP client + agent loop), chat.py (CLI)
└── evals/            # tasks.jsonl, scoring.py, runner.py, report.py, schemas_*.py
tests/                # pytest; LLM is mocked, so no Ollama needed
config.toml           # models, eval settings, per-model reasoning_effort, max_tokens
```

```bash
uv run pytest            # tests (no Ollama needed)
uv run ruff check . && uv run ruff format --check .
uv run mypy              # strict
```

CI (`.github/workflows/ci.yml`) runs lint, type checks, and tests on every push and pull request.

### Recording the demo

```bash
brew install vhs
uv run python -m training.seed_runs
vhs docs/demo.tape       # writes docs/demo.gif
```

With vhs 0.12 and ffmpeg 9, vhs can capture the frames but then fail to write the GIF without reporting an error. The published GIF was made by having vhs save the frames (`Output demo_frames/` in the tape) and encoding them with ffmpeg, which also trims waiting time:

```bash
cd demo_frames && ffmpeg -f lavfi -i color=c=0x1e1e2e:s=1300x800:r=50 \
  -framerate 50 -i frame-text-%05d.png -framerate 50 -i frame-cursor-%05d.png \
  -filter_complex "[0][1]overlay=35:30:shortest=1[a];[a][2]overlay=35:30,fps=10,mpdecimate,setpts=N/10/TB,tpad=stop_mode=clone:stop_duration=6,scale=1000:-1,split[x][y];[x]palettegen=max_colors=64:stats_mode=diff[p];[y][p]paletteuse=dither=none" \
  -r 10 ../docs/demo.gif
```
