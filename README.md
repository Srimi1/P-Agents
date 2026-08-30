# P-Agents — Universal Multi-Agent Harness (Rust)

> Creating and learning to create agents and subagents in a harness, step by step.
> Be ready to watch my journey.

A model-agnostic AI agent harness and multi-agent orchestrator built on `tokio`.
One static binary: streaming ReAct loop, isolated sub-agents, gated tool
execution, and a replayable transcript of everything that happened.

---

## What it does

- **Two native providers.** Anthropic's Messages API and any OpenAI-compatible
  endpoint (OpenAI, Ollama, vLLM, LocalAI, DeepSeek, Gemini's compatibility
  layer), both with real SSE streaming and token-usage accounting.
- **Six personas.** A Lead Planner that decomposes goals and delegates, plus
  Software Engineer, Verifier, Egoist Critic, Researcher, and Data Analyst
  specialists. More can be added from a config file.
- **Context isolation.** A sub-agent starts with its persona prompt and its task
  and nothing else — it never inherits the parent's transcript. Only its final
  answer returns, as a single tool observation.
- **Parallel delegation.** `run_parallel_subagents` fans independent subtasks out
  concurrently under a configurable ceiling and merges the answers back in the
  order they were requested.
- **Human-in-the-loop approval.** Tools that write files or run shell commands
  are gated behind a `[y/n/a]` prompt. The gate fails closed: no reachable
  approver means denied. A denial comes back to the model as an observation, so
  it adapts instead of crashing.
- **Session persistence.** Every message and usage report is appended to
  `.harness/sessions/<id>.jsonl` as it happens, and `--resume` replays it.
- **Context compaction.** Old tool observations are shrunk as the conversation
  approaches the model's window. Messages are never dropped, because removing an
  assistant turn that carries `tool_calls` invalidates the next request.

## Workspace layout

```text
crates/
├── agent_core/      LlmProvider trait, streaming ReAct loop, event and compaction hooks
│   └── providers/   anthropic, openai, mock
├── harness_core/    Tool trait and registry, filesystem/search/terminal tools, context manager
├── subagents/       Personas, delegation tools, orchestrator
├── runtime/         Event bus, approval gate, gated dispatcher, JSONL sessions
└── cli/             Config, provider factory, app assembly, streaming REPL
```

## Getting started

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Set an API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or use an OpenAI-compatible endpoint:

```bash
export OPENAI_API_KEY="sk-..."
```

For a local model, point the base URL at your server and use any placeholder key:

```bash
export OPENAI_API_KEY="ollama"
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="llama3.1"
```

### 3. Build and run

```bash
cargo build --release
```

Interactive REPL:

```bash
cargo run -p cli
```

One prompt, then exit:

```bash
cargo run -p cli -- --prompt "Review this workspace and list the three riskiest files"
```

No API key handy? The scripted offline provider exercises the whole loop:

```bash
cargo run -p cli -- --mock
```

## REPL commands

| Command | Effect |
| --- | --- |
| `/plan <goal>` | Force an explicit decomposition before any work starts |
| `/critic [text]` | Have the Egoist challenge the last answer, or the given text |
| `/verify [text]` | Have the Verifier check the last answer, or the given text |
| `/model [name]` | Hot-swap the model; `openai:gpt-4o` switches provider too |
| `/resume <id>` | Replay a previous session's lead transcript |
| `/usage` | Token usage so far, per agent |
| `/session` | Current session id and transcript path |
| `/help`, `/exit` | Command list, quit |

Anything else goes to the Lead Planner.

## Configuration

Copy [`config.default.toml`](config.default.toml) to
`~/.config/harness/config.toml`. Every value in it is already the built-in
default, so an empty file changes nothing.

Precedence runs lowest to highest: built-in defaults, then the config file, then
environment variables (`HARNESS_LIMITS__MAX_ITERATIONS=40`), then CLI flags.

API keys come from the environment by convention. A key placed in the config
file works but logs a warning.

## CLI flags

| Flag | Effect |
| --- | --- |
| `--prompt <text>` | Run once and exit instead of opening the REPL |
| `--provider <name>` | `anthropic`, `openai`, or `mock` |
| `--model <name>` | Override the configured model |
| `--config <path>` | Use a specific config file |
| `--yolo` | Approve every tool call without asking |
| `--resume <id>` | Restore a session by id or unambiguous prefix |
| `--session-dir <path>` | Where transcripts are written |
| `--mock` | Offline scripted provider |

`--yolo` lets agents write files and run shell commands with no confirmation.
It exists for CI and scripted runs; do not use it on a machine you care about
while pointing an agent at a broad task.

## How delegation works

1. The user gives a goal to the **Lead Planner**.
2. The planner calls `spawn_subagent(role, task)` or `run_parallel_subagents`.
3. Each sub-agent gets a fresh history — its persona prompt plus its task — and
   runs its own ReAct loop against the same gated tool dispatcher, so its tool
   calls face the same approval policy as the lead's.
4. The sub-agent's final answer returns to the planner as one tool observation.
5. The planner synthesizes the results and answers.

Sub-agents receive the tool registry *without* the delegation tools, which is
what keeps delegation one level deep rather than unbounded.

## Tools

| Tool | Approval | Notes |
| --- | --- | --- |
| `read_file` | no | Line-numbered, supports `offset` and `limit` |
| `list_directory` | no | Gitignore-aware, optional recursion and depth cap |
| `grep_search` | no | Regex over a gitignore-aware walk, skips binaries |
| `find_files_by_name` | no | Glob against both file name and relative path |
| `write_file` | yes | Creates parent directories |
| `edit_file_block` | yes | Exact-match replacement; refuses ambiguous matches |
| `run_bash_command` | yes | Timeout kills the child; optional background mode |

## Known limitations

Worth knowing before you point this at something important.

- **There is no filesystem sandbox.** The read-only tools will happily read any
  path the process can reach, including `~/.ssh` and files outside the working
  directory, and they do not prompt. The approval gate covers writing and shell
  execution, not reading. Run the harness as a user that only has access to what
  you are willing to expose.
- **A session grant is per-tool, not per-agent.** Answering `a` to an approval
  clears that tool for every agent, lead and sub-agents alike, until you exit.
  Use `y` if you only meant this one call.
- **Command timeouts kill the shell, not its descendants.** A timed-out
  `run_bash_command` kills the `sh` it started; a grandchild that shell spawned
  keeps running until it finishes on its own.
- **Background commands are not fully detached.** They share the harness's
  process group, so Ctrl-C in the terminal signals them too.
- **`--yolo` disables the gate entirely**, including for sub-agents. It is for
  CI and scripted runs.
- **Nothing here has been exercised against a live API in this build.** Both
  providers are covered end-to-end against a loopback HTTP server that asserts
  the exact request they emit, but no real endpoint was contacted.

## Development

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
./scripts/e2e.sh
```

`scripts/e2e.sh` runs the full gate: build, tests, clippy, an offline one-shot
run that asserts the artifact and session log, and a scripted REPL session that
exercises the approval prompt and denial recovery. If `ANTHROPIC_API_KEY` is
set it finishes with one live single-turn call.
