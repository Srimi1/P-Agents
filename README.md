# Universal Multi-Agent Harness (Rust)

A model-agnostic, high-performance **AI Agent Harness & Multi-Agent Orchestrator** built in Rust with `tokio`.

---

## 🌟 Features

- **Model Agnostic:** Connect to OpenAI, Anthropic, Gemini, DeepSeek, or local Ollama instances via the universal `LlmProvider` trait.
- **Hierarchical Multi-Agent Orchestration:** 
  - **Lead Planner ("Normal Thinker"):** Decomposes user goals and delegates tasks.
  - **Software Engineer Agent:** Writes, refactors, and edits codebase files.
  - **Verifier Agent:** Runs tests, checks compilation, and verifies diffs.
  - **Critic Agent ("Egoist"):** Challenges assumptions, finds edge cases and security risks.
  - **Research Tracker Agent:** Explores documentation and structure.
  - **Data Analyst Agent:** Evaluates logs, metrics, and data outputs.
- **Context Isolation:** Sub-agents run in isolated memory contexts, returning only synthesized outputs to prevent parent context bloat.
- **Safe Environment Tools:** File reading/writing, directory listings, and sandboxed terminal command execution with timeouts.
- **Event-Driven Architecture:** Event bus for real-time token streaming and UI bridges.

---

## 📁 Workspace Architecture

```text
.
├── Cargo.toml                  # Workspace manifest
├── crates/
│   ├── agent_core/             # Universal LlmProvider & ReAct execution loop
│   ├── harness_core/           # ToolEngine, File System, Terminal, Context Manager
│   ├── subagents/              # Sub-agent personas & delegation tools
│   ├── runtime/                # EventBus, Security Manager, Session persistence
│   └── cli/                    # User-facing REPL and CLI binary
└── tests/
```

---

## 🚀 Getting Started

### 1. Prerequisites
Install Rust (if not already installed):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. Configure Environment Variables
Set your API key and chosen model:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key"
export LLM_MODEL="gpt-4o"
export LLM_BASE_URL="https://api.openai.com/v1"

# Or for Local Ollama
export OPENAI_API_KEY="ollama"
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_MODEL="llama3.1"
```

### 3. Build and Run
```bash
# Build the workspace
cargo build --release

# Run the interactive REPL harness
cargo run -p cli

# Or execute a single prompt directly
cargo run -p cli -- --prompt "Review the files in the directory and recommend improvements"
```

---

## 📖 How Delegation Works Under the Hood

1. The user gives a goal to the **Lead Planner**.
2. The Lead Planner invokes the tool `spawn_subagent(role="engineer", task="...")`.
3. The harness spawns an isolated sub-agent with the requested persona and executes its private ReAct loop.
4. When finished, the sub-agent returns its result string back into the Lead Planner's conversation as a clean tool observation.
5. The Lead Planner synthesizes all sub-agent results and presents the final answer to the user.
