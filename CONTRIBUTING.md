# Contributing to P-Agents

Thanks for considering a contribution! This is a Rust workspace (`tokio`-based harness) — the guide below keeps your change fast to review and safe to land.

## Quick start

```bash
# 1. Fork + clone
git clone https://github.com/<you>/P-Agents.git && cd P-Agents

# 2. Toolchain (rust-toolchain.toml pins stable + clippy + rustfmt)
rustup show   # installs automatically on first run

# 3. Run the full gate (same as CI)
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
./scripts/e2e.sh          # offline; add ANTHROPIC_API_KEY for live check

# Or run the app without a key:
cargo run -p cli -- --mock --prompt "Create the artifact file"
```

## Project layout

```
crates/
  agent_core/     LlmProvider trait, ReAct loop, SSE streaming, compaction
  harness_core/   Tool trait + file/search/pty tools + WorkspacePolicy (sandbox)
  subagents/      PersonaRegistry, delegation tools, MultiAgentOrchestrator
  runtime/        EventBus, ApprovalGate, GatedDispatcher, SessionStore (JSONL)
  cli/            Config (config.default.toml), provider factory, REPL, app wiring
config.default.toml  # all defaults — copy to ~/.config/harness/config.toml to override
scripts/e2e.sh       # build + tests + offline binary + REPL denial-recovery gate
```

## How to contribute

1. **Open an issue first** for anything non-trivial — helps avoid wasted work.
2. **Branch from `main`**: `git checkout -b feat/short-name` or `fix/short-name`.
3. **Keep changes focused** — one concern per PR. Separate refactor from behavior.
4. **Add tests** for new logic. Prefer `#[tokio::test]` against the harness, not mocks.
5. **Run the gate locally** before pushing (see Quick start).

## Commit & PR hygiene

* Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `test:`.
* PR title follows the same prefix — it becomes the squash-merge subject.
* Fill out the PR template checklist (tests, clippy, fmt, docs).
* Keep PRs < 400 lines where possible; split large ones and link them.

## Coding standards

* `cargo fmt` is required (CI checks `--check`).
* `cargo clippy -- -D warnings` must pass — no `#[allow(...)]` without a comment.
* `tracing` over `println!` in library crates; `colored`/`indicatif` only in `cli`.
* File tools must go through `WorkspacePolicy` — never `std::fs` directly in new tools.
* `run_bash_command` is intentionally unconstrained — don't add path checks that give false confidence.
* Security-sensitive changes: read `SECURITY.md` and call out the risk in the PR description.

## Reporting bugs / requesting features

Use the issue templates: **Bug report** and **Feature request**. Include:
* `rustc --version`, `cargo --version`, OS, and how you ran it (`--mock` vs live provider).
* Minimal reproduction and expected vs actual behavior.
* For panics: `RUST_BACKTRACE=1` output.

## Development tips

* Config precedence: built-ins < `~/.config/harness/config.toml` < env (`HARNESS_*__*`) < CLI flags.
* Session transcripts live in `.harness/sessions/<id>.jsonl` — `--resume <id>` replays them.
* `--yolo` disables the approval gate but **not** `WorkspacePolicy` containment.
* To add a persona: add a file under `crates/subagents/personas/` and register it in `PersonaRegistry`.

## License

By contributing you agree your contributions are licensed under the MIT License (see `LICENSE`).

## Questions?

Open a [Discussion](https://github.com/Srimi1/P-Agents/discussions) or ask in your PR — maintainers respond within 2–3 days.
