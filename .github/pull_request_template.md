<!-- Thanks for contributing to P-Agents! Fill this out, delete sections that don't apply. -->

## Summary

<!-- What does this change? Why? Link issue: Closes #... -->

## Changes

- 
- 

## How tested

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `./scripts/e2e.sh` (offline) — paste result or check if not applicable
- [ ] Manual check with `cargo run -p cli -- --mock --prompt "..."`

## Risk & security

- [ ] No new `run_bash_command` surface or file-tool bypass
- [ ] If touching `WorkspacePolicy` / approval gate, called out below

<!-- Notes on risk, rollback, follow-ups -->

## Screenshots / logs

<!-- If UI/REPL behavior changed, paste relevant output -->

## Checklist

- [ ] Title uses Conventional Commits (`feat:`, `fix:`, `docs:` …)
- [ ] Added/updated tests and docs (`README.md`, `config.default.toml` comments, crate docs)
- [ ] No secrets or `.env` committed; `.gitignore` respected
