# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

`main` is the only actively developed branch.

## Known design limits (not bugs)

Read these before filing — they are documented trade-offs in `README.md#Known limitations`:

* `run_bash_command` is **not** filesystem-contained; only the approval gate restricts it. `--yolo` removes that gate.
* Hard links inside `workspace_roots` can name a file outside it — path containment cannot catch this.
* Process-group kill on timeout has a narrow PID-reuse race; full isolation needs cgroups/job objects.

## Reporting a Vulnerability

**Do not open a public issue for sensitive vulnerabilities.**

* **Preferred:** Email **srimi.dev+security@gmail.com** with subject `[P-Agents security] <summary>`.
* **Alternative:** [Private vulnerability reporting](https://github.com/Srimi1/P-Agents/security/advisories/new) (GitHub Security Advisory).

Include:
* Affected commit / version, OS, and provider (`anthropic` / `openai` / `mock`).
* Steps to reproduce, impact assessment, and any suggested fix.
* Whether you want public credit.

### What to expect

* Acknowledgement within **48 hours**.
* Triage + initial assessment within **5 business days**.
* Fix timeline depends on severity — critical RCE / sandbox escape is prioritized for a patch within **14 days**.
* We will coordinate disclosure; please allow **90 days** before public disclosure unless otherwise agreed.
* You will be credited in the advisory and `CHANGELOG.md` unless you opt out.

## Scope

In scope: sandbox escape via `WorkspacePolicy`, approval-gate bypass, session-store injection, provider request forgery, dependency supply-chain issues in this repo.

Out of scope: model-output prompt injection (inherent to LLM), social engineering, physical access, upstream provider vulnerabilities.

## Hardening tips for operators

* Never run `--yolo` on a machine you care about while pointing an agent at a broad task.
* Set `permissions.sandbox = true` and keep `workspace_roots` narrow (default = cwd).
* Use `permissions.grant_scope = "agent"` if you want `a` (always-allow) scoped per-agent rather than per-tool.
* Pin `rust-toolchain.toml` and run `cargo audit` / `cargo deny` in CI before deploying a fork.

Thanks for helping keep P-Agents safe.
