#!/usr/bin/env bash
# End-to-end gate for the multi-agent harness.
#
#   ./scripts/e2e.sh
#
# Runs entirely offline unless ANTHROPIC_API_KEY is set, in which case it
# finishes with one live single-turn call.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PATH="$HOME/.cargo/bin:$PATH"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }
pass() { printf '\033[0;32m    ok\033[0m %s\n' "$1"; }
fail() { printf '\033[0;31m    FAIL\033[0m %s\n' "$1" >&2; exit 1; }

step "Build (release)"
cargo build --release --workspace

step "Tests"
cargo test --workspace

step "Clippy"
cargo clippy --workspace --all-targets -- -D warnings

step "Formatting"
cargo fmt --all --check

HARNESS="$REPO_ROOT/target/release/harness"
[ -x "$HARNESS" ] || fail "harness binary missing at $HARNESS"

step "One-shot run against the scripted provider"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
(
  cd "$WORKDIR"
  "$HARNESS" --mock --yolo --prompt "Create the artifact file" > run.log 2>&1
) || fail "one-shot run exited non-zero"

[ -f "$WORKDIR/harness_mock_artifact.txt" ] || fail "the sub-agent did not write its artifact"
grep -q "mock harness ok" "$WORKDIR/harness_mock_artifact.txt" || fail "artifact contents are wrong"
pass "sub-agent wrote the artifact"

SESSION="$(find "$WORKDIR/.harness/sessions" -name '*.jsonl' | head -n 1)"
[ -n "$SESSION" ] || fail "no session transcript was written"
head -n 1 "$SESSION" | grep -q '"type":"meta"' || fail "session log does not start with a meta record"
grep -q '"type":"message"' "$SESSION" || fail "session log has no message records"
grep -q '"type":"usage"' "$SESSION" || fail "session log has no usage records"
pass "session transcript is well formed"

step "Scripted REPL session: approval prompt, denial recovery, usage"
REPL_DIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR" "$REPL_DIR"' EXIT
(
  cd "$REPL_DIR"
  # No --yolo, so write_file hits the gate. "n" denies it; the model is told and
  # recovers rather than the run failing.
  printf '/help\nCreate the artifact file\nn\n/usage\n/session\n/exit\n' \
    | "$HARNESS" --mock > repl.log 2>&1
) || fail "REPL session exited non-zero"

grep -q "APPROVAL" "$REPL_DIR/repl.log" || fail "the approval prompt never appeared"
grep -qi "denied by user" "$REPL_DIR/repl.log" || fail "the denial was not reported back to the model"
grep -q "delegating to" "$REPL_DIR/repl.log" || fail "delegation was not shown"
grep -q "/critic" "$REPL_DIR/repl.log" || fail "/help did not list the commands"
[ ! -f "$REPL_DIR/harness_mock_artifact.txt" ] || fail "a denied write reached the disk"
pass "approval gate held and the denial was recoverable"

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  step "Live single-turn call"
  LIVE_DIR="$(mktemp -d)"
  (
    cd "$LIVE_DIR"
    "$HARNESS" --prompt "Reply with exactly: ok" > live.log 2>&1
  ) || fail "live call failed (see $LIVE_DIR/live.log)"
  grep -qi "ok" "$LIVE_DIR/live.log" || fail "live call returned nothing usable"
  rm -rf "$LIVE_DIR"
  pass "live provider round-trip"
else
  step "Live call skipped (ANTHROPIC_API_KEY not set)"
fi

printf '\n\033[1;32mAll end-to-end checks passed.\033[0m\n'
