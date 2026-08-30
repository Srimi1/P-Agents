//! Human-in-the-loop approval gate. The gate lives on the agent side of the
//! channel; the UI owns the receiver and answers each request over a oneshot.

use anyhow::Result;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tracing::warn;
use uuid::Uuid;

/// Depth of the pending-approval queue. Requests block the calling agent
/// anyway, so a shallow queue is enough for a lead plus a few sub-agents.
const APPROVAL_QUEUE_DEPTH: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApprovalDecision {
    Approve,
    ApproveForSession,
    Deny,
}

/// How far a single "approve for the session" answer reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GrantScope {
    /// A grant covers this tool for every agent (today's behaviour).
    #[default]
    Tool,
    /// A grant covers this tool only for the agent that was asked.
    Agent,
}

/// Identity of a session grant. `Tool` scope drops the agent so any caller
/// matches; `Agent` scope keeps it so a grant cannot travel between agents.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GrantKey {
    agent_id: Option<String>,
    tool: String,
}

pub struct ApprovalRequest {
    pub request_id: String,
    pub agent_id: String,
    pub tool: String,
    pub arguments: Value,
    /// The blast radius an `ApproveForSession` answer to *this* request would
    /// have. Carried on the request so the UI can describe the consent it is
    /// asking for truthfully: the prompt cannot say "any agent" when the gate
    /// will only grant this one.
    pub scope: GrantScope,
    pub respond: oneshot::Sender<ApprovalDecision>,
}

#[derive(Clone)]
pub struct ApprovalGate {
    tx: mpsc::Sender<ApprovalRequest>,
    /// Tools the user has blanket-approved for the rest of the session, keyed
    /// according to `scope`.
    session_grants: Arc<Mutex<HashSet<GrantKey>>>,
    scope: GrantScope,
    yolo: bool,
}

impl ApprovalGate {
    /// Returns the gate handed to dispatchers plus the receiver the UI drains.
    /// `scope` fixes the blast radius of every `ApproveForSession` answer and
    /// cannot be changed afterwards, so grants can never be reinterpreted more
    /// broadly than the answer that created them.
    pub fn new(yolo: bool, scope: GrantScope) -> (Self, mpsc::Receiver<ApprovalRequest>) {
        let (tx, rx) = mpsc::channel(APPROVAL_QUEUE_DEPTH);
        (
            Self {
                tx,
                session_grants: Arc::new(Mutex::new(HashSet::new())),
                scope,
                yolo,
            },
            rx,
        )
    }

    /// A gate with no UI attached that approves everything. For `--yolo` runs
    /// and tests; the dropped receiver is never consulted because `yolo`
    /// short-circuits before any send.
    pub fn yolo() -> Self {
        let (gate, _rx) = Self::new(true, GrantScope::default());
        gate
    }

    pub fn is_yolo(&self) -> bool {
        self.yolo
    }

    pub fn grant_scope(&self) -> GrantScope {
        self.scope
    }

    pub async fn request(
        &self,
        agent_id: &str,
        tool: &str,
        args: &Value,
    ) -> Result<ApprovalDecision> {
        if self.yolo {
            return Ok(ApprovalDecision::Approve);
        }
        if self.is_granted(agent_id, tool) {
            return Ok(ApprovalDecision::Approve);
        }

        let (respond, answer) = oneshot::channel();
        let request = ApprovalRequest {
            request_id: Uuid::new_v4().to_string(),
            agent_id: agent_id.to_string(),
            tool: tool.to_string(),
            arguments: args.clone(),
            scope: self.scope,
            respond,
        };

        // No UI to ask means no consent: fail closed rather than assuming yes.
        if self.tx.send(request).await.is_err() {
            warn!(tool, "Approval channel closed; denying");
            return Ok(ApprovalDecision::Deny);
        }

        match answer.await {
            Ok(ApprovalDecision::ApproveForSession) => {
                self.grant(agent_id, tool);
                Ok(ApprovalDecision::ApproveForSession)
            }
            Ok(decision) => Ok(decision),
            Err(_) => {
                warn!(
                    tool,
                    "Approval responder dropped without answering; denying"
                );
                Ok(ApprovalDecision::Deny)
            }
        }
    }

    /// Under `GrantScope::Agent` the answer depends on who is asking: another
    /// agent's grant for the same tool does not count.
    pub fn is_granted(&self, agent_id: &str, tool: &str) -> bool {
        self.grants().contains(&self.key(agent_id, tool))
    }

    fn grant(&self, agent_id: &str, tool: &str) {
        let key = self.key(agent_id, tool);
        self.grants().insert(key);
    }

    fn key(&self, agent_id: &str, tool: &str) -> GrantKey {
        GrantKey {
            agent_id: match self.scope {
                GrantScope::Tool => None,
                GrantScope::Agent => Some(agent_id.to_string()),
            },
            tool: tool.to_string(),
        }
    }

    /// A poisoned lock only means some other task panicked mid-update; the set
    /// itself is still coherent, so recover rather than propagating the panic.
    fn grants(&self) -> std::sync::MutexGuard<'_, HashSet<GrantKey>> {
        self.session_grants
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    /// Answers every incoming request with `decision` and reports how many it saw.
    fn spawn_ui(
        mut rx: mpsc::Receiver<ApprovalRequest>,
        decision: ApprovalDecision,
    ) -> tokio::task::JoinHandle<usize> {
        tokio::spawn(async move {
            let mut seen = 0;
            while let Some(req) = rx.recv().await {
                seen += 1;
                assert!(!req.request_id.is_empty());
                let _ = req.respond.send(decision);
            }
            seen
        })
    }

    #[tokio::test]
    async fn approve_round_trips_through_the_channel() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let ui = spawn_ui(rx, ApprovalDecision::Approve);

        let decision = gate
            .request("lead", "run_bash_command", &json!({"cmd": "ls"}))
            .await
            .expect("request");
        assert_eq!(decision, ApprovalDecision::Approve);

        drop(gate);
        assert_eq!(ui.await.expect("ui task"), 1);
    }

    #[tokio::test]
    async fn deny_is_reported_verbatim() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let _ui = spawn_ui(rx, ApprovalDecision::Deny);

        let decision = gate
            .request("lead", "write_file", &json!({}))
            .await
            .unwrap();
        assert_eq!(decision, ApprovalDecision::Deny);
        assert!(!gate.is_granted("lead", "write_file"));
    }

    #[tokio::test]
    async fn approve_for_session_skips_the_ui_on_later_calls() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

        let first = gate
            .request("lead", "write_file", &json!({}))
            .await
            .unwrap();
        assert_eq!(first, ApprovalDecision::ApproveForSession);
        assert!(gate.is_granted("lead", "write_file"));

        let second = gate
            .request("sub-1", "write_file", &json!({}))
            .await
            .unwrap();
        assert_eq!(second, ApprovalDecision::Approve);
        // A different tool is still gated.
        let third = gate
            .request("lead", "run_bash_command", &json!({}))
            .await
            .unwrap();
        assert_eq!(third, ApprovalDecision::ApproveForSession);

        drop(gate);
        // Only the two first-time requests reached the UI.
        assert_eq!(ui.await.expect("ui task"), 2);
    }

    #[tokio::test]
    async fn tool_scope_is_the_default() {
        let (gate, _rx) = ApprovalGate::new(false, GrantScope::default());
        assert_eq!(gate.grant_scope(), GrantScope::Tool);
    }

    #[tokio::test]
    async fn agent_scope_grant_does_not_leak_to_another_agent() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
        let ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

        let first = gate
            .request("lead", "run_bash_command", &json!({}))
            .await
            .unwrap();
        assert_eq!(first, ApprovalDecision::ApproveForSession);
        assert!(gate.is_granted("lead", "run_bash_command"));
        assert!(!gate.is_granted("engineer-1", "run_bash_command"));

        // The granting agent is not asked again.
        let repeat = gate
            .request("lead", "run_bash_command", &json!({}))
            .await
            .unwrap();
        assert_eq!(repeat, ApprovalDecision::Approve);

        // A different agent still has to be asked.
        let other = gate
            .request("engineer-1", "run_bash_command", &json!({}))
            .await
            .unwrap();
        assert_eq!(other, ApprovalDecision::ApproveForSession);

        drop(gate);
        assert_eq!(ui.await.expect("ui task"), 2);
    }

    #[tokio::test]
    async fn tool_scope_grant_covers_every_agent() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

        gate.request("lead", "run_bash_command", &json!({}))
            .await
            .unwrap();
        assert!(gate.is_granted("engineer-1", "run_bash_command"));

        let other = gate
            .request("engineer-1", "run_bash_command", &json!({}))
            .await
            .unwrap();
        assert_eq!(other, ApprovalDecision::Approve);

        drop(gate);
        assert_eq!(ui.await.expect("ui task"), 1);
    }

    #[tokio::test]
    async fn agent_scope_denial_grants_nothing() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
        let _ui = spawn_ui(rx, ApprovalDecision::Deny);

        assert_eq!(
            gate.request("lead", "write_file", &json!({}))
                .await
                .unwrap(),
            ApprovalDecision::Deny
        );
        assert!(!gate.is_granted("lead", "write_file"));
        assert!(!gate.is_granted("engineer-1", "write_file"));
    }

    #[tokio::test]
    async fn dropped_ui_receiver_denies() {
        for scope in [GrantScope::Tool, GrantScope::Agent] {
            let (gate, rx) = ApprovalGate::new(false, scope);
            drop(rx);

            let decision = gate
                .request("lead", "write_file", &json!({}))
                .await
                .unwrap();
            assert_eq!(decision, ApprovalDecision::Deny, "scope {scope:?}");
        }
    }

    #[tokio::test]
    async fn dropped_responder_denies() {
        for scope in [GrantScope::Tool, GrantScope::Agent] {
            let (gate, mut rx) = ApprovalGate::new(false, scope);
            tokio::spawn(async move {
                // Take the request and drop it without ever answering.
                let _req = rx.recv().await;
            });

            let decision = gate
                .request("lead", "write_file", &json!({}))
                .await
                .unwrap();
            assert_eq!(decision, ApprovalDecision::Deny, "scope {scope:?}");
            assert!(!gate.is_granted("lead", "write_file"));
        }
    }

    #[tokio::test]
    async fn yolo_approves_without_a_ui() {
        let gate = ApprovalGate::yolo();
        assert!(gate.is_yolo());
        let decision = gate
            .request("lead", "run_bash_command", &json!({"cmd": "rm -rf /"}))
            .await
            .unwrap();
        assert_eq!(decision, ApprovalDecision::Approve);
    }

    #[tokio::test]
    async fn gate_clones_share_session_grants() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let _ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

        let clone = gate.clone();
        clone
            .request("sub-1", "write_file", &json!({}))
            .await
            .unwrap();
        assert!(gate.is_granted("lead", "write_file"));
    }

    #[tokio::test]
    async fn agent_scope_clones_share_the_granting_agents_grant_only() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
        let _ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

        let clone = gate.clone();
        clone
            .request("sub-1", "write_file", &json!({}))
            .await
            .unwrap();
        assert!(gate.is_granted("sub-1", "write_file"));
        assert!(!gate.is_granted("lead", "write_file"));
    }

    #[tokio::test]
    async fn the_ui_is_told_which_scope_its_answer_will_have() {
        for scope in [GrantScope::Tool, GrantScope::Agent] {
            let (gate, mut rx) = ApprovalGate::new(false, scope);
            let ui = tokio::spawn(async move {
                let req = rx.recv().await.expect("request");
                let seen = req.scope;
                let _ = req.respond.send(ApprovalDecision::Approve);
                seen
            });
            gate.request("lead", "write_file", &json!({}))
                .await
                .unwrap();
            assert_eq!(ui.await.unwrap(), scope);
        }
    }

    /// A grant key built by pasting the two strings together (`"lead-1:bash"`)
    /// would let one identity impersonate another by choosing a name that
    /// straddles the separator. The struct key must make that impossible for
    /// every separator an implementation might have picked.
    #[tokio::test]
    async fn agent_and_tool_cannot_be_confused_across_a_separator() {
        let pairs = [
            (("a", "b:c"), ("a:b", "c")),
            (("a", "b-c"), ("a-b", "c")),
            (("a", "b/c"), ("a/b", "c")),
            (("a", "b\u{0}c"), ("a\u{0}b", "c")),
            (("a", "b\nc"), ("a\nb", "c")),
            (("lead", "1:run_bash"), ("lead:1", "run_bash")),
        ];
        for ((grant_agent, grant_tool), (other_agent, other_tool)) in pairs {
            let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
            let _ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

            gate.request(grant_agent, grant_tool, &json!({}))
                .await
                .unwrap();
            assert!(gate.is_granted(grant_agent, grant_tool));
            assert!(
                !gate.is_granted(other_agent, other_tool),
                "{other_agent:?}/{other_tool:?} rode the grant for \
                 {grant_agent:?}/{grant_tool:?}"
            );
        }
    }

    /// Neighbouring identities: prefixes, suffixes, case, whitespace and the
    /// empty id must all be distinct principals under `Agent` scope.
    #[tokio::test]
    async fn agent_scope_grant_reaches_no_neighbouring_identity() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
        let _ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

        gate.request("lead", "run_bash_command", &json!({}))
            .await
            .unwrap();
        assert!(gate.is_granted("lead", "run_bash_command"));

        for impostor in [
            "",
            " ",
            "lead ",
            " lead",
            "lead\n",
            "Lead",
            "LEAD",
            "lead-1",
            "le",
            "leadx",
            "lead\u{0}",
            "lead\u{0}extra",
            "1ead",
            "l\u{0435}ad", // Cyrillic е
        ] {
            assert!(
                !gate.is_granted(impostor, "run_bash_command"),
                "{impostor:?} was treated as 'lead'"
            );
        }
        // ...and the tool side is exact too.
        for tool in ["run_bash", "run_bash_command2", "RUN_BASH_COMMAND", ""] {
            assert!(
                !gate.is_granted("lead", tool),
                "tool {tool:?} rode the grant"
            );
        }
    }

    /// A UI that says "always" once and "no" afterwards: the one grant must
    /// stick to exactly one agent, and the denials must record nothing, so the
    /// second agent is re-asked every single time.
    #[tokio::test]
    async fn one_always_answer_never_becomes_a_second_agents_consent() {
        let (gate, mut rx) = ApprovalGate::new(false, GrantScope::Agent);
        let asked = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = asked.clone();
        tokio::spawn(async move {
            let mut first = true;
            while let Some(req) = rx.recv().await {
                sink.lock().unwrap().push(req.agent_id.clone());
                let decision = if first {
                    first = false;
                    ApprovalDecision::ApproveForSession
                } else {
                    ApprovalDecision::Deny
                };
                let _ = req.respond.send(decision);
            }
        });

        let seq = ["lead", "engineer-1", "lead", "engineer-1", "engineer-2"];
        let mut decisions = Vec::new();
        for agent in seq {
            decisions.push(
                gate.request(agent, "run_bash_command", &json!({}))
                    .await
                    .unwrap(),
            );
        }

        assert_eq!(
            decisions,
            vec![
                ApprovalDecision::ApproveForSession,
                ApprovalDecision::Deny,
                ApprovalDecision::Approve, // rides its own grant, not re-asked
                ApprovalDecision::Deny,
                ApprovalDecision::Deny,
            ]
        );
        // Everyone but the granting lead's repeat call reached the UI.
        assert_eq!(
            *asked.lock().unwrap(),
            vec!["lead", "engineer-1", "engineer-1", "engineer-2"]
        );
    }

    /// Under `Agent` scope, holding a grant for one agent must not make the
    /// gate fail open for another once the UI disappears.
    #[tokio::test]
    async fn an_existing_grant_does_not_soften_fail_closed_for_others() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
        let ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);
        gate.request("lead", "run_bash_command", &json!({}))
            .await
            .unwrap();

        // The UI goes away with the lead's grant still on the books.
        ui.abort();
        let _ = ui.await;

        assert_eq!(
            gate.request("engineer-1", "run_bash_command", &json!({}))
                .await
                .unwrap(),
            ApprovalDecision::Deny
        );
        assert!(!gate.is_granted("engineer-1", "run_bash_command"));
    }

    /// Concurrent callers must never end up holding someone else's grant, and
    /// no interleaving may leave the set with a key for an agent that was
    /// never asked.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_requests_do_not_cross_grants() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
        let _ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

        let mut handles = Vec::new();
        for i in 0..32 {
            let gate = gate.clone();
            let agent = format!("agent-{}", i % 4);
            handles.push(tokio::spawn(async move {
                gate.request(&agent, "run_bash_command", &json!({}))
                    .await
                    .unwrap()
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        for i in 0..4 {
            assert!(gate.is_granted(&format!("agent-{i}"), "run_bash_command"));
        }
        for i in 4..8 {
            assert!(
                !gate.is_granted(&format!("agent-{i}"), "run_bash_command"),
                "agent-{i} was never asked yet holds a grant"
            );
        }
        assert!(!gate.is_granted("agent-0", "write_file"));
    }

    /// A panic while the grant set is locked must not wedge the gate, and must
    /// not invent a grant either.
    #[tokio::test]
    async fn a_poisoned_grant_set_stays_closed_rather_than_open() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
        let _ui = spawn_ui(rx, ApprovalDecision::Deny);

        let grants = gate.session_grants.clone();
        let _ = std::thread::spawn(move || {
            let _guard = grants.lock().unwrap();
            panic!("poison the lock");
        })
        .join();

        assert!(gate.session_grants.is_poisoned());
        assert!(!gate.is_granted("lead", "run_bash_command"));
        assert_eq!(
            gate.request("lead", "run_bash_command", &json!({}))
                .await
                .unwrap(),
            ApprovalDecision::Deny
        );
    }

    /// Huge and adversarial ids/tool names are data, not code paths: they must
    /// key normally instead of panicking or truncating into a collision.
    #[tokio::test]
    async fn pathological_identifiers_do_not_panic_or_collide() {
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
        let _ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

        let long_a = "x".repeat(100_000);
        let long_b = format!("{}y", "x".repeat(99_999));
        gate.request(&long_a, "run_bash_command", &json!({}))
            .await
            .unwrap();
        assert!(gate.is_granted(&long_a, "run_bash_command"));
        assert!(!gate.is_granted(&long_b, "run_bash_command"));

        // Non-UTF8-lookalike and emoji ids round-trip without incident.
        for id in ["🙂", "\u{202e}lead", "lead\u{200b}"] {
            assert!(!gate.is_granted(id, "run_bash_command"));
        }
    }
}
