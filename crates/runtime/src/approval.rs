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

pub struct ApprovalRequest {
    pub request_id: String,
    pub agent_id: String,
    pub tool: String,
    pub arguments: Value,
    pub respond: oneshot::Sender<ApprovalDecision>,
}

#[derive(Clone)]
pub struct ApprovalGate {
    tx: mpsc::Sender<ApprovalRequest>,
    /// Tools the user has blanket-approved for the rest of the session.
    session_grants: Arc<Mutex<HashSet<String>>>,
    yolo: bool,
}

impl ApprovalGate {
    /// Returns the gate handed to dispatchers plus the receiver the UI drains.
    pub fn new(yolo: bool) -> (Self, mpsc::Receiver<ApprovalRequest>) {
        let (tx, rx) = mpsc::channel(APPROVAL_QUEUE_DEPTH);
        (
            Self {
                tx,
                session_grants: Arc::new(Mutex::new(HashSet::new())),
                yolo,
            },
            rx,
        )
    }

    /// A gate with no UI attached that approves everything. For `--yolo` runs
    /// and tests; the dropped receiver is never consulted because `yolo`
    /// short-circuits before any send.
    pub fn yolo() -> Self {
        let (gate, _rx) = Self::new(true);
        gate
    }

    pub fn is_yolo(&self) -> bool {
        self.yolo
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
        if self.is_granted(tool) {
            return Ok(ApprovalDecision::Approve);
        }

        let (respond, answer) = oneshot::channel();
        let request = ApprovalRequest {
            request_id: Uuid::new_v4().to_string(),
            agent_id: agent_id.to_string(),
            tool: tool.to_string(),
            arguments: args.clone(),
            respond,
        };

        // No UI to ask means no consent: fail closed rather than assuming yes.
        if self.tx.send(request).await.is_err() {
            warn!(tool, "Approval channel closed; denying");
            return Ok(ApprovalDecision::Deny);
        }

        match answer.await {
            Ok(ApprovalDecision::ApproveForSession) => {
                self.grant(tool);
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

    pub fn is_granted(&self, tool: &str) -> bool {
        self.grants().contains(tool)
    }

    fn grant(&self, tool: &str) {
        self.grants().insert(tool.to_string());
    }

    /// A poisoned lock only means some other task panicked mid-update; the set
    /// itself is still coherent, so recover rather than propagating the panic.
    fn grants(&self) -> std::sync::MutexGuard<'_, HashSet<String>> {
        self.session_grants
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
        let (gate, rx) = ApprovalGate::new(false);
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
        let (gate, rx) = ApprovalGate::new(false);
        let _ui = spawn_ui(rx, ApprovalDecision::Deny);

        let decision = gate
            .request("lead", "write_file", &json!({}))
            .await
            .unwrap();
        assert_eq!(decision, ApprovalDecision::Deny);
        assert!(!gate.is_granted("write_file"));
    }

    #[tokio::test]
    async fn approve_for_session_skips_the_ui_on_later_calls() {
        let (gate, rx) = ApprovalGate::new(false);
        let ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

        let first = gate
            .request("lead", "write_file", &json!({}))
            .await
            .unwrap();
        assert_eq!(first, ApprovalDecision::ApproveForSession);
        assert!(gate.is_granted("write_file"));

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
    async fn dropped_ui_receiver_denies() {
        let (gate, rx) = ApprovalGate::new(false);
        drop(rx);

        let decision = gate
            .request("lead", "write_file", &json!({}))
            .await
            .unwrap();
        assert_eq!(decision, ApprovalDecision::Deny);
    }

    #[tokio::test]
    async fn dropped_responder_denies() {
        let (gate, mut rx) = ApprovalGate::new(false);
        tokio::spawn(async move {
            // Take the request and drop it without ever answering.
            let _req = rx.recv().await;
        });

        let decision = gate
            .request("lead", "write_file", &json!({}))
            .await
            .unwrap();
        assert_eq!(decision, ApprovalDecision::Deny);
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
        let (gate, rx) = ApprovalGate::new(false);
        let _ui = spawn_ui(rx, ApprovalDecision::ApproveForSession);

        let clone = gate.clone();
        clone
            .request("sub-1", "write_file", &json!({}))
            .await
            .unwrap();
        assert!(gate.is_granted("write_file"));
    }
}
