//! Human-in-the-loop approval gate. Stub — see the runtime work package.

use serde_json::Value;
use tokio::sync::oneshot;

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

#[derive(Clone, Default)]
pub struct ApprovalGate;
