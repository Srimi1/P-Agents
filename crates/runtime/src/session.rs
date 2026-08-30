//! Append-only JSONL session log. Stub — see the runtime work package.

use agent_core::types::{ChatMessage, TokenUsage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionRecord {
    Meta {
        session_id: String,
        created_at: u64,
        model: String,
    },
    Message {
        agent_id: String,
        message: ChatMessage,
    },
    Usage {
        agent_id: String,
        usage: TokenUsage,
    },
}

pub struct SessionStore;
