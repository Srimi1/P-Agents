use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarnessEvent {
    AgentStateChanged {
        agent_id: String,
        state: String,
    },
    ToolStarted {
        agent_id: String,
        tool: String,
        arguments: serde_json::Value,
    },
    ToolFinished {
        agent_id: String,
        tool: String,
        result_preview: String,
    },
    SubAgentSpawned {
        parent_id: String,
        role: String,
    },
    TokenChunk {
        agent_id: String,
        chunk: String,
    },
    ApprovalRequired {
        request_id: String,
        tool: String,
        arguments: serde_json::Value,
    },
}

pub struct EventBus {
    sender: broadcast::Sender<HarnessEvent>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    pub fn publish(&self, event: HarnessEvent) {
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<HarnessEvent> {
        self.sender.subscribe()
    }
}
