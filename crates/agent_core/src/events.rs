use crate::types::{AgentState, ChatMessage, TokenUsage};
use serde_json::Value;
use tokio::sync::mpsc::UnboundedSender;

/// Everything an agent reports about itself while it runs. Emitted from
/// `Agent::run` and consumed by the runtime pump, which mirrors it onto the
/// broadcast bus and appends the durable records to the session log.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    StateChanged {
        agent_id: String,
        state: AgentState,
    },
    /// One chunk of streamed assistant text.
    TextDelta {
        agent_id: String,
        delta: String,
    },
    /// A message was appended to the agent's history. Drives session persistence.
    MessageAppended {
        agent_id: String,
        message: ChatMessage,
    },
    ToolStarted {
        agent_id: String,
        tool: String,
        arguments: Value,
    },
    ToolFinished {
        agent_id: String,
        tool: String,
        preview: String,
        is_error: bool,
    },
    UsageReport {
        agent_id: String,
        turn: TokenUsage,
        cumulative: TokenUsage,
    },
    SubAgentSpawned {
        parent_id: String,
        agent_id: String,
        role: String,
    },
    SubAgentFinished {
        agent_id: String,
        ok: bool,
    },
    /// History was compacted to stay inside the context budget.
    Compacted {
        agent_id: String,
        messages_before: usize,
        messages_after: usize,
    },
}

impl AgentEvent {
    /// The agent this event is about (for `SubAgentSpawned`, the child).
    pub fn agent_id(&self) -> &str {
        match self {
            AgentEvent::StateChanged { agent_id, .. }
            | AgentEvent::TextDelta { agent_id, .. }
            | AgentEvent::MessageAppended { agent_id, .. }
            | AgentEvent::ToolStarted { agent_id, .. }
            | AgentEvent::ToolFinished { agent_id, .. }
            | AgentEvent::UsageReport { agent_id, .. }
            | AgentEvent::SubAgentSpawned { agent_id, .. }
            | AgentEvent::SubAgentFinished { agent_id, .. }
            | AgentEvent::Compacted { agent_id, .. } => agent_id,
        }
    }
}

/// Unbounded so an agent never blocks on a slow renderer. Previews are
/// truncated at the source to keep the queue small.
pub type EventSink = UnboundedSender<AgentEvent>;

/// Fire-and-forget send. A closed channel means the UI is gone; agents keep
/// working rather than failing the task over a dropped receiver.
pub fn emit(sink: &Option<EventSink>, event: AgentEvent) {
    if let Some(tx) = sink {
        let _ = tx.send(event);
    }
}
