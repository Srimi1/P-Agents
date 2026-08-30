//! Fan-out bus. The pump task republishes every agent event here so any number
//! of renderers (REPL, logger, future TUI) can watch a run.

use agent_core::events::AgentEvent;
use agent_core::types::{AgentState, ChatMessage, TokenUsage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::broadcast;

/// Mirrors `AgentEvent` one-for-one.
///
/// Approval requests deliberately do not travel on this bus: they need a reply,
/// so they go to exactly one responder over `ApprovalGate`'s channel rather than
/// being broadcast to every subscriber.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HarnessEvent {
    StateChanged {
        agent_id: String,
        state: AgentState,
    },
    TextDelta {
        agent_id: String,
        delta: String,
    },
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
    Compacted {
        agent_id: String,
        messages_before: usize,
        messages_after: usize,
    },
}

impl From<AgentEvent> for HarnessEvent {
    fn from(event: AgentEvent) -> Self {
        match event {
            AgentEvent::StateChanged { agent_id, state } => {
                HarnessEvent::StateChanged { agent_id, state }
            }
            AgentEvent::TextDelta { agent_id, delta } => {
                HarnessEvent::TextDelta { agent_id, delta }
            }
            AgentEvent::MessageAppended { agent_id, message } => {
                HarnessEvent::MessageAppended { agent_id, message }
            }
            AgentEvent::ToolStarted {
                agent_id,
                tool,
                arguments,
            } => HarnessEvent::ToolStarted {
                agent_id,
                tool,
                arguments,
            },
            AgentEvent::ToolFinished {
                agent_id,
                tool,
                preview,
                is_error,
            } => HarnessEvent::ToolFinished {
                agent_id,
                tool,
                preview,
                is_error,
            },
            AgentEvent::UsageReport {
                agent_id,
                turn,
                cumulative,
            } => HarnessEvent::UsageReport {
                agent_id,
                turn,
                cumulative,
            },
            AgentEvent::SubAgentSpawned {
                parent_id,
                agent_id,
                role,
            } => HarnessEvent::SubAgentSpawned {
                parent_id,
                agent_id,
                role,
            },
            AgentEvent::SubAgentFinished { agent_id, ok } => {
                HarnessEvent::SubAgentFinished { agent_id, ok }
            }
            AgentEvent::Compacted {
                agent_id,
                messages_before,
                messages_after,
            } => HarnessEvent::Compacted {
                agent_id,
                messages_before,
                messages_after,
            },
        }
    }
}

pub struct EventBus {
    sender: broadcast::Sender<HarnessEvent>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Dropped when nobody is subscribed; a headless run is not an error.
    pub fn publish(&self, event: HarnessEvent) {
        let _ = self.sender.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<HarnessEvent> {
        self.sender.subscribe()
    }

    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::types::Role;
    use serde_json::json;

    #[test]
    fn every_agent_event_maps_to_a_harness_event() {
        let cases = vec![
            AgentEvent::StateChanged {
                agent_id: "a".into(),
                state: AgentState::Planning,
            },
            AgentEvent::TextDelta {
                agent_id: "a".into(),
                delta: "hi".into(),
            },
            AgentEvent::MessageAppended {
                agent_id: "a".into(),
                message: ChatMessage::user("hello"),
            },
            AgentEvent::ToolStarted {
                agent_id: "a".into(),
                tool: "read_file".into(),
                arguments: json!({"path": "x"}),
            },
            AgentEvent::ToolFinished {
                agent_id: "a".into(),
                tool: "read_file".into(),
                preview: "ok".into(),
                is_error: false,
            },
            AgentEvent::UsageReport {
                agent_id: "a".into(),
                turn: TokenUsage::new(1, 2),
                cumulative: TokenUsage::new(3, 4),
            },
            AgentEvent::SubAgentSpawned {
                parent_id: "a".into(),
                agent_id: "b".into(),
                role: "researcher".into(),
            },
            AgentEvent::SubAgentFinished {
                agent_id: "b".into(),
                ok: true,
            },
            AgentEvent::Compacted {
                agent_id: "a".into(),
                messages_before: 10,
                messages_after: 4,
            },
        ];

        for case in cases {
            let id = case.agent_id().to_string();
            let mapped = HarnessEvent::from(case);
            let encoded = serde_json::to_string(&mapped).expect("serialize");
            assert!(encoded.contains(&id), "lost agent id in {encoded}");
            serde_json::from_str::<HarnessEvent>(&encoded).expect("round trip");
        }
    }

    #[test]
    fn message_payload_survives_conversion() {
        let mapped = HarnessEvent::from(AgentEvent::MessageAppended {
            agent_id: "lead".into(),
            message: ChatMessage::assistant("done"),
        });
        match mapped {
            HarnessEvent::MessageAppended { agent_id, message } => {
                assert_eq!(agent_id, "lead");
                assert_eq!(message.role, Role::Assistant);
                assert_eq!(message.content.as_deref(), Some("done"));
            }
            other => panic!("unexpected mapping: {other:?}"),
        }
    }

    #[test]
    fn bus_fans_out_to_all_subscribers() {
        let bus = EventBus::new(8);
        let mut first = bus.subscribe();
        let mut second = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);

        bus.publish(HarnessEvent::TextDelta {
            agent_id: "a".into(),
            delta: "tok".into(),
        });

        for rx in [&mut first, &mut second] {
            match rx.try_recv().expect("delivered") {
                HarnessEvent::TextDelta { delta, .. } => assert_eq!(delta, "tok"),
                other => panic!("unexpected event: {other:?}"),
            }
        }
    }

    #[test]
    fn publishing_without_subscribers_is_not_an_error() {
        let bus = EventBus::new(4);
        bus.publish(HarnessEvent::SubAgentFinished {
            agent_id: "b".into(),
            ok: false,
        });
        assert_eq!(bus.subscriber_count(), 0);
    }
}
