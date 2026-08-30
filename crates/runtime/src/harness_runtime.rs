//! Runtime assembly: event bus, approval endpoints, session store, pump task.

use crate::approval::{ApprovalGate, ApprovalRequest};
use crate::event_bus::{EventBus, HarnessEvent};
use crate::gated_dispatcher::GatedDispatcher;
use crate::security::SecurityManager;
use crate::session::{SessionRecord, SessionStore};
use agent_core::events::{AgentEvent, EventSink};
use anyhow::Result;
use harness_core::HarnessToolRegistry;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::error;

/// Broadcast backlog. Slow subscribers lag rather than block the pump.
const BUS_CAPACITY: usize = 1024;

pub struct HarnessRuntime {
    bus: Arc<EventBus>,
    gate: ApprovalGate,
    security: Arc<SecurityManager>,
    sink: EventSink,
    stop: oneshot::Sender<()>,
    pump: JoinHandle<()>,
    session_id: String,
    session_path: PathBuf,
}

impl HarnessRuntime {
    pub async fn new(
        session_dir: &Path,
        model: &str,
        security: SecurityManager,
        yolo: bool,
    ) -> Result<(Self, mpsc::Receiver<ApprovalRequest>)> {
        let yolo = yolo || security.is_yolo();
        let security = Arc::new(security.with_yolo(yolo));
        let (gate, approvals) = ApprovalGate::new(yolo);
        let bus = Arc::new(EventBus::new(BUS_CAPACITY));

        let store = SessionStore::create(session_dir, model).await?;
        let session_id = store.session_id().to_string();
        let session_path = store.path().to_path_buf();

        let (sink, events) = mpsc::unbounded_channel::<AgentEvent>();
        let (stop, stop_rx) = oneshot::channel();
        let pump = tokio::spawn(pump(events, stop_rx, store, bus.clone()));

        Ok((
            Self {
                bus,
                gate,
                security,
                sink,
                stop,
                pump,
                session_id,
                session_path,
            },
            approvals,
        ))
    }

    /// Handed to every agent; cloning it is how sub-agents report in.
    pub fn event_sink(&self) -> EventSink {
        self.sink.clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<HarnessEvent> {
        self.bus.subscribe()
    }

    /// For events that originate outside an agent, e.g. `ApprovalRequested`.
    pub fn publish(&self, event: HarnessEvent) {
        self.bus.publish(event);
    }

    pub fn gate(&self) -> ApprovalGate {
        self.gate.clone()
    }

    pub fn security(&self) -> Arc<SecurityManager> {
        self.security.clone()
    }

    pub fn bus(&self) -> Arc<EventBus> {
        self.bus.clone()
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn session_path(&self) -> &Path {
        &self.session_path
    }

    /// The one dispatcher shared by the lead agent and all sub-agents.
    pub fn dispatcher(&self, registry: Arc<HarnessToolRegistry>) -> Arc<GatedDispatcher> {
        Arc::new(
            GatedDispatcher::new(registry, self.security(), self.gate())
                .with_events(self.sink.clone()),
        )
    }

    /// Stops the pump, letting it drain what is already queued, so the session
    /// file is complete on disk before the process exits.
    ///
    /// Waiting for the event channel to close would not work: every agent and
    /// the gated dispatcher hold an `EventSink` clone, and those outlive the
    /// runtime handle, so the channel is still open here. An explicit stop
    /// signal makes shutdown terminate regardless of who is still holding one.
    pub async fn shutdown(self) {
        let Self {
            sink, stop, pump, ..
        } = self;
        drop(sink);
        let _ = stop.send(());
        if let Err(err) = pump.await {
            error!(error = %err, "Session pump task failed");
        }
    }
}

/// Single writer for the session log: no lock contention, and records land in
/// the order the agents produced them.
async fn pump(
    mut events: mpsc::UnboundedReceiver<AgentEvent>,
    mut stop: oneshot::Receiver<()>,
    mut store: SessionStore,
    bus: Arc<EventBus>,
) {
    loop {
        // Biased so queued events are always preferred over the stop signal;
        // stop only wins once the channel is momentarily empty.
        tokio::select! {
            biased;

            event = events.recv() => match event {
                Some(event) => handle(event, &mut store, &bus).await,
                None => break,
            },
            _ = &mut stop => {
                while let Ok(event) = events.try_recv() {
                    handle(event, &mut store, &bus).await;
                }
                break;
            }
        }
    }
}

async fn handle(event: AgentEvent, store: &mut SessionStore, bus: &EventBus) {
    let record = match &event {
        AgentEvent::MessageAppended { agent_id, message } => Some(SessionRecord::Message {
            agent_id: agent_id.clone(),
            message: message.clone(),
        }),
        AgentEvent::UsageReport { agent_id, turn, .. } => Some(SessionRecord::Usage {
            agent_id: agent_id.clone(),
            usage: *turn,
        }),
        _ => None,
    };

    if let Some(record) = record {
        // A failed write must not take the run down; the UI still gets
        // everything on the bus.
        if let Err(err) = store.append(&record).await {
            error!(error = %err, "Failed to append session record");
        }
    }

    bus.publish(HarnessEvent::from(event));
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::types::{AgentState, ChatMessage, TokenUsage};
    use agent_core::{ToolCall, ToolDispatcher};
    use harness_core::Tool;
    use serde_json::json;

    struct EchoTool;

    #[async_trait::async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echo"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }
        fn requires_approval(&self) -> bool {
            true
        }
        async fn execute(&self, _args: serde_json::Value) -> Result<String> {
            Ok("echoed".into())
        }
    }

    async fn runtime(dir: &Path) -> (HarnessRuntime, mpsc::Receiver<ApprovalRequest>) {
        HarnessRuntime::new(dir, "test-model", SecurityManager::new(), false)
            .await
            .expect("runtime")
    }

    #[tokio::test]
    async fn pump_persists_messages_and_usage_and_shutdown_flushes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (rt, _approvals) = runtime(dir.path()).await;
        let path = rt.session_path().to_path_buf();
        let sink = rt.event_sink();

        sink.send(AgentEvent::MessageAppended {
            agent_id: "lead".into(),
            message: ChatMessage::user("hi"),
        })
        .unwrap();
        sink.send(AgentEvent::UsageReport {
            agent_id: "lead".into(),
            turn: TokenUsage::new(7, 3),
            cumulative: TokenUsage::new(7, 3),
        })
        .unwrap();
        // Not persisted, only broadcast.
        sink.send(AgentEvent::TextDelta {
            agent_id: "lead".into(),
            delta: "tok".into(),
        })
        .unwrap();
        drop(sink);

        rt.shutdown().await;

        let records = SessionStore::load(&path).await.unwrap();
        assert_eq!(records.len(), 3, "meta + message + usage");
        let history = SessionStore::rebuild_history(&records, "lead");
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].content.as_deref(), Some("hi"));
        match &records[2] {
            SessionRecord::Usage { usage, .. } => assert_eq!(usage.total_tokens, 10),
            other => panic!("expected usage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn agent_events_are_republished_on_the_bus() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (rt, _approvals) = runtime(dir.path()).await;
        let mut sub = rt.subscribe();
        let sink = rt.event_sink();

        sink.send(AgentEvent::StateChanged {
            agent_id: "lead".into(),
            state: AgentState::Planning,
        })
        .unwrap();

        match sub.recv().await.expect("event") {
            HarnessEvent::StateChanged { agent_id, state } => {
                assert_eq!(agent_id, "lead");
                assert_eq!(state, AgentState::Planning);
            }
            other => panic!("unexpected event: {other:?}"),
        }

        rt.shutdown().await;
    }

    #[tokio::test]
    async fn runtime_dispatcher_routes_through_the_approval_gate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (rt, mut approvals) = runtime(dir.path()).await;

        let mut registry = HarnessToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let dispatcher = rt.dispatcher(Arc::new(registry));

        let ui = tokio::spawn(async move {
            let req = approvals.recv().await.expect("approval request");
            assert_eq!(req.tool, "echo");
            assert_eq!(req.agent_id, "sub-1");
            let _ = req.respond.send(crate::approval::ApprovalDecision::Deny);
        });

        let out = dispatcher
            .dispatch(
                "sub-1",
                &ToolCall {
                    id: "1".into(),
                    name: "echo".into(),
                    arguments: json!({}),
                },
            )
            .await
            .unwrap();
        assert!(out.starts_with("DENIED by user"));

        ui.await.expect("ui task");
        rt.shutdown().await;
    }

    /// Every agent and the gated dispatcher hold a sink clone that outlives the
    /// runtime handle. Shutdown must not wait for those to be dropped.
    #[tokio::test]
    async fn shutdown_completes_while_sink_clones_are_still_alive() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (rt, _approvals) = runtime(dir.path()).await;
        let path = rt.session_path().to_path_buf();

        let held_by_an_agent = rt.event_sink();
        let held_by_a_dispatcher = rt.dispatcher(Arc::new(HarnessToolRegistry::new()));

        held_by_an_agent
            .send(AgentEvent::MessageAppended {
                agent_id: "lead".into(),
                message: ChatMessage::user("queued before shutdown"),
            })
            .unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(5), rt.shutdown())
            .await
            .expect("shutdown must not hang while sink clones are alive");

        // Still alive, which is the whole point of the test.
        drop(held_by_an_agent);
        drop(held_by_a_dispatcher);

        // The queued event was drained rather than lost.
        let records = SessionStore::load(&path).await.unwrap();
        assert!(
            records.iter().any(|r| matches!(
                r,
                SessionRecord::Message { message, .. }
                    if message.content.as_deref() == Some("queued before shutdown")
            )),
            "shutdown should flush what was already queued"
        );
    }

    #[tokio::test]
    async fn yolo_flag_reaches_gate_and_policy() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (rt, _approvals) =
            HarnessRuntime::new(dir.path(), "test-model", SecurityManager::new(), true)
                .await
                .unwrap();

        assert!(rt.gate().is_yolo());
        assert!(rt.security().is_yolo());
        assert!(!rt.security().needs_approval("echo", true));
        rt.shutdown().await;
    }

    #[tokio::test]
    async fn session_id_matches_the_written_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (rt, _approvals) = runtime(dir.path()).await;
        let id = rt.session_id().to_string();
        let found = SessionStore::find_by_id(dir.path(), &id).await.unwrap();
        assert_eq!(found, rt.session_path());
        rt.shutdown().await;
    }
}
