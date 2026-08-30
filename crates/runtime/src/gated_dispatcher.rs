//! Approval-enforcing tool dispatcher. Shared by the lead agent and every
//! sub-agent, so a sub-agent cannot route around the gate.

use crate::approval::{ApprovalDecision, ApprovalGate};
use crate::security::SecurityManager;
use agent_core::events::{emit, AgentEvent, EventSink};
use agent_core::types::AgentState;
use agent_core::{ToolCall, ToolDefinition, ToolDispatcher};
use anyhow::Result;
use async_trait::async_trait;
use harness_core::HarnessToolRegistry;
use std::sync::Arc;
use tracing::info;

pub struct GatedDispatcher {
    inner: Arc<HarnessToolRegistry>,
    security: Arc<SecurityManager>,
    gate: ApprovalGate,
    /// `Agent::run` already brackets every dispatch with ToolStarted/ToolFinished,
    /// so emitting those here would double them in the transcript. The sink is
    /// kept for approval-specific signalling only (the WaitingForApproval state,
    /// which the agent cannot know about).
    events: Option<EventSink>,
}

impl GatedDispatcher {
    pub fn new(
        inner: Arc<HarnessToolRegistry>,
        security: Arc<SecurityManager>,
        gate: ApprovalGate,
    ) -> Self {
        Self {
            inner,
            security,
            gate,
            events: None,
        }
    }

    pub fn with_events(mut self, sink: EventSink) -> Self {
        self.events = Some(sink);
        self
    }

    // Deliberately no accessor for the inner registry. `HarnessToolRegistry` is
    // itself a `ToolDispatcher`, so handing it out would let any holder of a
    // gated dispatcher obtain an ungated one and skip approval entirely.
}

fn denied_observation(tool: &str) -> String {
    format!(
        "DENIED by user: the '{tool}' call was not approved. Do not retry it; choose a different approach or ask the user."
    )
}

#[async_trait]
impl ToolDispatcher for GatedDispatcher {
    fn get_definitions(&self) -> Vec<ToolDefinition> {
        self.inner.get_definitions()
    }

    async fn dispatch(&self, agent_id: &str, tool_call: &ToolCall) -> Result<String> {
        let tool = self.inner.get_tool(&tool_call.name).ok_or_else(|| {
            anyhow::anyhow!(
                "Tool '{}' is not registered. Call one of the tools listed in your tool definitions.",
                tool_call.name
            )
        })?;

        if self
            .security
            .needs_approval(&tool_call.name, tool.requires_approval())
        {
            emit(
                &self.events,
                AgentEvent::StateChanged {
                    agent_id: agent_id.to_string(),
                    state: AgentState::WaitingForApproval,
                },
            );
            let decision = self
                .gate
                .request(agent_id, &tool_call.name, &tool_call.arguments)
                .await?;

            if decision == ApprovalDecision::Deny {
                // Info, not warn: a denial is the user exercising the gate, and
                // the REPL shows warnings by default.
                info!(agent = agent_id, tool = %tool_call.name, "Tool call denied by user");
                // An Ok observation, not an Err: the ReAct loop feeds it back to
                // the model, which then adapts instead of surfacing a failure.
                return Ok(denied_observation(&tool_call.name));
            }
            // Only announce execution once the call is actually going to run;
            // a denial leaves the agent in WaitingForApproval for agent.rs to
            // move on from, rather than claiming a tool ran that never did.
            emit(
                &self.events,
                AgentEvent::StateChanged {
                    agent_id: agent_id.to_string(),
                    state: AgentState::ExecutingTool,
                },
            );
            info!(agent = agent_id, tool = %tool_call.name, ?decision, "Tool call approved");
        }

        self.inner.dispatch(agent_id, tool_call).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approval::{ApprovalRequest, GrantScope};
    use harness_core::Tool;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use tokio::sync::mpsc;

    struct CountingTool {
        name: &'static str,
        requires_approval: bool,
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    impl CountingTool {
        fn new(name: &'static str, requires_approval: bool) -> Self {
            Self {
                name,
                requires_approval,
                calls: Arc::new(AtomicUsize::new(0)),
                fail: false,
            }
        }
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            "test tool"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            json!({"type": "object", "properties": {}})
        }
        fn requires_approval(&self) -> bool {
            self.requires_approval
        }
        async fn execute(&self, _args: serde_json::Value) -> Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.fail {
                anyhow::bail!("tool blew up");
            }
            Ok(format!("{} ran", self.name))
        }
    }

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: "call-1".into(),
            name: name.into(),
            arguments: json!({"x": 1}),
        }
    }

    fn spawn_ui(
        mut rx: mpsc::Receiver<ApprovalRequest>,
        decision: ApprovalDecision,
        seen: Arc<AtomicUsize>,
    ) {
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                seen.fetch_add(1, Ordering::SeqCst);
                let _ = req.respond.send(decision);
            }
        });
    }

    /// Like `spawn_ui`, but records which agent each prompt was raised for.
    fn spawn_recording_ui(
        mut rx: mpsc::Receiver<ApprovalRequest>,
        decision: ApprovalDecision,
    ) -> Arc<Mutex<Vec<String>>> {
        let asked: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = asked.clone();
        tokio::spawn(async move {
            while let Some(req) = rx.recv().await {
                if let Ok(mut seen) = sink.lock() {
                    seen.push(req.agent_id.clone());
                }
                let _ = req.respond.send(decision);
            }
        });
        asked
    }

    fn asked_agents(asked: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
        asked.lock().map(|seen| seen.clone()).unwrap_or_default()
    }

    /// Registry with one dangerous tool and one safe tool; returns their counters.
    fn registry() -> (Arc<HarnessToolRegistry>, Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let dangerous = CountingTool::new("run_bash_command", true);
        let safe = CountingTool::new("read_file", false);
        let dangerous_calls = dangerous.calls.clone();
        let safe_calls = safe.calls.clone();
        let mut reg = HarnessToolRegistry::new();
        reg.register(Arc::new(dangerous));
        reg.register(Arc::new(safe));
        (Arc::new(reg), dangerous_calls, safe_calls)
    }

    #[tokio::test]
    async fn denial_becomes_an_ok_observation() {
        let (reg, dangerous_calls, _) = registry();
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let seen = Arc::new(AtomicUsize::new(0));
        spawn_ui(rx, ApprovalDecision::Deny, seen.clone());

        let dispatcher = GatedDispatcher::new(reg, Arc::new(SecurityManager::new()), gate);
        let out = dispatcher
            .dispatch("lead", &call("run_bash_command"))
            .await
            .expect("denial must not be an Err");

        assert!(out.starts_with("DENIED by user: the 'run_bash_command' call"));
        assert!(out.contains("Do not retry it"));
        assert_eq!(dangerous_calls.load(Ordering::SeqCst), 0);
        assert_eq!(seen.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn approval_lets_the_tool_run() {
        let (reg, dangerous_calls, _) = registry();
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let seen = Arc::new(AtomicUsize::new(0));
        spawn_ui(rx, ApprovalDecision::Approve, seen.clone());

        let dispatcher = GatedDispatcher::new(reg, Arc::new(SecurityManager::new()), gate);
        let out = dispatcher
            .dispatch("lead", &call("run_bash_command"))
            .await
            .unwrap();
        assert_eq!(out, "run_bash_command ran");
        assert_eq!(dangerous_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn auto_approved_tool_never_reaches_the_gate() {
        let (reg, dangerous_calls, _) = registry();
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let seen = Arc::new(AtomicUsize::new(0));
        spawn_ui(rx, ApprovalDecision::Deny, seen.clone());

        let security = SecurityManager::new().with_auto_approved(["run_bash_command"]);
        let dispatcher = GatedDispatcher::new(reg, Arc::new(security), gate);
        let out = dispatcher
            .dispatch("lead", &call("run_bash_command"))
            .await
            .unwrap();

        assert_eq!(out, "run_bash_command ran");
        assert_eq!(dangerous_calls.load(Ordering::SeqCst), 1);
        assert_eq!(seen.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn tool_that_does_not_require_approval_is_not_gated() {
        let (reg, _, safe_calls) = registry();
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let seen = Arc::new(AtomicUsize::new(0));
        spawn_ui(rx, ApprovalDecision::Deny, seen.clone());

        let dispatcher = GatedDispatcher::new(reg, Arc::new(SecurityManager::new()), gate);
        assert_eq!(
            dispatcher
                .dispatch("lead", &call("read_file"))
                .await
                .unwrap(),
            "read_file ran"
        );
        assert_eq!(safe_calls.load(Ordering::SeqCst), 1);
        assert_eq!(seen.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn yolo_security_bypasses_the_gate() {
        let (reg, dangerous_calls, _) = registry();
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        drop(rx);

        let security = SecurityManager::new().with_yolo(true);
        let dispatcher = GatedDispatcher::new(reg, Arc::new(security), gate);
        dispatcher
            .dispatch("sub-1", &call("run_bash_command"))
            .await
            .unwrap();
        assert_eq!(dangerous_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn missing_ui_denies_rather_than_running_the_tool() {
        let (reg, dangerous_calls, _) = registry();
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        drop(rx);

        let dispatcher = GatedDispatcher::new(reg, Arc::new(SecurityManager::new()), gate);
        let out = dispatcher
            .dispatch("lead", &call("run_bash_command"))
            .await
            .unwrap();
        assert!(out.starts_with("DENIED by user"));
        assert_eq!(dangerous_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn unknown_tool_errors() {
        let (reg, _, _) = registry();
        let dispatcher =
            GatedDispatcher::new(reg, Arc::new(SecurityManager::new()), ApprovalGate::yolo());
        let err = dispatcher
            .dispatch("lead", &call("no_such_tool"))
            .await
            .expect_err("unknown tool must error");
        assert!(err.to_string().contains("no_such_tool"));
    }

    #[tokio::test]
    async fn tool_errors_propagate() {
        let mut failing = CountingTool::new("run_bash_command", false);
        failing.fail = true;
        let mut reg = HarnessToolRegistry::new();
        reg.register(Arc::new(failing));
        let dispatcher = GatedDispatcher::new(
            Arc::new(reg),
            Arc::new(SecurityManager::new()),
            ApprovalGate::yolo(),
        );
        let err = dispatcher
            .dispatch("lead", &call("run_bash_command"))
            .await
            .expect_err("tool failure must propagate");
        assert!(err.to_string().contains("tool blew up"));
    }

    #[tokio::test]
    async fn approval_emits_waiting_state_on_the_sink() {
        let (reg, _, _) = registry();
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let seen = Arc::new(AtomicUsize::new(0));
        spawn_ui(rx, ApprovalDecision::Approve, seen);

        let (tx, mut events) = tokio::sync::mpsc::unbounded_channel();
        let dispatcher =
            GatedDispatcher::new(reg, Arc::new(SecurityManager::new()), gate).with_events(tx);
        dispatcher
            .dispatch("lead", &call("run_bash_command"))
            .await
            .unwrap();

        let states: Vec<AgentState> = std::iter::from_fn(|| events.try_recv().ok())
            .filter_map(|e| match e {
                AgentEvent::StateChanged { state, .. } => Some(state),
                _ => None,
            })
            .collect();
        assert_eq!(
            states,
            vec![AgentState::WaitingForApproval, AgentState::ExecutingTool]
        );
    }

    #[tokio::test]
    async fn get_definitions_delegates_to_the_registry() {
        let (reg, _, _) = registry();
        let dispatcher = GatedDispatcher::new(
            reg.clone(),
            Arc::new(SecurityManager::new()),
            ApprovalGate::yolo(),
        );
        let mut names: Vec<String> = dispatcher
            .get_definitions()
            .into_iter()
            .map(|d| d.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["read_file", "run_bash_command"]);
        assert!(dispatcher
            .get_definitions()
            .iter()
            .any(|d| d.name == "read_file"));
    }

    #[tokio::test]
    async fn agent_scope_session_grant_stops_at_the_granting_agent() {
        let (reg, dangerous_calls, _) = registry();
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
        let asked = spawn_recording_ui(rx, ApprovalDecision::ApproveForSession);

        let dispatcher = GatedDispatcher::new(reg, Arc::new(SecurityManager::new()), gate);
        for agent in ["lead", "lead", "engineer-1"] {
            dispatcher
                .dispatch(agent, &call("run_bash_command"))
                .await
                .unwrap();
        }

        assert_eq!(dangerous_calls.load(Ordering::SeqCst), 3);
        // The second "lead" call rode the grant; "engineer-1" had to be asked,
        // and was asked under its own identity rather than the lead's.
        assert_eq!(asked_agents(&asked), vec!["lead", "engineer-1"]);
    }

    #[tokio::test]
    async fn tool_scope_session_grant_covers_every_agent() {
        let (reg, dangerous_calls, _) = registry();
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Tool);
        let asked = spawn_recording_ui(rx, ApprovalDecision::ApproveForSession);

        let dispatcher = GatedDispatcher::new(reg, Arc::new(SecurityManager::new()), gate);
        for agent in ["lead", "engineer-1"] {
            dispatcher
                .dispatch(agent, &call("run_bash_command"))
                .await
                .unwrap();
        }

        assert_eq!(dangerous_calls.load(Ordering::SeqCst), 2);
        assert_eq!(asked_agents(&asked), vec!["lead"]);
    }

    #[tokio::test]
    async fn missing_ui_denies_under_every_scope() {
        for scope in [GrantScope::Tool, GrantScope::Agent] {
            let (reg, dangerous_calls, _) = registry();
            let (gate, rx) = ApprovalGate::new(false, scope);
            drop(rx);

            let dispatcher = GatedDispatcher::new(reg, Arc::new(SecurityManager::new()), gate);
            let out = dispatcher
                .dispatch("engineer-1", &call("run_bash_command"))
                .await
                .unwrap();
            assert!(out.starts_with("DENIED by user"), "scope {scope:?}");
            assert_eq!(dangerous_calls.load(Ordering::SeqCst), 0, "scope {scope:?}");
        }
    }

    #[tokio::test]
    async fn denial_grants_nothing_under_agent_scope() {
        let (reg, dangerous_calls, _) = registry();
        let (gate, rx) = ApprovalGate::new(false, GrantScope::Agent);
        let asked = spawn_recording_ui(rx, ApprovalDecision::Deny);

        let dispatcher = GatedDispatcher::new(reg, Arc::new(SecurityManager::new()), gate);
        for agent in ["lead", "lead"] {
            let out = dispatcher
                .dispatch(agent, &call("run_bash_command"))
                .await
                .unwrap();
            assert!(out.starts_with("DENIED by user"));
        }

        assert_eq!(dangerous_calls.load(Ordering::SeqCst), 0);
        assert_eq!(asked_agents(&asked), vec!["lead", "lead"]);
    }
}
