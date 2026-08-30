//! End-to-end coverage of `Agent::run` against scripted providers: tool
//! round-trips, self-healing on tool failure, the iteration cap, the event
//! stream, and the compaction hook.

use agent_core::{
    Agent, AgentEvent, AgentState, ChatMessage, HistoryCompactor, LlmResponse, MockProvider, Role,
    TokenUsage, ToolCall, ToolDefinition, ToolDispatcher,
};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{self, UnboundedReceiver};

const AGENT_ID: &str = "agent-under-test";

enum ToolOutcome {
    Reply(String),
    Fail(String),
}

struct ScriptedDispatcher {
    tool_name: String,
    outcome: ToolOutcome,
    calls: Mutex<Vec<(String, ToolCall)>>,
}

impl ScriptedDispatcher {
    fn replying(tool_name: &str, reply: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            outcome: ToolOutcome::Reply(reply.to_string()),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn failing(tool_name: &str, message: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            outcome: ToolOutcome::Fail(message.to_string()),
            calls: Mutex::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<(String, ToolCall)> {
        self.calls.lock().unwrap().clone()
    }

    fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl ToolDispatcher for ScriptedDispatcher {
    fn get_definitions(&self) -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: self.tool_name.clone(),
            description: "scripted test tool".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }]
    }

    async fn dispatch(&self, agent_id: &str, tool_call: &ToolCall) -> Result<String> {
        self.calls
            .lock()
            .unwrap()
            .push((agent_id.to_string(), tool_call.clone()));
        match &self.outcome {
            ToolOutcome::Reply(reply) => Ok(reply.clone()),
            ToolOutcome::Fail(message) => Err(anyhow::anyhow!("{}", message)),
        }
    }
}

/// A dispatcher that advertises no tools; used for the text-only paths.
struct NoTools;

#[async_trait]
impl ToolDispatcher for NoTools {
    fn get_definitions(&self) -> Vec<ToolDefinition> {
        Vec::new()
    }

    async fn dispatch(&self, _agent_id: &str, tool_call: &ToolCall) -> Result<String> {
        anyhow::bail!("unexpected dispatch of '{}'", tool_call.name)
    }
}

struct RecordingCompactor {
    /// 1-based index of the `should_compact` call that answers true.
    fires_on_call: usize,
    /// (history length, last_prompt_tokens) seen by each `should_compact` call.
    observed: Mutex<Vec<(usize, Option<usize>)>>,
    compactions: AtomicUsize,
}

impl RecordingCompactor {
    fn new(fires_on_call: usize) -> Self {
        Self {
            fires_on_call,
            observed: Mutex::new(Vec::new()),
            compactions: AtomicUsize::new(0),
        }
    }

    fn observed(&self) -> Vec<(usize, Option<usize>)> {
        self.observed.lock().unwrap().clone()
    }

    fn compactions(&self) -> usize {
        self.compactions.load(Ordering::SeqCst)
    }
}

impl HistoryCompactor for RecordingCompactor {
    fn should_compact(&self, history: &[ChatMessage], last_prompt_tokens: Option<usize>) -> bool {
        let mut observed = self.observed.lock().unwrap();
        observed.push((history.len(), last_prompt_tokens));
        observed.len() == self.fires_on_call
    }

    fn compact(&self, history: &mut Vec<ChatMessage>) {
        self.compactions.fetch_add(1, Ordering::SeqCst);
        let system = history.first().cloned();
        history.clear();
        if let Some(system) = system {
            history.push(system);
        }
        history.push(ChatMessage::user("<compacted transcript>"));
    }
}

fn tool_call_turn(id: &str, name: &str, arguments: Value, usage: TokenUsage) -> LlmResponse {
    LlmResponse {
        content: None,
        tool_calls: vec![ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments,
        }],
        usage: Some(usage),
        stop_reason: Some("tool_use".to_string()),
    }
}

fn text_turn(text: &str, usage: TokenUsage) -> LlmResponse {
    LlmResponse {
        content: Some(text.to_string()),
        tool_calls: Vec::new(),
        usage: Some(usage),
        stop_reason: Some("stop".to_string()),
    }
}

fn drain(rx: &mut UnboundedReceiver<AgentEvent>) -> Vec<AgentEvent> {
    let mut events = Vec::new();
    while let Ok(event) = rx.try_recv() {
        events.push(event);
    }
    events
}

/// Event kinds in order, with runs of `TextDelta` collapsed to a single entry
/// so assertions do not depend on the mock's chunk size.
fn shape(events: &[AgentEvent]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for event in events {
        let label = match event {
            AgentEvent::StateChanged { state, .. } => format!("StateChanged:{:?}", state),
            AgentEvent::TextDelta { .. } => "TextDelta".to_string(),
            AgentEvent::MessageAppended { .. } => "MessageAppended".to_string(),
            AgentEvent::ToolStarted { .. } => "ToolStarted".to_string(),
            AgentEvent::ToolFinished { .. } => "ToolFinished".to_string(),
            AgentEvent::UsageReport { .. } => "UsageReport".to_string(),
            AgentEvent::Compacted { .. } => "Compacted".to_string(),
            AgentEvent::SubAgentSpawned { .. } => "SubAgentSpawned".to_string(),
            AgentEvent::SubAgentFinished { .. } => "SubAgentFinished".to_string(),
        };
        if label == "TextDelta" && out.last().map(String::as_str) == Some("TextDelta") {
            continue;
        }
        out.push(label);
    }
    out
}

#[tokio::test]
async fn text_only_run_completes_and_records_the_answer() {
    let provider = Arc::new(MockProvider::with_text("The answer is 42."));
    let mut agent = Agent::new(
        AGENT_ID,
        "tester",
        "You are a test agent.",
        provider.clone(),
        Arc::new(NoTools),
    );

    let answer = agent.run("What is the answer?").await.unwrap();

    assert_eq!(answer, "The answer is 42.");
    assert_eq!(agent.state, AgentState::Completed);
    assert_eq!(provider.call_count(), 1);

    assert_eq!(agent.history.len(), 3);
    assert_eq!(agent.history[0].role, Role::System);
    assert_eq!(
        agent.history[0].content.as_deref(),
        Some("You are a test agent.")
    );
    assert_eq!(agent.history[1].role, Role::User);
    assert_eq!(
        agent.history[1].content.as_deref(),
        Some("What is the answer?")
    );
    assert_eq!(agent.history[2].role, Role::Assistant);
    assert_eq!(
        agent.history[2].content.as_deref(),
        Some("The answer is 42.")
    );
    assert!(agent.history[2].tool_calls.is_none());
    assert_eq!(agent.cumulative_usage, TokenUsage::new(10, 5));
}

#[tokio::test]
async fn tool_round_trip_feeds_the_observation_back_to_the_model() {
    let provider = Arc::new(MockProvider::new(vec![
        tool_call_turn(
            "call_weather_1",
            "get_weather",
            json!({"city": "Kyoto", "units": "c"}),
            TokenUsage::new(12, 4),
        ),
        text_turn("It is 22C in Kyoto.", TokenUsage::new(30, 6)),
    ]));
    let dispatcher = Arc::new(ScriptedDispatcher::replying("get_weather", "22C, clear"));

    let mut agent = Agent::new(
        AGENT_ID,
        "tester",
        "sys",
        provider.clone(),
        dispatcher.clone(),
    );
    let answer = agent.run("weather in Kyoto?").await.unwrap();

    assert_eq!(answer, "It is 22C in Kyoto.");
    assert_eq!(agent.state, AgentState::Completed);

    let calls = dispatcher.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, AGENT_ID);
    assert_eq!(calls[0].1.name, "get_weather");
    assert_eq!(calls[0].1.id, "call_weather_1");
    assert_eq!(calls[0].1.arguments, json!({"city": "Kyoto", "units": "c"}));

    // system, user, assistant(tool_calls), tool, assistant(final)
    assert_eq!(agent.history.len(), 5);
    let assistant_call = &agent.history[2];
    assert_eq!(assistant_call.role, Role::Assistant);
    let emitted = assistant_call.tool_calls.as_ref().expect("tool calls kept");
    assert_eq!(emitted.len(), 1);
    assert_eq!(emitted[0].id, "call_weather_1");

    let observation = &agent.history[3];
    assert_eq!(observation.role, Role::Tool);
    assert_eq!(observation.tool_call_id.as_deref(), Some("call_weather_1"));
    assert_eq!(observation.name.as_deref(), Some("get_weather"));
    assert_eq!(observation.content.as_deref(), Some("22C, clear"));

    assert_eq!(
        agent.history[4].content.as_deref(),
        Some("It is 22C in Kyoto.")
    );

    // The second turn must have shown the model the observation.
    let requests = provider.recorded_requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].len(), 4);
    assert_eq!(requests[1][3].role, Role::Tool);
}

#[tokio::test]
async fn tool_failure_is_reported_to_the_model_instead_of_failing_the_run() {
    let provider = Arc::new(MockProvider::new(vec![
        tool_call_turn(
            "call_read_1",
            "read_file",
            json!({"path": "/nope"}),
            TokenUsage::new(12, 4),
        ),
        text_turn("That file does not exist.", TokenUsage::new(20, 5)),
    ]));
    let dispatcher = Arc::new(ScriptedDispatcher::failing(
        "read_file",
        "no such file or directory",
    ));

    let mut agent = Agent::new(AGENT_ID, "tester", "sys", provider, dispatcher.clone());
    let answer = agent.run("read /nope").await.unwrap();

    assert_eq!(answer, "That file does not exist.");
    assert_eq!(agent.state, AgentState::Completed);
    assert_eq!(dispatcher.call_count(), 1);

    let observation = &agent.history[3];
    assert_eq!(observation.role, Role::Tool);
    assert_eq!(observation.tool_call_id.as_deref(), Some("call_read_1"));
    let text = observation.content.as_deref().unwrap_or_default();
    assert!(
        text.contains("Error executing tool 'read_file'") && text.contains("no such file"),
        "unexpected observation: {text}"
    );
}

#[tokio::test]
async fn unparseable_tool_arguments_are_never_dispatched() {
    let provider = Arc::new(MockProvider::new(vec![
        tool_call_turn(
            "call_bad_1",
            "run_query",
            Value::Null,
            TokenUsage::new(9, 3),
        ),
        text_turn("Recovered.", TokenUsage::new(15, 2)),
    ]));
    let dispatcher = Arc::new(ScriptedDispatcher::replying("run_query", "unreachable"));
    let (tx, mut rx) = mpsc::unbounded_channel();

    let mut agent =
        Agent::new(AGENT_ID, "tester", "sys", provider, dispatcher.clone()).with_events(tx);
    let answer = agent.run("query it").await.unwrap();

    assert_eq!(answer, "Recovered.");
    assert_eq!(
        dispatcher.call_count(),
        0,
        "tool must not run on null arguments"
    );

    let observation = &agent.history[3];
    assert_eq!(observation.role, Role::Tool);
    assert_eq!(observation.tool_call_id.as_deref(), Some("call_bad_1"));
    let text = observation.content.as_deref().unwrap_or_default();
    assert!(
        text.contains("not valid JSON") && text.contains("run_query"),
        "unexpected observation: {text}"
    );

    let events = drain(&mut rx);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolStarted { .. })),
        "a rejected call must not report as started"
    );
    let finished: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolFinished { tool, is_error, .. } => Some((tool.clone(), *is_error)),
            _ => None,
        })
        .collect();
    assert_eq!(finished, vec![("run_query".to_string(), true)]);
}

#[tokio::test]
async fn iteration_cap_terminates_the_loop_with_an_error() {
    let script: Vec<LlmResponse> = (0..3)
        .map(|i| {
            tool_call_turn(
                &format!("call_{i}"),
                "spin",
                json!({"n": i}),
                TokenUsage::new(5, 1),
            )
        })
        .collect();
    let provider = Arc::new(MockProvider::new(script));
    let dispatcher = Arc::new(ScriptedDispatcher::replying("spin", "still spinning"));

    let mut agent = Agent::new(
        AGENT_ID,
        "tester",
        "sys",
        provider.clone(),
        dispatcher.clone(),
    )
    .with_max_iterations(3);

    let err = agent.run("spin forever").await.unwrap_err();

    let message = err.to_string();
    assert!(
        message.contains("maximum iteration limit") && message.contains("(3)"),
        "unexpected error: {message}"
    );
    assert_eq!(agent.state, AgentState::Error);
    assert_eq!(dispatcher.call_count(), 3);
    assert_eq!(provider.call_count(), 3);
}

#[tokio::test]
async fn events_report_the_full_turn_sequence_and_cumulative_usage() {
    let mut first = tool_call_turn(
        "call_search_1",
        "search",
        json!({"q": "rust"}),
        TokenUsage::new(10, 5),
    );
    first.content = Some("Looking that up".to_string());

    let provider = Arc::new(
        MockProvider::new(vec![
            first,
            text_turn("Final answer", TokenUsage::new(20, 7)),
        ])
        .with_chunk_size(3),
    );
    let dispatcher = Arc::new(ScriptedDispatcher::replying("search", "one result"));
    let (tx, mut rx) = mpsc::unbounded_channel();

    let mut agent = Agent::new(AGENT_ID, "tester", "sys", provider, dispatcher).with_events(tx);
    let answer = agent.run("look it up").await.unwrap();
    assert_eq!(answer, "Final answer");

    let events = drain(&mut rx);
    assert!(events.iter().all(|e| e.agent_id() == AGENT_ID));

    assert_eq!(
        shape(&events),
        vec![
            "MessageAppended",
            "StateChanged:Planning",
            "StateChanged:StreamingResponse",
            "TextDelta",
            "UsageReport",
            "MessageAppended",
            "StateChanged:ExecutingTool",
            "ToolStarted",
            "ToolFinished",
            "MessageAppended",
            "StateChanged:Planning",
            "StateChanged:StreamingResponse",
            "TextDelta",
            "UsageReport",
            "MessageAppended",
            "StateChanged:Completed",
        ]
    );

    // Deltas arrive chunked, and each turn's chunks reassemble to its text.
    let mut runs: Vec<Vec<String>> = Vec::new();
    let mut in_run = false;
    for event in &events {
        match event {
            AgentEvent::TextDelta { delta, .. } => {
                if !in_run {
                    runs.push(Vec::new());
                    in_run = true;
                }
                runs.last_mut().expect("run started").push(delta.clone());
            }
            _ => in_run = false,
        }
    }
    assert_eq!(runs.len(), 2);
    assert!(runs[0].len() > 1, "text should stream in multiple chunks");
    assert_eq!(runs[0].concat(), "Looking that up");
    assert_eq!(runs[1].concat(), "Final answer");

    let started: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::ToolStarted {
                tool, arguments, ..
            } => Some((tool.clone(), arguments.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(started, vec![("search".to_string(), json!({"q": "rust"}))]);

    let usage: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            AgentEvent::UsageReport {
                turn, cumulative, ..
            } => Some((*turn, *cumulative)),
            _ => None,
        })
        .collect();
    assert_eq!(
        usage,
        vec![
            (TokenUsage::new(10, 5), TokenUsage::new(10, 5)),
            (TokenUsage::new(20, 7), TokenUsage::new(30, 12)),
        ]
    );
    assert_eq!(agent.cumulative_usage.total_tokens, 42);
}

#[tokio::test]
async fn compactor_is_consulted_each_iteration_and_reshapes_history() {
    let provider = Arc::new(MockProvider::new(vec![
        tool_call_turn(
            "call_note_1",
            "note",
            json!({"text": "remember"}),
            TokenUsage::new(13, 2),
        ),
        text_turn("Noted.", TokenUsage::new(4, 1)),
    ]));
    let dispatcher = Arc::new(ScriptedDispatcher::replying("note", "saved"));
    let compactor = Arc::new(RecordingCompactor::new(2));
    let (tx, mut rx) = mpsc::unbounded_channel();

    let mut agent = Agent::new(AGENT_ID, "tester", "sys", provider.clone(), dispatcher)
        .with_compactor(compactor.clone())
        .with_events(tx);

    let answer = agent.run("take a note").await.unwrap();
    assert_eq!(answer, "Noted.");

    // Consulted once per iteration, with the prompt size from the prior turn.
    assert_eq!(compactor.observed(), vec![(2, None), (4, Some(13))]);
    assert_eq!(compactor.compactions(), 1);

    let compacted: Vec<_> = drain(&mut rx)
        .into_iter()
        .filter_map(|e| match e {
            AgentEvent::Compacted {
                messages_before,
                messages_after,
                ..
            } => Some((messages_before, messages_after)),
            _ => None,
        })
        .collect();
    assert_eq!(compacted, vec![(4, 2)]);

    // The second turn saw the compacted transcript, not the original one.
    let requests = provider.recorded_requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].len(), 2);
    assert_eq!(
        requests[1][1].content.as_deref(),
        Some("<compacted transcript>")
    );
}

#[tokio::test]
async fn non_streaming_mode_uses_the_unary_path() {
    let provider = Arc::new(MockProvider::with_text("Same answer."));
    let (tx, mut rx) = mpsc::unbounded_channel();

    let mut agent = Agent::new(AGENT_ID, "tester", "sys", provider, Arc::new(NoTools))
        .with_streaming(false)
        .with_events(tx);

    let answer = agent.run("hello").await.unwrap();
    assert_eq!(answer, "Same answer.");
    assert_eq!(agent.state, AgentState::Completed);
    assert_eq!(agent.history[2].content.as_deref(), Some("Same answer."));

    assert_eq!(
        shape(&drain(&mut rx)),
        vec![
            "MessageAppended",
            "StateChanged:Planning",
            "UsageReport",
            "MessageAppended",
            "StateChanged:Completed",
        ],
        "the unary path emits neither TextDelta nor StreamingResponse"
    );
}

#[tokio::test]
async fn restore_history_reseeds_a_missing_system_prompt() {
    let provider = Arc::new(MockProvider::with_text("ok"));
    let mut agent = Agent::new(
        AGENT_ID,
        "tester",
        "guiding prompt",
        provider,
        Arc::new(NoTools),
    );

    agent.restore_history(vec![
        ChatMessage::user("earlier question"),
        ChatMessage::assistant("earlier answer"),
    ]);

    assert_eq!(agent.history.len(), 3);
    assert_eq!(agent.history[0].role, Role::System);
    assert_eq!(agent.history[0].content.as_deref(), Some("guiding prompt"));
    assert_eq!(
        agent.history[1].content.as_deref(),
        Some("earlier question")
    );

    // An empty transcript still gets the prompt back.
    agent.restore_history(Vec::new());
    assert_eq!(agent.history.len(), 1);
    assert_eq!(agent.history[0].role, Role::System);
}

#[tokio::test]
async fn restore_history_keeps_an_existing_system_prompt() {
    let provider = Arc::new(MockProvider::with_text("ok"));
    let mut agent = Agent::new(
        AGENT_ID,
        "tester",
        "guiding prompt",
        provider,
        Arc::new(NoTools),
    );

    agent.restore_history(vec![
        ChatMessage::system("persisted prompt"),
        ChatMessage::user("earlier question"),
    ]);

    assert_eq!(agent.history.len(), 2);
    assert_eq!(
        agent.history[0].content.as_deref(),
        Some("persisted prompt")
    );
}
