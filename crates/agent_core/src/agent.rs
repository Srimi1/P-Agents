use crate::compaction::HistoryCompactor;
use crate::events::{emit, AgentEvent, EventSink};
use crate::providers::{LlmProvider, StreamEvent};
use crate::types::{
    truncate_at_boundary, AgentState, ChatMessage, LlmResponse, TokenUsage, ToolCall,
    ToolDefinition,
};
use anyhow::Result;
use async_trait::async_trait;
use futures::StreamExt;
use std::sync::Arc;
use tracing::{info, warn};

/// How much of a tool result is echoed into the `ToolFinished` event. The full
/// output still goes into the agent's history; this only bounds the UI payload.
const TOOL_PREVIEW_BYTES: usize = 240;

/// Hard cap on a single tool observation stored in history. Compaction
/// deliberately leaves the most recent exchanges intact, so without a cap here
/// one multi-megabyte tool result in that protected tail can blow the context
/// window with nothing able to shrink it.
const MAX_OBSERVATION_BYTES: usize = 100_000;

#[async_trait]
pub trait ToolDispatcher: Send + Sync {
    fn get_definitions(&self) -> Vec<ToolDefinition>;
    /// `agent_id` identifies the caller so approval prompts and events can name
    /// which agent (lead or sub-agent) is asking.
    async fn dispatch(&self, agent_id: &str, tool_call: &ToolCall) -> Result<String>;
}

pub struct Agent {
    pub id: String,
    pub name: String,
    pub system_prompt: String,
    pub provider: Arc<dyn LlmProvider>,
    pub dispatcher: Arc<dyn ToolDispatcher>,
    pub history: Vec<ChatMessage>,
    pub state: AgentState,
    pub max_iterations: usize,
    pub temperature: Option<f32>,
    /// When false the agent uses the provider's unary path. Sub-agents run
    /// non-streaming by default so concurrent token streams don't interleave.
    pub streaming: bool,
    pub events: Option<EventSink>,
    pub compactor: Option<Arc<dyn HistoryCompactor>>,
    pub cumulative_usage: TokenUsage,
    /// Prompt size reported by the most recent turn, when the provider said.
    last_prompt_tokens: Option<usize>,
}

impl Agent {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        system_prompt: impl Into<String>,
        provider: Arc<dyn LlmProvider>,
        dispatcher: Arc<dyn ToolDispatcher>,
    ) -> Self {
        let system_str = system_prompt.into();
        Self {
            id: id.into(),
            name: name.into(),
            system_prompt: system_str.clone(),
            provider,
            dispatcher,
            history: vec![ChatMessage::system(system_str)],
            state: AgentState::Idle,
            max_iterations: 20,
            temperature: None,
            streaming: true,
            events: None,
            compactor: None,
            cumulative_usage: TokenUsage::default(),
            last_prompt_tokens: None,
        }
    }

    pub fn with_events(mut self, sink: EventSink) -> Self {
        self.events = Some(sink);
        self
    }

    pub fn with_compactor(mut self, compactor: Arc<dyn HistoryCompactor>) -> Self {
        self.compactor = Some(compactor);
        self
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn with_streaming(mut self, streaming: bool) -> Self {
        self.streaming = streaming;
        self
    }

    pub fn with_temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Swaps the model mid-session (`/model` in the REPL). History is preserved;
    /// the new provider sees it on the next turn.
    pub fn set_provider(&mut self, provider: Arc<dyn LlmProvider>) {
        self.provider = provider;
    }

    /// Replaces history wholesale, e.g. when resuming a persisted session. The
    /// system prompt is re-seeded if the restored transcript lacks one.
    pub fn restore_history(&mut self, mut history: Vec<ChatMessage>) {
        if history
            .first()
            .map(|m| m.role != crate::types::Role::System)
            .unwrap_or(true)
        {
            history.insert(0, ChatMessage::system(self.system_prompt.clone()));
        }
        self.history = history;
    }

    fn set_state(&mut self, state: AgentState) {
        self.state = state;
        emit(
            &self.events,
            AgentEvent::StateChanged {
                agent_id: self.id.clone(),
                state,
            },
        );
    }

    fn push_history(&mut self, message: ChatMessage) {
        emit(
            &self.events,
            AgentEvent::MessageAppended {
                agent_id: self.id.clone(),
                message: message.clone(),
            },
        );
        self.history.push(message);
    }

    fn record_usage(&mut self, usage: Option<TokenUsage>) {
        let Some(turn) = usage else { return };
        self.last_prompt_tokens = Some(turn.prompt_tokens);
        self.cumulative_usage.accumulate(&turn);
        emit(
            &self.events,
            AgentEvent::UsageReport {
                agent_id: self.id.clone(),
                turn,
                cumulative: self.cumulative_usage,
            },
        );
    }

    fn maybe_compact(&mut self) {
        let Some(compactor) = self.compactor.clone() else {
            return;
        };
        if !compactor.should_compact(&self.history, self.last_prompt_tokens) {
            return;
        }
        let before = self.history.len();
        compactor.compact(&mut self.history);
        emit(
            &self.events,
            AgentEvent::Compacted {
                agent_id: self.id.clone(),
                messages_before: before,
                messages_after: self.history.len(),
            },
        );
    }

    /// One model turn. Streams when enabled, emitting `TextDelta` as tokens
    /// arrive, and returns the assembled response either way.
    async fn model_turn(&mut self, tools: &[ToolDefinition]) -> Result<LlmResponse> {
        if !self.streaming {
            return self
                .provider
                .complete(&self.history, tools, self.temperature)
                .await;
        }

        self.set_state(AgentState::StreamingResponse);
        let mut stream = self
            .provider
            .stream(&self.history, tools, self.temperature)
            .await?;

        let mut final_response: Option<LlmResponse> = None;
        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::TextDelta(delta) => {
                    emit(
                        &self.events,
                        AgentEvent::TextDelta {
                            agent_id: self.id.clone(),
                            delta,
                        },
                    );
                }
                StreamEvent::Done(resp) => final_response = Some(*resp),
                // Tool-call fragments and interim usage are assembled by the
                // provider and arrive complete on `Done`.
                StreamEvent::ToolCallStarted { .. }
                | StreamEvent::ToolCallArgsDelta { .. }
                | StreamEvent::Usage(_) => {}
            }
        }

        final_response.ok_or_else(|| anyhow::anyhow!("provider stream ended without a Done event"))
    }

    pub async fn run(&mut self, user_input: &str) -> Result<String> {
        // Providers disagree about blank turns: Anthropic drops them and then
        // rejects the request for having no messages, while OpenAI accepts one.
        // Refusing here keeps the behaviour the same everywhere.
        if user_input.trim().is_empty() {
            anyhow::bail!("Agent received an empty prompt");
        }

        self.push_history(ChatMessage::user(user_input));
        self.set_state(AgentState::Planning);

        let tool_definitions = self.dispatcher.get_definitions();

        for iteration in 0..self.max_iterations {
            self.maybe_compact();

            info!(
                agent = %self.name,
                iteration = iteration + 1,
                "Querying LLM provider"
            );

            let response = match self.model_turn(&tool_definitions).await {
                Ok(response) => response,
                Err(err) => {
                    self.set_state(AgentState::Error);
                    return Err(err);
                }
            };
            self.record_usage(response.usage);

            // Case 1: final text, no tool calls.
            if response.tool_calls.is_empty() {
                let final_text = response.content.clone().unwrap_or_default();
                self.push_history(ChatMessage::assistant(final_text.clone()));
                self.set_state(AgentState::Completed);
                return Ok(final_text);
            }

            // Case 2: one or more tool calls. Any prose the model emitted
            // alongside them is preserved on the same assistant turn.
            self.push_history(ChatMessage::assistant_with_tool_calls(
                response.content.clone(),
                response.tool_calls.clone(),
            ));

            for tool_call in &response.tool_calls {
                self.set_state(AgentState::ExecutingTool);

                // Arguments that failed to parse upstream arrive as Null. Report
                // that to the model rather than calling the tool with garbage.
                if tool_call.arguments.is_null() {
                    warn!(tool = %tool_call.name, "Model produced unparseable tool arguments");
                    let observation = format!(
                        "Error: arguments for tool '{}' were not valid JSON. Re-issue the call with well-formed JSON arguments.",
                        tool_call.name
                    );
                    emit(
                        &self.events,
                        AgentEvent::ToolFinished {
                            agent_id: self.id.clone(),
                            tool: tool_call.name.clone(),
                            preview: observation.clone(),
                            is_error: true,
                        },
                    );
                    self.push_history(ChatMessage::tool_response(
                        &tool_call.id,
                        &tool_call.name,
                        observation,
                    ));
                    continue;
                }

                info!(
                    agent = %self.name,
                    tool = %tool_call.name,
                    args = %tool_call.arguments,
                    "Executing tool call"
                );
                emit(
                    &self.events,
                    AgentEvent::ToolStarted {
                        agent_id: self.id.clone(),
                        tool: tool_call.name.clone(),
                        arguments: tool_call.arguments.clone(),
                    },
                );

                let (observation, is_error) = match self
                    .dispatcher
                    .dispatch(&self.id, tool_call)
                    .await
                {
                    Ok(res) => (res, false),
                    Err(err) => {
                        warn!(tool = %tool_call.name, error = %err, "Tool execution returned error");
                        (
                            format!("Error executing tool '{}': {}", tool_call.name, err),
                            true,
                        )
                    }
                };

                emit(
                    &self.events,
                    AgentEvent::ToolFinished {
                        agent_id: self.id.clone(),
                        tool: tool_call.name.clone(),
                        preview: truncate_at_boundary(&observation, TOOL_PREVIEW_BYTES).to_string(),
                        is_error,
                    },
                );

                self.push_history(ChatMessage::tool_response(
                    &tool_call.id,
                    &tool_call.name,
                    cap_observation(observation),
                ));
            }

            self.set_state(AgentState::Planning);
        }

        self.set_state(AgentState::Error);
        anyhow::bail!(
            "Agent reached maximum iteration limit ({}) without finishing.",
            self.max_iterations
        );
    }
}

/// Bounds a single tool result before it enters history, leaving a marker so the
/// model knows output was dropped rather than the tool having produced nothing.
fn cap_observation(observation: String) -> String {
    if observation.len() <= MAX_OBSERVATION_BYTES {
        return observation;
    }
    let kept = truncate_at_boundary(&observation, MAX_OBSERVATION_BYTES);
    format!(
        "{kept}\n... [TRUNCATED: {} bytes omitted; the tool produced more output than one \
         observation may carry. Narrow the request to see the rest.]",
        observation.len() - kept.len()
    )
}
