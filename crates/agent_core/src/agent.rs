use crate::provider::LlmProvider;
use crate::types::{AgentState, ChatMessage, ToolCall, ToolDefinition};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tracing::{info, warn};

#[async_trait]
pub trait ToolDispatcher: Send + Sync {
    fn get_definitions(&self) -> Vec<ToolDefinition>;
    async fn dispatch(&self, tool_call: &ToolCall) -> Result<String>;
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
        }
    }

    pub async fn run(&mut self, user_input: &str) -> Result<String> {
        self.history.push(ChatMessage::user(user_input));
        self.state = AgentState::Planning;

        let tool_definitions = self.dispatcher.get_definitions();

        for iteration in 0..self.max_iterations {
            info!(
                agent = %self.name,
                iteration = iteration + 1,
                "Querying LLM provider"
            );

            let response = self
                .provider
                .complete(&self.history, &tool_definitions, None)
                .await?;

            // Case 1: Model produced final text without tool calls
            if response.tool_calls.is_empty() {
                let final_text = response.content.unwrap_or_default();
                self.history.push(ChatMessage::assistant(final_text.clone()));
                self.state = AgentState::Completed;
                return Ok(final_text);
            }

            // Case 2: Model triggered one or more tool calls
            self.history.push(ChatMessage::assistant_tool_calls(response.tool_calls.clone()));

            for tool_call in &response.tool_calls {
                self.state = AgentState::ExecutingTool;
                info!(
                    agent = %self.name,
                    tool = %tool_call.name,
                    args = %tool_call.arguments,
                    "Executing tool call"
                );

                let observation = match self.dispatcher.dispatch(tool_call).await {
                    Ok(res) => res,
                    Err(err) => {
                        warn!(tool = %tool_call.name, error = %err, "Tool execution returned error");
                        format!("Error executing tool '{}': {}", tool_call.name, err)
                    }
                };

                self.history.push(ChatMessage::tool_response(
                    &tool_call.id,
                    &tool_call.name,
                    observation,
                ));
            }

            self.state = AgentState::Planning;
        }

        self.state = AgentState::Error;
        anyhow::bail!("Agent reached maximum iteration limit ({}) without finishing.", self.max_iterations);
    }
}
