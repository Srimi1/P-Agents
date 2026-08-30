pub mod anthropic;
pub mod mock;
pub mod openai;

use crate::types::{ChatMessage, LlmResponse, ToolCall, ToolDefinition, TokenUsage};
use anyhow::Result;
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::collections::BTreeMap;
use std::pin::Pin;

pub use anthropic::AnthropicProvider;
pub use mock::MockProvider;
pub use openai::GenericOpenAiProvider;

/// Incremental output from a provider. `Done` is always the final item of a
/// well-behaved stream and carries the fully assembled response.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    TextDelta(String),
    ToolCallStarted {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgsDelta {
        index: usize,
        json_fragment: String,
    },
    Usage(TokenUsage),
    Done(Box<LlmResponse>),
}

pub type LlmStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;

    async fn stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        temperature: Option<f32>,
    ) -> Result<LlmStream>;

    /// Drains `stream()` and returns the assembled response. Providers only
    /// override this when a cheaper unary endpoint is worth the extra code.
    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        temperature: Option<f32>,
    ) -> Result<LlmResponse> {
        let mut stream = self.stream(messages, tools, temperature).await?;
        let mut last_done: Option<LlmResponse> = None;
        while let Some(event) = stream.next().await {
            if let StreamEvent::Done(resp) = event? {
                last_done = Some(*resp);
            }
        }
        last_done.ok_or_else(|| anyhow::anyhow!("provider stream ended without a Done event"))
    }
}

/// Assembles `StreamEvent`s into an `LlmResponse`.
///
/// Both wire formats deliver tool calls as a stream of fragments keyed by index:
/// a start event carrying id/name, then any number of partial JSON strings. This
/// collects them in index order and parses once at the end.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    text: String,
    /// index -> (id, name, raw argument JSON accumulated so far)
    tool_calls: BTreeMap<usize, (String, String, String)>,
    usage: Option<TokenUsage>,
    stop_reason: Option<String>,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_text(&mut self, delta: &str) {
        self.text.push_str(delta);
    }

    pub fn start_tool_call(&mut self, index: usize, id: String, name: String) {
        let entry = self
            .tool_calls
            .entry(index)
            .or_insert_with(|| (String::new(), String::new(), String::new()));
        // Some servers repeat the index with an empty id/name on later chunks;
        // never clobber a value we already have with an empty one.
        if !id.is_empty() {
            entry.0 = id;
        }
        if !name.is_empty() {
            entry.1 = name;
        }
    }

    pub fn push_tool_args(&mut self, index: usize, fragment: &str) {
        let entry = self
            .tool_calls
            .entry(index)
            .or_insert_with(|| (String::new(), String::new(), String::new()));
        entry.2.push_str(fragment);
    }

    pub fn set_usage(&mut self, usage: TokenUsage) {
        self.usage = Some(usage);
    }

    pub fn set_stop_reason(&mut self, reason: impl Into<String>) {
        self.stop_reason = Some(reason.into());
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Finishes assembly. Tool calls whose arguments failed to parse are kept
    /// with a `Null` argument value so the agent loop can report the failure
    /// back to the model instead of silently pretending the call was empty.
    pub fn finish(self) -> LlmResponse {
        let tool_calls = self
            .tool_calls
            .into_iter()
            .enumerate()
            .map(|(ordinal, (index, (id, name, raw)))| {
                let arguments = if raw.trim().is_empty() {
                    serde_json::json!({})
                } else {
                    serde_json::from_str(&raw).unwrap_or(serde_json::Value::Null)
                };
                ToolCall {
                    // Servers that stream tool calls without ids (some Ollama
                    // builds) still need a stable id to correlate results.
                    id: if id.is_empty() {
                        format!("call_{}_{}", ordinal, index)
                    } else {
                        id
                    },
                    name,
                    arguments,
                }
            })
            .collect();

        LlmResponse {
            content: if self.text.is_empty() {
                None
            } else {
                Some(self.text)
            },
            tool_calls,
            usage: self.usage,
            stop_reason: self.stop_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_text_and_tool_calls_in_index_order() {
        let mut acc = StreamAccumulator::new();
        acc.push_text("Let me ");
        acc.push_text("check.");
        // Deliberately out of order: index 1 arrives before index 0.
        acc.start_tool_call(1, "b".into(), "second".into());
        acc.push_tool_args(1, r#"{"y":"#);
        acc.start_tool_call(0, "a".into(), "first".into());
        acc.push_tool_args(0, r#"{"x":1}"#);
        acc.push_tool_args(1, r#"2}"#);
        acc.set_usage(TokenUsage::new(3, 4));
        acc.set_stop_reason("tool_use");

        let resp = acc.finish();
        assert_eq!(resp.content.as_deref(), Some("Let me check."));
        assert_eq!(resp.tool_calls.len(), 2);
        assert_eq!(resp.tool_calls[0].name, "first");
        assert_eq!(resp.tool_calls[0].arguments["x"], 1);
        assert_eq!(resp.tool_calls[1].name, "second");
        assert_eq!(resp.tool_calls[1].arguments["y"], 2);
        assert_eq!(resp.usage.unwrap().total_tokens, 7);
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));
    }

    #[test]
    fn empty_arguments_become_empty_object() {
        let mut acc = StreamAccumulator::new();
        acc.start_tool_call(0, "id".into(), "no_args".into());
        let resp = acc.finish();
        assert_eq!(resp.tool_calls[0].arguments, serde_json::json!({}));
    }

    #[test]
    fn malformed_arguments_survive_as_null_for_the_loop_to_report() {
        let mut acc = StreamAccumulator::new();
        acc.start_tool_call(0, "id".into(), "broken".into());
        acc.push_tool_args(0, "{not json");
        let resp = acc.finish();
        assert_eq!(resp.tool_calls[0].arguments, serde_json::Value::Null);
    }

    #[test]
    fn missing_ids_are_synthesized() {
        let mut acc = StreamAccumulator::new();
        acc.start_tool_call(0, String::new(), "anon".into());
        let resp = acc.finish();
        assert!(!resp.tool_calls[0].id.is_empty());
    }

    #[test]
    fn later_empty_id_does_not_clobber_earlier_value() {
        let mut acc = StreamAccumulator::new();
        acc.start_tool_call(0, "real_id".into(), "tool".into());
        acc.start_tool_call(0, String::new(), String::new());
        let resp = acc.finish();
        assert_eq!(resp.tool_calls[0].id, "real_id");
        assert_eq!(resp.tool_calls[0].name, "tool");
    }
}
