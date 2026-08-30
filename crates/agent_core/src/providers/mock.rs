use super::{LlmProvider, LlmStream, StreamEvent};
use crate::types::{ChatMessage, LlmResponse, TokenUsage, ToolDefinition};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::Mutex;

/// A provider that replays scripted responses instead of calling a network.
///
/// Every request's message list is recorded, which is what lets the sub-agent
/// tests assert context isolation: a sub-agent must never have seen the
/// parent's history.
pub struct MockProvider {
    model: String,
    script: Mutex<VecDeque<LlmResponse>>,
    recorded: Mutex<Vec<Vec<ChatMessage>>>,
    /// Bytes per simulated token chunk when streaming.
    chunk_size: usize,
    /// Optional per-response delay, used to stagger concurrent sub-agents.
    delay: Option<std::time::Duration>,
}

impl MockProvider {
    pub fn new(responses: Vec<LlmResponse>) -> Self {
        Self {
            model: "mock-model".to_string(),
            script: Mutex::new(responses.into()),
            recorded: Mutex::new(Vec::new()),
            chunk_size: 8,
            delay: None,
        }
    }

    /// Convenience for the common "reply with this text and stop" case.
    pub fn with_text(text: impl Into<String>) -> Self {
        Self::new(vec![LlmResponse {
            content: Some(text.into()),
            tool_calls: Vec::new(),
            usage: Some(TokenUsage::new(10, 5)),
            stop_reason: Some("stop".to_string()),
        }])
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_delay(mut self, delay: std::time::Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }

    /// Message lists this provider was called with, oldest first.
    pub fn recorded_requests(&self) -> Vec<Vec<ChatMessage>> {
        self.recorded.lock().unwrap().clone()
    }

    pub fn call_count(&self) -> usize {
        self.recorded.lock().unwrap().len()
    }

    /// Every user/system/tool text this provider was ever shown. Handy for
    /// asserting that a string did *not* leak into another agent's context.
    pub fn all_seen_text(&self) -> String {
        self.recorded
            .lock()
            .unwrap()
            .iter()
            .flatten()
            .filter_map(|m| m.content.clone())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn next_response(&self, messages: &[ChatMessage]) -> Result<LlmResponse> {
        self.recorded.lock().unwrap().push(messages.to_vec());
        self.script
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("MockProvider script exhausted after {} call(s)", self.call_count()))
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    fn provider_name(&self) -> &str {
        "mock"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    async fn stream(
        &self,
        messages: &[ChatMessage],
        _tools: &[ToolDefinition],
        _temperature: Option<f32>,
    ) -> Result<LlmStream> {
        let response = self.next_response(messages)?;
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }

        // Chop the scripted text into chunks so consumers exercise the real
        // incremental path. Chunking is by char to stay UTF-8 safe.
        let mut events: Vec<Result<StreamEvent>> = Vec::new();
        if let Some(text) = &response.content {
            let chars: Vec<char> = text.chars().collect();
            for chunk in chars.chunks(self.chunk_size) {
                events.push(Ok(StreamEvent::TextDelta(chunk.iter().collect())));
            }
        }
        for (index, call) in response.tool_calls.iter().enumerate() {
            events.push(Ok(StreamEvent::ToolCallStarted {
                index,
                id: call.id.clone(),
                name: call.name.clone(),
            }));
            events.push(Ok(StreamEvent::ToolCallArgsDelta {
                index,
                json_fragment: call.arguments.to_string(),
            }));
        }
        if let Some(usage) = response.usage {
            events.push(Ok(StreamEvent::Usage(usage)));
        }
        events.push(Ok(StreamEvent::Done(Box::new(response))));

        Ok(Box::pin(futures::stream::iter(events)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn streams_text_in_chunks_then_done() {
        let provider = MockProvider::with_text("hello world").with_chunk_size(4);
        let mut stream = provider.stream(&[ChatMessage::user("hi")], &[], None).await.unwrap();

        let mut deltas = Vec::new();
        let mut done = None;
        while let Some(ev) = stream.next().await {
            match ev.unwrap() {
                StreamEvent::TextDelta(d) => deltas.push(d),
                StreamEvent::Done(r) => done = Some(*r),
                _ => {}
            }
        }
        assert!(deltas.len() > 1, "text should arrive in multiple chunks");
        assert_eq!(deltas.concat(), "hello world");
        assert_eq!(done.unwrap().content.as_deref(), Some("hello world"));
    }

    #[tokio::test]
    async fn complete_drains_the_stream() {
        let provider = MockProvider::with_text("done");
        let resp = provider.complete(&[ChatMessage::user("hi")], &[], None).await.unwrap();
        assert_eq!(resp.content.as_deref(), Some("done"));
    }

    #[tokio::test]
    async fn records_requests_for_isolation_assertions() {
        let provider = MockProvider::new(vec![
            LlmResponse { content: Some("a".into()), ..Default::default() },
            LlmResponse { content: Some("b".into()), ..Default::default() },
        ]);
        provider.complete(&[ChatMessage::user("first")], &[], None).await.unwrap();
        provider.complete(&[ChatMessage::user("second")], &[], None).await.unwrap();

        assert_eq!(provider.call_count(), 2);
        assert!(provider.all_seen_text().contains("first"));
        assert!(provider.all_seen_text().contains("second"));
    }

    #[tokio::test]
    async fn exhausted_script_is_an_error_not_a_panic() {
        let provider = MockProvider::new(vec![]);
        let err = provider.complete(&[ChatMessage::user("hi")], &[], None).await.unwrap_err();
        assert!(err.to_string().contains("script exhausted"));
    }
}
