use super::{LlmProvider, LlmStream};
use crate::types::{ChatMessage, ToolDefinition};
use anyhow::Result;
use async_trait::async_trait;

/// Generic OpenAI-compatible provider (OpenAI, Ollama, DeepSeek, LocalAI, vLLM,
/// Gemini's OpenAI-compatible endpoint).
pub struct GenericOpenAiProvider {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub client: reqwest::Client,
}

impl GenericOpenAiProvider {
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl LlmProvider for GenericOpenAiProvider {
    fn provider_name(&self) -> &str {
        "openai_compatible"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    async fn stream(
        &self,
        _messages: &[ChatMessage],
        _tools: &[ToolDefinition],
        _temperature: Option<f32>,
    ) -> Result<LlmStream> {
        anyhow::bail!("openai provider streaming not implemented yet")
    }
}
