use super::{LlmProvider, LlmStream};
use crate::types::{ChatMessage, ToolDefinition};
use anyhow::Result;
use async_trait::async_trait;

pub const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
pub const ANTHROPIC_VERSION: &str = "2023-06-01";
pub const DEFAULT_MAX_TOKENS: usize = 8192;

/// Native Anthropic Messages API provider.
pub struct AnthropicProvider {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: usize,
    pub client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            model: model.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn provider_name(&self) -> &str {
        "anthropic"
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
        anyhow::bail!("anthropic provider streaming not implemented yet")
    }
}
