use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            api_key: env::var("OPENAI_API_KEY")
                .or_else(|_| env::var("ANTHROPIC_API_KEY"))
                .unwrap_or_else(|_| "dummy_key_or_set_env".to_string()),
            base_url: env::var("LLM_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            model: env::var("LLM_MODEL")
                .unwrap_or_else(|_| "gpt-4o".to_string()),
        }
    }
}

impl HarnessConfig {
    pub fn load() -> Result<Self> {
        Ok(Self::default())
    }
}
