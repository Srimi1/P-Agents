use async_trait::async_trait;
use crate::types::{ChatMessage, LlmResponse, ToolDefinition};
use anyhow::Result;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;

    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        temperature: Option<f32>,
    ) -> Result<LlmResponse>;
}

/// Generic OpenAI-compatible Provider (Works with OpenAI, Ollama, DeepSeek, LocalAI, vLLM)
pub struct GenericOpenAiProvider {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub client: reqwest::Client,
}

impl GenericOpenAiProvider {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>, model: impl Into<String>) -> Self {
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

    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        temperature: Option<f32>,
    ) -> Result<LlmResponse> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));

        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": temperature.unwrap_or(0.2),
        });

        if !tools.is_empty() {
            let tools_json: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools_json);
        }

        let res = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !res.status().is_success() {
            let err_text = res.text().await?;
            anyhow::bail!("LLM API returned error: {}", err_text);
        }

        let json_resp: serde_json::Value = res.json().await?;
        let choice = json_resp["choices"]
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("No choices returned from LLM"))?;

        let message = &choice["message"];
        let content = message["content"].as_str().map(|s| s.to_string());

        let mut tool_calls = Vec::new();
        if let Some(tcs) = message["tool_calls"].as_array() {
            for tc in tcs {
                let id = tc["id"].as_str().unwrap_or("").to_string();
                let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let args_raw = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let arguments: serde_json::Value = serde_json::from_str(args_raw).unwrap_or(serde_json::json!({}));
                tool_calls.push(crate::types::ToolCall { id, name, arguments });
            }
        }

        Ok(LlmResponse {
            content,
            tool_calls,
            usage: None,
        })
    }
}
