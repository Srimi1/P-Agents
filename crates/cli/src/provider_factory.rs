use crate::config::HarnessConfig;
use agent_core::providers::{AnthropicProvider, GenericOpenAiProvider, MockProvider};
use agent_core::{LlmProvider, LlmResponse, TokenUsage};
use anyhow::Result;
use std::sync::Arc;

/// Builds the provider named by `override_provider`, falling back to the
/// configured default. A model override applies to whichever provider is chosen.
pub fn make_provider(
    config: &HarnessConfig,
    override_provider: Option<&str>,
    override_model: Option<&str>,
) -> Result<Arc<dyn LlmProvider>> {
    let name = override_provider.unwrap_or(&config.provider.default);

    match name {
        "mock" => Ok(Arc::new(scripted_mock())),
        "anthropic" => {
            let key = config.api_key_for("anthropic").ok_or_else(|| {
                anyhow::anyhow!(
                    "No Anthropic API key. Set ANTHROPIC_API_KEY, or run with --provider openai / --mock."
                )
            })?;
            let model = override_model.unwrap_or(&config.provider.anthropic.model);
            Ok(Arc::new(
                AnthropicProvider::new(key, config.provider.anthropic.base_url.clone(), model)
                    .with_max_tokens(config.provider.anthropic.max_tokens),
            ))
        }
        "openai" => {
            let key = config.api_key_for("openai").ok_or_else(|| {
                anyhow::anyhow!(
                    "No OpenAI API key. Set OPENAI_API_KEY, or run with --provider anthropic / --mock. \
                     For a local server (Ollama, vLLM) set OPENAI_API_KEY to any placeholder and point \
                     LLM_BASE_URL at it."
                )
            })?;
            let model = override_model.unwrap_or(&config.provider.openai.model);
            Ok(Arc::new(GenericOpenAiProvider::new(
                key,
                config.provider.openai.base_url.clone(),
                model,
            )))
        }
        other => anyhow::bail!(
            "Unknown provider '{}'. Expected 'anthropic', 'openai', or 'mock'.",
            other
        ),
    }
}

/// The `--mock` provider: a fixed script that exercises the full loop offline —
/// delegate to a sub-agent, have it write a file, then answer. Used by the smoke
/// tests and for trying the REPL without an API key.
pub fn scripted_mock() -> MockProvider {
    MockProvider::new(vec![
        LlmResponse {
            content: Some("I'll delegate the file write to an engineer.".to_string()),
            tool_calls: vec![agent_core::ToolCall {
                id: "call_spawn_1".to_string(),
                name: "spawn_subagent".to_string(),
                arguments: serde_json::json!({
                    "role": "engineer",
                    "task": "Create harness_mock_artifact.txt containing the text 'mock harness ok'."
                }),
            }],
            usage: Some(TokenUsage::new(120, 30)),
            stop_reason: Some("tool_use".to_string()),
        },
        // The sub-agent's own two turns.
        LlmResponse {
            content: None,
            tool_calls: vec![agent_core::ToolCall {
                id: "call_write_1".to_string(),
                name: "write_file".to_string(),
                arguments: serde_json::json!({
                    "path": "harness_mock_artifact.txt",
                    "content": "mock harness ok\n"
                }),
            }],
            usage: Some(TokenUsage::new(80, 25)),
            stop_reason: Some("tool_use".to_string()),
        },
        LlmResponse {
            content: Some("Wrote harness_mock_artifact.txt.".to_string()),
            usage: Some(TokenUsage::new(95, 12)),
            stop_reason: Some("end_turn".to_string()),
            ..Default::default()
        },
        // Back in the lead agent.
        LlmResponse {
            content: Some(
                "Done. The engineer created harness_mock_artifact.txt with the requested contents."
                    .to_string(),
            ),
            usage: Some(TokenUsage::new(210, 20)),
            stop_reason: Some("end_turn".to_string()),
            ..Default::default()
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Arc<dyn LlmProvider>` is not `Debug`, so `unwrap_err` is unavailable.
    fn expect_err(result: Result<Arc<dyn LlmProvider>>) -> String {
        match result {
            Ok(provider) => panic!("expected an error, got provider {}", provider.provider_name()),
            Err(err) => err.to_string(),
        }
    }

    #[test]
    fn mock_provider_needs_no_key() {
        let config = HarnessConfig::default();
        let provider = make_provider(&config, Some("mock"), None).unwrap();
        assert_eq!(provider.provider_name(), "mock");
    }

    #[test]
    fn unknown_provider_is_rejected() {
        let config = HarnessConfig::default();
        let err = expect_err(make_provider(&config, Some("gemini"), None));
        assert!(err.contains("Unknown provider"));
    }

    #[test]
    fn model_override_wins_over_config() {
        let mut config = HarnessConfig::default();
        config.provider.openai.api_key = Some("test-key".to_string());
        let provider = make_provider(&config, Some("openai"), Some("gpt-4o-mini")).unwrap();
        assert_eq!(provider.model_name(), "gpt-4o-mini");
    }

    #[test]
    fn missing_key_produces_an_actionable_error() {
        let mut config = HarnessConfig::default();
        config.provider.anthropic.api_key = None;
        // Only meaningful when the ambient env has no key; skip otherwise.
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            return;
        }
        let err = expect_err(make_provider(&config, Some("anthropic"), None));
        assert!(err.contains("ANTHROPIC_API_KEY"));
    }
}
