use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tracing::warn;

/// Layered configuration. Precedence, lowest to highest:
/// defaults -> `~/.config/harness/config.toml` -> environment -> CLI flags.
/// CLI flags are applied by `main` after `load`, since clap owns them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct HarnessConfig {
    pub provider: ProviderConfig,
    pub limits: LimitsConfig,
    pub permissions: PermissionsConfig,
    /// Extra personas merged into the spawn tool's role list.
    pub personas: BTreeMap<String, PersonaConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ProviderConfig {
    /// "anthropic" or "openai".
    pub default: String,
    pub anthropic: AnthropicConfig,
    pub openai: OpenAiConfig,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            default: "anthropic".to_string(),
            anthropic: AnthropicConfig::default(),
            openai: OpenAiConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AnthropicConfig {
    pub model: String,
    pub base_url: String,
    pub max_tokens: usize,
    /// Prefer the `ANTHROPIC_API_KEY` environment variable; a key set here is
    /// honoured but warned about, since config files get committed by accident.
    pub api_key: Option<String>,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-5".to_string(),
            base_url: agent_core::providers::anthropic::DEFAULT_ANTHROPIC_BASE_URL.to_string(),
            max_tokens: agent_core::providers::anthropic::DEFAULT_MAX_TOKENS,
            api_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OpenAiConfig {
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LimitsConfig {
    pub max_iterations: usize,
    pub max_context_tokens: usize,
    pub max_parallel_subagents: usize,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            max_context_tokens: 128_000,
            max_parallel_subagents: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PermissionsConfig {
    /// Tools that never prompt, even when they declare `requires_approval`.
    pub auto_approve: Vec<String>,
    /// Approve everything without asking. Intended for CI and scripted runs.
    pub yolo: bool,
}

impl Default for PermissionsConfig {
    fn default() -> Self {
        Self {
            // Read-only tools don't declare approval anyway; listing them keeps
            // the intent explicit and gives users an obvious place to edit.
            auto_approve: vec![
                "read_file".to_string(),
                "list_directory".to_string(),
                "grep_search".to_string(),
                "find_files_by_name".to_string(),
            ],
            yolo: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaConfig {
    #[serde(default)]
    pub display_name: Option<String>,
    pub prompt: String,
}

impl HarnessConfig {
    /// Default config file location, honouring `XDG_CONFIG_HOME`.
    pub fn default_path() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
        Some(base.join("harness").join("config.toml"))
    }

    /// Loads defaults, then the config file if present, then environment
    /// overrides. A missing config file is normal, not an error.
    pub fn load(explicit_path: Option<&Path>) -> Result<Self> {
        let path = explicit_path.map(PathBuf::from).or_else(Self::default_path);

        let mut builder = ::config::Config::builder();
        // Serializing the defaults means every field has a value before any
        // layer is applied, so a partial config file only overrides what it sets.
        builder = builder.add_source(::config::Config::try_from(&Self::default())?);

        if let Some(path) = &path {
            if path.exists() {
                builder = builder.add_source(
                    ::config::File::from(path.as_path()).format(::config::FileFormat::Toml),
                );
            } else if explicit_path.is_some() {
                anyhow::bail!("Config file not found: {}", path.display());
            }
        }

        // HARNESS_PROVIDER__DEFAULT=openai, HARNESS_LIMITS__MAX_ITERATIONS=40, ...
        builder = builder.add_source(
            ::config::Environment::with_prefix("HARNESS")
                .separator("__")
                .try_parsing(true)
                .list_separator(",")
                .with_list_parse_key("permissions.auto_approve"),
        );

        let mut config: HarnessConfig = builder
            .build()
            .context("failed to assemble harness configuration")?
            .try_deserialize()
            .context("harness configuration is malformed")?;

        config.apply_legacy_env();
        config.validate()?;
        Ok(config)
    }

    /// The env vars the pre-config version of this tool used. Kept working so
    /// existing setups don't break.
    fn apply_legacy_env(&mut self) {
        if let Ok(url) = std::env::var("LLM_BASE_URL") {
            self.provider.openai.base_url = url;
        }
        if let Ok(model) = std::env::var("LLM_MODEL") {
            self.provider.openai.model = model;
        }
    }

    fn validate(&self) -> Result<()> {
        if !matches!(
            self.provider.default.as_str(),
            "anthropic" | "openai" | "mock"
        ) {
            anyhow::bail!(
                "Unknown provider '{}'. Expected 'anthropic', 'openai', or 'mock'.",
                self.provider.default
            );
        }
        if self.limits.max_iterations == 0 {
            anyhow::bail!("limits.max_iterations must be greater than zero");
        }
        if self.limits.max_context_tokens == 0 {
            anyhow::bail!("limits.max_context_tokens must be greater than zero");
        }
        if self.provider.anthropic.max_tokens == 0 {
            anyhow::bail!("provider.anthropic.max_tokens must be greater than zero");
        }
        Ok(())
    }

    /// Resolves the API key for a provider: environment first, config file as a
    /// fallback that warns.
    pub fn api_key_for(&self, provider: &str) -> Option<String> {
        let (env_var, from_file) = match provider {
            "anthropic" => ("ANTHROPIC_API_KEY", &self.provider.anthropic.api_key),
            _ => ("OPENAI_API_KEY", &self.provider.openai.api_key),
        };
        if let Ok(key) = std::env::var(env_var) {
            if !key.is_empty() {
                return Some(key);
            }
        }
        if let Some(key) = from_file {
            warn!(
                "Using an API key from the config file. Prefer the {} environment variable.",
                env_var
            );
            return Some(key.clone());
        }
        None
    }

    pub fn model_for(&self, provider: &str) -> &str {
        match provider {
            "anthropic" => &self.provider.anthropic.model,
            _ => &self.provider.openai.model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let config = HarnessConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.provider.default, "anthropic");
        assert_eq!(config.limits.max_parallel_subagents, 4);
        assert!(config
            .permissions
            .auto_approve
            .contains(&"read_file".to_string()));
    }

    #[test]
    fn rejects_unknown_provider() {
        let mut config = HarnessConfig::default();
        config.provider.default = "gemini".to_string();
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("Unknown provider"));
    }

    #[test]
    fn rejects_zero_limits() {
        let mut config = HarnessConfig::default();
        config.limits.max_iterations = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn partial_config_file_only_overrides_what_it_sets() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[provider]\ndefault = \"openai\"\n\n[limits]\nmax_iterations = 7\n",
        )
        .unwrap();

        let config = HarnessConfig::load(Some(&path)).unwrap();
        assert_eq!(config.provider.default, "openai");
        assert_eq!(config.limits.max_iterations, 7);
        // Untouched fields keep their defaults.
        assert_eq!(config.limits.max_parallel_subagents, 4);
        assert_eq!(config.provider.anthropic.model, "claude-sonnet-5");
    }

    #[test]
    fn custom_personas_are_parsed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[personas.dba]\ndisplay_name = \"DatabaseExpert\"\nprompt = \"You tune queries.\"\n",
        )
        .unwrap();

        let config = HarnessConfig::load(Some(&path)).unwrap();
        let persona = config.personas.get("dba").expect("persona should parse");
        assert_eq!(persona.display_name.as_deref(), Some("DatabaseExpert"));
        assert_eq!(persona.prompt, "You tune queries.");
    }

    #[test]
    fn explicit_missing_path_is_an_error() {
        let err = HarnessConfig::load(Some(Path::new("/nonexistent/harness.toml"))).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }
}
