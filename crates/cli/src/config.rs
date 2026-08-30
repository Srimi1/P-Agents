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
    /// "anthropic", "openai", or "gemini".
    pub default: String,
    pub anthropic: AnthropicConfig,
    pub openai: OpenAiConfig,
    pub gemini: GeminiConfig,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            default: "anthropic".to_string(),
            anthropic: AnthropicConfig::default(),
            openai: OpenAiConfig::default(),
            gemini: GeminiConfig::default(),
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

/// Gemini speaks the OpenAI chat format on a compatibility endpoint, so it
/// reuses the OpenAI client rather than needing a provider of its own.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeminiConfig {
    pub model: String,
    pub base_url: String,
    pub api_key: Option<String>,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            model: "gemini-2.5-flash".to_string(),
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai".to_string(),
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
    /// Confine the file tools to `workspace_roots`. Reads are not gated by the
    /// approval prompt, so without this an agent can read any file the process
    /// can reach.
    pub sandbox: bool,
    /// Roots the file tools may touch. Empty means the working directory.
    pub workspace_roots: Vec<PathBuf>,
    /// How far an "always approve" answer reaches: "tool" covers that tool for
    /// every agent, "agent" covers it only for the agent that was asked.
    pub grant_scope: String,
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
            sandbox: true,
            workspace_roots: Vec::new(),
            grant_scope: "tool".to_string(),
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
                // The prefix separator defaults to `separator`, which would
                // demand `HARNESS__LIMITS__...`. One underscore after the
                // prefix, two between levels.
                .prefix_separator("_")
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
            "anthropic" | "openai" | "gemini" | "mock"
        ) {
            anyhow::bail!(
                "Unknown provider '{}'. Expected 'anthropic', 'openai', 'gemini', or 'mock'.",
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
        if !matches!(self.permissions.grant_scope.as_str(), "tool" | "agent") {
            anyhow::bail!(
                "Unknown permissions.grant_scope '{}'. Expected 'tool' or 'agent'.",
                self.permissions.grant_scope
            );
        }
        Ok(())
    }

    /// Resolves the API key for a provider: environment first, config file as a
    /// fallback that warns.
    pub fn api_key_for(&self, provider: &str) -> Option<String> {
        let (env_var, from_file) = match provider {
            "anthropic" => ("ANTHROPIC_API_KEY", &self.provider.anthropic.api_key),
            "gemini" => ("GEMINI_API_KEY", &self.provider.gemini.api_key),
            _ => ("OPENAI_API_KEY", &self.provider.openai.api_key),
        };
        if let Ok(key) = std::env::var(env_var) {
            if !key.is_empty() {
                return Some(key);
            }
        }
        // An empty key in the file is the same as no key: treating it as one
        // trades a clear startup error for a confusing 401 mid-turn.
        if let Some(key) = from_file.as_ref().filter(|k| !k.is_empty()) {
            warn!(
                "Using an API key from the config file. Prefer the {} environment variable.",
                env_var
            );
            return Some(key.clone());
        }
        None
    }

    /// Builds the filesystem containment policy. Defaults to the working
    /// directory when no roots are configured, so a fresh install is confined
    /// rather than open.
    pub fn workspace_policy(&self) -> Result<harness_core::WorkspacePolicy> {
        if !self.permissions.sandbox {
            return Ok(harness_core::WorkspacePolicy::unrestricted());
        }
        if self.permissions.workspace_roots.is_empty() {
            return harness_core::WorkspacePolicy::current_dir();
        }
        harness_core::WorkspacePolicy::with_roots(&self.permissions.workspace_roots)
    }

    pub fn grant_scope(&self) -> runtime::GrantScope {
        match self.permissions.grant_scope.as_str() {
            "agent" => runtime::GrantScope::Agent,
            _ => runtime::GrantScope::Tool,
        }
    }

    pub fn model_for(&self, provider: &str) -> &str {
        match provider {
            "anthropic" => &self.provider.anthropic.model,
            "gemini" => &self.provider.gemini.model,
            _ => &self.provider.openai.model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn environment_overrides_use_a_single_underscore_after_the_prefix() {
        // Guards the documented form; the config crate's default would have
        // required HARNESS__PROVIDER__ANTHROPIC__MAX_TOKENS instead.
        // Env is process-wide, so this uses a key no other test asserts on.
        let key = "HARNESS_PROVIDER__ANTHROPIC__MAX_TOKENS";
        std::env::set_var(key, "4321");
        let loaded = HarnessConfig::load(None);
        std::env::remove_var(key);
        assert_eq!(
            loaded
                .expect("config should load")
                .provider
                .anthropic
                .max_tokens,
            4321
        );
    }

    #[test]
    fn an_empty_api_key_in_the_file_is_treated_as_absent() {
        let mut config = HarnessConfig::default();
        config.provider.anthropic.api_key = Some(String::new());
        if std::env::var("ANTHROPIC_API_KEY").is_ok() {
            return;
        }
        assert!(config.api_key_for("anthropic").is_none());
    }

    #[test]
    fn the_sandbox_is_on_by_default_and_confines_to_the_working_directory() {
        let config = HarnessConfig::default();
        assert!(config.permissions.sandbox, "containment must not be opt-in");
        let policy = config.workspace_policy().expect("policy");
        assert!(!policy.is_unrestricted());
        assert!(policy.resolve("/etc/passwd").is_err());
    }

    #[test]
    fn the_sandbox_can_be_turned_off_explicitly() {
        let mut config = HarnessConfig::default();
        config.permissions.sandbox = false;
        assert!(config.workspace_policy().expect("policy").is_unrestricted());
    }

    #[test]
    fn grant_scope_parses_and_is_validated() {
        let mut config = HarnessConfig::default();
        assert_eq!(config.grant_scope(), runtime::GrantScope::Tool);
        config.permissions.grant_scope = "agent".to_string();
        assert!(config.validate().is_ok());
        assert_eq!(config.grant_scope(), runtime::GrantScope::Agent);
        config.permissions.grant_scope = "everything".to_string();
        assert!(config.validate().is_err());
    }

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
        config.provider.default = "hal9000".to_string();
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
