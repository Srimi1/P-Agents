use crate::config::HarnessConfig;
use crate::provider_factory::make_provider;
use agent_core::{Agent, ChatMessage, HistoryCompactor, LlmProvider, ToolDispatcher};
use anyhow::Result;
use harness_core::ContextManager;
use runtime::{ApprovalRequest, HarnessRuntime, SecurityManager, SessionStore};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use subagents::{build_tool_registry, MultiAgentOrchestrator, PersonaRegistry, LEAD_AGENT_ID};
use tokio::sync::mpsc;

/// Where session transcripts live, relative to the working directory.
pub const SESSION_DIR: &str = ".harness/sessions";

/// Everything wired together: runtime, orchestrator, gated dispatcher, and the
/// lead agent. Split out from the REPL so integration tests can drive a full
/// harness without a terminal.
pub struct HarnessApp {
    pub config: HarnessConfig,
    pub runtime: HarnessRuntime,
    pub orchestrator: MultiAgentOrchestrator,
    pub dispatcher: Arc<dyn ToolDispatcher>,
    pub compactor: Arc<dyn HistoryCompactor>,
    pub lead: Agent,
    pub provider_name: String,
    pub model_name: String,
}

pub struct AppOptions {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub yolo: bool,
    pub session_dir: PathBuf,
}

impl Default for AppOptions {
    fn default() -> Self {
        Self {
            provider: None,
            model: None,
            yolo: false,
            session_dir: PathBuf::from(SESSION_DIR),
        }
    }
}

impl HarnessApp {
    pub async fn new(
        config: HarnessConfig,
        options: AppOptions,
    ) -> Result<(Self, mpsc::Receiver<ApprovalRequest>)> {
        let provider = make_provider(
            &config,
            options.provider.as_deref(),
            options.model.as_deref(),
        )?;
        Self::with_provider(config, options, provider).await
    }

    /// Assembles the harness around an already-built provider. Integration
    /// tests use this to drive a scripted model without touching a network.
    pub async fn with_provider(
        config: HarnessConfig,
        options: AppOptions,
        provider: Arc<dyn LlmProvider>,
    ) -> Result<(Self, mpsc::Receiver<ApprovalRequest>)> {
        let provider_name = provider.provider_name().to_string();
        let model_name = provider.model_name().to_string();

        let yolo = options.yolo || config.permissions.yolo;
        let security = SecurityManager::new()
            .with_yolo(yolo)
            .with_auto_approved(config.permissions.auto_approve.clone());

        let (runtime, approvals) =
            HarnessRuntime::new(&options.session_dir, &model_name, security, yolo).await?;

        let registry = Arc::new(build_tool_registry());
        let dispatcher: Arc<dyn ToolDispatcher> = runtime.dispatcher(registry);

        let compactor: Arc<dyn HistoryCompactor> =
            Arc::new(ContextManager::new(config.limits.max_context_tokens));

        let orchestrator = build_orchestrator(&config, provider);
        let lead = orchestrator.create_lead_agent(
            Arc::clone(&dispatcher),
            Some(runtime.event_sink()),
            Some(Arc::clone(&compactor)),
        )?;

        Ok((
            Self {
                config,
                runtime,
                orchestrator,
                dispatcher,
                compactor,
                lead,
                provider_name,
                model_name,
            },
            approvals,
        ))
    }

    pub async fn run_prompt(&mut self, prompt: &str) -> Result<String> {
        self.lead.run(prompt).await
    }

    /// Hot-swaps the model. The lead agent is rebuilt so sub-agents spawned
    /// afterwards use the new provider too; the transcript carries across.
    pub fn swap_model(&mut self, provider: Option<&str>, model: Option<&str>) -> Result<()> {
        let new_provider = make_provider(&self.config, provider, model)?;
        self.provider_name = new_provider.provider_name().to_string();
        self.model_name = new_provider.model_name().to_string();

        self.orchestrator.set_provider(Arc::clone(&new_provider));
        let history = std::mem::take(&mut self.lead.history);
        let mut lead = self.orchestrator.create_lead_agent(
            Arc::clone(&self.dispatcher),
            Some(self.runtime.event_sink()),
            Some(Arc::clone(&self.compactor)),
        )?;
        lead.restore_history(history);
        lead.cumulative_usage = self.lead.cumulative_usage;
        self.lead = lead;
        Ok(())
    }

    /// Replays a previous session's lead transcript into the current agent.
    pub async fn resume(&mut self, session_dir: &Path, session_id: &str) -> Result<usize> {
        let path = SessionStore::find_by_id(session_dir, session_id).await?;
        let records = SessionStore::load(&path).await?;
        let history = SessionStore::rebuild_history(&records, LEAD_AGENT_ID);
        if history.is_empty() {
            anyhow::bail!(
                "Session {} has no messages for the lead agent",
                path.display()
            );
        }
        let restored = history.len();
        self.lead.restore_history(history);
        Ok(restored)
    }

    /// Runs a single specialist persona once, outside the lead's transcript.
    /// Backs `/critic` and `/verify`.
    pub async fn run_persona_once(&self, role: &str, task: &str) -> Result<String> {
        let mut agent = self.orchestrator.create_persona_agent(
            role,
            Arc::clone(&self.dispatcher),
            Some(self.runtime.event_sink()),
            Some(Arc::clone(&self.compactor)),
        )?;
        agent.run(task).await
    }

    /// The most recent assistant answer, used as the default subject for
    /// `/critic` and `/verify` when the user gives no text of their own.
    pub fn last_answer(&self) -> Option<&str> {
        self.lead
            .history
            .iter()
            .rev()
            .find(|m| m.role == agent_core::Role::Assistant && m.content.is_some())
            .and_then(|m| m.content.as_deref())
    }

    pub fn history(&self) -> &[ChatMessage] {
        &self.lead.history
    }

    pub async fn shutdown(self) {
        self.runtime.shutdown().await;
    }
}

fn build_orchestrator(
    config: &HarnessConfig,
    provider: Arc<dyn LlmProvider>,
) -> MultiAgentOrchestrator {
    let mut personas = PersonaRegistry::new();
    for (role, persona) in &config.personas {
        let display_name = persona.display_name.clone().unwrap_or_else(|| role.clone());
        personas.insert(role.clone(), display_name, persona.prompt.clone());
    }

    MultiAgentOrchestrator::new(provider)
        .with_personas(personas)
        .with_max_parallel(config.limits.max_parallel_subagents)
        .with_max_iterations(config.limits.max_iterations)
}
