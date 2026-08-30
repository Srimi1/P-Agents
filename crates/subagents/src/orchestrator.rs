use crate::delegation_tool::{RunParallelSubAgentsTool, SpawnSubAgentTool, SubAgentFactory};
use crate::persona_registry::PersonaRegistry;
use agent_core::compaction::HistoryCompactor;
use agent_core::events::EventSink;
use agent_core::{Agent, LlmProvider, ToolCall, ToolDefinition, ToolDispatcher};
use anyhow::Result;
use async_trait::async_trait;
use harness_core::{
    BashCommandTool, EditFileBlockTool, FindFilesByNameTool, GrepSearchTool, HarnessToolRegistry,
    ListDirTool, ReadFileTool, Tool, WriteFileTool,
};
use std::collections::HashMap;
use std::sync::Arc;

/// The default environment tools every agent gets. The caller wraps the result
/// in the runtime's `GatedDispatcher` so approval policy applies to lead and
/// sub-agents alike.
pub fn build_tool_registry() -> HarnessToolRegistry {
    let mut registry = HarnessToolRegistry::new();
    registry.register(Arc::new(ReadFileTool));
    registry.register(Arc::new(WriteFileTool));
    registry.register(Arc::new(EditFileBlockTool));
    registry.register(Arc::new(ListDirTool));
    registry.register(Arc::new(GrepSearchTool));
    registry.register(Arc::new(FindFilesByNameTool));
    registry.register(Arc::new(BashCommandTool));
    registry
}

/// Adds the delegation tools on top of an existing dispatcher without putting
/// them into the shared registry — which is precisely what keeps sub-agents
/// from spawning sub-agents of their own.
///
/// Delegation itself touches nothing outside the process, so these tools run
/// unwrapped; everything the spawned agent then does still goes through `base`.
pub struct LeadDispatcher {
    base: Arc<dyn ToolDispatcher>,
    extra: HashMap<String, Arc<dyn Tool>>,
}

impl LeadDispatcher {
    pub fn new(base: Arc<dyn ToolDispatcher>, extra: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            base,
            extra: extra
                .into_iter()
                .map(|t| (t.name().to_string(), t))
                .collect(),
        }
    }
}

#[async_trait]
impl ToolDispatcher for LeadDispatcher {
    fn get_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = self.base.get_definitions();
        defs.extend(self.extra.values().map(|t| ToolDefinition {
            name: t.name().to_string(),
            description: t.description().to_string(),
            parameters: t.parameters_schema(),
        }));
        defs
    }

    async fn dispatch(&self, agent_id: &str, tool_call: &ToolCall) -> Result<String> {
        match self.extra.get(&tool_call.name) {
            Some(tool) => tool.execute(tool_call.arguments.clone()).await,
            None => self.base.dispatch(agent_id, tool_call).await,
        }
    }
}

pub const LEAD_AGENT_ID: &str = "lead-planner";

pub struct MultiAgentOrchestrator {
    pub provider: Arc<dyn LlmProvider>,
    personas: Arc<PersonaRegistry>,
    max_parallel: usize,
    max_iterations: usize,
}

impl MultiAgentOrchestrator {
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self {
            provider,
            personas: Arc::new(PersonaRegistry::new()),
            max_parallel: 4,
            max_iterations: 20,
        }
    }

    pub fn with_personas(mut self, personas: PersonaRegistry) -> Self {
        self.personas = Arc::new(personas);
        self
    }

    pub fn with_max_parallel(mut self, max_parallel: usize) -> Self {
        self.max_parallel = max_parallel.max(1);
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations.max(1);
        self
    }

    pub fn personas(&self) -> &PersonaRegistry {
        &self.personas
    }

    /// Swaps the model. The caller must rebuild its agents afterwards (moving
    /// the old history across with `Agent::restore_history`) so sub-agents pick
    /// up the new provider too.
    pub fn set_provider(&mut self, provider: Arc<dyn LlmProvider>) {
        self.provider = provider;
    }

    fn subagent_factory(
        &self,
        dispatcher: Arc<dyn ToolDispatcher>,
        events: Option<EventSink>,
        compactor: Option<Arc<dyn HistoryCompactor>>,
        parent_id: &str,
    ) -> Arc<SubAgentFactory> {
        Arc::new(
            SubAgentFactory::new(
                Arc::clone(&self.provider),
                dispatcher,
                Arc::clone(&self.personas),
                parent_id,
            )
            .with_events(events)
            .with_compactor(compactor)
            .with_max_iterations(self.max_iterations),
        )
    }

    /// The Lead Planner: every environment tool, plus delegation.
    ///
    /// `dispatcher` is the gated dispatcher built by the runtime; it is handed
    /// to sub-agents unchanged so their tool calls face the same approval gate.
    pub fn create_lead_agent(
        &self,
        dispatcher: Arc<dyn ToolDispatcher>,
        events: Option<EventSink>,
        compactor: Option<Arc<dyn HistoryCompactor>>,
    ) -> Result<Agent> {
        let factory = self.subagent_factory(
            Arc::clone(&dispatcher),
            events.clone(),
            compactor.clone(),
            LEAD_AGENT_ID,
        );

        let lead_dispatcher = Arc::new(LeadDispatcher::new(
            dispatcher,
            vec![
                Arc::new(SpawnSubAgentTool::new(Arc::clone(&factory))) as Arc<dyn Tool>,
                Arc::new(RunParallelSubAgentsTool::new(factory, self.max_parallel)) as Arc<dyn Tool>,
            ],
        ));

        let mut agent = Agent::new(
            LEAD_AGENT_ID,
            "LeadPlanner",
            self.personas.planner_prompt(),
            Arc::clone(&self.provider),
            lead_dispatcher,
        )
        .with_max_iterations(self.max_iterations);

        if let Some(events) = events {
            agent = agent.with_events(events);
        }
        if let Some(compactor) = compactor {
            agent = agent.with_compactor(compactor);
        }
        Ok(agent)
    }

    /// A single specialist agent with no delegation tools. Backs the `/critic`,
    /// `/verify` and `/plan` slash commands.
    pub fn create_persona_agent(
        &self,
        role: &str,
        dispatcher: Arc<dyn ToolDispatcher>,
        events: Option<EventSink>,
        compactor: Option<Arc<dyn HistoryCompactor>>,
    ) -> Result<Agent> {
        let persona = self.personas.get(role).ok_or_else(|| {
            anyhow::anyhow!(
                "Unknown persona '{}'. Available: {}.",
                role,
                self.personas.roles().join(", ")
            )
        })?;

        let mut agent = Agent::new(
            format!("{}-oneshot", role),
            persona.display_name.clone(),
            persona.prompt.clone(),
            Arc::clone(&self.provider),
            dispatcher,
        )
        .with_max_iterations(self.max_iterations);

        if let Some(events) = events {
            agent = agent.with_events(events);
        }
        if let Some(compactor) = compactor {
            agent = agent.with_compactor(compactor);
        }
        Ok(agent)
    }
}
