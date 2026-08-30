use crate::delegation_tool::SpawnSubAgentTool;
use crate::personas::get_planner_prompt;
use agent_core::{Agent, LlmProvider};
use anyhow::Result;
use harness_core::{BashCommandTool, HarnessToolRegistry, ListDirTool, ReadFileTool, WriteFileTool};
use std::sync::Arc;

pub struct MultiAgentOrchestrator {
    pub provider: Arc<dyn LlmProvider>,
}

impl MultiAgentOrchestrator {
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self { provider }
    }

    /// Builds the primary Lead Planner agent with full sub-agent delegation capability
    pub fn create_lead_agent(&self) -> Result<Agent> {
        let mut tool_registry = HarnessToolRegistry::new();

        // Register default environment tools
        tool_registry.register(Arc::new(ReadFileTool));
        tool_registry.register(Arc::new(WriteFileTool));
        tool_registry.register(Arc::new(ListDirTool));
        tool_registry.register(Arc::new(BashCommandTool));

        let registry_arc = Arc::new(tool_registry);

        // Register the sub-agent delegation tool
        let subagent_tool = Arc::new(SpawnSubAgentTool::new(
            Arc::clone(&self.provider),
            Arc::clone(&registry_arc) as Arc<dyn agent_core::ToolDispatcher>,
        ));

        // Create a new dispatcher that includes subagent delegation
        let mut full_registry = HarnessToolRegistry::new();
        full_registry.register(Arc::new(ReadFileTool));
        full_registry.register(Arc::new(WriteFileTool));
        full_registry.register(Arc::new(ListDirTool));
        full_registry.register(Arc::new(BashCommandTool));
        full_registry.register(subagent_tool);

        let lead_dispatcher = Arc::new(full_registry);

        let planner = Agent::new(
            "lead-planner",
            "LeadPlanner",
            get_planner_prompt(),
            Arc::clone(&self.provider),
            lead_dispatcher,
        );

        Ok(planner)
    }
}
