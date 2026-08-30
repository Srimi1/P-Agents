use crate::personas::*;
use agent_core::{Agent, LlmProvider, ToolDispatcher};
use anyhow::Result;
use async_trait::async_trait;
use harness_core::Tool;
use serde_json::json;
use std::sync::Arc;
use tracing::info;

pub struct SpawnSubAgentTool {
    pub provider: Arc<dyn LlmProvider>,
    pub dispatcher: Arc<dyn ToolDispatcher>,
}

impl SpawnSubAgentTool {
    pub fn new(provider: Arc<dyn LlmProvider>, dispatcher: Arc<dyn ToolDispatcher>) -> Self {
        Self {
            provider,
            dispatcher,
        }
    }
}

#[async_trait]
impl Tool for SpawnSubAgentTool {
    fn name(&self) -> &str {
        "spawn_subagent"
    }

    fn description(&self) -> &str {
        "Spawns an isolated specialist sub-agent to execute a specific subtask. Roles available: 'engineer', 'verifier', 'critic', 'researcher', 'analyst'."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["engineer", "verifier", "critic", "researcher", "analyst"],
                    "description": "The specialist role to spawn."
                },
                "task": {
                    "type": "string",
                    "description": "Clear, detailed task instructions for the subagent."
                }
            },
            "required": ["role", "task"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let role = args["role"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'role' parameter"))?;
        let task = args["task"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'task' parameter"))?;

        let (name, system_prompt) = match role {
            "engineer" => ("SoftwareEngineer", get_engineer_prompt()),
            "verifier" => ("Verifier", get_verifier_prompt()),
            "critic" => ("EgoistCritic", get_critic_prompt()),
            "researcher" => ("Researcher", get_researcher_prompt()),
            "analyst" => ("DataAnalyst", get_analyst_prompt()),
            _ => ("GeneralSpecialist", "You are a helpful specialist agent."),
        };

        info!(role = %role, "Spawning isolated sub-agent");

        let mut subagent = Agent::new(
            format!("subagent-{}", role),
            name,
            system_prompt,
            Arc::clone(&self.provider),
            Arc::clone(&self.dispatcher),
        );

        // Run sub-agent in isolated context
        let result = subagent.run(task).await?;
        info!(role = %role, "Sub-agent completed task successfully");

        Ok(format!("[Sub-Agent ({}) Result]:\n{}", name, result))
    }
}
