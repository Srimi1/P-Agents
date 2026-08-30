use agent_core::{ToolCall, ToolDefinition, ToolDispatcher};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn parameters_schema(&self) -> serde_json::Value;
    fn requires_approval(&self) -> bool {
        false
    }
    async fn execute(&self, args: serde_json::Value) -> Result<String>;
}

#[derive(Default)]
pub struct HarnessToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl HarnessToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }
}

#[async_trait]
impl ToolDispatcher for HarnessToolRegistry {
    fn get_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|t| ToolDefinition {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters_schema(),
            })
            .collect()
    }

    async fn dispatch(&self, _agent_id: &str, tool_call: &ToolCall) -> Result<String> {
        let tool = self
            .tools
            .get(&tool_call.name)
            .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found in registry", tool_call.name))?;

        tool.execute(tool_call.arguments.clone()).await
    }
}
