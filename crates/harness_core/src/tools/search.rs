use crate::tool_registry::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;

pub struct GrepSearchTool;

#[async_trait]
impl Tool for GrepSearchTool {
    fn name(&self) -> &str {
        "grep_search"
    }

    fn description(&self) -> &str {
        "Searches file contents for a regular expression."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {}, "required": [] })
    }

    async fn execute(&self, _args: serde_json::Value) -> Result<String> {
        anyhow::bail!("grep_search not implemented yet")
    }
}

pub struct FindFilesByNameTool;

#[async_trait]
impl Tool for FindFilesByNameTool {
    fn name(&self) -> &str {
        "find_files_by_name"
    }

    fn description(&self) -> &str {
        "Finds files whose path matches a glob pattern."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {}, "required": [] })
    }

    async fn execute(&self, _args: serde_json::Value) -> Result<String> {
        anyhow::bail!("find_files_by_name not implemented yet")
    }
}
