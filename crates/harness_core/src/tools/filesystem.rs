use crate::tool_registry::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::path::Path;
use tokio::fs;

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads the contents of a file at the given path."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'path' argument"))?;

        let path = Path::new(path_str);
        if !path.exists() {
            anyhow::bail!("File not found: {}", path_str);
        }

        let content = fs::read_to_string(path).await?;
        Ok(content)
    }
}

pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Creates or overwrites a file with the specified content."
    }

    fn requires_approval(&self) -> bool {
        true
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to write."
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write."
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let path_str = args["path"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'path' argument"))?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'content' argument"))?;

        let path = Path::new(path_str);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        fs::write(path, content).await?;
        Ok(format!("Successfully wrote {} bytes to '{}'", content.len(), path_str))
    }
}

pub struct ListDirTool;

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "Lists entries inside a directory."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let path_str = args["path"].as_str().unwrap_or(".");
        let mut dir = fs::read_dir(path_str).await?;
        let mut entries = Vec::new();

        while let Some(entry) = dir.next_entry().await? {
            let file_name = entry.file_name().to_string_lossy().to_string();
            let file_type = entry.file_type().await?;
            let kind = if file_type.is_dir() { "DIR" } else { "FILE" };
            entries.push(format!("[{}] {}", kind, file_name));
        }

        Ok(entries.join("\n"))
    }
}

pub struct EditFileBlockTool;

#[async_trait]
impl Tool for EditFileBlockTool {
    fn name(&self) -> &str {
        "edit_file_block"
    }

    fn description(&self) -> &str {
        "Replaces an exact block of text inside a file."
    }

    fn requires_approval(&self) -> bool {
        true
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {}, "required": [] })
    }

    async fn execute(&self, _args: serde_json::Value) -> Result<String> {
        anyhow::bail!("edit_file_block not implemented yet")
    }
}
