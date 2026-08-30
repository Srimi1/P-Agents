use crate::tool_registry::Tool;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::timeout;

pub struct BashCommandTool;

#[async_trait]
impl Tool for BashCommandTool {
    fn name(&self) -> &str {
        "run_bash_command"
    }

    fn description(&self) -> &str {
        "Executes a bash command in the workspace directory with a timeout."
    }

    fn requires_approval(&self) -> bool {
        true
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (optional, default: current directory)."
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (optional, default: 30)."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let command_str = args["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'command' argument"))?;
        let cwd = args["cwd"].as_str().unwrap_or(".");
        let timeout_secs = args["timeout_secs"].as_u64().unwrap_or(30);

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(command_str).current_dir(cwd);

        let output_future = cmd.output();
        let output = match timeout(Duration::from_secs(timeout_secs), output_future).await {
            Ok(res) => res?,
            Err(_) => anyhow::bail!("Command timed out after {} seconds.", timeout_secs),
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        let mut result = format!("Exit Code: {}\n", exit_code);
        if !stdout.is_empty() {
            result.push_str(&format!("STDOUT:\n{}\n", stdout));
        }
        if !stderr.is_empty() {
            result.push_str(&format!("STDERR:\n{}\n", stderr));
        }

        Ok(result)
    }
}
