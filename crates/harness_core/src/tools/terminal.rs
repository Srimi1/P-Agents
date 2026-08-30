use crate::tool_registry::Tool;
use agent_core::truncate_at_boundary;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::time::timeout;

/// Combined stdout+stderr budget handed back to the model.
const MAX_OUTPUT_BYTES: usize = 16_000;

const DEFAULT_TIMEOUT_SECS: u64 = 30;

static BACKGROUND_JOBS: OnceLock<Mutex<HashMap<u32, Child>>> = OnceLock::new();

fn background_jobs() -> MutexGuard<'static, HashMap<u32, Child>> {
    // A poisoned lock only means some other caller panicked mid-insert; the map itself
    // stays valid, so recovering is preferable to propagating the panic.
    BACKGROUND_JOBS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Kills a background job started by `run_bash_command` and forgets it.
/// Returns false when the pid is not a job this process started.
pub async fn kill_background_job(pid: u32) -> Result<bool> {
    let child = background_jobs().remove(&pid);
    match child {
        Some(mut child) => {
            child.kill().await?;
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Pids of the background jobs this process started, in unspecified order.
pub fn background_job_pids() -> Vec<u32> {
    background_jobs().keys().copied().collect()
}

pub struct BashCommandTool;

#[async_trait]
impl Tool for BashCommandTool {
    fn name(&self) -> &str {
        "run_bash_command"
    }

    fn description(&self) -> &str {
        "Executes a bash command in the workspace directory with a timeout, or starts it in the background and returns its pid."
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
                    "description": "Timeout in seconds for foreground commands (optional, default: 30). Ignored when background is true."
                },
                "background": {
                    "type": "boolean",
                    "description": "Start the command detached and return its pid immediately instead of waiting (optional, default: false)."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let command_str = args["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'command' argument"))?;
        if command_str.trim().is_empty() {
            anyhow::bail!("'command' must not be empty");
        }

        let cwd = match &args["cwd"] {
            serde_json::Value::Null => ".",
            value => value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("'cwd' must be a string"))?,
        };

        let timeout_secs = match &args["timeout_secs"] {
            serde_json::Value::Null => DEFAULT_TIMEOUT_SECS,
            value => value
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("'timeout_secs' must be a positive integer"))?,
        };
        if timeout_secs == 0 {
            anyhow::bail!("'timeout_secs' must be greater than 0");
        }

        let background = match &args["background"] {
            serde_json::Value::Null => false,
            value => value
                .as_bool()
                .ok_or_else(|| anyhow::anyhow!("'background' must be a boolean"))?,
        };

        // A bad cwd otherwise surfaces as an opaque OS error from spawn().
        let metadata = tokio::fs::metadata(cwd)
            .await
            .map_err(|e| anyhow::anyhow!("Working directory '{}' is unusable: {}", cwd, e))?;
        if !metadata.is_dir() {
            anyhow::bail!("Working directory '{}' is not a directory", cwd);
        }

        if background {
            run_background(command_str, cwd)
        } else {
            run_foreground(command_str, cwd, timeout_secs).await
        }
    }
}

fn run_background(command_str: &str, cwd: &str) -> Result<String> {
    let child = Command::new("sh")
        .arg("-c")
        .arg(command_str)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(false)
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to start background command: {}", e))?;

    let pid = match child.id() {
        Some(pid) => pid,
        None => anyhow::bail!("Background command exited before a pid could be observed"),
    };

    background_jobs().insert(pid, child);

    Ok(format!(
        "Started background process with pid {}. Its output is not captured; redirect it to a file in the command if you need it, and check on the process with `ps -p {}`.\n",
        pid, pid
    ))
}

async fn run_foreground(command_str: &str, cwd: &str, timeout_secs: u64) -> Result<String> {
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command_str)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to start command: {}", e))?;

    let mut stdout_pipe = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("Child stdout was not captured"))?;
    let mut stderr_pipe = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("Child stderr was not captured"))?;

    let mut stdout_buf = Vec::new();
    let mut stderr_buf = Vec::new();

    // Both pipes must be drained while waiting, otherwise a command that fills the
    // pipe buffer blocks forever instead of exiting.
    let collect = async {
        let (_, _, status) = tokio::try_join!(
            stdout_pipe.read_to_end(&mut stdout_buf),
            stderr_pipe.read_to_end(&mut stderr_buf),
            child.wait(),
        )?;
        Ok::<_, std::io::Error>(status)
    };

    let outcome = timeout(Duration::from_secs(timeout_secs), collect).await;

    let status = match outcome {
        Ok(res) => res?,
        Err(_) => {
            // kill_on_drop would also reap it, but killing explicitly makes the
            // guarantee stated in the error message unconditional.
            let killed = child.kill().await.is_ok();
            anyhow::bail!(
                "Command timed out after {} seconds; the process was {}.",
                timeout_secs,
                if killed { "killed" } else { "already gone" }
            );
        }
    };

    let stdout = String::from_utf8_lossy(&stdout_buf);
    let stderr = String::from_utf8_lossy(&stderr_buf);
    Ok(format_output(status.code().unwrap_or(-1), &stdout, &stderr))
}

fn format_output(exit_code: i32, stdout: &str, stderr: &str) -> String {
    let original_len = stdout.len() + stderr.len();
    let (stdout, stderr) = cap_output(stdout, stderr);

    let mut result = format!("Exit Code: {}\n", exit_code);
    if !stdout.is_empty() {
        result.push_str(&format!("STDOUT:\n{}\n", stdout));
    }
    if !stderr.is_empty() {
        result.push_str(&format!("STDERR:\n{}\n", stderr));
    }
    if original_len > MAX_OUTPUT_BYTES {
        result.push_str(&format!(
            "[output truncated: {} bytes of combined stdout/stderr capped at {} bytes]\n",
            original_len, MAX_OUTPUT_BYTES
        ));
    }
    result
}

/// Splits the output budget between the two streams, letting either use the slack
/// the other leaves behind.
fn cap_output<'a>(stdout: &'a str, stderr: &'a str) -> (&'a str, &'a str) {
    if stdout.len() + stderr.len() <= MAX_OUTPUT_BYTES {
        return (stdout, stderr);
    }

    let half = MAX_OUTPUT_BYTES / 2;
    let (stdout_budget, stderr_budget) = if stderr.len() <= half {
        (MAX_OUTPUT_BYTES - stderr.len(), stderr.len())
    } else if stdout.len() <= half {
        (stdout.len(), MAX_OUTPUT_BYTES - stdout.len())
    } else {
        (half, MAX_OUTPUT_BYTES - half)
    };

    (
        truncate_at_boundary(stdout, stdout_budget),
        truncate_at_boundary(stderr, stderr_budget),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn shell_available() -> bool {
        std::process::Command::new("sh")
            .arg("-c")
            .arg("exit 0")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn extract_pid(report: &str) -> u32 {
        let digits: String = report
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits.parse().expect("report should contain a pid")
    }

    #[tokio::test]
    async fn captures_stdout_and_zero_exit_code() {
        if !shell_available() {
            return;
        }
        let tool = BashCommandTool;
        let out = tool
            .execute(json!({ "command": "echo hello" }))
            .await
            .expect("command should run");

        assert!(out.contains("Exit Code: 0"), "{}", out);
        assert!(out.contains("STDOUT:\nhello"), "{}", out);
        assert!(!out.contains("STDERR:"), "{}", out);
    }

    #[tokio::test]
    async fn reports_non_zero_exit_code_and_stderr() {
        if !shell_available() {
            return;
        }
        let tool = BashCommandTool;
        let out = tool
            .execute(json!({ "command": "echo boom 1>&2; exit 3" }))
            .await
            .expect("command should run");

        assert!(out.contains("Exit Code: 3"), "{}", out);
        assert!(out.contains("STDERR:\nboom"), "{}", out);
    }

    #[tokio::test]
    async fn runs_in_the_requested_cwd() {
        if !shell_available() {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let tool = BashCommandTool;
        let out = tool
            .execute(json!({
                "command": "printf marker > marker.txt",
                "cwd": dir.path().to_string_lossy(),
            }))
            .await
            .expect("command should run");

        assert!(out.contains("Exit Code: 0"), "{}", out);
        assert!(dir.path().join("marker.txt").exists());
    }

    #[tokio::test]
    async fn timeout_kills_the_child() {
        if !shell_available() {
            return;
        }
        let tool = BashCommandTool;
        let started = Instant::now();
        let err = tool
            .execute(json!({ "command": "sleep 5", "timeout_secs": 1 }))
            .await
            .expect_err("sleep 5 must not finish within 1 second");

        let message = err.to_string();
        assert!(message.contains("timed out after 1 seconds"), "{}", message);
        assert!(message.contains("killed"), "{}", message);
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "timeout should fire long before the command would finish"
        );
    }

    #[tokio::test]
    async fn timed_out_command_does_not_keep_running() {
        if !shell_available() {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let tool = BashCommandTool;
        tool.execute(json!({
            "command": "sleep 2; printf done > survived.txt",
            "cwd": dir.path().to_string_lossy(),
            "timeout_secs": 1,
        }))
        .await
        .expect_err("command must time out");

        tokio::time::sleep(Duration::from_millis(2500)).await;
        assert!(
            !dir.path().join("survived.txt").exists(),
            "the killed command must not have completed its second statement"
        );
    }

    #[tokio::test]
    async fn missing_cwd_is_a_clear_error() {
        let tool = BashCommandTool;
        let err = tool
            .execute(json!({ "command": "echo hi", "cwd": "/definitely/not/a/real/dir" }))
            .await
            .expect_err("missing cwd must fail");

        let message = err.to_string();
        assert!(
            message.contains("/definitely/not/a/real/dir"),
            "{}",
            message
        );
        assert!(message.contains("Working directory"), "{}", message);
    }

    #[tokio::test]
    async fn cwd_pointing_at_a_file_is_rejected() {
        let file = tempfile::NamedTempFile::new().expect("temp file");
        let tool = BashCommandTool;
        let err = tool
            .execute(json!({
                "command": "echo hi",
                "cwd": file.path().to_string_lossy(),
            }))
            .await
            .expect_err("a file is not a working directory");

        assert!(err.to_string().contains("is not a directory"), "{}", err);
    }

    #[tokio::test]
    async fn large_output_is_truncated_and_reported() {
        if !shell_available() {
            return;
        }
        // Pure shell string doubling keeps this portable: 32 * 2^10 = 32768 bytes.
        let script = "s=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; \
                      for i in 1 2 3 4 5 6 7 8 9 10; do s=\"$s$s\"; done; printf '%s' \"$s\"";
        let tool = BashCommandTool;
        let out = tool
            .execute(json!({ "command": script, "timeout_secs": 20 }))
            .await
            .expect("command should run");

        assert!(out.contains("Exit Code: 0"), "{}", out);
        assert!(out.contains("[output truncated:"), "{}", out);
        assert!(out.contains("32768 bytes"), "{}", out);
        assert!(
            out.len() < MAX_OUTPUT_BYTES + 512,
            "capped output should stay near the budget, got {} bytes",
            out.len()
        );
    }

    #[tokio::test]
    async fn background_returns_immediately_with_a_pid() {
        if !shell_available() {
            return;
        }
        let tool = BashCommandTool;
        let started = Instant::now();
        let out = tool
            .execute(json!({ "command": "sleep 5", "background": true }))
            .await
            .expect("background command should start");

        assert!(
            started.elapsed() < Duration::from_secs(2),
            "background start should not wait for the command"
        );
        assert!(out.contains("pid"), "{}", out);

        let pid = extract_pid(&out);
        assert!(pid > 0);
        assert!(background_job_pids().contains(&pid));

        assert!(kill_background_job(pid).await.expect("kill should succeed"));
        assert!(!background_job_pids().contains(&pid));
        assert!(!kill_background_job(pid)
            .await
            .expect("second kill is a no-op"));
    }

    #[tokio::test]
    async fn rejects_bad_arguments() {
        let tool = BashCommandTool;

        assert!(tool.execute(json!({})).await.is_err());
        assert!(tool.execute(json!({ "command": "   " })).await.is_err());
        assert!(tool
            .execute(json!({ "command": "echo hi", "timeout_secs": 0 }))
            .await
            .is_err());
        assert!(tool
            .execute(json!({ "command": "echo hi", "timeout_secs": -1 }))
            .await
            .is_err());
        assert!(tool
            .execute(json!({ "command": "echo hi", "cwd": 7 }))
            .await
            .is_err());
        assert!(tool
            .execute(json!({ "command": "echo hi", "background": "yes" }))
            .await
            .is_err());
    }

    #[test]
    fn cap_output_leaves_small_output_untouched() {
        let (out, err) = cap_output("hello", "world");
        assert_eq!(out, "hello");
        assert_eq!(err, "world");
    }

    #[test]
    fn cap_output_lets_one_stream_use_the_other_slack() {
        let stdout = "a".repeat(MAX_OUTPUT_BYTES * 2);
        let (out, err) = cap_output(&stdout, "tiny");
        assert_eq!(out.len(), MAX_OUTPUT_BYTES - 4);
        assert_eq!(err, "tiny");
    }

    #[test]
    fn cap_output_splits_when_both_streams_are_large() {
        let stdout = "a".repeat(MAX_OUTPUT_BYTES);
        let stderr = "b".repeat(MAX_OUTPUT_BYTES);
        let (out, err) = cap_output(&stdout, &stderr);
        assert_eq!(out.len() + err.len(), MAX_OUTPUT_BYTES);
    }

    #[test]
    fn cap_output_respects_utf8_boundaries() {
        let stdout = "\u{65e5}".repeat(MAX_OUTPUT_BYTES);
        let (out, err) = cap_output(&stdout, "");
        assert!(out.len() <= MAX_OUTPUT_BYTES);
        assert!(stdout.starts_with(out));
        assert!(err.is_empty());
    }

    #[test]
    fn tool_metadata_declares_approval_and_schema() {
        let tool = BashCommandTool;
        assert_eq!(tool.name(), "run_bash_command");
        assert!(tool.requires_approval());

        let schema = tool.parameters_schema();
        assert_eq!(schema["required"], json!(["command"]));
        for key in ["command", "cwd", "timeout_secs", "background"] {
            assert!(
                schema["properties"][key].is_object(),
                "schema should describe {}",
                key
            );
        }
    }
}
