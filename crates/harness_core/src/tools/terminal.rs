use crate::tool_registry::Tool;
use crate::workspace::WorkspacePolicy;
use agent_core::truncate_at_boundary;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
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
    kill_registered_job(pid, move |mut child| async move {
        // Signal the whole group, not just the shell, so nothing the job started
        // outlives it. The pid is the registry key, which only matches because
        // this process recorded it from a live child of its own, so a caller
        // cannot steer this at an unrelated group. `child.id()` is deliberately
        // not used: it is None once the leader has been reaped, and a job whose
        // leader exited while leaving work behind is exactly the case that still
        // needs the group killed.
        //
        // While the leader is unreaped its pid cannot be recycled, so signalling
        // is unconditionally safe. Once it has been reaped the number is only
        // still ours for as long as the group has members, so probe first rather
        // than firing SIGKILL at whatever now owns that pid. The probe leaves a
        // narrow race that is inherent to addressing a group by pid at all;
        // closing it entirely needs cgroups or job objects.
        let group_killed = if child.id().is_some() || process_group_is_alive(pid) {
            kill_process_group(pid)
        } else {
            false
        };
        let outcome = child.kill().await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to kill background job {}; its process group {} signalled: {}",
                pid,
                if group_killed { "was" } else { "could not be" },
                e
            )
        });
        (child, outcome)
    })
    .await
}

/// Registry bookkeeping around a kill, kept separate from the killing itself so
/// the failure ordering can be exercised.
///
/// The map cannot be held across the await without making the returned future
/// !Send, so the child is taken out first. It goes back if the kill fails: a job
/// is only forgotten once it is actually dead, and dropping it from the registry
/// after a failed kill would leave a running process nothing can reach any more.
async fn kill_registered_job<F, Fut>(pid: u32, kill: F) -> Result<bool>
where
    F: FnOnce(Child) -> Fut,
    Fut: std::future::Future<Output = (Child, Result<()>)>,
{
    let child = background_jobs().remove(&pid);
    let child = match child {
        Some(child) => child,
        None => return Ok(false),
    };

    let (child, outcome) = kill(child).await;
    match outcome {
        Ok(()) => Ok(true),
        Err(e) => {
            background_jobs().insert(pid, child);
            Err(e)
        }
    }
}

/// Puts the child in its own process group. Two things follow from that: a
/// timeout can signal the whole tree instead of only the shell, and a Ctrl-C on
/// the harness's terminal no longer reaches commands the harness started.
#[cfg(unix)]
fn detach_process_group(command: &mut Command) {
    // SAFETY: the closure runs in the forked child between fork and exec, where
    // only async-signal-safe calls are permitted. `setpgid` is async-signal-safe
    // and touches no memory shared with the parent, which is exactly the
    // contract `pre_exec` imposes.
    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) == 0 {
                Ok(())
            } else {
                Err(std::io::Error::last_os_error())
            }
        });
    }
}

#[cfg(not(unix))]
fn detach_process_group(_command: &mut Command) {}

/// Signals every process in the group led by `pid`. The child's pid is its own
/// process-group id because of `detach_process_group`. Returns false when the
/// group is already gone, or on platforms without process groups.
#[cfg(unix)]
fn kill_process_group(pid: u32) -> bool {
    // SAFETY: `killpg` is a plain syscall wrapper taking two integers; it
    // dereferences nothing and cannot violate any Rust invariant.
    unsafe { libc::killpg(pid as i32, libc::SIGKILL) == 0 }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: u32) -> bool {
    false
}

/// True when the group led by `pid` still has members. Signal 0 delivers nothing
/// and performs only the existence and permission check.
#[cfg(unix)]
fn process_group_is_alive(pid: u32) -> bool {
    // SAFETY: as `kill_process_group` — two integers, nothing dereferenced.
    unsafe { libc::killpg(pid as i32, 0) == 0 }
}

#[cfg(not(unix))]
fn process_group_is_alive(_pid: u32) -> bool {
    false
}

/// Kills the child's whole process group and then reaps the child. `killpg` does
/// not wait, so without the reap the child stays a zombie holding its pid.
///
/// `group_pid` is recorded from the live `Child` at spawn, never supplied by a
/// caller. Using the recorded number instead of `child.id()` is what makes this
/// correct after the child has been waited on, which is precisely when a
/// surviving grandchild still needs the group signalled. It stays safe to reuse:
/// the kernel does not recycle a pid while that number is still a live
/// process-group id, so this can only reach the group we started, or nothing.
async fn kill_group_and_reap(child: &mut Child, group_pid: Option<u32>) -> bool {
    let group_killed = group_pid.map(kill_process_group).unwrap_or(false);
    let reaped = child.kill().await.is_ok();
    group_killed || reaped
}

/// Pids of the background jobs this process started, in unspecified order.
/// Finished jobs are reaped first, so this reflects what is still running.
pub fn background_job_pids() -> Vec<u32> {
    let mut jobs = background_jobs();
    reap_finished(&mut jobs);
    jobs.keys().copied().collect()
}

/// Drops jobs that are completely finished. Without this the map grows without
/// bound and every finished job stays a zombie holding its pid, because nothing
/// ever waits on a background `Child`.
///
/// "Finished" means the whole group is gone, not just the leader. Judging by the
/// leader alone silently forgets a job that is still running: `run_bash_command`
/// spawns `sh -c <command>`, and a command ending in `&` makes that shell exit
/// immediately while the real work continues in the same group. Dropping the
/// entry there is the same failure the kill-ordering fix exists to prevent — a
/// live process that nothing can reach or kill any more.
fn reap_finished(jobs: &mut HashMap<u32, Child>) {
    jobs.retain(|pid, child| {
        if !matches!(child.try_wait(), Ok(Some(_))) {
            return true;
        }
        // The leader has been reaped, so its pid is free again — but a pid that
        // is still a live process-group id is not handed out to a new process,
        // so this probe cannot land on an unrelated group. Once the last member
        // exits the probe fails and the entry finally goes.
        process_group_is_alive(*pid)
    });
}

pub struct BashCommandTool {
    policy: Arc<WorkspacePolicy>,
}

impl BashCommandTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

impl Default for BashCommandTool {
    fn default() -> Self {
        Self::new(Arc::new(WorkspacePolicy::unrestricted()))
    }
}

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

        let requested_cwd = match &args["cwd"] {
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

        // This confines where the command *starts*, not what it can reach: the
        // shell can still read anything the user can. That is what requiring
        // approval on this tool is for, and it is the intended model.
        let cwd = self.policy.resolve(requested_cwd).map_err(|e| {
            anyhow::anyhow!(
                "Working directory '{}' is not allowed: {}",
                requested_cwd,
                e
            )
        })?;

        // A bad cwd otherwise surfaces as an opaque OS error from spawn().
        let metadata = tokio::fs::metadata(&cwd).await.map_err(|e| {
            anyhow::anyhow!("Working directory '{}' is unusable: {}", cwd.display(), e)
        })?;
        if !metadata.is_dir() {
            anyhow::bail!("Working directory '{}' is not a directory", cwd.display());
        }

        if background {
            run_background(command_str, &cwd)
        } else {
            run_foreground(command_str, &cwd, timeout_secs).await
        }
    }
}

fn run_background(command_str: &str, cwd: &Path) -> Result<String> {
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(command_str)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .kill_on_drop(false);
    detach_process_group(&mut command);

    let child = command
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to start background command: {}", e))?;

    let pid = match child.id() {
        Some(pid) => pid,
        None => anyhow::bail!("Background command exited before a pid could be observed"),
    };

    {
        let mut jobs = background_jobs();
        reap_finished(&mut jobs);
        jobs.insert(pid, child);
    }

    Ok(format!(
        "Started background process with pid {}. Its output is not captured; redirect it to a file in the command if you need it, and check on the process with `ps -p {}`.\n",
        pid, pid
    ))
}

async fn run_foreground(command_str: &str, cwd: &Path, timeout_secs: u64) -> Result<String> {
    let mut command = Command::new("sh");
    command
        .arg("-c")
        .arg(command_str)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    detach_process_group(&mut command);

    let mut child = command
        .spawn()
        .map_err(|e| anyhow::anyhow!("Failed to start command: {}", e))?;

    // Recorded from the live child, and recorded now: `child.id()` starts
    // returning None the moment the child has been waited on, and the timeout
    // path still needs a group to signal after that point.
    let group_pid = child.id();

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
    // pipe buffer blocks forever instead of exiting. Draining is bounded: the pipes
    // are read to completion but only the first MAX_OUTPUT_BYTES of each are kept,
    // so `yes` (multiple GB/s through a pipe) cannot exhaust memory before the
    // timeout fires.
    //
    // The wait is sequenced *after* the drains rather than joined with them. A
    // completed `wait()` fuses the `Child`, which makes `child.id()` return None
    // and leaves the timeout path with no process-group id to signal. That is not
    // hypothetical: `sh -c '... &'` exits immediately while the grandchild it
    // started keeps the inherited pipe open, so the wait would win the join every
    // time and the group kill would silently become a no-op. Completion already
    // requires both drains to hit EOF, so waiting afterwards costs nothing.
    let collect = async {
        let (stdout_total, stderr_total) = tokio::try_join!(
            drain_capped(&mut stdout_pipe, &mut stdout_buf, MAX_OUTPUT_BYTES),
            drain_capped(&mut stderr_pipe, &mut stderr_buf, MAX_OUTPUT_BYTES),
        )?;
        let status = child.wait().await?;
        Ok::<_, std::io::Error>((stdout_total, stderr_total, status))
    };

    let outcome = timeout(Duration::from_secs(timeout_secs), collect).await;

    let (stdout_total, stderr_total, status) = match outcome {
        Ok(res) => res?,
        Err(_) => {
            // kill_on_drop would reap the shell, but only the shell: anything it
            // started would survive as an orphan. Signalling the group is what
            // makes the guarantee in the error message true.
            let killed = kill_group_and_reap(&mut child, group_pid).await;
            anyhow::bail!(
                "Command timed out after {} seconds; the process group was {}.",
                timeout_secs,
                if killed { "killed" } else { "already gone" }
            );
        }
    };

    let stdout = String::from_utf8_lossy(&stdout_buf);
    let stderr = String::from_utf8_lossy(&stderr_buf);
    // The totals come from the pipes, not from the retained buffers, so the report
    // stays honest about how much was thrown away.
    Ok(format_output(
        status.code().unwrap_or(-1),
        &stdout,
        &stderr,
        stdout_total.saturating_add(stderr_total),
    ))
}

/// Reads `reader` to EOF so the child never blocks on a full pipe, but retains at
/// most `cap` bytes in `buf`. Returns the total number of bytes the stream
/// produced, which may be far larger than what was retained.
async fn drain_capped<R: AsyncRead + Unpin>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    cap: usize,
) -> std::io::Result<usize> {
    let mut chunk = [0u8; 8192];
    let mut total: usize = 0;
    loop {
        let n = reader.read(&mut chunk).await?;
        if n == 0 {
            return Ok(total);
        }
        total = total.saturating_add(n);
        if buf.len() < cap {
            let take = std::cmp::min(n, cap - buf.len());
            buf.extend_from_slice(&chunk[..take]);
        }
    }
}

fn format_output(exit_code: i32, stdout: &str, stderr: &str, original_len: usize) -> String {
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
        let tool = BashCommandTool::default();
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
        let tool = BashCommandTool::default();
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
        let tool = BashCommandTool::default();
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
        let tool = BashCommandTool::default();
        let started = Instant::now();
        let err = tool
            .execute(json!({ "command": "sleep 5", "timeout_secs": 1 }))
            .await
            .expect_err("sleep 5 must not finish within 1 second");

        let message = err.to_string();
        assert!(message.contains("timed out after 1 seconds"), "{}", message);
        assert!(
            message.contains("the process group was killed"),
            "{}",
            message
        );
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
        let tool = BashCommandTool::default();
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
    async fn timeout_kills_the_whole_process_group() {
        if !shell_available() {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let tool = BashCommandTool::default();
        // The inner shell is a grandchild of the harness. Killing only the direct
        // child leaves it running and it writes the file a moment later.
        tool.execute(json!({
            "command": "sh -c 'sleep 2; printf leaked > leaked.txt' & sleep 2",
            "cwd": dir.path().to_string_lossy(),
            "timeout_secs": 1,
        }))
        .await
        .expect_err("command must time out");

        tokio::time::sleep(Duration::from_millis(2500)).await;
        assert!(
            !dir.path().join("leaked.txt").exists(),
            "a grandchild of the timed-out command outlived the process-group kill"
        );
    }

    /// The shell exits at once and the grandchild it started inherits the stdout
    /// pipe. If the child is waited on concurrently with the drains, the wait
    /// wins, the `Child` fuses, `child.id()` becomes None and the group kill
    /// degrades to a no-op while the error still claims the group was killed.
    #[tokio::test]
    async fn timeout_kills_the_group_even_when_the_shell_exits_first() {
        if !shell_available() {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let tool = BashCommandTool::default();
        let err = tool
            .execute(json!({
                "command": "sh -c 'sleep 2; printf leaked > leaked.txt' &",
                "cwd": dir.path().to_string_lossy(),
                "timeout_secs": 1,
            }))
            .await
            .expect_err("the held-open pipe must keep the command from completing");
        assert!(
            err.to_string().contains("the process group was killed"),
            "{}",
            err
        );

        tokio::time::sleep(Duration::from_millis(2500)).await;
        assert!(
            !dir.path().join("leaked.txt").exists(),
            "a grandchild survived a timeout whose error claimed the group was killed"
        );
    }

    /// A background job whose leader exits but which left work behind must stay
    /// killable. Deciding liveness from the leader alone forgets it while it runs.
    #[tokio::test]
    async fn a_background_job_outliving_its_shell_stays_killable() {
        if !shell_available() {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let tool = BashCommandTool::default();
        let report = tool
            .execute(json!({
                "command": "sh -c 'sleep 3; printf orphan > orphan.txt' &",
                "cwd": dir.path().to_string_lossy(),
                "background": true,
            }))
            .await
            .expect("background command should start");
        let pid = extract_pid(&report);

        // Long enough for the leading shell to have exited.
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(
            background_job_pids().contains(&pid),
            "a job whose group is still running must not be reaped away"
        );
        assert!(
            kill_background_job(pid).await.expect("kill should succeed"),
            "the job must still be killable after its leader exited"
        );

        tokio::time::sleep(Duration::from_millis(3000)).await;
        assert!(
            !dir.path().join("orphan.txt").exists(),
            "the surviving member of the job's group was never killed"
        );
        assert!(!background_job_pids().contains(&pid));
    }

    #[tokio::test]
    async fn cwd_outside_the_workspace_is_refused() {
        let root = tempfile::tempdir().expect("root");
        let outside = tempfile::tempdir().expect("outside");
        let policy = Arc::new(WorkspacePolicy::with_roots([root.path()]).expect("policy"));
        let tool = BashCommandTool::new(policy);

        let err = tool
            .execute(json!({
                "command": "printf escaped > escaped.txt",
                "cwd": outside.path().to_string_lossy(),
            }))
            .await
            .expect_err("a cwd outside the roots must be refused");

        let message = err.to_string();
        assert!(message.contains("Working directory"), "{}", message);
        assert!(
            message.contains("outside the allowed workspace"),
            "{}",
            message
        );
        assert!(!outside.path().join("escaped.txt").exists());
    }

    #[tokio::test]
    async fn cwd_inside_the_workspace_is_allowed() {
        if !shell_available() {
            return;
        }
        let root = tempfile::tempdir().expect("root");
        std::fs::create_dir_all(root.path().join("sub")).expect("mkdir");
        let policy = Arc::new(WorkspacePolicy::with_roots([root.path()]).expect("policy"));
        let tool = BashCommandTool::new(policy);

        let out = tool
            .execute(json!({
                "command": "printf marker > marker.txt",
                "cwd": root.path().join("sub").to_string_lossy(),
            }))
            .await
            .expect("a cwd inside the root should run");

        assert!(out.contains("Exit Code: 0"), "{}", out);
        assert!(root.path().join("sub/marker.txt").exists());
    }

    #[tokio::test]
    async fn kill_background_job_is_false_for_an_unknown_pid() {
        assert!(!kill_background_job(0)
            .await
            .expect("unknown pid is not an error"));
        assert!(!kill_background_job(u32::MAX)
            .await
            .expect("unknown pid is not an error"));
    }

    #[tokio::test]
    async fn a_failed_kill_keeps_the_job_in_the_registry() {
        if !shell_available() {
            return;
        }
        let tool = BashCommandTool::default();
        let report = tool
            .execute(json!({ "command": "sleep 5", "background": true }))
            .await
            .expect("background command should start");
        let pid = extract_pid(&report);

        let err = kill_registered_job(pid, |child| async move {
            (child, Err(anyhow::anyhow!("signal refused")))
        })
        .await
        .expect_err("a failed kill must surface as an error");
        assert!(err.to_string().contains("signal refused"), "{}", err);
        assert!(
            background_job_pids().contains(&pid),
            "a job whose kill failed must stay reachable instead of being forgotten while it runs"
        );

        assert!(kill_background_job(pid).await.expect("kill should succeed"));
        assert!(!background_job_pids().contains(&pid));
    }

    #[tokio::test]
    async fn a_successful_kill_forgets_the_job() {
        if !shell_available() {
            return;
        }
        let tool = BashCommandTool::default();
        let report = tool
            .execute(json!({ "command": "sleep 5", "background": true }))
            .await
            .expect("background command should start");
        let pid = extract_pid(&report);

        assert!(
            kill_registered_job(pid, |child| async move { (child, Ok(())) })
                .await
                .expect("a successful kill is not an error")
        );
        assert!(!background_job_pids().contains(&pid));

        // The stub did not actually kill it, so clean up for real.
        kill_process_group(pid);
    }

    #[tokio::test]
    async fn missing_cwd_is_a_clear_error() {
        let tool = BashCommandTool::default();
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
        let tool = BashCommandTool::default();
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
        let tool = BashCommandTool::default();
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
        let tool = BashCommandTool::default();
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

    #[cfg(unix)]
    #[tokio::test]
    async fn background_job_leads_its_own_process_group() {
        if !shell_available() {
            return;
        }
        let tool = BashCommandTool::default();
        let out = tool
            .execute(json!({ "command": "sleep 5", "background": true }))
            .await
            .expect("background command should start");
        let pid = extract_pid(&out);

        // pre_exec runs in the forked child, so give it a moment to land.
        tokio::time::sleep(Duration::from_millis(200)).await;
        // SAFETY: getpgid only reads kernel state for the given pid.
        let (job_group, our_group) = unsafe { (libc::getpgid(pid as i32), libc::getpgid(0)) };

        assert_eq!(
            job_group, pid as i32,
            "the job must lead its own process group"
        );
        assert_ne!(
            job_group, our_group,
            "a Ctrl-C on the harness's group must not reach background jobs"
        );

        assert!(kill_background_job(pid).await.expect("kill should succeed"));
    }

    #[tokio::test]
    async fn rejects_bad_arguments() {
        let tool = BashCommandTool::default();

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

    #[tokio::test]
    async fn drain_capped_retains_only_the_cap_but_reports_the_whole_stream() {
        // 1 MiB of input through a 64-byte cap: a read_to_end-style implementation
        // would retain all 1 MiB, which is exactly the runaway-memory failure mode.
        let source = vec![b'x'; 1024 * 1024];
        let mut reader = &source[..];
        let mut buf = Vec::new();
        let total = drain_capped(&mut reader, &mut buf, 64)
            .await
            .expect("draining a slice cannot fail");

        assert_eq!(total, source.len());
        assert_eq!(buf.len(), 64);
    }

    #[tokio::test]
    async fn runaway_output_does_not_get_buffered_whole() {
        if !shell_available() {
            return;
        }
        // 4 MiB emitted for real through the pipe; only the cap may survive.
        let script = "s=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; \
                      for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17; \
                      do s=\"$s$s\"; done; printf '%s' \"$s\"";
        let tool = BashCommandTool::default();
        let out = tool
            .execute(json!({ "command": script, "timeout_secs": 30 }))
            .await
            .expect("command should run");

        assert!(out.contains("[output truncated:"), "{}", out);
        assert!(out.contains("4194304 bytes"), "{}", out);
        assert!(
            out.len() < MAX_OUTPUT_BYTES + 512,
            "capped output should stay near the budget, got {} bytes",
            out.len()
        );
    }

    #[test]
    fn kill_background_job_future_is_send() {
        fn assert_send<T: Send>(_: T) {}
        // Holding the BACKGROUND_JOBS std::sync guard across the `.await` would make
        // this future !Send and risk deadlocking a multi-threaded runtime; the guard
        // must stay a statement-scoped temporary.
        assert_send(kill_background_job(0));
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
        let tool = BashCommandTool::default();
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
