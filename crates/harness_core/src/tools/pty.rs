//! Interactive terminal sessions.
//!
//! `run_bash_command` runs a command to completion with no stdin, so it cannot
//! answer anything that asks a question: a REPL, an installer, `git rebase -i`,
//! a plain `are you sure? [y/N]`. These tools keep a program alive on a pty
//! between tool calls, so the agent can read a prompt and then reply to it.

use crate::tool_registry::Tool;
use crate::workspace::WorkspacePolicy;
use agent_core::truncate_at_boundary;
use anyhow::Result;
use async_trait::async_trait;
use portable_pty::{native_pty_system, Child, CommandBuilder, ExitStatus, MasterPty, PtySize};
use serde_json::json;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Bytes retained per session between reads. The most recent bytes are kept
/// rather than the oldest: a prompt lives at the end of the stream.
const MAX_BUFFER_BYTES: usize = 256 * 1024;

/// Budget for a single tool result, applied after the buffer cap.
const MAX_REPORT_BYTES: usize = 16_000;

/// Cap on what one wait accumulates across its polls. The per-session buffer
/// bounds what is retained *between* calls, but a wait drains that buffer every
/// `POLL_INTERVAL_MS` and concatenates the pieces, so without a second cap here
/// a program that prints without pause makes a single `pty_send` allocate at the
/// rate of the pty for the whole of `wait_ms` — hundreds of megabytes at the
/// maximum wait, of which `render` then shows the last 16 KB.
const MAX_COLLECT_BYTES: usize = MAX_BUFFER_BYTES;

/// Open sessions allowed at once. Each holds a pty pair, a child and a thread,
/// so an agent looping on `pty_start` would otherwise run the process out of
/// file descriptors.
const MAX_SESSIONS: usize = 8;

const MAX_INPUT_BYTES: usize = 64 * 1024;

const DEFAULT_COLS: u16 = 120;
const DEFAULT_ROWS: u16 = 30;

const START_SETTLE_MS: u64 = 400;
const DEFAULT_WAIT_MS: u64 = 500;
const MAX_WAIT_MS: u64 = 30_000;
const POLL_INTERVAL_MS: u64 = 20;

/// How long output must stay quiet before a wait returns early. Without this
/// every turn of an interactive exchange pays the full `wait_ms`.
const IDLE_QUIET_MS: u64 = 120;

/// A write only blocks when the program has stopped draining its terminal input.
/// Bounding it keeps `pty_send` from hanging on a wedged child.
const WRITE_TIMEOUT_MS: u64 = 5_000;

const READER_JOIN_ATTEMPTS: u32 = 20;
const READER_JOIN_POLL_MS: u64 = 25;

const REAP_ATTEMPTS: u32 = 40;
const REAP_POLL_MS: u64 = 50;

/// Bytes read off the pty that no `pty_send` has taken yet.
///
/// Contents are bytes, not text: a read can land in the middle of a multi-byte
/// character, and so can the drop that enforces the cap. Decoding is deferred to
/// `take`, which hands back only complete characters and keeps a partial tail
/// for the next read.
#[derive(Default)]
struct OutputBuffer {
    bytes: Vec<u8>,
    dropped: usize,
    eof: bool,
    /// Set on close so a reader thread that outlives its join stops retaining.
    stopped: bool,
}

impl OutputBuffer {
    fn push(&mut self, chunk: &[u8]) {
        self.bytes.extend_from_slice(chunk);
        if self.bytes.len() > MAX_BUFFER_BYTES {
            let excess = self.bytes.len() - MAX_BUFFER_BYTES;
            self.bytes.drain(..excess);
            self.dropped = self.dropped.saturating_add(excess);
        }
    }

    /// Removes and decodes everything that can be decoded, returning the text and
    /// the number of bytes dropped since the last call.
    fn take(&mut self) -> (String, usize) {
        // A trailing partial sequence is still going to be completed by the next
        // read, so it stays in the buffer; at EOF nothing more is coming and it
        // is decoded lossily instead.
        let held_back = if self.eof {
            0
        } else {
            incomplete_tail_len(&self.bytes)
        };
        let split = self.bytes.len() - held_back;
        let text = String::from_utf8_lossy(&self.bytes[..split]).into_owned();
        self.bytes.drain(..split);
        (text, std::mem::take(&mut self.dropped))
    }

    fn has_unread(&self) -> bool {
        !self.bytes.is_empty() || self.dropped > 0
    }
}

/// Length of an unfinished UTF-8 sequence at the end of `bytes`, or 0. A
/// sequence is at most four bytes, so this looks back at most three.
fn incomplete_tail_len(bytes: &[u8]) -> usize {
    let max_back = std::cmp::min(3, bytes.len());
    for back in 1..=max_back {
        let byte = bytes[bytes.len() - back];
        if byte < 0x80 {
            return 0;
        }
        if byte >= 0xC0 {
            let needed = if byte >= 0xF0 {
                4
            } else if byte >= 0xE0 {
                3
            } else {
                2
            };
            return if needed > back { back } else { 0 };
        }
    }
    0
}

/// The parts of a session that outlive the registry lock. Every field is
/// individually lockable so a tool call can take what it needs, drop the
/// registry guard, and only then await.
struct SessionShared {
    output: Arc<Mutex<OutputBuffer>>,
    writer: Mutex<Box<dyn Write + Send>>,
    child: Mutex<Box<dyn Child + Send + Sync>>,
    /// A write that timed out leaves its blocking thread holding `writer`. This
    /// makes the next send say so instead of blocking on the lock in turn.
    write_in_flight: AtomicBool,
    /// Recorded at spawn. The child is a session leader, so this is also its
    /// process-group id, which is what a kill has to reach: killing the leader
    /// alone would leave anything it started attached to the pty.
    pid: Option<u32>,
    command: String,
}

struct PtySession {
    shared: Arc<SessionShared>,
    /// Held only so the pty stays open for the life of the session and is
    /// released when it is dropped.
    #[allow(dead_code)]
    master: Box<dyn MasterPty + Send>,
    reader: Option<JoinHandle<()>>,
}

static SESSIONS: OnceLock<Mutex<HashMap<String, PtySession>>> = OnceLock::new();

fn sessions() -> MutexGuard<'static, HashMap<String, PtySession>> {
    // A poisoned lock only means another caller panicked mid-update; the map
    // itself is still valid, so recovering beats propagating the panic.
    SESSIONS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn lock_output(output: &Mutex<OutputBuffer>) -> MutexGuard<'_, OutputBuffer> {
    output
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn lock_child(
    child: &Mutex<Box<dyn Child + Send + Sync>>,
) -> MutexGuard<'_, Box<dyn Child + Send + Sync>> {
    child
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

/// The counter makes ids unique within the process; the clock and pid make them
/// impractical to guess, so a session can only be addressed by an agent that was
/// handed its id. Ids are otherwise opaque.
fn new_session_id() -> String {
    let counter = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let salt = nanos.rotate_left(23).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ ((std::process::id() as u64) << 32)
        ^ counter;
    format!("pty-{:x}-{:08x}", counter, salt as u32)
}

/// Signals every process in the group led by `pid`. The child calls `setsid`
/// before exec, so its pid is its own process-group id. Returns false when the
/// group is already gone, or on platforms without process groups.
#[cfg(unix)]
fn kill_process_group(pid: u32) -> bool {
    // SAFETY: `killpg` is a plain syscall wrapper over two integers; it
    // dereferences nothing and cannot violate any Rust invariant. The pid was
    // recorded from a live child of this process, so a caller cannot steer this
    // at an unrelated group.
    unsafe { libc::killpg(pid as i32, libc::SIGKILL) == 0 }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: u32) -> bool {
    false
}

/// The exit status once the child has one. `try_wait` caches it, so this stays
/// correct after the child has been reaped and keeps a finished session from
/// being reported as running.
fn poll_exit(shared: &SessionShared) -> Option<ExitStatus> {
    lock_child(&shared.child).try_wait().unwrap_or_default()
}

fn status_line(shared: &SessionShared) -> String {
    match poll_exit(shared) {
        Some(status) => format!("exited ({})", status),
        None => match shared.pid {
            Some(pid) => format!("running (pid {})", pid),
            None => "running".to_string(),
        },
    }
}

/// Kills the process group and reaps the child, so a closed session leaves
/// neither a running program nor a zombie behind. `pid` is the group id recorded
/// at spawn.
fn terminate(child: &mut Box<dyn Child + Send + Sync>, pid: Option<u32>) -> String {
    if let Ok(Some(status)) = child.try_wait() {
        return format!("already exited ({})", status);
    }

    // The group, not just the leader: a program that spawned helpers sharing the
    // pty would otherwise keep the terminal alive after the session is gone.
    let group_killed = pid.map(kill_process_group).unwrap_or(false);
    if !group_killed {
        let _ = child.kill();
    }

    for attempt in 0..REAP_ATTEMPTS {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(REAP_POLL_MS));
        }
        if let Ok(Some(status)) = child.try_wait() {
            return format!("killed ({})", status);
        }
    }
    "kill signalled, but the process has not been reaped".to_string()
}

/// Waits for the reader thread to end, which it does once the child is gone and
/// the last handle on the slave side is closed. Bounded: a thread that somehow
/// outlives its process is detached rather than blocking the close forever.
fn join_reader(reader: Option<JoinHandle<()>>) -> bool {
    let handle = match reader {
        Some(handle) => handle,
        None => return true,
    };
    for attempt in 0..READER_JOIN_ATTEMPTS {
        if handle.is_finished() {
            return handle.join().is_ok();
        }
        if attempt + 1 < READER_JOIN_ATTEMPTS {
            std::thread::sleep(Duration::from_millis(READER_JOIN_POLL_MS));
        }
    }
    false
}

/// Reads the pty on its own thread. `portable-pty`'s reader is blocking, and a
/// program that never prints anything would otherwise wedge whatever polled it.
fn spawn_reader(
    mut reader: Box<dyn Read + Send>,
    output: Arc<Mutex<OutputBuffer>>,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut chunk = [0u8; 8192];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    let mut buffer = lock_output(&output);
                    if buffer.stopped {
                        break;
                    }
                    buffer.push(&chunk[..n]);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                // A pty whose slave side has closed reports EIO rather than EOF.
                Err(_) => break,
            }
        }
        lock_output(&output).eof = true;
    })
}

/// Drops sessions that have exited and have nothing left to report. Anything
/// still unread is kept: the agent has not seen it yet, and losing a program's
/// last words to a background sweep would be worse than holding the slot.
fn reap_drained(map: &mut HashMap<String, PtySession>) {
    map.retain(|_, session| {
        if poll_exit(&session.shared).is_none() {
            return true;
        }
        let buffer = lock_output(&session.shared.output);
        buffer.has_unread() || !buffer.eof
    });
}

fn capacity_error(map: &HashMap<String, PtySession>) -> anyhow::Error {
    let open = map
        .iter()
        .map(|(id, session)| {
            format!(
                "{} ({})",
                id,
                truncate_at_boundary(&session.shared.command, 48)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    anyhow::anyhow!(
        "At most {} interactive sessions may be open at once. Close one with pty_close first. Open sessions: {}",
        MAX_SESSIONS,
        open
    )
}

struct StartedSession {
    session: PtySession,
    shared: Arc<SessionShared>,
}

fn spawn_session(command: &str, cwd: &Path, cols: u16, rows: u16) -> Result<StartedSession> {
    let pair = native_pty_system()
        .openpty(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| anyhow::anyhow!("Failed to open a pty: {}", e))?;

    let mut builder = CommandBuilder::new("sh");
    builder.arg("-c");
    builder.arg(command);
    builder.cwd(cwd);
    // Curses programs abort outright without a terminal type, and the harness's
    // own environment may not have one when it runs detached.
    if builder.get_env("TERM").is_none() {
        builder.env("TERM", "xterm-256color");
    }

    let mut child = pair
        .slave
        .spawn_command(builder)
        .map_err(|e| anyhow::anyhow!("Failed to start '{}' on a pty: {}", command, e))?;
    // The slave handle has to go now: while it is open the kernel keeps the
    // terminal alive, and the reader would never see EOF after the child exits.
    drop(pair.slave);

    let pid = child.process_id();

    // The program is already running, so from here on a failure has to take it
    // back down: nothing else has a handle on it yet.
    let reader = match pair.master.try_clone_reader() {
        Ok(reader) => reader,
        Err(e) => {
            terminate(&mut child, pid);
            anyhow::bail!("Failed to read from the pty: {}", e);
        }
    };
    let writer = match pair.master.take_writer() {
        Ok(writer) => writer,
        Err(e) => {
            terminate(&mut child, pid);
            anyhow::bail!("Failed to write to the pty: {}", e);
        }
    };

    let output = Arc::new(Mutex::new(OutputBuffer::default()));
    let handle = spawn_reader(reader, Arc::clone(&output));

    let shared = Arc::new(SessionShared {
        output,
        writer: Mutex::new(writer),
        child: Mutex::new(child),
        write_in_flight: AtomicBool::new(false),
        pid,
        command: command.to_string(),
    });

    Ok(StartedSession {
        session: PtySession {
            shared: Arc::clone(&shared),
            master: pair.master,
            reader: Some(handle),
        },
        shared,
    })
}

/// Kills a session and releases everything it holds. The blocking parts — the
/// reap and the thread join — run off the runtime.
async fn shutdown(mut session: PtySession) -> (String, bool) {
    let reader = session.reader.take();
    let outcome = tokio::task::spawn_blocking(move || {
        let killed = {
            let mut child = lock_child(&session.shared.child);
            terminate(&mut child, session.shared.pid)
        };
        let joined = join_reader(reader);
        // Dropping the session here releases the master pty, the writer and the
        // child handle on the same thread that just reaped them.
        drop(session);
        (killed, joined)
    })
    .await;

    outcome.unwrap_or_else(|e| (format!("kill outcome unknown: {}", e), false))
}

/// Collects output for up to `wait_ms`, returning early once the program has
/// answered and gone quiet, or once it is gone for good.
async fn collect_output(shared: &SessionShared, wait_ms: u64) -> (String, usize) {
    let deadline = Instant::now() + Duration::from_millis(wait_ms);
    let mut text = String::new();
    let mut dropped = 0usize;
    let mut last_growth = Instant::now();

    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let step = std::cmp::min(remaining, Duration::from_millis(POLL_INTERVAL_MS));
        if !step.is_zero() {
            tokio::time::sleep(step).await;
        }

        // The guard is scoped to this block: it must not be alive at the await
        // above or the next one, which would make this future !Send and risk
        // deadlocking the runtime.
        let (chunk, chunk_dropped, eof) = {
            let mut buffer = lock_output(&shared.output);
            let (chunk, chunk_dropped) = buffer.take();
            (chunk, chunk_dropped, buffer.eof)
        };
        if !chunk.is_empty() || chunk_dropped > 0 {
            last_growth = Instant::now();
            text.push_str(&chunk);
            dropped = dropped.saturating_add(chunk_dropped);
            // Keep only the most recent bytes, as the session buffer does: the
            // end is where the prompt is, and everything before it is about to
            // be thrown away by `render` anyway. Counting the discard into
            // `dropped` keeps the reported total honest.
            if text.len() > MAX_COLLECT_BYTES {
                let cut = text.len() - tail_at_boundary(&text, MAX_COLLECT_BYTES).len();
                text.drain(..cut);
                dropped = dropped.saturating_add(cut);
            }
        }

        // EOF means the terminal has no writers left, so nothing more can arrive.
        if eof {
            break;
        }
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        if !text.is_empty()
            && now.duration_since(last_growth) >= Duration::from_millis(IDLE_QUIET_MS)
        {
            break;
        }
    }

    (text, dropped)
}

async fn write_input(shared: &Arc<SessionShared>, input: &str, submit: bool) -> Result<()> {
    let mut bytes = input.as_bytes().to_vec();
    if submit {
        bytes.push(b'\n');
    }
    if bytes.len() > MAX_INPUT_BYTES {
        anyhow::bail!(
            "'input' is {} bytes; at most {} may be sent at once",
            bytes.len(),
            MAX_INPUT_BYTES
        );
    }

    if shared.write_in_flight.swap(true, Ordering::SeqCst) {
        anyhow::bail!(
            "An earlier write to this session is still blocked because the program is not reading its input. Close the session with pty_close."
        );
    }

    let target = Arc::clone(shared);
    let write = tokio::task::spawn_blocking(move || {
        let outcome = (|| -> std::io::Result<()> {
            let mut writer = target
                .writer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            writer.write_all(&bytes)?;
            writer.flush()
        })();
        target.write_in_flight.store(false, Ordering::SeqCst);
        outcome
    });

    // A blocked write keeps its thread and the flag, so the timeout ends this
    // call without leaving the next one to block on the writer lock instead.
    match tokio::time::timeout(Duration::from_millis(WRITE_TIMEOUT_MS), write).await {
        Ok(Ok(Ok(()))) => Ok(()),
        Ok(Ok(Err(e))) => anyhow::bail!("Failed to send input to the session: {}", e),
        Ok(Err(e)) => anyhow::bail!("The write task failed: {}", e),
        Err(_) => anyhow::bail!(
            "Sending input blocked for more than {} ms; the program is not consuming its input.",
            WRITE_TIMEOUT_MS
        ),
    }
}

/// Keeps the most recent `max_bytes` of `text`, split on a character boundary.
/// The tail is the useful end: that is where the prompt is.
fn tail_at_boundary(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut start = text.len() - max_bytes;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

fn render(session_id: &str, status: &str, text: &str, dropped: usize) -> String {
    let shown = tail_at_boundary(text, MAX_REPORT_BYTES);
    let lost = dropped.saturating_add(text.len() - shown.len());

    let mut report = format!("Session: {}\nStatus: {}\n", session_id, status);
    if shown.is_empty() {
        report.push_str("OUTPUT: (nothing new)\n");
    } else {
        report.push_str("OUTPUT:\n");
        report.push_str(shown);
        if !shown.ends_with('\n') {
            report.push('\n');
        }
    }
    if lost > 0 {
        report.push_str(&format!(
            "[output truncated: {} earlier bytes dropped, showing the most recent {} bytes]\n",
            lost,
            shown.len()
        ));
    }
    report
}

fn read_dimension(value: &serde_json::Value, default: u16, name: &str) -> Result<u16> {
    match value {
        serde_json::Value::Null => Ok(default),
        value => {
            let size = value
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("'{}' must be a positive integer", name))?;
            if size == 0 || size > u16::MAX as u64 {
                anyhow::bail!("'{}' must be between 1 and {}", name, u16::MAX);
            }
            Ok(size as u16)
        }
    }
}

fn read_session_id(args: &serde_json::Value) -> Result<&str> {
    let id = args["session_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing 'session_id' argument"))?;
    if id.trim().is_empty() {
        anyhow::bail!("'session_id' must not be empty");
    }
    Ok(id)
}

fn unknown_session(session_id: &str) -> anyhow::Error {
    // No sweep here: a session that has exited is still open and still
    // closable, so it belongs in this listing.
    let open = {
        let map = sessions();
        map.keys().cloned().collect::<Vec<_>>().join(", ")
    };
    if open.is_empty() {
        anyhow::anyhow!(
            "No interactive session '{}'. There are no open sessions; start one with pty_start.",
            session_id
        )
    } else {
        anyhow::anyhow!(
            "No interactive session '{}'. Open sessions: {}",
            session_id,
            open
        )
    }
}

pub struct PtyStartTool {
    policy: Arc<WorkspacePolicy>,
}

impl PtyStartTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

impl Default for PtyStartTool {
    fn default() -> Self {
        Self::new(Arc::new(WorkspacePolicy::unrestricted()))
    }
}

#[async_trait]
impl Tool for PtyStartTool {
    fn name(&self) -> &str {
        "pty_start"
    }

    fn description(&self) -> &str {
        "Starts a program on an interactive terminal and returns a session id plus whatever it printed first, so prompts can be answered with pty_send."
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
                    "description": "Shell command to run on the terminal."
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (optional, default: current directory)."
                },
                "cols": {
                    "type": "integer",
                    "description": "Terminal width in columns (optional, default: 120)."
                },
                "rows": {
                    "type": "integer",
                    "description": "Terminal height in rows (optional, default: 30)."
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'command' argument"))?;
        if command.trim().is_empty() {
            anyhow::bail!("'command' must not be empty");
        }

        let requested_cwd = match &args["cwd"] {
            serde_json::Value::Null => ".",
            value => value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("'cwd' must be a string"))?,
        };
        let cols = read_dimension(&args["cols"], DEFAULT_COLS, "cols")?;
        let rows = read_dimension(&args["rows"], DEFAULT_ROWS, "rows")?;

        // As with run_bash_command this confines where the program *starts*, not
        // what it can reach once running; requiring approval is what covers that.
        let cwd = self.policy.resolve(requested_cwd).map_err(|e| {
            anyhow::anyhow!(
                "Working directory '{}' is not allowed: {}",
                requested_cwd,
                e
            )
        })?;

        let metadata = tokio::fs::metadata(&cwd).await.map_err(|e| {
            anyhow::anyhow!("Working directory '{}' is unusable: {}", cwd.display(), e)
        })?;
        if !metadata.is_dir() {
            anyhow::bail!("Working directory '{}' is not a directory", cwd.display());
        }

        // Checked before spawning so a refusal has no side effects at all, and
        // again at insert so two concurrent starts cannot both take the last slot.
        {
            let mut map = sessions();
            // Only swept under pressure. Sweeping on every start would silently
            // retire a session the moment its program exited and its last output
            // was read, so the pty_close that followed would fail with "no
            // interactive session" for a session the agent still held.
            if map.len() >= MAX_SESSIONS {
                reap_drained(&mut map);
            }
            if map.len() >= MAX_SESSIONS {
                return Err(capacity_error(&map));
            }
        }

        let session_id = new_session_id();
        let started = spawn_session(command, &cwd, cols, rows)?;
        let shared = Arc::clone(&started.shared);

        // The session is only handed to the registry once there is room for it,
        // so a refusal here still owns it and can shut it down.
        let mut pending = Some(started.session);
        let refused = {
            let mut map = sessions();
            if map.len() >= MAX_SESSIONS {
                reap_drained(&mut map);
            }
            match pending.take() {
                Some(session) if map.len() < MAX_SESSIONS => {
                    map.insert(session_id.clone(), session);
                    None
                }
                session => {
                    pending = session;
                    Some(capacity_error(&map))
                }
            }
        };
        if let Some(error) = refused {
            if let Some(session) = pending {
                shutdown(session).await;
            }
            return Err(error);
        }

        let (text, dropped) = collect_output(&shared, START_SETTLE_MS).await;
        Ok(render(&session_id, &status_line(&shared), &text, dropped))
    }
}

pub struct PtySendTool {
    #[allow(dead_code)]
    policy: Arc<WorkspacePolicy>,
}

impl PtySendTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

impl Default for PtySendTool {
    fn default() -> Self {
        Self::new(Arc::new(WorkspacePolicy::unrestricted()))
    }
}

#[async_trait]
impl Tool for PtySendTool {
    fn name(&self) -> &str {
        "pty_send"
    }

    fn description(&self) -> &str {
        "Sends input to an interactive terminal session and returns whatever it printed since the last read. Call it with no input to read further output."
    }

    fn requires_approval(&self) -> bool {
        true
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session id returned by pty_start."
                },
                "input": {
                    "type": "string",
                    "description": "Text to type into the terminal. Omit it, or pass an empty string, to just read new output. Use control characters directly (for example \"\\u0003\" for Ctrl-C)."
                },
                "submit": {
                    "type": "boolean",
                    "description": "Append a newline to 'input', as pressing Enter would (optional, default: true). Set it to false to send a bare keystroke."
                },
                "wait_ms": {
                    "type": "integer",
                    "description": "How long to wait for output, in milliseconds (optional, default: 500, maximum: 30000). Returns sooner once the program goes quiet or exits."
                }
            },
            "required": ["session_id"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let session_id = read_session_id(&args)?;

        let input = match &args["input"] {
            serde_json::Value::Null => "",
            value => value
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("'input' must be a string"))?,
        };
        let submit = match &args["submit"] {
            serde_json::Value::Null => true,
            value => value
                .as_bool()
                .ok_or_else(|| anyhow::anyhow!("'submit' must be a boolean"))?,
        };
        let wait_ms = match &args["wait_ms"] {
            serde_json::Value::Null => DEFAULT_WAIT_MS,
            value => value
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("'wait_ms' must be a positive integer"))?,
        };
        let wait_ms = std::cmp::min(wait_ms, MAX_WAIT_MS);

        // Only the handle is taken out of the registry; the guard is gone before
        // anything is awaited.
        let shared = {
            let map = sessions();
            map.get(session_id).map(|s| Arc::clone(&s.shared))
        };
        let shared = match shared {
            Some(shared) => shared,
            None => return Err(unknown_session(session_id)),
        };

        if !input.is_empty() {
            // Writing to a terminal whose program is gone fails with an opaque
            // I/O error; saying so plainly is more useful.
            if let Some(status) = poll_exit(&shared) {
                let (text, dropped) = collect_output(&shared, 0).await;
                return Ok(render(
                    session_id,
                    &format!("exited ({}); input was not sent", status),
                    &text,
                    dropped,
                ));
            }
            write_input(&shared, input, submit).await?;
        }

        let (text, dropped) = collect_output(&shared, wait_ms).await;
        Ok(render(session_id, &status_line(&shared), &text, dropped))
    }
}

pub struct PtyCloseTool {
    #[allow(dead_code)]
    policy: Arc<WorkspacePolicy>,
}

impl PtyCloseTool {
    pub fn new(policy: Arc<WorkspacePolicy>) -> Self {
        Self { policy }
    }
}

impl Default for PtyCloseTool {
    fn default() -> Self {
        Self::new(Arc::new(WorkspacePolicy::unrestricted()))
    }
}

#[async_trait]
impl Tool for PtyCloseTool {
    fn name(&self) -> &str {
        "pty_close"
    }

    fn description(&self) -> &str {
        "Closes an interactive terminal session, killing the program if it is still running, and returns any output that was never read."
    }

    fn requires_approval(&self) -> bool {
        true
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session id returned by pty_start."
                }
            },
            "required": ["session_id"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let session_id = read_session_id(&args)?;

        let session = { sessions().remove(session_id) };
        let session = match session {
            Some(session) => session,
            None => return Err(unknown_session(session_id)),
        };

        // Kept alive past the shutdown so the program's last words can still be
        // reported; the reader thread has finished writing into it by then.
        let output = Arc::clone(&session.shared.output);
        let (killed, joined) = shutdown(session).await;

        let (text, dropped) = {
            let mut buffer = lock_output(&output);
            if !joined {
                // The thread outlived the join, so stop it retaining anything more.
                buffer.stopped = true;
            }
            buffer.take()
        };

        let status = if joined {
            format!("closed, {}", killed)
        } else {
            format!("closed, {} (terminal reader still draining)", killed)
        };
        Ok(render(session_id, &status, &text, dropped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::Mutex as AsyncMutex;

    /// The registry is process-global and two tests assert on its whole
    /// contents, so the suite runs one pty test at a time. Async so no std guard
    /// is ever held across an await, even here.
    static TEST_LOCK: AsyncMutex<()> = AsyncMutex::const_new(());

    /// Serializes the suite and clears anything a previously failing test left
    /// behind, so one failure cannot cascade into the session cap.
    async fn exclusive() -> tokio::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().await;
        let leftovers: Vec<String> = { sessions().keys().cloned().collect() };
        for id in leftovers {
            let _ = PtyCloseTool::default()
                .execute(json!({ "session_id": id }))
                .await;
        }
        guard
    }

    fn shell_available() -> bool {
        std::process::Command::new("sh")
            .arg("-c")
            .arg("exit 0")
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn session_id_of(report: &str) -> String {
        report
            .lines()
            .find_map(|line| line.strip_prefix("Session: "))
            .map(|id| id.trim().to_string())
            .unwrap_or_default()
    }

    async fn start(command: &str) -> String {
        let report = PtyStartTool::default()
            .execute(json!({ "command": command }))
            .await
            .expect("session should start");
        let id = session_id_of(&report);
        assert!(!id.is_empty(), "{}", report);
        id
    }

    async fn send(id: &str, args: serde_json::Value) -> String {
        PtySendTool::default()
            .execute(args)
            .await
            .unwrap_or_else(|e| panic!("send to {} failed: {}", id, e))
    }

    async fn close(id: &str) -> String {
        PtyCloseTool::default()
            .execute(json!({ "session_id": id }))
            .await
            .unwrap_or_else(|e| panic!("close of {} failed: {}", id, e))
    }

    #[tokio::test]
    async fn round_trips_a_line_through_cat() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let id = start("cat").await;

        let report = send(
            &id,
            json!({ "session_id": id, "input": "hello pty", "wait_ms": 1000 }),
        )
        .await;
        assert!(report.contains("hello pty"), "{}", report);
        assert!(report.contains("running"), "{}", report);

        let closed = close(&id).await;
        assert!(closed.contains("closed"), "{}", closed);
    }

    /// The whole point of the tool: answering a program that stops and asks.
    #[tokio::test]
    async fn answers_a_prompt_from_an_interactive_program() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let start_report = PtyStartTool::default()
            .execute(json!({ "command": "printf 'name? '; read name; echo got:$name" }))
            .await
            .expect("session should start");
        let id = session_id_of(&start_report);
        assert!(start_report.contains("name?"), "{}", start_report);

        let report = send(
            &id,
            json!({ "session_id": id, "input": "world", "wait_ms": 2000 }),
        )
        .await;
        assert!(report.contains("got:world"), "{}", report);

        close(&id).await;
    }

    #[tokio::test]
    async fn an_exited_program_is_reported_with_its_status_and_still_closes() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let id = start("echo finishing; exit 3").await;

        let report = send(&id, json!({ "session_id": id, "wait_ms": 2000 })).await;
        assert!(report.contains("exited"), "{}", report);
        assert!(report.contains("code 3"), "{}", report);

        let closed = close(&id).await;
        assert!(closed.contains("closed"), "{}", closed);
        assert!(closed.contains("already exited"), "{}", closed);
    }

    #[tokio::test]
    async fn an_exited_session_does_not_swallow_input_silently() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let id = start("exit 0").await;

        let report = send(
            &id,
            json!({ "session_id": id, "input": "too late", "wait_ms": 500 }),
        )
        .await;
        assert!(report.contains("input was not sent"), "{}", report);

        close(&id).await;
    }

    #[tokio::test]
    async fn unknown_session_ids_error_cleanly() {
        let _guard = exclusive().await;
        let send_err = PtySendTool::default()
            .execute(json!({ "session_id": "pty-nope", "input": "hi" }))
            .await
            .expect_err("an unknown session must be an error");
        assert!(
            send_err.to_string().contains("No interactive session"),
            "{}",
            send_err
        );

        let close_err = PtyCloseTool::default()
            .execute(json!({ "session_id": "pty-nope" }))
            .await
            .expect_err("an unknown session must be an error");
        assert!(
            close_err.to_string().contains("No interactive session"),
            "{}",
            close_err
        );
    }

    #[tokio::test]
    async fn cwd_outside_the_workspace_is_refused() {
        let _guard = exclusive().await;
        let root = tempfile::tempdir().expect("root");
        let outside = tempfile::tempdir().expect("outside");
        let policy = Arc::new(WorkspacePolicy::with_roots([root.path()]).expect("policy"));

        let err = PtyStartTool::new(policy)
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
        assert!(
            sessions().is_empty(),
            "a refused start must open no session"
        );
    }

    #[tokio::test]
    async fn cwd_inside_the_workspace_is_allowed() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let root = tempfile::tempdir().expect("root");
        let policy = Arc::new(WorkspacePolicy::with_roots([root.path()]).expect("policy"));

        let report = PtyStartTool::new(policy)
            .execute(json!({
                "command": "pwd; cat",
                "cwd": root.path().to_string_lossy(),
            }))
            .await
            .expect("a cwd inside the root should run");
        let id = session_id_of(&report);
        close(&id).await;
    }

    #[tokio::test]
    async fn runaway_output_is_capped_and_reported() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        // Pure shell doubling keeps this portable: 32 * 2^15 = 1 MiB, four times
        // the retained buffer.
        let script = "s=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; \
                      for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do s=\"$s$s\"; done; \
                      printf '%s' \"$s\"; sleep 5";
        let start_report = PtyStartTool::default()
            .execute(json!({ "command": script }))
            .await
            .expect("session should start");
        let id = session_id_of(&start_report);

        let mut seen_truncation = start_report.contains("[output truncated:");
        for _ in 0..5 {
            if seen_truncation {
                break;
            }
            let report = send(&id, json!({ "session_id": id, "wait_ms": 1000 })).await;
            assert!(
                report.len() < MAX_REPORT_BYTES + 1024,
                "a report must stay near its budget, got {} bytes",
                report.len()
            );
            seen_truncation = report.contains("[output truncated:");
        }
        assert!(seen_truncation, "1 MiB of output must report dropped bytes");

        close(&id).await;
    }

    #[tokio::test]
    async fn multi_byte_output_survives_the_buffer() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        // Written as octal escapes so the bytes reach the terminal exactly as
        // encoded, whatever the shell's own idea of the locale is.
        let start_report = PtyStartTool::default()
            .execute(json!({
                "command": "printf '\\346\\227\\245\\346\\234\\254\\350\\252\\236'; cat"
            }))
            .await
            .expect("session should start");
        let id = session_id_of(&start_report);

        let echoed = send(
            &id,
            json!({ "session_id": id, "input": "漢字", "wait_ms": 1500 }),
        )
        .await;

        assert!(start_report.contains("日本語"), "{}", start_report);
        assert!(echoed.contains("漢字"), "{}", echoed);
        for report in [&start_report, &echoed] {
            assert!(
                !report.contains('\u{fffd}'),
                "no character may be split by the buffer: {}",
                report
            );
        }

        close(&id).await;
    }

    #[tokio::test]
    async fn exceeding_the_session_cap_is_a_clear_error() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let mut ids = Vec::new();
        for _ in 0..MAX_SESSIONS {
            // Prompting first lets each start return as soon as its program is
            // ready instead of waiting out the full settle eight times over.
            ids.push(start("printf 'ready '; cat").await);
        }

        let err = PtyStartTool::default()
            .execute(json!({ "command": "cat" }))
            .await
            .expect_err("the session cap must be enforced");
        let message = err.to_string();
        assert!(
            message.contains("At most 8 interactive sessions"),
            "{}",
            message
        );
        for id in &ids {
            assert!(
                message.contains(id),
                "{} should be listed in {}",
                id,
                message
            );
        }

        for id in &ids {
            close(id).await;
        }
        assert!(sessions().is_empty());
    }

    #[tokio::test]
    async fn a_closed_session_is_gone_and_its_program_is_dead() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let dir = tempfile::tempdir().expect("tempdir");
        let marker = dir.path().join("survived.txt");
        let id = PtyStartTool::default()
            .execute(json!({
                "command": format!("sleep 1; printf survived > {}", marker.display()),
                "cwd": dir.path().to_string_lossy(),
            }))
            .await
            .map(|report| session_id_of(&report))
            .expect("session should start");

        close(&id).await;
        assert!(sessions().get(&id).is_none());

        tokio::time::sleep(Duration::from_millis(1400)).await;
        assert!(
            !marker.exists(),
            "the killed program must not have finished its work"
        );

        let err = PtyCloseTool::default()
            .execute(json!({ "session_id": id }))
            .await
            .expect_err("closing twice must be an error, not a panic");
        assert!(
            err.to_string().contains("No interactive session"),
            "{}",
            err
        );
    }

    #[tokio::test]
    async fn rejects_bad_arguments() {
        let _guard = exclusive().await;
        let start_tool = PtyStartTool::default();
        assert!(start_tool.execute(json!({})).await.is_err());
        assert!(start_tool
            .execute(json!({ "command": "  " }))
            .await
            .is_err());
        assert!(start_tool
            .execute(json!({ "command": "cat", "cols": 0 }))
            .await
            .is_err());
        assert!(start_tool
            .execute(json!({ "command": "cat", "rows": -4 }))
            .await
            .is_err());
        assert!(start_tool
            .execute(json!({ "command": "cat", "cwd": 7 }))
            .await
            .is_err());

        let send_tool = PtySendTool::default();
        assert!(send_tool.execute(json!({})).await.is_err());
        assert!(send_tool
            .execute(json!({ "session_id": "" }))
            .await
            .is_err());
        assert!(send_tool
            .execute(json!({ "session_id": "x", "submit": "yes" }))
            .await
            .is_err());
        assert!(send_tool
            .execute(json!({ "session_id": "x", "wait_ms": -1 }))
            .await
            .is_err());

        assert!(PtyCloseTool::default().execute(json!({})).await.is_err());
        assert!(sessions().is_empty());
    }

    #[test]
    fn tool_futures_are_send() {
        fn assert_send<T: Send>(_: T) {}
        // Holding the SESSIONS guard across an await would make these futures
        // !Send and could deadlock a multi-threaded runtime.
        let start = PtyStartTool::default();
        let send = PtySendTool::default();
        let close = PtyCloseTool::default();
        assert_send(start.execute(json!({ "command": "cat" })));
        assert_send(send.execute(json!({ "session_id": "x" })));
        assert_send(close.execute(json!({ "session_id": "x" })));
    }

    #[test]
    fn tool_metadata_declares_approval_and_schemas() {
        let start = PtyStartTool::default();
        let send = PtySendTool::default();
        let close = PtyCloseTool::default();

        assert_eq!(start.name(), "pty_start");
        assert_eq!(send.name(), "pty_send");
        assert_eq!(close.name(), "pty_close");
        assert!(start.requires_approval());
        assert!(send.requires_approval());
        assert!(close.requires_approval());

        assert_eq!(start.parameters_schema()["required"], json!(["command"]));
        assert_eq!(send.parameters_schema()["required"], json!(["session_id"]));
        assert_eq!(close.parameters_schema()["required"], json!(["session_id"]));
        for key in ["command", "cwd", "cols", "rows"] {
            assert!(start.parameters_schema()["properties"][key].is_object());
        }
        for key in ["session_id", "input", "submit", "wait_ms"] {
            assert!(send.parameters_schema()["properties"][key].is_object());
        }
    }

    #[test]
    fn the_buffer_keeps_the_most_recent_bytes_and_counts_the_rest() {
        let mut buffer = OutputBuffer::default();
        buffer.push(&vec![b'a'; MAX_BUFFER_BYTES]);
        buffer.push(b"tail");

        let (text, dropped) = buffer.take();
        assert_eq!(dropped, 4);
        assert_eq!(text.len(), MAX_BUFFER_BYTES);
        assert!(text.ends_with("tail"));
        assert!(!buffer.has_unread());
    }

    #[test]
    fn a_character_split_across_reads_is_not_mangled() {
        let encoded = "日".as_bytes();
        let mut buffer = OutputBuffer::default();
        buffer.push(&encoded[..1]);

        let (partial, _) = buffer.take();
        assert!(
            partial.is_empty(),
            "an incomplete character must be held back"
        );

        buffer.push(&encoded[1..]);
        let (whole, _) = buffer.take();
        assert_eq!(whole, "日");
    }

    #[test]
    fn a_partial_character_is_flushed_once_nothing_more_can_arrive() {
        let mut buffer = OutputBuffer::default();
        buffer.push(&"日".as_bytes()[..1]);
        buffer.eof = true;

        let (text, _) = buffer.take();
        assert_eq!(text, "\u{fffd}");
    }

    #[test]
    fn incomplete_tails_are_measured_correctly() {
        assert_eq!(incomplete_tail_len(b"plain ascii"), 0);
        assert_eq!(incomplete_tail_len("日本語".as_bytes()), 0);
        assert_eq!(incomplete_tail_len(&"日".as_bytes()[..1]), 1);
        assert_eq!(incomplete_tail_len(&"日".as_bytes()[..2]), 2);
        assert_eq!(incomplete_tail_len(&"😀".as_bytes()[..3]), 3);
        assert_eq!(incomplete_tail_len(b""), 0);
    }

    #[test]
    fn the_report_keeps_the_tail_where_the_prompt_is() {
        let text = format!("{}password: ", "x".repeat(MAX_REPORT_BYTES * 2));
        let report = render("pty-1-0", "running (pid 1)", &text, 0);

        assert!(report.ends_with("bytes]\n"), "{}", &report[..80]);
        assert!(report.contains("password: "));
        assert!(report.contains("[output truncated:"));
        assert!(report.len() < MAX_REPORT_BYTES + 512);
    }

    #[test]
    fn tail_at_boundary_never_splits_a_character() {
        let text = "日".repeat(64);
        let tail = tail_at_boundary(&text, 10);
        assert!(tail.len() <= 10);
        assert!(text.ends_with(tail));
    }

    /// A program that prints without pausing must not make one wait allocate at
    /// the pty's throughput for the whole of `wait_ms`. Before the collect cap
    /// this reached ~110 MiB per 10 s of wait, so ~330 MiB at the 30 s maximum.
    #[tokio::test]
    async fn a_flooding_program_cannot_grow_one_wait_without_bound() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let id = start("yes abcdefghijklmnopqrstuvwxyz0123456789").await;
        let shared = {
            sessions()
                .get(&id)
                .map(|s| Arc::clone(&s.shared))
                .expect("session should be registered")
        };

        let (text, dropped) = collect_output(&shared, 2_000).await;
        assert!(
            text.len() <= MAX_COLLECT_BYTES,
            "one wait accumulated {} bytes, above the {} byte cap",
            text.len(),
            MAX_COLLECT_BYTES
        );
        // The flood really did happen, so the cap is what held it down.
        assert!(
            dropped > MAX_COLLECT_BYTES,
            "expected a flood to report dropped bytes, got {}",
            dropped
        );
        assert!(
            !text.contains('\u{fffd}'),
            "the cap must split on a boundary"
        );

        close(&id).await;
    }

    /// An exited session is still an open session: the agent has its id and has
    /// to be able to close it, however many other sessions it starts meanwhile.
    #[tokio::test]
    async fn an_exited_session_stays_closable_after_later_starts() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let done = start("echo bye; exit 0").await;
        let drained = send(&done, json!({ "session_id": done, "wait_ms": 1500 })).await;
        assert!(drained.contains("exited"), "{}", drained);

        // A later start must not quietly retire the finished session.
        let other = start("cat").await;
        assert!(
            sessions().contains_key(&done),
            "an exited session must survive a later pty_start"
        );

        let closed = PtyCloseTool::default()
            .execute(json!({ "session_id": done }))
            .await
            .expect("an exited session must still be closable");
        assert!(closed.contains("closed"), "{}", closed);

        close(&other).await;
    }

    /// Closing must reap the child, not just signal it: an unreaped kill leaves a
    /// zombie and reports that it could not confirm the death.
    #[tokio::test]
    async fn closing_a_running_session_reaps_the_child() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let id = start("sleep 60").await;

        let closed = close(&id).await;
        assert!(
            closed.contains("killed ("),
            "close must report a reaped status: {}",
            closed
        );
        assert!(
            !closed.contains("has not been reaped"),
            "the child must be reaped, not left a zombie: {}",
            closed
        );
    }

    /// The cap is checked before spawning and again at insert, so starts racing
    /// for the last slot cannot both take it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn concurrent_starts_cannot_exceed_the_session_cap() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let mut tasks = Vec::new();
        for _ in 0..24 {
            tasks.push(tokio::spawn(async {
                PtyStartTool::default()
                    .execute(json!({ "command": "printf ready; cat" }))
                    .await
            }));
        }

        let mut ids = Vec::new();
        for task in tasks {
            if let Ok(Ok(report)) = task.await {
                ids.push(session_id_of(&report));
            }
        }
        assert!(
            ids.len() <= MAX_SESSIONS,
            "{} starts succeeded against a cap of {}",
            ids.len(),
            MAX_SESSIONS
        );
        assert!(
            sessions().len() <= MAX_SESSIONS,
            "the registry holds {} sessions, above the cap",
            sessions().len()
        );

        for id in &ids {
            close(id).await;
        }
        assert!(sessions().is_empty());
    }

    #[tokio::test]
    async fn oversized_input_is_refused_rather_than_written() {
        if !shell_available() {
            return;
        }
        let _guard = exclusive().await;
        let id = start("cat").await;

        let err = PtySendTool::default()
            .execute(json!({ "session_id": id, "input": "x".repeat(MAX_INPUT_BYTES + 1) }))
            .await
            .expect_err("input above the limit must be refused");
        assert!(err.to_string().contains("may be sent at once"), "{}", err);

        // The session is untouched and still usable.
        let report = send(
            &id,
            json!({ "session_id": id, "input": "still here", "wait_ms": 1000 }),
        )
        .await;
        assert!(report.contains("still here"), "{}", report);

        close(&id).await;
    }

    #[test]
    fn session_ids_are_unique() {
        let ids: std::collections::HashSet<String> = (0..64).map(|_| new_session_id()).collect();
        assert_eq!(ids.len(), 64);
    }
}
