use crate::app::{HarnessApp, SESSION_DIR};
use agent_core::TokenUsage;
use anyhow::Result;
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use runtime::{ApprovalDecision, ApprovalRequest, HarnessEvent};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use subagents::LEAD_AGENT_ID;
use tokio::sync::mpsc;
use tokio::sync::Mutex as AsyncMutex;

/// Grace period after a turn so the renderer can drain the bus before the main
/// loop prints the final answer. Events arrive over a channel, so without this
/// the answer can overtake the last tool line.
const RENDER_SETTLE: Duration = Duration::from_millis(60);

/// Ceiling on how long a turn waits for the renderer to catch up. Only reached
/// if the renderer has died, in which case the REPL carries on regardless.
const SYNC_TIMEOUT: Duration = Duration::from_secs(3);

/// Not a real agent. A state change stamped with this id is pushed through the
/// event pipeline after each turn; because the channel and the bus are both
/// FIFO, seeing it means everything the turn produced has already been drawn.
const SYNC_AGENT_ID: &str = "__repl_sync__";

type Lines = Arc<AsyncMutex<mpsc::UnboundedReceiver<String>>>;
type UsageTable = Arc<Mutex<HashMap<String, TokenUsage>>>;

/// Shared with the renderer so it knows whose tokens to print as the primary
/// stream. Everyone else is dimmed and prefixed with their agent id.
#[derive(Clone)]
struct Focus {
    agent_id: Arc<Mutex<String>>,
    /// Whether the focused agent has streamed any text this turn. When it has,
    /// the REPL must not print the answer again underneath it.
    streamed: Arc<std::sync::atomic::AtomicBool>,
    sync: Arc<tokio::sync::Notify>,
}

impl Focus {
    fn new() -> Self {
        Self {
            agent_id: Arc::new(Mutex::new(LEAD_AGENT_ID.to_string())),
            streamed: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            sync: Arc::new(tokio::sync::Notify::new()),
        }
    }

    fn begin_turn(&self, agent_id: &str) {
        *self.agent_id.lock().unwrap_or_else(|p| p.into_inner()) = agent_id.to_string();
        self.streamed
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    fn is_focused(&self, agent_id: &str) -> bool {
        *self.agent_id.lock().unwrap_or_else(|p| p.into_inner()) == agent_id
    }

    fn mark_streamed(&self) {
        self.streamed
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    fn did_stream(&self) -> bool {
        self.streamed.load(std::sync::atomic::Ordering::SeqCst)
    }
}

pub async fn start_repl(
    mut app: HarnessApp,
    approvals: mpsc::Receiver<ApprovalRequest>,
) -> Result<()> {
    banner(&app);

    let lines = spawn_stdin_reader();
    let usage: UsageTable = Arc::new(Mutex::new(HashMap::new()));
    let focus = Focus::new();
    let renderer = tokio::spawn(render_loop(
        app.runtime.subscribe(),
        approvals,
        Arc::clone(&lines),
        Arc::clone(&usage),
        focus.clone(),
    ));
    let sink = app.runtime.event_sink();

    loop {
        print!("\n{} ", "harness>".bold().blue());
        io::stdout().flush()?;

        let Some(input) = read_line(&lines).await else {
            break;
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        match parse_command(trimmed) {
            Command::Exit => break,
            Command::Help => print_help(),
            Command::Usage => print_usage(&usage),
            Command::Session => println!(
                "  session {} at {}",
                app.runtime.session_id().dimmed(),
                app.runtime.session_path().display().to_string().dimmed()
            ),
            Command::Model(spec) => match swap(&mut app, spec.as_deref()) {
                Ok(()) => println!(
                    "  {} now {} via {}",
                    "model".green(),
                    app.model_name.bold(),
                    app.provider_name
                ),
                Err(err) => println!("  {} {}", "model change failed:".red(), err),
            },
            Command::Resume(id) => match app.resume(std::path::Path::new(SESSION_DIR), &id).await {
                Ok(n) => println!(
                    "  {} {} messages from session {}",
                    "restored".green(),
                    n,
                    id
                ),
                Err(err) => println!("  {} {}", "resume failed:".red(), err),
            },
            Command::Persona { role, text } => {
                let subject = match text {
                    Some(text) => text,
                    None => match app.last_answer() {
                        Some(answer) => format!(
                            "Review the following proposal from the lead agent:\n\n{answer}"
                        ),
                        None => {
                            println!(
                                "  {}",
                                "nothing to review yet; give /critic some text".yellow()
                            );
                            continue;
                        }
                    },
                };
                announce(&role);
                focus.begin_turn(&format!("{role}-oneshot"));
                let result = app.run_persona_once(&role, &subject).await;
                sync_renderer(&sink, &focus).await;
                match result {
                    Ok(answer) => finish(&answer, &focus),
                    Err(err) => println!("\n{} {}", "error:".bold().red(), err),
                }
            }
            Command::Prompt(text) => {
                announce("lead planner");
                focus.begin_turn(LEAD_AGENT_ID);
                let result = app.run_prompt(&text).await;
                sync_renderer(&sink, &focus).await;
                match result {
                    Ok(answer) => finish(&answer, &focus),
                    Err(err) => println!("\n{} {}", "error:".bold().red(), err),
                }
            }
        }
    }

    println!("{}", "\nExiting harness.".yellow());
    renderer.abort();
    app.shutdown().await;
    Ok(())
}

fn banner(app: &HarnessApp) {
    let rule = "─".repeat(60);
    println!("{}", rule.cyan());
    println!("{}", "Universal Multi-Agent Harness".bold().green());
    println!(
        "  model    {} via {}",
        app.model_name.bold(),
        app.provider_name
    );
    println!(
        "  personas {}",
        app.orchestrator.personas().roles().join(", ")
    );
    println!(
        "  session  {}",
        app.runtime.session_path().display().to_string().dimmed()
    );
    println!("  {}", "/help for commands, /exit to quit".dimmed());
    println!("{}", rule.cyan());
}

fn announce(who: &str) {
    println!("\n{}", format!("▸ {who} working…").italic().magenta());
}

/// Closes out a turn. When the focused agent already streamed its answer to the
/// screen, repeating it here would show it twice, so only a rule is drawn.
fn finish(answer: &str, focus: &Focus) {
    if focus.did_stream() {
        println!("\n{}", "─".repeat(60).dimmed());
    } else {
        println!("\n{}\n{}", "◆ Final answer".bold().green(), answer);
    }
}

/// Pushes a sentinel through the event pipeline and waits for the renderer to
/// draw it, so the turn's output is on screen before the prompt returns.
async fn sync_renderer(sink: &agent_core::EventSink, focus: &Focus) {
    let notified = focus.sync.notified();
    let sent = sink
        .send(agent_core::AgentEvent::StateChanged {
            agent_id: SYNC_AGENT_ID.to_string(),
            state: agent_core::AgentState::Idle,
        })
        .is_ok();

    if sent && tokio::time::timeout(SYNC_TIMEOUT, notified).await.is_ok() {
        return;
    }
    // No renderer, or it never acknowledged: fall back to a short grace period
    // rather than blocking the REPL.
    tokio::time::sleep(RENDER_SETTLE).await;
}

// ---------------------------------------------------------------- commands

enum Command {
    Prompt(String),
    Persona { role: String, text: Option<String> },
    Model(Option<String>),
    Resume(String),
    Usage,
    Session,
    Help,
    Exit,
}

fn parse_command(input: &str) -> Command {
    if !input.starts_with('/') {
        return match input {
            "exit" | "quit" => Command::Exit,
            other => Command::Prompt(other.to_string()),
        };
    }

    let (head, rest) = input.split_once(char::is_whitespace).unwrap_or((input, ""));
    let rest = rest.trim();
    let arg = if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    };

    match head {
        "/exit" | "/quit" => Command::Exit,
        "/help" | "/?" => Command::Help,
        "/usage" => Command::Usage,
        "/session" => Command::Session,
        "/model" => Command::Model(arg),
        "/resume" => match arg {
            Some(id) => Command::Resume(id),
            None => Command::Help,
        },
        // Forces decomposition before any work happens.
        "/plan" => Command::Prompt(format!(
            "Before doing any work, decompose this goal into an explicit numbered plan of atomic \
             steps and say which specialist should own each one. Then execute the plan.\n\n{}",
            arg.unwrap_or_default()
        )),
        "/critic" => Command::Persona {
            role: "critic".to_string(),
            text: arg,
        },
        "/verify" => Command::Persona {
            role: "verifier".to_string(),
            text: arg,
        },
        _ => Command::Help,
    }
}

fn print_help() {
    let rows = [
        (
            "/plan <goal>",
            "force an explicit decomposition before work starts",
        ),
        (
            "/critic [text]",
            "have the Egoist challenge the last answer, or the given text",
        ),
        (
            "/verify [text]",
            "have the Verifier check the last answer, or the given text",
        ),
        (
            "/model [name]",
            "hot-swap the model; bare /model reverts to the configured default",
        ),
        (
            "/resume <id>",
            "replay a previous session's lead transcript",
        ),
        ("/usage", "token usage so far, per agent"),
        ("/session", "current session id and transcript path"),
        ("/exit", "quit"),
    ];
    println!();
    for (cmd, help) in rows {
        println!("  {:<16} {}", cmd.bold().cyan(), help.dimmed());
    }
    println!(
        "\n  {}",
        "Anything else is sent to the Lead Planner, which delegates to specialists.".dimmed()
    );
}

fn print_usage(usage: &UsageTable) {
    let table = usage.lock().unwrap_or_else(|p| p.into_inner());
    if table.is_empty() {
        println!("  {}", "no usage reported yet".dimmed());
        return;
    }

    let mut rows: Vec<_> = table.iter().collect();
    rows.sort_by(|a, b| a.0.cmp(b.0));
    let mut total = TokenUsage::default();
    println!();
    for (agent, usage) in rows {
        total.accumulate(usage);
        println!(
            "  {:<20} in {:>7}  out {:>7}  total {:>8}",
            agent.bold(),
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens
        );
    }
    println!(
        "  {:<20} in {:>7}  out {:>7}  total {:>8}",
        "ALL".bold().green(),
        total.prompt_tokens,
        total.completion_tokens,
        total.total_tokens
    );
}

fn swap(app: &mut HarnessApp, spec: Option<&str>) -> Result<()> {
    // "/model openai:gpt-4o" switches provider and model together;
    // "/model gpt-4o" keeps the current provider.
    match spec {
        None => app.swap_model(None, None),
        Some(spec) => match spec.split_once(':') {
            Some((provider, model)) => app.swap_model(Some(provider), Some(model)),
            None => app.swap_model(Some(&app.provider_name.clone()), Some(spec)),
        },
    }
}

// ------------------------------------------------------------------ input

/// stdin has exactly one reader for the whole process. The REPL loop takes
/// lines between turns and the renderer takes them during approval prompts;
/// they never contend, because the REPL is inside `agent.run` while approvals
/// happen.
fn spawn_stdin_reader() -> Lines {
    let (tx, rx) = mpsc::unbounded_channel();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(line) => {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    Arc::new(AsyncMutex::new(rx))
}

async fn read_line(lines: &Lines) -> Option<String> {
    lines.lock().await.recv().await
}

// --------------------------------------------------------------- renderer

#[derive(Default)]
struct Renderer {
    /// Partial lines for background agents, buffered so their output never
    /// interleaves mid-word with the focused agent's stream.
    sub_buffers: HashMap<String, String>,
    lead_mid_line: bool,
    spinner: Option<ProgressBar>,
}

impl Renderer {
    fn clear_spinner(&mut self) {
        if let Some(spinner) = self.spinner.take() {
            spinner.finish_and_clear();
        }
    }

    /// Ends the lead's streamed line before printing a structured line under it.
    fn break_line(&mut self) {
        self.clear_spinner();
        if self.lead_mid_line {
            println!();
            self.lead_mid_line = false;
        }
    }

    fn flush_all_subs(&mut self) {
        let ids: Vec<String> = self.sub_buffers.keys().cloned().collect();
        for id in ids {
            self.flush_sub(&id);
        }
    }

    fn flush_sub(&mut self, agent_id: &str) {
        if let Some(buffer) = self.sub_buffers.get_mut(agent_id) {
            if buffer.trim().is_empty() {
                buffer.clear();
                return;
            }
            let text = std::mem::take(buffer);
            self.break_line();
            for line in text.lines() {
                println!("  {} {}", format!("[{agent_id}]").dimmed(), line.dimmed());
            }
        }
    }
}

async fn render_loop(
    mut bus: tokio::sync::broadcast::Receiver<HarnessEvent>,
    mut approvals: mpsc::Receiver<ApprovalRequest>,
    lines: Lines,
    usage: UsageTable,
    focus: Focus,
) {
    let mut renderer = Renderer::default();

    loop {
        let pending = tokio::select! {
            // Approvals win: an agent is blocked waiting on the answer.
            biased;

            request = approvals.recv() => request,
            event = bus.recv() => {
                match event {
                    Ok(event) => handle_event(event, &mut renderer, &usage, &focus),
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        renderer.break_line();
                        println!("  {}", format!("… {n} events skipped (renderer fell behind)").dimmed());
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
                None
            }
        };

        let Some(request) = pending else { continue };

        // Draw whatever is already queued before asking. Consent is meaningless
        // if the tool call that prompted it has not appeared on screen yet.
        while let Ok(event) = bus.try_recv() {
            handle_event(event, &mut renderer, &usage, &focus);
        }
        renderer.break_line();
        let decision = prompt_for_approval(&request, &lines).await;
        let _ = request.respond.send(decision);
    }
}

fn handle_event(event: HarnessEvent, renderer: &mut Renderer, usage: &UsageTable, focus: &Focus) {
    // The turn-boundary sentinel: everything queued ahead of it is now drawn.
    if event_agent_id(&event) == Some(SYNC_AGENT_ID) {
        renderer.flush_all_subs();
        renderer.break_line();
        focus.sync.notify_one();
        return;
    }

    match event {
        HarnessEvent::TextDelta { agent_id, delta } => {
            if focus.is_focused(&agent_id) {
                renderer.clear_spinner();
                print!("{delta}");
                let _ = io::stdout().flush();
                renderer.lead_mid_line = !delta.ends_with('\n');
                if !delta.trim().is_empty() {
                    focus.mark_streamed();
                }
            } else {
                let buffer = renderer.sub_buffers.entry(agent_id.clone()).or_default();
                buffer.push_str(&delta);
                if buffer.contains('\n') {
                    let complete = buffer.rfind('\n').map(|i| i + 1).unwrap_or(0);
                    let ready: String = buffer.drain(..complete).collect();
                    renderer.break_line();
                    for line in ready.lines() {
                        println!("  {} {}", format!("[{agent_id}]").dimmed(), line.dimmed());
                    }
                }
            }
        }
        HarnessEvent::ToolStarted {
            agent_id,
            tool,
            arguments,
        } => {
            renderer.break_line();
            let who = agent_label(&agent_id, focus);
            let spinner = ProgressBar::new_spinner();
            spinner.set_style(
                ProgressStyle::with_template("  {spinner:.cyan} {msg}")
                    .unwrap_or_else(|_| ProgressStyle::default_spinner()),
            );
            spinner.set_message(format!(
                "{}{} {}",
                who,
                tool.bold(),
                summarize_args(&arguments).dimmed()
            ));
            spinner.enable_steady_tick(Duration::from_millis(90));
            renderer.spinner = Some(spinner);
        }
        HarnessEvent::ToolFinished {
            agent_id,
            tool,
            preview,
            is_error,
        } => {
            renderer.clear_spinner();
            // A denial comes back as a successful observation, but showing it
            // with a green tick would read as "the tool ran".
            let denied = preview.starts_with("DENIED by user");
            let mark = if is_error {
                "✗".red()
            } else if denied {
                "⊘".yellow()
            } else {
                "✓".green()
            };
            let who = agent_label(&agent_id, focus);
            let first_line = preview.lines().next().unwrap_or("").trim();
            println!("  {} {}{} {}", mark, who, tool.bold(), first_line.dimmed());
        }
        HarnessEvent::SubAgentSpawned { agent_id, role, .. } => {
            renderer.break_line();
            println!(
                "  {} {} {}",
                "↳".cyan(),
                "delegating to".dimmed(),
                format!("{role} ({agent_id})").bold()
            );
        }
        HarnessEvent::SubAgentFinished { agent_id, ok } => {
            renderer.flush_sub(&agent_id);
            renderer.break_line();
            let mark = if ok { "✓".green() } else { "✗".red() };
            println!("  {} {} {}", mark, agent_id.bold(), "finished".dimmed());
        }
        HarnessEvent::UsageReport {
            agent_id,
            cumulative,
            ..
        } => {
            usage
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(agent_id, cumulative);
        }
        HarnessEvent::Compacted {
            messages_before,
            messages_after,
            ..
        } => {
            renderer.break_line();
            println!(
                "  {}",
                format!("⟳ compacted context ({messages_before} → {messages_after} messages)")
                    .dimmed()
            );
        }
        HarnessEvent::StateChanged { .. } | HarnessEvent::MessageAppended { .. } => {}
    }
}

/// Background agents are labelled; the focused agent's own lines are not, since
/// its output is the main stream the user is reading.
fn agent_label(agent_id: &str, focus: &Focus) -> String {
    if focus.is_focused(agent_id) {
        String::new()
    } else {
        format!("[{agent_id}] ")
    }
}

fn event_agent_id(event: &HarnessEvent) -> Option<&str> {
    match event {
        HarnessEvent::StateChanged { agent_id, .. }
        | HarnessEvent::TextDelta { agent_id, .. }
        | HarnessEvent::MessageAppended { agent_id, .. }
        | HarnessEvent::ToolStarted { agent_id, .. }
        | HarnessEvent::ToolFinished { agent_id, .. }
        | HarnessEvent::UsageReport { agent_id, .. }
        | HarnessEvent::SubAgentSpawned { agent_id, .. }
        | HarnessEvent::SubAgentFinished { agent_id, .. }
        | HarnessEvent::Compacted { agent_id, .. } => Some(agent_id),
    }
}

/// One-line gist of a tool call's arguments for the status line.
fn summarize_args(arguments: &serde_json::Value) -> String {
    let Some(object) = arguments.as_object() else {
        return String::new();
    };
    let parts: Vec<String> = object
        .iter()
        .take(3)
        .map(|(key, value)| {
            let rendered = match value {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            let flat = rendered.replace('\n', " ");
            format!("{key}={}", agent_core::truncate_at_boundary(&flat, 48))
        })
        .collect();
    parts.join(" ")
}

async fn prompt_for_approval(request: &ApprovalRequest, lines: &Lines) -> ApprovalDecision {
    let rule = "─".repeat(60);
    println!("\n{}", rule.yellow());
    println!(
        "{} {} requested by {}",
        "APPROVAL".bold().yellow(),
        request.tool.bold(),
        request.agent_id
    );
    match serde_json::to_string_pretty(&request.arguments) {
        Ok(pretty) => {
            for line in pretty.lines().take(20) {
                println!("  {}", line.dimmed());
            }
        }
        Err(_) => println!("  {}", request.arguments.to_string().dimmed()),
    }
    println!("{}", rule.yellow());
    print!(
        "{} ",
        "allow? [y]es / [n]o / [a]lways (this tool, any agent, rest of session):".bold()
    );
    let _ = io::stdout().flush();

    // Deliberately no "discard type-ahead" step here. Draining buffered lines
    // would swallow the answer whenever input arrives faster than the prompt is
    // drawn, which is always the case for piped input.
    match read_line(lines).await {
        Some(answer) => match answer.trim().to_ascii_lowercase().as_str() {
            "y" | "yes" => ApprovalDecision::Approve,
            "a" | "always" => ApprovalDecision::ApproveForSession,
            // Anything else, including a bare Enter, is a refusal: an ambiguous
            // response must never authorize a write or a shell command.
            _ => ApprovalDecision::Deny,
        },
        // stdin closed mid-prompt.
        None => ApprovalDecision::Deny,
    }
}

/// Waits for the renderer to catch up before the caller prints its own output.
pub async fn settle() {
    tokio::time::sleep(RENDER_SETTLE).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_a_prompt() {
        assert!(
            matches!(parse_command("build me a parser"), Command::Prompt(t) if t == "build me a parser")
        );
    }

    #[test]
    fn bare_exit_words_quit() {
        assert!(matches!(parse_command("exit"), Command::Exit));
        assert!(matches!(parse_command("/quit"), Command::Exit));
    }

    #[test]
    fn persona_commands_carry_optional_text() {
        match parse_command("/critic this design is wrong") {
            Command::Persona { role, text } => {
                assert_eq!(role, "critic");
                assert_eq!(text.as_deref(), Some("this design is wrong"));
            }
            _ => panic!("expected a persona command"),
        }
        match parse_command("/verify") {
            Command::Persona { role, text } => {
                assert_eq!(role, "verifier");
                assert!(text.is_none());
            }
            _ => panic!("expected a persona command"),
        }
    }

    #[test]
    fn plan_wraps_the_goal_in_a_decomposition_instruction() {
        match parse_command("/plan ship the CLI") {
            Command::Prompt(text) => {
                assert!(text.contains("ship the CLI"));
                assert!(text.contains("numbered plan"));
            }
            _ => panic!("expected a prompt"),
        }
    }

    #[test]
    fn model_accepts_bare_and_provider_qualified_forms() {
        assert!(matches!(parse_command("/model"), Command::Model(None)));
        assert!(matches!(parse_command("/model gpt-4o"), Command::Model(Some(m)) if m == "gpt-4o"));
        assert!(
            matches!(parse_command("/model openai:gpt-4o"), Command::Model(Some(m)) if m == "openai:gpt-4o")
        );
    }

    #[test]
    fn resume_without_an_id_shows_help() {
        assert!(matches!(parse_command("/resume"), Command::Help));
        assert!(matches!(parse_command("/resume abc123"), Command::Resume(id) if id == "abc123"));
    }

    #[test]
    fn unknown_slash_command_shows_help() {
        assert!(matches!(parse_command("/nonsense"), Command::Help));
    }

    #[test]
    fn argument_summary_is_single_line_and_bounded() {
        let args = serde_json::json!({
            "path": "src/main.rs",
            "content": "line one\nline two\nline three",
        });
        let summary = summarize_args(&args);
        assert!(!summary.contains('\n'));
        assert!(summary.contains("path=src/main.rs"));
    }

    #[test]
    fn argument_summary_tolerates_non_objects() {
        assert_eq!(summarize_args(&serde_json::Value::Null), "");
    }
}
