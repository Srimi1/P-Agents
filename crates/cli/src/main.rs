use anyhow::Result;
use clap::Parser;
use cli::app::{AppOptions, HarnessApp, SESSION_DIR};
use cli::config::HarnessConfig;
use cli::repl::{settle, start_repl};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "harness",
    version = "0.1.0",
    about = "Universal Multi-Agent Harness in Rust"
)]
struct CliArgs {
    #[arg(
        short,
        long,
        help = "Run one prompt and exit instead of opening the REPL"
    )]
    prompt: Option<String>,

    #[arg(long, help = "Provider to use: anthropic, openai, or mock")]
    provider: Option<String>,

    #[arg(long, help = "Model name, overriding the configured default")]
    model: Option<String>,

    #[arg(long, help = "Path to a config.toml, overriding the default location")]
    config: Option<PathBuf>,

    #[arg(long, help = "Approve every tool call without asking")]
    yolo: bool,

    #[arg(long, help = "Resume a previous session by id or id prefix")]
    resume: Option<String>,

    #[arg(long, help = "Use the offline scripted mock provider")]
    mock: bool,

    #[arg(long, help = "Directory for session transcripts", default_value = SESSION_DIR)]
    session_dir: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Default to warn so tracing lines don't fight with the REPL's own output;
    // RUST_LOG still overrides it.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let args = CliArgs::parse();
    let config = HarnessConfig::load(args.config.as_deref())?;

    let options = AppOptions {
        provider: if args.mock {
            Some("mock".to_string())
        } else {
            args.provider.clone()
        },
        model: args.model.clone(),
        yolo: args.yolo,
        session_dir: args.session_dir.clone(),
    };

    let (mut app, approvals) = HarnessApp::new(config, options).await?;

    if let Some(id) = &args.resume {
        let restored = app.resume(&args.session_dir, id).await?;
        println!("Restored {restored} messages from session {id}.");
    }

    match args.prompt {
        Some(prompt) => {
            // One-shot runs still need an approval responder, or a gated tool
            // call would hang. Without a terminal to ask, deny by policy unless
            // --yolo was passed.
            let auto_deny = tokio::spawn(async move {
                let mut approvals = approvals;
                while let Some(request) = approvals.recv().await {
                    eprintln!(
                        "Denying '{}' from {}: non-interactive run without --yolo.",
                        request.tool, request.agent_id
                    );
                    let _ = request.respond.send(runtime::ApprovalDecision::Deny);
                }
            });

            let result = app.run_prompt(&prompt).await;
            settle().await;
            auto_deny.abort();
            match result {
                Ok(answer) => println!("\n{answer}"),
                Err(err) => {
                    app.shutdown().await;
                    return Err(err);
                }
            }
            app.shutdown().await;
        }
        None => start_repl(app, approvals).await?,
    }

    Ok(())
}
