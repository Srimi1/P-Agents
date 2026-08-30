mod repl;

use agent_core::ToolDispatcher;
use anyhow::Result;
use clap::Parser;
use cli::config::HarnessConfig;
use cli::provider_factory::make_provider;
use repl::start_interactive_repl;
use std::sync::Arc;
use subagents::{build_tool_registry, MultiAgentOrchestrator};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "harness",
    version = "0.1.0",
    about = "Universal Multi-Agent Harness in Rust"
)]
struct CliArgs {
    #[arg(short, long, help = "Direct prompt to execute without entering the REPL")]
    prompt: Option<String>,

    #[arg(long, help = "Provider to use: anthropic, openai, or mock")]
    provider: Option<String>,

    #[arg(long, help = "Model name, overriding the configured default")]
    model: Option<String>,

    #[arg(long, help = "Path to a config.toml, overriding the default location")]
    config: Option<std::path::PathBuf>,

    #[arg(long, help = "Run the offline scripted mock provider")]
    mock: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = CliArgs::parse();
    let config = HarnessConfig::load(args.config.as_deref())?;

    let provider_name = if args.mock {
        Some("mock")
    } else {
        args.provider.as_deref()
    };
    let provider = make_provider(&config, provider_name, args.model.as_deref())?;

    let orchestrator = MultiAgentOrchestrator::new(provider)
        .with_max_parallel(config.limits.max_parallel_subagents)
        .with_max_iterations(config.limits.max_iterations);

    let dispatcher: Arc<dyn ToolDispatcher> = Arc::new(build_tool_registry());

    if let Some(prompt) = args.prompt {
        let mut lead_agent = orchestrator.create_lead_agent(dispatcher, None, None)?;
        println!("Executing prompt: {}", prompt);
        let result = lead_agent.run(&prompt).await?;
        println!("\nResult:\n{}", result);
    } else {
        start_interactive_repl(orchestrator, dispatcher).await?;
    }

    Ok(())
}
