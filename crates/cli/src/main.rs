mod config;
mod repl;

use agent_core::GenericOpenAiProvider;
use anyhow::Result;
use clap::Parser;
use config::HarnessConfig;
use repl::start_interactive_repl;
use std::sync::Arc;
use subagents::MultiAgentOrchestrator;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "harness", version = "0.1.0", about = "Universal Multi-Agent Harness in Rust")]
struct CliArgs {
    #[arg(short, long, help = "Direct prompt to execute without entering REPL")]
    prompt: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let args = CliArgs::parse();
    let config = HarnessConfig::load()?;

    let provider = Arc::new(GenericOpenAiProvider::new(
        config.api_key,
        config.base_url,
        config.model,
    ));

    let orchestrator = MultiAgentOrchestrator::new(provider);

    if let Some(prompt) = args.prompt {
        let mut lead_agent = orchestrator.create_lead_agent()?;
        println!("⚙️ Executing prompt: {}", prompt);
        let result = lead_agent.run(&prompt).await?;
        println!("\n🤖 Result:\n{}", result);
    } else {
        start_interactive_repl(orchestrator).await?;
    }

    Ok(())
}
