use agent_core::ToolDispatcher;
use anyhow::Result;
use colored::*;
use std::io::{self, Write};
use std::sync::Arc;
use subagents::MultiAgentOrchestrator;

pub async fn start_interactive_repl(
    orchestrator: MultiAgentOrchestrator,
    dispatcher: Arc<dyn ToolDispatcher>,
) -> Result<()> {
    println!("{}", "=========================================================".cyan());
    println!("{}", "🚀 Universal Rust Multi-Agent Harness Initialized!".bold().green());
    println!("{}", "Personas active: Planner, Engineer, Verifier, Critic, Researcher, Analyst".yellow());
    println!("{}", "Type your goal or prompt below (or 'exit' to quit):".cyan());
    println!("{}", "=========================================================\n".cyan());

    let mut lead_agent = orchestrator.create_lead_agent(dispatcher, None, None)?;

    loop {
        print!("{}", "harness> ".bold().blue());
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed == "exit" || trimmed == "quit" {
            println!("{}", "Exiting harness. Goodbye!".yellow());
            break;
        }

        println!("\n{}", "⚙️ Lead Planner orchestrating task...".italic().magenta());

        match lead_agent.run(trimmed).await {
            Ok(answer) => {
                println!("\n{}\n{}", "🤖 Final Response:".bold().green(), answer);
                println!("{}", "---------------------------------------------------------\n".dimmed());
            }
            Err(err) => {
                println!("\n{} {}\n", "❌ Error during execution:".bold().red(), err);
            }
        }
    }

    Ok(())
}
