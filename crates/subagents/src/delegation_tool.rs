use crate::persona_registry::PersonaRegistry;
use agent_core::compaction::HistoryCompactor;
use agent_core::events::{emit, AgentEvent, EventSink};
use agent_core::{Agent, LlmProvider, ToolDispatcher};
use anyhow::Result;
use async_trait::async_trait;
use harness_core::Tool;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::info;

/// Shared machinery for building and running an isolated sub-agent.
///
/// Context isolation lives here: a sub-agent is constructed with a fresh
/// history containing only its persona prompt and its own task. It never sees
/// the parent's transcript, and only its final answer travels back — as a
/// single tool observation.
pub struct SubAgentFactory {
    provider: Arc<dyn LlmProvider>,
    /// The gated dispatcher. Sub-agents get the same one as the lead, so their
    /// tool calls face the same approval policy. It deliberately excludes the
    /// delegation tools, which is what prevents unbounded recursion.
    dispatcher: Arc<dyn ToolDispatcher>,
    personas: Arc<PersonaRegistry>,
    events: Option<EventSink>,
    compactor: Option<Arc<dyn HistoryCompactor>>,
    parent_id: String,
    max_iterations: usize,
    counter: AtomicUsize,
}

impl SubAgentFactory {
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        dispatcher: Arc<dyn ToolDispatcher>,
        personas: Arc<PersonaRegistry>,
        parent_id: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            dispatcher,
            personas,
            events: None,
            compactor: None,
            parent_id: parent_id.into(),
            max_iterations: 20,
            counter: AtomicUsize::new(0),
        }
    }

    pub fn with_events(mut self, events: Option<EventSink>) -> Self {
        self.events = events;
        self
    }

    pub fn with_compactor(mut self, compactor: Option<Arc<dyn HistoryCompactor>>) -> Self {
        self.compactor = compactor;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    fn next_id(&self, role: &str) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed) + 1;
        format!("{}-{}", role, n)
    }

    /// Runs one sub-agent to completion and returns its answer.
    ///
    /// `streaming` is disabled for parallel batches so several agents' tokens
    /// don't interleave into an unreadable stream.
    async fn run_one(&self, role: &str, task: &str, streaming: bool) -> Result<String> {
        let persona = self
            .personas
            .get(role)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Unknown sub-agent role '{}'. Available roles: {}.",
                    role,
                    self.personas.roles().join(", ")
                )
            })?
            .clone();

        let agent_id = self.next_id(role);
        info!(role = %role, agent_id = %agent_id, "Spawning isolated sub-agent");
        emit(
            &self.events,
            AgentEvent::SubAgentSpawned {
                parent_id: self.parent_id.clone(),
                agent_id: agent_id.clone(),
                role: role.to_string(),
            },
        );

        let mut subagent = Agent::new(
            agent_id.clone(),
            persona.display_name.clone(),
            persona.prompt.clone(),
            Arc::clone(&self.provider),
            Arc::clone(&self.dispatcher),
        )
        .with_max_iterations(self.max_iterations)
        .with_streaming(streaming);

        if let Some(events) = &self.events {
            subagent = subagent.with_events(events.clone());
        }
        if let Some(compactor) = &self.compactor {
            subagent = subagent.with_compactor(Arc::clone(compactor));
        }

        let result = subagent.run(task).await;
        emit(
            &self.events,
            AgentEvent::SubAgentFinished {
                agent_id,
                ok: result.is_ok(),
            },
        );

        let answer = result?;
        Ok(format!(
            "[Sub-Agent ({}) Result]:\n{}",
            persona.display_name, answer
        ))
    }
}

/// Delegates one task to one specialist sub-agent.
pub struct SpawnSubAgentTool {
    factory: Arc<SubAgentFactory>,
    description: String,
}

impl SpawnSubAgentTool {
    pub fn new(factory: Arc<SubAgentFactory>) -> Self {
        let description = format!(
            "Spawns an isolated specialist sub-agent to execute a single subtask and returns its \
             final answer. The sub-agent starts with a clean context and sees only the task text \
             you give it, so the task must be self-contained. Roles available: {}.",
            factory.personas.roles().join(", ")
        );
        Self {
            factory,
            description,
        }
    }
}

#[async_trait]
impl Tool for SpawnSubAgentTool {
    fn name(&self) -> &str {
        "spawn_subagent"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": self.factory.personas.roles(),
                    "description": "The specialist role to spawn."
                },
                "task": {
                    "type": "string",
                    "description": "Self-contained task instructions. Include every file path and detail the sub-agent needs; it cannot see this conversation."
                }
            },
            "required": ["role", "task"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let role = args["role"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'role' parameter"))?;
        let task = args["task"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'task' parameter"))?;

        self.factory.run_one(role, task, true).await
    }
}

/// Fans several independent subtasks out concurrently.
pub struct RunParallelSubAgentsTool {
    factory: Arc<SubAgentFactory>,
    max_parallel: usize,
    description: String,
}

impl RunParallelSubAgentsTool {
    pub fn new(factory: Arc<SubAgentFactory>, max_parallel: usize) -> Self {
        let description = format!(
            "Runs several independent subtasks concurrently, each in its own isolated sub-agent, \
             and returns all their answers together. Use this instead of repeated spawn_subagent \
             calls when the subtasks do not depend on each other. Roles available: {}.",
            factory.personas.roles().join(", ")
        );
        Self {
            factory,
            max_parallel: max_parallel.max(1),
            description,
        }
    }
}

#[async_trait]
impl Tool for RunParallelSubAgentsTool {
    fn name(&self) -> &str {
        "run_parallel_subagents"
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": 1,
                    "description": "Independent subtasks to run at the same time.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": self.factory.personas.roles(),
                                "description": "The specialist role to spawn."
                            },
                            "task": {
                                "type": "string",
                                "description": "Self-contained task instructions for this sub-agent."
                            }
                        },
                        "required": ["role", "task"]
                    }
                }
            },
            "required": ["tasks"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let tasks = args["tasks"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Missing 'tasks' array parameter"))?;
        if tasks.is_empty() {
            anyhow::bail!("'tasks' must contain at least one subtask");
        }

        let mut specs = Vec::with_capacity(tasks.len());
        for (index, task) in tasks.iter().enumerate() {
            let role = task["role"].as_str().ok_or_else(|| {
                anyhow::anyhow!("Task {} is missing its 'role' parameter", index + 1)
            })?;
            let text = task["task"].as_str().ok_or_else(|| {
                anyhow::anyhow!("Task {} is missing its 'task' parameter", index + 1)
            })?;
            specs.push((index, role.to_string(), text.to_string()));
        }

        let permits = Arc::new(Semaphore::new(self.max_parallel));
        let mut set: JoinSet<(usize, String, Result<String>)> = JoinSet::new();

        for (index, role, task) in specs {
            let factory = Arc::clone(&self.factory);
            let permits = Arc::clone(&permits);
            set.spawn(async move {
                // The semaphore is never closed, so acquire only fails if we
                // close it ourselves; treat a failure as "run anyway".
                let _permit = permits.acquire_owned().await.ok();
                let result = factory.run_one(&role, &task, false).await;
                (index, role, result)
            });
        }

        let mut outcomes: Vec<Option<String>> = vec![None; tasks.len()];
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok((index, role, Ok(answer))) => {
                    outcomes[index] =
                        Some(format!("### Task {} ({})\n{}", index + 1, role, answer));
                }
                // A failed sub-agent is reported to the model as a result, not
                // propagated: the parent should be able to route around it.
                Ok((index, role, Err(err))) => {
                    outcomes[index] =
                        Some(format!("### Task {} ({}) FAILED\n{}", index + 1, role, err));
                }
                Err(join_err) => {
                    let slot = outcomes.iter().position(|o| o.is_none());
                    let message = format!("### A sub-agent task panicked\n{}", join_err);
                    match slot {
                        Some(index) => outcomes[index] = Some(message),
                        None => outcomes.push(Some(message)),
                    }
                }
            }
        }

        // Results are emitted in the order the caller listed them, not the order
        // they happened to finish, so the model can match answers to tasks.
        let merged = outcomes
            .into_iter()
            .map(|o| o.unwrap_or_else(|| "### Task produced no result".to_string()))
            .collect::<Vec<_>>()
            .join("\n\n");

        Ok(merged)
    }
}
