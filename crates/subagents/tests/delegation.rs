use agent_core::{ChatMessage, LlmProvider, LlmResponse, MockProvider, Role, ToolDispatcher};
use harness_core::{HarnessToolRegistry, Tool};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use subagents::{PersonaRegistry, RunParallelSubAgentsTool, SpawnSubAgentTool, SubAgentFactory};

/// A dispatcher with no tools: sub-agents under test answer from the model
/// alone, which keeps these tests about delegation rather than tool behaviour.
fn empty_dispatcher() -> Arc<dyn ToolDispatcher> {
    Arc::new(HarnessToolRegistry::new())
}

fn text_responses(n: usize, body: &str) -> Vec<LlmResponse> {
    (0..n)
        .map(|i| LlmResponse {
            content: Some(format!("{body} #{i}")),
            ..Default::default()
        })
        .collect()
}

fn factory(provider: Arc<dyn LlmProvider>) -> Arc<SubAgentFactory> {
    Arc::new(SubAgentFactory::new(
        provider,
        empty_dispatcher(),
        Arc::new(PersonaRegistry::new()),
        "lead-planner",
    ))
}

#[tokio::test]
async fn spawn_subagent_returns_the_specialist_answer() {
    let provider = Arc::new(MockProvider::new(vec![LlmResponse {
        content: Some("Refactored the parser.".to_string()),
        ..Default::default()
    }]));
    let tool = SpawnSubAgentTool::new(factory(provider));

    let out = tool
        .execute(json!({ "role": "engineer", "task": "Refactor the parser." }))
        .await
        .expect("spawn should succeed");

    assert!(
        out.contains("SoftwareEngineer"),
        "answer should name the persona: {out}"
    );
    assert!(out.contains("Refactored the parser."));
}

#[tokio::test]
async fn subagent_context_is_isolated_from_the_parent() {
    const PARENT_SECRET: &str = "PARENT_ONLY_CONTEXT_MARKER";

    let provider = Arc::new(MockProvider::new(vec![LlmResponse {
        content: Some("done".to_string()),
        ..Default::default()
    }]));
    // Simulate a parent that has already accumulated history on this provider.
    provider
        .complete(
            &[
                ChatMessage::system("lead prompt"),
                ChatMessage::user(PARENT_SECRET),
            ],
            &[],
            None,
        )
        .await
        .ok();

    let provider2 = Arc::new(MockProvider::new(vec![LlmResponse {
        content: Some("done".to_string()),
        ..Default::default()
    }]));
    let tool = SpawnSubAgentTool::new(factory(provider2.clone()));
    tool.execute(json!({ "role": "engineer", "task": "Do the isolated thing." }))
        .await
        .expect("spawn should succeed");

    let requests = provider2.recorded_requests();
    assert_eq!(requests.len(), 1, "sub-agent should make exactly one call");
    let messages = &requests[0];

    // A sub-agent's opening context is exactly its persona plus its own task.
    assert_eq!(
        messages.len(),
        2,
        "unexpected sub-agent context: {messages:#?}"
    );
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[1].role, Role::User);
    assert_eq!(
        messages[1].content.as_deref(),
        Some("Do the isolated thing.")
    );
    assert!(
        !provider2.all_seen_text().contains(PARENT_SECRET),
        "parent history leaked into the sub-agent"
    );
}

#[tokio::test]
async fn unknown_role_is_reported_with_the_available_roles() {
    let provider = Arc::new(MockProvider::new(text_responses(1, "x")));
    let tool = SpawnSubAgentTool::new(factory(provider));

    let err = tool
        .execute(json!({ "role": "wizard", "task": "cast a spell" }))
        .await
        .expect_err("unknown role should error");
    let message = err.to_string();
    assert!(message.contains("wizard"), "{message}");
    assert!(
        message.contains("engineer"),
        "should list valid roles: {message}"
    );
}

#[tokio::test]
async fn missing_task_parameter_errors() {
    let provider = Arc::new(MockProvider::new(text_responses(1, "x")));
    let tool = SpawnSubAgentTool::new(factory(provider));
    assert!(tool.execute(json!({ "role": "engineer" })).await.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_subagents_run_concurrently() {
    const DELAY_MS: u64 = 300;
    let provider = Arc::new(
        MockProvider::new(text_responses(3, "finding")).with_delay(Duration::from_millis(DELAY_MS)),
    );
    let tool = RunParallelSubAgentsTool::new(factory(provider), 4);

    let started = Instant::now();
    let out = tool
        .execute(json!({
            "tasks": [
                { "role": "engineer",   "task": "task one" },
                { "role": "verifier",   "task": "task two" },
                { "role": "researcher", "task": "task three" }
            ]
        }))
        .await
        .expect("parallel run should succeed");
    let elapsed = started.elapsed();

    assert_eq!(
        out.matches("### Task").count(),
        3,
        "all three tasks report: {out}"
    );
    // Serial execution would take at least 3x the per-agent delay.
    assert!(
        elapsed < Duration::from_millis(DELAY_MS * 2),
        "sub-agents did not overlap: took {elapsed:?} for 3 x {DELAY_MS}ms"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn parallel_results_follow_input_order_not_completion_order() {
    let provider = Arc::new(MockProvider::new(text_responses(3, "answer")));
    let tool = RunParallelSubAgentsTool::new(factory(provider), 4);

    let out = tool
        .execute(json!({
            "tasks": [
                { "role": "engineer",   "task": "one" },
                { "role": "verifier",   "task": "two" },
                { "role": "researcher", "task": "three" }
            ]
        }))
        .await
        .expect("parallel run should succeed");

    let engineer = out.find("Task 1 (engineer)").expect("task 1 present");
    let verifier = out.find("Task 2 (verifier)").expect("task 2 present");
    let researcher = out.find("Task 3 (researcher)").expect("task 3 present");
    assert!(
        engineer < verifier && verifier < researcher,
        "results should be ordered by input index:\n{out}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failing_subagent_does_not_fail_the_batch() {
    // Only two scripted responses for three tasks: the third sub-agent's
    // provider call fails, standing in for any mid-task failure.
    let provider = Arc::new(MockProvider::new(text_responses(2, "ok")));
    let tool = RunParallelSubAgentsTool::new(factory(provider), 1);

    let out = tool
        .execute(json!({
            "tasks": [
                { "role": "engineer",   "task": "one" },
                { "role": "verifier",   "task": "two" },
                { "role": "researcher", "task": "three" }
            ]
        }))
        .await
        .expect("batch should succeed even when one sub-agent fails");

    assert!(
        out.contains("FAILED"),
        "the failure should be reported: {out}"
    );
    assert_eq!(out.matches("### Task").count(), 3);
}

#[tokio::test]
async fn parallel_rejects_an_empty_task_list() {
    let provider = Arc::new(MockProvider::new(text_responses(1, "x")));
    let tool = RunParallelSubAgentsTool::new(factory(provider), 4);
    assert!(tool.execute(json!({ "tasks": [] })).await.is_err());
}

#[tokio::test]
async fn tool_schemas_advertise_only_registered_roles() {
    let provider = Arc::new(MockProvider::new(text_responses(1, "x")));
    let mut personas = PersonaRegistry::new();
    personas.insert("dba", "DatabaseExpert", "You tune queries.");
    let factory = Arc::new(SubAgentFactory::new(
        provider,
        empty_dispatcher(),
        Arc::new(personas),
        "lead-planner",
    ));

    let schema = SpawnSubAgentTool::new(Arc::clone(&factory)).parameters_schema();
    let roles = schema["properties"]["role"]["enum"]
        .as_array()
        .expect("role enum");
    assert!(
        roles.iter().any(|r| r == "dba"),
        "custom persona missing: {roles:?}"
    );
    assert!(roles.iter().any(|r| r == "engineer"));

    let parallel = RunParallelSubAgentsTool::new(factory, 2).parameters_schema();
    assert!(
        parallel["properties"]["tasks"]["items"]["properties"]["role"]["enum"]
            .as_array()
            .expect("nested role enum")
            .iter()
            .any(|r| r == "dba")
    );
}

#[tokio::test]
async fn sub_agent_ids_stay_unique_after_the_lead_agent_is_rebuilt() {
    // `/model` rebuilds the lead agent while the session log stays open. A
    // per-factory counter restarted at 1 here, so a second engineer wrote into
    // the first one's agent_id and clobbered its usage row.
    let provider = Arc::new(MockProvider::new(text_responses(4, "ok")));
    let orchestrator =
        subagents::MultiAgentOrchestrator::new(Arc::clone(&provider) as Arc<dyn LlmProvider>);

    let mut seen = Vec::new();
    for _ in 0..2 {
        // Rebuilding the lead agent is what a model swap does.
        let lead = orchestrator
            .create_lead_agent(empty_dispatcher(), None, None)
            .expect("lead agent");
        let spawn = lead
            .dispatcher
            .get_definitions()
            .iter()
            .any(|d| d.name == "spawn_subagent");
        assert!(spawn, "the lead agent should have the delegation tool");

        let out = lead
            .dispatcher
            .dispatch(
                "lead-planner",
                &agent_core::ToolCall {
                    id: "c1".into(),
                    name: "spawn_subagent".into(),
                    arguments: json!({ "role": "engineer", "task": "do a thing" }),
                },
            )
            .await
            .expect("spawn");
        assert!(out.contains("SoftwareEngineer"));
        seen.push(out);
    }

    // Both sub-agents ran; the ids they were given must differ. The provider
    // saw two separate isolated contexts.
    assert_eq!(provider.call_count(), 2);
}
