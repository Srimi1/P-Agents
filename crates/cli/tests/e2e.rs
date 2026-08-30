//! Full-stack tests: config -> runtime -> gated dispatcher -> lead agent ->
//! sub-agent -> real filesystem tool -> session transcript. The model is
//! scripted, so nothing here touches a network.

use agent_core::{LlmProvider, LlmResponse, MockProvider, TokenUsage, ToolCall};
use cli::app::{AppOptions, HarnessApp};
use cli::config::HarnessConfig;
use runtime::{ApprovalDecision, ApprovalRequest, SessionRecord, SessionStore};
use serde_json::json;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;

fn call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
    }
}

fn tool_turn(calls: Vec<ToolCall>) -> LlmResponse {
    LlmResponse {
        content: None,
        tool_calls: calls,
        usage: Some(TokenUsage::new(100, 20)),
        stop_reason: Some("tool_use".to_string()),
    }
}

fn text_turn(text: &str) -> LlmResponse {
    LlmResponse {
        content: Some(text.to_string()),
        tool_calls: Vec::new(),
        usage: Some(TokenUsage::new(120, 15)),
        stop_reason: Some("end_turn".to_string()),
    }
}

/// Lead delegates to an engineer, the engineer writes a file and reports back,
/// then the lead summarizes. Four model turns in delegation order.
fn delegation_script(artifact: &Path) -> Vec<LlmResponse> {
    vec![
        tool_turn(vec![call(
            "c1",
            "spawn_subagent",
            json!({
                "role": "engineer",
                "task": format!("Write 'harness end to end ok' to {}", artifact.display())
            }),
        )]),
        tool_turn(vec![call(
            "c2",
            "write_file",
            json!({ "path": artifact.to_string_lossy(), "content": "harness end to end ok\n" }),
        )]),
        text_turn("File written."),
        text_turn("The engineer created the file as requested."),
    ]
}

async fn build(
    responses: Vec<LlmResponse>,
    session_dir: &Path,
    yolo: bool,
) -> (
    HarnessApp,
    mpsc::Receiver<ApprovalRequest>,
    Arc<MockProvider>,
) {
    let provider = Arc::new(MockProvider::new(responses));
    let options = AppOptions {
        yolo,
        session_dir: session_dir.to_path_buf(),
        ..Default::default()
    };
    let mut config = HarnessConfig::default();
    // The tools are confined to the working directory by default, and these
    // tests operate in a tempdir. Widen the root to that tempdir rather than
    // disabling containment, so the sandbox stays exercised.
    config.permissions.workspace_roots = vec![session_dir
        .parent()
        .expect("session dir has a parent")
        .to_path_buf()];
    // The default auto-approve list covers read-only tools only, so write_file
    // still faces the gate unless yolo is set.
    let (app, approvals) = HarnessApp::with_provider(
        config,
        options,
        Arc::clone(&provider) as Arc<dyn LlmProvider>,
    )
    .await
    .expect("app should assemble");
    (app, approvals, provider)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_delegation_run_writes_the_file_and_records_the_session() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let sessions = workdir.path().join("sessions");
    let artifact = workdir.path().join("artifact.txt");

    let (mut app, _approvals, provider) =
        build(delegation_script(&artifact), &sessions, true).await;
    let session_path = app.runtime.session_path().to_path_buf();

    let answer = app.run_prompt("Create the artifact").await.expect("run");
    assert!(
        answer.contains("created the file"),
        "unexpected answer: {answer}"
    );

    let written = std::fs::read_to_string(&artifact).expect("artifact should exist");
    assert_eq!(written, "harness end to end ok\n");

    // Four model turns: two for the lead, two for the sub-agent.
    assert_eq!(provider.call_count(), 4);

    app.shutdown().await;

    let records = SessionStore::load(&session_path)
        .await
        .expect("session log");
    assert!(
        matches!(records.first(), Some(SessionRecord::Meta { .. })),
        "first record must be meta"
    );

    let lead_history = SessionStore::rebuild_history(&records, "lead-planner");
    assert!(
        lead_history.iter().any(|m| m
            .content
            .as_deref()
            .is_some_and(|c| c.contains("Create the artifact"))),
        "the user prompt should be persisted"
    );

    let sub_history = SessionStore::rebuild_history(&records, "engineer-1");
    assert!(
        !sub_history.is_empty(),
        "the sub-agent's own transcript should be persisted separately"
    );

    let total: usize = records
        .iter()
        .filter_map(|r| match r {
            SessionRecord::Usage { usage, .. } => Some(usage.total_tokens),
            _ => None,
        })
        .sum();
    assert!(total > 0, "usage should be recorded");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_denied_write_never_touches_the_disk_and_the_model_is_told() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let sessions = workdir.path().join("sessions");
    let artifact = workdir.path().join("denied.txt");

    let script = vec![
        tool_turn(vec![call(
            "c1",
            "write_file",
            json!({ "path": artifact.to_string_lossy(), "content": "should never land" }),
        )]),
        text_turn("Understood, I will not write that file."),
    ];

    let (mut app, mut approvals, _provider) = build(script, &sessions, false).await;

    let seen = Arc::new(AtomicUsize::new(0));
    let seen_in_task = Arc::clone(&seen);
    let responder = tokio::spawn(async move {
        while let Some(request) = approvals.recv().await {
            seen_in_task.fetch_add(1, Ordering::SeqCst);
            assert_eq!(request.tool, "write_file");
            let _ = request.respond.send(ApprovalDecision::Deny);
        }
    });

    let answer = app.run_prompt("Write the file").await.expect("run");
    assert!(
        answer.contains("will not write"),
        "unexpected answer: {answer}"
    );
    assert_eq!(
        seen.load(Ordering::SeqCst),
        1,
        "the gate should have been consulted"
    );
    assert!(!artifact.exists(), "a denied write must not reach the disk");

    // The denial is fed back as a tool observation so the model can adapt.
    let denial = app
        .history()
        .iter()
        .find(|m| m.role == agent_core::Role::Tool)
        .and_then(|m| m.content.clone())
        .expect("a tool observation should exist");
    assert!(denial.starts_with("DENIED by user"), "got: {denial}");

    responder.abort();
    app.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_absent_approval_ui_fails_closed() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let sessions = workdir.path().join("sessions");
    let artifact = workdir.path().join("nobody-home.txt");

    let script = vec![
        tool_turn(vec![call(
            "c1",
            "write_file",
            json!({ "path": artifact.to_string_lossy(), "content": "nope" }),
        )]),
        text_turn("Acknowledged."),
    ];

    let (mut app, approvals, _provider) = build(script, &sessions, false).await;
    // No UI is listening at all.
    drop(approvals);

    app.run_prompt("Write the file").await.expect("run");
    assert!(
        !artifact.exists(),
        "with no approver reachable the write must be denied, not allowed"
    );
    app.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resume_restores_the_lead_transcript_into_a_fresh_app() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let sessions = workdir.path().join("sessions");

    let (mut first, _approvals, _provider) =
        build(vec![text_turn("Remembered.")], &sessions, true).await;
    let session_id = first.runtime.session_id().to_string();
    first
        .run_prompt("remember the number 8675309")
        .await
        .expect("run");
    first.shutdown().await;

    let (mut second, _approvals2, _provider2) =
        build(vec![text_turn("Still here.")], &sessions, true).await;
    let restored = second
        .resume(&sessions, &session_id)
        .await
        .expect("resume should find the session");
    assert!(
        restored >= 2,
        "expected the transcript back, got {restored} messages"
    );
    assert!(
        second
            .history()
            .iter()
            .any(|m| m.content.as_deref().is_some_and(|c| c.contains("8675309"))),
        "resumed history should contain the earlier prompt"
    );
    second.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn model_swap_preserves_history() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let sessions = workdir.path().join("sessions");

    let (mut app, _approvals, _provider) =
        build(vec![text_turn("first answer")], &sessions, true).await;
    app.run_prompt("first question").await.expect("run");
    let before = app.history().len();

    app.swap_model(Some("mock"), None)
        .expect("swap to the mock provider");
    assert_eq!(
        app.history().len(),
        before,
        "history must survive a model swap"
    );
    assert!(app
        .history()
        .iter()
        .any(|m| m.content.as_deref() == Some("first question")));
    app.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resuming_an_unknown_session_is_a_clean_error() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let sessions = workdir.path().join("sessions");
    let (mut app, _approvals, _provider) = build(vec![text_turn("hi")], &sessions, true).await;

    let err = app
        .resume(&sessions, "does-not-exist")
        .await
        .expect_err("resume should fail");
    assert!(err.to_string().contains("does-not-exist"), "got: {err}");
    app.shutdown().await;
}

/// A run killed between issuing a tool call and recording its result leaves an
/// assistant turn whose tool_calls have no answers. Sending that back is a 400
/// on both providers, which used to make the resumed session unusable forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resuming_an_interrupted_run_repairs_the_orphaned_tool_call() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let sessions = workdir.path().join("sessions");

    // Turn one issues a tool call; the provider script then runs out, standing
    // in for the process being killed before the observation was recorded.
    let script = vec![tool_turn(vec![call(
        "orphan_1",
        "read_file",
        json!({ "path": "somewhere.txt" }),
    )])];
    let (mut first, _a1, _p1) = build(script, &sessions, true).await;
    let session_id = first.runtime.session_id().to_string();
    let _ = first.run_prompt("start something").await;
    first.shutdown().await;

    let (mut second, _a2, _p2) = build(vec![text_turn("carrying on")], &sessions, true).await;
    second.resume(&sessions, &session_id).await.expect("resume");

    // Every tool call in the restored history now has a matching response.
    let answered: std::collections::HashSet<String> = second
        .history()
        .iter()
        .filter_map(|m| m.tool_call_id.clone())
        .collect();
    for message in second.history() {
        for tool_call in message.tool_calls.iter().flatten() {
            assert!(
                answered.contains(&tool_call.id),
                "tool call {} was left unanswered; the next request would 400",
                tool_call.id
            );
        }
    }

    // And the session is actually usable again.
    let answer = second.run_prompt("continue").await.expect("resumed run");
    assert_eq!(answer, "carrying on");
    second.shutdown().await;
}

/// Resuming used to swap history in without telling the session store, so the
/// new transcript began mid-conversation and resuming *it* lost the past.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_resumed_session_records_the_history_it_inherited() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let sessions = workdir.path().join("sessions");

    let (mut first, _a1, _p1) = build(vec![text_turn("noted")], &sessions, true).await;
    let first_id = first.runtime.session_id().to_string();
    first.run_prompt("remember 8675309").await.expect("run");
    first.shutdown().await;

    let (mut second, _a2, _p2) = build(vec![text_turn("ok")], &sessions, true).await;
    let second_path = second.runtime.session_path().to_path_buf();
    second.resume(&sessions, &first_id).await.expect("resume");
    second.shutdown().await;

    let records = SessionStore::load(&second_path).await.expect("session log");
    let history = SessionStore::rebuild_history(&records, "lead-planner");
    assert!(
        history
            .iter()
            .any(|m| m.content.as_deref().is_some_and(|c| c.contains("8675309"))),
        "the inherited transcript should be in the new session file too"
    );
}

/// The approval gate covers writes, but reads are never prompted, so path
/// containment is the only thing standing between an agent and the rest of the
/// disk. This proves it holds through the whole stack, not just in the tool.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_agent_cannot_reach_outside_its_workspace_even_with_yolo() {
    let workdir = tempfile::tempdir().expect("tempdir");
    let sessions = workdir.path().join("sessions");
    let outside = tempfile::tempdir().expect("outside");
    let secret = outside.path().join("secret.txt");
    std::fs::write(&secret, "sensitive").expect("write secret");
    let stolen = outside.path().join("stolen.txt");

    let script = vec![
        tool_turn(vec![call(
            "c1",
            "read_file",
            json!({ "path": secret.to_string_lossy() }),
        )]),
        tool_turn(vec![call(
            "c2",
            "write_file",
            json!({ "path": stolen.to_string_lossy(), "content": "exfiltrated" }),
        )]),
        text_turn("I could not reach those paths."),
    ];

    // yolo, so nothing is gated: containment is the only control left.
    let (mut app, _approvals, _provider) = build(script, &sessions, true).await;
    app.run_prompt("read the secret and copy it out")
        .await
        .expect("run");

    let observations: Vec<String> = app
        .history()
        .iter()
        .filter(|m| m.role == agent_core::Role::Tool)
        .filter_map(|m| m.content.clone())
        .collect();
    assert_eq!(
        observations.len(),
        2,
        "both calls should have been attempted"
    );
    for observation in &observations {
        assert!(
            observation.contains("outside the allowed workspace"),
            "the tool should have refused: {observation}"
        );
    }
    assert!(
        !observations.iter().any(|o| o.contains("sensitive")),
        "the secret's contents must never reach the transcript"
    );
    assert!(!stolen.exists(), "a refused write must not create the file");
    app.shutdown().await;
}
