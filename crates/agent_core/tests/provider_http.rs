//! Drives both providers against a real loopback HTTP server.
//!
//! The unit tests cover wire-format conversion and SSE decoding in isolation.
//! This covers the seam between them: the request a provider actually puts on
//! the socket, and the response handling all the way back to an `LlmResponse`.
//! Nothing here reaches the network beyond 127.0.0.1.

use agent_core::providers::{AnthropicProvider, GenericOpenAiProvider};
use agent_core::{ChatMessage, LlmProvider, ToolDefinition};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

struct CapturedRequest {
    request_line: String,
    headers: Vec<(String, String)>,
    body: Value,
}

impl CapturedRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Serves exactly one request, then returns what it saw. `status` and `body`
/// are what it replies with.
async fn serve_once(
    status: &str,
    content_type: &str,
    body: String,
) -> (String, JoinHandle<CapturedRequest>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let status = status.to_string();
    let content_type = content_type.to_string();
    let reply = body;

    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept");

        // Read headers, then exactly Content-Length bytes of body.
        let mut buf = Vec::new();
        let header_end = loop {
            let mut chunk = [0u8; 1024];
            let n = socket.read(&mut chunk).await.expect("read");
            assert!(n > 0, "client closed before sending a complete request");
            buf.extend_from_slice(&chunk[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                break pos + 4;
            }
        };

        let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let mut lines = head.lines();
        let request_line = lines.next().unwrap_or_default().to_string();
        let headers: Vec<(String, String)> = lines
            .filter_map(|line| line.split_once(':'))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();

        let content_length: usize = headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
            .and_then(|(_, v)| v.parse().ok())
            .unwrap_or(0);

        while buf.len() < header_end + content_length {
            let mut chunk = [0u8; 1024];
            let n = socket.read(&mut chunk).await.expect("read body");
            assert!(n > 0, "client closed mid-body");
            buf.extend_from_slice(&chunk[..n]);
        }

        let body: Value = serde_json::from_slice(&buf[header_end..header_end + content_length])
            .unwrap_or(Value::Null);

        let head = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            reply.len()
        );
        let _ = socket.write_all(head.as_bytes()).await;
        let _ = socket.write_all(reply.as_bytes()).await;
        let _ = socket.flush().await;
        let _ = socket.shutdown().await;

        CapturedRequest {
            request_line,
            headers,
            body,
        }
    });

    (format!("http://{addr}"), handle)
}

fn sample_history() -> Vec<ChatMessage> {
    vec![
        ChatMessage::system("You are terse."),
        ChatMessage::user("say ok"),
    ]
}

fn sample_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "read_file".to_string(),
        description: "Reads a file".to_string(),
        parameters: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
    }]
}

#[tokio::test]
async fn anthropic_sends_a_well_formed_request_and_assembles_the_reply() {
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":11,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: ping\n",
        "data: {\"type\":\"ping\"}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    let (base_url, server) = serve_once("200 OK", "text/event-stream", sse.to_string()).await;
    let provider = AnthropicProvider::new("test-key", base_url, "claude-test").with_max_tokens(256);

    let response = provider
        .complete(&sample_history(), &sample_tools(), Some(0.3))
        .await
        .expect("complete should succeed");

    assert_eq!(response.content.as_deref(), Some("ok"));
    assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));
    let usage = response.usage.expect("usage should be reported");
    assert_eq!(usage.prompt_tokens, 11);
    assert_eq!(usage.completion_tokens, 3);

    let request = server.await.expect("server task");
    assert!(
        request.request_line.starts_with("POST /v1/messages "),
        "got: {}",
        request.request_line
    );
    assert_eq!(request.header("x-api-key"), Some("test-key"));
    assert_eq!(request.header("anthropic-version"), Some("2023-06-01"));

    // System prompts are hoisted out of the message array entirely.
    assert_eq!(request.body["system"], json!("You are terse."));
    assert_eq!(request.body["messages"].as_array().unwrap().len(), 1);
    assert_eq!(request.body["messages"][0]["role"], json!("user"));
    assert_eq!(request.body["max_tokens"], json!(256));
    assert_eq!(request.body["stream"], json!(true));
    // Anthropic names the schema field input_schema, not parameters.
    assert_eq!(request.body["tools"][0]["name"], json!("read_file"));
    assert!(request.body["tools"][0]["input_schema"].is_object());
    assert!(request.body["tools"][0]["parameters"].is_null());
}

#[tokio::test]
async fn openai_sends_a_well_formed_streaming_request_and_assembles_the_reply() {
    let sse = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"o\"}}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"k\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}}\n\n",
        "data: [DONE]\n\n",
    );

    let (base_url, server) = serve_once("200 OK", "text/event-stream", sse.to_string()).await;
    let provider = GenericOpenAiProvider::new("sk-test", base_url, "gpt-test");

    let response = provider
        .stream(&sample_history(), &sample_tools(), Some(0.2))
        .await
        .expect("stream should open");
    let response = drain(response).await;

    assert_eq!(response.content.as_deref(), Some("ok"));
    assert_eq!(response.stop_reason.as_deref(), Some("stop"));
    assert_eq!(response.usage.expect("usage").total_tokens, 9);

    let request = server.await.expect("server task");
    assert!(
        request.request_line.starts_with("POST /chat/completions "),
        "got: {}",
        request.request_line
    );
    assert_eq!(request.header("authorization"), Some("Bearer sk-test"));
    assert_eq!(request.body["stream"], json!(true));
    assert_eq!(request.body["stream_options"]["include_usage"], json!(true));
    // OpenAI keeps the system turn in the message array.
    assert_eq!(request.body["messages"][0]["role"], json!("system"));
    assert_eq!(request.body["tools"][0]["type"], json!("function"));
    assert!(request.body["tools"][0]["function"]["parameters"].is_object());
}

#[tokio::test]
async fn openai_replays_tool_calls_with_arguments_encoded_as_a_string() {
    let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let (base_url, server) = serve_once("200 OK", "text/event-stream", sse.to_string()).await;
    let provider = GenericOpenAiProvider::new("sk-test", base_url, "gpt-test");

    let history = vec![
        ChatMessage::user("read it"),
        ChatMessage::assistant_tool_calls(vec![agent_core::ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({"path": "src/main.rs"}),
        }]),
        ChatMessage::tool_response("call_1", "read_file", "fn main() {}"),
    ];

    let stream = provider.stream(&history, &[], None).await.expect("stream");
    drain(stream).await;

    let request = server.await.expect("server task");
    let call = &request.body["messages"][1]["tool_calls"][0];
    assert_eq!(call["type"], json!("function"));
    // The whole point: arguments go on the wire as a JSON *string*, not an object.
    let arguments = call["function"]["arguments"]
        .as_str()
        .expect("arguments must be a string on the wire");
    assert_eq!(
        serde_json::from_str::<Value>(arguments).expect("and it must parse"),
        json!({"path": "src/main.rs"})
    );
    assert_eq!(request.body["messages"][2]["role"], json!("tool"));
    assert_eq!(request.body["messages"][2]["tool_call_id"], json!("call_1"));
}

#[tokio::test]
async fn anthropic_surfaces_a_rejected_request_with_its_body() {
    let error_body =
        r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#;
    let (base_url, server) = serve_once(
        "429 Too Many Requests",
        "application/json",
        error_body.to_string(),
    )
    .await;
    let provider = AnthropicProvider::new("test-key", base_url, "claude-test");

    let err = provider
        .complete(&sample_history(), &[], None)
        .await
        .expect_err("a 429 must be an error");
    let message = err.to_string();
    assert!(
        message.contains("429"),
        "status should be reported: {message}"
    );
    assert!(
        message.contains("slow down"),
        "the server's explanation should reach the caller: {message}"
    );

    server.await.expect("server task");
}

#[tokio::test]
async fn openai_surfaces_a_rejected_request_with_its_body() {
    let error_body = r#"{"error":{"message":"model not found","type":"invalid_request_error"}}"#;
    let (base_url, server) =
        serve_once("404 Not Found", "application/json", error_body.to_string()).await;
    let provider = GenericOpenAiProvider::new("sk-test", base_url, "nope");

    // `LlmStream` is not Debug, so unwrap the Result by hand.
    let message = match provider.stream(&sample_history(), &[], None).await {
        Ok(_) => panic!("a 404 must be an error"),
        Err(err) => err.to_string(),
    };
    assert!(message.contains("404"), "got: {message}");
    assert!(message.contains("model not found"), "got: {message}");

    server.await.expect("server task");
}

/// A server that reports no usage at all, which is what Ollama and older
/// OpenAI-compatible servers do.
#[tokio::test]
async fn openai_tolerates_a_server_that_never_reports_usage() {
    let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
    let (base_url, server) = serve_once("200 OK", "text/event-stream", sse.to_string()).await;
    // An empty key stands in for a local server that issues none.
    let provider = GenericOpenAiProvider::new("", base_url, "llama");

    let stream = provider
        .stream(&sample_history(), &[], None)
        .await
        .expect("stream");
    let response = drain(stream).await;
    assert_eq!(response.content.as_deref(), Some("hi"));
    assert!(
        response.usage.is_none(),
        "absent usage must not be invented"
    );

    let request = server.await.expect("server task");
    assert!(
        request.header("authorization").is_none(),
        "no bearer header should be sent when there is no key"
    );
}

async fn drain(mut stream: agent_core::LlmStream) -> agent_core::LlmResponse {
    use futures::StreamExt;
    let mut done = None;
    while let Some(event) = stream.next().await {
        if let agent_core::StreamEvent::Done(response) = event.expect("stream event") {
            done = Some(*response);
        }
    }
    done.expect("the stream must end with a Done event")
}
