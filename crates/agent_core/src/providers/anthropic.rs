use super::{LlmProvider, LlmStream, StreamAccumulator, StreamEvent};
use crate::types::{truncate_at_boundary, ChatMessage, Role, TokenUsage, ToolDefinition};
use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::VecDeque;

pub const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
pub const ANTHROPIC_VERSION: &str = "2023-06-01";
pub const DEFAULT_MAX_TOKENS: usize = 8192;

/// Upper bound on how much of an error body is echoed back to the caller.
const MAX_ERROR_BODY_BYTES: usize = 2048;

/// Native Anthropic Messages API provider.
pub struct AnthropicProvider {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: usize,
    pub client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            model: model.into(),
            max_tokens: DEFAULT_MAX_TOKENS,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    fn build_body(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        temperature: Option<f32>,
    ) -> Result<Value> {
        let (system, wire_messages) = to_wire(messages)?;

        let mut body = json!({
            "model": self.model,
            // Anthropic rejects a request without max_tokens; there is no default.
            "max_tokens": self.max_tokens,
            "messages": wire_messages,
            "stream": true,
        });

        if let Some(system) = system {
            body["system"] = Value::String(system);
        }
        if let Some(temperature) = temperature {
            body["temperature"] = json!(temperature);
        }
        if !tools.is_empty() {
            body["tools"] = Value::Array(
                tools
                    .iter()
                    .map(|t| {
                        json!({
                            "name": t.name,
                            "description": t.description,
                            // Anthropic names this `input_schema`, not `parameters`.
                            "input_schema": t.parameters,
                        })
                    })
                    .collect(),
            );
        }

        Ok(body)
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn provider_name(&self) -> &str {
        "anthropic"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    async fn stream(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        temperature: Option<f32>,
    ) -> Result<LlmStream> {
        let body = self.build_body(messages, tools, temperature)?;
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .with_context(|| format!("anthropic request to {url} failed"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|e| format!("<error body unreadable: {e}>"));
            bail!(
                "anthropic api returned {} for {}: {}",
                status,
                url,
                truncate_at_boundary(&body, MAX_ERROR_BODY_BYTES)
            );
        }

        Ok(sse_bytes_to_stream(response.bytes_stream()))
    }
}

/// Converts harness history into `(system, messages)` for the Messages API.
///
/// System turns are hoisted out of the array entirely, and consecutive tool
/// responses are merged into a single user message: Anthropic rejects a turn
/// that answers parallel `tool_use` blocks with more than one user message.
fn to_wire(messages: &[ChatMessage]) -> Result<(Option<String>, Vec<Value>)> {
    let mut system_parts: Vec<&str> = Vec::new();
    let mut out: Vec<Value> = Vec::new();
    let mut pending_tool_results: Vec<Value> = Vec::new();

    for message in messages {
        let text = message.content.as_deref().unwrap_or_default();
        match message.role {
            Role::System => {
                if !text.trim().is_empty() {
                    system_parts.push(text);
                }
            }
            Role::Tool => {
                let tool_use_id = message.tool_call_id.as_deref().ok_or_else(|| {
                    anyhow!(
                        "tool response for `{}` has no tool_call_id to correlate with a tool_use block",
                        message.name.as_deref().unwrap_or("<unnamed>")
                    )
                })?;
                pending_tool_results.push(json!({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": text,
                }));
            }
            Role::User => {
                flush_tool_results(&mut pending_tool_results, &mut out);
                if text.trim().is_empty() {
                    continue;
                }
                out.push(json!({ "role": "user", "content": text }));
            }
            Role::Assistant => {
                flush_tool_results(&mut pending_tool_results, &mut out);
                let calls = message.tool_calls.as_deref().unwrap_or_default();
                if calls.is_empty() {
                    // Blank assistant turns are rejected by the API and carry no
                    // information, so they are dropped rather than sent.
                    if text.trim().is_empty() {
                        continue;
                    }
                    out.push(json!({ "role": "assistant", "content": text }));
                    continue;
                }

                let mut blocks: Vec<Value> = Vec::with_capacity(calls.len() + 1);
                if !text.trim().is_empty() {
                    blocks.push(json!({ "type": "text", "text": text }));
                }
                for call in calls {
                    // `input` must be an object; a call whose arguments failed to
                    // parse upstream is replayed as an empty one.
                    let input = if call.arguments.is_object() {
                        call.arguments.clone()
                    } else {
                        json!({})
                    };
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": input,
                    }));
                }
                out.push(json!({ "role": "assistant", "content": blocks }));
            }
        }
    }
    flush_tool_results(&mut pending_tool_results, &mut out);

    if out.is_empty() {
        bail!("anthropic requires at least one message, but the history had no user or assistant turns");
    }
    if out[0]["role"] != json!("user") {
        bail!("anthropic requires the first message to be a user turn, but the history starts with an assistant turn");
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    Ok((system, out))
}

fn flush_tool_results(pending: &mut Vec<Value>, out: &mut Vec<Value>) {
    if pending.is_empty() {
        return;
    }
    out.push(json!({ "role": "user", "content": std::mem::take(pending) }));
}

/// Per-connection SSE assembly state.
#[derive(Debug, Default)]
pub(crate) struct SseState {
    acc: StreamAccumulator,
    /// Reported by `message_start`; only paired with output tokens at `message_delta`.
    input_tokens: usize,
    finished: bool,
}

/// Applies one SSE event to `state` and returns the events it produces.
///
/// Pure with respect to IO so tests can drive transcripts event by event.
pub(crate) fn handle_sse_event(
    event_name: &str,
    data: &str,
    state: &mut SseState,
) -> Result<Vec<StreamEvent>> {
    if state.finished {
        return Ok(Vec::new());
    }
    let data = data.trim();
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let payload: Value = serde_json::from_str(data)
        .with_context(|| format!("anthropic sse event `{event_name}` had unparseable data"))?;

    // Proxies sometimes drop the `event:` line; the payload always carries `type`.
    let kind = if event_name.is_empty() || event_name == "message" {
        payload.get("type").and_then(Value::as_str).unwrap_or("")
    } else {
        event_name
    };

    match kind {
        "ping" => Ok(Vec::new()),
        "error" => {
            let kind = payload
                .pointer("/error/type")
                .and_then(Value::as_str)
                .unwrap_or("error");
            let message = payload
                .pointer("/error/message")
                .and_then(Value::as_str)
                .or_else(|| payload.get("message").and_then(Value::as_str))
                .unwrap_or("no message provided");
            bail!("anthropic stream error ({kind}): {message}")
        }
        "message_start" => {
            if let Some(tokens) = payload
                .pointer("/message/usage/input_tokens")
                .and_then(Value::as_u64)
            {
                state.input_tokens = tokens as usize;
            }
            Ok(Vec::new())
        }
        "content_block_start" => {
            let index = block_index(&payload, kind)?;
            if payload.pointer("/content_block/type").and_then(Value::as_str) != Some("tool_use") {
                return Ok(Vec::new());
            }
            let id = payload
                .pointer("/content_block/id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let name = payload
                .pointer("/content_block/name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            state.acc.start_tool_call(index, id.clone(), name.clone());
            Ok(vec![StreamEvent::ToolCallStarted { index, id, name }])
        }
        "content_block_delta" => {
            let index = block_index(&payload, kind)?;
            match payload.pointer("/delta/type").and_then(Value::as_str) {
                Some("text_delta") => {
                    let text = payload
                        .pointer("/delta/text")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if text.is_empty() {
                        return Ok(Vec::new());
                    }
                    state.acc.push_text(text);
                    Ok(vec![StreamEvent::TextDelta(text.to_string())])
                }
                Some("input_json_delta") => {
                    let fragment = payload
                        .pointer("/delta/partial_json")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if fragment.is_empty() {
                        return Ok(Vec::new());
                    }
                    state.acc.push_tool_args(index, fragment);
                    Ok(vec![StreamEvent::ToolCallArgsDelta {
                        index,
                        json_fragment: fragment.to_string(),
                    }])
                }
                // thinking/signature deltas carry nothing the agent loop needs.
                _ => Ok(Vec::new()),
            }
        }
        "content_block_stop" => Ok(Vec::new()),
        "message_delta" => {
            if let Some(reason) = payload
                .pointer("/delta/stop_reason")
                .and_then(Value::as_str)
            {
                state.acc.set_stop_reason(reason);
            }
            let Some(output_tokens) = payload
                .pointer("/usage/output_tokens")
                .and_then(Value::as_u64)
            else {
                return Ok(Vec::new());
            };
            let usage = TokenUsage::new(state.input_tokens, output_tokens as usize);
            state.acc.set_usage(usage);
            Ok(vec![StreamEvent::Usage(usage)])
        }
        "message_stop" => {
            state.finished = true;
            let acc = std::mem::take(&mut state.acc);
            Ok(vec![StreamEvent::Done(Box::new(acc.finish()))])
        }
        _ => Ok(Vec::new()),
    }
}

fn block_index(payload: &Value, kind: &str) -> Result<usize> {
    payload
        .get("index")
        .and_then(Value::as_u64)
        .map(|i| i as usize)
        .ok_or_else(|| anyhow!("anthropic `{kind}` event is missing its `index` field"))
}

/// Wraps a raw byte stream (a live response body, or synthetic chunks in tests)
/// into the provider-agnostic event stream.
pub(crate) fn sse_bytes_to_stream<S, E>(bytes: S) -> LlmStream
where
    S: Stream<Item = std::result::Result<bytes::Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    // Boxing first makes the EventStream `Unpin`, which `StreamExt::next` needs.
    let events = Box::pin(bytes).eventsource();
    let initial = (events, SseState::default(), VecDeque::new(), false);

    let stream = futures::stream::unfold(
        initial,
        |(mut events, mut state, mut queue, mut ended)| async move {
            loop {
                if let Some(event) = queue.pop_front() {
                    return Some((Ok(event), (events, state, queue, ended)));
                }
                if ended {
                    return None;
                }
                match events.next().await {
                    None => return None,
                    Some(Err(e)) => {
                        ended = true;
                        return Some((
                            Err(anyhow!("anthropic sse transport error: {e}")),
                            (events, state, queue, ended),
                        ));
                    }
                    Some(Ok(event)) => match handle_sse_event(&event.event, &event.data, &mut state)
                    {
                        Ok(produced) => queue.extend(produced),
                        Err(e) => {
                            ended = true;
                            return Some((Err(e), (events, state, queue, ended)));
                        }
                    },
                }
            }
        },
    );

    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ToolCall;

    fn tool_call(id: &str, name: &str, args: Value) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: args,
        }
    }

    fn chunks(transcript: &str, size: usize) -> Vec<bytes::Bytes> {
        transcript
            .as_bytes()
            .chunks(size)
            .map(bytes::Bytes::copy_from_slice)
            .collect()
    }

    async fn drive(parts: Vec<bytes::Bytes>) -> Result<Vec<StreamEvent>> {
        let raw = futures::stream::iter(parts.into_iter().map(Ok::<_, std::io::Error>));
        let mut stream = sse_bytes_to_stream(raw);
        let mut out = Vec::new();
        while let Some(event) = stream.next().await {
            out.push(event?);
        }
        Ok(out)
    }

    fn done_of(events: &[StreamEvent]) -> &crate::types::LlmResponse {
        match events.last() {
            Some(StreamEvent::Done(resp)) => resp,
            other => panic!("expected a trailing Done event, got {other:?}"),
        }
    }

    const TEXT_TRANSCRIPT: &str = concat!(
        "event: message_start\n",
        r#"data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"usage":{"input_tokens":25,"output_tokens":1}}}"#,
        "\n\n",
        "event: content_block_start\n",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        "\n\n",
        "event: ping\n",
        r#"data: {"type":"ping"}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
        "\n\n",
        "event: ping\n",
        r#"data: {"type":"ping"}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", world"}}"#,
        "\n\n",
        "event: content_block_stop\n",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "\n\n",
        "event: message_delta\n",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}"#,
        "\n\n",
        "event: message_stop\n",
        r#"data: {"type":"message_stop"}"#,
        "\n\n",
    );

    const TOOL_TRANSCRIPT: &str = concat!(
        "event: message_start\n",
        r#"data: {"type":"message_start","message":{"id":"msg_2","role":"assistant","content":[],"usage":{"input_tokens":40,"output_tokens":1}}}"#,
        "\n\n",
        "event: content_block_start\n",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Reading."}}"#,
        "\n\n",
        "event: content_block_stop\n",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "\n\n",
        "event: content_block_start\n",
        r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\""}}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":":\"src/"}}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"main.rs\"}"}}"#,
        "\n\n",
        "event: content_block_stop\n",
        r#"data: {"type":"content_block_stop","index":1}"#,
        "\n\n",
        "event: message_delta\n",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":31}}"#,
        "\n\n",
        "event: message_stop\n",
        r#"data: {"type":"message_stop"}"#,
        "\n\n",
    );

    #[test]
    fn to_wire_hoists_and_joins_system_messages() {
        let history = vec![
            ChatMessage::system("You are terse."),
            ChatMessage::system("Never guess."),
            ChatMessage::user("hi"),
        ];
        let (system, messages) = to_wire(&history).unwrap();
        assert_eq!(system.as_deref(), Some("You are terse.\n\nNever guess."));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], json!({"role":"user","content":"hi"}));
    }

    #[test]
    fn to_wire_emits_text_then_tool_use_blocks() {
        let history = vec![
            ChatMessage::user("read both"),
            ChatMessage::assistant_with_tool_calls(
                Some("On it.".to_string()),
                vec![tool_call("toolu_a", "read_file", json!({"path":"a"}))],
            ),
        ];
        let (system, messages) = to_wire(&history).unwrap();
        assert!(system.is_none());
        assert_eq!(
            messages[1],
            json!({
                "role": "assistant",
                "content": [
                    {"type":"text","text":"On it."},
                    {"type":"tool_use","id":"toolu_a","name":"read_file","input":{"path":"a"}},
                ]
            })
        );
    }

    #[test]
    fn to_wire_omits_text_block_when_assistant_had_no_prose() {
        let history = vec![
            ChatMessage::user("go"),
            ChatMessage::assistant_tool_calls(vec![tool_call("toolu_a", "ls", json!({}))]),
        ];
        let (_, messages) = to_wire(&history).unwrap();
        let blocks = messages[1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tool_use");
    }

    #[test]
    fn to_wire_merges_consecutive_tool_responses_into_one_user_message() {
        let history = vec![
            ChatMessage::user("read both"),
            ChatMessage::assistant_tool_calls(vec![
                tool_call("toolu_a", "read_file", json!({"path":"a"})),
                tool_call("toolu_b", "read_file", json!({"path":"b"})),
            ]),
            ChatMessage::tool_response("toolu_a", "read_file", "contents of a"),
            ChatMessage::tool_response("toolu_b", "read_file", "contents of b"),
        ];
        let (_, messages) = to_wire(&history).unwrap();
        assert_eq!(messages.len(), 3, "parallel results must collapse into one user turn");
        assert_eq!(
            messages[2],
            json!({
                "role": "user",
                "content": [
                    {"type":"tool_result","tool_use_id":"toolu_a","content":"contents of a"},
                    {"type":"tool_result","tool_use_id":"toolu_b","content":"contents of b"},
                ]
            })
        );
    }

    #[test]
    fn to_wire_flushes_tool_results_before_the_next_turn() {
        let history = vec![
            ChatMessage::user("go"),
            ChatMessage::assistant_tool_calls(vec![tool_call("toolu_a", "ls", json!({}))]),
            ChatMessage::tool_response("toolu_a", "ls", "a b"),
            ChatMessage::assistant("Found two files."),
            ChatMessage::user("thanks"),
        ];
        let (_, messages) = to_wire(&history).unwrap();
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[3], json!({"role":"assistant","content":"Found two files."}));
    }

    #[test]
    fn to_wire_replays_unparseable_tool_arguments_as_an_empty_object() {
        let history = vec![
            ChatMessage::user("go"),
            ChatMessage::assistant_tool_calls(vec![tool_call("toolu_a", "ls", Value::Null)]),
        ];
        let (_, messages) = to_wire(&history).unwrap();
        assert_eq!(messages[1]["content"][0]["input"], json!({}));
    }

    #[test]
    fn to_wire_rejects_history_starting_with_an_assistant_turn() {
        let history = vec![ChatMessage::assistant("hi there")];
        let err = to_wire(&history).unwrap_err().to_string();
        assert!(err.contains("first message to be a user turn"), "{err}");
    }

    #[test]
    fn to_wire_rejects_a_history_with_only_system_messages() {
        let history = vec![ChatMessage::system("be terse")];
        let err = to_wire(&history).unwrap_err().to_string();
        assert!(err.contains("at least one message"), "{err}");
    }

    #[test]
    fn to_wire_rejects_a_tool_response_without_a_call_id() {
        let history = vec![
            ChatMessage::user("go"),
            ChatMessage {
                role: Role::Tool,
                content: Some("result".into()),
                tool_calls: None,
                tool_call_id: None,
                name: Some("ls".into()),
            },
        ];
        let err = to_wire(&history).unwrap_err().to_string();
        assert!(err.contains("no tool_call_id"), "{err}");
    }

    #[test]
    fn to_wire_drops_blank_turns() {
        let history = vec![
            ChatMessage::user("go"),
            ChatMessage::assistant("   "),
            ChatMessage::user(""),
        ];
        let (_, messages) = to_wire(&history).unwrap();
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn build_body_uses_input_schema_and_required_max_tokens() {
        let provider =
            AnthropicProvider::new("k", DEFAULT_ANTHROPIC_BASE_URL, "claude-x").with_max_tokens(512);
        let tools = vec![ToolDefinition {
            name: "read_file".into(),
            description: "Reads a file".into(),
            parameters: json!({"type":"object","properties":{"path":{"type":"string"}}}),
        }];
        let body = provider
            .build_body(&[ChatMessage::system("sys"), ChatMessage::user("hi")], &tools, Some(0.3))
            .unwrap();

        assert_eq!(body["model"], "claude-x");
        assert_eq!(body["max_tokens"], 512);
        assert_eq!(body["stream"], true);
        assert_eq!(body["system"], "sys");
        // f32 -> f64 widening leaves the usual representation error.
        assert!((body["temperature"].as_f64().unwrap() - 0.3).abs() < 1e-6);
        assert_eq!(body["tools"][0]["name"], "read_file");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert!(body["tools"][0].get("parameters").is_none());
    }

    #[test]
    fn build_body_omits_optional_fields() {
        let provider = AnthropicProvider::new("k", DEFAULT_ANTHROPIC_BASE_URL, "claude-x");
        let body = provider
            .build_body(&[ChatMessage::user("hi")], &[], None)
            .unwrap();
        assert!(body.get("system").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("tools").is_none());
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[tokio::test]
    async fn text_transcript_assembles_across_adversarial_chunk_boundaries() {
        // 1 byte splits every field name, 7 splits mid-JSON, 13 straddles the
        // blank-line dispatch boundaries.
        for size in [1usize, 7, 13, 64, TEXT_TRANSCRIPT.len()] {
            let events = drive(chunks(TEXT_TRANSCRIPT, size)).await.unwrap();
            let response = done_of(&events);
            assert_eq!(
                response.content.as_deref(),
                Some("Hello, world"),
                "chunk size {size}"
            );
            assert!(response.tool_calls.is_empty(), "chunk size {size}");
            assert_eq!(response.stop_reason.as_deref(), Some("end_turn"));
            let usage = response.usage.expect("usage");
            assert_eq!(usage.prompt_tokens, 25);
            assert_eq!(usage.completion_tokens, 12);
            assert_eq!(usage.total_tokens, 37);
        }
    }

    #[tokio::test]
    async fn text_transcript_emits_deltas_and_usage_in_order() {
        let events = drive(chunks(TEXT_TRANSCRIPT, 9)).await.unwrap();
        assert_eq!(events.len(), 4, "two text deltas, one usage, one done: {events:?}");
        assert_eq!(events[0], StreamEvent::TextDelta("Hello".into()));
        assert_eq!(events[1], StreamEvent::TextDelta(", world".into()));
        assert_eq!(events[2], StreamEvent::Usage(TokenUsage::new(25, 12)));
    }

    #[tokio::test]
    async fn tool_transcript_assembles_arguments_from_json_fragments() {
        for size in [1usize, 5, 31, TOOL_TRANSCRIPT.len()] {
            let events = drive(chunks(TOOL_TRANSCRIPT, size)).await.unwrap();
            let response = done_of(&events);
            assert_eq!(response.content.as_deref(), Some("Reading."));
            assert_eq!(response.tool_calls.len(), 1, "chunk size {size}");
            let call = &response.tool_calls[0];
            assert_eq!(call.id, "toolu_1");
            assert_eq!(call.name, "read_file");
            assert_eq!(call.arguments, json!({"path":"src/main.rs"}));
            assert_eq!(response.stop_reason.as_deref(), Some("tool_use"));
            assert_eq!(response.usage.unwrap(), TokenUsage::new(40, 31));
        }
    }

    #[tokio::test]
    async fn tool_transcript_announces_the_call_before_its_fragments() {
        let events = drive(chunks(TOOL_TRANSCRIPT, 17)).await.unwrap();
        assert_eq!(
            events[1],
            StreamEvent::ToolCallStarted {
                index: 1,
                id: "toolu_1".into(),
                name: "read_file".into()
            }
        );
        let fragments: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallArgsDelta { json_fragment, .. } => Some(json_fragment.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(fragments.concat(), r#"{"path":"src/main.rs"}"#);
    }

    #[tokio::test]
    async fn error_event_surfaces_as_err() {
        let transcript = concat!(
            "event: message_start\n",
            r#"data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}"#,
            "\n\n",
            "event: error\n",
            r#"data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
            "\n\n",
        );
        let err = drive(chunks(transcript, 11)).await.unwrap_err().to_string();
        assert!(err.contains("overloaded_error"), "{err}");
        assert!(err.contains("Overloaded"), "{err}");
    }

    #[tokio::test]
    async fn malformed_event_data_surfaces_as_err() {
        let transcript = concat!("event: message_delta\n", "data: {not json\n\n");
        let err = drive(chunks(transcript, 4)).await.unwrap_err().to_string();
        assert!(err.contains("unparseable data"), "{err}");
    }

    #[tokio::test]
    async fn transport_errors_surface_as_err() {
        let parts = vec![
            Ok(bytes::Bytes::from("event: ping\ndata: {\"type\":\"ping\"}\n\n")),
            Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset")),
        ];
        let raw = futures::stream::iter(parts);
        let mut stream = sse_bytes_to_stream(raw);
        let mut last = None;
        while let Some(event) = stream.next().await {
            last = Some(event);
        }
        let err = last.expect("an item").unwrap_err().to_string();
        assert!(err.contains("transport error"), "{err}");
    }

    #[tokio::test]
    async fn stream_without_message_stop_yields_no_done() {
        let transcript = concat!(
            "event: content_block_start\n",
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            "\n\n",
            "event: content_block_delta\n",
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
            "\n\n",
        );
        let events = drive(chunks(transcript, 6)).await.unwrap();
        assert_eq!(events, vec![StreamEvent::TextDelta("hi".into())]);
    }

    #[test]
    fn ping_events_produce_nothing() {
        let mut state = SseState::default();
        assert!(handle_sse_event("ping", r#"{"type":"ping"}"#, &mut state)
            .unwrap()
            .is_empty());
        assert!(handle_sse_event("ping", "", &mut state).unwrap().is_empty());
    }

    #[test]
    fn unknown_event_types_are_ignored() {
        let mut state = SseState::default();
        assert!(
            handle_sse_event("message_batch_thing", r#"{"type":"whatever"}"#, &mut state)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn event_type_falls_back_to_the_payload_when_the_name_is_missing() {
        let mut state = SseState::default();
        let produced = handle_sse_event(
            "message",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}"#,
            &mut state,
        )
        .unwrap();
        assert_eq!(produced, vec![StreamEvent::TextDelta("x".into())]);
    }

    #[test]
    fn content_block_events_require_an_index() {
        let mut state = SseState::default();
        let err = handle_sse_event(
            "content_block_delta",
            r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}"#,
            &mut state,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("missing its `index`"), "{err}");
    }

    #[test]
    fn events_after_message_stop_are_ignored() {
        let mut state = SseState::default();
        let done = handle_sse_event("message_stop", r#"{"type":"message_stop"}"#, &mut state).unwrap();
        assert_eq!(done.len(), 1);
        assert!(handle_sse_event(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"late"}}"#,
            &mut state
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn message_delta_without_usage_still_records_the_stop_reason() {
        let mut state = SseState::default();
        let produced = handle_sse_event(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens"}}"#,
            &mut state,
        )
        .unwrap();
        assert!(produced.is_empty());
        let done = handle_sse_event("message_stop", r#"{"type":"message_stop"}"#, &mut state).unwrap();
        match &done[0] {
            StreamEvent::Done(resp) => assert_eq!(resp.stop_reason.as_deref(), Some("max_tokens")),
            other => panic!("expected Done, got {other:?}"),
        }
    }
}
