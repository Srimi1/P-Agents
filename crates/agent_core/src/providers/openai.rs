use super::{LlmProvider, LlmStream, StreamAccumulator, StreamEvent};
use crate::types::{
    truncate_at_boundary, ChatMessage, LlmResponse, Role, TokenUsage, ToolCall, ToolDefinition,
};
use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use eventsource_stream::{Event, Eventsource};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::pin::Pin;

/// Sentinel the OpenAI SSE endpoint sends as its last `data:` line.
const DONE_SENTINEL: &str = "[DONE]";
const MAX_ERROR_BODY_BYTES: usize = 2048;

/// Generic OpenAI-compatible provider (OpenAI, Ollama, DeepSeek, LocalAI, vLLM,
/// Gemini's OpenAI-compatible endpoint).
pub struct GenericOpenAiProvider {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub client: reqwest::Client,
}

impl GenericOpenAiProvider {
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    fn build_body(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        temperature: Option<f32>,
        stream: bool,
    ) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": to_wire(messages),
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools_to_wire(tools));
            body["tool_choice"] = json!("auto");
        }
        if let Some(temperature) = temperature {
            body["temperature"] = json!(temperature);
        }
        if stream {
            body["stream"] = json!(true);
            body["stream_options"] = json!({ "include_usage": true });
        }
        body
    }

    async fn send(&self, body: &Value) -> Result<reqwest::Response> {
        let mut request = self.client.post(self.endpoint()).json(body);
        // Local servers (Ollama, LocalAI) reject a bearer header they never issued.
        if !self.api_key.is_empty() {
            request = request.bearer_auth(&self.api_key);
        }
        let response = request
            .send()
            .await
            .with_context(|| format!("request to {} failed", self.endpoint()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|e| format!("<body could not be read: {e}>"));
            bail!(
                "openai-compatible endpoint {} returned {}: {}",
                self.endpoint(),
                status,
                truncate_at_boundary(&body, MAX_ERROR_BODY_BYTES)
            );
        }
        Ok(response)
    }
}

#[async_trait]
impl LlmProvider for GenericOpenAiProvider {
    fn provider_name(&self) -> &str {
        "openai_compatible"
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
        let body = self.build_body(messages, tools, temperature, true);
        let response = self.send(&body).await?;
        Ok(sse_bytes_to_stream(response.bytes_stream()))
    }

    async fn complete(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        temperature: Option<f32>,
    ) -> Result<LlmResponse> {
        let body = self.build_body(messages, tools, temperature, false);
        let response = self.send(&body).await?;
        let parsed: Value = response
            .json()
            .await
            .context("openai-compatible response was not valid JSON")?;
        parse_completion_body(&parsed)
    }
}

/// Converts our messages into the OpenAI chat wire format.
///
/// Our `ToolCall::arguments` is a `Value`, but OpenAI requires the arguments of
/// a tool call to be a *JSON string*; sending the object straight through is
/// silently accepted by some servers and rejected by others.
fn to_wire(messages: &[ChatMessage]) -> Vec<Value> {
    messages
        .iter()
        // `{"role":"assistant","content":null}` with no tool calls is a 400. The
        // agent loop never produces one, but a restored transcript can.
        .filter(|message| {
            message.role != Role::Assistant
                || message.content.as_deref().is_some_and(|c| !c.is_empty())
                || message.tool_calls.as_ref().is_some_and(|c| !c.is_empty())
        })
        .map(|message| match message.role {
            Role::Tool => json!({
                "role": "tool",
                "tool_call_id": message.tool_call_id.clone().unwrap_or_default(),
                "content": message.content.clone().unwrap_or_default(),
            }),
            role => {
                let mut wire = json!({
                    "role": role_str(role),
                    "content": match &message.content {
                        Some(content) => Value::String(content.clone()),
                        None => Value::Null,
                    },
                });
                if let Some(tool_calls) = &message.tool_calls {
                    if !tool_calls.is_empty() {
                        wire["tool_calls"] =
                            Value::Array(tool_calls.iter().map(tool_call_to_wire).collect());
                    }
                }
                if let Some(name) = &message.name {
                    wire["name"] = Value::String(name.clone());
                }
                wire
            }
        })
        .collect()
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn tool_call_to_wire(tool_call: &ToolCall) -> Value {
    let arguments = match &tool_call.arguments {
        // A model that produced unparseable arguments leaves `Null` behind; an
        // empty object is the only thing the server will accept back.
        Value::Null => "{}".to_string(),
        // An empty argument blob is only meaningful to the server as `{}`.
        Value::String(raw) if raw.trim().is_empty() => "{}".to_string(),
        Value::String(raw) if is_encoded_json_document(raw) => {
            // Already-encoded arguments must not be encoded a second time.
            raw.clone()
        }
        // Anything else (including a bare scalar string, which is NOT a valid
        // encoded argument document) is encoded now, so the field we send is
        // always parseable JSON text.
        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
    };
    json!({
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": arguments,
        }
    })
}

/// True when `raw` is already the encoded form of a JSON object or array, i.e.
/// something the server will accept verbatim as a tool-call `arguments` string.
fn is_encoded_json_document(raw: &str) -> bool {
    serde_json::from_str::<Value>(raw).is_ok_and(|v| v.is_object() || v.is_array())
}

/// Normalises a wire `arguments` value into the argument text it represents.
///
/// The OpenAI schema says this is a string, but several compatible servers
/// (llama.cpp, some LocalAI builds) send the object itself. Serialising it is
/// strictly better than dropping it and dispatching the tool with no arguments.
fn arguments_text(value: &Value) -> Option<String> {
    match value {
        Value::String(raw) => Some(raw.clone()),
        Value::Null => None,
        other => serde_json::to_string(other).ok(),
    }
}

fn tools_to_wire(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            })
        })
        .collect()
}

struct SseState {
    events: Pin<Box<dyn Stream<Item = Result<Event>> + Send>>,
    acc: StreamAccumulator,
    pending: VecDeque<Result<StreamEvent>>,
    finished: bool,
}

/// Turns a raw byte stream of SSE frames into an `LlmStream`.
///
/// Split out from `stream()` so the wire decoding can be driven from tests with
/// a synthetic byte stream instead of a socket.
pub(crate) fn sse_bytes_to_stream<S, E>(bytes: S) -> LlmStream
where
    S: Stream<Item = Result<bytes::Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + Sync + 'static,
{
    let events = Box::pin(bytes)
        .eventsource()
        .map(|event| event.map_err(|e| anyhow!("sse decode failed: {e}")));

    let state = SseState {
        events: Box::pin(events),
        acc: StreamAccumulator::new(),
        pending: VecDeque::new(),
        finished: false,
    };

    Box::pin(futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(item) = state.pending.pop_front() {
                return Some((item, state));
            }
            if state.finished {
                return None;
            }
            match state.events.next().await {
                Some(Ok(event)) => {
                    if is_stream_terminator(&event.data) {
                        state.finish();
                        continue;
                    }
                    match handle_sse_message(&event.data, &mut state.acc) {
                        Ok(events) => state.pending.extend(events.into_iter().map(Ok)),
                        Err(e) => {
                            state.finished = true;
                            state.pending.push_back(Err(e));
                        }
                    }
                }
                Some(Err(e)) => {
                    state.finished = true;
                    state.pending.push_back(Err(e));
                }
                // Servers that just close the connection still owe us a Done.
                None => state.finish(),
            }
        }
    }))
}

impl SseState {
    fn finish(&mut self) {
        self.finished = true;
        let acc = std::mem::take(&mut self.acc);
        self.pending
            .push_back(Ok(StreamEvent::Done(Box::new(acc.finish()))));
    }
}

fn is_stream_terminator(data: &str) -> bool {
    data.trim() == DONE_SENTINEL
}

/// Decodes one SSE `data:` payload, folding it into `acc` and returning the
/// events it produced. Pure apart from `acc`, so it is directly unit-testable.
fn handle_sse_message(data: &str, acc: &mut StreamAccumulator) -> Result<Vec<StreamEvent>> {
    let data = data.trim();
    if data.is_empty() || is_stream_terminator(data) {
        return Ok(Vec::new());
    }

    let chunk: Value = serde_json::from_str(data)
        .with_context(|| format!("malformed SSE chunk: {}", truncate_at_boundary(data, 256)))?;

    if let Some(error) = chunk.get("error").filter(|e| !e.is_null()) {
        bail!("openai-compatible endpoint reported an error: {}", error);
    }

    let mut events = Vec::new();

    if let Some(choice) = chunk
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    {
        if let Some(delta) = choice.get("delta") {
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    acc.push_text(text);
                    events.push(StreamEvent::TextDelta(text.to_string()));
                }
            }
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for (ordinal, tool_call) in tool_calls.iter().enumerate() {
                    // Index is what correlates fragments; fall back to position
                    // for servers that omit it on single-call turns.
                    let index = tool_call
                        .get("index")
                        .and_then(Value::as_u64)
                        .map(|i| i as usize)
                        .unwrap_or(ordinal);
                    let function = tool_call.get("function");
                    let id = tool_call.get("id").and_then(Value::as_str).unwrap_or("");
                    let name = function
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if !id.is_empty() || !name.is_empty() {
                        acc.start_tool_call(index, id.to_string(), name.to_string());
                        events.push(StreamEvent::ToolCallStarted {
                            index,
                            id: id.to_string(),
                            name: name.to_string(),
                        });
                    }
                    if let Some(fragment) = function
                        .and_then(|f| f.get("arguments"))
                        .and_then(arguments_text)
                    {
                        if !fragment.is_empty() {
                            acc.push_tool_args(index, &fragment);
                            events.push(StreamEvent::ToolCallArgsDelta {
                                index,
                                json_fragment: fragment,
                            });
                        }
                    }
                }
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            acc.set_stop_reason(reason);
        }
    }

    // Usage rides on a final chunk with an empty `choices` array, and Ollama
    // omits it entirely.
    if let Some(usage) = chunk.get("usage").and_then(parse_usage) {
        acc.set_usage(usage);
        events.push(StreamEvent::Usage(usage));
    }

    Ok(events)
}

fn parse_usage(usage: &Value) -> Option<TokenUsage> {
    let prompt_tokens = usage.get("prompt_tokens").and_then(Value::as_u64);
    let completion_tokens = usage.get("completion_tokens").and_then(Value::as_u64);
    if prompt_tokens.is_none() && completion_tokens.is_none() {
        return None;
    }
    let prompt_tokens = prompt_tokens.unwrap_or(0) as usize;
    let completion_tokens = completion_tokens.unwrap_or(0) as usize;
    Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        // Saturating, and only evaluated when `total_tokens` is absent: a server
        // reporting absurd counts must not abort the turn with an overflow panic.
        total_tokens: usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .map(|t| t as usize)
            .unwrap_or_else(|| prompt_tokens.saturating_add(completion_tokens)),
    })
}

/// Parses the non-streaming `/chat/completions` body.
fn parse_completion_body(body: &Value) -> Result<LlmResponse> {
    if let Some(error) = body.get("error").filter(|e| !e.is_null()) {
        bail!("openai-compatible endpoint reported an error: {}", error);
    }

    let choice = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| anyhow!("openai-compatible response contained no choices"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| anyhow!("openai-compatible choice contained no message"))?;

    let content = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|c| !c.is_empty())
        .map(str::to_string);

    let mut tool_calls = Vec::new();
    if let Some(raw_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for (ordinal, raw) in raw_calls.iter().enumerate() {
            let function = raw.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let raw_arguments = function
                .and_then(|f| f.get("arguments"))
                .and_then(arguments_text)
                .unwrap_or_default();
            let arguments = if raw_arguments.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&raw_arguments).unwrap_or(Value::Null)
            };
            let id = raw
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            tool_calls.push(ToolCall {
                id: if id.is_empty() {
                    format!("call_{ordinal}")
                } else {
                    id
                },
                name,
                arguments,
            });
        }
    }

    Ok(LlmResponse {
        content,
        tool_calls,
        usage: body.get("usage").and_then(parse_usage),
        stop_reason: choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use std::io::{Error as IoError, ErrorKind};

    fn byte_chunks(payload: &str, size: usize) -> Vec<Result<Bytes, IoError>> {
        payload
            .as_bytes()
            .chunks(size)
            .map(|c| Ok(Bytes::copy_from_slice(c)))
            .collect()
    }

    async fn drain(mut stream: LlmStream) -> Result<Vec<StreamEvent>> {
        let mut out = Vec::new();
        while let Some(event) = stream.next().await {
            out.push(event?);
        }
        Ok(out)
    }

    fn done_of(events: &[StreamEvent]) -> LlmResponse {
        match events.last() {
            Some(StreamEvent::Done(resp)) => (**resp).clone(),
            other => panic!("expected Done as last event, got {other:?}"),
        }
    }

    #[test]
    fn assistant_tool_call_serializes_arguments_as_a_json_string() {
        let messages = vec![ChatMessage::assistant_with_tool_calls(
            Some("checking".to_string()),
            vec![ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: json!({ "path": "src/main.rs" }),
            }],
        )];

        let wire = to_wire(&messages);
        assert_eq!(
            wire[0],
            json!({
                "role": "assistant",
                "content": "checking",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{\"path\":\"src/main.rs\"}"
                    }
                }]
            })
        );
        // The arguments field must be a string, never an object.
        assert!(wire[0]["tool_calls"][0]["function"]["arguments"].is_string());
    }

    #[test]
    fn tool_response_becomes_role_tool_with_tool_call_id() {
        let messages = vec![ChatMessage::tool_response(
            "call_1",
            "read_file",
            "fn main()",
        )];
        assert_eq!(
            to_wire(&messages)[0],
            json!({
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "fn main()"
            })
        );
    }

    #[test]
    fn tool_call_without_content_keeps_a_null_content_field() {
        let messages = vec![ChatMessage::assistant_tool_calls(vec![ToolCall {
            id: "c".to_string(),
            name: "noop".to_string(),
            arguments: json!({}),
        }])];
        let wire = to_wire(&messages);
        assert!(wire[0]["content"].is_null());
        assert_eq!(wire[0]["tool_calls"][0]["function"]["arguments"], "{}");
    }

    #[test]
    fn null_arguments_degrade_to_an_empty_object_string() {
        let call = ToolCall {
            id: "c".to_string(),
            name: "broken".to_string(),
            arguments: Value::Null,
        };
        assert_eq!(tool_call_to_wire(&call)["function"]["arguments"], "{}");
    }

    #[test]
    fn preencoded_string_arguments_are_not_double_encoded() {
        let call = ToolCall {
            id: "c".to_string(),
            name: "t".to_string(),
            arguments: Value::String("{\"a\":1}".to_string()),
        };
        assert_eq!(
            tool_call_to_wire(&call)["function"]["arguments"],
            "{\"a\":1}"
        );
    }

    #[test]
    fn system_and_user_messages_round_trip() {
        let messages = vec![ChatMessage::system("be terse"), ChatMessage::user("hi")];
        let wire = to_wire(&messages);
        assert_eq!(wire[0], json!({"role": "system", "content": "be terse"}));
        assert_eq!(wire[1], json!({"role": "user", "content": "hi"}));
    }

    #[test]
    fn tools_use_the_function_envelope() {
        let tools = vec![ToolDefinition {
            name: "grep".to_string(),
            description: "search".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }];
        assert_eq!(
            tools_to_wire(&tools)[0],
            json!({
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}}
                }
            })
        );
    }

    #[test]
    fn body_includes_stream_options_only_when_streaming() {
        let provider = GenericOpenAiProvider::new("", "http://localhost:11434/v1/", "llama3");
        let streaming = provider.build_body(&[ChatMessage::user("hi")], &[], Some(0.2), true);
        assert_eq!(streaming["stream"], json!(true));
        assert_eq!(streaming["stream_options"]["include_usage"], json!(true));
        // f32 widens to f64 on the wire, so compare with a tolerance.
        assert!((streaming["temperature"].as_f64().unwrap() - 0.2).abs() < 1e-6);
        assert!(streaming.get("tools").is_none());

        let unary = provider.build_body(&[ChatMessage::user("hi")], &[], None, false);
        assert!(unary.get("stream").is_none());
        assert!(unary.get("temperature").is_none());
        assert_eq!(
            provider.endpoint(),
            "http://localhost:11434/v1/chat/completions"
        );
    }

    #[tokio::test]
    async fn text_stream_survives_adversarial_byte_boundaries() {
        // Multibyte content forces splits inside UTF-8 sequences.
        let payload = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"héllo \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"世界 👍\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":4,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n",
        );

        for size in [1usize, 2, 3, 5, 7, 13, 64, 4096] {
            let events = drain(sse_bytes_to_stream(futures::stream::iter(byte_chunks(
                payload, size,
            ))))
            .await
            .unwrap_or_else(|e| panic!("chunk size {size} failed: {e}"));

            let done = done_of(&events);
            assert_eq!(
                done.content.as_deref(),
                Some("héllo 世界 👍"),
                "chunk size {size}"
            );
            assert_eq!(
                done.stop_reason.as_deref(),
                Some("stop"),
                "chunk size {size}"
            );
            let usage = done.usage.expect("usage");
            assert_eq!(usage.prompt_tokens, 11);
            assert_eq!(usage.total_tokens, 15);
            assert!(events.iter().any(|e| matches!(e, StreamEvent::Usage(_))));
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, StreamEvent::Done(_)))
                    .count(),
                1
            );
        }
    }

    #[tokio::test]
    async fn tool_call_arguments_split_across_fragments_reassemble() {
        let payload = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_a\",\"type\":\"function\",\"function\":{\"name\":\"write_file\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":\\\"a.txt\\\",\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"body\\\":\\\"hi\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        for size in [1usize, 9, 4096] {
            let events = drain(sse_bytes_to_stream(futures::stream::iter(byte_chunks(
                payload, size,
            ))))
            .await
            .unwrap();
            let done = done_of(&events);
            assert_eq!(done.tool_calls.len(), 1);
            assert_eq!(done.tool_calls[0].id, "call_a");
            assert_eq!(done.tool_calls[0].name, "write_file");
            assert_eq!(done.tool_calls[0].arguments["path"], "a.txt");
            assert_eq!(done.tool_calls[0].arguments["body"], "hi");
            assert_eq!(done.stop_reason.as_deref(), Some("tool_calls"));
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, StreamEvent::ToolCallStarted { .. }))
                    .count(),
                1,
                "the empty follow-up fragments must not restart the call"
            );
        }
    }

    #[tokio::test]
    async fn parallel_tool_calls_keep_their_indices() {
        let payload = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"second\",\"arguments\":\"{\\\"y\\\":2}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"first\",\"arguments\":\"{\\\"x\\\":1}\"}}]}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let events = drain(sse_bytes_to_stream(futures::stream::iter(byte_chunks(
            payload, 3,
        ))))
        .await
        .unwrap();
        let done = done_of(&events);
        assert_eq!(done.tool_calls.len(), 2);
        assert_eq!(done.tool_calls[0].name, "first");
        assert_eq!(done.tool_calls[1].name, "second");
    }

    #[tokio::test]
    async fn missing_usage_chunk_yields_none() {
        let payload = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let events = drain(sse_bytes_to_stream(futures::stream::iter(byte_chunks(
            payload, 4096,
        ))))
        .await
        .unwrap();
        let done = done_of(&events);
        assert_eq!(done.content.as_deref(), Some("ok"));
        assert!(done.usage.is_none());
        assert!(!events.iter().any(|e| matches!(e, StreamEvent::Usage(_))));
    }

    #[tokio::test]
    async fn done_sentinel_terminates_and_ignores_trailing_frames() {
        let payload = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"ignored\"}}]}\n\n",
        );
        let events = drain(sse_bytes_to_stream(futures::stream::iter(byte_chunks(
            payload, 4096,
        ))))
        .await
        .unwrap();
        assert!(matches!(events.last(), Some(StreamEvent::Done(_))));
        assert_eq!(done_of(&events).content.as_deref(), Some("a"));
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, StreamEvent::TextDelta(_)))
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn stream_closed_without_sentinel_still_emits_done() {
        let payload = "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n";
        let events = drain(sse_bytes_to_stream(futures::stream::iter(byte_chunks(
            payload, 4096,
        ))))
        .await
        .unwrap();
        assert_eq!(done_of(&events).content.as_deref(), Some("a"));
    }

    #[tokio::test]
    async fn transport_error_surfaces_as_a_stream_error() {
        let chunks: Vec<Result<Bytes, IoError>> = vec![
            Ok(Bytes::from_static(
                b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
            )),
            Err(IoError::new(ErrorKind::ConnectionReset, "peer went away")),
        ];
        let result = drain(sse_bytes_to_stream(futures::stream::iter(chunks))).await;
        let err = result.expect_err("transport error must propagate");
        assert!(err.to_string().contains("peer went away"), "{err}");
    }

    #[tokio::test]
    async fn malformed_chunk_is_reported_not_swallowed() {
        let payload = "data: {not json}\n\n";
        let result = drain(sse_bytes_to_stream(futures::stream::iter(byte_chunks(
            payload, 4096,
        ))))
        .await;
        let err = result.expect_err("malformed chunk must error");
        assert!(err.to_string().contains("malformed SSE chunk"), "{err}");
    }

    #[tokio::test]
    async fn error_payload_in_stream_becomes_an_error() {
        let payload = "data: {\"error\":{\"message\":\"context length exceeded\"}}\n\n";
        let result = drain(sse_bytes_to_stream(futures::stream::iter(byte_chunks(
            payload, 4096,
        ))))
        .await;
        let err = result.expect_err("error payload must error");
        assert!(err.to_string().contains("context length exceeded"), "{err}");
    }

    #[test]
    fn handle_sse_message_tolerates_empty_and_sentinel_payloads() {
        let mut acc = StreamAccumulator::new();
        assert!(handle_sse_message("", &mut acc).unwrap().is_empty());
        assert!(handle_sse_message("  ", &mut acc).unwrap().is_empty());
        assert!(handle_sse_message("[DONE]", &mut acc).unwrap().is_empty());
        assert!(acc.finish().content.is_none());
    }

    #[test]
    fn handle_sse_message_falls_back_to_position_when_index_is_absent() {
        let mut acc = StreamAccumulator::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"id":"a","function":{"name":"t","arguments":"{}"}}]}}]}"#;
        let events = handle_sse_message(data, &mut acc).unwrap();
        assert!(matches!(
            events[0],
            StreamEvent::ToolCallStarted { index: 0, .. }
        ));
        assert_eq!(acc.finish().tool_calls[0].name, "t");
    }

    #[test]
    fn absurd_token_counts_do_not_panic() {
        // A hostile or buggy endpoint must not be able to abort the turn with an
        // arithmetic overflow (which panics in debug builds).
        let usage = parse_usage(&json!({
            "prompt_tokens": u64::MAX,
            "completion_tokens": u64::MAX,
            "total_tokens": 3
        }))
        .unwrap();
        assert_eq!(usage.total_tokens, 3);
        let summed =
            parse_usage(&json!({"prompt_tokens": u64::MAX, "completion_tokens": u64::MAX}))
                .unwrap();
        assert_eq!(summed.total_tokens, usize::MAX);
    }

    #[test]
    fn object_arguments_are_preserved_not_silently_dropped() {
        // llama.cpp/LocalAI-style servers sometimes send `arguments` as an
        // object. Dropping it would dispatch the tool with no arguments at all.
        let body = json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "c1",
                        "function": {"name": "write", "arguments": {"path": "a.txt"}}
                    }]
                }
            }]
        });
        let resp = parse_completion_body(&body).unwrap();
        assert_eq!(resp.tool_calls[0].arguments["path"], "a.txt");
    }

    #[test]
    fn object_arguments_in_a_stream_are_preserved() {
        let mut acc = StreamAccumulator::new();
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"t","arguments":{"k":7}}}]}}]}"#;
        handle_sse_message(data, &mut acc).unwrap();
        assert_eq!(acc.finish().tool_calls[0].arguments["k"], 7);
    }

    #[test]
    fn string_arguments_that_are_not_an_object_are_re_encoded() {
        // `arguments` on the wire must always be a JSON *document*. A bare
        // scalar string must be re-encoded, not passed through as raw text.
        let call = ToolCall {
            id: "c".to_string(),
            name: "t".to_string(),
            arguments: Value::String("hello".to_string()),
        };
        let wire = tool_call_to_wire(&call);
        let raw = wire["function"]["arguments"].as_str().unwrap();
        serde_json::from_str::<Value>(raw).expect("arguments must be valid JSON text");
        assert_eq!(raw, "\"hello\"");

        let empty = ToolCall {
            id: "c".to_string(),
            name: "t".to_string(),
            arguments: Value::String("   ".to_string()),
        };
        assert_eq!(tool_call_to_wire(&empty)["function"]["arguments"], "{}");
    }

    #[test]
    fn usage_without_total_is_summed() {
        let usage = parse_usage(&json!({"prompt_tokens": 5, "completion_tokens": 2})).unwrap();
        assert_eq!(usage.total_tokens, 7);
        assert!(parse_usage(&Value::Null).is_none());
        assert!(parse_usage(&json!({})).is_none());
    }

    #[test]
    fn completion_body_parses_content_tool_calls_and_usage() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "on it",
                    "tool_calls": [{
                        "id": "call_9",
                        "type": "function",
                        "function": {"name": "ls", "arguments": "{\"dir\":\".\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}
        });
        let resp = parse_completion_body(&body).unwrap();
        assert_eq!(resp.content.as_deref(), Some("on it"));
        assert_eq!(resp.tool_calls[0].id, "call_9");
        assert_eq!(resp.tool_calls[0].arguments["dir"], ".");
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_calls"));
        assert_eq!(resp.usage.unwrap().total_tokens, 11);
    }

    #[test]
    fn completion_body_handles_null_content_and_absent_usage() {
        let body = json!({
            "choices": [{"message": {"role": "assistant", "content": null}, "finish_reason": "stop"}]
        });
        let resp = parse_completion_body(&body).unwrap();
        assert!(resp.content.is_none());
        assert!(resp.tool_calls.is_empty());
        assert!(resp.usage.is_none());
    }

    #[test]
    fn completion_body_rejects_missing_choices_and_errors() {
        assert!(parse_completion_body(&json!({"choices": []})).is_err());
        assert!(parse_completion_body(&json!({})).is_err());
        let err = parse_completion_body(&json!({"error": {"message": "bad key"}})).unwrap_err();
        assert!(err.to_string().contains("bad key"), "{err}");
    }

    #[test]
    fn completion_body_keeps_unparseable_arguments_as_null() {
        let body = json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{"function": {"name": "t", "arguments": "{oops"}}]
                }
            }]
        });
        let resp = parse_completion_body(&body).unwrap();
        assert_eq!(resp.tool_calls[0].arguments, Value::Null);
        // Servers that omit ids still need something to correlate results with.
        assert!(!resp.tool_calls[0].id.is_empty());
    }
}
