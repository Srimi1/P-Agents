use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }

    /// An assistant turn that produced both prose and tool calls. Anthropic emits
    /// these routinely; OpenAI can too when the model narrates before calling.
    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.filter(|c| !c.is_empty()),
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
            name: None,
        }
    }

    pub fn tool_response(
        tool_call_id: impl Into<String>,
        name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: Some(name.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl TokenUsage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }

    /// Folds another turn's usage into this running total. Counts come from a
    /// remote server, so they are saturated rather than trusted not to overflow.
    pub fn accumulate(&mut self, other: &TokenUsage) {
        self.prompt_tokens = self.prompt_tokens.saturating_add(other.prompt_tokens);
        self.completion_tokens = self
            .completion_tokens
            .saturating_add(other.completion_tokens);
        self.total_tokens = self.total_tokens.saturating_add(other.total_tokens);
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LlmResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<TokenUsage>,
    /// Provider-reported stop reason, when available (`end_turn`, `tool_use`,
    /// `max_tokens`, `stop`, `length`, …). Used to detect truncated turns.
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    Idle,
    Planning,
    WaitingForApproval,
    ExecutingTool,
    StreamingResponse,
    Completed,
    Error,
}

/// Truncates `s` to at most `max_bytes`, stepping back to the nearest UTF-8
/// character boundary so the slice can never panic on multi-byte input.
pub fn truncate_at_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_respects_char_boundaries() {
        // "日本語" is 3 bytes per char; cutting at 4 must step back to 3.
        let s = "日本語テキスト";
        let out = truncate_at_boundary(s, 4);
        assert_eq!(out, "日");
        assert!(s.starts_with(out));
    }

    #[test]
    fn truncate_is_identity_when_short_enough() {
        assert_eq!(truncate_at_boundary("hello", 100), "hello");
        assert_eq!(truncate_at_boundary("hello", 5), "hello");
    }

    #[test]
    fn truncate_can_return_empty() {
        assert_eq!(truncate_at_boundary("日", 1), "");
    }

    #[test]
    fn usage_saturates_instead_of_overflowing() {
        // Providers are remote; absurd counts must not panic the stream task.
        let huge = TokenUsage::new(usize::MAX, usize::MAX);
        assert_eq!(huge.total_tokens, usize::MAX);

        let mut running = TokenUsage::new(usize::MAX, 0);
        running.accumulate(&TokenUsage::new(usize::MAX, 0));
        assert_eq!(running.prompt_tokens, usize::MAX);
    }

    #[test]
    fn usage_accumulates() {
        let mut total = TokenUsage::default();
        total.accumulate(&TokenUsage::new(10, 5));
        total.accumulate(&TokenUsage::new(20, 7));
        assert_eq!(total.prompt_tokens, 30);
        assert_eq!(total.completion_tokens, 12);
        assert_eq!(total.total_tokens, 42);
    }
}
