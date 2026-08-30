use agent_core::{truncate_at_boundary, ChatMessage, HistoryCompactor, Role};

/// Rough bytes-per-token ratio for English-ish text and JSON. Only used when the
/// provider did not report a prompt size.
const BYTES_PER_TOKEN: usize = 4;

/// Per-message envelope cost (role, delimiters) the estimator adds on top of content.
const MESSAGE_OVERHEAD_TOKENS: usize = 4;

const DEFAULT_COMPACT_RATIO: f64 = 0.8;
const DEFAULT_KEEP_RECENT: usize = 6;
const DEFAULT_TOOL_CONTENT_BUDGET: usize = 2000;

const MARKER_PREFIX: &str = "... [TRUNCATED: ";
const MARKER_SUFFIX: &str = " chars omitted]";

/// Chars/4 heuristic. Deliberately provider-agnostic: it only has to be good
/// enough to decide *whether* to compact when no real count is available.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(BYTES_PER_TOKEN)
}

/// Estimated prompt size of a whole history, including serialized tool call arguments.
pub fn estimate_history_tokens(history: &[ChatMessage]) -> usize {
    history
        .iter()
        .map(|msg| {
            let content = msg.content.as_deref().map(estimate_tokens).unwrap_or(0);
            let calls = msg
                .tool_calls
                .as_ref()
                .map(|calls| {
                    calls
                        .iter()
                        .map(|c| {
                            estimate_tokens(&c.name) + estimate_tokens(&c.arguments.to_string())
                        })
                        .sum::<usize>()
                })
                .unwrap_or(0);
            content + calls + MESSAGE_OVERHEAD_TOKENS
        })
        .sum()
}

/// Keeps an agent's history inside the model's context window by shrinking old
/// tool observations. Messages are never removed: dropping an assistant turn that
/// carries `tool_calls`, or any of its matching tool responses, makes the next
/// request invalid for both OpenAI and Anthropic.
pub struct ContextManager {
    pub max_tokens: usize,
    compact_ratio: f64,
    keep_recent: usize,
    tool_content_budget: usize,
}

impl ContextManager {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            compact_ratio: DEFAULT_COMPACT_RATIO,
            keep_recent: DEFAULT_KEEP_RECENT,
            tool_content_budget: DEFAULT_TOOL_CONTENT_BUDGET,
        }
    }

    /// Fraction of `max_tokens` at which compaction kicks in. Clamped to a usable
    /// range so a bad config cannot compact on every turn or never compact at all.
    pub fn with_compact_ratio(mut self, ratio: f64) -> Self {
        self.compact_ratio = if ratio.is_finite() {
            ratio.clamp(0.05, 1.0)
        } else {
            DEFAULT_COMPACT_RATIO
        };
        self
    }

    /// Number of trailing messages left fully intact so the model keeps its
    /// immediate working context.
    pub fn with_keep_recent(mut self, keep_recent: usize) -> Self {
        self.keep_recent = keep_recent;
        self
    }

    /// Bytes of content retained from each truncated tool observation.
    pub fn with_tool_content_budget(mut self, budget: usize) -> Self {
        self.tool_content_budget = budget;
        self
    }

    pub fn compact_ratio(&self) -> f64 {
        self.compact_ratio
    }

    pub fn keep_recent(&self) -> usize {
        self.keep_recent
    }

    pub fn tool_content_budget(&self) -> usize {
        self.tool_content_budget
    }

    fn threshold_tokens(&self) -> usize {
        (self.max_tokens as f64 * self.compact_ratio) as usize
    }

    /// Truncates the content of tool observations older than the protected tail.
    pub fn compact_messages(&self, messages: &mut [ChatMessage]) {
        let protected_from = messages.len().saturating_sub(self.keep_recent);
        for msg in messages.iter_mut().take(protected_from) {
            if msg.role != Role::Tool {
                continue;
            }
            let Some(content) = msg.content.as_deref() else {
                continue;
            };
            if content.len() <= self.tool_content_budget || is_truncated(content) {
                continue;
            }
            let head = truncate_at_boundary(content, self.tool_content_budget);
            let omitted = content.chars().count() - head.chars().count();
            let replacement = format!("{head}\n{MARKER_PREFIX}{omitted}{MARKER_SUFFIX}");
            // The marker costs ~32 bytes, so for content only slightly over the
            // budget the "compacted" form is *larger* than the original. Never
            // spend tokens to compact.
            if replacement.len() >= content.len() {
                continue;
            }
            msg.content = Some(replacement);
        }
    }
}

/// A message this manager already shrank; re-truncating it would eat the marker
/// and lie about the omitted count.
fn is_truncated(content: &str) -> bool {
    content
        .rsplit('\n')
        .next()
        .is_some_and(|last| last.starts_with(MARKER_PREFIX) && last.ends_with(MARKER_SUFFIX))
}

impl HistoryCompactor for ContextManager {
    fn should_compact(&self, history: &[ChatMessage], last_prompt_tokens: Option<usize>) -> bool {
        let used = last_prompt_tokens.unwrap_or_else(|| estimate_history_tokens(history));
        used > self.threshold_tokens()
    }

    fn compact(&self, history: &mut Vec<ChatMessage>) {
        self.compact_messages(history.as_mut_slice());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::ToolCall;
    use serde_json::json;

    fn tool_msg(id: &str, content: impl Into<String>) -> ChatMessage {
        ChatMessage::tool_response(id, "read_file", content)
    }

    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: "read_file".to_string(),
            arguments: json!({"path": "a.rs"}),
        }
    }

    /// History with `n` filler exchanges after the system message, each an
    /// assistant tool_calls turn followed by its tool response.
    fn history_with_tool_pairs(n: usize, content: &str) -> Vec<ChatMessage> {
        let mut history = vec![ChatMessage::system("SYSTEM PROMPT")];
        for i in 0..n {
            let id = format!("call_{i}");
            history.push(ChatMessage::assistant_tool_calls(vec![call(&id)]));
            history.push(tool_msg(&id, content));
        }
        history
    }

    #[test]
    fn truncates_multibyte_content_without_panicking() {
        // Cut point lands mid-character for every candidate budget in this string.
        let content = "日".repeat(3000);
        let mut history = history_with_tool_pairs(4, &content);
        let manager = ContextManager::new(1000);

        manager.compact(&mut history);

        let first_tool = history[2].content.as_deref().unwrap();
        assert!(first_tool.len() < content.len());
        assert!(first_tool.starts_with('日'));
        assert!(first_tool.ends_with(MARKER_SUFFIX));
        // 2000 bytes of 3-byte chars steps back to 666 chars retained.
        assert!(first_tool.contains("[TRUNCATED: 2334 chars omitted]"));
    }

    #[test]
    fn only_tool_messages_are_shrunk() {
        let long = "x".repeat(9000);
        let mut history = vec![
            ChatMessage::system(long.clone()),
            ChatMessage::user(long.clone()),
            ChatMessage::assistant(long.clone()),
            tool_msg("call_0", long.clone()),
            ChatMessage::user("go on"),
        ];
        let manager = ContextManager::new(1000).with_keep_recent(1);

        manager.compact(&mut history);

        assert_eq!(history[0].content.as_deref(), Some(long.as_str()));
        assert_eq!(history[1].content.as_deref(), Some(long.as_str()));
        assert_eq!(history[2].content.as_deref(), Some(long.as_str()));
        assert!(history[3].content.as_deref().unwrap().len() < long.len());
    }

    #[test]
    fn recent_messages_stay_intact() {
        let long = "y".repeat(9000);
        let mut history = history_with_tool_pairs(5, &long);
        let manager = ContextManager::new(1000).with_keep_recent(6);

        manager.compact(&mut history);

        let len = history.len();
        for msg in &history[len - 6..] {
            if let Some(content) = &msg.content {
                assert_eq!(content, &long);
            }
        }
        // The oldest tool observation sits outside the protected tail.
        assert!(history[2].content.as_deref().unwrap().len() < long.len());
        assert_eq!(history[0].content.as_deref(), Some("SYSTEM PROMPT"));
    }

    #[test]
    fn keeps_tool_responses_paired_with_their_assistant_call() {
        let long = "z".repeat(9000);
        let mut history = history_with_tool_pairs(6, &long);
        let before = history.len();
        let manager = ContextManager::new(1000);

        manager.compact(&mut history);

        assert_eq!(history.len(), before);
        for pair in history[1..].chunks(2) {
            let calls = pair[0].tool_calls.as_ref().expect("assistant tool_calls");
            assert_eq!(pair[1].role, Role::Tool);
            assert_eq!(pair[1].tool_call_id.as_deref(), Some(calls[0].id.as_str()));
            assert!(pair[1].content.is_some());
        }
    }

    #[test]
    fn compaction_is_idempotent() {
        let mut history = history_with_tool_pairs(6, &"q".repeat(9000));
        let manager = ContextManager::new(1000);

        manager.compact(&mut history);
        let once = history.clone();
        manager.compact(&mut history);

        assert_eq!(history, once);
    }

    #[test]
    fn content_below_budget_is_left_alone() {
        let mut history = history_with_tool_pairs(6, "small output");
        let manager = ContextManager::new(1000);

        manager.compact(&mut history);

        assert_eq!(history[2].content.as_deref(), Some("small output"));
    }

    #[test]
    fn compaction_never_makes_a_message_larger() {
        // Just over the budget: the ~32-byte marker would cost more than the
        // 10 bytes it saves, so the message must be left exactly as it was.
        let content = "x".repeat(2010);
        let mut history = history_with_tool_pairs(6, &content);
        let manager = ContextManager::new(1000);

        manager.compact(&mut history);

        assert_eq!(history[2].content.as_deref(), Some(content.as_str()));

        // Far enough over the budget that truncation is a real saving.
        let big = "x".repeat(4000);
        let mut history = history_with_tool_pairs(6, &big);
        manager.compact(&mut history);
        assert!(history[2].content.as_deref().unwrap().len() < big.len());
    }

    #[test]
    fn tool_message_without_content_is_skipped() {
        let mut history = vec![
            ChatMessage {
                role: Role::Tool,
                content: None,
                tool_calls: None,
                tool_call_id: Some("call_0".into()),
                name: Some("read_file".into()),
            },
            ChatMessage::user("next"),
        ];
        let manager = ContextManager::new(1000).with_keep_recent(1);

        manager.compact(&mut history);

        assert!(history[0].content.is_none());
    }

    #[test]
    fn empty_and_short_histories_are_safe() {
        let manager = ContextManager::new(1000);
        let mut empty: Vec<ChatMessage> = Vec::new();
        manager.compact(&mut empty);
        assert!(empty.is_empty());

        let mut one = vec![ChatMessage::system("s")];
        manager.compact(&mut one);
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn should_compact_uses_reported_prompt_tokens() {
        let manager = ContextManager::new(1000);
        let history = vec![ChatMessage::user("hi")];

        assert!(manager.should_compact(&history, Some(801)));
        assert!(!manager.should_compact(&history, Some(800)));
        assert!(!manager.should_compact(&history, Some(10)));
    }

    #[test]
    fn should_compact_falls_back_to_estimate() {
        let manager = ContextManager::new(1000);
        let small = vec![ChatMessage::user("hi")];
        assert!(!manager.should_compact(&small, None));

        // 4000 chars ~ 1000 estimated tokens, past the 800 threshold.
        let big = vec![ChatMessage::user("a".repeat(4000))];
        assert!(manager.should_compact(&big, None));
    }

    #[test]
    fn compact_ratio_is_configurable_and_clamped() {
        let manager = ContextManager::new(1000).with_compact_ratio(0.5);
        let history = vec![ChatMessage::user("hi")];
        assert!(manager.should_compact(&history, Some(501)));
        assert!(!manager.should_compact(&history, Some(500)));

        assert_eq!(
            ContextManager::new(1000)
                .with_compact_ratio(-1.0)
                .compact_ratio(),
            0.05
        );
        assert_eq!(
            ContextManager::new(1000)
                .with_compact_ratio(9.0)
                .compact_ratio(),
            1.0
        );
        assert_eq!(
            ContextManager::new(1000)
                .with_compact_ratio(f64::NAN)
                .compact_ratio(),
            DEFAULT_COMPACT_RATIO
        );
    }

    #[test]
    fn tool_content_budget_is_configurable() {
        let mut history = history_with_tool_pairs(6, &"w".repeat(500));
        let manager = ContextManager::new(1000).with_tool_content_budget(100);

        manager.compact(&mut history);

        let first = history[2].content.as_deref().unwrap();
        assert!(first.starts_with(&"w".repeat(100)));
        assert!(first.contains("[TRUNCATED: 400 chars omitted]"));
        assert_eq!(manager.tool_content_budget(), 100);
        assert_eq!(manager.keep_recent(), DEFAULT_KEEP_RECENT);
    }

    #[test]
    fn estimate_tokens_uses_chars_not_bytes() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        // 4 three-byte chars still count as 4 chars.
        assert_eq!(estimate_tokens("日本語だ"), 1);
    }

    #[test]
    fn estimate_history_includes_tool_call_arguments() {
        let plain = vec![ChatMessage::assistant("ok")];
        let with_calls = vec![ChatMessage::assistant_with_tool_calls(
            Some("ok".to_string()),
            vec![call("call_0")],
        )];
        assert!(estimate_history_tokens(&with_calls) > estimate_history_tokens(&plain));
        assert_eq!(estimate_history_tokens(&[]), 0);
    }

    #[test]
    fn manager_is_usable_as_a_boxed_compactor() {
        let compactor: std::sync::Arc<dyn HistoryCompactor> =
            std::sync::Arc::new(ContextManager::new(1000));
        let mut history = history_with_tool_pairs(6, &"v".repeat(9000));
        compactor.compact(&mut history);
        assert!(history[2].content.as_deref().unwrap().len() < 9000);
    }
}
