use crate::types::ChatMessage;

/// Strategy for keeping an agent's history inside the model's context window.
/// `agent_core` only defines the hook; `harness_core::ContextManager` supplies
/// the implementation so the loop stays free of policy.
pub trait HistoryCompactor: Send + Sync {
    /// `last_prompt_tokens` is the provider-reported prompt size from the most
    /// recent turn, when the provider reported one.
    fn should_compact(&self, history: &[ChatMessage], last_prompt_tokens: Option<usize>) -> bool;

    fn compact(&self, history: &mut Vec<ChatMessage>);
}
