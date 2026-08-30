use agent_core::types::ChatMessage;

pub struct ContextManager {
    pub max_tokens: usize,
}

impl ContextManager {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }

    /// Truncates overly large tool observation outputs to save context window tokens
    pub fn compact_messages(&self, messages: &mut [ChatMessage]) {
        for msg in messages.iter_mut() {
            if msg.role == agent_core::types::Role::Tool {
                if let Some(content) = &msg.content {
                    if content.len() > 4000 {
                        let truncated = format!(
                            "{}\n... [TRUNCATED: Remaining {} chars omitted to preserve context]",
                            &content[..2000],
                            content.len() - 2000
                        );
                        msg.content = Some(truncated);
                    }
                }
            }
        }
    }
}
