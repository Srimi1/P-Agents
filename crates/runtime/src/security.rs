//! Policy layer deciding which tool calls have to be shown to the user.

use crate::approval::GrantScope;
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct SecurityManager {
    /// Tools the operator pre-cleared for this run, overriding the tool's own
    /// `requires_approval`.
    auto_approved: HashSet<String>,
    /// Blast radius of an "approve for session" answer. Carried here so config
    /// owns it; the `ApprovalGate` has to be built with the same value, since
    /// the gate is what actually keys the grants.
    grant_scope: GrantScope,
    yolo: bool,
}

impl SecurityManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_yolo(mut self, yolo: bool) -> Self {
        self.yolo = yolo;
        self
    }

    pub fn with_grant_scope(mut self, scope: GrantScope) -> Self {
        self.grant_scope = scope;
        self
    }

    pub fn with_auto_approved<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.auto_approved.extend(tools.into_iter().map(Into::into));
        self
    }

    pub fn auto_approve(&mut self, tool: impl Into<String>) {
        self.auto_approved.insert(tool.into());
    }

    pub fn is_yolo(&self) -> bool {
        self.yolo
    }

    pub fn grant_scope(&self) -> GrantScope {
        self.grant_scope
    }

    pub fn is_auto_approved(&self, tool_name: &str) -> bool {
        self.auto_approved.contains(tool_name)
    }

    /// `tool_requires_approval` is the tool's own declaration; policy can only
    /// waive it, never impose approval on a tool that does not ask for it.
    pub fn needs_approval(&self, tool_name: &str, tool_requires_approval: bool) -> bool {
        if self.yolo || self.is_auto_approved(tool_name) {
            return false;
        }
        tool_requires_approval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_defers_to_the_tool() {
        let sec = SecurityManager::new();
        assert!(sec.needs_approval("run_bash_command", true));
        assert!(!sec.needs_approval("read_file", false));
    }

    #[test]
    fn yolo_waives_everything() {
        let sec = SecurityManager::new().with_yolo(true);
        assert!(sec.is_yolo());
        assert!(!sec.needs_approval("run_bash_command", true));
    }

    #[test]
    fn auto_approved_tools_skip_the_gate() {
        let sec = SecurityManager::new().with_auto_approved(["write_file", "list_dir"]);
        assert!(!sec.needs_approval("write_file", true));
        assert!(sec.needs_approval("run_bash_command", true));
        assert!(sec.is_auto_approved("list_dir"));
    }

    #[test]
    fn auto_approval_never_adds_a_prompt() {
        let mut sec = SecurityManager::new();
        sec.auto_approve("read_file");
        assert!(!sec.needs_approval("read_file", false));
        assert!(!sec.needs_approval("unknown_tool", false));
    }

    #[test]
    fn clones_are_independent_snapshots() {
        let sec = SecurityManager::new().with_auto_approved(["write_file"]);
        let mut other = sec.clone();
        other.auto_approve("run_bash_command");
        assert!(sec.needs_approval("run_bash_command", true));
        assert!(!other.needs_approval("run_bash_command", true));
    }

    #[test]
    fn grant_scope_defaults_to_tool_wide() {
        assert_eq!(SecurityManager::new().grant_scope(), GrantScope::Tool);
    }

    #[test]
    fn grant_scope_is_opt_in() {
        let sec = SecurityManager::new().with_grant_scope(GrantScope::Agent);
        assert_eq!(sec.grant_scope(), GrantScope::Agent);
        // Narrowing grants must not silently waive approval anywhere.
        assert!(sec.needs_approval("run_bash_command", true));
    }

    #[test]
    fn grant_scope_survives_the_other_builders() {
        let sec = SecurityManager::new()
            .with_grant_scope(GrantScope::Agent)
            .with_auto_approved(["read_file"])
            .with_yolo(false);
        assert_eq!(sec.grant_scope(), GrantScope::Agent);
        assert_eq!(sec.clone().grant_scope(), GrantScope::Agent);
    }
}
