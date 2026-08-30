
pub struct SecurityManager {
    pub require_approval_for_write: bool,
    pub require_approval_for_terminal: bool,
}

impl Default for SecurityManager {
    fn default() -> Self {
        Self {
            require_approval_for_write: false,
            require_approval_for_terminal: true,
        }
    }
}

impl SecurityManager {
    pub fn check_permission(&self, tool_name: &str) -> bool {
        match tool_name {
            "run_bash_command" => !self.require_approval_for_terminal,
            "write_file" => !self.require_approval_for_write,
            _ => true,
        }
    }
}
