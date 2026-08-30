use crate::personas::*;
use std::collections::BTreeMap;

/// A persona the orchestrator can spawn: the display name that shows up in
/// transcripts, plus the system prompt that defines its behaviour.
#[derive(Debug, Clone)]
pub struct Persona {
    pub display_name: String,
    pub prompt: String,
}

/// The set of roles `spawn_subagent` will accept. Built-ins are always present;
/// config-defined personas are merged on top and may override a built-in prompt.
#[derive(Debug, Clone)]
pub struct PersonaRegistry {
    personas: BTreeMap<String, Persona>,
}

impl Default for PersonaRegistry {
    fn default() -> Self {
        let mut personas = BTreeMap::new();
        let builtins: [(&str, &str, &str); 5] = [
            ("engineer", "SoftwareEngineer", get_engineer_prompt()),
            ("verifier", "Verifier", get_verifier_prompt()),
            ("critic", "EgoistCritic", get_critic_prompt()),
            ("researcher", "Researcher", get_researcher_prompt()),
            ("analyst", "DataAnalyst", get_analyst_prompt()),
        ];
        for (role, display_name, prompt) in builtins {
            personas.insert(
                role.to_string(),
                Persona {
                    display_name: display_name.to_string(),
                    prompt: prompt.to_string(),
                },
            );
        }
        Self { personas }
    }
}

impl PersonaRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces a persona. Custom personas from `config.toml` land here.
    pub fn insert(&mut self, role: impl Into<String>, display_name: impl Into<String>, prompt: impl Into<String>) {
        let role = role.into();
        self.personas.insert(
            role,
            Persona {
                display_name: display_name.into(),
                prompt: prompt.into(),
            },
        );
    }

    pub fn get(&self, role: &str) -> Option<&Persona> {
        self.personas.get(role)
    }

    /// Role names, sorted — used to build the tool's JSON-schema enum so the
    /// model can only ask for personas that exist.
    pub fn roles(&self) -> Vec<String> {
        self.personas.keys().cloned().collect()
    }

    pub fn planner_prompt(&self) -> &'static str {
        get_planner_prompt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_roles_are_present() {
        let registry = PersonaRegistry::new();
        for role in ["engineer", "verifier", "critic", "researcher", "analyst"] {
            assert!(registry.get(role).is_some(), "missing built-in role {role}");
        }
        assert_eq!(registry.roles().len(), 5);
    }

    #[test]
    fn custom_persona_can_be_added_and_can_override_a_builtin() {
        let mut registry = PersonaRegistry::new();
        registry.insert("dba", "DatabaseExpert", "You tune queries.");
        assert_eq!(registry.get("dba").unwrap().display_name, "DatabaseExpert");
        assert!(registry.roles().contains(&"dba".to_string()));

        registry.insert("critic", "Critic", "Custom critic prompt.");
        assert_eq!(registry.get("critic").unwrap().prompt, "Custom critic prompt.");
        assert_eq!(registry.roles().len(), 6);
    }

    #[test]
    fn unknown_role_is_none() {
        assert!(PersonaRegistry::new().get("wizard").is_none());
    }
}
