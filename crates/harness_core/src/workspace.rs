//! Path containment for the filesystem tools.
//!
//! The approval gate covers writing and shell execution, but reads are not
//! gated at all — an agent that can call `read_file` with any path it likes can
//! exfiltrate `~/.ssh/id_rsa` without the user ever seeing a prompt. This
//! module bounds every tool-supplied path to a set of allowed roots.
//!
//! Shell commands are deliberately out of scope: nothing here can stop
//! `cat ~/.ssh/id_rsa` inside `run_bash_command`. That tool is controlled by
//! requiring approval instead, so the user sees the command before it runs.

use anyhow::{Context, Result};
use std::path::{Component, Path, PathBuf};

/// Allowed filesystem roots. An empty root set means unrestricted, which is
/// what a user gets by explicitly opting out in config.
#[derive(Debug, Clone, Default)]
pub struct WorkspacePolicy {
    roots: Vec<PathBuf>,
}

impl WorkspacePolicy {
    /// No containment at all. Every path a tool is handed is accepted.
    pub fn unrestricted() -> Self {
        Self { roots: Vec::new() }
    }

    /// Confines tools to `roots`. Each root is canonicalized, so a root given
    /// through a symlink still matches paths resolved through it.
    pub fn with_roots<I, P>(roots: I) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut canonical = Vec::new();
        for root in roots {
            let root = root.as_ref();
            let resolved = root
                .canonicalize()
                .with_context(|| format!("workspace root {} does not exist", root.display()))?;
            canonical.push(resolved);
        }
        Ok(Self { roots: canonical })
    }

    /// The working directory, which is what a harness with no configured roots
    /// confines itself to.
    pub fn current_dir() -> Result<Self> {
        let cwd = std::env::current_dir().context("cannot determine the working directory")?;
        Self::with_roots([cwd])
    }

    pub fn is_unrestricted(&self) -> bool {
        self.roots.is_empty()
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Resolves a tool-supplied path and proves it lands inside a root.
    ///
    /// The path need not exist: the longest existing prefix is canonicalized
    /// (which resolves any symlinks, so a link pointing outside the workspace
    /// is caught) and the remainder appended. `..` in the non-existent tail is
    /// rejected outright rather than normalized, since there is nothing to
    /// resolve it against.
    pub fn resolve(&self, path: &str) -> Result<PathBuf> {
        if path.is_empty() {
            anyhow::bail!("path must not be empty");
        }

        let requested = PathBuf::from(path);
        let absolute = if requested.is_absolute() {
            requested
        } else {
            std::env::current_dir()
                .context("cannot determine the working directory")?
                .join(requested)
        };

        let resolved = canonicalize_lexically(&absolute)?;

        if self.is_unrestricted() {
            return Ok(resolved);
        }

        if self.roots.iter().any(|root| resolved.starts_with(root)) {
            return Ok(resolved);
        }

        anyhow::bail!(
            "path '{}' resolves to {}, which is outside the allowed workspace ({}). \
             Ask the user to widen `permissions.workspace_roots` if this is intended.",
            path,
            resolved.display(),
            self.roots
                .iter()
                .map(|r| r.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Canonicalizes the deepest existing ancestor of `path` and re-appends the
/// components that do not exist yet, so paths for files about to be created can
/// still be checked.
fn canonicalize_lexically(path: &Path) -> Result<PathBuf> {
    if let Ok(resolved) = path.canonicalize() {
        return Ok(resolved);
    }

    let mut tail: Vec<&std::ffi::OsStr> = Vec::new();
    let mut cursor = path;
    loop {
        match cursor.parent() {
            Some(parent) => {
                let name = cursor.file_name().ok_or_else(|| {
                    anyhow::anyhow!("path '{}' has no final component", path.display())
                })?;
                // `..` past the existing part cannot be resolved against a real
                // directory, so allowing it would let a caller climb out.
                if name == std::ffi::OsStr::new("..") {
                    anyhow::bail!(
                        "path '{}' uses '..' below a directory that does not exist",
                        path.display()
                    );
                }
                tail.push(name);
                if let Ok(resolved) = parent.canonicalize() {
                    let mut out = resolved;
                    for component in tail.iter().rev() {
                        out.push(component);
                    }
                    return Ok(out);
                }
                cursor = parent;
            }
            // Reached the filesystem root without finding anything that exists.
            None => {
                anyhow::bail!("path '{}' cannot be resolved", path.display());
            }
        }
    }
}

/// True when `path` contains a `..` component, before any resolution. Used for
/// error messages that are clearer than a containment failure.
pub fn has_parent_traversal(path: &str) -> bool {
    Path::new(path)
        .components()
        .any(|c| matches!(c, Component::ParentDir))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace() -> (tempfile::TempDir, WorkspacePolicy) {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(dir.path().join("src")).expect("mkdir");
        std::fs::write(dir.path().join("src/main.rs"), "fn main() {}").expect("write");
        let policy = WorkspacePolicy::with_roots([dir.path()]).expect("policy");
        (dir, policy)
    }

    #[test]
    fn accepts_a_path_inside_the_workspace() {
        let (dir, policy) = workspace();
        let resolved = policy
            .resolve(dir.path().join("src/main.rs").to_str().unwrap())
            .expect("inside the root");
        assert!(resolved.ends_with("src/main.rs"));
    }

    #[test]
    fn accepts_a_path_that_does_not_exist_yet() {
        let (dir, policy) = workspace();
        let target = dir.path().join("src/new/deeply/nested.rs");
        let resolved = policy.resolve(target.to_str().unwrap()).expect("new file");
        assert!(resolved.ends_with("src/new/deeply/nested.rs"));
    }

    #[test]
    fn rejects_an_absolute_path_outside_the_workspace() {
        let (_dir, policy) = workspace();
        let err = policy.resolve("/etc/passwd").expect_err("must be refused");
        assert!(err.to_string().contains("outside the allowed workspace"));
    }

    #[test]
    fn rejects_climbing_out_with_parent_components() {
        let (dir, policy) = workspace();
        let escape = dir.path().join("src/../../..").join("etc/passwd");
        assert!(policy.resolve(escape.to_str().unwrap()).is_err());
    }

    #[test]
    fn rejects_a_symlink_pointing_outside_the_workspace() {
        let (dir, policy) = workspace();
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("secret.txt"), "sensitive").expect("write");

        let link = dir.path().join("escape");
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.path(), &link).expect("symlink");

        // Canonicalization follows the link, so containment sees the real target.
        let err = policy
            .resolve(link.join("secret.txt").to_str().unwrap())
            .expect_err("a symlink must not be an escape hatch");
        assert!(err.to_string().contains("outside the allowed workspace"));
    }

    #[test]
    fn unrestricted_policy_allows_anything_that_resolves() {
        let policy = WorkspacePolicy::unrestricted();
        assert!(policy.is_unrestricted());
        assert!(policy.resolve("/etc").is_ok());
    }

    #[test]
    fn multiple_roots_are_all_honoured() {
        let a = tempfile::tempdir().expect("a");
        let b = tempfile::tempdir().expect("b");
        let policy = WorkspacePolicy::with_roots([a.path(), b.path()]).expect("policy");
        assert!(policy
            .resolve(a.path().join("x.txt").to_str().unwrap())
            .is_ok());
        assert!(policy
            .resolve(b.path().join("y.txt").to_str().unwrap())
            .is_ok());
        assert!(policy.resolve("/etc/passwd").is_err());
    }

    #[test]
    fn a_missing_root_is_rejected_at_construction() {
        assert!(WorkspacePolicy::with_roots(["/definitely/not/here"]).is_err());
    }

    #[test]
    fn empty_paths_are_rejected() {
        let (_dir, policy) = workspace();
        assert!(policy.resolve("").is_err());
    }

    #[test]
    fn parent_traversal_is_detectable_for_messaging() {
        assert!(has_parent_traversal("../etc/passwd"));
        assert!(has_parent_traversal("src/../../x"));
        assert!(!has_parent_traversal("src/main.rs"));
    }
}
