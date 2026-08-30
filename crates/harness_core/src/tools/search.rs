use crate::tool_registry::Tool;
use agent_core::truncate_at_boundary;
use anyhow::{Context, Result};
use async_trait::async_trait;
use globset::{Glob, GlobMatcher};
use ignore::WalkBuilder;
use regex::RegexBuilder;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

const DEFAULT_GREP_MAX_RESULTS: usize = 100;
const DEFAULT_FIND_MAX_RESULTS: usize = 200;
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
const MAX_LINE_BYTES: usize = 512;
const BINARY_SNIFF_BYTES: usize = 8192;
const TRUNCATED_LINE_MARKER: &str = "…[line truncated]";

fn optional_str<'a>(args: &'a Value, key: &str, default: &'a str) -> Result<&'a str> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => v
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("'{}' must be a string", key)),
    }
}

fn optional_bool(args: &Value, key: &str) -> Result<bool> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(false),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| anyhow::anyhow!("'{}' must be a boolean", key)),
    }
}

fn optional_limit(args: &Value, key: &str, default: usize) -> Result<usize> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => {
            let n = v
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("'{}' must be a non-negative integer", key))?;
            if n == 0 {
                anyhow::bail!("'{}' must be greater than zero", key);
            }
            Ok(n as usize)
        }
    }
}

fn compile_glob(pattern: &str, key: &str) -> Result<GlobMatcher> {
    Ok(Glob::new(pattern)
        .with_context(|| format!("Invalid glob for '{}': {}", key, pattern))?
        .compile_matcher())
}

fn walker(root: &Path) -> ignore::Walk {
    let mut builder = WalkBuilder::new(root);
    // Mirrors ListDirTool's walk so the three tools agree on what exists:
    // dotfiles are searchable (.github, .env.example, .cargo/config.toml), only
    // `.git` itself is pruned. `parents`/`git_global` are off so that results
    // depend solely on the tree being searched, never on a .gitignore above the
    // root or the user's global excludes file -- silent omissions are worse than
    // extra hits for a search tool. require_git(false) keeps in-tree .gitignore
    // files meaningful even when the tree is not a git checkout.
    builder
        .hidden(false)
        .parents(false)
        .git_global(false)
        .git_ignore(true)
        .git_exclude(true)
        .require_git(false)
        .follow_links(false)
        .filter_entry(|entry| entry.file_name() != ".git")
        .sort_by_file_path(|a, b| a.cmp(b));
    builder.build()
}

fn relative_display(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    if rel.as_os_str().is_empty() {
        // `root` is the file itself, so strip_prefix yields "". Fall back to the
        // file name rather than emitting a nameless ":12: ..." line.
        return match path.file_name() {
            Some(name) => name.to_string_lossy().to_string(),
            None => path.to_string_lossy().to_string(),
        };
    }
    rel.to_string_lossy().to_string()
}

/// Matches either the path relative to the search root or the bare file name, so that
/// both "*.rs" and "src/**/*.rs" behave as a caller would expect.
fn glob_matches(matcher: &GlobMatcher, rel: &str, path: &Path) -> bool {
    if matcher.is_match(rel) {
        return true;
    }
    match path.file_name() {
        Some(name) => matcher.is_match(Path::new(name)),
        None => false,
    }
}

fn resolve_root(path_str: &str) -> Result<PathBuf> {
    let root = PathBuf::from(path_str);
    if !root.exists() {
        anyhow::bail!("Path not found: {}", path_str);
    }
    Ok(root)
}

fn read_text_file(path: &Path) -> Option<String> {
    use std::io::Read;

    let file = std::fs::File::open(path).ok()?;
    // Fast path: skip huge files without reading them at all.
    if file.metadata().ok()?.len() > MAX_FILE_BYTES {
        return None;
    }
    // Hard cap on the actual read as well. The metadata check above races with
    // writers, so a file that grows between stat and read must not be able to
    // pull an unbounded amount into memory. Reading one byte past the cap lets
    // us detect (and drop) a file that outgrew its metadata.
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES + 1).read_to_end(&mut bytes).ok()?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return None;
    }
    let sniff_len = bytes.len().min(BINARY_SNIFF_BYTES);
    if bytes[..sniff_len].contains(&0) {
        return None;
    }
    String::from_utf8(bytes).ok()
}

/// Trims and length-caps a matched line, marking it when content was dropped so
/// a truncated line is never mistaken for the file's real contents.
fn format_line(line: &str) -> String {
    let trimmed = line.trim();
    let capped = truncate_at_boundary(trimmed, MAX_LINE_BYTES);
    if capped.len() == trimmed.len() {
        capped.to_string()
    } else {
        format!("{}{}", capped, TRUNCATED_LINE_MARKER)
    }
}

pub struct GrepSearchTool;

#[async_trait]
impl Tool for GrepSearchTool {
    fn name(&self) -> &str {
        "grep_search"
    }

    fn description(&self) -> &str {
        "Searches file contents for a regular expression."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for."
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search. Defaults to the current directory."
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Match case-insensitively. Defaults to false."
                },
                "include_glob": {
                    "type": "string",
                    "description": "Only search files matching this glob, e.g. '*.rs' or 'src/**/*.rs'."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return. Defaults to 100."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'pattern' argument"))?
            .to_string();
        let path_str = optional_str(&args, "path", ".")?.to_string();
        let case_insensitive = optional_bool(&args, "case_insensitive")?;
        let max_results = optional_limit(&args, "max_results", DEFAULT_GREP_MAX_RESULTS)?;
        let include_glob = match args.get("include_glob") {
            None | Some(Value::Null) => None,
            Some(_) => Some(compile_glob(
                optional_str(&args, "include_glob", "")?,
                "include_glob",
            )?),
        };

        let regex = RegexBuilder::new(&pattern)
            .case_insensitive(case_insensitive)
            .build()
            .with_context(|| format!("Invalid regular expression: {}", pattern))?;

        let root = resolve_root(&path_str)?;

        let search_pattern = pattern.clone();
        tokio::task::spawn_blocking(move || {
            let mut lines: Vec<String> = Vec::new();
            let mut truncated = false;

            'walk: for entry in walker(&root) {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    continue;
                }

                let path = entry.path();
                let rel = relative_display(&root, path);
                if let Some(matcher) = &include_glob {
                    if !glob_matches(matcher, &rel, path) {
                        continue;
                    }
                }

                let content = match read_text_file(path) {
                    Some(c) => c,
                    None => continue,
                };

                for (idx, line) in content.lines().enumerate() {
                    if !regex.is_match(line) {
                        continue;
                    }
                    if lines.len() >= max_results {
                        truncated = true;
                        break 'walk;
                    }
                    lines.push(format!("{}:{}: {}", rel, idx + 1, format_line(line)));
                }
            }

            if lines.is_empty() {
                return Ok(format!("No matches found for pattern '{}'", search_pattern));
            }

            let mut out = lines.join("\n");
            if truncated {
                out.push_str(&format!(
                    "\n[Results truncated at {} matches; narrow the pattern or raise max_results]",
                    max_results
                ));
            }
            Ok(out)
        })
        .await
        .context("grep_search worker failed")?
    }
}

pub struct FindFilesByNameTool;

#[async_trait]
impl Tool for FindFilesByNameTool {
    fn name(&self) -> &str {
        "find_files_by_name"
    }

    fn description(&self) -> &str {
        "Finds files whose path matches a glob pattern."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob matched against the file name and the path relative to 'path', e.g. '*.rs' or 'src/**/*.rs'."
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search. Defaults to the current directory."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of paths to return. Defaults to 200."
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'pattern' argument"))?
            .to_string();
        let path_str = optional_str(&args, "path", ".")?.to_string();
        let max_results = optional_limit(&args, "max_results", DEFAULT_FIND_MAX_RESULTS)?;
        let matcher = compile_glob(&pattern, "pattern")?;
        let root = resolve_root(&path_str)?;

        tokio::task::spawn_blocking(move || {
            let mut paths: Vec<String> = Vec::new();

            for entry in walker(&root) {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    continue;
                }

                let path = entry.path();
                let rel = relative_display(&root, path);
                if glob_matches(&matcher, &rel, path) {
                    paths.push(rel);
                }
            }

            if paths.is_empty() {
                return Ok(format!("No files matched glob '{}'", pattern));
            }

            paths.sort();
            let truncated = paths.len() > max_results;
            paths.truncate(max_results);

            let mut out = paths.join("\n");
            if truncated {
                out.push_str(&format!(
                    "\n[Results truncated at {} files; narrow the pattern or raise max_results]",
                    max_results
                ));
            }
            Ok(out)
        })
        .await
        .context("find_files_by_name worker failed")?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write(root: &Path, rel: &str, content: &str) {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, content).unwrap();
    }

    fn fixture() -> TempDir {
        let dir = TempDir::new().unwrap();
        let root = dir.path();
        write(root, "src/main.rs", "fn main() {\n    let needle = 1;\n}\n");
        write(root, "src/deep/mod.rs", "// NEEDLE in a comment\n");
        write(root, "notes.txt", "plain needle text\n");
        dir
    }

    async fn grep(args: serde_json::Value) -> Result<String> {
        GrepSearchTool.execute(args).await
    }

    async fn find(args: serde_json::Value) -> Result<String> {
        FindFilesByNameTool.execute(args).await
    }

    #[tokio::test]
    async fn grep_reports_path_and_line_number() {
        let dir = fixture();
        let out = grep(json!({ "pattern": "needle", "path": dir.path() }))
            .await
            .unwrap();
        assert!(
            out.contains("src/main.rs:2: let needle = 1;"),
            "unexpected output: {}",
            out
        );
        assert!(out.contains("notes.txt:1: plain needle text"));
        // Case-sensitive by default.
        assert!(!out.contains("src/deep/mod.rs"));
    }

    #[tokio::test]
    async fn grep_case_insensitive() {
        let dir = fixture();
        let out = grep(json!({
            "pattern": "needle",
            "path": dir.path(),
            "case_insensitive": true
        }))
        .await
        .unwrap();
        assert!(out.contains("src/deep/mod.rs:1: // NEEDLE in a comment"));
    }

    #[tokio::test]
    async fn grep_include_glob_filters_by_extension() {
        let dir = fixture();
        let out = grep(json!({
            "pattern": "needle",
            "path": dir.path(),
            "include_glob": "*.rs"
        }))
        .await
        .unwrap();
        assert!(out.contains("src/main.rs"));
        assert!(!out.contains("notes.txt"));
    }

    #[tokio::test]
    async fn grep_include_glob_filters_by_subpath() {
        let dir = fixture();
        let out = grep(json!({
            "pattern": "needle",
            "path": dir.path(),
            "include_glob": "src/deep/**",
            "case_insensitive": true
        }))
        .await
        .unwrap();
        assert!(out.contains("src/deep/mod.rs"));
        assert!(!out.contains("src/main.rs"));
    }

    #[tokio::test]
    async fn grep_skips_gitignored_files() {
        let dir = fixture();
        write(dir.path(), ".gitignore", "secret.txt\nbuild/\n");
        write(dir.path(), "secret.txt", "needle hidden here\n");
        write(dir.path(), "build/out.txt", "needle in build output\n");

        let out = grep(json!({ "pattern": "needle", "path": dir.path() }))
            .await
            .unwrap();
        assert!(!out.contains("secret.txt"), "unexpected output: {}", out);
        assert!(!out.contains("build/out.txt"), "unexpected output: {}", out);
        assert!(out.contains("notes.txt"));
    }

    #[tokio::test]
    async fn grep_reports_truncation() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "many.txt", &"needle\n".repeat(20));

        let out = grep(json!({ "pattern": "needle", "path": dir.path(), "max_results": 3 }))
            .await
            .unwrap();
        let body: Vec<&str> = out.lines().filter(|l| !l.starts_with('[')).collect();
        assert_eq!(body.len(), 3);
        assert!(out.contains("Results truncated at 3 matches"));
    }

    #[tokio::test]
    async fn grep_skips_binary_files() {
        let dir = TempDir::new().unwrap();
        let mut bytes = b"needle".to_vec();
        bytes.push(0);
        bytes.extend_from_slice(b"needle again");
        fs::write(dir.path().join("blob.bin"), bytes).unwrap();
        write(dir.path(), "plain.txt", "needle\n");

        let out = grep(json!({ "pattern": "needle", "path": dir.path() }))
            .await
            .unwrap();
        assert!(!out.contains("blob.bin"), "unexpected output: {}", out);
        assert!(out.contains("plain.txt:1: needle"));
    }

    #[tokio::test]
    async fn grep_skips_oversized_files() {
        let dir = TempDir::new().unwrap();
        let mut big = String::from("needle\n");
        big.push_str(&"a".repeat(MAX_FILE_BYTES as usize + 1));
        write(dir.path(), "big.txt", &big);

        let out = grep(json!({ "pattern": "needle", "path": dir.path() }))
            .await
            .unwrap();
        assert_eq!(out, "No matches found for pattern 'needle'");
    }

    #[tokio::test]
    async fn grep_rejects_invalid_regex() {
        let dir = fixture();
        let err = grep(json!({ "pattern": "a(b", "path": dir.path() }))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("Invalid regular expression"),
            "unexpected error: {}",
            err
        );
    }

    #[tokio::test]
    async fn grep_rejects_missing_pattern_and_bad_arguments() {
        let dir = fixture();
        assert!(grep(json!({ "path": dir.path() })).await.is_err());
        assert!(grep(json!({ "pattern": "x", "max_results": 0 }))
            .await
            .is_err());
        assert!(grep(json!({ "pattern": "x", "case_insensitive": "yes" }))
            .await
            .is_err());
        assert!(grep(json!({ "pattern": "x", "path": "/no/such/dir/here" }))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn find_matches_bare_name_glob() {
        let dir = fixture();
        let out = find(json!({ "pattern": "*.rs", "path": dir.path() }))
            .await
            .unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines, vec!["src/deep/mod.rs", "src/main.rs"]);
    }

    #[tokio::test]
    async fn find_matches_path_glob() {
        let dir = fixture();
        write(dir.path(), "root.rs", "");
        let out = find(json!({ "pattern": "src/**/*.rs", "path": dir.path() }))
            .await
            .unwrap();
        assert!(
            out.contains("src/deep/mod.rs"),
            "unexpected output: {}",
            out
        );
        assert!(!out.contains("root.rs"), "unexpected output: {}", out);
    }

    #[tokio::test]
    async fn find_skips_gitignored_files() {
        let dir = fixture();
        write(dir.path(), ".gitignore", "ignored.rs\n");
        write(dir.path(), "ignored.rs", "");
        let out = find(json!({ "pattern": "*.rs", "path": dir.path() }))
            .await
            .unwrap();
        assert!(!out.contains("ignored.rs"), "unexpected output: {}", out);
    }

    #[tokio::test]
    async fn find_reports_truncation_and_no_matches() {
        let dir = fixture();
        let out = find(json!({ "pattern": "*.rs", "path": dir.path(), "max_results": 1 }))
            .await
            .unwrap();
        assert!(out.starts_with("src/deep/mod.rs"));
        assert!(out.contains("Results truncated at 1 files"));

        let empty = find(json!({ "pattern": "*.zzz", "path": dir.path() }))
            .await
            .unwrap();
        assert_eq!(empty, "No files matched glob '*.zzz'");
    }

    #[tokio::test]
    async fn find_rejects_invalid_glob() {
        let dir = fixture();
        let err = find(json!({ "pattern": "src/[", "path": dir.path() }))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("Invalid glob"),
            "unexpected error: {}",
            err
        );
    }

    #[tokio::test]
    async fn grep_names_the_file_when_path_is_a_single_file() {
        let dir = fixture();
        let file = dir.path().join("notes.txt");
        let out = grep(json!({ "pattern": "needle", "path": file }))
            .await
            .unwrap();
        assert_eq!(out, "notes.txt:1: plain needle text");
    }

    #[tokio::test]
    async fn find_names_the_file_when_path_is_a_single_file() {
        let dir = fixture();
        let file = dir.path().join("notes.txt");
        let out = find(json!({ "pattern": "*.txt", "path": file }))
            .await
            .unwrap();
        assert_eq!(out, "notes.txt");
    }

    #[tokio::test]
    async fn search_sees_dotfiles_but_never_descends_into_dot_git() {
        let dir = fixture();
        write(dir.path(), ".env.example", "needle in a dotfile\n");
        write(
            dir.path(),
            ".github/workflows/ci.yml",
            "needle in a dot dir\n",
        );
        write(
            dir.path(),
            ".git/objects/blob",
            "needle inside the git dir\n",
        );

        let out = grep(json!({ "pattern": "needle", "path": dir.path() }))
            .await
            .unwrap();
        assert!(
            out.contains(".env.example:1: needle in a dotfile"),
            "dotfiles must be searchable: {}",
            out
        );
        assert!(
            out.contains(".github/workflows/ci.yml:1: needle in a dot dir"),
            "dot directories must be searchable: {}",
            out
        );
        assert!(!out.contains(".git/objects"), "must skip .git: {}", out);

        let listed = find(json!({ "pattern": "**/*", "path": dir.path() }))
            .await
            .unwrap();
        assert!(listed.contains(".env.example"), "unexpected: {}", listed);
        assert!(!listed.contains(".git/objects"), "unexpected: {}", listed);
    }

    #[tokio::test]
    async fn gitignore_outside_the_search_root_does_not_hide_results() {
        // A .gitignore in a *parent* of the search root must not silently drop matches,
        // otherwise results depend on state the caller never asked about.
        let outer = TempDir::new().unwrap();
        write(outer.path(), ".gitignore", "notes.txt\n");
        let inner = outer.path().join("project");
        write(&inner, "notes.txt", "needle text\n");

        let out = grep(json!({ "pattern": "needle", "path": inner }))
            .await
            .unwrap();
        assert_eq!(out, "notes.txt:1: needle text");
    }

    #[tokio::test]
    async fn grep_marks_lines_it_had_to_truncate() {
        let dir = TempDir::new().unwrap();
        let long = format!("needle {}", "x".repeat(MAX_LINE_BYTES + 100));
        write(dir.path(), "long.txt", &long);

        let out = grep(json!({ "pattern": "needle", "path": dir.path() }))
            .await
            .unwrap();
        assert!(
            out.starts_with("long.txt:1: needle "),
            "unexpected: {}",
            out
        );
        assert!(
            out.ends_with(TRUNCATED_LINE_MARKER),
            "truncated lines must be marked: {}",
            &out[out.len() - 40..]
        );
    }

    #[tokio::test]
    async fn grep_does_not_mark_lines_that_fit() {
        let dir = fixture();
        let out = grep(json!({ "pattern": "needle", "path": dir.path() }))
            .await
            .unwrap();
        assert!(!out.contains(TRUNCATED_LINE_MARKER), "unexpected: {}", out);
    }

    #[tokio::test]
    async fn search_tools_do_not_require_approval() {
        assert!(!GrepSearchTool.requires_approval());
        assert!(!FindFilesByNameTool.requires_approval());
        assert_eq!(GrepSearchTool.name(), "grep_search");
        assert_eq!(FindFilesByNameTool.name(), "find_files_by_name");
    }
}
