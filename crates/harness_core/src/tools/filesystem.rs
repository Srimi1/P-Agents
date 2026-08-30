use crate::tool_registry::Tool;
use agent_core::truncate_at_boundary;
use anyhow::{Context, Result};
use async_trait::async_trait;
use ignore::WalkBuilder;
use serde_json::json;
use std::path::{Path, PathBuf};
use tokio::fs;

/// A single minified line must not be allowed to dominate the model's context.
const MAX_LINE_BYTES: usize = 2000;
const DEFAULT_READ_LINES: usize = 2000;
const MAX_LIST_ENTRIES: usize = 500;
/// Whole-file read guard. `read_file` slurps the file before paging it, so without
/// this a single huge file (or a character device such as /dev/zero, whose reported
/// length is 0) can exhaust the harness's memory.
const MAX_READ_BYTES: u64 = 16 * 1024 * 1024;

fn require_str<'a>(args: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    args[key]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing or non-string '{}' argument", key))
}

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Reads a text file and returns its contents with 1-based line numbers, so lines can be cited exactly. Use offset and limit to page through large files."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read."
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start reading from. Defaults to 1."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return."
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let path_str = require_str(&args, "path")?;
        let path = Path::new(path_str);

        let metadata = fs::metadata(path)
            .await
            .with_context(|| format!("File not found: '{}'", path_str))?;
        if metadata.is_dir() {
            anyhow::bail!("'{}' is a directory, not a file", path_str);
        }
        // Character devices, FIFOs and sockets are neither dirs nor regular files;
        // reading them can block forever or never reach EOF.
        if !metadata.is_file() {
            anyhow::bail!("'{}' is not a regular file", path_str);
        }
        if metadata.len() > MAX_READ_BYTES {
            anyhow::bail!(
                "'{}' is {} bytes, which exceeds the {} byte read limit; use grep_search to locate the relevant region instead",
                path_str,
                metadata.len(),
                MAX_READ_BYTES
            );
        }

        let content = fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read '{}' as UTF-8 text", path_str))?;

        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        if total == 0 {
            return Ok(format!("['{}' is empty]", path_str));
        }

        let start = match args.get("offset") {
            Some(serde_json::Value::Null) | None => 1usize,
            Some(v) => v
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("'offset' must be a positive integer"))?
                .max(1) as usize,
        };
        let limit = match args.get("limit") {
            Some(serde_json::Value::Null) | None => DEFAULT_READ_LINES,
            Some(v) => {
                let n = v
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("'limit' must be a positive integer"))?;
                if n == 0 {
                    anyhow::bail!("'limit' must be at least 1");
                }
                n as usize
            }
        };

        if start > total {
            anyhow::bail!(
                "offset {} is past the end of '{}', which has {} lines",
                start,
                path_str,
                total
            );
        }

        let end = start.saturating_sub(1).saturating_add(limit).min(total);
        let mut out = String::new();
        for (i, line) in lines[start - 1..end].iter().enumerate() {
            let shown = truncate_at_boundary(line, MAX_LINE_BYTES);
            out.push_str(&format!("{}\t{}", start + i, shown));
            if shown.len() < line.len() {
                out.push_str(" [line truncated]");
            }
            out.push('\n');
        }

        if start > 1 || end < total {
            out.push_str(&format!(
                "\n[Showed lines {}-{} of {} total lines in '{}']",
                start, end, total, path_str
            ));
        }

        Ok(out)
    }
}

pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Creates or overwrites a file with the specified content, creating parent directories as needed."
    }

    fn requires_approval(&self) -> bool {
        true
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to write."
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write."
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let path_str = require_str(&args, "path")?;
        let content = require_str(&args, "content")?;

        let path = Path::new(path_str);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).await.with_context(|| {
                    format!("Failed to create parent directories for '{}'", path_str)
                })?;
            }
        }

        fs::write(path, content)
            .await
            .with_context(|| format!("Failed to write '{}'", path_str))?;
        Ok(format!("Wrote {} bytes to '{}'", content.len(), path_str))
    }
}

pub struct ListDirTool;

#[async_trait]
impl Tool for ListDirTool {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn description(&self) -> &str {
        "Lists directory entries, honouring .gitignore and skipping .git. Set recursive to descend, optionally bounded by max_depth."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Defaults to the current directory."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Descend into subdirectories. Defaults to false (immediate children only)."
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth when recursive; 1 means immediate children."
                }
            },
            "required": []
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let path_str = args["path"].as_str().unwrap_or(".").to_string();
        let recursive = args["recursive"].as_bool().unwrap_or(false);
        let max_depth = match args.get("max_depth") {
            Some(serde_json::Value::Null) | None => None,
            Some(v) => Some(
                v.as_u64()
                    .ok_or_else(|| anyhow::anyhow!("'max_depth' must be a positive integer"))?
                    .max(1) as usize,
            ),
        };

        let root = PathBuf::from(&path_str);
        let metadata = fs::metadata(&root)
            .await
            .with_context(|| format!("Directory not found: '{}'", path_str))?;
        if !metadata.is_dir() {
            anyhow::bail!("'{}' is not a directory", path_str);
        }

        let depth = if recursive { max_depth } else { Some(1) };
        let walk_root = root.clone();

        let (mut entries, total, skipped) =
            tokio::task::spawn_blocking(move || -> Result<(Vec<(bool, String)>, usize, usize)> {
                let mut builder = WalkBuilder::new(&walk_root);
                builder
                    .hidden(false)
                    .parents(false)
                    .git_global(false)
                    .git_ignore(true)
                    .git_exclude(true)
                    // .gitignore files apply even when the tree is not a git checkout.
                    .require_git(false)
                    .follow_links(false)
                    .max_depth(depth)
                    .filter_entry(|entry| entry.file_name() != ".git");

                let mut collected = Vec::new();
                let mut skipped = 0usize;
                for result in builder.build() {
                    // One unreadable subdirectory must not destroy the whole listing:
                    // count it and keep walking rather than aborting.
                    let entry = match result {
                        Ok(entry) => entry,
                        Err(_) => {
                            skipped += 1;
                            continue;
                        }
                    };
                    if entry.depth() == 0 {
                        continue;
                    }
                    let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
                    let rel = entry
                        .path()
                        .strip_prefix(&walk_root)
                        .unwrap_or_else(|_| entry.path());
                    collected.push((is_dir, rel.to_string_lossy().to_string()));
                }

                collected.sort_by(|a, b| a.1.cmp(&b.1));
                let total = collected.len();
                Ok((collected, total, skipped))
            })
            .await
            .context("Directory walk task failed")??;

        let skipped_note = if skipped > 0 {
            format!(
                "\n\n[{} path(s) under '{}' could not be read and were skipped]",
                skipped, path_str
            )
        } else {
            String::new()
        };

        if entries.is_empty() {
            return Ok(format!("[No entries under '{}']{}", path_str, skipped_note));
        }

        let truncated = entries.len() > MAX_LIST_ENTRIES;
        entries.truncate(MAX_LIST_ENTRIES);

        let mut out = entries
            .into_iter()
            .map(|(is_dir, name)| {
                let kind = if is_dir { "DIR" } else { "FILE" };
                format!("[{}] {}", kind, name)
            })
            .collect::<Vec<_>>()
            .join("\n");

        if truncated {
            out.push_str(&format!(
                "\n\n[Truncated: showing the first {} of {} entries under '{}']",
                MAX_LIST_ENTRIES, total, path_str
            ));
        }
        out.push_str(&skipped_note);

        Ok(out)
    }
}

pub struct EditFileBlockTool;

#[async_trait]
impl Tool for EditFileBlockTool {
    fn name(&self) -> &str {
        "edit_file_block"
    }

    fn description(&self) -> &str {
        "Replaces an exact block of text inside a file. old_string must be unique unless replace_all is set."
    }

    fn requires_approval(&self) -> bool {
        true
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to edit."
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace, including surrounding context needed to make it unique."
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text."
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence instead of requiring a unique match. Defaults to false."
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn execute(&self, args: serde_json::Value) -> Result<String> {
        let path_str = require_str(&args, "path")?;
        let old_string = require_str(&args, "old_string")?;
        let new_string = require_str(&args, "new_string")?;
        let replace_all = args["replace_all"].as_bool().unwrap_or(false);

        if old_string.is_empty() {
            anyhow::bail!("'old_string' must not be empty");
        }
        if old_string == new_string {
            anyhow::bail!("'old_string' and 'new_string' are identical; nothing to do");
        }

        let path = Path::new(path_str);
        let content = fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read '{}' for editing", path_str))?;

        let matches = content.matches(old_string).count();
        if matches == 0 {
            anyhow::bail!(
                "No occurrence of old_string found in '{}'; the file was not modified",
                path_str
            );
        }
        if matches > 1 && !replace_all {
            anyhow::bail!(
                "old_string is ambiguous: {} matches found in '{}'. Add surrounding context to make it unique, or set replace_all to true. The file was not modified",
                matches,
                path_str
            );
        }

        let (updated, replaced) = if replace_all {
            (content.replace(old_string, new_string), matches)
        } else {
            (content.replacen(old_string, new_string, 1), 1)
        };

        fs::write(path, &updated)
            .await
            .with_context(|| format!("Failed to write '{}'", path_str))?;

        Ok(format!(
            "Made {} replacement(s) in '{}'",
            replaced, path_str
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn path_of(dir: &tempfile::TempDir, name: &str) -> String {
        dir.path().join(name).to_string_lossy().to_string()
    }

    #[tokio::test]
    async fn read_numbers_lines() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "a.txt");
        std::fs::write(&file, "alpha\nbeta\n").unwrap();

        let out = ReadFileTool.execute(json!({ "path": file })).await.unwrap();
        assert_eq!(out, "1\talpha\n2\tbeta\n");
    }

    #[tokio::test]
    async fn read_honours_offset_and_limit_and_reports_range() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "many.txt");
        let body: String = (1..=10).map(|n| format!("line{}\n", n)).collect();
        std::fs::write(&file, body).unwrap();

        let out = ReadFileTool
            .execute(json!({ "path": file.clone(), "offset": 3, "limit": 2 }))
            .await
            .unwrap();

        assert!(out.contains("3\tline3\n"), "{}", out);
        assert!(out.contains("4\tline4\n"), "{}", out);
        assert!(!out.contains("line5"), "{}", out);
        assert!(
            out.contains("Showed lines 3-4 of 10 total lines"),
            "{}",
            out
        );
    }

    #[tokio::test]
    async fn read_without_limit_has_no_truncation_note() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "small.txt");
        std::fs::write(&file, "one\ntwo\n").unwrap();

        let out = ReadFileTool.execute(json!({ "path": file })).await.unwrap();
        assert!(!out.contains("Showed lines"), "{}", out);
    }

    #[tokio::test]
    async fn read_offset_past_end_errors() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "short.txt");
        std::fs::write(&file, "only\n").unwrap();

        let err = ReadFileTool
            .execute(json!({ "path": file, "offset": 9 }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("past the end"), "{}", err);
    }

    #[tokio::test]
    async fn read_missing_path_errors() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "nope.txt");

        let err = ReadFileTool
            .execute(json!({ "path": file }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("File not found"), "{}", err);
    }

    #[tokio::test]
    async fn read_directory_errors() {
        let dir = tempdir().unwrap();
        let err = ReadFileTool
            .execute(json!({ "path": dir.path().to_string_lossy() }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("is a directory"), "{}", err);
    }

    #[tokio::test]
    async fn read_multibyte_line_truncates_without_panic() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "utf8.txt");
        let long_line = "あ".repeat(1000);
        std::fs::write(&file, format!("日本語テキスト\n{}\n", long_line)).unwrap();

        let out = ReadFileTool.execute(json!({ "path": file })).await.unwrap();

        assert!(out.contains("1\t日本語テキスト"), "{}", out);
        assert!(out.contains("[line truncated]"), "{}", out);
        // Cut must land on a char boundary, so no replacement chars appear.
        assert!(!out.contains('\u{fffd}'));
    }

    #[tokio::test]
    async fn read_rejects_zero_limit() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "x.txt");
        std::fs::write(&file, "a\n").unwrap();

        let err = ReadFileTool
            .execute(json!({ "path": file, "limit": 0 }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least 1"), "{}", err);
    }

    #[tokio::test]
    async fn write_creates_parent_directories() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "nested/deeper/out.txt");

        let out = WriteFileTool
            .execute(json!({ "path": file.clone(), "content": "hello" }))
            .await
            .unwrap();

        assert!(out.contains("5 bytes"), "{}", out);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "hello");
        assert!(WriteFileTool.requires_approval());
    }

    #[tokio::test]
    async fn write_requires_content() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "out.txt");

        let err = WriteFileTool
            .execute(json!({ "path": file }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("'content'"), "{}", err);
    }

    #[tokio::test]
    async fn edit_replaces_single_match() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "code.rs");
        std::fs::write(&file, "let a = 1;\nlet b = 2;\n").unwrap();

        let out = EditFileBlockTool
            .execute(json!({
                "path": file.clone(),
                "old_string": "let b = 2;",
                "new_string": "let b = 3;"
            }))
            .await
            .unwrap();

        assert!(out.contains("Made 1 replacement"), "{}", out);
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "let a = 1;\nlet b = 3;\n"
        );
        assert!(EditFileBlockTool.requires_approval());
    }

    #[tokio::test]
    async fn edit_with_zero_matches_errors_and_leaves_file() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "code.rs");
        let original = "let a = 1;\n";
        std::fs::write(&file, original).unwrap();

        let err = EditFileBlockTool
            .execute(json!({
                "path": file.clone(),
                "old_string": "let z = 9;",
                "new_string": "let z = 8;"
            }))
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("No occurrence"), "{}", err);
        assert!(err.contains("code.rs"), "{}", err);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), original);
    }

    #[tokio::test]
    async fn edit_with_ambiguous_match_errors_and_leaves_file() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "code.rs");
        let original = "value\nvalue\n";
        std::fs::write(&file, original).unwrap();

        let err = EditFileBlockTool
            .execute(json!({
                "path": file.clone(),
                "old_string": "value",
                "new_string": "other"
            }))
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("ambiguous"), "{}", err);
        assert!(err.contains("2 matches"), "{}", err);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), original);
    }

    #[tokio::test]
    async fn edit_replace_all_replaces_every_match() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "code.rs");
        std::fs::write(&file, "value\nvalue\n").unwrap();

        let out = EditFileBlockTool
            .execute(json!({
                "path": file.clone(),
                "old_string": "value",
                "new_string": "other",
                "replace_all": true
            }))
            .await
            .unwrap();

        assert!(out.contains("Made 2 replacement"), "{}", out);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "other\nother\n");
    }

    #[tokio::test]
    async fn edit_missing_file_errors() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "absent.rs");

        let err = EditFileBlockTool
            .execute(json!({
                "path": file,
                "old_string": "a",
                "new_string": "b"
            }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("absent.rs"), "{}", err);
    }

    #[tokio::test]
    async fn edit_rejects_empty_old_string() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "code.rs");
        std::fs::write(&file, "x\n").unwrap();

        let err = EditFileBlockTool
            .execute(json!({
                "path": file,
                "old_string": "",
                "new_string": "b"
            }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("must not be empty"), "{}", err);
    }

    fn build_tree(dir: &tempfile::TempDir) {
        let root = dir.path();
        std::fs::write(root.join(".gitignore"), "ignored.txt\ntarget/\n").unwrap();
        std::fs::write(root.join("keep.txt"), "keep").unwrap();
        std::fs::write(root.join("ignored.txt"), "hidden from listing").unwrap();
        std::fs::create_dir_all(root.join("src/inner")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("src/inner/deep.rs"), "deep").unwrap();
        std::fs::create_dir_all(root.join("target")).unwrap();
        std::fs::write(root.join("target/artifact.bin"), "junk").unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/HEAD"), "ref: refs/heads/main").unwrap();
    }

    #[tokio::test]
    async fn list_directory_respects_gitignore_and_skips_git() {
        let dir = tempdir().unwrap();
        build_tree(&dir);

        let out = ListDirTool
            .execute(json!({ "path": dir.path().to_string_lossy(), "recursive": true }))
            .await
            .unwrap();

        assert!(out.contains("[FILE] keep.txt"), "{}", out);
        assert!(out.contains("[DIR] src"), "{}", out);
        assert!(out.contains("[FILE] src/inner/deep.rs"), "{}", out);
        assert!(!out.contains("ignored.txt"), "{}", out);
        assert!(!out.contains("target"), "{}", out);
        assert!(!out.contains(".git/"), "{}", out);
        assert!(!out.contains("HEAD"), "{}", out);
    }

    #[tokio::test]
    async fn list_directory_honours_max_depth() {
        let dir = tempdir().unwrap();
        build_tree(&dir);

        let shallow = ListDirTool
            .execute(json!({
                "path": dir.path().to_string_lossy(),
                "recursive": true,
                "max_depth": 1
            }))
            .await
            .unwrap();
        assert!(shallow.contains("[DIR] src"), "{}", shallow);
        assert!(!shallow.contains("main.rs"), "{}", shallow);

        let deeper = ListDirTool
            .execute(json!({
                "path": dir.path().to_string_lossy(),
                "recursive": true,
                "max_depth": 2
            }))
            .await
            .unwrap();
        assert!(deeper.contains("src/main.rs"), "{}", deeper);
        assert!(!deeper.contains("deep.rs"), "{}", deeper);
    }

    #[tokio::test]
    async fn list_directory_defaults_to_immediate_children() {
        let dir = tempdir().unwrap();
        build_tree(&dir);

        let out = ListDirTool
            .execute(json!({ "path": dir.path().to_string_lossy() }))
            .await
            .unwrap();
        assert!(out.contains("[DIR] src"), "{}", out);
        assert!(!out.contains("main.rs"), "{}", out);
    }

    #[tokio::test]
    async fn list_directory_is_sorted_and_named_per_blueprint() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("b.txt"), "b").unwrap();
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();

        let out = ListDirTool
            .execute(json!({ "path": dir.path().to_string_lossy() }))
            .await
            .unwrap();
        assert_eq!(out, "[FILE] a.txt\n[FILE] b.txt");
        assert_eq!(ListDirTool.name(), "list_directory");
    }

    #[tokio::test]
    async fn list_directory_truncates_large_listings() {
        let dir = tempdir().unwrap();
        for n in 0..(MAX_LIST_ENTRIES + 10) {
            std::fs::write(dir.path().join(format!("f{:04}.txt", n)), "x").unwrap();
        }

        let out = ListDirTool
            .execute(json!({ "path": dir.path().to_string_lossy() }))
            .await
            .unwrap();
        assert!(out.contains("Truncated: showing the first 500"), "{}", out);
        assert!(
            out.contains(&format!("of {} entries", MAX_LIST_ENTRIES + 10)),
            "{}",
            out
        );
    }

    #[tokio::test]
    async fn list_directory_rejects_non_directory() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "f.txt");
        std::fs::write(&file, "x").unwrap();

        let err = ListDirTool
            .execute(json!({ "path": file }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("is not a directory"), "{}", err);
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn list_directory_survives_unreadable_subdirectory() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("visible.txt"), "x").unwrap();
        let locked = dir.path().join("locked");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::write(locked.join("inner.txt"), "y").unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();

        let res = ListDirTool
            .execute(json!({ "path": dir.path().to_string_lossy(), "recursive": true }))
            .await;
        // Restore before asserting so the TempDir can always clean itself up.
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();

        let out = res.expect("one unreadable subdir must not fail the whole listing");
        assert!(out.contains("[FILE] visible.txt"), "{}", out);
        assert!(
            out.contains("could not be read and were skipped"),
            "{}",
            out
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn read_rejects_non_regular_file() {
        // /dev/zero reports len() == 0 and is neither a dir nor a regular file;
        // slurping it would allocate until the process dies.
        let err = ReadFileTool
            .execute(json!({ "path": "/dev/zero" }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a regular file"), "{}", err);
    }

    #[tokio::test]
    async fn read_rejects_oversized_file() {
        let dir = tempdir().unwrap();
        let file = path_of(&dir, "huge.txt");
        let f = std::fs::File::create(&file).unwrap();
        // Sparse allocation: no bytes are actually written to disk.
        f.set_len(MAX_READ_BYTES + 1).unwrap();
        drop(f);

        let err = ReadFileTool
            .execute(json!({ "path": file }))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("exceeds"), "{}", err);
    }

    #[tokio::test]
    async fn list_directory_reports_empty_tree() {
        let dir = tempdir().unwrap();
        let out = ListDirTool
            .execute(json!({ "path": dir.path().to_string_lossy() }))
            .await
            .unwrap();
        assert!(out.contains("No entries under"), "{}", out);
    }
}
