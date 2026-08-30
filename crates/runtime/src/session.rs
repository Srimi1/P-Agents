//! Append-only JSONL session log. One record per line so a crashed run leaves
//! everything up to the last complete line replayable.

use agent_core::types::{ChatMessage, TokenUsage};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::fs::{File, OpenOptions};
use tokio::io::AsyncWriteExt;
use tracing::warn;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SessionRecord {
    Meta {
        session_id: String,
        created_at: u64,
        model: String,
    },
    Message {
        agent_id: String,
        message: ChatMessage,
    },
    Usage {
        agent_id: String,
        usage: TokenUsage,
    },
}

pub struct SessionStore {
    session_id: String,
    path: PathBuf,
    file: File,
}

impl SessionStore {
    pub async fn create(dir: &Path, model: &str) -> Result<Self> {
        tokio::fs::create_dir_all(dir)
            .await
            .with_context(|| format!("creating session directory {}", dir.display()))?;

        let session_id = Uuid::new_v4().to_string();
        let path = dir.join(format!("{session_id}.jsonl"));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .with_context(|| format!("opening session log {}", path.display()))?;

        let mut store = Self {
            session_id,
            path,
            file,
        };
        let meta = SessionRecord::Meta {
            session_id: store.session_id.clone(),
            created_at: unix_seconds(),
            model: model.to_string(),
        };
        store.append(&meta).await?;
        Ok(store)
    }

    pub async fn append(&mut self, rec: &SessionRecord) -> Result<()> {
        let mut line = serde_json::to_string(rec).context("serializing session record")?;
        line.push('\n');
        self.file
            .write_all(line.as_bytes())
            .await
            .with_context(|| format!("writing to session log {}", self.path.display()))?;
        self.file
            .flush()
            .await
            .with_context(|| format!("flushing session log {}", self.path.display()))?;
        Ok(())
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub async fn load(path: &Path) -> Result<Vec<SessionRecord>> {
        let raw = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("reading session log {}", path.display()))?;

        let mut records = Vec::new();
        let mut lines = raw.lines().filter(|l| !l.trim().is_empty()).peekable();
        while let Some(line) = lines.next() {
            match serde_json::from_str::<SessionRecord>(line) {
                Ok(rec) => records.push(rec),
                // A run killed mid-write leaves a half-written final line.
                // Everything before it is still good.
                Err(err) if lines.peek().is_none() => {
                    warn!(
                        path = %path.display(),
                        error = %err,
                        "Dropping truncated final session record"
                    );
                }
                Err(err) => {
                    return Err(err).with_context(|| {
                        format!(
                            "parsing record {} of session log {}",
                            records.len() + 1,
                            path.display()
                        )
                    })
                }
            }
        }
        Ok(records)
    }

    /// Messages belonging to `agent_id`, in the order they were appended.
    pub fn rebuild_history(records: &[SessionRecord], agent_id: &str) -> Vec<ChatMessage> {
        records
            .iter()
            .filter_map(|rec| match rec {
                SessionRecord::Message {
                    agent_id: id,
                    message,
                } if id == agent_id => Some(message.clone()),
                _ => None,
            })
            .collect()
    }

    /// Resolves `--resume <id>`, accepting an unambiguous id prefix.
    pub async fn find_by_id(dir: &Path, id: &str) -> Result<PathBuf> {
        let exact = dir.join(format!("{id}.jsonl"));
        if exact.is_file() {
            return Ok(exact);
        }

        let mut entries = tokio::fs::read_dir(dir)
            .await
            .with_context(|| format!("listing session directory {}", dir.display()))?;
        let mut matches = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .with_context(|| format!("listing session directory {}", dir.display()))?
        {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if path
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|stem| stem.starts_with(id))
            {
                matches.push(path);
            }
        }

        match matches.len() {
            0 => anyhow::bail!("no session matching '{id}' in {}", dir.display()),
            1 => Ok(matches.remove(0)),
            n => anyhow::bail!(
                "'{id}' matches {n} sessions in {}; pass the full session id",
                dir.display()
            ),
        }
    }
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::types::Role;

    fn msg(agent_id: &str, text: &str) -> SessionRecord {
        SessionRecord::Message {
            agent_id: agent_id.to_string(),
            message: ChatMessage::user(text),
        }
    }

    #[tokio::test]
    async fn create_writes_meta_and_round_trips_records() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut store = SessionStore::create(dir.path(), "claude-test")
            .await
            .unwrap();
        let path = store.path().to_path_buf();
        assert!(path.starts_with(dir.path()));
        assert!(path.to_string_lossy().contains(store.session_id()));

        store.append(&msg("lead", "hello")).await.unwrap();
        store
            .append(&SessionRecord::Usage {
                agent_id: "lead".into(),
                usage: TokenUsage::new(10, 5),
            })
            .await
            .unwrap();

        let records = SessionStore::load(&path).await.unwrap();
        assert_eq!(records.len(), 3);
        match &records[0] {
            SessionRecord::Meta {
                session_id,
                created_at,
                model,
            } => {
                assert_eq!(session_id, store.session_id());
                assert_eq!(model, "claude-test");
                assert!(*created_at > 1_600_000_000);
            }
            other => panic!("expected meta first, got {other:?}"),
        }
        match &records[1] {
            SessionRecord::Message { agent_id, message } => {
                assert_eq!(agent_id, "lead");
                assert_eq!(message.role, Role::User);
                assert_eq!(message.content.as_deref(), Some("hello"));
            }
            other => panic!("expected message, got {other:?}"),
        }
        match &records[2] {
            SessionRecord::Usage { usage, .. } => assert_eq!(usage.total_tokens, 15),
            other => panic!("expected usage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn create_makes_missing_directories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let nested = dir.path().join("a").join("b");
        let store = SessionStore::create(&nested, "m").await.unwrap();
        assert!(store.path().is_file());
    }

    #[tokio::test]
    async fn rebuild_history_filters_by_agent_and_keeps_order() {
        let records = vec![
            SessionRecord::Meta {
                session_id: "s".into(),
                created_at: 0,
                model: "m".into(),
            },
            msg("lead", "one"),
            msg("sub-1", "not mine"),
            SessionRecord::Usage {
                agent_id: "lead".into(),
                usage: TokenUsage::new(1, 1),
            },
            msg("lead", "two"),
        ];

        let history = SessionStore::rebuild_history(&records, "lead");
        let texts: Vec<_> = history
            .iter()
            .map(|m| m.content.clone().unwrap_or_default())
            .collect();
        assert_eq!(texts, vec!["one", "two"]);
        assert!(SessionStore::rebuild_history(&records, "nobody").is_empty());
    }

    #[tokio::test]
    async fn load_tolerates_a_truncated_final_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut store = SessionStore::create(dir.path(), "m").await.unwrap();
        store.append(&msg("lead", "kept")).await.unwrap();
        let path = store.path().to_path_buf();
        drop(store);

        let mut raw = tokio::fs::read_to_string(&path).await.unwrap();
        raw.push_str("{\"type\":\"message\",\"agent_id\":\"lead\",\"mess");
        tokio::fs::write(&path, raw).await.unwrap();

        let records = SessionStore::load(&path).await.unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(
            SessionStore::rebuild_history(&records, "lead")
                .first()
                .and_then(|m| m.content.clone())
                .as_deref(),
            Some("kept")
        );
    }

    #[tokio::test]
    async fn load_rejects_corruption_before_the_last_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("broken.jsonl");
        tokio::fs::write(&path, "not json\n{\"type\":\"usage\",\"agent_id\":\"a\",\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n")
            .await
            .unwrap();
        assert!(SessionStore::load(&path).await.is_err());
    }

    #[tokio::test]
    async fn load_of_a_missing_file_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(SessionStore::load(&dir.path().join("nope.jsonl"))
            .await
            .is_err());
    }

    #[tokio::test]
    async fn find_by_id_resolves_exact_and_prefix() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SessionStore::create(dir.path(), "m").await.unwrap();
        let id = store.session_id().to_string();

        let exact = SessionStore::find_by_id(dir.path(), &id).await.unwrap();
        assert_eq!(exact, store.path());

        let prefix = SessionStore::find_by_id(dir.path(), &id[..8])
            .await
            .unwrap();
        assert_eq!(prefix, store.path());

        assert!(SessionStore::find_by_id(dir.path(), "deadbeef")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn find_by_id_rejects_ambiguous_prefixes() {
        let dir = tempfile::tempdir().expect("tempdir");
        tokio::fs::write(dir.path().join("abc-1.jsonl"), "")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("abc-2.jsonl"), "")
            .await
            .unwrap();
        let err = SessionStore::find_by_id(dir.path(), "abc")
            .await
            .expect_err("ambiguous");
        assert!(err.to_string().contains("matches 2 sessions"));
    }
}
