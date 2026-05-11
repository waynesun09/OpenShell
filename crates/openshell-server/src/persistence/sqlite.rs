// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    DraftChunkRecord, ObjectRecord, PersistenceResult, PolicyRecord, current_time_ms, map_db_error,
    map_migrate_error,
};
use crate::policy_store::{
    draft_chunk_payload_from_record, draft_chunk_record_from_parts, policy_payload_from_record,
    policy_record_from_parts,
};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Row, SqlitePool};
use std::str::FromStr;

static SQLITE_MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations/sqlite");

const POLICY_OBJECT_TYPE: &str = "sandbox_policy";
const DRAFT_CHUNK_OBJECT_TYPE: &str = "draft_policy_chunk";

#[derive(Debug, Clone)]
pub struct SqliteStore {
    pool: SqlitePool,
}

impl SqliteStore {
    /// Test-only accessor for raw pool access (e.g., installing failure
    /// triggers to drive fault-injection tests in sibling modules).
    #[cfg(test)]
    pub(crate) fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub async fn connect(url: &str) -> PersistenceResult<Self> {
        let is_in_memory = url.contains(":memory:") || url.contains("mode=memory");
        let max_connections = if is_in_memory { 1 } else { 5 };

        let options = SqliteConnectOptions::from_str(url)
            .map_err(|e| map_db_error(&e))?
            .create_if_missing(true);

        let mut pool_options = SqlitePoolOptions::new()
            .max_connections(max_connections)
            .min_connections(max_connections);

        if is_in_memory {
            pool_options = pool_options.idle_timeout(None).max_lifetime(None);
        }

        let pool = pool_options
            .connect_with(options)
            .await
            .map_err(|e| map_db_error(&e))?;

        Ok(Self { pool })
    }

    pub async fn migrate(&self) -> PersistenceResult<()> {
        SQLITE_MIGRATOR
            .run(&self.pool)
            .await
            .map_err(|e| map_migrate_error(&e))
    }

    pub async fn put(
        &self,
        object_type: &str,
        id: &str,
        name: &str,
        payload: &[u8],
        labels: Option<&str>,
    ) -> PersistenceResult<()> {
        let now_ms = current_time_ms()?;

        sqlx::query(
            r#"
INSERT INTO "objects" ("object_type", "id", "name", "payload", "created_at_ms", "updated_at_ms", "labels")
VALUES (?1, ?2, ?3, ?4, ?5, ?5, ?6)
ON CONFLICT ("object_type", "name") WHERE "name" IS NOT NULL DO UPDATE SET
    "payload" = excluded."payload",
    "updated_at_ms" = excluded."updated_at_ms",
    "labels" = excluded."labels"
"#,
        )
        .bind(object_type)
        .bind(id)
        .bind(name)
        .bind(payload)
        .bind(now_ms)
        .bind(labels.unwrap_or("{}"))
        .execute(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;
        Ok(())
    }

    pub async fn get(
        &self,
        object_type: &str,
        id: &str,
    ) -> PersistenceResult<Option<ObjectRecord>> {
        let row = sqlx::query(
            r#"
SELECT "object_type", "id", "name", "payload", "created_at_ms", "updated_at_ms", "labels"
FROM "objects"
WHERE "object_type" = ?1 AND "id" = ?2
"#,
        )
        .bind(object_type)
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        Ok(row.map(row_to_object_record))
    }

    pub async fn get_by_name(
        &self,
        object_type: &str,
        name: &str,
    ) -> PersistenceResult<Option<ObjectRecord>> {
        let row = sqlx::query(
            r#"
SELECT "object_type", "id", "name", "payload", "created_at_ms", "updated_at_ms", "labels"
FROM "objects"
WHERE "object_type" = ?1 AND "name" = ?2
"#,
        )
        .bind(object_type)
        .bind(name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        Ok(row.map(row_to_object_record))
    }

    pub async fn delete(&self, object_type: &str, id: &str) -> PersistenceResult<bool> {
        let result = sqlx::query(
            r#"
DELETE FROM "objects"
WHERE "object_type" = ?1 AND "id" = ?2
"#,
        )
        .bind(object_type)
        .bind(id)
        .execute(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn delete_by_name(&self, object_type: &str, name: &str) -> PersistenceResult<bool> {
        let result = sqlx::query(
            r#"
DELETE FROM "objects"
WHERE "object_type" = ?1 AND "name" = ?2
"#,
        )
        .bind(object_type)
        .bind(name)
        .execute(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn list(
        &self,
        object_type: &str,
        limit: u32,
        offset: u32,
    ) -> PersistenceResult<Vec<ObjectRecord>> {
        let rows = sqlx::query(
            r#"
SELECT "object_type", "id", "name", "payload", "created_at_ms", "updated_at_ms", "labels"
FROM "objects"
WHERE "object_type" = ?1
ORDER BY "created_at_ms" ASC, "name" ASC
LIMIT ?2 OFFSET ?3
"#,
        )
        .bind(object_type)
        .bind(i64::from(limit))
        .bind(i64::from(offset))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        Ok(rows.into_iter().map(row_to_object_record).collect())
    }
    pub async fn list_with_selector(
        &self,
        object_type: &str,
        label_selector: &str,
        limit: u32,
        offset: u32,
    ) -> PersistenceResult<Vec<ObjectRecord>> {
        use super::parse_label_selector;

        let required_labels = parse_label_selector(label_selector)?;
        let all_records = self.list(object_type, u32::MAX, 0).await?;

        let filtered: Vec<ObjectRecord> = all_records
            .into_iter()
            .filter(|record| {
                let labels_json = record.labels.as_deref().unwrap_or("{}");
                let labels: std::collections::HashMap<String, String> =
                    serde_json::from_str(labels_json).unwrap_or_default();

                required_labels
                    .iter()
                    .all(|(key, value)| labels.get(key).is_some_and(|v| v == value))
            })
            .skip(offset as usize)
            .take(limit as usize)
            .collect();

        Ok(filtered)
    }
    pub async fn put_policy_revision(
        &self,
        id: &str,
        sandbox_id: &str,
        version: i64,
        payload: &[u8],
        hash: &str,
    ) -> PersistenceResult<()> {
        let now_ms = current_time_ms()?;
        let record = PolicyRecord {
            id: id.to_string(),
            sandbox_id: sandbox_id.to_string(),
            version,
            policy_payload: payload.to_vec(),
            policy_hash: hash.to_string(),
            status: "pending".to_string(),
            load_error: None,
            created_at_ms: now_ms,
            loaded_at_ms: None,
        };
        let wrapped_payload = policy_payload_from_record(&record)?;

        sqlx::query(
            r#"
INSERT INTO "objects" (
    "object_type", "id", "scope", "version", "status", "payload", "created_at_ms", "updated_at_ms"
)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)
"#,
        )
        .bind(POLICY_OBJECT_TYPE)
        .bind(id)
        .bind(sandbox_id)
        .bind(version)
        .bind("pending")
        .bind(wrapped_payload)
        .bind(now_ms)
        .execute(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;
        Ok(())
    }

    pub async fn get_latest_policy(
        &self,
        sandbox_id: &str,
    ) -> PersistenceResult<Option<PolicyRecord>> {
        let row = sqlx::query(
            r#"
SELECT "id", "scope", "version", "status", "payload", "created_at_ms"
FROM "objects"
WHERE "object_type" = ?1 AND "scope" = ?2
ORDER BY "version" DESC, "created_at_ms" DESC
LIMIT 1
"#,
        )
        .bind(POLICY_OBJECT_TYPE)
        .bind(sandbox_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        row.map(row_to_policy_record).transpose()
    }

    pub async fn get_latest_loaded_policy(
        &self,
        sandbox_id: &str,
    ) -> PersistenceResult<Option<PolicyRecord>> {
        let row = sqlx::query(
            r#"
SELECT "id", "scope", "version", "status", "payload", "created_at_ms"
FROM "objects"
WHERE "object_type" = ?1 AND "scope" = ?2 AND "status" = 'loaded'
ORDER BY "version" DESC, "created_at_ms" DESC
LIMIT 1
"#,
        )
        .bind(POLICY_OBJECT_TYPE)
        .bind(sandbox_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        row.map(row_to_policy_record).transpose()
    }

    pub async fn get_policy_by_version(
        &self,
        sandbox_id: &str,
        version: i64,
    ) -> PersistenceResult<Option<PolicyRecord>> {
        let row = sqlx::query(
            r#"
SELECT "id", "scope", "version", "status", "payload", "created_at_ms"
FROM "objects"
WHERE "object_type" = ?1 AND "scope" = ?2 AND "version" = ?3
"#,
        )
        .bind(POLICY_OBJECT_TYPE)
        .bind(sandbox_id)
        .bind(version)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        row.map(row_to_policy_record).transpose()
    }

    pub async fn list_policies(
        &self,
        sandbox_id: &str,
        limit: u32,
        offset: u32,
    ) -> PersistenceResult<Vec<PolicyRecord>> {
        let rows = sqlx::query(
            r#"
SELECT "id", "scope", "version", "status", "payload", "created_at_ms"
FROM "objects"
WHERE "object_type" = ?1 AND "scope" = ?2
ORDER BY "version" DESC, "created_at_ms" DESC
LIMIT ?3 OFFSET ?4
"#,
        )
        .bind(POLICY_OBJECT_TYPE)
        .bind(sandbox_id)
        .bind(i64::from(limit))
        .bind(i64::from(offset))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        rows.into_iter().map(row_to_policy_record).collect()
    }

    pub async fn update_policy_status(
        &self,
        sandbox_id: &str,
        version: i64,
        status: &str,
        load_error: Option<&str>,
        loaded_at_ms: Option<i64>,
    ) -> PersistenceResult<bool> {
        let Some(mut record) = self.get_policy_by_version(sandbox_id, version).await? else {
            return Ok(false);
        };

        record.status = status.to_string();
        record.load_error = load_error.map(ToOwned::to_owned);
        record.loaded_at_ms = loaded_at_ms;
        let payload = policy_payload_from_record(&record)?;
        let now_ms = current_time_ms()?;

        let result = sqlx::query(
            r#"
UPDATE "objects"
SET "status" = ?4, "payload" = ?5, "updated_at_ms" = ?6
WHERE "object_type" = ?1 AND "scope" = ?2 AND "version" = ?3
"#,
        )
        .bind(POLICY_OBJECT_TYPE)
        .bind(sandbox_id)
        .bind(version)
        .bind(status)
        .bind(payload)
        .bind(now_ms)
        .execute(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn supersede_older_policies(
        &self,
        sandbox_id: &str,
        before_version: i64,
    ) -> PersistenceResult<u64> {
        let now_ms = current_time_ms()?;
        let result = sqlx::query(
            r#"
UPDATE "objects"
SET "status" = 'superseded', "updated_at_ms" = ?4
WHERE "object_type" = ?1
  AND "scope" = ?2
  AND "version" < ?3
  AND "status" IN ('pending', 'loaded')
"#,
        )
        .bind(POLICY_OBJECT_TYPE)
        .bind(sandbox_id)
        .bind(before_version)
        .bind(now_ms)
        .execute(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;
        Ok(result.rows_affected())
    }

    pub async fn put_draft_chunk(&self, chunk: &DraftChunkRecord) -> PersistenceResult<()> {
        let payload = draft_chunk_payload_from_record(chunk)?;
        sqlx::query(
            r#"
INSERT INTO "objects" (
    "object_type", "id", "scope", "status", "dedup_key", "hit_count", "payload", "created_at_ms", "updated_at_ms"
)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
ON CONFLICT ("object_type", "scope", "dedup_key") WHERE "dedup_key" IS NOT NULL DO UPDATE SET
    "hit_count" = "objects"."hit_count" + excluded."hit_count",
    "updated_at_ms" = excluded."updated_at_ms"
"#,
        )
        .bind(DRAFT_CHUNK_OBJECT_TYPE)
        .bind(&chunk.id)
        .bind(&chunk.sandbox_id)
        .bind(&chunk.status)
        .bind(draft_chunk_dedup_key(chunk))
        .bind(i64::from(chunk.hit_count))
        .bind(payload)
        .bind(chunk.first_seen_ms)
        .bind(chunk.last_seen_ms)
        .execute(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;
        Ok(())
    }

    pub async fn get_draft_chunk(&self, id: &str) -> PersistenceResult<Option<DraftChunkRecord>> {
        let row = sqlx::query(
            r#"
SELECT "id", "scope", "status", "hit_count", "payload", "created_at_ms", "updated_at_ms"
FROM "objects"
WHERE "object_type" = ?1 AND "id" = ?2
"#,
        )
        .bind(DRAFT_CHUNK_OBJECT_TYPE)
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        row.map(row_to_draft_chunk_record).transpose()
    }

    pub async fn find_pending_draft_chunk_for_key(
        &self,
        sandbox_id: &str,
        host: &str,
        port: i32,
        binary: &str,
    ) -> PersistenceResult<Option<DraftChunkRecord>> {
        let dedup_key = draft_chunk_dedup_key_for_status("pending", host, port, binary);
        let row = sqlx::query(
            r#"
SELECT "id", "scope", "status", "hit_count", "payload", "created_at_ms", "updated_at_ms"
FROM "objects"
WHERE "object_type" = ?1 AND "scope" = ?2 AND "status" = 'pending' AND "dedup_key" = ?3
"#,
        )
        .bind(DRAFT_CHUNK_OBJECT_TYPE)
        .bind(sandbox_id)
        .bind(dedup_key)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        row.map(row_to_draft_chunk_record).transpose()
    }

    /// Find any other approved draft chunk for `(sandbox_id, host, port, binary)`
    /// excluding the chunk identified by `exclude_chunk_id`. Approved chunks
    /// have `dedup_key=NULL` (issue #1245), so the partial unique index does
    /// not constrain them — multiple approved chunks can coexist for the same
    /// key. Callers that intend to mutate a rule contributed to by one chunk
    /// use this to detect when another decided chunk is also contributing.
    pub async fn find_other_approved_chunk_for_key(
        &self,
        sandbox_id: &str,
        host: &str,
        port: i32,
        binary: &str,
        exclude_chunk_id: &str,
    ) -> PersistenceResult<Option<DraftChunkRecord>> {
        let rows = sqlx::query(
            r#"
SELECT "id", "scope", "status", "hit_count", "payload", "created_at_ms", "updated_at_ms"
FROM "objects"
WHERE "object_type" = ?1 AND "scope" = ?2 AND "status" = 'approved' AND "id" != ?3
"#,
        )
        .bind(DRAFT_CHUNK_OBJECT_TYPE)
        .bind(sandbox_id)
        .bind(exclude_chunk_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        for row in rows {
            let record = row_to_draft_chunk_record(row)?;
            if record.host == host && record.port == port && record.binary == binary {
                return Ok(Some(record));
            }
        }
        Ok(None)
    }

    pub async fn list_draft_chunks(
        &self,
        sandbox_id: &str,
        status_filter: Option<&str>,
    ) -> PersistenceResult<Vec<DraftChunkRecord>> {
        let rows = if let Some(status) = status_filter {
            sqlx::query(
                r#"
SELECT "id", "scope", "status", "hit_count", "payload", "created_at_ms", "updated_at_ms"
FROM "objects"
WHERE "object_type" = ?1 AND "scope" = ?2 AND "status" = ?3
ORDER BY "created_at_ms" DESC
"#,
            )
            .bind(DRAFT_CHUNK_OBJECT_TYPE)
            .bind(sandbox_id)
            .bind(status)
            .fetch_all(&self.pool)
            .await
        } else {
            sqlx::query(
                r#"
SELECT "id", "scope", "status", "hit_count", "payload", "created_at_ms", "updated_at_ms"
FROM "objects"
WHERE "object_type" = ?1 AND "scope" = ?2
ORDER BY "created_at_ms" DESC
"#,
            )
            .bind(DRAFT_CHUNK_OBJECT_TYPE)
            .bind(sandbox_id)
            .fetch_all(&self.pool)
            .await
        }
        .map_err(|e| map_db_error(&e))?;

        rows.into_iter().map(row_to_draft_chunk_record).collect()
    }

    pub async fn update_draft_chunk_status(
        &self,
        id: &str,
        status: &str,
        decided_at_ms: Option<i64>,
    ) -> PersistenceResult<bool> {
        let Some(mut record) = self.get_draft_chunk(id).await? else {
            return Ok(false);
        };

        record.status = status.to_string();
        record.decided_at_ms = decided_at_ms;
        record.last_seen_ms = current_time_ms()?;
        let payload = draft_chunk_payload_from_record(&record)?;
        let dedup_key = draft_chunk_dedup_key(&record);

        let result = sqlx::query(
            r#"
UPDATE "objects"
SET "status" = ?3, "payload" = ?4, "updated_at_ms" = ?5, "dedup_key" = ?6
WHERE "object_type" = ?1 AND "id" = ?2
"#,
        )
        .bind(DRAFT_CHUNK_OBJECT_TYPE)
        .bind(id)
        .bind(status)
        .bind(payload)
        .bind(record.last_seen_ms)
        .bind(dedup_key)
        .execute(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn update_draft_chunk_rule(
        &self,
        id: &str,
        proposed_rule: &[u8],
    ) -> PersistenceResult<bool> {
        let Some(mut record) = self.get_draft_chunk(id).await? else {
            return Ok(false);
        };

        if record.status != "pending" {
            return Ok(false);
        }

        record.proposed_rule = proposed_rule.to_vec();
        record.last_seen_ms = current_time_ms()?;
        let payload = draft_chunk_payload_from_record(&record)?;

        let result = sqlx::query(
            r#"
UPDATE "objects"
SET "payload" = ?3, "updated_at_ms" = ?4
WHERE "object_type" = ?1 AND "id" = ?2 AND "status" = 'pending'
"#,
        )
        .bind(DRAFT_CHUNK_OBJECT_TYPE)
        .bind(id)
        .bind(payload)
        .bind(record.last_seen_ms)
        .execute(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;
        Ok(result.rows_affected() > 0)
    }

    pub async fn delete_draft_chunks(
        &self,
        sandbox_id: &str,
        status: &str,
    ) -> PersistenceResult<u64> {
        let result = sqlx::query(
            r#"
DELETE FROM "objects"
WHERE "object_type" = ?1 AND "scope" = ?2 AND "status" = ?3
"#,
        )
        .bind(DRAFT_CHUNK_OBJECT_TYPE)
        .bind(sandbox_id)
        .bind(status)
        .execute(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;
        Ok(result.rows_affected())
    }

    pub async fn get_draft_version(&self, sandbox_id: &str) -> PersistenceResult<i64> {
        let rows = sqlx::query(
            r#"
SELECT "payload"
FROM "objects"
WHERE "object_type" = ?1 AND "scope" = ?2
"#,
        )
        .bind(DRAFT_CHUNK_OBJECT_TYPE)
        .bind(sandbox_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| map_db_error(&e))?;

        let mut max_version = 0_i64;
        for row in rows {
            let payload: Vec<u8> = row.get("payload");
            let wrapper = draft_chunk_record_from_parts(
                String::new(),
                sandbox_id.to_string(),
                String::new(),
                0,
                &payload,
                0,
                0,
            )?;
            max_version = max_version.max(wrapper.draft_version);
        }
        Ok(max_version)
    }
}

fn draft_chunk_dedup_key(chunk: &DraftChunkRecord) -> Option<String> {
    draft_chunk_dedup_key_for_status(&chunk.status, &chunk.host, chunk.port, &chunk.binary)
}

fn draft_chunk_dedup_key_for_status(
    status: &str,
    host: &str,
    port: i32,
    binary: &str,
) -> Option<String> {
    // Only pending chunks participate in dedup. Approved and rejected chunks
    // get NULL so they don't absorb future submissions for the same
    // (host, port, binary) — see issue #1245.
    (status == "pending").then(|| format!("{host}|{port}|{binary}"))
}

fn row_to_object_record(row: sqlx::sqlite::SqliteRow) -> ObjectRecord {
    ObjectRecord {
        object_type: row.get("object_type"),
        id: row.get("id"),
        name: row.get("name"),
        payload: row.get("payload"),
        created_at_ms: row.get("created_at_ms"),
        updated_at_ms: row.get("updated_at_ms"),
        labels: row.get("labels"),
    }
}

fn row_to_policy_record(row: sqlx::sqlite::SqliteRow) -> PersistenceResult<PolicyRecord> {
    let id: String = row.get("id");
    let sandbox_id: String = row.get("scope");
    let version: i64 = row.get("version");
    let status: String = row.get("status");
    let payload: Vec<u8> = row.get("payload");
    let created_at_ms: i64 = row.get("created_at_ms");
    policy_record_from_parts(id, sandbox_id, version, status, &payload, created_at_ms)
}

fn row_to_draft_chunk_record(row: sqlx::sqlite::SqliteRow) -> PersistenceResult<DraftChunkRecord> {
    let id: String = row.get("id");
    let sandbox_id: String = row.get("scope");
    let status: String = row.get("status");
    let hit_count: i64 = row.get("hit_count");
    let payload: Vec<u8> = row.get("payload");
    let created_at_ms: i64 = row.get("created_at_ms");
    let updated_at_ms: i64 = row.get("updated_at_ms");
    draft_chunk_record_from_parts(
        id,
        sandbox_id,
        status,
        hit_count,
        &payload,
        created_at_ms,
        updated_at_ms,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn fetch_dedup_key(store: &SqliteStore, id: &str) -> Option<String> {
        sqlx::query_scalar(r#"SELECT "dedup_key" FROM "objects" WHERE "id" = ?1"#)
            .bind(id)
            .fetch_one(&store.pool)
            .await
            .unwrap()
    }

    fn make_chunk(id: &str, sandbox_id: &str, status: &str) -> DraftChunkRecord {
        DraftChunkRecord {
            id: id.to_string(),
            sandbox_id: sandbox_id.to_string(),
            draft_version: 1,
            status: status.to_string(),
            rule_name: "allow_internal_api".to_string(),
            proposed_rule: Vec::new(),
            rationale: "test".to_string(),
            security_notes: String::new(),
            confidence: 0.9,
            created_at_ms: 100,
            decided_at_ms: None,
            host: "internal-api.example.com".to_string(),
            port: 443,
            binary: "/usr/bin/curl".to_string(),
            hit_count: 1,
            first_seen_ms: 100,
            last_seen_ms: 100,
        }
    }

    /// Pending chunks carry a `dedup_key`; approving must clear it so a fresh
    /// pending submission for the same key can land as a new row. Issue #1245.
    #[tokio::test]
    async fn approving_pending_chunk_clears_dedup_key() {
        let store = SqliteStore::connect("sqlite::memory:").await.unwrap();
        store.migrate().await.unwrap();

        let chunk = make_chunk("chunk-1", "sb-1", "pending");
        store.put_draft_chunk(&chunk).await.unwrap();
        assert_eq!(
            fetch_dedup_key(&store, "chunk-1").await,
            Some("internal-api.example.com|443|/usr/bin/curl".to_string()),
            "pending chunk must hold its dedup slot"
        );

        store
            .update_draft_chunk_status("chunk-1", "approved", Some(200))
            .await
            .unwrap();
        assert_eq!(
            fetch_dedup_key(&store, "chunk-1").await,
            None,
            "approved chunk must release its dedup slot"
        );
    }

    /// Rejected chunks must also release their dedup slot. Otherwise a future
    /// denial for the same (host, port, binary) is silently absorbed.
    #[tokio::test]
    async fn rejecting_pending_chunk_clears_dedup_key() {
        let store = SqliteStore::connect("sqlite::memory:").await.unwrap();
        store.migrate().await.unwrap();

        let chunk = make_chunk("chunk-1", "sb-1", "pending");
        store.put_draft_chunk(&chunk).await.unwrap();

        store
            .update_draft_chunk_status("chunk-1", "rejected", Some(200))
            .await
            .unwrap();
        assert_eq!(
            fetch_dedup_key(&store, "chunk-1").await,
            None,
            "rejected chunk must release its dedup slot"
        );
    }

    /// `find_pending_draft_chunk_for_key` returns the pending peer for a
    /// (sandbox, host, port, binary) tuple and ignores decided chunks.
    #[tokio::test]
    async fn find_pending_draft_chunk_for_key_returns_pending_peer() {
        let store = SqliteStore::connect("sqlite::memory:").await.unwrap();
        store.migrate().await.unwrap();

        let approved = make_chunk("chunk-approved", "sb-1", "pending");
        store.put_draft_chunk(&approved).await.unwrap();
        store
            .update_draft_chunk_status("chunk-approved", "approved", Some(150))
            .await
            .unwrap();

        // No peer yet — only the now-approved chunk exists.
        assert!(
            store
                .find_pending_draft_chunk_for_key(
                    "sb-1",
                    "internal-api.example.com",
                    443,
                    "/usr/bin/curl"
                )
                .await
                .unwrap()
                .is_none()
        );

        let pending = make_chunk("chunk-pending", "sb-1", "pending");
        store.put_draft_chunk(&pending).await.unwrap();
        let peer = store
            .find_pending_draft_chunk_for_key(
                "sb-1",
                "internal-api.example.com",
                443,
                "/usr/bin/curl",
            )
            .await
            .unwrap()
            .expect("pending peer must be found");
        assert_eq!(peer.id, "chunk-pending");
    }

    /// `find_other_approved_chunk_for_key` filters approved rows in Rust by
    /// deserializing each payload, so a wrong field comparison would silently
    /// match unrelated rows and block legitimate operator actions. Cover the
    /// four near-miss axes (sandbox, host, port, binary) plus the
    /// exclude-self guard and a positive match.
    #[tokio::test]
    async fn find_other_approved_chunk_for_key_ignores_unrelated_chunks() {
        async fn put_approved(store: &SqliteStore, chunk: DraftChunkRecord) {
            let id = chunk.id.clone();
            store.put_draft_chunk(&chunk).await.unwrap();
            store
                .update_draft_chunk_status(&id, "approved", Some(200))
                .await
                .unwrap();
        }

        let store = SqliteStore::connect("sqlite::memory:").await.unwrap();
        store.migrate().await.unwrap();

        // Self — must be excluded by id.
        put_approved(&store, make_chunk("chunk-self", "sb-1", "pending")).await;
        // Different sandbox.
        put_approved(&store, make_chunk("chunk-other-sandbox", "sb-2", "pending")).await;
        // Different host.
        let mut other_host = make_chunk("chunk-other-host", "sb-1", "pending");
        other_host.host = "different-host.example.com".to_string();
        put_approved(&store, other_host).await;
        // Different port.
        let mut other_port = make_chunk("chunk-other-port", "sb-1", "pending");
        other_port.port = 8443;
        put_approved(&store, other_port).await;
        // Different binary.
        let mut other_binary = make_chunk("chunk-other-binary", "sb-1", "pending");
        other_binary.binary = "/usr/bin/wget".to_string();
        put_approved(&store, other_binary).await;

        // Nothing else approved matches → None.
        assert!(
            store
                .find_other_approved_chunk_for_key(
                    "sb-1",
                    "internal-api.example.com",
                    443,
                    "/usr/bin/curl",
                    "chunk-self",
                )
                .await
                .unwrap()
                .is_none(),
            "must not match different sandbox/host/port/binary or self"
        );

        // Add a true peer and confirm it is returned.
        put_approved(&store, make_chunk("chunk-peer", "sb-1", "pending")).await;
        let peer = store
            .find_other_approved_chunk_for_key(
                "sb-1",
                "internal-api.example.com",
                443,
                "/usr/bin/curl",
                "chunk-self",
            )
            .await
            .unwrap()
            .expect("matching approved peer must be returned");
        assert_eq!(peer.id, "chunk-peer");

        // And excluding the peer instead returns self (the only remaining match).
        let self_match = store
            .find_other_approved_chunk_for_key(
                "sb-1",
                "internal-api.example.com",
                443,
                "/usr/bin/curl",
                "chunk-peer",
            )
            .await
            .unwrap()
            .expect("with the peer excluded, self must match");
        assert_eq!(self_match.id, "chunk-self");
    }

    /// Migration 005 clears `dedup_key` from any decided rows seeded before
    /// the runtime fix landed, but must leave pending rows alone. The test
    /// loads the migration SQL directly from the file (via `include_str!`)
    /// so a drift between the file and the test's expectations would be
    /// caught — and seeds one row of each status, including a pending row
    /// that the migration must NOT touch.
    #[tokio::test]
    async fn migration_clears_dedup_key_on_legacy_decided_rows() {
        const MIGRATION_005: &str =
            include_str!("../../migrations/sqlite/005_clear_dedup_for_decided_chunks.sql");

        let store = SqliteStore::connect("sqlite::memory:").await.unwrap();
        store.migrate().await.unwrap();

        let seed = |id: &'static str, status: &'static str, dedup: &'static str| {
            let pool = store.pool.clone();
            async move {
                sqlx::query(
                    r#"
INSERT INTO "objects" (
    "id", "object_type", "scope", "status", "dedup_key",
    "hit_count", "payload", "created_at_ms", "updated_at_ms"
)
VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
"#,
                )
                .bind(id)
                .bind(DRAFT_CHUNK_OBJECT_TYPE)
                .bind("sb-legacy")
                .bind(status)
                .bind(dedup)
                .bind(1_i64)
                .bind(Vec::<u8>::new())
                .bind(100_i64)
                .bind(100_i64)
                .execute(&pool)
                .await
                .unwrap();
            }
        };

        // Pre-fix gateway: every chunk carries a dedup_key regardless of
        // status. Use distinct keys so all three rows can coexist under the
        // partial unique index.
        seed("legacy-approved", "approved", "host-a|443|/usr/bin/curl").await;
        seed("legacy-rejected", "rejected", "host-r|443|/usr/bin/curl").await;
        seed("inflight-pending", "pending", "host-p|443|/usr/bin/curl").await;

        sqlx::raw_sql(MIGRATION_005)
            .execute(&store.pool)
            .await
            .unwrap();

        assert_eq!(
            fetch_dedup_key(&store, "legacy-approved").await,
            None,
            "migration must clear dedup_key on approved legacy rows"
        );
        assert_eq!(
            fetch_dedup_key(&store, "legacy-rejected").await,
            None,
            "migration must clear dedup_key on rejected legacy rows"
        );
        assert_eq!(
            fetch_dedup_key(&store, "inflight-pending").await,
            Some("host-p|443|/usr/bin/curl".to_string()),
            "migration must NOT clear dedup_key on in-flight pending rows"
        );
    }
}
