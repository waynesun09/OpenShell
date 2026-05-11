-- Issue #1245: only pending draft chunks should hold a dedup_key. Approved
-- and rejected chunks held the same (host, port, binary) key as pending ones,
-- so a fresh denial submission for the same key was silently absorbed by the
-- decided row's ON CONFLICT update. Clear the key on existing decided rows
-- so future submissions can insert as new pending chunks.
UPDATE objects
   SET dedup_key = NULL
 WHERE object_type = 'draft_policy_chunk'
   AND status IN ('approved', 'rejected')
   AND dedup_key IS NOT NULL;
