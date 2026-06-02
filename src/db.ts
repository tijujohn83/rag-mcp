import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { dbPath, EMBED_MODEL } from "./config.js";
import type { RagDb } from "./types.js";

// ── Index storage: SQLite + sqlite-vec + FTS5 ─────────────────────────────────
//
// Schema:
//   meta(key, value)                  — embed_model, embed_dim, schema_version
//   files(path PK, mtime)             — tracked files / archive entries
//   chunks(id PK, file, start, text, hash)
//   vec_chunks(rowid -> embedding)    — sqlite-vec virtual table
//   fts_chunks(text)                  — FTS5 external content for BM25
//
// Triggers on chunks keep fts_chunks in sync. vec_chunks is updated manually
// in insertChunkRow/removeFileChunks because sqlite-vec doesn't support
// triggers on virtual tables.

export function f32ToBuffer(f32: Float32Array): Buffer {
  return Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength);
}

export function openDb(): RagDb {
  const db = new Database(dbPath);
  sqliteVec.load(db);
  db.pragma("journal_mode = WAL");
  db.pragma("synchronous = NORMAL");

  db.exec(`
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
    CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, mtime REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      file TEXT NOT NULL,
      start INTEGER NOT NULL,
      text TEXT NOT NULL,
      hash TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file);
    CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
      text, content='chunks', content_rowid='id', tokenize='porter unicode61'
    );
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO fts_chunks(rowid, text) VALUES (new.id, new.text);
    END;
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
      INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES ('delete', old.id, old.text);
    END;
  `);

  // Determine stored embed model + dim, or stamp the meta on a fresh DB.
  const storedModel = (db.prepare("SELECT value FROM meta WHERE key = ?").get("embed_model") as { value?: string } | undefined)?.value;
  const storedDim = Number(
    (db.prepare("SELECT value FROM meta WHERE key = ?").get("embed_dim") as { value?: string } | undefined)?.value ?? 0
  );

  if (storedModel && storedModel !== EMBED_MODEL) {
    process.stderr.write(
      `[rag] WARNING: index was built with embed model '${storedModel}' but current model is '${EMBED_MODEL}'. ` +
      `Vector search results will be meaningless until you re-index with force:true.\n`
    );
  }

  // vec_chunks must be created with a fixed dim. We create it lazily once we
  // know the dim (either from meta or from the first embedding produced).
  if (storedDim > 0) {
    db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[${storedDim}]);`);
  }

  return { db, embedDim: storedDim, embedModel: storedModel ?? EMBED_MODEL };
}

/** Ensure vec_chunks table exists for the given dim, stamping meta on first call. */
export function ensureVecTable(rdb: RagDb, dim: number): void {
  if (rdb.embedDim === dim) return;
  if (rdb.embedDim > 0 && rdb.embedDim !== dim) {
    throw new Error(
      `Embedding dim mismatch: index has dim=${rdb.embedDim}, new embeddings are dim=${dim}. ` +
      `Drop the DB or use a different --index path.`
    );
  }
  rdb.db.exec(`CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[${dim}]);`);
  const upsert = rdb.db.prepare("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)");
  upsert.run("embed_model", EMBED_MODEL);
  upsert.run("embed_dim", String(dim));
  upsert.run("schema_version", "1");
  rdb.embedDim = dim;
  rdb.embedModel = EMBED_MODEL;
}

// ── DB primitives used by the tools ───────────────────────────────────────────

export function getFileMtime(rdb: RagDb, file: string): number | undefined {
  const row = rdb.db.prepare("SELECT mtime FROM files WHERE path = ?").get(file) as { mtime: number } | undefined;
  return row?.mtime;
}

export function setFileMtime(rdb: RagDb, file: string, mtime: number): void {
  rdb.db.prepare("INSERT OR REPLACE INTO files(path, mtime) VALUES (?, ?)").run(file, mtime);
}

export function listFilesUnderPrefix(rdb: RagDb, prefix: string): { path: string; mtime: number }[] {
  return rdb.db.prepare("SELECT path, mtime FROM files WHERE path LIKE ?").all(prefix + "%") as { path: string; mtime: number }[];
}

export function hashExists(rdb: RagDb, hash: string): boolean {
  return (rdb.db.prepare("SELECT 1 FROM chunks WHERE hash = ? LIMIT 1").get(hash) as unknown) !== undefined;
}

/** True if this file already has at least one chunk produced by the vision LLM
 *  (detected via the [LLM_CAPTION] sentinel prefix). */
export function hasLlmCaption(rdb: RagDb, file: string, sentinel: string): boolean {
  return (rdb.db
    .prepare("SELECT 1 FROM chunks WHERE file = ? AND text LIKE ? LIMIT 1")
    .get(file, sentinel + "%") as unknown) !== undefined;
}

export function removeFileChunks(rdb: RagDb, file: string): number {
  const ids = (rdb.db.prepare("SELECT id FROM chunks WHERE file = ?").all(file) as { id: number }[]).map(r => r.id);
  if (ids.length === 0) return 0;
  const delVec = rdb.db.prepare("DELETE FROM vec_chunks WHERE rowid = ?");
  const delChunk = rdb.db.prepare("DELETE FROM chunks WHERE id = ?");
  const tx = rdb.db.transaction(() => {
    for (const id of ids) { try { delVec.run(BigInt(id)); } catch { /* vec table may not exist yet */ } delChunk.run(id); }
  });
  tx();
  return ids.length;
}

export function removePrefix(rdb: RagDb, prefix: string): number {
  const ids = (rdb.db.prepare("SELECT id FROM chunks WHERE file LIKE ?").all(prefix + "%") as { id: number }[]).map(r => r.id);
  const delVec = rdb.db.prepare("DELETE FROM vec_chunks WHERE rowid = ?");
  const tx = rdb.db.transaction(() => {
    for (const id of ids) { try { delVec.run(BigInt(id)); } catch { /* */ } }
    rdb.db.prepare("DELETE FROM chunks WHERE file LIKE ?").run(prefix + "%");
    rdb.db.prepare("DELETE FROM files WHERE path LIKE ?").run(prefix + "%");
  });
  tx();
  return ids.length;
}

export interface InsertChunkArgs { file: string; start: number; text: string; hash: string; embedding: Float32Array; }

/** Insert one chunk + its embedding. Caller must have called ensureVecTable. */
export function insertChunkRow(rdb: RagDb, c: InsertChunkArgs): number {
  const info = rdb.db.prepare("INSERT INTO chunks(file, start, text, hash) VALUES (?, ?, ?, ?)")
    .run(c.file, c.start, c.text, c.hash);
  const id = Number(info.lastInsertRowid);
  // sqlite-vec requires a BigInt rowid binding (it rejects JS numbers as "non-integer").
  rdb.db.prepare("INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)").run(BigInt(id), f32ToBuffer(c.embedding));
  return id;
}

export function totalChunks(rdb: RagDb): number {
  return (rdb.db.prepare("SELECT count(*) AS n FROM chunks").get() as { n: number }).n;
}
export function totalFiles(rdb: RagDb): number {
  return (rdb.db.prepare("SELECT count(DISTINCT file) AS n FROM chunks").get() as { n: number }).n;
}
export function totalTrackedFiles(rdb: RagDb): number {
  return (rdb.db.prepare("SELECT count(*) AS n FROM files").get() as { n: number }).n;
}
