import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import * as path from "path";
import { normalizePath, ARCHIVE_ENTRY_SEP } from "../paths.js";
import { embed } from "../embedding.js";
import { rerankScores } from "../reranker.js";
import { rrfFuse } from "../fusion.js";
import { f32ToBuffer, getFileMtime, totalChunks } from "../db.js";
import type { Chunk, RagDb, Ranked } from "../types.js";

export function registerSearchTool(server: McpServer, rdb: RagDb): void {
  server.tool(
    "rag_search",
    "Semantic search over locally indexed files. Returns the most relevant code/text chunks for a query.",
    {
      query: z.string().describe("Natural language or code query"),
      topK: z.number().optional().describe("Number of chunks to return. Default: 5"),
      directory: z.string().optional().describe("Restrict search to files under this directory path"),
      extensions: z.array(z.string()).optional().describe("Restrict search to these file extensions (e.g. [\".pdf\", \".md\"])."),
      fileGlob: z.string().optional().describe("Restrict to files matching this glob pattern (e.g. \"**/payslip*.pdf\")."),
      mtimeFrom: z.number().optional().describe("Only include files modified on or after this Unix epoch ms."),
      mtimeTo: z.number().optional().describe("Only include files modified on or before this Unix epoch ms."),
      minScore: z.number().optional().describe("Minimum similarity score threshold (0–1) for vector mode. Default: 0.3"),
      maxPerFile: z.number().optional().describe("Maximum results from any single file. Default: 2"),
      mode: z.enum(["hybrid", "vector", "lexical"]).optional().describe("Search mode: hybrid (RRF), vector only, or lexical (BM25) only. Default: hybrid"),
      rerank: z.boolean().optional().describe("Re-rank top candidates with a cross-encoder. Default: false"),
      rerankTopN: z.number().optional().describe("Candidates to feed into the reranker. Default: 30"),
    },
    async ({ query, topK = 5, directory: rawDirectory, extensions, fileGlob, mtimeFrom, mtimeTo, minScore = 0.3, maxPerFile = 2, mode = "hybrid", rerank = false, rerankTopN = 30 }) => {
      const chunkCount = totalChunks(rdb);
      if (chunkCount === 0) {
        return { content: [{ type: "text", text: "Index is empty. Run rag_index first." }] };
      }

      const dirPrefix = rawDirectory ? normalizePath(rawDirectory) + "/" : null;
      const extSet = extensions && extensions.length > 0 ? new Set(extensions.map(e => e.toLowerCase())) : null;
      let globRx: RegExp | null = null;
      if (fileGlob) {
        const escape = (s: string) => s.replace(/[.+^${}()|[\]\\]/g, "\\$&");
        let re = "";
        for (let i = 0; i < fileGlob.length; i++) {
          const c = fileGlob[i];
          if (c === "*") {
            if (fileGlob[i + 1] === "*") { re += ".*"; i++; }
            else re += "[^/]*";
          } else if (c === "?") re += "[^/]";
          else re += escape(c);
        }
        globRx = new RegExp("^" + re + "$", "i");
      }

      const passes = (file: string): boolean => {
        if (extSet && !extSet.has(path.extname(file).toLowerCase())) return false;
        if (globRx && !globRx.test(file)) return false;
        if (mtimeFrom != null || mtimeTo != null) {
          const m = getFileMtime(rdb, file);
          if (m == null) return false;
          if (mtimeFrom != null && m < mtimeFrom) return false;
          if (mtimeTo != null && m > mtimeTo) return false;
        }
        return true;
      };

      // Over-fetch from each ranker so post-filtering + per-file cap have enough to work with.
      const POOL = Math.max(topK * 10, 50);
      const dirClause = dirPrefix ? "c.file LIKE ?" : "1";
      const dirParam: unknown[] = dirPrefix ? [dirPrefix + "%"] : [];

      // ── Vector ranking via sqlite-vec ────────────────────────────────────────
      const vectorList: Ranked[] = [];
      if ((mode === "vector" || mode === "hybrid") && rdb.embedDim > 0) {
        const qEmb = await embed(query);
        const k = POOL * 4;
        const rows = rdb.db.prepare(`
          SELECT v.rowid AS id, v.distance AS dist, c.file AS file
          FROM vec_chunks v
          JOIN chunks c ON c.id = v.rowid
          WHERE v.embedding MATCH ? AND v.k = ?
            AND ${dirClause}
          ORDER BY v.distance
        `).all(f32ToBuffer(qEmb), k, ...dirParam) as { id: number; dist: number; file: string }[];

        for (const r of rows) {
          if (!passes(r.file)) continue;
          // Convert L2 distance on normalised embeddings to cosine similarity in [0,1].
          // For unit vectors: cosine = 1 - dist^2 / 2.
          const score = 1 - (r.dist * r.dist) / 2;
          if (mode === "vector" && score < minScore) continue;
          vectorList.push({ chunkId: r.id, score });
          if (vectorList.length >= POOL) break;
        }
      }

      // ── Lexical ranking via FTS5 BM25 ────────────────────────────────────────
      const lexicalList: Ranked[] = [];
      if (mode === "lexical" || mode === "hybrid") {
        const ftsQuery = (query.match(/[A-Za-z0-9_]+/g) ?? []).map(t => `"${t}"`).join(" OR ");
        if (ftsQuery) {
          try {
            const rows = rdb.db.prepare(`
              SELECT c.id AS id, bm25(fts_chunks) AS bm, c.file AS file
              FROM fts_chunks
              JOIN chunks c ON c.id = fts_chunks.rowid
              WHERE fts_chunks MATCH ? AND ${dirClause}
              ORDER BY bm
              LIMIT ?
            `).all(ftsQuery, ...dirParam, POOL * 4) as { id: number; bm: number; file: string }[];

            for (const r of rows) {
              if (!passes(r.file)) continue;
              lexicalList.push({ chunkId: r.id, score: -r.bm });
              if (lexicalList.length >= POOL) break;
            }
          } catch (err) {
            process.stderr.write(`[rag] FTS5 query failed: ${String(err)}\n`);
          }
        }
      }

      // ── Fuse ─────────────────────────────────────────────────────────────────
      let fused: Ranked[];
      if (mode === "hybrid") {
        const rrf = rrfFuse([vectorList, lexicalList]);
        fused = [...rrf.entries()].map(([chunkId, score]) => ({ chunkId, score })).sort((a, b) => b.score - a.score);
      } else if (mode === "vector") {
        fused = vectorList;
      } else {
        fused = lexicalList;
      }

      if (fused.length === 0) {
        return { content: [{ type: "text", text: `No results for "${query}" in ${mode} mode. Try a different query or mode.` }] };
      }

      // ── Optional cross-encoder rerank of the top-N fused candidates ──────────
      let rerankApplied = false;
      if (rerank) {
        const head = fused.slice(0, Math.max(topK, rerankTopN));
        const ids = head.map(r => r.chunkId);
        const placeholders = ids.map(() => "?").join(",");
        const rows = rdb.db.prepare(`SELECT id, text FROM chunks WHERE id IN (${placeholders})`).all(...ids) as { id: number; text: string }[];
        const textById = new Map(rows.map(r => [r.id, r.text]));
        try {
          const passages = head.map(r => textById.get(r.chunkId) ?? "");
          const scores = await rerankScores(query, passages);
          const reranked = head
            .map((r, i) => ({ chunkId: r.chunkId, score: scores[i] }))
            .sort((a, b) => b.score - a.score);
          fused = [...reranked, ...fused.slice(head.length)];
          rerankApplied = true;
        } catch (err) {
          process.stderr.write(`[rag] reranker failed, falling back to fused scores: ${String(err)}\n`);
        }
      }

      // ── Hydrate top candidates with their text + apply per-file cap ──────────
      const hydrateIds = fused.slice(0, topK * Math.max(maxPerFile, 1) * 4).map(r => r.chunkId);
      const idToChunk = new Map<number, Chunk>();
      if (hydrateIds.length > 0) {
        const ph = hydrateIds.map(() => "?").join(",");
        const rows = rdb.db.prepare(`SELECT id, file, start, text, hash FROM chunks WHERE id IN (${ph})`).all(...hydrateIds) as Array<{ id: number; file: string; start: number; text: string; hash: string }>;
        for (const r of rows) idToChunk.set(r.id, { file: r.file, start: r.start, text: r.text, hash: r.hash });
      }

      const fileCounts = new Map<string, number>();
      const results: Array<{ chunk: Chunk; score: number }> = [];
      for (const r of fused) {
        if (results.length >= topK) break;
        const c = idToChunk.get(r.chunkId);
        if (!c) continue;
        const count = fileCounts.get(c.file) ?? 0;
        if (count < maxPerFile) {
          results.push({ chunk: c, score: r.score });
          fileCounts.set(c.file, count + 1);
        }
      }

      if (results.length === 0) {
        return { content: [{ type: "text", text: `No results for "${query}" in ${mode} mode. Try a different query or mode.` }] };
      }

      const fenceLang = (f: string) => {
        const base = f.split(ARCHIVE_ENTRY_SEP).pop() ?? f;
        const e = path.extname(base).replace(".", "");
        return ["pdf", "docx", "pptx", "xlsx", "rtf", "odt", "epub", "zip", "7z"].includes(e) ? "text" : (e || "text");
      };
      const scoreLabel = rerankApplied ? `${mode}+rerank` : mode;
      const text = results
        .map(r => `**${r.chunk.file}** (${scoreLabel}: ${r.score.toFixed(3)})\n\`\`\`${fenceLang(r.chunk.file)}\n${r.chunk.text}\n\`\`\``)
        .join("\n\n---\n\n");

      return { content: [{ type: "text", text }] };
    }
  );
}
