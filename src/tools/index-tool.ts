import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { glob } from "glob";
import { statSync } from "fs";
import * as path from "path";
import {
  normalizePath, hashText, IMAGE_EXTS, ARCHIVE_CONTAINER_EXTS,
  ARCHIVE_ENTRY_SEP, DEFAULT_INDEX_EXTS, LLM_CAPTION_SENTINEL,
} from "../paths.js";
import { LLM_MAX_CONCURRENCY } from "../config.js";
import { chunkText } from "../chunking.js";
import { embedBatch } from "../embedding.js";
import { buildIgnoreFilters, isGitignored } from "../ignore-filter.js";
import { extractText, enumerateArchiveEntries } from "../extractors.js";
import { getLlmAvailable } from "../llm-caption.js";
import {
  GIT_LOG_ENTRY, GIT_BRANCHES_ENTRY,
  isGitAvailable, findGitRepos, getGitTipHash, exportGitLog, exportGitBranches,
} from "../git-export.js";
import {
  ensureVecTable, getFileMtime, setFileMtime, listFilesUnderPrefix,
  hashExists, hasLlmCaption, removeFileChunks, insertChunkRow, totalChunks,
} from "../db.js";
import type { RagDb } from "../types.js";

export function registerIndexTool(server: McpServer, rdb: RagDb): void {
  server.tool(
    "rag_index",
    [
      "Index files from a local directory for semantic search.",
      "By default only re-indexes files that changed since the last run (incremental — fast on repeat runs).",
      "Use force:true to re-index all files unconditionally.",
      `Supported by default: ${DEFAULT_INDEX_EXTS.join(" ")}`,
    ].join(" "),
    {
      directory: z.string().describe("Absolute path to the directory to index"),
      extensions: z
        .array(z.string())
        .optional()
        .describe("File extensions to include, e.g. [\".cs\", \".json\", \".pdf\", \".docx\"]. Defaults to common source and document types."),
      passwords: z
        .array(z.string())
        .optional()
        .describe("Passwords to try when opening encrypted PDFs. Each is attempted in order."),
      ignorePatterns: z
        .array(z.string())
        .optional()
        .describe("Additional .gitignore-style patterns to skip files (e.g. [\"~$*\", \"*.tmp\", \"**/backup/**\"])."),
      ocrLang: z
        .string()
        .optional()
        .describe("Tesseract OCR language(s), e.g. \"eng\", \"dan\", \"eng+dan\". Default: \"eng+dan\". Required language data must be installed."),
      recursive: z.boolean().optional().describe("Recurse into subdirectories. Default: true"),
      force: z
        .boolean()
        .optional()
        .describe("Re-index all files even if unchanged. Default: false (incremental)"),
      captionImages: z
        .boolean()
        .optional()
        .describe("Use the local vision LLM to caption image files and enhance scanned PDFs. When the LLM is reachable, images and scanned PDFs that lack a caption are re-processed automatically on subsequent runs. Default: true"),
      indexGitHistory: z
        .boolean()
        .optional()
        .describe("Also index the full git log plus a branch/tag summary for every git repository found under the directory. Stored as virtual files \"<repoRoot>!git-log\" and \"<repoRoot>!git-branches\". Default: true"),
    },
    async ({ directory: rawDirectory, extensions, passwords, ignorePatterns, ocrLang = "eng+dan", recursive = true, force = false, captionImages = true, indexGitHistory = true }) => {
      const directory = normalizePath(rawDirectory);
      const dirPrefix = directory.endsWith("/") ? directory : directory + "/";
      const exts = new Set(extensions ?? DEFAULT_INDEX_EXTS);
      const pattern = recursive ? "**/*" : "*";

      const files = await glob(pattern, {
        cwd: directory,
        absolute: true,
        nodir: true,
        ignore: ["**/.git/**", "**/node_modules/**"],
      });

      const ignoreFilters = await buildIgnoreFilters(directory, ignorePatterns);
      const matching = files.filter(f => exts.has(path.extname(f).toLowerCase()) && !isGitignored(f, ignoreFilters));

      // Remove chunks for files that no longer exist in this directory.
      // Virtual archive entries (containing "!") survive while their parent archive does;
      // missing entries within an archive are pruned when that archive is re-processed.
      const normalizedMatching = matching.map(normalizePath);
      const matchingSet = new Set(normalizedMatching);
      const tracked = listFilesUnderPrefix(rdb, dirPrefix);
      const staleFiles = tracked.map(t => t.path).filter(p => {
        if (matchingSet.has(p)) return false;
        const bang = p.indexOf(ARCHIVE_ENTRY_SEP);
        if (bang >= 0) {
          const inner = p.slice(bang + 1);
          // Git virtual entries are cleaned up separately below, after we know which repos exist.
          if (inner === GIT_LOG_ENTRY || inner === GIT_BRANCHES_ENTRY) return false;
          return !matchingSet.has(p.slice(0, bang));
        }
        return true;
      });
      for (const f of staleFiles) {
        removeFileChunks(rdb, f);
        rdb.db.prepare("DELETE FROM files WHERE path = ?").run(f);
      }

      let indexed = 0;
      let reused = 0;
      let skipped = 0;
      const errors: { file: string; error: string }[] = [];

      const log = (msg: string) => process.stderr.write(`[rag] ${msg}\n`);
      const total = normalizedMatching.length;
      log(`Scanning ${directory} — ${total} matching file(s), ${staleFiles.length} stale removed`);
      const t0 = Date.now();

      const BATCH = 32;

      /** Ingest one logical unit (real file or archive entry). */
      const ingestUnit = async (virtualPath: string, mtime: number, text: string, displayName: string, i: number, total: number, fileT0: number): Promise<number> => {
        if (!text.trim()) {
          log(`[${i}/${total}] skipped (empty) ${displayName}`);
          return 0;
        }
        // Drop any previously-indexed chunks for this unit before re-ingesting.
        removeFileChunks(rdb, virtualPath);

        const fileChunks = chunkText(text);
        // Hash-dedup against the DB before embedding (saves work on identical text).
        const fresh: { text: string; start: number; hash: string }[] = [];
        let dupCount = 0;
        const seenInBatch = new Set<string>();
        for (const c of fileChunks) {
          const h = hashText(c.text);
          if (seenInBatch.has(h) || hashExists(rdb, h)) { dupCount++; continue; }
          fresh.push({ ...c, hash: h });
          seenInBatch.add(h);
        }
        for (let j = 0; j < fresh.length; j += BATCH) {
          const batch = fresh.slice(j, j + BATCH);
          const embeddings = await embedBatch(batch.map(c => c.text));
          ensureVecTable(rdb, embeddings[0].length);
          const tx = rdb.db.transaction(() => {
            for (let k = 0; k < batch.length; k++) {
              insertChunkRow(rdb, {
                file: virtualPath,
                start: batch[k].start,
                text: batch[k].text,
                hash: batch[k].hash,
                embedding: embeddings[k],
              });
            }
          });
          tx();
        }
        setFileMtime(rdb, virtualPath, mtime);
        log(`[${i}/${total}] done ${displayName} (${fresh.length} new chunks, ${dupCount} dup-skipped, ${Math.round((Date.now() - fileT0) / 1000)}s)`);
        return fresh.length;
      };

      // Process a single file (or archive container). Designed to be safe to run
      // in parallel with itself: better-sqlite3 calls are synchronous on the JS
      // event loop, and all shared counters/arrays are mutated synchronously too.
      const processFile = async (file: string, i: number): Promise<void> => {
        const rel = path.relative(directory, file).replace(/\\/g, "/");
        const fileExt = path.extname(file).toLowerCase();

        try {
          const mtime = statSync(file).mtimeMs;

          // Archive container — fan out into per-entry units
          if (ARCHIVE_CONTAINER_EXTS.has(fileExt)) {
            const prefixKey = file + ARCHIVE_ENTRY_SEP;
            const existingEntries = listFilesUnderPrefix(rdb, prefixKey);

            // Check whether any entry inside the archive could newly benefit from a caption.
            let archiveNeedsCaption = false;
            if (captionImages && existingEntries.length > 0 && await getLlmAvailable()) {
              for (const ex of existingEntries) {
                const exExt = path.extname(ex.path.split(ARCHIVE_ENTRY_SEP).pop() ?? "").toLowerCase();
                if ((IMAGE_EXTS.has(exExt) || exExt === ".pdf") && !hasLlmCaption(rdb, ex.path, LLM_CAPTION_SENTINEL)) {
                  archiveNeedsCaption = true;
                  break;
                }
              }
            }

            // If archive unchanged and entries are tracked and no caption upgrade needed, skip.
            if (!force && !archiveNeedsCaption && existingEntries.length > 0 && existingEntries.every(e => e.mtime === mtime)) {
              reused++;
              return;
            }

            const fileT0 = Date.now();
            log(`[${i}/${total}] indexing archive ${rel}${archiveNeedsCaption ? " (caption upgrade)" : ""}`);
            const entries = await enumerateArchiveEntries(file, passwords, ocrLang, captionImages);
            const currentKeys = new Set(entries.map(e => file + ARCHIVE_ENTRY_SEP + e.entryPath));

            // Prune entries that no longer exist inside the archive
            for (const ex of existingEntries) {
              if (!currentKeys.has(ex.path)) {
                removeFileChunks(rdb, ex.path);
                rdb.db.prepare("DELETE FROM files WHERE path = ?").run(ex.path);
              }
            }

            for (const e of entries) {
              const vPath = file + ARCHIVE_ENTRY_SEP + e.entryPath;
              const vDisplay = `${rel}${ARCHIVE_ENTRY_SEP}${e.entryPath}`;
              await ingestUnit(vPath, mtime, e.text, vDisplay, i, total, fileT0);
            }
            indexed++;
            return;
          }

          // Regular file. Files whose mtime matches are skipped, UNLESS they are an
          // image or PDF that could now gain a vision-LLM caption (LLM became available
          // since the last run). This makes captioning self-healing across runs.
          const mtimeMatch = getFileMtime(rdb, file) === mtime;
          const captionEligible = IMAGE_EXTS.has(fileExt) || fileExt === ".pdf";
          const needsCaption = captionImages && captionEligible && mtimeMatch
            && await getLlmAvailable()
            && !hasLlmCaption(rdb, file, LLM_CAPTION_SENTINEL);

          if (!force && mtimeMatch && !needsCaption) {
            reused++;
            return;
          }

          const fileT0 = Date.now();
          log(`[${i}/${total}] indexing ${rel}${needsCaption ? " (caption upgrade)" : ""}`);
          const text = await extractText(file, passwords, ocrLang, captionImages);
          await ingestUnit(file, mtime, text, rel, i, total, fileT0);
          indexed++;
        } catch (err) {
          errors.push({ file, error: String(err) });
          skipped++;
          log(`[${i}/${total}] FAILED ${rel}: ${String(err)}`);
        }
      };

      // File-level concurrency. Matched to the vision-LLM semaphore so image-heavy
      // directories can keep both LLM slots busy. For text-only runs the small
      // parallelism still helps overlap I/O with embedding work.
      const FILE_CONCURRENCY = LLM_MAX_CONCURRENCY;
      let nextIdx = 0;
      const worker = async () => {
        while (true) {
          const myIdx = nextIdx++;
          if (myIdx >= normalizedMatching.length) return;
          await processFile(normalizedMatching[myIdx], myIdx + 1);
        }
      };
      await Promise.all(Array.from({ length: FILE_CONCURRENCY }, worker));

      // ── Git repository history indexing ─────────────────────────────────────
      let gitRepos = 0;
      let gitIndexed = 0;
      let gitReused = 0;
      if (indexGitHistory) {
        if (!(await isGitAvailable())) {
          log("git not available on PATH — skipping git history indexing");
        } else {
          const repos = await findGitRepos(directory);
          gitRepos = repos.length;
          log(`Found ${gitRepos} git repository/repositories under ${directory}`);
          const liveGitPaths = new Set<string>();
          let gi = 0;
          for (const repo of repos) {
            gi++;
            const logPath = repo + ARCHIVE_ENTRY_SEP + GIT_LOG_ENTRY;
            const branchesPath = repo + ARCHIVE_ENTRY_SEP + GIT_BRANCHES_ENTRY;
            liveGitPaths.add(logPath);
            liveGitPaths.add(branchesPath);
            const relRepo = path.relative(directory, repo).replace(/\\/g, "/") || ".";
            try {
              const tipHash = await getGitTipHash(repo);
              const upToDate =
                tipHash !== 0 &&
                getFileMtime(rdb, logPath) === tipHash &&
                getFileMtime(rdb, branchesPath) === tipHash;
              if (!force && upToDate) {
                gitReused++;
                log(`[git ${gi}/${gitRepos}] up-to-date ${relRepo}`);
                continue;
              }
              const fileT0 = Date.now();
              log(`[git ${gi}/${gitRepos}] exporting log ${relRepo}`);
              const logText = await exportGitLog(repo);
              await ingestUnit(logPath, tipHash, logText, `${relRepo}!${GIT_LOG_ENTRY}`, gi, gitRepos, fileT0);
              const branchesText = await exportGitBranches(repo);
              await ingestUnit(branchesPath, tipHash, branchesText, `${relRepo}!${GIT_BRANCHES_ENTRY}`, gi, gitRepos, fileT0);
              gitIndexed++;
            } catch (err) {
              errors.push({ file: logPath, error: String(err) });
              skipped++;
              log(`[git ${gi}/${gitRepos}] FAILED ${relRepo}: ${String(err)}`);
            }
          }
          // Prune git virtual entries whose repo no longer exists under this directory.
          const trackedGit = listFilesUnderPrefix(rdb, dirPrefix).filter(t => {
            const bang = t.path.indexOf(ARCHIVE_ENTRY_SEP);
            if (bang < 0) return false;
            const inner = t.path.slice(bang + 1);
            return inner === GIT_LOG_ENTRY || inner === GIT_BRANCHES_ENTRY;
          });
          for (const t of trackedGit) {
            if (liveGitPaths.has(t.path)) continue;
            removeFileChunks(rdb, t.path);
            rdb.db.prepare("DELETE FROM files WHERE path = ?").run(t.path);
          }
        }
      }

      log(`Finished: indexed=${indexed} reused=${reused} skipped=${skipped} git(repos=${gitRepos} indexed=${gitIndexed} reused=${gitReused}) in ${Math.round((Date.now() - t0) / 1000)}s`);

      const chunkCount = totalChunks(rdb);
      const lines = [
        `Indexed ${indexed} files, reused ${reused} unchanged, ${skipped} failed/empty.`,
        `Git repos: ${gitRepos} found, ${gitIndexed} (re)indexed, ${gitReused} unchanged.`,
        `Directory: ${directory}`,
        `Total chunks in index: ${chunkCount}`,
      ];
      if (errors.length > 0) {
        lines.push("", `Errors (first ${Math.min(errors.length, 10)} of ${errors.length}):`,
          ...errors.slice(0, 10).map(e => `  ${path.basename(e.file)}: ${e.error}`)
        );
      }

      return { content: [{ type: "text", text: lines.join("\n") }] };
    }
  );
}
