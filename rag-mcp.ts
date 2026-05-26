import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { pipeline, env } from "@huggingface/transformers";
import { glob } from "glob";
import { readFileSync, writeFileSync, existsSync, statSync } from "fs";
import { readFile } from "fs/promises";
import * as path from "path";
import * as url from "url";
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse") as (buf: Buffer) => Promise<{ text: string }>;
import mammoth from "mammoth";
import ignore, { Ignore } from "ignore";

// ── Model cache ───────────────────────────────────────────────────────────────

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
env.cacheDir = path.join(__dirname, ".model-cache");

// ── Types ─────────────────────────────────────────────────────────────────────

interface Chunk {
  file: string;
  start: number;
  text: string;
  embedding: Float32Array;
}

interface VectorIndex {
  files: Record<string, { mtime: number }>;
  chunks: Chunk[];
}

// On-disk representation — embeddings stored as base64 to halve JSON size
type StoredChunk = Omit<Chunk, "embedding"> & { embedding: string | number[] };
type StoredIndex = { files?: Record<string, { mtime: number }>; chunks: StoredChunk[] };

// ── Embedding ─────────────────────────────────────────────────────────────────

let _embedder: Awaited<ReturnType<typeof pipeline>> | null = null;

async function getEmbedder() {
  if (!_embedder) {
    // all-MiniLM-L6-v2: ~80 MB, downloads once to .model-cache, then runs locally
    _embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");
  }
  return _embedder;
}

async function embed(text: string): Promise<Float32Array> {
  const e = await getEmbedder();
  const result = await e(text, { pooling: "mean", normalize: true });
  return new Float32Array(result.data as Float32Array);
}

async function embedBatch(texts: string[]): Promise<Float32Array[]> {
  if (texts.length === 0) return [];
  if (texts.length === 1) return [await embed(texts[0])];
  try {
    const e = await getEmbedder();
    const result = await e(texts, { pooling: "mean", normalize: true });
    const dims = (result.dims as number[])[result.dims.length - 1];
    const floats = result.data as Float32Array;
    return texts.map((_, i) => floats.slice(i * dims, (i + 1) * dims));
  } catch {
    // Fallback if the pipeline doesn't support batching
    return Promise.all(texts.map(t => embed(t)));
  }
}

// ── Dot product (embeddings are L2-normalised so dot product = cosine) ────────

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

// ── Text chunking (paragraph-aware) ──────────────────────────────────────────

function chunkText(text: string, maxSize = 800, overlap = 100): { text: string; start: number }[] {
  const chunks: { text: string; start: number }[] = [];
  let windowStart = 0;

  while (windowStart < text.length) {
    const windowEnd = windowStart + maxSize;

    if (windowEnd >= text.length) {
      const chunk = text.slice(windowStart);
      if (chunk.trim()) chunks.push({ text: chunk, start: windowStart });
      break;
    }

    // Prefer paragraph boundary (\n\n), then line boundary (\n), then hard split
    let breakAt = text.lastIndexOf("\n\n", windowEnd);
    if (breakAt <= windowStart) breakAt = text.lastIndexOf("\n", windowEnd);
    if (breakAt <= windowStart) breakAt = windowEnd;

    const chunk = text.slice(windowStart, breakAt);
    if (chunk.trim()) chunks.push({ text: chunk, start: windowStart });

    // Advance with overlap; Math.max guards against infinite loop on no-boundary edge cases
    windowStart = Math.max(windowStart + 1, breakAt - overlap);
  }

  return chunks;
}

// ── Gitignore filtering ──────────────────────────────────────────────────────

interface IgnoreFilter { base: string; ig: Ignore; }

async function buildIgnoreFilters(rootDir: string): Promise<IgnoreFilter[]> {
  const filters: IgnoreFilter[] = [];

  // .gitignore files inside the indexed directory (nested repos / subfolders)
  // Ancestor .gitignore files are intentionally excluded: a workspace-level
  // .gitignore may exclude entire sub-folders (e.g. /be/ /fe/) that the user
  // explicitly wants to index, so we only respect rules within rootDir itself.
  const nested = await glob("**/.gitignore", { cwd: rootDir, absolute: true, dot: true });

  for (const gip of nested) {
    try {
      const ig = ignore();
      ig.add(readFileSync(gip, "utf-8"));
      filters.push({ base: path.dirname(gip), ig });
    } catch { /* unreadable — skip */ }
  }

  return filters;
}

function isGitignored(file: string, filters: IgnoreFilter[]): boolean {
  return filters.some(({ base, ig }) => {
    // Use path.sep suffix to avoid false prefix matches (e.g. "foo" matching "foo-bar")
    if (!file.startsWith(base + path.sep) && file !== base) return false;
    const rel = path.relative(base, file).replace(/\\/g, "/");
    if (rel.startsWith("../")) return false;
    return ig.ignores(rel);
  });
}

// ── File text extraction ─────────────────────────────────────────────────────

async function extractText(file: string): Promise<string> {
  const ext = path.extname(file).toLowerCase();
  if (ext === ".pdf") {
    const buf = await readFile(file);
    const data = await pdfParse(buf);
    return data.text;
  }
  if (ext === ".docx") {
    const buf = await readFile(file);
    const result = await mammoth.extractRawText({ buffer: buf });
    return result.value;
  }
  // plain text (includes .cs, .json, .ts, .md, .txt, .yml, etc.)
  return readFileSync(file, "utf-8");
}

// ── Path normalisation (forward slashes, consistent casing via resolve) ────────

// glob returns forward-slash paths on Windows; path.resolve returns backslashes.
// Normalise everything to forward slashes so startsWith comparisons are reliable.
const normalizePath = (p: string) => path.resolve(p).replace(/\\/g, "/");

// ── Base64 Float32 serialization ──────────────────────────────────────────────

function encodeEmbedding(f32: Float32Array): string {
  return Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength).toString("base64");
}

function decodeEmbedding(stored: string | number[]): Float32Array {
  if (Array.isArray(stored)) return new Float32Array(stored); // migrate old number[] format
  const buf = Buffer.from(stored, "base64");
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

// ── Index storage ──────────────────────────────────────────────────────────────

const argIndex = process.argv.indexOf("--index");
const indexPath = argIndex !== -1 && process.argv[argIndex + 1]
  ? process.argv[argIndex + 1]
  : path.join(__dirname, "rag-index.json");

function loadIndex(): VectorIndex {
  if (existsSync(indexPath)) {
    const raw: StoredIndex = JSON.parse(readFileSync(indexPath, "utf-8"));
    return {
      files: raw.files ?? {},
      chunks: raw.chunks.map(c => ({ ...c, embedding: decodeEmbedding(c.embedding) })),
    };
  }
  return { files: {}, chunks: [] };
}

function saveIndex(idx: VectorIndex) {
  const stored: StoredIndex = {
    files: idx.files,
    chunks: idx.chunks.map(c => ({ ...c, embedding: encodeEmbedding(c.embedding) })),
  };
  writeFileSync(indexPath, JSON.stringify(stored));
}

let index = loadIndex();

// ── MCP Server ────────────────────────────────────────────────────────────────

const server = new McpServer({ name: "rag", version: "1.0.0" });

// ── Tool: rag_index ───────────────────────────────────────────────────────────

server.tool(
  "rag_index",
  [
    "Index files from a local directory for semantic search.",
    "By default only re-indexes files that changed since the last run (incremental — fast on repeat runs).",
    "Use force:true to re-index all files unconditionally.",
    "Supported by default: .cs .json .ts .tsx .js .md .txt .yml .yaml .html .scss .css .sql .ps1 .bicep .xml .config .csproj .razor",
  ].join(" "),
  {
    directory: z.string().describe("Absolute path to the directory to index"),
    extensions: z
      .array(z.string())
      .optional()
      .describe("File extensions to include, e.g. [\".cs\", \".json\", \".pdf\", \".docx\"]. Defaults to common source and document types."),
    recursive: z.boolean().optional().describe("Recurse into subdirectories. Default: true"),
    force: z
      .boolean()
      .optional()
      .describe("Re-index all files even if unchanged. Default: false (incremental)"),
  },
  async ({ directory: rawDirectory, extensions, recursive = true, force = false }) => {
    const directory = normalizePath(rawDirectory);
    const dirPrefix = directory.endsWith("/") ? directory : directory + "/";
    const exts = new Set(extensions ?? [".cs", ".json", ".ts", ".tsx", ".md", ".txt", ".yml", ".yaml", ".pdf", ".docx", ".js", ".html", ".scss", ".css", ".sql", ".ps1", ".bicep", ".xml", ".config", ".csproj", ".razor"]);
    const pattern = recursive ? "**/*" : "*";

    const files = await glob(pattern, {
      cwd: directory,
      absolute: true,
      nodir: true,
      // Fast-path: skip directories glob should never descend into regardless of .gitignore
      ignore: ["**/.git/**", "**/node_modules/**"],
    });

    const ignoreFilters = await buildIgnoreFilters(directory);
    const matching = files.filter(f => exts.has(path.extname(f)) && !isGitignored(f, ignoreFilters));

    // Remove chunks for files that no longer exist in this directory (deleted files)
    const normalizedMatching = matching.map(normalizePath);
    const matchingSet = new Set(normalizedMatching);
    const staleFiles = Object.keys(index.files).filter(f => f.startsWith(dirPrefix) && !matchingSet.has(f));
    for (const f of staleFiles) {
      index.chunks = index.chunks.filter(c => c.file !== f);
      delete index.files[f];
    }

    let indexed = 0;
    let reused = 0;
    let skipped = 0;
    const errors: { file: string; error: string }[] = [];

    // First pass: collect chunks from changed/new files
    const pending: { file: string; text: string; start: number }[] = [];

    for (const file of normalizedMatching) {
      try {
        const mtime = statSync(file).mtimeMs;

        if (!force && index.files[file]?.mtime === mtime) {
          reused++;
          continue;
        }

        const text = await extractText(file);
        if (!text.trim()) { skipped++; continue; }

        // Drop old chunks for this file before collecting new ones
        index.chunks = index.chunks.filter(c => c.file !== file);

        for (const chunk of chunkText(text)) {
          pending.push({ file, text: chunk.text, start: chunk.start });
        }
        index.files[file] = { mtime };
        indexed++;
      } catch (err) {
        errors.push({ file, error: String(err) });
        skipped++;
      }
    }

    // Second pass: embed all pending chunks in batches of 32
    const BATCH = 32;
    for (let i = 0; i < pending.length; i += BATCH) {
      const batch = pending.slice(i, i + BATCH);
      const embeddings = await embedBatch(batch.map(c => c.text));
      for (let j = 0; j < batch.length; j++) {
        index.chunks.push({ ...batch[j], embedding: embeddings[j] });
      }
    }

    saveIndex(index);

    const lines = [
      `Indexed ${indexed} files, reused ${reused} unchanged, ${skipped} failed/empty.`,
      `Directory: ${directory}`,
      `Total chunks in index: ${index.chunks.length}`,
    ];
    if (errors.length > 0) {
      lines.push("", `Errors (first ${Math.min(errors.length, 10)} of ${errors.length}):`,
        ...errors.slice(0, 10).map(e => `  ${path.basename(e.file)}: ${e.error}`)
      );
    }

    return { content: [{ type: "text", text: lines.join("\n") }] };
  }
);

// ── Tool: rag_search ──────────────────────────────────────────────────────────

server.tool(
  "rag_search",
  "Semantic search over locally indexed files. Returns the most relevant code/text chunks for a query.",
  {
    query: z.string().describe("Natural language or code query"),
    topK: z.number().optional().describe("Number of chunks to return. Default: 5"),
    directory: z
      .string()
      .optional()
      .describe("Restrict search to files under this directory path"),
    minScore: z
      .number()
      .optional()
      .describe("Minimum similarity score threshold (0–1). Default: 0.3"),
    maxPerFile: z
      .number()
      .optional()
      .describe("Maximum results from any single file (prevents one large file monopolising results). Default: 2"),
  },
  async ({ query, topK = 5, directory: rawDirectory, minScore = 0.3, maxPerFile = 2 }) => {
    let candidates = index.chunks;

    if (rawDirectory) {
      const dirPrefix = normalizePath(rawDirectory) + "/";
      candidates = candidates.filter(c => c.file.startsWith(dirPrefix));
    }

    if (candidates.length === 0) {
      return {
        content: [{ type: "text", text: "No indexed files found. Run rag_index first." }],
      };
    }

    const qEmb = await embed(query);

    const scored = candidates
      .map(c => ({ ...c, score: dot(qEmb, c.embedding) }))
      .filter(c => c.score >= minScore)
      .sort((a, b) => b.score - a.score);

    // Apply per-file cap to prevent result monopolisation
    const fileCounts = new Map<string, number>();
    const results: typeof scored = [];
    for (const r of scored) {
      if (results.length >= topK) break;
      const count = fileCounts.get(r.file) ?? 0;
      if (count < maxPerFile) {
        results.push(r);
        fileCounts.set(r.file, count + 1);
      }
    }

    if (results.length === 0) {
      return {
        content: [{ type: "text", text: `No results above score ${minScore}. Try a different query or lower minScore.` }],
      };
    }

    const fenceLang = (f: string) => {
      const e = path.extname(f).replace(".", "");
      return ["pdf", "docx"].includes(e) ? "text" : (e || "text");
    };
    const text = results
      .map(r => `**${r.file}** (score: ${r.score.toFixed(3)})\n\`\`\`${fenceLang(r.file)}\n${r.text}\n\`\`\``)
      .join("\n\n---\n\n");

    return { content: [{ type: "text", text }] };
  }
);

// ── Tool: rag_status ──────────────────────────────────────────────────────────

server.tool(
  "rag_status",
  "Show what directories and files are currently indexed.",
  {},
  async () => {
    if (index.chunks.length === 0) {
      return { content: [{ type: "text", text: "Index is empty. Run rag_index to add files." }] };
    }

    const fileCounts = new Map<string, number>();
    for (const c of index.chunks) {
      fileCounts.set(c.file, (fileCounts.get(c.file) ?? 0) + 1);
    }

    const dirCounts = new Map<string, number>();
    for (const [file] of fileCounts) {
      const parts = file.replace(/\\/g, "/").split("/");
      const dir = parts.slice(0, -1).join("/");
      dirCounts.set(dir, (dirCounts.get(dir) ?? 0) + 1);
    }

    const lines = [
      `Total chunks: ${index.chunks.length}`,
      `Total files: ${fileCounts.size}`,
      `Tracked files (mtime): ${Object.keys(index.files).length}`,
      `Index file: ${indexPath}`,
      "",
      "Top directories by file count:",
      ...[...dirCounts.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 20)
        .map(([d, n]) => `  ${d}  (${n} files)`),
    ];

    return { content: [{ type: "text", text: lines.join("\n") }] };
  }
);

// ── Tool: rag_remove ─────────────────────────────────────────────────────────

server.tool(
  "rag_remove",
  "Remove all indexed entries for a directory (e.g. when you no longer need that codebase indexed).",
  {
    directory: z.string().describe("Absolute path prefix to remove from the index"),
  },
  async ({ directory }) => {
    const before = index.chunks.length;
    const dirPrefix = normalizePath(directory) + "/";
    index.chunks = index.chunks.filter(c => !c.file.startsWith(dirPrefix));
    for (const key of Object.keys(index.files)) {
      if (key.startsWith(dirPrefix)) delete index.files[key];
    }
    const removed = before - index.chunks.length;
    saveIndex(index);
    return {
      content: [{ type: "text", text: `Removed ${removed} chunks for ${directory}. Total remaining: ${index.chunks.length}` }],
    };
  }
);

// ── Start ─────────────────────────────────────────────────────────────────────

const transport = new StdioServerTransport();
await server.connect(transport);
