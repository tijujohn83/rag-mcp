import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { pipeline, env } from "@huggingface/transformers";
import { glob } from "glob";
import { readFileSync, writeFileSync, existsSync, statSync } from "fs";
import { readFile, writeFile, unlink, mkdtemp, rm } from "fs/promises";
import { tmpdir } from "os";
import { execFile } from "child_process";
import { promisify } from "util";
import * as path from "path";
import * as url from "url";
import { createRequire } from "module";
const execFileAsync = promisify(execFile);
const require = createRequire(import.meta.url);
interface PDFImage { dataUrl?: string; width: number; height: number; kind: number; }
interface PDFImageResult { pages: { pageNumber: number; images: PDFImage[] }[]; total: number; }
interface PDFScreenshotPage { pageNumber: number; width: number; height: number; data: Uint8Array; dataUrl?: string; }
interface PDFScreenshotResult { pages: PDFScreenshotPage[]; total: number; }
const { PDFParse } = require("pdf-parse") as {
  PDFParse: new (opts: { data: Buffer; password?: string }) => {
    load(): Promise<void>;
    getText(): Promise<{ text: string }>;
    getImage(opts?: Record<string, unknown>): Promise<PDFImageResult>;
    getScreenshot(opts?: { scale?: number; imageBuffer?: boolean; imageDataUrl?: boolean }): Promise<PDFScreenshotResult>;
  };
};
const { createWorker } = require("tesseract.js") as { createWorker: (lang: string) => Promise<{ recognize(src: Buffer | string): Promise<{ data: { text: string } }>; terminate(): Promise<void> }> };
const AdmZip = require("adm-zip") as new (input: string) => { getEntries(): { isDirectory: boolean; entryName: string; getData(): Buffer }[] };
const { path7za } = require("7zip-bin") as { path7za: string };
import mammoth from "mammoth";
import ignore, { Ignore } from "ignore";
import ExcelJS from "exceljs";

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

// ── OCR (tesseract.js) ───────────────────────────────────────────────────────

let _ocrWorker: Awaited<ReturnType<typeof createWorker>> | null = null;

async function getOcrWorker() {
  if (!_ocrWorker) {
    _ocrWorker = await createWorker("eng");
  }
  return _ocrWorker;
}

/** OCR a scanned/image-based PDF by rasterising each page to a PNG with pdf-to-img,
 *  then running tesseract on the rendered pages. Returns concatenated text. */
/** OCR a scanned/image-based PDF by rasterising each page via pdf-parse's getScreenshot
 *  (which uses node-canvas under the hood), then running tesseract on the PNG buffers. */
async function ocrPdfImages(buf: Buffer, passwords: string[] = []): Promise<string> {
  const parser = await openPdf(buf, passwords);
  const shot = await parser.getScreenshot({ scale: 2, imageBuffer: true, imageDataUrl: false });
  const worker = await getOcrWorker();
  const pageTexts: string[] = [];
  for (const page of shot.pages) {
    if (!page.data || page.data.length === 0) continue;
    try {
      const { data: { text } } = await worker.recognize(Buffer.from(page.data));
      if (text.trim()) pageTexts.push(text.trim());
    } catch { /* skip unreadable page */ }
  }
  return pageTexts.join("\n\n");
}

/** OCR a standalone image file (png, jpg, tiff, etc.) */
async function ocrImageFile(file: string): Promise<string> {
  const worker = await getOcrWorker();
  const { data: { text } } = await worker.recognize(file);
  return text;
}

/** Open a PDF, trying each password if the document is encrypted. Returns the loaded parser. */
async function openPdf(buf: Buffer, passwords: string[] = []): Promise<InstanceType<typeof PDFParse>> {
  const candidates = [undefined, ...passwords];
  let lastErr: unknown;
  for (const pw of candidates) {
    try {
      const parser = new PDFParse(pw === undefined ? { data: buf } : { data: buf, password: pw });
      await parser.load();
      return parser;
    } catch (err) {
      lastErr = err;
      const name = (err as { name?: string } | null)?.name;
      // Only retry with next password on password-related errors
      if (name !== "PasswordException") throw err;
    }
  }
  throw lastErr;
}

// ── File text extraction ─────────────────────────────────────────────────────

const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"]);

// Extensions supported for extraction within archives
const ARCHIVE_SUPPORTED_EXTS = new Set([
  ".pdf", ".docx", ".txt", ".md", ".json", ".ts", ".tsx", ".js", ".cs",
  ".html", ".css", ".scss", ".yml", ".yaml", ".xml", ".sql", ".ps1",
  ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
  ".rtf", ".odt", ".pptx", ".xlsx", ".epub",
  ".csv", ".tsv", ".tex", ".rst", ".adoc", ".org",
]);

// ── Document extractors for richer formats ──────────────────────────────────

/** Strip XML/HTML tags, decode common entities, collapse whitespace. */
function stripMarkup(s: string): string {
  return s
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(parseInt(n, 10)))
    .replace(/[ \t]+/g, " ")
    .replace(/\s*\n\s*/g, "\n")
    .trim();
}

/** Strip RTF control words and groups, leaving plain text. */
function extractRtfText(buf: Buffer): string {
  let s = buf.toString("utf-8");
  // Decode \'xx hex escapes (latin-1 best-effort)
  s = s.replace(/\\'([0-9a-fA-F]{2})/g, (_, h) => String.fromCharCode(parseInt(h, 16)));
  // Convert paragraph/line breaks to newlines
  s = s.replace(/\\par[d]?\b/g, "\n").replace(/\\line\b/g, "\n");
  // Remove remaining control words (\word optionally followed by number/space)
  s = s.replace(/\\[a-zA-Z]+-?\d* ?/g, "");
  // Remove control symbols and braces
  s = s.replace(/\\[^a-zA-Z]/g, "").replace(/[{}]/g, "");
  return s.replace(/[ \t]+/g, " ").replace(/\s*\n\s*/g, "\n").trim();
}

/** Extract text from a zip-based document by reading specific entries and stripping markup. */
function extractZipMarkupText(buf: Buffer, entryFilter: (name: string) => boolean): string {
  const tmpFile = path.join(tmpdir(), `rag-zip-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  writeFileSync(tmpFile, buf);
  try {
    const zip = new AdmZip(tmpFile);
    const parts: string[] = [];
    for (const entry of zip.getEntries()) {
      if (entry.isDirectory) continue;
      if (!entryFilter(entry.entryName)) continue;
      try {
        const text = stripMarkup(entry.getData().toString("utf-8"));
        if (text) parts.push(text);
      } catch { /* skip */ }
    }
    return parts.join("\n\n");
  } finally {
    try { require("fs").unlinkSync(tmpFile); } catch { /* ignore */ }
  }
}

/** Extract text from .pptx — slide XML files under ppt/slides/. */
function extractPptxText(buf: Buffer): string {
  return extractZipMarkupText(buf, n => /^ppt\/slides\/slide\d+\.xml$/i.test(n));
}

/** Extract text from .odt — content.xml. */
function extractOdtText(buf: Buffer): string {
  return extractZipMarkupText(buf, n => n.toLowerCase() === "content.xml");
}

/** Extract text from .epub — all .xhtml/.html entries. */
function extractEpubText(buf: Buffer): string {
  return extractZipMarkupText(buf, n => /\.(x?html|htm)$/i.test(n));
}

/** Extract text from .xlsx — all cells from all worksheets. */
async function extractXlsxText(buf: Buffer): Promise<string> {
  const wb = new ExcelJS.Workbook();
  await wb.xlsx.load(buf as unknown as ArrayBuffer);
  const parts: string[] = [];
  wb.eachSheet(sheet => {
    const rows: string[] = [];
    sheet.eachRow(row => {
      const vals: string[] = [];
      row.eachCell(cell => {
        const v = cell.value;
        if (v == null) return;
        if (typeof v === "object" && "text" in (v as object)) vals.push(String((v as { text: unknown }).text));
        else if (typeof v === "object" && "richText" in (v as object)) {
          vals.push(((v as { richText: { text: string }[] }).richText).map(r => r.text).join(""));
        } else if (typeof v === "object" && "result" in (v as object)) vals.push(String((v as { result: unknown }).result ?? ""));
        else vals.push(String(v));
      });
      if (vals.length) rows.push(vals.join("\t"));
    });
    if (rows.length) parts.push(`[${sheet.name}]\n${rows.join("\n")}`);
  });
  return parts.join("\n\n");
}

/** Extract text from a buffer given its file extension — used for archive entries. */
async function extractTextFromBuffer(buf: Buffer, ext: string, passwords: string[] = []): Promise<string> {
  if (ext === ".pdf") {
    const parser = await openPdf(buf, passwords);
    const data = await parser.getText();
    const textContent = data.text.replace(/--\s*\d+\s+of\s+\d+\s*--/g, "").trim();
    if (!textContent) return ocrPdfImages(buf, passwords);
    return data.text;
  }
  if (ext === ".docx") {
    const result = await mammoth.extractRawText({ buffer: buf });
    return result.value;
  }
  if (ext === ".rtf") return extractRtfText(buf);
  if (ext === ".odt") return extractOdtText(buf);
  if (ext === ".pptx") return extractPptxText(buf);
  if (ext === ".epub") return extractEpubText(buf);
  if (ext === ".xlsx") return extractXlsxText(buf);
  if (IMAGE_EXTS.has(ext)) {
    // Tesseract needs a file path — write to temp, OCR, clean up
    const tmpFile = path.join(tmpdir(), `rag-ocr-${Date.now()}-${Math.random().toString(36).slice(2)}${ext}`);
    try {
      await writeFile(tmpFile, buf);
      return await ocrImageFile(tmpFile);
    } finally {
      await unlink(tmpFile).catch(() => {});
    }
  }
  return buf.toString("utf-8");
}

/** Extract and concatenate text from all supported entries in a ZIP file. */
async function extractZipText(file: string, passwords: string[] = []): Promise<string> {
  const zip = new AdmZip(file);
  const entries = zip.getEntries();
  const texts: string[] = [];
  for (const entry of entries) {
    if (entry.isDirectory) continue;
    const ext = path.extname(entry.entryName).toLowerCase();
    if (!ARCHIVE_SUPPORTED_EXTS.has(ext)) continue;
    try {
      const buf = entry.getData();
      const text = await extractTextFromBuffer(buf, ext, passwords);
      if (text.trim()) texts.push(`[${entry.entryName}]\n${text.trim()}`);
    } catch { /* skip unreadable entries */ }
  }
  return texts.join("\n\n");
}

/** Extract and concatenate text from all supported files in a 7-Zip archive. */
async function extract7zText(file: string, passwords: string[] = []): Promise<string> {
  const tmpDir = await mkdtemp(path.join(tmpdir(), "rag-7z-"));
  try {
    await execFileAsync(path7za, ["x", file, `-o${tmpDir}`, "-y"]);
    const extracted = await glob("**/*", { cwd: tmpDir, absolute: true, nodir: true });
    const texts: string[] = [];
    for (const f of extracted) {
      const ext = path.extname(f).toLowerCase();
      if (!ARCHIVE_SUPPORTED_EXTS.has(ext)) continue;
      try {
        const text = await extractText(f, passwords);
        if (text.trim()) texts.push(`[${path.relative(tmpDir, f)}]\n${text.trim()}`);
      } catch { /* skip */ }
    }
    return texts.join("\n\n");
  } finally {
    await rm(tmpDir, { recursive: true, force: true });
  }
}

async function extractText(file: string, passwords: string[] = []): Promise<string> {
  const ext = path.extname(file).toLowerCase();
  if (ext === ".zip") return extractZipText(file, passwords);
  if (ext === ".7z") return extract7zText(file, passwords);
  if (ext === ".pdf") {
    const buf = await readFile(file);
    const parser = await openPdf(buf, passwords);
    const data = await parser.getText();
    // If PDF has no embedded text (scanned/image-based), fall back to OCR
    const textContent = data.text.replace(/--\s*\d+\s+of\s+\d+\s*--/g, "").trim();
    if (!textContent) {
      return ocrPdfImages(buf, passwords);
    }
    return data.text;
  }
  if (ext === ".docx") {
    const buf = await readFile(file);
    const result = await mammoth.extractRawText({ buffer: buf });
    return result.value;
  }
  if (ext === ".rtf") return extractRtfText(await readFile(file));
  if (ext === ".odt") return extractOdtText(await readFile(file));
  if (ext === ".pptx") return extractPptxText(await readFile(file));
  if (ext === ".epub") return extractEpubText(await readFile(file));
  if (ext === ".xlsx") return extractXlsxText(await readFile(file));
  if (IMAGE_EXTS.has(ext)) {
    return ocrImageFile(file);
  }
  // plain text (includes .cs, .json, .ts, .md, .txt, .yml, .csv, .tsv, .tex, .rst, .adoc, .org, etc.)
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
    "Supported by default: .cs .json .ts .tsx .js .md .txt .yml .yaml .html .scss .css .sql .ps1 .bicep .xml .config .csproj .razor .pdf .docx .rtf .odt .pptx .xlsx .epub .csv .tsv .tex .rst .adoc .org .png .jpg .jpeg .tiff .tif .bmp .webp .zip .7z",
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
    recursive: z.boolean().optional().describe("Recurse into subdirectories. Default: true"),
    force: z
      .boolean()
      .optional()
      .describe("Re-index all files even if unchanged. Default: false (incremental)"),
  },
  async ({ directory: rawDirectory, extensions, passwords, recursive = true, force = false }) => {
    const directory = normalizePath(rawDirectory);
    const dirPrefix = directory.endsWith("/") ? directory : directory + "/";
    const exts = new Set(extensions ?? [".cs", ".json", ".ts", ".tsx", ".md", ".txt", ".yml", ".yaml", ".pdf", ".docx", ".rtf", ".odt", ".pptx", ".xlsx", ".epub", ".csv", ".tsv", ".tex", ".rst", ".adoc", ".org", ".js", ".html", ".scss", ".css", ".sql", ".ps1", ".bicep", ".xml", ".config", ".csproj", ".razor", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".zip", ".7z"]);
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

        const text = await extractText(file, passwords);
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
