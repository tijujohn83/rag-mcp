import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { pipeline, env } from "@huggingface/transformers";
import { glob } from "glob";
import { readFileSync, writeFileSync, existsSync, statSync } from "fs";
import { readFile, writeFile, unlink, mkdtemp, rm } from "fs/promises";
import { createHash } from "crypto";
import { tmpdir } from "os";
import { execFile } from "child_process";
import { promisify } from "util";
import * as path from "path";
import * as url from "url";
import { createRequire } from "module";
import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
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
  hash: string;
}

// ── Embedding ─────────────────────────────────────────────────────────────────

// Default model: Xenova/all-MiniLM-L6-v2 (~80 MB, 384 dims). Override with
// --embed-model <name> on the command line or RAG_EMBED_MODEL env var.
const DEFAULT_EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";

function resolveEmbedModel(): string {
  const argI = process.argv.indexOf("--embed-model");
  if (argI !== -1 && process.argv[argI + 1]) return process.argv[argI + 1];
  if (process.env.RAG_EMBED_MODEL) return process.env.RAG_EMBED_MODEL;
  return DEFAULT_EMBED_MODEL;
}

const EMBED_MODEL = resolveEmbedModel();

let _embedder: Awaited<ReturnType<typeof pipeline>> | null = null;

async function getEmbedder() {
  if (!_embedder) {
    // Downloads once to .model-cache, then runs locally.
    _embedder = await pipeline("feature-extraction", EMBED_MODEL);
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

// ── Cross-encoder reranker (optional, lazy-loaded) ────────────────────────────
//
// A cross-encoder scores a (query, passage) pair jointly, giving much sharper
// relevance signals than vector cosine. We use it to re-order the top-N
// candidates produced by hybrid/vector/lexical search. Model downloads once
// to .model-cache, then runs locally. Override with --rerank-model or
// RAG_RERANK_MODEL. Default: Xenova/ms-marco-MiniLM-L-6-v2 (~80 MB).

const DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2";

function resolveRerankModel(): string {
  const argI = process.argv.indexOf("--rerank-model");
  if (argI !== -1 && process.argv[argI + 1]) return process.argv[argI + 1];
  if (process.env.RAG_RERANK_MODEL) return process.env.RAG_RERANK_MODEL;
  return DEFAULT_RERANK_MODEL;
}

const RERANK_MODEL = resolveRerankModel();

let _reranker: Awaited<ReturnType<typeof pipeline>> | null = null;

async function getReranker() {
  if (!_reranker) {
    process.stderr.write(`[rag] loading reranker '${RERANK_MODEL}' (first run downloads ~80 MB)...\n`);
    _reranker = await pipeline("text-classification", RERANK_MODEL);
  }
  return _reranker;
}

/** Score (query, passage) pairs with a cross-encoder. Returns one score per passage. */
async function rerankScores(query: string, passages: string[]): Promise<number[]> {
  if (passages.length === 0) return [];
  const r = await getReranker() as unknown as (input: unknown) => Promise<unknown>;
  const pairs = passages.map(p => ({ text: query, text_pair: p }));
  // Transformers.js returns either an array or an array-of-arrays depending on model head.
  const out = await r(pairs);
  const arr = Array.isArray(out) ? out : [out];
  return arr.map((item: unknown) => {
    if (Array.isArray(item)) {
      // Multi-label: take the "LABEL_1"/relevance score, else max
      const rel = (item as Array<{ label: string; score: number }>).find(x => /1|relevant|true/i.test(x.label));
      return rel ? rel.score : Math.max(...(item as Array<{ score: number }>).map(x => x.score));
    }
    return (item as { score: number }).score;
  });
}

// ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
//
// Combines vector (cosine via sqlite-vec) and lexical (BM25 via FTS5) ranked
// lists into a single fused score. Both rankers run as SQL queries against the
// sqlite store; this function only fuses ranks, not raw scores, so the very
// different score scales of the two rankers don't need to be normalised.

/** Reciprocal Rank Fusion: combine multiple ranked lists into a single score. */
function rrfFuse(rankings: Array<Array<{ chunkId: number; score: number }>>, k = 60): Map<number, number> {
  const fused = new Map<number, number>();
  for (const list of rankings) {
    const sorted = [...list].sort((a, b) => b.score - a.score);
    for (let r = 0; r < sorted.length; r++) {
      const id = sorted[r].chunkId;
      fused.set(id, (fused.get(id) ?? 0) + 1 / (k + r + 1));
    }
  }
  return fused;
}

// ── Text chunking (sentence + paragraph aware) ──────────────────────────────

function chunkText(text: string, maxSize = 1000, overlap = 150): { text: string; start: number }[] {
  const chunks: { text: string; start: number }[] = [];
  let windowStart = 0;

  while (windowStart < text.length) {
    const windowEnd = windowStart + maxSize;

    if (windowEnd >= text.length) {
      const chunk = text.slice(windowStart);
      if (chunk.trim()) chunks.push({ text: chunk, start: windowStart });
      break;
    }

    // Prefer break order: paragraph (\n\n) → sentence end (.?! followed by space) → line (\n) → hard cut
    let breakAt = text.lastIndexOf("\n\n", windowEnd);
    if (breakAt <= windowStart) {
      const slice = text.slice(windowStart, windowEnd);
      // Sentence-ending punctuation followed by whitespace and a capital/quote start
      const re = /[.!?]["')\]]?\s+(?=["'(\[]?[A-ZÀ-ÖØ-Þ])/g;
      let lastIdx = -1;
      let lastLen = 0;
      let m: RegExpExecArray | null;
      while ((m = re.exec(slice)) !== null) {
        lastIdx = m.index;
        lastLen = m[0].length;
      }
      if (lastIdx >= 0) breakAt = windowStart + lastIdx + lastLen;
    }
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

async function buildIgnoreFilters(rootDir: string, userPatterns: string[] = []): Promise<IgnoreFilter[]> {
  const filters: IgnoreFilter[] = [];

  // User-supplied patterns apply across the entire rootDir.
  if (userPatterns.length > 0) {
    const ig = ignore();
    ig.add(userPatterns);
    filters.push({ base: rootDir, ig });
  }

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

const _ocrWorkers = new Map<string, Awaited<ReturnType<typeof createWorker>>>();

/** Returns a cached tesseract worker for the given language code (e.g. "eng", "eng+dan"). */
async function getOcrWorker(lang: string = "eng") {
  let w = _ocrWorkers.get(lang);
  if (!w) {
    w = await createWorker(lang);
    _ocrWorkers.set(lang, w);
  }
  return w;
}

/** OCR a scanned/image-based PDF by rasterising each page to a PNG with pdf-to-img,
 *  then running tesseract on the rendered pages. Returns concatenated text. */
/** OCR a scanned/image-based PDF by rasterising each page via pdf-parse's getScreenshot
 *  (which uses node-canvas under the hood), then running tesseract on the PNG buffers. */
async function ocrPdfImages(buf: Buffer, passwords: string[] = [], ocrLang: string = "eng"): Promise<string> {
  const parser = await openPdf(buf, passwords);
  const shot = await parser.getScreenshot({ scale: 2, imageBuffer: true, imageDataUrl: false });
  const worker = await getOcrWorker(ocrLang);
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
async function ocrImageFile(file: string, ocrLang: string = "eng"): Promise<string> {
  const worker = await getOcrWorker(ocrLang);
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

// ── Vision LLM captioning (optional, OpenAI-compatible /v1/chat/completions) ─
//
// When indexing image files (and scanned PDFs) we optionally call a local vision
// LLM to generate a rich natural-language description. The caption is stored
// alongside any OCR text as an ordinary chunk, prefixed with the sentinel
// "[LLM_CAPTION]" so we can later detect files that already have one.
//
// The LLM is OPTIONAL: if the server isn't reachable or the model isn't loaded,
// captioning is silently skipped and indexing proceeds normally. On the next
// run where the LLM is available, files that lack a sentinel get re-indexed
// automatically (see incremental skip logic in rag_index).
//
// No function in this section throws — every failure path returns null.

const LLM_CAPTION_SENTINEL = "[LLM_CAPTION]";

interface LlmConfig {
  url: string;          // e.g. "http://127.0.0.1:1234"
  model: string;
  maxTokens: number;
  maxPdfPages: number;
}

function resolveLlmConfig(): LlmConfig {
  const argVal = (flag: string): string | undefined => {
    const i = process.argv.indexOf(flag);
    return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : undefined;
  };
  const url = argVal("--llm-url") ?? process.env.RAG_LLM_URL ?? "http://127.0.0.1:1234";
  const model = argVal("--llm-model") ?? process.env.RAG_LLM_MODEL ?? "qwen3.5-9b-uncensored-hauhaucs-aggressive@q8_0";
  // 0 or negative means "omit from request" — let the model use its full context budget.
  const maxTokens = Number(argVal("--llm-max-tokens") ?? process.env.RAG_LLM_MAX_TOKENS ?? 0);
  const maxPdfPages = Number(argVal("--llm-image-pages") ?? process.env.RAG_LLM_IMAGE_PAGES ?? 3);
  return { url: url.replace(/\/+$/, ""), model, maxTokens, maxPdfPages };
}

const LLM_CFG = resolveLlmConfig();

// Cached availability — pinged once per server run.
let _llmAvailable: boolean | null = null;

async function checkLlmAvailable(cfg: LlmConfig): Promise<boolean> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 3000);
    const res = await fetch(`${cfg.url}/v1/models`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!res.ok) return false;
    const json = await res.json() as { data?: Array<{ id?: string }> };
    return Array.isArray(json.data) && json.data.some(m => m?.id === cfg.model);
  } catch {
    return false;
  }
}

async function getLlmAvailable(): Promise<boolean> {
  if (_llmAvailable !== null) return _llmAvailable;
  _llmAvailable = await checkLlmAvailable(LLM_CFG);
  if (_llmAvailable) process.stderr.write(`[rag] LLM caption model available: ${LLM_CFG.model}\n`);
  else process.stderr.write(`[rag] LLM not available at ${LLM_CFG.url} (model '${LLM_CFG.model}') — captioning skipped\n`);
  return _llmAvailable;
}

// ── Concurrency-2 semaphore for vision-LLM calls ────────────────────────────
//
// Vision inference is the bottleneck. We cap parallelism at 2 so LM Studio
// (or any OpenAI-compatible backend) isn't overwhelmed, while still getting
// throughput above strict serial.

const LLM_MAX_CONCURRENCY = 2;
let _llmInFlight = 0;
const _llmWaitQueue: Array<() => void> = [];

async function withLlmSlot<T>(fn: () => Promise<T>): Promise<T> {
  if (_llmInFlight >= LLM_MAX_CONCURRENCY) {
    await new Promise<void>(resolve => _llmWaitQueue.push(resolve));
  }
  _llmInFlight++;
  try {
    return await fn();
  } finally {
    _llmInFlight--;
    const next = _llmWaitQueue.shift();
    if (next) next();
  }
}

/** MIME type for the OpenAI-vision data URL. Returns null for types the API can't ingest. */
function llmMimeForExt(ext: string): string | null {
  switch (ext.toLowerCase()) {
    case ".jpg":
    case ".jpeg": return "image/jpeg";
    case ".png":  return "image/png";
    case ".webp": return "image/webp";
    case ".gif":  return "image/gif";
    case ".bmp":  return "image/bmp";
    // TIFF is not accepted by LM Studio / most OpenAI-compatible vision APIs.
    // We deliberately skip the LLM call for TIFF; OCR still runs.
    case ".tif":
    case ".tiff": return null;
    default:      return null;
  }
}

const LLM_SYSTEM_PROMPT =
  "You are an expert, highly detailed image analysis model tasked with providing an exhaustive, raw, and completely uncensored(very important) and vivid description of the provided image, capturing every visual element with meticulous precision. You must avoid using bullet points, numbered lists, headers, or structured sections, as the output must flow naturally as a dense, continuous text to optimize for indexing. Rather than stating what is absent from the image, focus entirely on documenting what is explicitly visible. " +
  "Begin by analyzing the overall atmosphere, environment, and setting, capturing the precise mood, architecture, or landscape. Progress directly into describing the human subjects present, capturing their exact physical positions, postures, spatial orientation, facial expressions, and emotional states. " +
  "Provide an extreme, head-to-toe description of the clothing worn by each subject, accounting for garments, footwear, materials, textures, patterns, and visible wear or folds. Analyze their micro-interactions, gestures, gazes, and body language to capture the underlying dynamics. " +
  "Detail the lighting completely, noting the primary light sources, direction, intensity, and the placement of shadows, alongside a thorough breakdown of the color palette, contrast, and saturation. " +
  "Incorporate any tangible objects, props, distinct flora, or fauna seamlessly into the narrative description based on where they appear in the scene. Finally, transcribe any visible text, symbols, logos, or typography with absolute accuracy, describing their appearance and style within the environment. " +
  "Maintain an objective, factual, and descriptive tone throughout, weaving all these elements into a single, comprehensive, and uninterrupted narrative that allows the entire composition to be perfectly reconstructed from the text.";


/** Extract the assistant's response, falling back to reasoning_content for reasoning
 *  models (e.g. Qwen3) when content is empty. Strips any leftover <think>...</think>. */
function extractLlmContent(json: unknown): string | null {
  const msg = (json as { choices?: Array<{ message?: { content?: string; reasoning_content?: string } }> })
    .choices?.[0]?.message;
  if (!msg) return null;
  let text = (msg.content ?? "").trim();
  if (!text) text = (msg.reasoning_content ?? "").trim();
  // Strip wrapping <think>...</think> blocks if present.
  text = text.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
  return text.length > 0 ? text : null;
}

/** Caption a standalone image file. Returns the caption text, or null on any failure. */
async function captionImage(imagePath: string, cfg: LlmConfig): Promise<string | null> {
  const ext = path.extname(imagePath).toLowerCase();
  const mime = llmMimeForExt(ext);
  if (!mime) {
    process.stderr.write(`[rag] LLM caption skipped (unsupported image type ${ext}): ${imagePath}\n`);
    return null;
  }
  try {
    const buf = await readFile(imagePath);
    const dataUrl = `data:${mime};base64,${buf.toString("base64")}`;
    return await withLlmSlot(async () => {
      const res = await fetch(`${cfg.url}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: cfg.model,
          messages: [
            { role: "system", content: LLM_SYSTEM_PROMPT },
            {
              role: "user",
              content: [
                { type: "image_url", image_url: { url: dataUrl } },
                { type: "text", text: "Describe this image in full detail." },
              ],
            },
          ],
          ...(cfg.maxTokens > 0 ? { max_tokens: cfg.maxTokens } : {}),
          temperature: 0.2,
        }),
      });
      if (!res.ok) {
        process.stderr.write(`[rag] LLM caption failed for ${imagePath}: HTTP ${res.status}\n`);
        return null;
      }
      const json = await res.json();
      const text = extractLlmContent(json);
      if (!text) process.stderr.write(`[rag] LLM caption empty for ${imagePath}\n`);
      return text;
    });
  } catch (err) {
    process.stderr.write(`[rag] LLM caption failed for ${imagePath}: ${String(err)}\n`);
    return null;
  }
}

/** For scanned/image-heavy PDFs: render first N pages and send to the LLM with OCR text as hint.
 *  Returns the LLM's corrected/enhanced transcription, or null on any failure. */
async function captionPdfPages(buf: Buffer, ocrText: string, passwords: string[], cfg: LlmConfig): Promise<string | null> {
  try {
    const parser = await openPdf(buf, passwords);
    const shot = await parser.getScreenshot({ scale: 2, imageBuffer: true, imageDataUrl: false });
    const pages = shot.pages.filter(p => p.data && p.data.length > 0).slice(0, cfg.maxPdfPages);
    if (pages.length === 0) return null;

    const imageParts = pages.map(p => ({
      type: "image_url" as const,
      image_url: { url: `data:image/png;base64,${Buffer.from(p.data).toString("base64")}` },
    }));

    return await withLlmSlot(async () => {
      const res = await fetch(`${cfg.url}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: cfg.model,
          messages: [
            { role: "system", content: LLM_SYSTEM_PROMPT },
            {
              role: "user",
              content: [
                ...imageParts,
                {
                  type: "text",
                  text:
                    "The OCR produced this text (may be incomplete or garbled):\n\n" +
                    ocrText +
                    "\n\nRead the pages carefully and produce the accurate, complete text. " +
                    "Include all text you can see that OCR may have missed, and describe " +
                    "any non-text visual content (diagrams, photos, signatures, stamps).",
                },
              ],
            },
          ],
          ...(cfg.maxTokens > 0 ? { max_tokens: cfg.maxTokens } : {}),
          temperature: 0.2,
        }),
      });
      if (!res.ok) {
        process.stderr.write(`[rag] LLM PDF enhancement failed: HTTP ${res.status}\n`);
        return null;
      }
      const json = await res.json();
      const text = extractLlmContent(json);
      if (text) {
        process.stderr.write(`[rag] LLM enhanced PDF (pages: ${pages.length})\n`);
        return text;
      }
      return null;
    });
  } catch (err) {
    process.stderr.write(`[rag] LLM PDF enhancement failed: ${String(err)}\n`);
    return null;
  }
}

// ── File text extraction ─────────────────────────────────────────────────────

const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"]);

// Extensions treated as archive containers — their entries are indexed individually.
const ARCHIVE_CONTAINER_EXTS = new Set([".zip", ".7z"]);

// Extensions supported for extraction within archives
const ARCHIVE_SUPPORTED_EXTS = new Set([
  ".pdf", ".docx", ".txt", ".md", ".json", ".ts", ".tsx", ".js", ".cs",
  ".html", ".css", ".scss", ".yml", ".yaml", ".xml", ".sql", ".ps1",
  ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
  ".rtf", ".odt", ".pptx", ".xlsx", ".epub",
  ".csv", ".tsv", ".tex", ".rst", ".adoc", ".org",
]);

// Separator between an archive path and an entry within it: e.g. "foo.zip!inner.pdf".
const ARCHIVE_ENTRY_SEP = "!";

/** sha256 of text, first 16 hex chars — used for chunk de-duplication. */
function hashText(s: string): string {
  return createHash("sha256").update(s).digest("hex").slice(0, 16);
}

/** A unit to be indexed: either a real file on disk, or a virtual entry within an archive. */
interface IndexUnit {
  virtualPath: string;        // e.g. "C:/foo.pdf" or "C:/bar.zip!doc/inner.pdf"
  mtime: number;
  load: () => Promise<string>;
}

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
async function extractTextFromBuffer(buf: Buffer, ext: string, passwords: string[] = [], ocrLang: string = "eng", useLlmCaption: boolean = true): Promise<string> {
  if (ext === ".pdf") {
    const parser = await openPdf(buf, passwords);
    const data = await parser.getText();
    const textContent = data.text.replace(/--\s*\d+\s+of\s+\d+\s*--/g, "").trim();
    if (textContent) return data.text;
    // Scanned PDF: OCR always; then optionally enhance with LLM.
    const ocrText = await ocrPdfImages(buf, passwords, ocrLang);
    if (useLlmCaption && await getLlmAvailable()) {
      const llm = await captionPdfPages(buf, ocrText, passwords, LLM_CFG);
      if (llm) return `${LLM_CAPTION_SENTINEL}\n${llm}`;
    }
    return ocrText;
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
    // Tesseract needs a file path — write to temp, OCR, optionally caption, clean up.
    const tmpFile = path.join(tmpdir(), `rag-ocr-${Date.now()}-${Math.random().toString(36).slice(2)}${ext}`);
    try {
      await writeFile(tmpFile, buf);
      const ocrText = await ocrImageFile(tmpFile, ocrLang);
      if (useLlmCaption && await getLlmAvailable()) {
        const caption = await captionImage(tmpFile, LLM_CFG);
        if (caption) {
          return `${LLM_CAPTION_SENTINEL}\n${caption}\n\n[OCR]\n${ocrText}`.trim();
        }
      }
      return ocrText;
    } finally {
      await unlink(tmpFile).catch(() => {});
    }
  }
  return buf.toString("utf-8");
}

/** Enumerate supported entries inside an archive. Returns one item per supported entry,
 *  each tagged with the archive-relative entry path. */
async function enumerateArchiveEntries(
  file: string,
  passwords: string[] = [],
  ocrLang: string = "eng",
  useLlmCaption: boolean = true,
): Promise<Array<{ entryPath: string; text: string }>> {
  const ext = path.extname(file).toLowerCase();
  const out: Array<{ entryPath: string; text: string }> = [];

  if (ext === ".zip") {
    const zip = new AdmZip(file);
    for (const entry of zip.getEntries()) {
      if (entry.isDirectory) continue;
      const eExt = path.extname(entry.entryName).toLowerCase();
      if (!ARCHIVE_SUPPORTED_EXTS.has(eExt)) continue;
      try {
        const buf = entry.getData();
        const text = await extractTextFromBuffer(buf, eExt, passwords, ocrLang, useLlmCaption);
        if (text.trim()) out.push({ entryPath: entry.entryName.replace(/\\/g, "/"), text });
      } catch { /* skip unreadable entries */ }
    }
    return out;
  }

  if (ext === ".7z") {
    const tmpDir = await mkdtemp(path.join(tmpdir(), "rag-7z-"));
    try {
      await execFileAsync(path7za, ["x", file, `-o${tmpDir}`, "-y"]);
      const extracted = await glob("**/*", { cwd: tmpDir, absolute: true, nodir: true });
      for (const f of extracted) {
        const eExt = path.extname(f).toLowerCase();
        if (!ARCHIVE_SUPPORTED_EXTS.has(eExt)) continue;
        try {
          const text = await extractText(f, passwords, ocrLang, useLlmCaption);
          if (text.trim()) out.push({ entryPath: path.relative(tmpDir, f).replace(/\\/g, "/"), text });
        } catch { /* skip */ }
      }
      return out;
    } finally {
      await rm(tmpDir, { recursive: true, force: true });
    }
  }

  return out;
}

async function extractText(file: string, passwords: string[] = [], ocrLang: string = "eng", useLlmCaption: boolean = true): Promise<string> {
  const ext = path.extname(file).toLowerCase();
  // Archive containers are handled separately by the indexer via enumerateArchiveEntries.
  // If called directly here, fall back to concatenating entries (legacy behaviour).
  if (ARCHIVE_CONTAINER_EXTS.has(ext)) {
    const entries = await enumerateArchiveEntries(file, passwords, ocrLang, useLlmCaption);
    return entries.map(e => `[${e.entryPath}]\n${e.text.trim()}`).join("\n\n");
  }
  if (ext === ".pdf") {
    const buf = await readFile(file);
    const parser = await openPdf(buf, passwords);
    const data = await parser.getText();
    // Embedded text present → fastest, most accurate path (no OCR, no LLM).
    const textContent = data.text.replace(/--\s*\d+\s+of\s+\d+\s*--/g, "").trim();
    if (textContent.length > 100) return data.text;
    // Scanned / thin-text PDF: OCR always; optionally enhance with LLM.
    const ocrText = await ocrPdfImages(buf, passwords, ocrLang);
    if (useLlmCaption && await getLlmAvailable()) {
      const llm = await captionPdfPages(buf, ocrText, passwords, LLM_CFG);
      if (llm) return `${LLM_CAPTION_SENTINEL}\n${llm}`;
    }
    return ocrText || data.text;
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
    const ocrText = await ocrImageFile(file, ocrLang);
    if (useLlmCaption && await getLlmAvailable()) {
      const caption = await captionImage(file, LLM_CFG);
      if (caption) {
        return `${LLM_CAPTION_SENTINEL}\n${caption}\n\n[OCR]\n${ocrText}`.trim();
      }
    }
    return ocrText;
  }
  // plain text (includes .cs, .json, .ts, .md, .txt, .yml, .csv, .tsv, .tex, .rst, .adoc, .org, etc.)
  return readFileSync(file, "utf-8");
}

// ── Path normalisation (forward slashes, consistent casing via resolve) ────────

// glob returns forward-slash paths on Windows; path.resolve returns backslashes.
// Normalise everything to forward slashes so startsWith comparisons are reliable.
const normalizePath = (p: string) => path.resolve(p).replace(/\\/g, "/");

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
// in writeChunk/removeFileChunks because sqlite-vec doesn't support triggers
// on virtual tables.

const argIndex = process.argv.indexOf("--index");
const indexArg = argIndex !== -1 && process.argv[argIndex + 1] ? process.argv[argIndex + 1] : null;

// Resolve DB path. If user passed --index foo.json, switch to foo.db automatically.
const dbPath: string = (() => {
  if (indexArg) {
    return indexArg.toLowerCase().endsWith(".json")
      ? indexArg.replace(/\.json$/i, ".db")
      : indexArg;
  }
  return path.join(__dirname, "rag-index.db");
})();

function f32ToBuffer(f32: Float32Array): Buffer {
  return Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength);
}

interface RagDb {
  db: Database.Database;
  embedDim: number;
  embedModel: string;
}

function openDb(): RagDb {
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
function ensureVecTable(rdb: RagDb, dim: number): void {
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

function getFileMtime(rdb: RagDb, file: string): number | undefined {
  const row = rdb.db.prepare("SELECT mtime FROM files WHERE path = ?").get(file) as { mtime: number } | undefined;
  return row?.mtime;
}

function setFileMtime(rdb: RagDb, file: string, mtime: number): void {
  rdb.db.prepare("INSERT OR REPLACE INTO files(path, mtime) VALUES (?, ?)").run(file, mtime);
}

function listFilesUnderPrefix(rdb: RagDb, prefix: string): { path: string; mtime: number }[] {
  return rdb.db.prepare("SELECT path, mtime FROM files WHERE path LIKE ?").all(prefix + "%") as { path: string; mtime: number }[];
}

function hashExists(rdb: RagDb, hash: string): boolean {
  return (rdb.db.prepare("SELECT 1 FROM chunks WHERE hash = ? LIMIT 1").get(hash) as unknown) !== undefined;
}

/** True if this file already has at least one chunk produced by the vision LLM
 *  (detected via the [LLM_CAPTION] sentinel prefix). Used to skip files that
 *  are already captioned, and conversely to re-process files that aren't. */
function hasLlmCaption(rdb: RagDb, file: string): boolean {
  return (rdb.db
    .prepare("SELECT 1 FROM chunks WHERE file = ? AND text LIKE ? LIMIT 1")
    .get(file, LLM_CAPTION_SENTINEL + "%") as unknown) !== undefined;
}

function removeFileChunks(rdb: RagDb, file: string): number {
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

function removePrefix(rdb: RagDb, prefix: string): number {
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

interface InsertChunkArgs { file: string; start: number; text: string; hash: string; embedding: Float32Array; }

/** Insert one chunk + its embedding. Caller must have called ensureVecTable. */
function insertChunkRow(rdb: RagDb, c: InsertChunkArgs): number {
  const info = rdb.db.prepare("INSERT INTO chunks(file, start, text, hash) VALUES (?, ?, ?, ?)")
    .run(c.file, c.start, c.text, c.hash);
  const id = Number(info.lastInsertRowid);
  // sqlite-vec requires a BigInt rowid binding (it rejects JS numbers as "non-integer").
  rdb.db.prepare("INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)").run(BigInt(id), f32ToBuffer(c.embedding));
  return id;
}

function totalChunks(rdb: RagDb): number {
  return (rdb.db.prepare("SELECT count(*) AS n FROM chunks").get() as { n: number }).n;
}
function totalFiles(rdb: RagDb): number {
  return (rdb.db.prepare("SELECT count(DISTINCT file) AS n FROM chunks").get() as { n: number }).n;
}
function totalTrackedFiles(rdb: RagDb): number {
  return (rdb.db.prepare("SELECT count(*) AS n FROM files").get() as { n: number }).n;
}

// Initialise DB on startup.
const rdb = openDb();

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
    ignorePatterns: z
      .array(z.string())
      .optional()
      .describe("Additional .gitignore-style patterns to skip files (e.g. [\"~$*\", \"*.tmp\", \"**/backup/**\"])."),
    ocrLang: z
      .string()
      .optional()
      .describe("Tesseract OCR language(s), e.g. \"eng\", \"dan\", \"eng+dan\". Default: \"eng\". Required language data must be installed."),
    recursive: z.boolean().optional().describe("Recurse into subdirectories. Default: true"),
    force: z
      .boolean()
      .optional()
      .describe("Re-index all files even if unchanged. Default: false (incremental)"),
    captionImages: z
      .boolean()
      .optional()
      .describe("Use the local vision LLM to caption image files and enhance scanned PDFs. When the LLM is reachable, images and scanned PDFs that lack a caption are re-processed automatically on subsequent runs. Default: true"),
  },
  async ({ directory: rawDirectory, extensions, passwords, ignorePatterns, ocrLang = "eng", recursive = true, force = false, captionImages = true }) => {
    const directory = normalizePath(rawDirectory);
    const dirPrefix = directory.endsWith("/") ? directory : directory + "/";
    const exts = new Set(extensions ?? [".cs", ".json", ".ts", ".tsx", ".md", ".txt", ".yml", ".yaml", ".pdf", ".docx", ".rtf", ".odt", ".pptx", ".xlsx", ".epub", ".csv", ".tsv", ".tex", ".rst", ".adoc", ".org", ".js", ".html", ".scss", ".css", ".sql", ".ps1", ".bicep", ".xml", ".config", ".csproj", ".razor", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".zip", ".7z"]);
    const pattern = recursive ? "**/*" : "*";

    const files = await glob(pattern, {
      cwd: directory,
      absolute: true,
      nodir: true,
      ignore: ["**/.git/**", "**/node_modules/**"],
    });

    const ignoreFilters = await buildIgnoreFilters(directory, ignorePatterns);
    const matching = files.filter(f => exts.has(path.extname(f)) && !isGitignored(f, ignoreFilters));

    // Remove chunks for files that no longer exist in this directory.
    // Virtual archive entries (containing "!") survive while their parent archive does;
    // missing entries within an archive are pruned when that archive is re-processed.
    const normalizedMatching = matching.map(normalizePath);
    const matchingSet = new Set(normalizedMatching);
    const tracked = listFilesUnderPrefix(rdb, dirPrefix);
    const staleFiles = tracked.map(t => t.path).filter(p => {
      if (matchingSet.has(p)) return false;
      const bang = p.indexOf(ARCHIVE_ENTRY_SEP);
      if (bang >= 0) return !matchingSet.has(p.slice(0, bang));
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

    // Log to stderr — stdout is reserved for the MCP stdio protocol.
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
              if ((IMAGE_EXTS.has(exExt) || exExt === ".pdf") && !hasLlmCaption(rdb, ex.path)) {
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
          && !hasLlmCaption(rdb, file);

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

    // File-level concurrency. Matched to the vision-LLM semaphore (LLM_MAX_CONCURRENCY)
    // so image-heavy directories can keep both LLM slots busy. For text-only runs the
    // small parallelism still helps overlap I/O with embedding work.
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

    log(`Finished: indexed=${indexed} reused=${reused} skipped=${skipped} in ${Math.round((Date.now() - t0) / 1000)}s`);

    const chunkCount = totalChunks(rdb);
    const lines = [
      `Indexed ${indexed} files, reused ${reused} unchanged, ${skipped} failed/empty.`,
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
    extensions: z
      .array(z.string())
      .optional()
      .describe("Restrict search to these file extensions (e.g. [\".pdf\", \".md\"])."),
    fileGlob: z
      .string()
      .optional()
      .describe("Restrict to files matching this glob pattern (e.g. \"**/payslip*.pdf\")."),
    mtimeFrom: z
      .number()
      .optional()
      .describe("Only include files modified on or after this Unix epoch ms."),
    mtimeTo: z
      .number()
      .optional()
      .describe("Only include files modified on or before this Unix epoch ms."),
    minScore: z
      .number()
      .optional()
      .describe("Minimum similarity score threshold (0–1) for vector mode. Default: 0.3"),
    maxPerFile: z
      .number()
      .optional()
      .describe("Maximum results from any single file (prevents one large file monopolising results). Default: 2"),
    mode: z
      .enum(["hybrid", "vector", "lexical"])
      .optional()
      .describe("Search mode: hybrid combines vector + BM25 via RRF (best for most queries); vector = semantic only; lexical = BM25 keyword only. Default: hybrid"),
    rerank: z
      .boolean()
      .optional()
      .describe("Re-rank top candidates with a cross-encoder for sharper relevance (slower, ~80 MB model downloaded on first use). Default: false"),
    rerankTopN: z
      .number()
      .optional()
      .describe("Number of fused candidates to feed into the cross-encoder reranker. Default: 30"),
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

    /** Post-filter a candidate row by user-supplied extension/glob/mtime filters. */
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
    // The directory prefix is the only filter we can push into SQL cheaply.
    const dirClause = dirPrefix ? "c.file LIKE ?" : "1";
    const dirParam: unknown[] = dirPrefix ? [dirPrefix + "%"] : [];

    interface Ranked { chunkId: number; score: number; }

    // ── Vector ranking via sqlite-vec ────────────────────────────────────────
    let vectorList: Ranked[] = [];
    if ((mode === "vector" || mode === "hybrid") && rdb.embedDim > 0) {
      const qEmb = await embed(query);
      // sqlite-vec: pull a broader pool then filter, since MATCH + WHERE-on-joined-cols isn't reliable.
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
        // Convert L2 distance on normalised embeddings to a cosine-similarity score in [0, 1].
        // For unit vectors: cosine = 1 - dist^2 / 2.
        const score = 1 - (r.dist * r.dist) / 2;
        if (mode === "vector" && score < minScore) continue;
        vectorList.push({ chunkId: r.id, score });
        if (vectorList.length >= POOL) break;
      }
    }

    // ── Lexical ranking via FTS5 BM25 ────────────────────────────────────────
    let lexicalList: Ranked[] = [];
    if (mode === "lexical" || mode === "hybrid") {
      // Quote each token to avoid FTS5 syntax errors on user input.
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
            // bm25() returns negative values where smaller (more negative) = more relevant; negate.
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

// ── Tool: rag_status ──────────────────────────────────────────────────────────

server.tool(
  "rag_status",
  "Show what directories and files are currently indexed.",
  {},
  async () => {
    const chunkCount = totalChunks(rdb);
    if (chunkCount === 0) {
      return { content: [{ type: "text", text: "Index is empty. Run rag_index to add files." }] };
    }

    const dirRows = rdb.db.prepare("SELECT DISTINCT file FROM chunks").all() as { file: string }[];

    const dirCounts = new Map<string, number>();
    for (const { file } of dirRows) {
      const parts = file.replace(/\\/g, "/").split("/");
      const dir = parts.slice(0, -1).join("/");
      dirCounts.set(dir, (dirCounts.get(dir) ?? 0) + 1);
    }

    const lines = [
      `Total chunks: ${chunkCount}`,
      `Total files: ${totalFiles(rdb)}`,
      `Tracked files (mtime): ${totalTrackedFiles(rdb)}`,
      `Index file: ${dbPath}`,
      `Embed model (current): ${EMBED_MODEL}`,
      `Embed model (index):   ${rdb.embedModel}`,
      `Embed dim:             ${rdb.embedDim}`,
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
    const dirPrefix = normalizePath(directory) + "/";
    const removed = removePrefix(rdb, dirPrefix);
    const remaining = totalChunks(rdb);
    return {
      content: [{ type: "text", text: `Removed ${removed} chunks for ${directory}. Total remaining: ${remaining}` }],
    };
  }
);

// ── Start ─────────────────────────────────────────────────────────────────────

const transport = new StdioServerTransport();
await server.connect(transport);
