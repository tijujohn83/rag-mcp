import { readFile, mkdtemp, rm } from "fs/promises";
import { writeFileSync } from "fs";
import { tmpdir } from "os";
import * as path from "path";
import { execFile } from "child_process";
import { promisify } from "util";
import { createRequire } from "module";
import { glob } from "glob";
import mammoth from "mammoth";
import ExcelJS from "exceljs";
import {
  IMAGE_EXTS, HEIC_EXTS, ARCHIVE_CONTAINER_EXTS, ARCHIVE_SUPPORTED_EXTS,
  LLM_CAPTION_SENTINEL,
} from "./paths.js";
import { openPdf } from "./pdf.js";
import { ocrPdfImages, ocrImageFile } from "./ocr.js";
import { captionImage, captionPdfPages, getLlmAvailable } from "./llm-caption.js";
import { LLM_CFG } from "./config.js";
import { prepareImageFile, prepareImageBuffer } from "./image-prep.js";

const execFileAsync = promisify(execFile);
const require = createRequire(import.meta.url);

const AdmZip = require("adm-zip") as new (input: string) => {
  getEntries(): { isDirectory: boolean; entryName: string; getData(): Buffer }[];
};
const { path7za } = require("7zip-bin") as { path7za: string };

/** Strip XML/HTML tags, decode common entities, collapse whitespace. */
export function stripMarkup(s: string): string {
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
export function extractRtfText(buf: Buffer): string {
  let s = buf.toString("utf-8");
  s = s.replace(/\\'([0-9a-fA-F]{2})/g, (_, h) => String.fromCharCode(parseInt(h, 16)));
  s = s.replace(/\\par[d]?\b/g, "\n").replace(/\\line\b/g, "\n");
  s = s.replace(/\\[a-zA-Z]+-?\d* ?/g, "");
  s = s.replace(/\\[^a-zA-Z]/g, "").replace(/[{}]/g, "");
  return s.replace(/[ \t]+/g, " ").replace(/\s*\n\s*/g, "\n").trim();
}

/** Extract text from a zip-based document by reading specific entries and stripping markup. */
export function extractZipMarkupText(buf: Buffer, entryFilter: (name: string) => boolean): string {
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

export function extractPptxText(buf: Buffer): string {
  return extractZipMarkupText(buf, n => /^ppt\/slides\/slide\d+\.xml$/i.test(n));
}

export function extractOdtText(buf: Buffer): string {
  return extractZipMarkupText(buf, n => n.toLowerCase() === "content.xml");
}

export function extractEpubText(buf: Buffer): string {
  return extractZipMarkupText(buf, n => /\.(x?html|htm)$/i.test(n));
}

export async function extractXlsxText(buf: Buffer): Promise<string> {
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
export async function extractTextFromBuffer(
  buf: Buffer,
  ext: string,
  passwords: string[] = [],
  ocrLang: string = "eng",
  useLlmCaption: boolean = true,
): Promise<string> {
  if (ext === ".pdf") {
    const parser = await openPdf(buf, passwords);
    const data = await parser.getText();
    const textContent = data.text.replace(/--\s*\d+\s+of\s+\d+\s*--/g, "").trim();
    if (textContent) return data.text;
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
    // Tesseract needs a file path. For HEIC, prepareImageBuffer converts to JPEG first;
    // for everything else it writes the original buffer to a temp file.
    const prepared = await prepareImageBuffer(buf, ext);
    try {
      const ocrText = await ocrImageFile(prepared.path, ocrLang);
      if (useLlmCaption && await getLlmAvailable()) {
        // The prepared file is already a vision-LLM-friendly format (JPEG for HEIC,
        // original ext otherwise), so captionImage can read it directly.
        const caption = await captionImage(prepared.path, LLM_CFG);
        if (caption) {
          return `${LLM_CAPTION_SENTINEL}\n${caption}\n\n[OCR]\n${ocrText}`.trim();
        }
      }
      return ocrText;
    } finally {
      await prepared.cleanup();
    }
  }
  return buf.toString("utf-8");
}

/** Enumerate supported entries inside an archive. Returns one item per supported entry,
 *  each tagged with the archive-relative entry path. */
export async function enumerateArchiveEntries(
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

export async function extractText(
  file: string,
  passwords: string[] = [],
  ocrLang: string = "eng",
  useLlmCaption: boolean = true,
): Promise<string> {
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
    // HEIC/HEIF: convert to JPEG temp file; everything else passes through unchanged.
    const prepared = HEIC_EXTS.has(ext) ? await prepareImageFile(file) : null;
    const target = prepared?.path ?? file;
    try {
      const ocrText = await ocrImageFile(target, ocrLang);
      if (useLlmCaption && await getLlmAvailable()) {
        const caption = await captionImage(target, LLM_CFG);
        if (caption) {
          return `${LLM_CAPTION_SENTINEL}\n${caption}\n\n[OCR]\n${ocrText}`.trim();
        }
      }
      return ocrText;
    } finally {
      if (prepared) await prepared.cleanup();
    }
  }
  // plain text (includes .cs, .json, .ts, .md, .txt, .yml, .csv, .tsv, .tex, .rst, .adoc, .org, etc.)
  return (await readFile(file)).toString("utf-8");
}
