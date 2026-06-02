import * as path from "path";
import { createHash } from "crypto";

// glob returns forward-slash paths on Windows; path.resolve returns backslashes.
// Normalise everything to forward slashes so startsWith comparisons are reliable.
export const normalizePath = (p: string) => path.resolve(p).replace(/\\/g, "/");

/** sha256 of text, first 16 hex chars — used for chunk de-duplication. */
export function hashText(s: string): string {
  return createHash("sha256").update(s).digest("hex").slice(0, 16);
}

// Separator between an archive path and an entry within it: e.g. "foo.zip!inner.pdf".
export const ARCHIVE_ENTRY_SEP = "!";

// Sentinel prefix marking chunks produced by the vision LLM caption pipeline.
export const LLM_CAPTION_SENTINEL = "[LLM_CAPTION]";

export const IMAGE_EXTS = new Set([
  ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif",
  ".heic", ".heif",
]);

// Images that need conversion to JPEG/PNG before OCR or vision-LLM ingestion.
export const HEIC_EXTS = new Set([".heic", ".heif"]);

export const ARCHIVE_CONTAINER_EXTS = new Set([".zip", ".7z"]);

export const ARCHIVE_SUPPORTED_EXTS = new Set([
  ".pdf", ".docx", ".txt", ".md", ".json", ".ts", ".tsx", ".js", ".cs",
  ".html", ".css", ".scss", ".yml", ".yaml", ".xml", ".sql", ".ps1",
  ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif",
  ".heic", ".heif",
  ".rtf", ".odt", ".pptx", ".xlsx", ".epub",
  ".csv", ".tsv", ".tex", ".rst", ".adoc", ".org",
]);

// Default extensions used by rag_index when the caller doesn't specify any.
export const DEFAULT_INDEX_EXTS = [
  ".cs", ".json", ".ts", ".tsx", ".md", ".txt", ".yml", ".yaml",
  ".pdf", ".docx", ".rtf", ".odt", ".pptx", ".xlsx", ".epub",
  ".csv", ".tsv", ".tex", ".rst", ".adoc", ".org",
  ".js", ".html", ".scss", ".css", ".sql", ".ps1", ".bicep",
  ".xml", ".config", ".csproj", ".razor",
  ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif",
  ".heic", ".heif",
  ".zip", ".7z",
  ".chm",
];
