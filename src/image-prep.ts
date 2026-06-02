import { readFile, writeFile, unlink } from "fs/promises";
import { tmpdir } from "os";
import * as path from "path";
import { createRequire } from "module";
import { HEIC_EXTS } from "./paths.js";

const require = createRequire(import.meta.url);

// heic-convert is pure JS (uses libheif compiled to wasm). Lazy-loaded so users
// who never index HEIC images don't pay the import cost.
type HeicConvertFn = (opts: {
  buffer: Buffer | ArrayBuffer | Uint8Array;
  format: "JPEG" | "PNG";
  quality?: number;
}) => Promise<ArrayBuffer | Uint8Array>;

let _heicConvert: HeicConvertFn | null = null;
function getHeicConvert(): HeicConvertFn {
  if (!_heicConvert) {
    // CJS module — default export is the function itself.
    const mod = require("heic-convert") as HeicConvertFn | { default: HeicConvertFn };
    _heicConvert = typeof mod === "function" ? mod : mod.default;
  }
  return _heicConvert;
}

/** Convert a HEIC/HEIF buffer to a JPEG buffer (quality 0.9). */
export async function heicBufferToJpeg(buf: Buffer): Promise<Buffer> {
  const out = await getHeicConvert()({ buffer: buf, format: "JPEG", quality: 0.9 });
  return Buffer.isBuffer(out) ? out : Buffer.from(out as ArrayBuffer);
}

/** Result of preparing an image for OCR / vision-LLM ingestion. */
export interface PreparedImage {
  /** Absolute path to a JPEG/PNG file usable by tesseract.js and OpenAI-vision APIs. */
  path: string;
  /** MIME type matching the prepared file. */
  mime: string;
  /** Original-or-converted buffer (lazy-loaded only if requested). */
  buffer: () => Promise<Buffer>;
  /** Releases any temporary file created during preparation. Idempotent. */
  cleanup: () => Promise<void>;
}

function mimeForExt(ext: string): string {
  switch (ext.toLowerCase()) {
    case ".jpg":
    case ".jpeg": return "image/jpeg";
    case ".png":  return "image/png";
    case ".webp": return "image/webp";
    case ".gif":  return "image/gif";
    case ".bmp":  return "image/bmp";
    case ".tif":
    case ".tiff": return "image/tiff";
    default:      return "application/octet-stream";
  }
}

/** Prepare an on-disk image file. HEIC/HEIF are converted to a temp JPEG. */
export async function prepareImageFile(file: string): Promise<PreparedImage> {
  const ext = path.extname(file).toLowerCase();
  if (!HEIC_EXTS.has(ext)) {
    return {
      path: file,
      mime: mimeForExt(ext),
      buffer: () => readFile(file),
      cleanup: async () => {},
    };
  }
  const buf = await readFile(file);
  const jpeg = await heicBufferToJpeg(buf);
  const tmpFile = path.join(tmpdir(), `rag-heic-${Date.now()}-${Math.random().toString(36).slice(2)}.jpg`);
  await writeFile(tmpFile, jpeg);
  return {
    path: tmpFile,
    mime: "image/jpeg",
    buffer: async () => jpeg,
    cleanup: () => unlink(tmpFile).catch(() => {}),
  };
}

/** Prepare an in-memory image buffer. HEIC/HEIF are converted to JPEG and
 *  written to a temp file for OCR. */
export async function prepareImageBuffer(buf: Buffer, ext: string): Promise<PreparedImage> {
  const lower = ext.toLowerCase();
  if (!HEIC_EXTS.has(lower)) {
    const tmpFile = path.join(tmpdir(), `rag-img-${Date.now()}-${Math.random().toString(36).slice(2)}${lower}`);
    await writeFile(tmpFile, buf);
    return {
      path: tmpFile,
      mime: mimeForExt(lower),
      buffer: async () => buf,
      cleanup: () => unlink(tmpFile).catch(() => {}),
    };
  }
  const jpeg = await heicBufferToJpeg(buf);
  const tmpFile = path.join(tmpdir(), `rag-heic-${Date.now()}-${Math.random().toString(36).slice(2)}.jpg`);
  await writeFile(tmpFile, jpeg);
  return {
    path: tmpFile,
    mime: "image/jpeg",
    buffer: async () => jpeg,
    cleanup: () => unlink(tmpFile).catch(() => {}),
  };
}
