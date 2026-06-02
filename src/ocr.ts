import { createRequire } from "module";
import { openPdf } from "./pdf.js";

const require = createRequire(import.meta.url);

const { createWorker } = require("tesseract.js") as {
  createWorker: (lang: string) => Promise<{
    recognize(src: Buffer | string): Promise<{ data: { text: string } }>;
    terminate(): Promise<void>;
  }>;
};

const _ocrWorkers = new Map<string, Awaited<ReturnType<typeof createWorker>>>();

/** Returns a cached tesseract worker for the given language code (e.g. "eng", "eng+dan"). */
export async function getOcrWorker(lang: string = "eng") {
  let w = _ocrWorkers.get(lang);
  if (!w) {
    w = await createWorker(lang);
    _ocrWorkers.set(lang, w);
  }
  return w;
}

/** OCR a scanned/image-based PDF by rasterising each page via pdf-parse's getScreenshot
 *  (which uses node-canvas under the hood), then running tesseract on the PNG buffers. */
export async function ocrPdfImages(buf: Buffer, passwords: string[] = [], ocrLang: string = "eng"): Promise<string> {
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

/** OCR a standalone image file. Caller is responsible for ensuring `file` points at
 *  a tesseract-compatible format (JPEG/PNG/TIFF/BMP/etc.) — see image-prep.ts for
 *  HEIC conversion. */
export async function ocrImageFile(file: string, ocrLang: string = "eng"): Promise<string> {
  const worker = await getOcrWorker(ocrLang);
  const { data: { text } } = await worker.recognize(file);
  return text;
}
