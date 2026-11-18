import { createRequire } from "module";
import { openPdf } from "./pdf.js";
import { ROOT_DIR } from "./config.js";

const require = createRequire(import.meta.url);

interface OcrWorker {
  recognize(src: Buffer | string): Promise<{ data: { text: string } }>;
  terminate(): Promise<void>;
}

const { createWorker } = require("tesseract.js") as {
  createWorker: (lang: string, oem?: number, options?: Record<string, unknown>) => Promise<OcrWorker>;
};

const _ocrWorkers = new Map<string, Promise<OcrWorker>>();

/** Returns a cached tesseract worker for the given language code (e.g. "eng", "eng+dan"). */
export async function getOcrWorker(lang: string = "eng"): Promise<OcrWorker> {
  let w = _ocrWorkers.get(lang);
  if (!w) {
    w = new Promise<OcrWorker>((resolve, reject) => {
      createWorker(lang, undefined, {
        // Traineddata ships with the repo. Without absolute paths tesseract resolves
        // them against the cwd, misses, and downloads from a CDN.
        langPath: ROOT_DIR,
        cachePath: ROOT_DIR,
        // Tesseract throws from a worker message listener when this is unset, which
        // Node rethrows on nextTick and kills the process past any await.
        // tesseract.js swallows load failures, so without this reject createWorker never settles.
        errorHandler: (err: unknown) => {
          process.stderr.write(`[rag] OCR worker error: ${String(err)}\n`);
          reject(err instanceof Error ? err : new Error(String(err)));
        },
      }).then(resolve, reject);
    }).catch((err: unknown) => {
      _ocrWorkers.delete(lang);
      throw err;
    });
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
