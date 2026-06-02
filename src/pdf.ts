import { createRequire } from "module";

const require = createRequire(import.meta.url);

export interface PDFImage { dataUrl?: string; width: number; height: number; kind: number; }
export interface PDFImageResult { pages: { pageNumber: number; images: PDFImage[] }[]; total: number; }
export interface PDFScreenshotPage { pageNumber: number; width: number; height: number; data: Uint8Array; dataUrl?: string; }
export interface PDFScreenshotResult { pages: PDFScreenshotPage[]; total: number; }

export const { PDFParse } = require("pdf-parse") as {
  PDFParse: new (opts: { data: Buffer; password?: string }) => {
    load(): Promise<void>;
    getText(): Promise<{ text: string }>;
    getImage(opts?: Record<string, unknown>): Promise<PDFImageResult>;
    getScreenshot(opts?: { scale?: number; imageBuffer?: boolean; imageDataUrl?: boolean }): Promise<PDFScreenshotResult>;
  };
};

export type PDFParser = InstanceType<typeof PDFParse>;

/** Open a PDF, trying each password if the document is encrypted. Returns the loaded parser. */
export async function openPdf(buf: Buffer, passwords: string[] = []): Promise<PDFParser> {
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
