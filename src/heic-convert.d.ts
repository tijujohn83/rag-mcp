// Type shim for heic-convert (no published @types/heic-convert).
declare module "heic-convert" {
  interface HeicConvertOpts {
    buffer: Buffer | ArrayBuffer | Uint8Array;
    format: "JPEG" | "PNG";
    quality?: number;
  }
  const heicConvert: (opts: HeicConvertOpts) => Promise<ArrayBuffer | Uint8Array>;
  export default heicConvert;
}
