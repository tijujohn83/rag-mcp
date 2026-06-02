import { pipeline } from "@huggingface/transformers";
import { EMBED_MODEL } from "./config.js";

let _embedder: any = null;

async function getEmbedder(): Promise<any> {
  if (!_embedder) {
    // Downloads once to .model-cache, then runs locally.
    _embedder = await pipeline("feature-extraction", EMBED_MODEL);
  }
  return _embedder;
}

export async function embed(text: string): Promise<Float32Array> {
  const e = await getEmbedder();
  const result = await e(text, { pooling: "mean", normalize: true });
  return new Float32Array(result.data as Float32Array);
}

export async function embedBatch(texts: string[]): Promise<Float32Array[]> {
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
