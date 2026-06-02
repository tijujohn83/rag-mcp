import { pipeline } from "@huggingface/transformers";
import { RERANK_MODEL } from "./config.js";

// A cross-encoder scores a (query, passage) pair jointly, giving much sharper
// relevance signals than vector cosine. We use it to re-order the top-N
// candidates produced by hybrid/vector/lexical search. Model downloads once
// to .model-cache, then runs locally.

let _reranker: Awaited<ReturnType<typeof pipeline>> | null = null;

async function getReranker() {
  if (!_reranker) {
    process.stderr.write(`[rag] loading reranker '${RERANK_MODEL}' (first run downloads ~80 MB)...\n`);
    _reranker = await pipeline("text-classification", RERANK_MODEL);
  }
  return _reranker;
}

/** Score (query, passage) pairs with a cross-encoder. Returns one score per passage. */
export async function rerankScores(query: string, passages: string[]): Promise<number[]> {
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
