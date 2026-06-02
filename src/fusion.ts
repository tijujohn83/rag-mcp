// Reciprocal Rank Fusion — combine multiple ranked lists into a single fused score.
// Fuses ranks (not raw scores) so the very different scales of vector-cosine and
// BM25 don't need to be normalised.
export function rrfFuse(
  rankings: Array<Array<{ chunkId: number; score: number }>>,
  k = 60,
): Map<number, number> {
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
