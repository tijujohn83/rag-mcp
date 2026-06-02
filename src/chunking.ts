/** Sentence + paragraph-aware text chunker.
 *  Produces overlapping windows of roughly `maxSize` characters, preferring
 *  paragraph boundaries, then sentence ends, then line breaks, then hard cuts. */
export function chunkText(text: string, maxSize = 1000, overlap = 150): { text: string; start: number }[] {
  const chunks: { text: string; start: number }[] = [];
  let windowStart = 0;

  while (windowStart < text.length) {
    const windowEnd = windowStart + maxSize;

    if (windowEnd >= text.length) {
      const chunk = text.slice(windowStart);
      if (chunk.trim()) chunks.push({ text: chunk, start: windowStart });
      break;
    }

    // Prefer break order: paragraph (\n\n) → sentence end (.?! followed by space) → line (\n) → hard cut
    let breakAt = text.lastIndexOf("\n\n", windowEnd);
    if (breakAt <= windowStart) {
      const slice = text.slice(windowStart, windowEnd);
      // Sentence-ending punctuation followed by whitespace and a capital/quote start
      const re = /[.!?]["')\]]?\s+(?=["'(\[]?[A-ZÀ-ÖØ-Þ])/g;
      let lastIdx = -1;
      let lastLen = 0;
      let m: RegExpExecArray | null;
      while ((m = re.exec(slice)) !== null) {
        lastIdx = m.index;
        lastLen = m[0].length;
      }
      if (lastIdx >= 0) breakAt = windowStart + lastIdx + lastLen;
    }
    if (breakAt <= windowStart) breakAt = text.lastIndexOf("\n", windowEnd);
    if (breakAt <= windowStart) breakAt = windowEnd;

    const chunk = text.slice(windowStart, breakAt);
    if (chunk.trim()) chunks.push({ text: chunk, start: windowStart });
    // Advance with overlap; Math.max guards against infinite loop on no-boundary edge cases
    windowStart = Math.max(windowStart + 1, breakAt - overlap);
  }

  return chunks;
}
