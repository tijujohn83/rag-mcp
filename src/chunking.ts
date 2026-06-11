// Minimum informative chunk length (after stripping page markers + whitespace).
// Chunks below this are tail-end noise from overlap windows or scanned-PDF artifacts
// like "ness 2\n\n-- 8 of 12 --" and add no retrievable signal.
const MIN_INFORMATIVE_LEN = 30;

/** Strip PDF "-- N of M --" page markers and collapse whitespace.
 *  Used only to judge whether a chunk carries enough signal to keep. */
export function informativeLength(text: string): number {
  return text
    .replace(/--\s*\d+\s+of\s+\d+\s*--/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .length;
}

/** Returns true if a chunk is too short / marker-only to be useful for retrieval. */
export function isLowInformation(text: string): boolean {
  return informativeLength(text) < MIN_INFORMATIVE_LEN;
}

/** Detect JSON-Lines: most non-empty lines parse as standalone JSON objects/arrays.
 *  Samples the first ~20 non-empty lines to stay cheap on large files. */
function looksLikeJsonl(text: string): boolean {
  const lines: string[] = [];
  for (const raw of text.split("\n")) {
    const t = raw.trim();
    if (!t) continue;
    lines.push(t);
    if (lines.length >= 20) break;
  }
  if (lines.length < 2) return false;
  let ok = 0;
  for (const l of lines) {
    if ((l.startsWith("{") && l.endsWith("}")) || (l.startsWith("[") && l.endsWith("]"))) {
      try { JSON.parse(l); ok++; } catch { /* not jsonl */ }
    }
  }
  return ok / lines.length >= 0.7;
}

/** Chunk JSON-Lines text by grouping whole lines up to ~maxSize chars.
 *  Each chunk contains one or more complete JSON records — never a partial one. */
function chunkJsonl(text: string, maxSize: number): { text: string; start: number }[] {
  const chunks: { text: string; start: number }[] = [];
  const lines = text.split("\n");
  let buf: string[] = [];
  let bufLen = 0;
  let chunkStart = 0;
  let cursor = 0;

  const flush = () => {
    if (buf.length === 0) return;
    const chunkText = buf.join("\n");
    if (!isLowInformation(chunkText)) chunks.push({ text: chunkText, start: chunkStart });
    buf = [];
    bufLen = 0;
  };

  for (const line of lines) {
    const lineLen = line.length + 1; // +1 for newline
    if (buf.length === 0) chunkStart = cursor;
    // A single line longer than maxSize is kept whole; we don't split JSON records.
    if (bufLen > 0 && bufLen + lineLen > maxSize) {
      flush();
      chunkStart = cursor;
    }
    buf.push(line);
    bufLen += lineLen;
    cursor += lineLen;
  }
  flush();
  return chunks;
}

/** Sentence + paragraph-aware text chunker.
 *  Produces overlapping windows of roughly `maxSize` characters, preferring
 *  paragraph boundaries, then sentence ends, then line breaks, then hard cuts.
 *  Auto-switches to line-aware chunking for JSON-Lines input so each record
 *  stays whole. Drops trailing low-information fragments. */
export function chunkText(text: string, maxSize = 1000, overlap = 150): { text: string; start: number }[] {
  if (looksLikeJsonl(text)) return chunkJsonl(text, maxSize);

  const chunks: { text: string; start: number }[] = [];
  let windowStart = 0;

  while (windowStart < text.length) {
    const windowEnd = windowStart + maxSize;

    if (windowEnd >= text.length) {
      const chunk = text.slice(windowStart);
      if (chunk.trim() && !isLowInformation(chunk)) chunks.push({ text: chunk, start: windowStart });
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
    if (chunk.trim() && !isLowInformation(chunk)) chunks.push({ text: chunk, start: windowStart });
    // Advance with overlap; Math.max guards against infinite loop on no-boundary edge cases
    windowStart = Math.max(windowStart + 1, breakAt - overlap);
  }

  return chunks;
}
