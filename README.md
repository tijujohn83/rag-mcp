# rag-mcp

A local RAG (Retrieval-Augmented Generation) MCP server that provides semantic search over local files using on-device embeddings. No API keys, no cloud calls — everything runs locally.

## What it does

- **Indexes** a wide range of local file types into a SQLite-backed vector + full-text store
- **Searches** using hybrid vector (cosine via sqlite-vec) + BM25 keyword (FTS5) ranking fused with Reciprocal Rank Fusion (RRF)
- **Re-ranks** results with an optional cross-encoder for sharper relevance
- **Incremental** — only re-indexes files that have changed since the last run
- Exposes four tools via the [Model Context Protocol](https://modelcontextprotocol.io/): `rag_index`, `rag_search`, `rag_status`, `rag_remove`

## Supported file types

| Category | Extensions |
|---|---|
| Text / Markdown | `.txt` `.md` `.csv` `.tsv` `.tex` `.rst` `.adoc` `.org` |
| Source code | `.ts` `.tsx` `.js` `.cs` `.sql` `.ps1` `.bicep` `.yaml` `.yml` `.json` `.xml` `.html` `.css` `.scss` `.config` `.csproj` `.razor` |
| Documents | `.pdf` `.docx` `.rtf` `.odt` `.pptx` `.xlsx` `.epub` |
| Images (OCR + optional LLM caption) | `.png` `.jpg` `.jpeg` `.tiff` `.tif` `.bmp` `.webp` `.gif` |
| Archives | `.zip` `.7z` (entries indexed individually) |

PDFs with no embedded text are automatically OCR'd via Tesseract. Scanned image files are also OCR'd.
If a local vision LLM is configured and reachable, image files (and scanned PDFs) are additionally
captioned by the LLM — see [Vision LLM captioning](#vision-llm-captioning) below.

## Setup

```bash
npm install
```

Models are downloaded automatically on first use and cached in `.model-cache/`.

## Usage

Add the server to your MCP config (e.g. `.vscode/mcp.json`):

```json
{
  "servers": {
    "rag": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "tsx", "C:/path/to/rag-mcp/rag-mcp.ts"]
    }
  }
}
```

### CLI options

| Option | Env var | Default | Description |
|---|---|---|---|
| `--index <path>` | — | `rag-index.db` next to `rag-mcp.ts` | Path to the SQLite index file |
| `--embed-model <name>` | `RAG_EMBED_MODEL` | `Xenova/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `--rerank-model <name>` | `RAG_RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker model |
| `--llm-url <url>` | `RAG_LLM_URL` | `http://127.0.0.1:1234` | Base URL of the OpenAI-compatible vision LLM server (e.g. LM Studio) |
| `--llm-model <id>` | `RAG_LLM_MODEL` | `qwen3.5-9b-uncensored-hauhaucs-aggressive@q8_0` | Vision model ID to request |
| `--llm-max-tokens <n>` | `RAG_LLM_MAX_TOKENS` | `0` (omit — let the model use its full context) | Max tokens per caption response. Set > 0 to cap. |
| `--llm-image-pages <n>` | `RAG_LLM_IMAGE_PAGES` | `3` | Max scanned-PDF pages sent to the LLM per file |

## MCP Tools

### `rag_index`

Index a local directory (incremental by default).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `directory` | `string` | required | Absolute path to index |
| `extensions` | `string[]` | all supported | File extensions to include |
| `passwords` | `string[]` | — | Passwords to try on encrypted PDFs |
| `ignorePatterns` | `string[]` | — | Extra `.gitignore`-style patterns to skip |
| `ocrLang` | `string` | `"eng"` | Tesseract language(s), e.g. `"eng+dan"` |
| `recursive` | `boolean` | `true` | Recurse into subdirectories |
| `force` | `boolean` | `false` | Re-index all files even if unchanged |
| `captionImages` | `boolean` | `true` | Use the local vision LLM to caption images and enhance scanned PDFs (when the LLM is reachable) |

`.gitignore` files found inside the indexed directory are automatically respected.  
Archives (`.zip`, `.7z`) are unpacked and their supported entries indexed individually.

### `rag_search`

Semantic + keyword search over the index.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `string` | required | Natural language or code query |
| `topK` | `number` | `5` | Number of chunks to return |
| `directory` | `string` | — | Restrict to files under this path |
| `extensions` | `string[]` | — | Restrict to these extensions |
| `fileGlob` | `string` | — | Restrict to files matching a glob, e.g. `"**/payslip*.pdf"` |
| `mtimeFrom` | `number` | — | Only files modified on/after this Unix epoch ms |
| `mtimeTo` | `number` | — | Only files modified on/before this Unix epoch ms |
| `minScore` | `number` | `0.3` | Minimum cosine similarity (vector mode only) |
| `maxPerFile` | `number` | `2` | Max results from any single file |
| `mode` | `"hybrid"\|"vector"\|"lexical"` | `"hybrid"` | Search mode |
| `rerank` | `boolean` | `false` | Re-rank with cross-encoder (sharper, slower) |
| `rerankTopN` | `number` | `30` | Candidates to feed into the reranker |

**Search modes:**
- `hybrid` — combines vector cosine + BM25 via Reciprocal Rank Fusion (best for most queries)
- `vector` — semantic similarity only (good for conceptual queries)
- `lexical` — BM25 keyword only (good for exact term matching)

### `rag_status`

Show what is currently indexed: total chunks, files, index path, embed model, and top directories.

### `rag_remove`

Remove all indexed entries for a directory prefix.

| Parameter | Type | Description |
|---|---|---|
| `directory` | `string` | Absolute path prefix to remove |

## Tech stack

| Package | Purpose |
|---|---|
| `@modelcontextprotocol/sdk` | MCP server transport |
| `@huggingface/transformers` | Local embedding + reranker inference |
| `better-sqlite3` | SQLite index store |
| `sqlite-vec` | Vector similarity search (ANN via sqlite extension) |
| `pdf-parse` | PDF text + screenshot extraction |
| `mammoth` | DOCX text extraction |
| `tesseract.js` | OCR for scanned PDFs and images |
| `exceljs` | XLSX text extraction |
| `adm-zip` / `7zip-bin` | ZIP / 7z archive unpacking |
| `ignore` | `.gitignore`-style file filtering |
| `glob` | File discovery |
| `zod` | Schema validation |

## Files

- `rag-mcp.ts` — main server source
- `rag-index.db` — generated SQLite index (gitignored, rebuilt on demand)
- `.model-cache/` — cached HuggingFace model weights (gitignored, ~80 MB per model)

---

## Keywords explained

### RAG (Retrieval-Augmented Generation)
A pattern where an AI model is given relevant context retrieved from a knowledge base before generating a response. Instead of relying solely on what the model memorised during training, it can "look up" current, local, or private information at query time. This server is the retrieval half of that pipeline.

### Embeddings
Every chunk of text is converted into a list of numbers (a vector) called an **embedding**. The numbers encode the *meaning* of the text — similar meanings produce vectors that point in similar directions in high-dimensional space. This is how "puppy" and "dog" can match a query about "canines" even though none of the exact words overlap.

### all-MiniLM-L6-v2 (default embedding model)
A compact sentence-transformer model from HuggingFace (~80 MB). It reads a text chunk and outputs a 384-dimensional embedding vector. "MiniLM" means it is a distilled (smaller, faster) version of a larger transformer; "L6" means it has 6 layers; "v2" is the version. It runs entirely on-device via `@huggingface/transformers` (ONNX runtime).

### Vector search / cosine similarity
Once everything is embedded, finding relevant chunks for a query means finding the vectors geometrically closest to the query vector. **Cosine similarity** measures the angle between two vectors — 1.0 means identical direction (same meaning), 0.0 means perpendicular (unrelated). `sqlite-vec` is a SQLite extension that stores and queries these vectors efficiently using approximate nearest-neighbour search.

### BM25 / FTS5 (lexical search)
**BM25** (Best Match 25) is a classical keyword-ranking formula used by search engines. It scores a document by how often query terms appear in it, down-weighted if those terms are very common across the whole corpus. SQLite's built-in **FTS5** (Full-Text Search 5) module runs BM25 queries with a **Porter stemmer** (so "running" matches "run") and `unicode61` tokenisation. Good at exact term matching — function names, error codes, IDs.

### Reciprocal Rank Fusion (RRF)
Vector and BM25 searches each produce a ranked list with incompatible score scales. RRF ignores raw scores and fuses rankings by position: each chunk earns $\frac{1}{k + \text{rank}}$ from each ranker (k=60 is a smoothing constant). A chunk that ranks highly in both lists gets a much higher combined score than one that appears in only one. This is the "hybrid" in hybrid search.

$$\text{RRF score} = \sum_{\text{ranker}} \frac{1}{60 + \text{rank}}$$

### Cross-encoder reranking
The bi-encoder approach (embed query → embed chunks → compare vectors) is fast but approximate — the query and chunk are embedded independently. A **cross-encoder** reads the query and a candidate chunk *together* in a single pass, giving much sharper relevance scores at the cost of speed. It is used as a second stage: the top ~30 fused candidates are re-scored, then re-sorted. The default model is `ms-marco-MiniLM-L-6-v2`, trained on the MS MARCO passage-ranking dataset.

### ms-marco-MiniLM-L-6-v2 (default reranker model)
A cross-encoder fine-tuned on the Microsoft MAchine Reading COmprehension (MS MARCO) dataset — a large collection of real Bing queries paired with relevant passages. It outputs a single relevance score for a (query, passage) pair. Same MiniLM-L6 architecture as the embedder, ~80 MB, runs locally.

### Chunking
Large documents are split into overlapping **chunks** (default: 1000 characters, 150-character overlap) before embedding — most embedding models have a token-length limit, and smaller chunks produce more precise retrieval. Chunk boundaries prefer paragraph breaks, then sentence endings, then line breaks, to avoid cutting mid-thought.

### OCR (Optical Character Recognition)
Scanned PDFs and image files contain pixels, not text. **Tesseract** (via `tesseract.js`) rasterises each PDF page or reads the image and extracts the text before it is chunked and embedded. The `ocrLang` parameter selects the Tesseract language pack (e.g. `"eng+dan"` for English + Danish).

### Incremental indexing / mtime tracking
Each indexed file's modification timestamp (`mtime`) is stored in the database. On subsequent runs, a file is skipped if its `mtime` is unchanged — only new or modified files are re-processed. Use `force: true` to override and re-index everything.

---

## Vision LLM captioning

When indexing image files (and scanned PDFs whose embedded text is empty or thin), the
server can optionally call a local vision LLM via an OpenAI-compatible
`/v1/chat/completions` endpoint (LM Studio, Ollama with an OpenAI shim, llama.cpp server,
etc.) to produce a rich natural-language description of the visual content.

The caption is stored as a normal text chunk in the index, prefixed with `[LLM_CAPTION]`,
so it participates in both vector and BM25 search alongside everything else. No schema
change and no separate "image" table.

### How OCR and the LLM interact

| File type | What gets stored |
|---|---|
| PDF with embedded text (> 100 chars) | Embedded text only — fastest, most accurate. No OCR, no LLM. |
| Scanned PDF, LLM down | OCR text only. |
| Scanned PDF, LLM up | LLM transcription (given OCR as a hint, plus rendered pages). Supersedes OCR. |
| Image, LLM down | OCR text only. |
| Image, LLM up | `[LLM_CAPTION]\n<description>\n\n[OCR]\n<ocr text>` — both stored side-by-side. |
| TIFF / TIF | Always OCR only — LM Studio and most OpenAI-compatible vision APIs don't accept TIFF. |

### Self-healing across runs

The LLM is **optional**. If it isn't running when you first index, images are stored with
OCR only. On a later run when the LLM is up, only the un-captioned image and scanned-PDF
files are re-processed — normal text files stay skipped on `mtime` match. You never need
`force: true` to pick up newly available captions.

### Concurrency

Vision inference is the bottleneck. At most **2** caption requests run in parallel,
enforced by an internal semaphore, so the LLM backend isn't overwhelmed.

### Disabling per-run

Pass `captionImages: false` to `rag_index` to skip the LLM entirely for that run (useful
for fast re-indexing of text-only directories).

### Content policy

This tool is designed for individual users searching their own local corpora. The default
system prompt instructs the model to describe images literally and without sanitisation,
because a softened caption could give the user false reassurance about content that is
actually problematic. You are responsible for the data you index.
