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
| Images (OCR) | `.png` `.jpg` `.jpeg` `.tiff` `.tif` `.bmp` `.webp` |
| Archives | `.zip` `.7z` (entries indexed individually) |

PDFs with no embedded text are automatically OCR'd via Tesseract. Scanned image files are also OCR'd.

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
