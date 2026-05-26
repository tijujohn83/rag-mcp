# rag-mcp

A local RAG (Retrieval-Augmented Generation) MCP server that provides semantic search over local files using on-device embeddings. No API keys, no cloud calls — everything runs locally.

## What it does

- Indexes local files (`.txt`, `.md`, `.pdf`, `.docx`, and more) into a vector store
- Generates embeddings using [all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) (~80 MB, downloaded once)
- Exposes a `search` tool via the [Model Context Protocol](https://modelcontextprotocol.io/) so Claude and other MCP clients can query your local knowledge base

## Setup

```bash
npm install
```

The embedding model is downloaded automatically on first use and cached in `.model-cache/`.

## Usage with Claude Code

Add the server to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "rag": {
      "command": "npx",
      "args": ["tsx", "/path/to/rag-mcp.ts"],
      "env": {}
    }
  }
}
```

Then ask Claude to index a folder:

```
index /path/to/your/docs
```

And search it:

```
search for authentication flow
```

## Tech stack

| Package | Purpose |
|---|---|
| `@modelcontextprotocol/sdk` | MCP server transport |
| `@huggingface/transformers` | Local embedding inference |
| `pdf-parse` | PDF text extraction |
| `mammoth` | DOCX text extraction |
| `glob` | File discovery |
| `zod` | Schema validation |

## Files

- `rag-mcp.ts` — main server source
- `rag-index.json` — generated vector index (gitignored, rebuilt on demand)
- `.model-cache/` — cached HuggingFace model weights (gitignored)
