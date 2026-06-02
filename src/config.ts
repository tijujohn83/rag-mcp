import * as path from "path";
import * as url from "url";
import { env } from "@huggingface/transformers";

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
// src/ → project root
export const ROOT_DIR = path.resolve(__dirname, "..");

// Model cache lives at <project>/.model-cache (downloads once, then offline).
env.cacheDir = path.join(ROOT_DIR, ".model-cache");

function argVal(flag: string): string | undefined {
  const i = process.argv.indexOf(flag);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : undefined;
}

// Default model: Xenova/all-MiniLM-L6-v2 (~80 MB, 384 dims). Override with
// --embed-model <name> on the command line or RAG_EMBED_MODEL env var.
const DEFAULT_EMBED_MODEL = "Xenova/all-MiniLM-L6-v2";
export const EMBED_MODEL =
  argVal("--embed-model") ?? process.env.RAG_EMBED_MODEL ?? DEFAULT_EMBED_MODEL;

const DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2";
export const RERANK_MODEL =
  argVal("--rerank-model") ?? process.env.RAG_RERANK_MODEL ?? DEFAULT_RERANK_MODEL;

export interface LlmConfig {
  url: string;
  model: string;
  maxTokens: number;
  maxPdfPages: number;
}

export const LLM_CFG: LlmConfig = {
  url: (argVal("--llm-url") ?? process.env.RAG_LLM_URL ?? "http://127.0.0.1:1234").replace(/\/+$/, ""),
  model: argVal("--llm-model") ?? process.env.RAG_LLM_MODEL ?? "qwen3.5-9b-uncensored-hauhaucs-aggressive@q8_0",
  // 0 or negative means "omit from request" — let the model use its full context budget.
  maxTokens: Number(argVal("--llm-max-tokens") ?? process.env.RAG_LLM_MAX_TOKENS ?? 0),
  maxPdfPages: Number(argVal("--llm-image-pages") ?? process.env.RAG_LLM_IMAGE_PAGES ?? 3),
};

// Concurrency cap for vision-LLM calls; matched to file-level concurrency in the indexer.
export const LLM_MAX_CONCURRENCY = 2;

// Resolve DB path. If user passed --index foo.json, switch to foo.db automatically.
export const dbPath: string = (() => {
  const indexArg = argVal("--index");
  if (indexArg) {
    return indexArg.toLowerCase().endsWith(".json")
      ? indexArg.replace(/\.json$/i, ".db")
      : indexArg;
  }
  return path.join(ROOT_DIR, "rag-index.db");
})();
