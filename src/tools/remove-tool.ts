import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { normalizePath } from "../paths.js";
import { removePrefix, totalChunks } from "../db.js";
import type { RagDb } from "../types.js";

export function registerRemoveTool(server: McpServer, rdb: RagDb): void {
  server.tool(
    "rag_remove",
    "Remove all indexed entries for a directory (e.g. when you no longer need that codebase indexed).",
    {
      directory: z.string().describe("Absolute path prefix to remove from the index"),
    },
    async ({ directory }) => {
      const dirPrefix = normalizePath(directory) + "/";
      const removed = removePrefix(rdb, dirPrefix);
      const remaining = totalChunks(rdb);
      return {
        content: [{ type: "text", text: `Removed ${removed} chunks for ${directory}. Total remaining: ${remaining}` }],
      };
    }
  );
}
