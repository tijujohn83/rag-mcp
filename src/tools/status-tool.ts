import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { dbPath, EMBED_MODEL } from "../config.js";
import { totalChunks, totalFiles, totalTrackedFiles } from "../db.js";
import type { RagDb } from "../types.js";

export function registerStatusTool(server: McpServer, rdb: RagDb): void {
  server.tool(
    "rag_status",
    "Show what directories and files are currently indexed.",
    {},
    async () => {
      const chunkCount = totalChunks(rdb);
      if (chunkCount === 0) {
        return { content: [{ type: "text", text: "Index is empty. Run rag_index to add files." }] };
      }

      const dirRows = rdb.db.prepare("SELECT DISTINCT file FROM chunks").all() as { file: string }[];

      const dirCounts = new Map<string, number>();
      for (const { file } of dirRows) {
        const parts = file.replace(/\\/g, "/").split("/");
        const dir = parts.slice(0, -1).join("/");
        dirCounts.set(dir, (dirCounts.get(dir) ?? 0) + 1);
      }

      const lines = [
        `Total chunks: ${chunkCount}`,
        `Total files: ${totalFiles(rdb)}`,
        `Tracked files (mtime): ${totalTrackedFiles(rdb)}`,
        `Index file: ${dbPath}`,
        `Embed model (current): ${EMBED_MODEL}`,
        `Embed model (index):   ${rdb.embedModel}`,
        `Embed dim:             ${rdb.embedDim}`,
        "",
        "Top directories by file count:",
        ...[...dirCounts.entries()]
          .sort((a, b) => b[1] - a[1])
          .slice(0, 20)
          .map(([d, n]) => `  ${d}  (${n} files)`),
      ];

      return { content: [{ type: "text", text: lines.join("\n") }] };
    }
  );
}
