import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { openDb } from "./src/db.js";
import { registerIndexTool } from "./src/tools/index-tool.js";
import { registerSearchTool } from "./src/tools/search-tool.js";
import { registerStatusTool } from "./src/tools/status-tool.js";
import { registerRemoveTool } from "./src/tools/remove-tool.js";

const rdb = openDb();
const server = new McpServer({ name: "rag", version: "1.0.0" });
registerIndexTool(server, rdb);
registerSearchTool(server, rdb);
registerStatusTool(server, rdb);
registerRemoveTool(server, rdb);
const transport = new StdioServerTransport();
await server.connect(transport);
