import { readFileSync } from "fs";
import { glob } from "glob";
import * as path from "path";
import ignore, { Ignore } from "ignore";

export interface IgnoreFilter { base: string; ig: Ignore; }

export async function buildIgnoreFilters(rootDir: string, userPatterns: string[] = []): Promise<IgnoreFilter[]> {
  const filters: IgnoreFilter[] = [];

  // User-supplied patterns apply across the entire rootDir.
  if (userPatterns.length > 0) {
    const ig = ignore();
    ig.add(userPatterns);
    filters.push({ base: rootDir, ig });
  }

  // .gitignore files inside the indexed directory (nested repos / subfolders).
  // Ancestor .gitignore files are intentionally excluded: a workspace-level
  // .gitignore may exclude entire sub-folders (e.g. /be/ /fe/) that the user
  // explicitly wants to index, so we only respect rules within rootDir itself.
  const nested = await glob("**/.gitignore", { cwd: rootDir, absolute: true, dot: true });

  for (const gip of nested) {
    try {
      const ig = ignore();
      ig.add(readFileSync(gip, "utf-8"));
      filters.push({ base: path.dirname(gip), ig });
    } catch { /* unreadable — skip */ }
  }

  return filters;
}

export function isGitignored(file: string, filters: IgnoreFilter[]): boolean {
  return filters.some(({ base, ig }) => {
    // Use path.sep suffix to avoid false prefix matches (e.g. "foo" matching "foo-bar")
    if (!file.startsWith(base + path.sep) && file !== base) return false;
    const rel = path.relative(base, file).replace(/\\/g, "/");
    if (rel.startsWith("../")) return false;
    return ig.ignores(rel);
  });
}
