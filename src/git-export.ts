import { execFile } from "child_process";
import { promisify } from "util";
import { glob } from "glob";
import * as path from "path";
import { normalizePath, hashText } from "./paths.js";

const execFileAsync = promisify(execFile);

// Discovers every git repository under a root (including nested ones such as
// submodules and worktrees) and exports the full history + branch/tag summary
// as plain text. The text is fed through the same chunking + embedding pipeline
// as regular files, stored under the virtual paths "<repoRoot>!git-log" and
// "<repoRoot>!git-branches" so all existing search/pruning logic works.

export const GIT_LOG_ENTRY = "git-log";
export const GIT_BRANCHES_ENTRY = "git-branches";
// Cap the export sizes to keep the index bounded on huge histories.
const GIT_LOG_MAX_BYTES = 32 * 1024 * 1024;        // 32 MB (one line per commit)
const GIT_BRANCHES_MAX_BYTES = 16 * 1024 * 1024;   // 16 MB
const GIT_EXEC_MAX_BUFFER = 64 * 1024 * 1024;      // 64 MB stdout buffer

let _gitAvailable: boolean | null = null;

export async function isGitAvailable(): Promise<boolean> {
  if (_gitAvailable !== null) return _gitAvailable;
  try {
    await execFileAsync("git", ["--version"], { maxBuffer: 1 * 1024 * 1024 });
    _gitAvailable = true;
  } catch {
    _gitAvailable = false;
  }
  return _gitAvailable;
}

/** Find every git repository at or under rootDir. Returns absolute repo roots. */
export async function findGitRepos(rootDir: string): Promise<string[]> {
  // Match both directory ".git" and file ".git" (worktrees / submodules).
  const hits = await glob("**/.git", {
    cwd: rootDir,
    absolute: true,
    dot: true,
    ignore: ["**/node_modules/**"],
  });
  const repos = new Set<string>();
  for (const hit of hits) {
    const repoRoot = normalizePath(path.dirname(hit));
    repos.add(repoRoot);
  }
  return [...repos].sort();
}

/** Cheap "has the repo moved since last run" signature: hash of all ref tips. */
export async function getGitTipHash(repoRoot: string): Promise<number> {
  try {
    const { stdout } = await execFileAsync(
      "git",
      ["-C", repoRoot, "for-each-ref", "--format=%(objectname) %(refname)"],
      { maxBuffer: 32 * 1024 * 1024 }
    );
    // Reduce to a safe-integer hash (first 13 hex chars of sha256 → 52 bits).
    const hex = hashText(stdout).slice(0, 13);
    return parseInt(hex, 16);
  } catch {
    return 0;
  }
}

/** Export the git log as one compact line per commit. Returns plain text.
 *  Format: "<shortSHA> <ISO author date> <author name> | <subject>"
 *  Intentionally omits: parents, committer, tree, refs, full body, --stat, patches.
 *  This keeps the index small and embedding fast while still being searchable. */
export async function exportGitLog(repoRoot: string): Promise<string> {
  const format = "%h %aI %an | %s";
  const args = [
    "-c", "i18n.logOutputEncoding=UTF-8",
    "-C", repoRoot,
    "log",
    "--all",
    "--date=iso-strict",
    "--no-color",
    `--pretty=format:${format}`,
  ];
  const { stdout } = await execFileAsync("git", args, {
    maxBuffer: GIT_EXEC_MAX_BUFFER,
    encoding: "utf-8",
  });
  if (stdout.length > GIT_LOG_MAX_BYTES) {
    return stdout.slice(0, GIT_LOG_MAX_BYTES) +
      `\n\n---TRUNCATED at ${GIT_LOG_MAX_BYTES} bytes---\n`;
  }
  return stdout;
}

/** One-line-per-ref summary of every branch and tag. Small and cheap. */
export async function exportGitBranches(repoRoot: string): Promise<string> {
  const fmt =
    "%(refname) %(objectname) %(committerdate:iso-strict) %(authorname) %(subject)";
  const { stdout } = await execFileAsync(
    "git",
    [
      "-C", repoRoot,
      "for-each-ref",
      `--format=${fmt}`,
      "refs/heads", "refs/remotes", "refs/tags",
    ],
    { maxBuffer: GIT_BRANCHES_MAX_BYTES, encoding: "utf-8" }
  );
  if (stdout.length > GIT_BRANCHES_MAX_BYTES) {
    return stdout.slice(0, GIT_BRANCHES_MAX_BYTES) + "\n---TRUNCATED---\n";
  }
  return stdout;
}
