// Standalone validation of the git export commands used by rag-mcp.ts.
// Runs each command the way exportGitLog / exportGitBranches / getGitTipHash do.
const { execFile } = require("child_process");
const { promisify } = require("util");
const { createHash } = require("crypto");
const path = require("path");
const { glob } = require("glob");
const execFileAsync = promisify(execFile);

const GIT_LOG_FORMAT =
  "commit %H%n" +
  "parents %P%n" +
  "refs %D%n" +
  "author %an <%ae>%n" +
  "author-date %aI%n" +
  "committer %cn <%ce>%n" +
  "commit-date %cI%n" +
  "tree %T%n%n" +
  "%B%n---END---";

const BRANCH_FORMAT =
  "%(refname) %(objectname) %(committerdate:iso-strict) %(authorname) %(subject)";

const normalize = p => path.resolve(p).replace(/\\/g, "/");

async function findGitRepos(rootDir) {
  const hits = await glob("**/.git", {
    cwd: rootDir, absolute: true, dot: true,
    ignore: ["**/node_modules/**"],
  });
  return [...new Set(hits.map(h => normalize(path.dirname(h))))].sort();
}

async function tipHash(repo) {
  const { stdout } = await execFileAsync(
    "git", ["-C", repo, "for-each-ref", "--format=%(objectname) %(refname)"],
    { maxBuffer: 32 * 1024 * 1024 }
  );
  const hex = createHash("sha256").update(stdout).digest("hex").slice(0, 13);
  return parseInt(hex, 16);
}

async function exportLog(repo) {
  const { stdout } = await execFileAsync(
    "git",
    [
      "-c", "i18n.logOutputEncoding=UTF-8",
      "-C", repo, "log", "--all", "--decorate=full",
      "--date=iso-strict", "--stat", "--no-color",
      `--pretty=format:${GIT_LOG_FORMAT}`,
    ],
    { maxBuffer: 512 * 1024 * 1024, encoding: "utf-8" }
  );
  return stdout;
}

async function exportBranches(repo) {
  const { stdout } = await execFileAsync(
    "git",
    ["-C", repo, "for-each-ref", `--format=${BRANCH_FORMAT}`,
     "refs/heads", "refs/remotes", "refs/tags"],
    { maxBuffer: 16 * 1024 * 1024, encoding: "utf-8" }
  );
  return stdout;
}

async function main() {
  const root = process.argv[2] || "c:/dev/rag-mcp";
  console.log("Root:", normalize(root));

  // git available?
  const v = await execFileAsync("git", ["--version"]);
  console.log("git:", v.stdout.trim());

  const repos = await findGitRepos(root);
  console.log(`Repos found: ${repos.length}`);
  for (const r of repos.slice(0, 5)) console.log("  -", r);
  if (repos.length === 0) {
    console.error("No repos found — aborting");
    process.exit(1);
  }

  for (const repo of repos.slice(0, 3)) {
    console.log("\n=================");
    console.log("Repo:", repo);
    const h1 = await tipHash(repo);
    const h2 = await tipHash(repo);
    console.log("tipHash:", h1, "stable:", h1 === h2 ? "yes" : "NO");

    const log = await exportLog(repo);
    const records = log.split("---END---").filter(s => s.trim()).length;
    console.log(`log bytes: ${log.length}, commit records: ${records}`);
    // Sanity-check structure
    const looksRight = /^commit [0-9a-f]{40}\nparents/.test(log) &&
                       /\nauthor-date \d{4}-\d{2}-\d{2}T/.test(log) &&
                       /\ncommit-date \d{4}-\d{2}-\d{2}T/.test(log);
    console.log("format sanity:", looksRight ? "OK" : "BAD");
    console.log("--- first record ---");
    console.log(log.split("---END---")[0].slice(0, 800));

    const br = await exportBranches(repo);
    const brLines = br.split("\n").filter(Boolean).length;
    console.log(`branches/refs: ${brLines} lines`);
    console.log("--- first 5 ref lines ---");
    console.log(br.split("\n").slice(0, 5).join("\n"));
  }
}

main().catch(e => { console.error("FAILED:", e); process.exit(1); });
