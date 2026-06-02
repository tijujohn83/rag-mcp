# Plan: Index Git History During `rag_index`

## Goal

When `rag_index` scans a directory, for every git repository found at the root
or in any subdirectory (recursively), export the full git history and feed it
into the same indexing pipeline so it becomes searchable via `rag_search`.

Non-git directories are unaffected.

## High-Level Behaviour

1. While walking `directory`, discover every git repository:
   - A directory is a git repo if it contains a `.git` folder **or** a `.git`
     file (worktrees / submodules).
   - Nested repos count too (e.g. `/dev` contains `be/abat-api/.git`,
     `fe/simcorp-portal/.git`, etc.).
2. For each repo, export the full log from the beginning of time, across all
   branches, including diffs/stats as configured.
3. Write the export(s) into the index as **virtual files** keyed by the repo
   root, so they live alongside normal files and reuse all existing chunking,
   embedding, dedup, search, and pruning logic.
4. Make it incremental: only re-export when the repo's tip(s) changed since
   the last run.
5. Make it controllable: opt-in/out via a tool parameter; sensible defaults.

## Detection

Add a helper:

```ts
async function findGitRepos(root: string): Promise<string[]>
```

- Walk with `glob("**/.git", { cwd: root, absolute: true, dot: true, ignore: ["**/node_modules/**"] })`.
- Accept both directory `.git` and file `.git` (worktree pointer).
- Return repo roots (parent of `.git`).
- De-dup; sort by depth so parents are processed before children.
- Add `**/.git/**` is already in the file-scan ignore list — keep it.

## Export Strategy

We export **two** virtual files per repo so the LLM can answer questions
about commits, authors, parents, refs, branches, tags, and merges.

### 1. Commit log — `<repoRoot>!git-log`

One `git log` invocation with a custom pretty format that captures every
field we want as searchable text:

```
git -C <repoRoot> \
    -c i18n.logOutputEncoding=UTF-8 \
    log \
    --all \
    --decorate=full \
    --date=iso-strict \
    --stat \
    --no-color \
    --pretty=format:'commit %H%nparents %P%nrefs %D%nauthor %an <%ae>%nauthor-date %aI%ncommitter %cn <%ce>%ncommit-date %cI%ntree %T%n%n%B%n---END---'
```

Produces one self-describing record per commit, e.g.:

```
commit 9ab12...
parents 7cd44... 5ef00...
refs HEAD -> refs/heads/main, refs/remotes/origin/main, tag: v1.2.0
author Tomas <tjoh@simcorp.com>
author-date 2025-11-04T10:22:13+01:00
committer GitHub <noreply@github.com>
commit-date 2025-11-04T10:22:13+01:00
tree abc123...

Merge pull request #287579 from feature/foo

Fixes AB#123456
Co-authored-by: Alice <alice@...>

 src/foo.ts | 12 ++++++++----
---END---
```

Why each field:

| Field | Enables answering |
|---|---|
| `commit %H` | "what does commit `<sha>` do" |
| `parents %P` | Merge detection (≥2 parents), ancestry questions |
| `refs %D` (`--decorate=full`) | Which branches/tags point at this commit |
| `author` / `committer` | "commits by X" |
| `author-date` / `commit-date` (ISO) | Time-bounded questions |
| `%B` (full body) | Trailers like `Signed-off-by`, `Co-authored-by`, `Closes #...`, `AB#...` |
| `--stat` | "which files changed" |
| `---END---` sentinel | Stable chunk boundary marker |

**`--all`** covers every local + remote branch and tag refs but skips stashes.

**Full diffs (`-p`) are deliberately excluded** — they balloon the index and
are rarely needed for the question set. Add later behind an opt-in flag if
needed.

### 2. Branch / tag summary — `<repoRoot>!git-branches`

A tiny companion file so "list all branches", "what's the tip of branch X",
"what tags exist" are directly answerable without scanning the whole log:

```
git -C <repoRoot> for-each-ref \
    --format='%(refname) %(objectname) %(committerdate:iso-strict) %(authorname) %(subject)' \
    refs/heads refs/remotes refs/tags
```

One line per ref. Cheap, small, refreshed on the same incremental schedule
as the log.

### What this does NOT solve

- **Parent branch a feature branch was cut from** — git genuinely does not
  store this. Heuristics (walk merge-bases of every branch tip pair) are out
  of scope for v1. Users can still infer it from merge-commit subjects
  ("Merge pull request #X from feature/foo") which **are** indexed.
- **Structured filtering** (e.g. SQL-style `WHERE author='Alice'`) — the
  values are present as text and findable by hybrid/lexical search, but not
  exposed as columns.

### Execution

Run both commands via `execFile` (`execFileAsync` already imported):
- Stream stdout to a temp file under `os.tmpdir()` to avoid heap blow-ups
  on large repos, then read the file back as UTF-8.
- Cap at ~200 MB; warn and truncate if exceeded.

## Virtual-Path Convention

Reuse the existing archive-style virtual path so all existing storage,
pruning, mtime, and search filters work unmodified. Two entries per repo:

```
<repoRoot>!git-log
<repoRoot>!git-branches
```

Where `ARCHIVE_ENTRY_SEP` is `!`. Examples:

- `c:/dev/be/abat-api!git-log`
- `c:/dev/be/abat-api!git-branches`
- `c:/dev/fe/simcorp-portal!git-log`

Benefits:

- `rag_search` with `directory: "c:/dev/be/abat-api"` naturally matches
  the virtual file (the path starts with the dir prefix).
- `listFilesUnderPrefix` / stale-file pruning continue to work.
- No new table needed.

## Incremental Re-Export

Avoid re-exporting an unchanged repo on every run.

- Compute a "tip hash" cheaply:
  `git -C <repoRoot> rev-list --all --max-count=1` (latest commit on any ref)
  or, more thorough: hash of `git for-each-ref --format='%(objectname) %(refname)'`.
- Store that hash as the `mtime` field for the virtual path (cast string→number
  via a stable hash, OR add a new column). **Simpler:** repurpose `mtime` to be
  a numeric hash of the tip-list string (`hashText(...)` → take first 13 hex
  chars → parse as int).
- Skip export when the stored value matches and `force !== true`.

## Chunking & Storage

No changes needed — the existing `chunkText` + `ingestUnit` flow handles
plain text. The `---END---` sentinel between commits provides a stable
boundary the existing chunker will tend to break on.

(Future refinement: a git-log-aware splitter that guarantees one commit per
chunk when the commit fits within the chunk size budget, so metadata never
gets separated from its body.)

## Tool Surface

Add one optional parameter to `rag_index`:

```ts
indexGitHistory: z.boolean().optional()
  .describe("Also index full git log for every git repo found under the directory. Default: true")
```

Default **true** per user request. Allow opt-out for users who don't want it.

(Optional future: `gitLogArgs: string[]` to override the export command.)

## Implementation Steps

1. **Helper: `findGitRepos(rootDir)`**
   - New function near `buildIgnoreFilters`. Returns absolute repo roots.

2. **Helper: `getGitTipHash(repoRoot)`**
   - Runs `git for-each-ref --format='%(objectname)'`, concatenates, hashes
     via existing `hashText`, returns a number (parseInt of first 13 hex
     chars — fits in JS safe int).

3. **Helper: `exportGitLog(repoRoot)`**
   - Streams the custom-format `git log` (see Export Strategy) stdout to a
     temp file, then reads it back as UTF-8.
   - Returns the text.
   - Handles errors (not a repo, git not installed, shallow clone) by
     throwing — caller logs and continues.

4. **Helper: `exportGitBranches(repoRoot)`**
   - Runs `git for-each-ref` with the format from Export Strategy.
   - Returns the (small) text.

5. **Integrate into `rag_index` handler**, after the existing file scan and
   stale-file cleanup, before the per-file worker loop:

   ```ts
   if (indexGitHistory) {
     const repos = await findGitRepos(directory);
     for (const repo of repos) {
       const logPath      = normalizePath(repo) + ARCHIVE_ENTRY_SEP + "git-log";
       const branchesPath = normalizePath(repo) + ARCHIVE_ENTRY_SEP + "git-branches";
       const tipHash = await getGitTipHash(repo);
       const upToDate =
         getFileMtime(rdb, logPath) === tipHash &&
         getFileMtime(rdb, branchesPath) === tipHash;
       if (!force && upToDate) { reused += 2; continue; }
       try {
         const relRepo = path.relative(directory, repo).replace(/\\/g, "/") || ".";
         const logText = await exportGitLog(repo);
         await ingestUnit(logPath, tipHash, logText, `${relRepo}!git-log`,
                          ++gitI, repos.length * 2, Date.now());
         const branchesText = await exportGitBranches(repo);
         await ingestUnit(branchesPath, tipHash, branchesText, `${relRepo}!git-branches`,
                          ++gitI, repos.length * 2, Date.now());
         indexed += 2;
       } catch (err) {
         errors.push({ file: logPath, error: String(err) });
         skipped++;
       }
     }
   }
   ```

6. **Stale-repo cleanup**
   - Extend the existing stale-file loop: any tracked path ending in
     `!git-log` or `!git-branches` whose repo root no longer exists (or no
     longer contains `.git`) gets pruned.

7. **Update tool description**
   - Mention git-history + branch indexing in the `rag_index` description.

8. **README**
   - Add a short section: "Git history indexing — what fields are captured,
     how to disable, how it's stored, how incremental works, what kinds of
     questions it answers."

## Edge Cases

| Case | Handling |
|---|---|
| `git` not on PATH | Catch ENOENT, log once, set a session flag to skip further attempts. |
| Bare repo (`.git` is a file with `gitdir: ...`) | `git -C <parent> log` still works — no special code needed. |
| Submodules | Each submodule's `.git` file is detected by the glob and processed as a separate repo. Acceptable. |
| Worktrees | Same — treated as separate repos. |
| Huge repo (gigabytes of log) | Stream to temp file; cap at e.g. 200 MB and warn if exceeded. |
| Shallow clone | Log still works, returns what's available. No special handling. |
| Encoding | Force `-c i18n.logOutputEncoding=UTF-8`; read temp file as UTF-8. |
| User edits then re-runs | Tip hash changes → re-export. Old chunks removed by `removeFileChunks` inside `ingestUnit`. |

## Question Coverage Matrix

| Question type | Covered by v1? |
|---|---|
| "What does commit `<sha>` do" | ✅ via `commit %H` + body |
| "Commits authored by X" | ✅ text match on `author` line |
| "Commits mentioning AB#123" / PR number | ✅ in body |
| "Files changed in commit X" | ✅ via `--stat` |
| "Trailers (Co-authored-by, Signed-off-by, Closes …)" | ✅ in body |
| "Which branches/tags point at commit X" | ✅ via `refs %D` |
| "Is commit X a merge" | ✅ via `parents %P` (≥2 SHAs) |
| "When was branch X merged into main" | ✅ via merge commit's subject + commit-date |
| "List all branches / tags" | ✅ via `!git-branches` |
| "Parent branch a feature was cut from" | ⚠️ only if mentioned in merge subject |
| Structured `WHERE author=… AND date>…` filtering | ❌ — text search only |

## Out of Scope (for v1)

- Per-commit chunking with dedicated metadata columns (author, date, sha)
  for structured filtering.
- Indexing patch contents (`-p`).
- Stash / notes / reflog.
- Heuristic parent-branch reconstruction via merge-base walks.
- Updating `rag_search` to surface commit metadata in result formatting.

These can be added later without breaking the v1 storage layout.

## Test / Verification

1. Run `rag_index` on `c:/dev/rag-mcp` — should detect 1 repo, produce one
   `!git-log` virtual file.
2. Run again — should report `reused` (no re-export).
3. Make a dummy commit — re-run, should re-export.
4. Run on `c:/dev` — should discover dozens of repos under `be/` and `fe/`.
5. `rag_search` for a known commit message — should return a chunk from the
   matching `!git-log` file.
6. `rag_search` with `directory: "c:/dev/be/abat-api"` — git log of that
   repo only is in scope.
