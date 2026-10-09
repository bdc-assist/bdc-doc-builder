# CLI reference

One entry point does everything: `r_doc_builder.ingest`. It embeds documents and pushes
them to r-doc-mcp's ingest API, and with `--build` it first runs the preprocessing
pipeline over the sources listed in `sources.yaml`. `r_doc_builder.pipeline` is also runnable
on its own when you only want the preprocessing.

All commands run from the repo root with `uv run`.

## Before you start

- `.env` filled in (`cp .env.example .env`). See [Environment](#environment).
- r-doc-mcp's API running with the same `INGEST_TOKEN`:
  `uv run uvicorn r_doc_mcp.api:app --port 8000` in that repo.
- For `--build`: the sources listed in `sources.yaml` (git sources are cloned automatically
  into `REPOS_DIR`), plus a completion LLM configured for the contextualizer unless you pass
  `--no-contextualize`.

## `python -m r_doc_builder.ingest`

```text
uv run python -m r_doc_builder.ingest [paths ...] [--doc-type T] [--reset] [--summarize] [--dry-run]
        [--build [--yaml F] [--sources S ...] [--no-contextualize] [--pull] [--data-dir D]]
```

Pushes every supported file under `paths`. With `--build`, preprocesses first and pushes
what that wrote, then any `paths` given as well. At least one of `paths` or `--build` is
required.

| Option | What it does |
| --- | --- |
| `paths` | Files or directories. Directories are searched recursively for `.pkl .md .mdx .txt .pdf`. |
| `--doc-type T` | `doc_type` metadata for every chunk pushed in this run. Default: whatever each record carries. |
| `--reset` | Drop the remote collection before pushing. Runs after `--build` finishes, so a failed build leaves the DB untouched. |
| `--summarize` | For `.pkl` records with no contextualized chunk, embed an LLM summary of the content instead of the raw content. One LLM call per record. |
| `--dry-run` | Report what the push would embed, update and delete (and which sources would lose chunks), then stop: nothing embedded, nothing written. Not with `--reset`. With `--build` the pipeline still runs and writes its `.pkl` files; with `--summarize` the summary LLM calls are still made. |
| `--build` | Run the preprocessing pipeline before pushing. The options below are passed through to it. |
| `--yaml F` | Root source list. Default `<CONFIG_DIR>/sources.yaml`. |
| `--sources S ...` | Which doc_types to build; `untyped` = rows without one. Default all. All yaml files are walked first, so a doc_type declared in an included file is selectable. |
| `--no-contextualize` | Skip the per-chunk LLM contextualizer. Much faster and cheaper; retrieval quality drops. |
| `--pull` | `git pull --ff-only` existing clones of git sources. Without it, clones are reused as they are; missing clones are always cloned. |
| `--data-dir D` | Where the `.pkl` files are written. Default `./data/<config>/`, the `CONFIG_DIR` folder name (`PREPROC_DATA_DIR`). |

Without `--doc-type`, every chunk keeps the doc_type its record carries (set by the
`doc_type` of `sources.yaml` entries); ad-hoc files and `untyped.pkl` records are pushed
without one.

Don't combine `--doc-type` with `--build --sources all`: it would label every source the same.

### Examples

See what a push would change before doing it:

```bash
uv run python -m r_doc_builder.ingest data/bdc/ --dry-run
```

Rebuild the database from scratch:

```bash
uv run python -m r_doc_builder.ingest --build --reset
```

Same, after pulling the latest commits into any existing git clones, without the LLM
contextualizer:

```bash
uv run python -m r_doc_builder.ingest --build --pull --no-contextualize --reset
```

Refresh only the `docs` and `faq` doc_types. The two are re-preprocessed and upserted;
everything else in the DB stays as it is:

```bash
uv run python -m r_doc_builder.ingest --build --sources docs faq
```

Re-push the existing `data/<config>/*.pkl` without re-preprocessing, for example after switching
embedding models. Git clones under `data/repos/` are skipped; only the `.pkl` files (and any
ad-hoc files you drop in that folder) are pushed:

```bash
uv run python -m r_doc_builder.ingest data/config/ --reset   # data/bdc/ with CONFIG_DIR=examples/bdc
```

Push one file, or an ad-hoc directory of markdown with an explicit type:

```bash
uv run python -m r_doc_builder.ingest data/docs.pkl
uv run python -m r_doc_builder.ingest ./some/pages --doc-type page
```

## `python -m r_doc_builder.pipeline`

```text
uv run python -m r_doc_builder.pipeline [--yaml F] [--sources S ...] [--no-contextualize] [--pull] [--data-dir D]
```

Preprocessing only: writes `<data-dir>/<doc_type>.pkl` and never talks to the database.
Useful when the MCP server isn't reachable yet, or to preprocess on one machine and push
from another. The options are the same ones `ingest --build` passes through.

| Option | What it does |
| --- | --- |
| `--yaml F` | Root source list. Default `<CONFIG_DIR>/sources.yaml`. |
| `--sources S ...` | Which doc_types to build; `untyped` = rows without one. Default all. All yaml files are walked first, so a doc_type declared in an included file is selectable. |
| `--no-contextualize` | Skip the per-chunk LLM contextualizer. Much faster and cheaper; retrieval quality drops. |
| `--pull` | `git pull --ff-only` existing clones of git sources. Without it, clones are reused as they are; missing clones are always cloned. |
| `--data-dir D` | Where the `.pkl` files are written. Default `./data/<config>/`, the `CONFIG_DIR` folder name (`PREPROC_DATA_DIR`). |

Contextualizing reuses the previous build: a chunk whose text and document are unchanged since
the last build into the same `--data-dir` keeps its context sentence (any doc_type's `.pkl`
counts), so only new or edited documents call the LLM — and unchanged chunks keep the same
embedded text, so the push skips them. Changing the `contextualize_chunk` prompt or
`COMPLETION_MODEL` re-contextualizes everything; so does building into an empty `--data-dir`. A chunk whose
last contextualize call failed (raw text kept) is retried on the next build.

Source types and the yaml format are described in the README.

It prints the `ingest` command for what it wrote. Preprocess one doc_type and push it:

```bash
uv run python -m r_doc_builder.pipeline --sources faq
uv run python -m r_doc_builder.ingest data/faq.pkl
```

## How a push behaves

- Chunk ids are derived from `source`, `page_url`, and content, so pushing the same file
  again updates its chunks instead of duplicating them.
- Before embedding anything, ingest asks r-doc-mcp what it holds for the file's documents and
  compares chunk by chunk. Every chunk carries `embed_hash`, a fingerprint of the embedding model
  and the exact text embedded:
  - same `embed_hash` and metadata as stored: skipped;
  - same `embed_hash`, other metadata (title, date, doc_type, `--doc-type`): metadata updated in
    place, no embedding call;
  - not stored, or another `embed_hash` (edited text, a new context sentence, another embedding
    model): embedded and upserted.

  Each file prints `N to embed, N to update, N unchanged, N to delete`.
- Chunks a pushed document no longer produces are deleted, after its new chunks are stored.
  Documents that are not part of the push are never touched; a document removed from its source
  entirely stays in the DB until `--reset`: run a `--build --reset` now and then.
- A document (same `source` and `page_url`) present in two pushed files is refused before anything
  is written: it is usually a stale `.pkl` left by a renamed or emptied doc_type. Delete it.
- Chunks are embedded (about `EMBEDDING_BATCH_TOKENS` tokens per request) and upserted
  `PUSH_BATCH` at a time, so a run that dies loses at most one batch: rerun the same command
  **without `--reset`** and it carries on where it stopped.
- A failed embedding call or a dropped connection to r-doc-mcp is retried `RETRIES` times with
  exponential backoff, then the run aborts. HTTP errors from r-doc-mcp (bad token, server error)
  abort at once.
- Metadata is filtered to scalar values (str, int, float, bool) before pushing. Nested
  values in the `.pkl` files are dropped. A key a record no longer has is removed from the
  stored chunk.
- The first push after upgrading to this version re-embeds everything once: chunks stored by
  older versions carry no `embed_hash`. The first `--build` also re-contextualizes everything
  (old `.pkl` files have no `context_hash`), so build once with the new version and push those files.
- After pushing with an older version of this tool, push with `--reset`: an older push keeps the
  stored `embed_hash` while replacing the vector, so a later push could skip a chunk it should
  re-embed.

## Environment

Read from `.env`; real environment variables win. The full list with comments is in
`.env.example`.

| Variable | Used by | Meaning |
| --- | --- | --- |
| `DOC_MCP_URL` | push | r-doc-mcp base URL (default `http://127.0.0.1:8000`) |
| `INGEST_TOKEN` | push | Bearer token; must match the server's |
| `PUSH_BATCH` | push | Chunks per upsert request, the resume slice size, and the batch size of lookup/update/delete requests (default 200) |
| `PUSH_TIMEOUT` | push | Seconds allowed for one ingest API request (default 300) |
| `EMBEDDING_URL`, `EMBEDDING_MODEL`, `EMBEDDING_MODEL_PROVIDER` | push | Embedding endpoint. **Must match what r-doc-mcp queries with.** |
| `EMBEDDING_BATCH_TOKENS` | push | Token budget per embedding request (default 6000) |
| `COMPLETION_URL`, `COMPLETION_MODEL`, `COMPLETION_MODEL_PROVIDER`, `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `AZURE_API_VERSION` | `--build`, `--summarize` | Completion LLM for the contextualizer and summarizer |
| `CONTEXT_CHAR_LIMIT` | `--build` | Truncation of the document context handed to the contextualizer (default 16000) |
| `SUMMARY_MIN_CHARS` | `--summarize` | Texts shorter than this are embedded as-is instead of summarized (default 300) |
| `COMPLETION_TEMPERATURE` | `--build`, `--summarize` | Sampling temperature of the contextualizer/summarizer LLM (default 0) |
| `RETRIES` | `--build`, push | Attempts per LLM, embedding or ingest API call before giving up (default 5) |
| `REQUEST_TIMEOUT` | `--build` | Seconds allowed for one source download: web page, caption track (default 60) |
| `USER_AGENT` | `--build` | User-Agent sent with web page, Freshdesk and transcript downloads (default `Mozilla/5.0`) |
| `CONFIG_DIR` | `--build`, `--summarize` | Folder holding `sources.yaml`, `prompts.yaml` and `build.yaml` (default `config`; `examples/bdc` selects the BDC ones) |
| `REPOS_DIR` | `--build` | Where git sources are cloned (default `./data/repos/`) |
| `PREPROC_DATA_DIR` | `--build` | Default for `--data-dir` (default `./data/<config>/`, `<config>` = the `CONFIG_DIR` folder name, so examples don't overwrite each other) |

Chunk size and overlap, the HTML tags to drop and to read, and the caption language are not
environment variables: they shape the built corpus, so they live in `<CONFIG_DIR>/build.yaml`
with the rest of the project's setup. Rebuild with `--reset` after changing them.

## Rehearsal and rollout

Pushes change the database in place. Before the first push with a new version of this tool to a
database that matters, rehearse on a copy. The commands use the BDC config and run in Git Bash;
r-doc-mcp commands run in that repo, everything else in this one, each repo with its own `.env`
for the embedding endpoint.

0. **Build once with this version**, because old `.pkl` files carry no `context_hash`; the rehearsal
   and the production push both use these files:

   ```bash
   CONFIG_DIR=examples/bdc uv run python -m r_doc_builder.pipeline --pull
   ```

1. **Copy the database.** Stop writes to it first (scale the r-doc-mcp deployment to zero, or make
   sure no push is running), then copy its `DB_PATH` (e.g. `kubectl cp` from the pod) to
   `./before-db` and again to `./rehearsal-db`. `./before-db` is never pushed to. Serve both, each
   in its own terminal (r-doc-mcp repo):

   ```bash
   DB_PATH=./before-db    CONFIG_DIR=examples/bdc INGEST_TOKEN=rehearse uv run uvicorn r_doc_mcp.api:app --port 8099
   DB_PATH=./rehearsal-db CONFIG_DIR=examples/bdc INGEST_TOKEN=rehearse uv run uvicorn r_doc_mcp.api:app --port 8100
   ```

2. **Dry run**: on the first run after upgrading, expect every chunk "to embed" (no `embed_hash`
   stored yet) and nothing to delete. That premise holds when `data/bdc/` is what the DB was last
   pushed from; a DB pushed by older versions without `--reset` holds orphaned chunks, which show up
   as deletes (and the count drops). That is expected: review them in the dry run.

   ```bash
   DOC_MCP_URL=http://127.0.0.1:8100 INGEST_TOKEN=rehearse CONFIG_DIR=examples/bdc uv run python -m r_doc_builder.ingest data/bdc/ --dry-run
   ```

3. **Push, then push again** (same command without `--dry-run`, twice). The second run must print
   `0 to embed, 0 to update, ... 0 to delete` for every file, and `curl http://127.0.0.1:8100/health`
   must show the same `documents` count as `curl http://127.0.0.1:8099/health`.
4. **Same answers**: save this once as `answers.py` (in this repo):

   ```python
   import sys
   import requests
   base = sys.argv[1]
   questions = [("What is PIC-SURE and what can I do with it in BDC?", {}),
                ("Whats the difference between picsure open access and authorized access?", {"mode": "keyword"}),
                ("What are the latest BDC events, and are any more coming up?", {"date_from": "2025-01-01"}),
                ("How do I upload my own data to BDC?", {})]
   for q, extra in questions:
       hits = requests.post(f"{base}/search", json={"query": q, **extra}).json()
       print(q, [h["metadata"].get("source") for h in hits], sep="
  ")
   ```

   Run it against each server into a file and compare:

   ```bash
   uv run python answers.py http://127.0.0.1:8099 > before.txt
   uv run python answers.py http://127.0.0.1:8100 > after.txt
   diff before.txt after.txt
   ```

   No output means the same answers: nothing changed in the content, so nothing should change in the results.
5. **Compare with a fresh push**: delete `./oracle-db` if it exists, serve the empty `./oracle-db`
   on port 8101 the same way, push the same `.pkl` files to it, stop the 8100 and 8101 servers,
   then (r-doc-mcp repo):

   ```bash
   uv run python tests/compare_db.py ./rehearsal-db ./oracle-db --collection bdc --atol 1e-4
   ```

   Expect `same`. (`--atol`: a real embedding endpoint can differ in the last float digits.)
6. **Kill a run**: copy `data/bdc/` to `./rehearsal-data` and do everything below with
   `--data-dir ./rehearsal-data` (build and push), so no test edit reaches production. Restart the
   8100 server, edit a few pages in a source clone, rebuild
   (`--build --sources <type> --data-dir ./rehearsal-data`), start the push of `./rehearsal-data` and stop
   it part-way (Ctrl-C, or drop the port-forward), rerun it, then repeat step 5 with a new, empty
   `./oracle-db` and every `.pkl` file of `./rehearsal-data`, so the fresh push holds the edits too.
   End by undoing the clone edits: `git -C data/repos/<clone> checkout .`.

Then production:

1. **Back up** the database directory with writes stopped (or snapshot its volume). That backup is
   the rollback.
2. Deploy r-doc-mcp first; `/health` must show the same document count as before.
3. Run `answers.py` against production once before pushing, into a file: those are the "before"
   answers (`uv run python answers.py <prod url> > before.txt`).
4. Run the push with `--dry-run` against production and read the counts and the sources that would
   lose chunks. The push command, with the build from step 0 already in `data/bdc/`:

   ```bash
   DOC_MCP_URL=<prod url> INGEST_TOKEN=<token> CONFIG_DIR=examples/bdc uv run python -m r_doc_builder.ingest data/bdc/ --dry-run
   ```

5. Push: the same command without `--dry-run`. Check `/health`, run `answers.py` again into
   `after.txt` and `diff before.txt after.txt`: the answers should match, except where the pushed
   content really changed.
6. Keep the backup until the next refresh has succeeded. To roll back, restore the directory.
   The previous ingest version still works against the new r-doc-mcp (its endpoints are additive).

## Troubleshooting

- **`POST .../ingest/upsert -> 401`**: `INGEST_TOKEN` differs between the two repos.
- **`embedding call failed (ConnectionError); retrying`**: the embedding endpoint dropped
  the connection. The run retries on its own; check the endpoint if it keeps failing.
- **Dimension errors or nonsense results after switching embedding models**: vectors from
  different models don't mix. `--reset` and re-push with the new model, and change
  `EMBEDDING_*` on the server side too.
- **`--build` prints `warning: ...` lines**: those rows were skipped (bad link for the type,
  missing path, no captions). Fix the row or ignore; the build only aborts when nothing was
  built.
- **A push died part-way** (Ctrl-C, dropped tunnel, crash): rerun the same command without
  `--reset`. Chunks already stored are skipped; only the rest is embedded.
- **`ingest API call failed (ConnectionError); retrying`**: the connection to r-doc-mcp dropped
  (often the port-forward). The run retries on its own.
