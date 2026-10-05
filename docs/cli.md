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
uv run python -m r_doc_builder.ingest [paths ...] [--doc-type T] [--reset] [--summarize]
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

Source types and the yaml format are described in the README.

It prints the `ingest` command for what it wrote. Preprocess one doc_type and push it:

```bash
uv run python -m r_doc_builder.pipeline --sources faq
uv run python -m r_doc_builder.ingest data/faq.pkl
```

## How a push behaves

- Chunk ids are derived from `source`, `page_url`, and content, so pushing the same file
  again updates its chunks instead of duplicating them. Only `--reset` deletes anything.
- Embeddings go out in batches of about `EMBEDDING_BATCH_TOKENS` tokens; upserts in
  batches of `PUSH_BATCH` chunks. A failed embedding call is retried `RETRIES` times with
  exponential backoff, then the run aborts.
- Metadata is filtered to scalar values (str, int, float, bool) before pushing. Nested
  values in the `.pkl` files are dropped.

## Environment

Read from `.env`; real environment variables win. The full list with comments is in
`.env.example`.

| Variable | Used by | Meaning |
| --- | --- | --- |
| `DOC_MCP_URL` | push | r-doc-mcp base URL (default `http://127.0.0.1:8000`) |
| `INGEST_TOKEN` | push | Bearer token; must match the server's |
| `PUSH_BATCH` | push | Chunks per upsert request (default 200) |
| `PUSH_TIMEOUT` | push | Seconds allowed for one ingest API request (default 300) |
| `EMBEDDING_URL`, `EMBEDDING_MODEL`, `EMBEDDING_MODEL_PROVIDER` | push | Embedding endpoint. **Must match what r-doc-mcp queries with.** |
| `EMBEDDING_BATCH_TOKENS` | push | Token budget per embedding request (default 6000) |
| `COMPLETION_URL`, `COMPLETION_MODEL`, `COMPLETION_MODEL_PROVIDER`, `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `AZURE_API_VERSION` | `--build`, `--summarize` | Completion LLM for the contextualizer and summarizer |
| `CONTEXT_CHAR_LIMIT` | `--build` | Truncation of the document context handed to the contextualizer (default 16000) |
| `SUMMARY_MIN_CHARS` | `--summarize` | Texts shorter than this are embedded as-is instead of summarized (default 300) |
| `COMPLETION_TEMPERATURE` | `--build`, `--summarize` | Sampling temperature of the contextualizer/summarizer LLM (default 0) |
| `RETRIES` | `--build`, push | Attempts per LLM or embedding call before giving up (default 5) |
| `REQUEST_TIMEOUT` | `--build` | Seconds allowed for one source download: web page, caption track (default 60) |
| `USER_AGENT` | `--build` | User-Agent sent with web page, Freshdesk and transcript downloads (default `Mozilla/5.0`) |
| `CONFIG_DIR` | `--build`, `--summarize` | Folder holding `sources.yaml`, `prompts.yaml` and `build.yaml` (default `config`; `examples/bdc` selects the BDC ones) |
| `REPOS_DIR` | `--build` | Where git sources are cloned (default `./data/repos/`) |
| `PREPROC_DATA_DIR` | `--build` | Default for `--data-dir` (default `./data/<config>/`, `<config>` = the `CONFIG_DIR` folder name, so examples don't overwrite each other) |

Chunk size and overlap, the HTML tags to drop and to read, and the caption language are not
environment variables: they shape the built corpus, so they live in `<CONFIG_DIR>/build.yaml`
with the rest of the project's setup. Rebuild with `--reset` after changing them.

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
