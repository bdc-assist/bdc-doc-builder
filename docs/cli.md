# CLI reference

One entry point does everything: `bdc_doc_builder.ingest`. It embeds documents and pushes
them to bdc-doc-mcp's ingest API, and with `--build` it first runs the preprocessing
pipeline over the BDC sources. `bdc_doc_builder.preproc.pipeline` is also runnable on its
own when you only want the preprocessing.

All commands run from the repo root with `uv run`.

## Before you start

- `.env` filled in (`cp .env.example .env`). See [Environment](#environment).
- bdc-doc-mcp's API running with the same `INGEST_TOKEN`:
  `uv run uvicorn bdc_doc_mcp.api:app --port 8000` in that repo.
- For `--build`: the two source repos cloned next to this one (or `BDC_WEBSITE_DIR` /
  `BDC_GITBOOK_DIR` pointing at them), plus a completion LLM configured for the
  contextualizer unless you pass `--no-contextualize`.

## `python -m bdc_doc_builder.ingest`

```text
uv run python -m bdc_doc_builder.ingest [paths ...] [--doc-type T] [--reset] [--summarize]
        [--build [--sources S ...] [--no-contextualize] [--pull] [--data-dir D]]
```

Pushes every supported file under `paths`. With `--build`, preprocesses first and pushes
what that wrote, then any `paths` given as well. At least one of `paths` or `--build` is
required.

| Option | What it does |
| --- | --- |
| `paths` | Files or directories. Directories are searched recursively for `.pkl .md .mdx .txt .pdf`. |
| `--doc-type T` | `doc_type` metadata for every chunk pushed in this run. Default: derived per file from its name (table below). |
| `--reset` | Drop the remote collection before pushing. Runs after `--build` finishes, so a failed build leaves the DB untouched. |
| `--summarize` | For `.pkl` records with no contextualized chunk (fellows, events, latest updates, videos), embed an LLM summary of the content instead of the raw content. One LLM call per record. |
| `--build` | Run the preprocessing pipeline before pushing. The four options below are passed through to it. |
| `--sources S ...` | Which sources to preprocess: `all` (default) or any of `fellows latest_updates events pages freshdesk docs vids`. |
| `--no-contextualize` | Skip the per-chunk LLM contextualizer. Much faster and cheaper; retrieval quality drops. |
| `--pull` | `git pull --ff-only` the website and gitbook repos first. Fails rather than merging if a repo has local commits. |
| `--data-dir D` | Where the `.pkl` files are written. Default `./data/` (`PREPROC_DATA_DIR`). |

### `doc_type` defaults

When `--doc-type` is not given, each file's type comes from its name. Anything else gets `docs`.

| File | `doc_type` |
| --- | --- |
| `fellows.pkl` | `fellow` |
| `latest_updates.pkl` | `update` |
| `events.pkl` | `event` |
| `pages.pkl` | `page` |
| `freshdesk.pkl` | `faq` |
| `docs.pkl` | `docs` |
| `vids.pkl` | `video` |

Don't combine `--doc-type` with `--build --sources all`: it would label every source the same.

### Examples

Rebuild the database from scratch:

```bash
uv run python -m bdc_doc_builder.ingest --build --reset
```

Same, after pulling the latest website and gitbook commits, without the LLM contextualizer:

```bash
uv run python -m bdc_doc_builder.ingest --build --pull --no-contextualize --reset
```

Refresh only the gitbook docs and the FAQ. The two sources are re-preprocessed and
upserted; everything else in the DB stays as it is:

```bash
uv run python -m bdc_doc_builder.ingest --build --sources docs freshdesk
```

Re-push the existing `data/*.pkl` without re-preprocessing, for example after switching
embedding models:

```bash
uv run python -m bdc_doc_builder.ingest data/ --reset
```

Push one file, or an ad-hoc directory of markdown with an explicit type:

```bash
uv run python -m bdc_doc_builder.ingest data/docs.pkl
uv run python -m bdc_doc_builder.ingest ../interim-bdc-website/src/pages --doc-type page
```

## `python -m bdc_doc_builder.preproc.pipeline`

```text
uv run python -m bdc_doc_builder.preproc.pipeline [--sources S ...] [--no-contextualize] [--pull] [--data-dir D]
```

Preprocessing only: writes `<data-dir>/<source>.pkl` and never talks to the database.
Useful when the MCP server isn't reachable yet, or to preprocess on one machine and push
from another. The options are the same ones `ingest --build` passes through.

| Option | What it does |
| --- | --- |
| `--sources S ...` | Which sources to preprocess: `all` (default) or any of the names in the table below. |
| `--no-contextualize` | Skip the per-chunk LLM contextualizer. Much faster and cheaper; retrieval quality drops. Only affects sources marked as contextualized below. |
| `--pull` | `git pull --ff-only` the website and gitbook repos first. Fails rather than merging if a repo has local commits. |
| `--data-dir D` | Where the `.pkl` files are written. Default `./data/` (`PREPROC_DATA_DIR`). |

### Sources

| Source | Reads | Needs | Contextualized |
| --- | --- | --- | --- |
| `fellows` | `src/data/fellows/` in the website repo | `BDC_WEBSITE_DIR` | no |
| `latest_updates` | `src/data/latest-updates/` in the website repo | `BDC_WEBSITE_DIR` | no |
| `events` | `src/data/events/` in the website repo | `BDC_WEBSITE_DIR` | no |
| `pages` | hand-picked MDX under `src/pages/` in the website repo | `BDC_WEBSITE_DIR`, LLM | yes, per section |
| `docs` | markdown in the gitbook repo, chunked by header hierarchy | `BDC_GITBOOK_DIR`, LLM | yes |
| `freshdesk` | live scrape of bdcatalyst.freshdesk.com | network, LLM | yes |
| `vids` | Google Sheet of videos plus Drive SRT transcripts and YouTube upload dates | network, LLM | yes |

"LLM" means the completion model is called unless `--no-contextualize` is given.

It prints the `ingest` command for what it wrote. Preprocess one source and push it:

```bash
uv run python -m bdc_doc_builder.preproc.pipeline --sources vids
uv run python -m bdc_doc_builder.ingest data/vids.pkl
```

## How a push behaves

- Chunk ids are derived from `source`, `page_url`, and content, so pushing the same file
  again updates its chunks instead of duplicating them. Only `--reset` deletes anything.
- Embeddings go out in batches of about `EMBEDDING_BATCH_TOKENS` tokens; upserts in
  batches of `PUSH_BATCH` chunks. A failed embedding call is retried five times with
  exponential backoff (port-forward drops, idle timeouts), then the run aborts.
- Metadata is filtered to scalar values (str, int, float, bool) before pushing. Nested
  values in the `.pkl` files are dropped.

## Environment

Read from `.env`; real environment variables win. The full list with comments is in
`.env.example`.

| Variable | Used by | Meaning |
| --- | --- | --- |
| `DOC_MCP_URL` | push | bdc-doc-mcp base URL (default `http://127.0.0.1:8000`) |
| `INGEST_TOKEN` | push | Bearer token; must match the server's |
| `PUSH_BATCH` | push | Chunks per upsert request (default 200) |
| `EMBEDDING_URL`, `EMBEDDING_MODEL`, `EMBEDDING_MODEL_PROVIDER` | push | Embedding endpoint. **Must match what bdc-doc-mcp queries with.** |
| `EMBEDDING_BATCH_TOKENS` | push | Token budget per embedding request (default 6000) |
| `COMPLETION_URL`, `COMPLETION_MODEL`, `COMPLETION_MODEL_PROVIDER`, `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, `AZURE_API_VERSION` | `--build`, `--summarize` | Completion LLM for the contextualizer and summarizer |
| `CONTEXT_CHAR_LIMIT` | `--build` | Truncation of the document context handed to the contextualizer (default 16000) |
| `BDC_WEBSITE_DIR`, `BDC_GITBOOK_DIR` | `--build` | Source repo checkouts (default `../interim-bdc-website/`, `../bdc-gitbook/`) |
| `PREPROC_DATA_DIR` | `--build` | Default for `--data-dir` (default `./data/`) |

## Troubleshooting

- **`POST .../ingest/upsert -> 401`**: `INGEST_TOKEN` differs between the two repos.
- **`embedding call failed (ConnectionError); retrying`**: the Ollama tunnel dropped. The run
  retries on its own; re-establish the port-forward if it keeps failing.
- **Dimension errors or nonsense results after switching embedding models**: vectors from
  different models don't mix. `--reset` and re-push with the new model, and change
  `EMBEDDING_*` on the server side too.
- **`--build` fails on a missing directory**: the website or gitbook repo isn't cloned, or
  the matching `BDC_*_DIR` variable points elsewhere.
