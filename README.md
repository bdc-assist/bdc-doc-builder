# BDC Doc Builder

Builds the content of the [bdc-doc-mcp](https://github.com/bdc-assist/bdc-doc-mcp) doc RAG database.
Lives outside the serving security boundary: it scrapes/parses the BDC sources, contextualizes and
embeds the chunks, then pushes the finished `{id, content, embedding, metadata}` records to
bdc-doc-mcp's ingest API. It never touches the database directly.

```
bdc_doc_builder/config.py   env-driven embeddings/LLM clients
bdc_doc_builder/ingest.py   .pkl/.md/.mdx/.txt/.pdf → embeddings → push to bdc-doc-mcp
bdc_doc_builder/preproc/    source-specific preprocessing pipeline
data/                       preproc output (*.pkl), push input
```

## Setup

```bash
uv sync
cp .env.example .env    # then fill in keys/URLs
```

Source repos (only needed for preprocessing `--sources all`; pushing existing `.pkl` files
works without them). Clone next to this repo or point the env vars at them:

```bash
git clone https://github.com/stagecc/interim-bdc-website ../interim-bdc-website   # BDC_WEBSITE_DIR
git clone https://github.com/stagecc/bdc-gitbook ../bdc-gitbook                   # BDC_GITBOOK_DIR
```

Models: completion via the OpenAI API on Azure (`gpt-4o-mini` by default); embeddings via
Ollama on Sterling (RENCI VPN — `kubectl -n ner port-forward svc/ollama 11434:11434`) or a
local Ollama with `groonga/bge-m3-Q4_K_M-GGUF`.

**The embedding model must match the one bdc-doc-mcp uses for queries** — vectors from
different models don't mix (`bge-m3` is 1024-dim, `text-embedding-3-small` 1536). Switching
models means `--reset` and a full re-push on this side plus the matching `EMBEDDING_*`
change on the server side.

## Build & push

bdc-doc-mcp's API must be running (`uv run uvicorn bdc_doc_mcp.api:app --port 8000` over
there) with the same `INGEST_TOKEN` set.

Full rebuild from every source (needs the two source repos cloned; writes `data/*.pkl`,
then embeds and pushes them):

```bash
uv run python -m bdc_doc_builder.preproc.pipeline --sources all --ingest --reset
```

Re-push existing `.pkl` files without re-preprocessing:

```bash
uv run python -m bdc_doc_builder.preproc.pipeline --ingest-only --reset
```

Individual files or directories:

```bash
uv run python -m bdc_doc_builder.ingest ./data/docs.pkl --doc-type docs
uv run python -m bdc_doc_builder.ingest ../interim-bdc-website/src/pages --doc-type page --reset
```

`--no-contextualize` skips the per-chunk LLM call (much faster, weaker retrieval).

## Preprocessing

`bdc_doc_builder/preproc/` is the BDC_Chatbot pipeline, ported:

| Module | Source | Ported from (BDC_Chatbot) | Notes |
|---|---|---|---|
| `bdc_repo.py` | interim-bdc-website MDX | `utils/preproc/proc_BDC_repo.py` (verbatim-ish) | fellows, events, latest-updates, pages |
| `bdc_docs.py` | bdc-gitbook markdown | `utils/preproc/proc_BDC_docs.py` (module-level LLM init removed) | chunked by header hierarchy; needs the repo cloned |
| `freshdesk.py` | bdcatalyst.freshdesk.com | `utils/preproc/proc_freshdesk.py` | live scrape |
| `vids.py` | Google Sheet + Drive SRT | `utils/preproc/proc_BDC_vids.py` (GoogleSheetsReader class flattened) | video transcripts with timestamp URLs |
| `utils.py` | — | — | LLM chunk contextualizer + summarizer |
| `pipeline.py` | — | `utils/preproc_doc.py` | orchestrator |

## Tests

```bash
uv run python tests/test_ingest.py   # batching, chunk ids, push batching — no network
```
