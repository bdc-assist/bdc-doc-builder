# r-doc-builder

Builds the content of the r-doc-mcp doc RAG database. Lives outside the serving security
boundary: it reads the sources listed in `sources.yaml`, chunks and contextualizes them,
embeds the chunks, then pushes the finished `{id, content, embedding, metadata}` records to
r-doc-mcp's ingest API. It never touches the database directly.

```
r_doc_builder/config.py         env-driven embeddings/LLM clients
r_doc_builder/ingest.py         the CLI: [--build] → embed → push to r-doc-mcp
r_doc_builder/pipeline.py       sources.yaml → data/<config>/<doc_type>.pkl (never touches the DB)
r_doc_builder/sources.py        the source types: path, url, git, youtube, transcripts, freshdesk
r_doc_builder/contextualize.py  LLM chunk contextualizer / summarizer
config/prompts.yaml             the contextualizer / summarizer prompt texts (examples/bdc has the BDC version)
config/sources.yaml             what to build (template; CONFIG_DIR=examples/bdc selects the BDC one)
config/build.yaml               chunk size/overlap, html tags, caption language (examples/bdc has the BDC version)
data/<config>/                  pipeline output (*.pkl), ingest input; <config> = the CONFIG_DIR folder name
                                (config, bdc, ...), so examples never overwrite each other; git clones under data/repos/
docs/cli.md                     CLI reference
```

## Setup

```bash
uv sync
cp .env.example .env    # then fill in keys/URLs
```

Models: a completion LLM for the contextualizer (any OpenAI-compatible endpoint, Azure, vLLM
or Ollama) and an embedding model. **The embedding model must match the one r-doc-mcp uses
for queries** — vectors from different models don't mix. Switching models means `--reset`
and a full re-push on this side plus the matching `EMBEDDING_*` change on the server side.

To run the three repos together from scratch, follow "From clone to chat" in r-assist's README.

## sources.yaml

A YAML list of entries with `source_type`, optional `doc_type`, and `link`. `#` comments as usual;
relative links resolve against the file's own folder.

| source_type | link | notes |
| --- | --- | --- |
| `yaml` | another list like this one | one file per category is a tidy layout; its entries inherit this entry's doc_type unless they set their own |
| `path` | local file or directory | `.md .mdx .txt .pdf`, recursive, dot-directories skipped; markdown is chunked by header; a frontmatter-only file is indexed from its fields |
| `url` | one web page | main text extracted; `.pdf`/`.md`/`.txt` URLs are read as files |
| `git` | repo URL, optional `#subdir` | shallow-cloned into `data/repos/` (with `core.longpaths`, so deep trees work on Windows); `--pull` updates it; GitHub files get a `page_url` |
| `youtube` | video, playlist or channel URL | captions via yt-dlp, chunked into timestamped windows |
| `freshdesk` | help-center category or folder URL (`…/support/solutions/<id>`, `…/support/solutions/folders/<id>`) | public Freshdesk portal scrape, no API key; one document per article with `title`, `category`, `folder` metadata |
| `transcripts` | YAML list of `{video_url, transcript}`: a local `.yaml` or a YAML URL | one document per YouTube video from its SRT transcript (a Drive link, URL or path next to the list); chunked like `youtube`, title/date from YouTube |

`doc_type` is the label r-doc-mcp filters on (declare the types in its `config/doc_types.yaml`).
Omitted, blank or `none` inherits from the including `yaml` entry; entries that never get one are
written to `untyped.pkl` and pushed without a `doc_type`. An entry whose link doesn't fit its type
is warned and skipped; the build only fails when nothing at all was built.

`examples/bdc/sources.yaml` lists the NHLBI BioData Catalyst sources: the GitBook content repo
(`docs`) and the website repo's pages, news, events and fellows folders (`page`, `update`, `event`,
`fellow`), the Freshdesk FAQ category (`faq`), and the video transcripts sheet reduced to
`examples/bdc/videos.yaml` (`video`). Every original BDC source now has a generic source type. The
example ingests every markdown file of those folders; the original builder skipped GitBook's
`SUMMARY.md` and a few oversized pages, which the list cannot express.

`examples/fastapi/sources.yaml` builds the FastAPI docs from the English markdown in
github.com/fastapi/fastapi: the tutorial, advanced, how-to and deployment folders (`git` with
`#subdir`), the top-level pages through an included `pages.yaml` (`url` entries for raw `.md`
files, inheriting `page`), and the release notes (`release`). Its code samples are not indexed:
the pages pull them in with `{* docs_src/... *}` lines, which stay as text.

The contextualizer and summarizer wording lives in `prompts.yaml` (the copy in `CONFIG_DIR`); edit it freely but keep the `{placeholders}`.
How sources become chunks (chunk size and overlap, which HTML tags to drop and read, caption
language) is in `build.yaml` next to it; rebuild with `--reset` after changing it.

## Build & push

`r_doc_builder.ingest` is the CLI. r-doc-mcp's API must be running
(`uv run uvicorn r_doc_mcp.api:app --port 8000` over there) with the same `INGEST_TOKEN` set.

List your sources in `config/sources.yaml` (every entry in the template is commented out), then build and
push from scratch:

```bash
uv run python -m r_doc_builder.ingest --build --reset
```

Build with `pipeline`, push with `ingest` (inspect the `.pkl` files in between). Git clones under
`data/repos/` are skipped; only the `.pkl` files (and any ad-hoc files you drop in `data/`) are
pushed. Builds land in `data/<config>/`, named after the `CONFIG_DIR` folder (`data/config/` for
the template, `data/bdc/` for `examples/bdc`); push that folder, not all of `data/`.

```bash
uv run python -m r_doc_builder.pipeline
uv run python -m r_doc_builder.ingest data/config/ --reset
```

Refresh one doc_type, leaving the rest of the DB as it is:

```bash
uv run python -m r_doc_builder.ingest --build --sources faq
```

`--no-contextualize` skips the per-chunk LLM call (much faster, weaker retrieval). Every flag
and environment variable is in [docs/cli.md](docs/cli.md).

## Tests

```bash
uv run python tests/test_ingest.py     # batching, chunk ids, push batching — no network
uv run python tests/test_pipeline.py   # yaml walk, chunkers, source types (stubbed) — no network
uv run python tests/test_e2e.py        # incremental pushes vs a fresh push, real r-doc-mcp from ../bdc-doc-mcp — no network, ~2 min
```
