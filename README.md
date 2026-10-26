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

A YAML list of entries with `source_type`, optional `doc_type`, `link`, and optional `exclude`. `#`
comments as usual; relative links resolve against the file's own folder. `exclude` is a glob or a
list of globs matched against each document's `source` (for a `path` or `git` directory, the file's
path relative to the link; `*` also crosses `/`); matching documents are dropped, so one folder can be
split between doc_types.

| source_type | link | notes |
| --- | --- | --- |
| `yaml` | another list like this one | one file per category is a tidy layout; its entries inherit this entry's doc_type unless they set their own |
| `path` | local file or directory | `.md .mdx .txt .pdf`, recursive, dot-directories skipped; markdown is chunked by header; a frontmatter-only file is indexed from its fields; a frontmatter `url` (http/https) becomes the `page_url` |
| `url` | one web page | main text extracted; `.pdf`/`.md`/`.txt` URLs are read as files |
| `git` | repo URL, optional `#subdir` or `#path/to/file` | shallow-cloned into `data/repos/` (with `core.longpaths`, so deep trees work on Windows); `--pull` updates it; GitHub files get their GitHub page as `page_url` unless their frontmatter has a `url` |
| `youtube` | video, playlist or channel URL | captions via yt-dlp, chunked into timestamped windows |
| `freshdesk` | help-center category or folder URL (`…/support/solutions/<id>`, `…/support/solutions/folders/<id>`) | public Freshdesk portal scrape, no API key; one document per article with `title`, `category`, `folder` metadata |
| `transcripts` | YAML list of `{video_url, transcript}`: a local `.yaml` or a YAML URL | one document per YouTube video from its SRT transcript (a Drive link, URL or path next to the list); chunked like `youtube`, title/date from YouTube |

`doc_type` is the label r-doc-mcp filters on (declare the types in its `config/doc_types.yaml`).
Omitted, blank or `none` inherits from the including `yaml` entry; entries that never get one are
written to `untyped.pkl` and pushed without a `doc_type`. Each doc_type builds all or nothing: an
entry that fails to load or loads no documents is warned and its whole doc_type skipped (nothing of
it contextualized, written or pushed; the DB keeps its last good build, and r-doc-mcp's `/health`
its older `last_ingest`), while the other doc_types build and push. The run ends naming the skipped
doc_types; fix their entries and rebuild just those with `--build --sources <doc_type> ...`. `--reset`
is refused when a doc_type was skipped, as it would leave that doc_type empty. A malformed entry
(no `link`, a missing `yaml` include) stops the build before anything runs.

`examples/bdc/sources.yaml` lists the NHLBI BioData Catalyst sources. Docs and website come from
github.com/stagecc/bdc-llm, the Markdown export of the BDC GitBook and website (private: the clone
uses your git credentials): `docs/` (`docs`); `site/` (`page`) minus what the other rows take, which
are `site/news/latest-updates` (`update`), `site/news/events` (`event`), `site/about/bdc/fellows.md`
(`fellow`) and `site/help/faqs.md` (`faq`). Every bdc-llm file's frontmatter `url` is its live page,
so citations link to the website and GitBook rather than the private repo; events and updates get
their `date` from frontmatter too. Videos come from the transcripts sheet reduced to
`examples/bdc/videos.yaml` (`video`).

`examples/fastapi/sources.yaml` builds the FastAPI docs from the English markdown in
github.com/fastapi/fastapi: the tutorial, advanced, how-to and deployment folders (`git` with
`#subdir`), the top-level pages through an included `pages.yaml` (`url` entries for raw `.md`
files, inheriting `page`), and the release notes (`release`). Its code samples are not indexed:
the pages pull them in with `{* docs_src/... *}` lines, which stay as text.

`examples/fixture/` is a fake, BDC-shaped corpus for tests and rehearsals: invented text in all seven
BDC doc_types with their real metadata shapes, built offline (markdown, local transcripts). Its
`extras.yaml` holds ready-made rows for the shapes only networked sources produce; the tests write them
to `extras.pkl` at run time. r-doc-mcp's `examples/fixture` serves it in its own `fixture` collection.

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
uv run python tests/test_pipeline.py   # yaml walk, chunkers, source types (stubbed), the fixture corpus — no network
uv run python tests/test_fixture_cli.py  # the real CLI on the fixture corpus vs real r-doc-mcp from ../bdc-doc-mcp — offline, ~15 s
uv run python tests/test_e2e.py        # incremental pushes vs a fresh push, real r-doc-mcp from ../bdc-doc-mcp — no network, ~80 s
```

`test_fixture_cli.py` doubles as a safe rehearsal: it builds, pushes, edits and re-pushes the fake corpus
into a throwaway DB with fake embeddings. `FIXTURE_EMBEDDINGS=real` uses this repo's `.env` embedding
settings instead, to try a real endpoint without touching real data.
