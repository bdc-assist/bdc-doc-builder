import argparse
import pickle
import time
import uuid
from pathlib import Path

import requests
from tqdm import tqdm

from . import pipeline
from . import sources
from .config import (DOC_MCP_URL, EMBEDDING_BATCH_TOKENS, INGEST_TOKEN, PUSH_BATCH, PUSH_TIMEOUT, REPOS_DIR,
                     RETRIES, get_emb)

SUPPORTED = {".pkl", *sources.SUPPORTED}


def _scalar_meta(meta: dict) -> dict:
    # vector DBs only accept scalar metadata values
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


def load_pkl(path: Path, doc_type: str | None = None, use_summary: bool = False):
    """pipeline.py output: list of {'content': str, 'metadata': dict}. Embeds
    metadata['contextualized_chunk'] / ['text_to_embed'] when present, else an LLM summary
    if use_summary, else the content. doc_type, when given, overrides whatever the record carries."""
    with open(path, "rb") as f:
        rows = pickle.load(f)
    contents, metas, embed_texts = [], [], []
    for row in tqdm(rows, desc=f"preparing {path.name}"):
        meta = dict(row["metadata"])
        if doc_type:
            meta["doc_type"] = doc_type
        meta.setdefault("source", str(path.name))
        embed_text = meta.get("contextualized_chunk") or meta.get("text_to_embed")
        if not embed_text and use_summary:
            from .contextualize import get_summary
            embed_text = meta["summary"] = get_summary(row["content"])
        contents.append(row["content"])
        embed_texts.append(embed_text or row["content"])
        metas.append(_scalar_meta(meta))
    return contents, metas, embed_texts


def load_file(path: Path, doc_type: str | None = None):
    """An ad-hoc .md/.mdx/.txt/.pdf given on the command line: same readers as a `path`
    source row, no contextualizing."""
    doc = sources.load_file(path)
    if not doc:
        return [], [], []
    records = pipeline.to_records(doc, doc_type, contextualize=False)  # same metadata merge + date_num as pipeline output
    contents = [r["content"] for r in records]
    return contents, [_scalar_meta(r["metadata"]) for r in records], list(contents)


def iter_files(paths):
    repos = Path(REPOS_DIR).resolve()  # git clones live here; never push them as ad-hoc files
    for p in paths:
        p = Path(p)
        if p.is_dir():
            yield from (f for f in sorted(p.rglob("*"))
                        if f.suffix.lower() in SUPPORTED and repos not in f.resolve().parents)
        elif p.suffix.lower() in SUPPORTED:
            yield p
        else:
            print(f"skipping unsupported: {p}")


def _chunk_ids(contents, metas):
    """Content-derived ids so re-pushing a file updates chunks instead of duplicating.
    Repeated text within one file (identical boilerplate sections) gets an occurrence
    suffix — deterministic, and the server rejects duplicate ids in a single upsert."""
    ids, seen = [], {}
    for content, meta in zip(contents, metas):
        key = f"{meta.get('source')}::{meta.get('page_url', '')}::{content}"
        count = seen.get(key, 0)
        seen[key] = count + 1
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, key if not count else f"{key}::{count}")))
    return ids


def _embed_with_retry(emb, batch, attempts=RETRIES):
    """kubectl port-forward drops under normal conditions (VPN blips, idle timeouts,
    apiserver proxy restarts). Same backoff _invoke_llm uses. Unlike _invoke_llm
    there's no fallback for a missing embedding, so re-raise once attempts are
    exhausted."""
    for attempt in range(attempts):
        try:
            return emb.embed_documents(batch)
        except Exception as e:
            if attempt == attempts - 1:
                raise
            print(f"  embedding call failed ({type(e).__name__}); retrying in {2 ** attempt}s")
            time.sleep(2 ** attempt)


def _embed_batched(emb, texts, desc="embedding"):
    """Embed in request-sized batches. Endpoints cap tokens per request (8k on some
    gateways); a whole file at once blows that. Budget is estimated at ~4 chars/token."""
    budget = EMBEDDING_BATCH_TOKENS * 4
    vectors, batch, batch_chars = [], [], 0
    with tqdm(total=len(texts), desc=desc) as pbar:
        for text in texts:
            text = text[:budget]  # a single oversized chunk still has to fit one request
            if batch and batch_chars + len(text) > budget:
                vectors.extend(_embed_with_retry(emb, batch))
                pbar.update(len(batch))
                batch, batch_chars = [], 0
            batch.append(text)
            batch_chars += len(text)
        if batch:
            vectors.extend(_embed_with_retry(emb, batch))
            pbar.update(len(batch))
    if len(vectors) != len(texts):  # push_chunks zips them: a short reply would drop chunks silently
        raise RuntimeError(f"embedding returned {len(vectors)} vectors for {len(texts)} texts")
    return vectors


# --- push client: r-doc-mcp owns the DB; we only talk to its ingest API ---

def _api_post(path, payload):
    """kubectl port-forward drops (see _embed_with_retry) hit the ingest API too: retry connection
    errors and timeouts with the same backoff. Every ingest endpoint is idempotent, so a retried
    request is safe. An HTTP error (bad token, server bug) is an answer, not a dropped tunnel:
    raise it at once."""
    url = DOC_MCP_URL + path
    for attempt in range(RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=PUSH_TIMEOUT,
                                 headers={"Authorization": f"Bearer {INGEST_TOKEN}"})
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == RETRIES - 1:
                raise
            print(f"  ingest API call failed ({type(e).__name__}); retrying in {2 ** attempt}s")
            time.sleep(2 ** attempt)
    if not resp.ok:
        raise RuntimeError(f"POST {url} -> {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def reset_remote():
    """Drop the server-side collection so the next push starts empty."""
    print("resetting remote collection")
    _api_post("/ingest/reset", None)


def push_chunks(ids, contents, embeddings, metas, desc="pushing"):
    """Upsert chunks to r-doc-mcp in request-sized batches (embeddings are fat)."""
    batch = PUSH_BATCH
    for i in tqdm(range(0, len(ids), batch), desc=desc):
        _api_post("/ingest/upsert", [
            {"id": cid, "content": content, "embedding": emb, "metadata": meta}
            for cid, content, emb, meta in zip(ids[i:i + batch], contents[i:i + batch],
                                              embeddings[i:i + batch], metas[i:i + batch])
        ])


def ingest_paths(paths, doc_type: str | None = None, use_summary: bool = False) -> int:
    """Load, embed, and push all supported files under the given paths.
    doc_type, when given, applies to every chunk; otherwise records keep the doc_type they
    carry (none for ad-hoc files)."""
    emb = get_emb()
    total = 0
    for f in iter_files(paths):
        if f.suffix == ".pkl":
            contents, metas, embed_texts = load_pkl(f, doc_type, use_summary)
        else:
            contents, metas, embed_texts = load_file(f, doc_type)
        if not contents:
            continue
        embeddings = _embed_batched(emb, embed_texts, desc=f"embedding {f.name}")
        ids = _chunk_ids(contents, metas)
        push_chunks(ids, contents, embeddings, metas, desc=f"pushing {f.name}")
        total += len(contents)
        print(f"pushed {len(contents):4d} chunks from {f}")
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the r-doc-mcp database: embed documents and push them to its ingest API. "
                    "Push existing files, or --build to preprocess the sources in sources.yaml first. See docs/cli.md.")
    parser.add_argument("paths", nargs="*", help="files or directories to push (.pkl .md .mdx .txt .pdf)")
    parser.add_argument("--doc-type",
                        help="metadata doc_type for every chunk pushed in this run (default: whatever each record carries)")
    parser.add_argument("--reset", action="store_true", help="drop the remote collection before pushing")
    parser.add_argument("--summarize", action="store_true",
                        help="embed an LLM summary for .pkl records lacking a contextualized chunk")
    build = parser.add_argument_group(
        "build from source",
        "--build runs pipeline first and pushes what it wrote; the other options here pass through to it")
    build.add_argument("--build", action="store_true", help="preprocess sources.yaml before pushing")
    pipeline.add_args(build)
    args = parser.parse_args(argv)
    if not args.paths and not args.build:
        parser.error("nothing to do: give paths to push, or --build to preprocess the sources first")

    built = pipeline.build(args) if args.build else []  # before --reset: a failed build must not empty the DB
    if args.reset:
        reset_remote()
    total = ingest_paths(built + args.paths, args.doc_type, args.summarize)
    print(f"done: pushed {total} chunks to {DOC_MCP_URL}")


if __name__ == "__main__":
    main()
