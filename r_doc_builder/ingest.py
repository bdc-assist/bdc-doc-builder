import argparse
import hashlib
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
        # ponytail: summaries are made before ingest_paths asks the DB what it holds, and differ every
        # run, so --summarize records are re-summarized and re-embedded on every push
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
    with tqdm(total=len(texts), desc=desc, disable=desc is None) as pbar:
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


def _post_batches(path, rows):
    """POST rows to the ingest API in PUSH_BATCH-sized requests (embeddings are fat)."""
    for i in range(0, len(rows), PUSH_BATCH):
        _api_post(path, rows[i:i + PUSH_BATCH])


def push_chunks(ids, contents, embeddings, metas):
    """Upsert chunks to r-doc-mcp in request-sized batches."""
    _post_batches("/ingest/upsert", [{"id": cid, "content": content, "embedding": emb, "metadata": meta}
                                     for cid, content, emb, meta in zip(ids, contents, embeddings, metas)])


def _lookup(metas):
    """What the DB holds for this file's documents: {id: stored metadata}. Asked by source, then
    narrowed to the file's (source, page_url) pairs: two documents can share a source (the same
    relative path in two git repos), and one file's push must never touch the other's chunks."""
    docs = {(m.get("source"), m.get("page_url")) for m in metas}
    sources = sorted({s for s, _ in docs})
    stored = {}
    for i in range(0, len(sources), PUSH_BATCH):
        stored.update(_api_post("/ingest/lookup", {"sources": sources[i:i + PUSH_BATCH]}))
    return {cid: m for cid, m in stored.items() if (m.get("source"), m.get("page_url")) in docs}


def _plan(ids, metas, stored):
    """Diff a file's chunks against what the DB holds for its documents -> (indexes to embed,
    indexes whose metadata alone changed, stored ids the file no longer produces). A chunk is
    embedded only when its embed_hash (model + embedded text) differs from the stored one."""
    embed, update = [], []
    for i, (cid, meta) in enumerate(zip(ids, metas)):
        old = stored.get(cid)
        if old is None or old.get("embed_hash") != meta["embed_hash"]:
            embed.append(i)
        elif old != meta:
            update.append(i)
    return embed, update, sorted(stored.keys() - set(ids))


def _with_removals(meta, old):
    """Chroma merges metadata into what is stored: a key the new record dropped goes out as None,
    which deletes it."""
    return {**meta, **{k: None for k in old.keys() - meta.keys()}}


def ingest_paths(paths, doc_type: str | None = None, use_summary: bool = False) -> int:
    """Load, embed, and push all supported files under the given paths, embedding only what the DB
    doesn't hold yet: unchanged chunks are skipped, metadata-only changes updated in place, and
    chunks a pushed document no longer produces deleted. doc_type, when given, applies to every
    chunk; otherwise records keep the doc_type they carry (none for ad-hoc files).
    Returns the number of chunks embedded."""
    emb = get_emb()
    model = getattr(emb, "model", None) or ""
    total = 0
    for f in iter_files(paths):
        if f.suffix == ".pkl":
            contents, metas, embed_texts = load_pkl(f, doc_type, use_summary)
        else:
            contents, metas, embed_texts = load_file(f, doc_type)
        if not contents:
            continue
        ids = _chunk_ids(contents, metas)
        for meta, text in zip(metas, embed_texts):
            meta["embed_hash"] = hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()[:16]
        stored = _lookup(metas)
        embed, update, stale = _plan(ids, metas, stored)
        print(f"{f}: {len(embed)} to embed, {len(update)} to update, "
              f"{len(ids) - len(embed) - len(update)} unchanged, {len(stale)} to delete")
        _post_batches("/ingest/update", [{"id": ids[i], "metadata": _with_removals(metas[i], stored[ids[i]])}
                                         for i in update])
        # each slice is stored before the next is embedded: a killed run loses at most one slice,
        # and the rerun's lookup skips everything already pushed
        with tqdm(total=len(embed), desc=f"embedding {f.name}", disable=not embed) as bar:
            for start in range(0, len(embed), PUSH_BATCH):
                part = embed[start:start + PUSH_BATCH]
                vectors = _embed_batched(emb, [embed_texts[i] for i in part], desc=None)
                push_chunks([ids[i] for i in part], [contents[i] for i in part], vectors,
                            [_with_removals(metas[i], stored.get(ids[i], {})) for i in part])
                bar.update(len(part))
        _post_batches("/ingest/delete", stale)  # last: a document never loses chunks before their replacements are in
        total += len(embed)
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
    print(f"done: embedded {total} chunks; {DOC_MCP_URL} is up to date")


if __name__ == "__main__":
    main()
