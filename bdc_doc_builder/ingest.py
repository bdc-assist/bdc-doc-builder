import argparse
import os
import pickle
import re
import time
import uuid
from pathlib import Path

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .config import get_emb

SUPPORTED = {".pkl", ".md", ".mdx", ".txt", ".pdf"}

# file stem -> doc_type for the .pkl files preproc.pipeline writes (mirrors BDC_Chatbot's
# prepare_chromadb.py); anything else defaults to "docs" unless --doc-type is given
SOURCE_DOC_TYPES = {
    "fellows": "fellow",
    "latest_updates": "update",
    "events": "event",
    "pages": "page",
    "freshdesk": "faq",
    "docs": "docs",
    "vids": "video",
}

_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
_MDX_IMPORTS = re.compile(r"^\s*(?:import|export)\s.*$", re.M)
_JSX_TAGS = re.compile(r"</?[A-Z][^>]*>")


def _scalar_meta(meta: dict) -> dict:
    # vector DBs only accept scalar metadata values
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


def load_pkl(path: Path, doc_type: str, use_summary: bool = False):
    """BDC_Chatbot preproc output: list of {'content': str, 'metadata': dict}.
    Embeds metadata['contextualized_chunk'] / ['text_to_embed'] when present, else
    an LLM summary if use_summary — same precedence as the original loadPKL."""
    with open(path, "rb") as f:
        rows = pickle.load(f)
    contents, metas, embed_texts = [], [], []
    for row in tqdm(rows, desc=f"preparing {path.name}"):
        meta = dict(row["metadata"], doc_type=doc_type)
        meta.setdefault("source", str(path.name))  # docs.pkl records carry their own source path
        embed_text = meta.get("contextualized_chunk") or meta.get("text_to_embed")
        if not embed_text and use_summary:
            from .preproc.utils import get_summary
            embed_text = meta["summary"] = get_summary(row["content"])
        contents.append(row["content"])
        embed_texts.append(embed_text or row["content"])
        metas.append(_scalar_meta(meta))
    return contents, metas, embed_texts


def load_text(path: Path, doc_type: str):
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".mdx":
        text = _FRONTMATTER.sub("", text)
        text = _MDX_IMPORTS.sub("", text)
        text = _JSX_TAGS.sub("", text)
    chunks = _splitter.split_text(text)
    metas = [{"source": str(path), "doc_type": doc_type} for _ in chunks]
    return chunks, metas, list(chunks)


def load_pdf(path: Path, doc_type: str):
    from pypdf import PdfReader
    contents, metas = [], []
    for page_num, page in enumerate(PdfReader(path).pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for chunk in _splitter.split_text(text):
            contents.append(chunk)
            metas.append({"source": str(path.name), "page": page_num, "doc_type": doc_type})
    return contents, metas, list(contents)


def iter_files(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            yield from (f for f in sorted(p.rglob("*")) if f.suffix.lower() in SUPPORTED)
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


def _embed_with_retry(emb, batch, attempts=5):
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
    budget = int(os.getenv("EMBEDDING_BATCH_TOKENS", "6000")) * 4
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
    return vectors


# --- push client: bdc-doc-mcp owns the DB; we only talk to its ingest API ---

def _api_post(path, payload):
    url = os.getenv("DOC_MCP_URL", "http://127.0.0.1:8000").rstrip("/") + path
    resp = requests.post(url, json=payload, timeout=300,
                         headers={"Authorization": f"Bearer {os.getenv('INGEST_TOKEN', '')}"})
    if not resp.ok:
        raise RuntimeError(f"POST {url} -> {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def reset_remote():
    """Drop the server-side collection so the next push starts empty."""
    print("resetting remote collection")
    _api_post("/ingest/reset", None)


def push_chunks(ids, contents, embeddings, metas, desc="pushing"):
    """Upsert chunks to bdc-doc-mcp in request-sized batches (embeddings are fat)."""
    batch = int(os.getenv("PUSH_BATCH", "200"))
    for i in tqdm(range(0, len(ids), batch), desc=desc):
        _api_post("/ingest/upsert", [
            {"id": cid, "content": content, "embedding": emb, "metadata": meta}
            for cid, content, emb, meta in zip(ids[i:i + batch], contents[i:i + batch],
                                              embeddings[i:i + batch], metas[i:i + batch])
        ])


def ingest_paths(paths, doc_type: str | None = None, use_summary: bool = False) -> int:
    """Load, embed, and push all supported files under the given paths.
    doc_type applies to every chunk; None derives it per file from SOURCE_DOC_TYPES."""
    emb = get_emb()
    total = 0
    for f in iter_files(paths):
        file_doc_type = doc_type or SOURCE_DOC_TYPES.get(f.stem, "docs")
        if f.suffix == ".pkl":
            contents, metas, embed_texts = load_pkl(f, file_doc_type, use_summary)
        else:
            loader = load_pdf if f.suffix == ".pdf" else load_text
            contents, metas, embed_texts = loader(f, file_doc_type)
        if not contents:
            continue
        embeddings = _embed_batched(emb, embed_texts, desc=f"embedding {f.name}")
        ids = _chunk_ids(contents, metas)
        push_chunks(ids, contents, embeddings, metas, desc=f"pushing {f.name}")
        total += len(contents)
        print(f"pushed {len(contents):4d} chunks from {f}")
    return total


def main(argv=None):
    from .preproc import pipeline  # lazy: pulls in the scraper deps (pandas, bs4, ...)

    parser = argparse.ArgumentParser(
        description="Build the bdc-doc-mcp database: embed documents and push them to its ingest API. "
                    "Push existing files, or --build to preprocess the BDC sources first. See docs/cli.md.")
    parser.add_argument("paths", nargs="*", help="files or directories to push (.pkl .md .mdx .txt .pdf)")
    parser.add_argument("--doc-type",
                        help="metadata doc_type for every chunk; default derives from the file name "
                             f"({', '.join(f'{k}.pkl->{v}' for k, v in SOURCE_DOC_TYPES.items())}), else docs")
    parser.add_argument("--reset", action="store_true", help="drop the remote collection before pushing")
    parser.add_argument("--summarize", action="store_true",
                        help="embed an LLM summary for .pkl records lacking a contextualized chunk")
    build = parser.add_argument_group(
        "build from source",
        "--build runs preproc.pipeline first and pushes what it wrote; the other options here pass through to it")
    build.add_argument("--build", action="store_true", help="preprocess the BDC sources before pushing")
    pipeline.add_args(build)
    args = parser.parse_args(argv)
    if not args.paths and not args.build:
        parser.error("nothing to do: give paths to push, or --build to preprocess the sources first")

    built = pipeline.build(args) if args.build else []  # before --reset: a failed build must not empty the DB
    if args.reset:
        reset_remote()
    total = ingest_paths(built + args.paths, args.doc_type, args.summarize)
    print(f"done: pushed {total} chunks to {os.getenv('DOC_MCP_URL', 'http://127.0.0.1:8000')}")


if __name__ == "__main__":
    main()
