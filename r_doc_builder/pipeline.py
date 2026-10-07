"""sources.yaml -> data/<config>/<doc_type>.pkl, <config> = the CONFIG_DIR folder name. Never touches the DB;
pushing is ingest.py's job.

sources.yaml is a list of entries {source_type, doc_type, link}. source_type is one of yaml,
path, url, git, youtube, transcripts, freshdesk. doc_type is free text; omitted, blank or 'none' inherits the doc_type
of the `yaml` entry that included this file (entries that never get one land in untyped.pkl
with no doc_type). link is relative to the file it appears in.
"""
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

from . import sources
from .config import PREPROC_DATA_DIR, SOURCES_FILE
from .contextualize import contextualize_chunk
from .sources import warn

UNTYPED = "untyped"
_REMOTE = ("http://", "https://", "ssh://", "git@")


def read_rows(path, inherited=None, _seen=None):
    """Walk a sources yaml and the yaml files it includes -> [(source_type, doc_type|None, link, where)].
    Bad `yaml` entries are warned and skipped; other entries are validated later by sources.load_row."""
    path = Path(path)
    _seen = set() if _seen is None else _seen
    _seen.add(path.resolve())
    with open(path, encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    if not isinstance(entries, list):
        warn(f"{path}: expected a list of entries, got {type(entries).__name__}")
        return []
    rows = []
    for n, entry in enumerate(entries, start=1):
        where = f"{path}[{n}]"
        if not isinstance(entry, dict):
            warn(f"{where}: expected a mapping with source_type, doc_type, link")
            continue
        source_type = str(entry.get("source_type") or "").strip().lower()
        doc_type = str(entry.get("doc_type") or "").strip()
        link = str(entry.get("link") or "").strip()
        if not source_type or not link:
            warn(f"{where}: source_type and link are required")
            continue
        if doc_type.lower() in ("", "none"):
            doc_type = inherited
        if not link.startswith(_REMOTE):
            link = str(path.parent / link)  # relative to this file's folder
        if source_type == "yaml":
            if not link.lower().endswith((".yaml", ".yml")) or not Path(link).is_file():
                warn(f"{where}: yaml link must be an existing .yaml file: {link}")
                continue
            if Path(link).resolve() in _seen:
                warn(f"{where}: {link} already included, skipping")
                continue
            rows.extend(read_rows(link, doc_type, _seen))
        else:
            rows.append((source_type, doc_type, link, where))
    return rows


def to_records(doc, doc_type, contextualize):
    """One chunked document -> ingest records {content, metadata} (the .pkl shape)."""
    base = dict(doc["metadata"])
    if doc_type:
        base["doc_type"] = doc_type
    if base.get("date"):
        base["date_num"] = int(base["date"].replace("-", ""))  # chroma range filters are numeric-only
    records = []
    for chunk in tqdm(doc["chunks"], desc=f"contextualizing {base['source'][-40:]}", leave=False, disable=not contextualize):
        meta = {**base, **{k: v for k, v in chunk.items() if k != "content"}}
        if contextualize:
            meta["contextualized_chunk"] = contextualize_chunk(chunk["content"], doc["text"])
        records.append({"content": chunk["content"], "metadata": meta})
    return records


def add_args(parser):
    """Preprocessing options. Defined once here and attached to both this module's CLI and
    `ingest --build`, which then hands its parsed args straight to build()."""
    parser.add_argument("--yaml", default=SOURCES_FILE, help="root source list (default: %(default)s)")
    parser.add_argument("--sources", nargs="+", default=["all"], metavar="DOC_TYPE",
                        help=f"doc_types to build; '{UNTYPED}' = rows without one (default: all)")
    parser.add_argument("--no-contextualize", action="store_true",
                        help="skip the LLM contextualizer (much faster/cheaper, weaker retrieval)")
    parser.add_argument("--pull", action="store_true",
                        help="git pull --ff-only existing clones of git sources (default: reuse as-is)")
    parser.add_argument("--data-dir", default=PREPROC_DATA_DIR, help="where the .pkl files go (default: %(default)s)")


def build(args) -> list[Path]:
    """Build the selected doc_types into <data-dir>/<doc_type>.pkl; returns the paths written.
    Rows that fail are warned and skipped; exits non-zero only when nothing was built."""
    rows = read_rows(args.yaml)
    wanted = None if "all" in args.sources else set(args.sources)
    groups = defaultdict(list)
    for source_type, doc_type, link, where in tqdm(rows, desc="sources", unit="row"):
        key = doc_type or UNTYPED
        if wanted and key not in wanted:
            continue
        try:
            docs = sources.load_row(source_type, link, pull=args.pull)
        except Exception as e:
            warn(f"{where}: {type(e).__name__}: {e}")
            continue
        if not docs:  # wrong git subdir, JS-only page, scanned PDF: as lost as a row that fails
            warn(f"{where}: {source_type} {link} -> 0 documents")
            continue
        tqdm.write(f"{where}: {source_type} {link} -> {len(docs)} documents", file=sys.stderr)
        for doc in tqdm(docs, desc=f"{key} <- {link[-40:]}", unit="doc", leave=False):
            groups[key].extend(to_records(doc, doc_type, not args.no_contextualize))
    if not groups:
        sys.exit("no records built: no row produced a selected doc_type (see the warnings above, if any)")
    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, records in groups.items():
        path = out_dir / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(records, f)
        print(f"{key}: {len(records)} chunks -> {path}")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the sources listed in sources.yaml into .pkl files without pushing them "
                    "(`python -m r_doc_builder.ingest --build` does both)")
    add_args(parser)
    paths = build(parser.parse_args())
    print(f"\npush with: python -m r_doc_builder.ingest {' '.join(str(p) for p in paths)} [--reset]")


if __name__ == "__main__":
    main()
