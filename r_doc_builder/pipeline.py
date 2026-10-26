"""sources.yaml -> data/<config>/<doc_type>.pkl, <config> = the CONFIG_DIR folder name. Never touches the DB;
pushing is ingest.py's job.

sources.yaml is a list of entries {source_type, doc_type, link, exclude?}. source_type is one of yaml,
path, url, git, youtube, transcripts, freshdesk. doc_type is free text; omitted, blank or 'none' inherits the doc_type
of the `yaml` entry that included this file (entries that never get one land in untyped.pkl
with no doc_type). link is relative to the file it appears in. exclude: a glob or list of globs;
documents whose source (a path/git directory's file path relative to the link) matches one are dropped.
"""
import argparse
import pickle
import sys
from collections import defaultdict
from fnmatch import fnmatchcase
from pathlib import Path

import yaml
from tqdm import tqdm

from . import sources
from .config import PREPROC_DATA_DIR, SOURCES_FILE
from .contextualize import context_hash, contextualize_chunk
from .sources import warn

UNTYPED = "untyped"
_REMOTE = ("http://", "https://", "ssh://", "git@")


def read_rows(path, inherited=None, _seen=None):
    """Walk a sources yaml and the yaml files it includes -> [(source_type, doc_type|None, link, where, exclude)].
    A malformed entry raises ValueError: every source listed must build; other entries are validated
    later by sources.load_row."""
    path = Path(path)
    _seen = set() if _seen is None else _seen
    _seen.add(path.resolve())
    with open(path, encoding="utf-8") as f:
        entries = yaml.safe_load(f) or []
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list of entries, got {type(entries).__name__}")
    rows = []
    for n, entry in enumerate(entries, start=1):
        where = f"{path}[{n}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: expected a mapping with source_type, doc_type, link")
        source_type = str(entry.get("source_type") or "").strip().lower()
        doc_type = str(entry.get("doc_type") or "").strip()
        link = str(entry.get("link") or "").strip()
        if not source_type or not link:
            raise ValueError(f"{where}: source_type and link are required")
        exclude = entry.get("exclude") or []
        exclude = [exclude] if isinstance(exclude, str) else exclude
        if not isinstance(exclude, list):
            raise ValueError(f"{where}: exclude must be a glob or a list of globs")
        if doc_type.lower() in ("", "none"):
            doc_type = inherited
        if not link.startswith(_REMOTE):
            link = str(path.parent / link)  # relative to this file's folder
        if source_type == "yaml":
            if not link.lower().endswith((".yaml", ".yml")) or not Path(link).is_file():
                raise ValueError(f"{where}: yaml link must be an existing .yaml file: {link}")
            if Path(link).resolve() in _seen:
                warn(f"{where}: {link} already included, skipping")
                continue
            rows.extend(read_rows(link, doc_type, _seen))
        else:
            rows.append((source_type, doc_type, link, where, tuple(str(g) for g in exclude)))
    return rows


def previous_contexts(out_dir):
    """{(source, page_url, chunk content, context_hash): contextualized_chunk} from the .pkl files a
    previous build left in out_dir. Every doc_type's, so a source moved to another doc_type keeps
    its sentences. Reusing them skips the LLM call and keeps the embedded text identical, so
    ingest skips re-embedding those chunks too."""
    found = {}
    for path in sorted(Path(out_dir).glob("*.pkl")):
        try:
            with open(path, "rb") as f:
                for row in pickle.load(f):
                    meta = row["metadata"]
                    # a sentence equal to the content is the fallback of a failed LLM call: not reused, so a
                    # transient failure is retried next build (a chunk the content filter always blocks costs
                    # one call per build; its embedded text stays the raw content, so the push doesn't re-embed it)
                    if meta.get("contextualized_chunk") not in (None, "", row["content"]) and meta.get("context_hash"):
                        key = (meta.get("source"), meta.get("page_url"), row["content"], meta["context_hash"])
                        found[key] = meta["contextualized_chunk"]
        except Exception as e:
            warn(f"{path}: unreadable, its contexts are not reused ({type(e).__name__}: {e})")
    return found


def to_records(doc, doc_type, contextualize, previous=None):
    """One chunked document -> ingest records {content, metadata} (the .pkl shape). previous
    (see previous_contexts) supplies context sentences to reuse instead of calling the LLM."""
    base = dict(doc["metadata"])
    if doc_type:
        base["doc_type"] = doc_type
    if base.get("date"):
        base["date_num"] = int(base["date"].replace("-", ""))  # chroma range filters are numeric-only
    if contextualize:
        base["context_hash"] = context_hash(doc["text"])
    records = []
    for chunk in tqdm(doc["chunks"], desc=f"contextualizing {base['source'][-40:]}", leave=False, disable=not contextualize):
        meta = {**base, **{k: v for k, v in chunk.items() if k != "content"}}
        if contextualize:
            key = (meta["source"], meta.get("page_url"), chunk["content"], base["context_hash"])
            meta["contextualized_chunk"] = (previous or {}).get(key) or contextualize_chunk(chunk["content"], doc["text"])
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


def build(args) -> tuple[list[Path], list[str]]:
    """Build the selected doc_types into <data-dir>/<doc_type>.pkl -> (paths written, doc_types skipped).
    A doc_type is all or nothing: a row that fails or loads no documents is warned and its whole
    doc_type skipped (the previous .pkl and the DB keep its last good build, so it is never pushed
    half-loaded or stamped as freshly ingested) while the other doc_types build; rebuild it with
    --sources once fixed. Every row loads before any contextualizing, so a skipped doc_type costs no
    LLM calls. Exits non-zero when nothing was built."""
    try:
        rows = read_rows(args.yaml)
    except ValueError as e:
        sys.exit(f"build stopped, nothing written: {e}")
    wanted = None if "all" in args.sources else set(args.sources)
    out_dir = Path(args.data_dir)
    loaded, failed = [], set()
    for source_type, doc_type, link, where, exclude in tqdm(rows, desc="sources", unit="row"):
        key = doc_type or UNTYPED
        if wanted and key not in wanted:
            continue
        try:
            docs = sources.load_row(source_type, link, pull=args.pull)
        except Exception as e:
            warn(f"{where}: {type(e).__name__}: {e}")
            failed.add(key)
            continue
        docs = [d for d in docs if not any(fnmatchcase(d["metadata"]["source"], g) for g in exclude)]
        if not docs:  # wrong git subdir, JS-only page, scanned PDF: as lost as a row that fails
            warn(f"{where}: {source_type} {link} -> 0 documents")
            failed.add(key)
            continue
        tqdm.write(f"{where}: {source_type} {link} -> {len(docs)} documents", file=sys.stderr)
        loaded.append((key, doc_type, link, docs))
    skipped = sorted(failed)
    if skipped:
        warn(f"skipped {', '.join(skipped)}: a source above failed, so nothing of it is built; "
             f"once fixed, rebuild with --sources {' '.join(skipped)}")
    # sentences from the last build into this dir: unchanged chunks of unchanged documents keep theirs
    previous = {} if args.no_contextualize else previous_contexts(out_dir)
    groups = defaultdict(list)
    for key, doc_type, link, docs in loaded:
        if key in failed:
            continue
        for doc in tqdm(docs, desc=f"{key} <- {link[-40:]}", unit="doc", leave=False):
            groups[key].extend(to_records(doc, doc_type, not args.no_contextualize, previous))
    if not groups:
        sys.exit("no records built: no row produced a selected doc_type (see the warnings above, if any)")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, records in groups.items():
        path = out_dir / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(records, f)
        print(f"{key}: {len(records)} chunks -> {path}")
        paths.append(path)
    return paths, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the sources listed in sources.yaml into .pkl files without pushing them "
                    "(`python -m r_doc_builder.ingest --build` does both)")
    add_args(parser)
    paths, skipped = build(parser.parse_args())
    # --reset would drop a skipped doc_type's chunks with nothing to replace them
    print(f"\npush with: python -m r_doc_builder.ingest {' '.join(str(p) for p in paths)}{'' if skipped else ' [--reset]'}")


if __name__ == "__main__":
    main()
