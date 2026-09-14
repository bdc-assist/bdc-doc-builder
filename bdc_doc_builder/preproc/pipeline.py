import argparse
import os
import pickle
import re
import subprocess
from pathlib import Path

from tqdm import tqdm

from . import bdc_docs, bdc_repo, freshdesk, vids
from .utils import (contextualize_chunk, event_datetime, paths_to_urls,
                    split_by_sections, youtube_upload_date)

BASE_URL = "https://biodatacatalyst.nhlbi.nih.gov/"
GITHUB_BASE_URL = "https://github.com/stagecc/interim-bdc-website/tree/main/"
GITBOOK_BASE_URL = "https://bdcatalyst.gitbook.io/biodata-catalyst-documentation/"

WEBSITE_DIR = os.getenv("BDC_WEBSITE_DIR", "../interim-bdc-website/")
GITBOOK_DIR = os.getenv("BDC_GITBOOK_DIR", "../bdc-gitbook/")
DATA_DIR = os.getenv("PREPROC_DATA_DIR", "./data/")

# pages hand-picked in preproc_doc.py
PAGE_DIR_PATHS = ["use-bdc/analyze-data/"]
PAGE_FILE_PATHS = [
    "join-bdc/index.mdx", "use-bdc/share-data.mdx",
    "user-resources/terms-of-use.mdx", "user-resources/usage-costs.mdx", "user-resources/usage-terms.mdx",
    "about/key-collaborations.mdx", "about/overview.mdx", "about/research-communities.mdx",
    "use-bdc/explore-data/index.mdx",
]


def build_fellows():
    data_dir = os.path.join(WEBSITE_DIR, "src/data/")
    fellows = bdc_repo.get_fellow_files(
        Path(data_dir, "fellows").resolve(), data_dir, base_url=BASE_URL, remote_file_dir=GITHUB_BASE_URL
    )
    for fellow in fellows:
        project = fellow["metadata"].pop("project", None)
        if project:
            fellow["metadata"]["project_title"] = project.get("title")
            fellow["metadata"]["project_abstract"] = project.get("abstract")
    return fellows


def _build_dated_mdx(subdir):
    data_dir = os.path.join(WEBSITE_DIR, "src/data/")
    rows = bdc_repo.get_data_mdx_files(
        Path(data_dir, subdir).resolve(), data_dir, base_url=BASE_URL, remote_file_dir=GITHUB_BASE_URL
    )
    for row in rows:
        meta = row["metadata"]
        if hasattr(meta.get("date"), "strftime"):
            meta["date"] = meta["date"].strftime("%Y-%m-%d")
        if meta.get("date"):
            # chroma range filters ($gte/$lte) are numeric-only, so keep an int copy
            meta["date_num"] = int(str(meta["date"]).replace("-", ""))
        if isinstance(meta.get("tags"), list):
            meta["tags"] = ", ".join(str(t) for t in meta["tags"])
    return rows


def build_latest_updates():
    return _build_dated_mdx("latest-updates")


def build_events():
    rows = _build_dated_mdx("events")
    for row in rows:
        meta = row["metadata"]
        dt = event_datetime(meta.get("date"), meta.get("time"))
        if dt:
            meta["datetime"] = dt
    return rows


def build_pages(contextualize=True):
    pages_dir = os.path.join(WEBSITE_DIR, "src/pages/")
    mdx_paths, relative_paths = bdc_repo.get_all_mdx_paths(pages_dir, PAGE_DIR_PATHS, PAGE_FILE_PATHS)
    pages_url = paths_to_urls(BASE_URL, relative_paths)

    pages_data = []
    for i, path in tqdm(list(enumerate(mdx_paths)), desc="Processing pages"):
        header, page_content = bdc_repo.clean_mdx(path)
        header["file_path"] = bdc_repo.clean_path(path, pages_dir)
        header["page_url"] = pages_url[i]
        if "menu" in header:
            header["headings"] = ", ".join(pair["heading"] for pair in header["menu"])
            header["hrefs"] = ", ".join(pair["href"] for pair in header["menu"])
            del header["menu"]

        for section in split_by_sections(page_content):
            meta = dict(header)
            if contextualize:
                meta["contextualized_chunk"] = contextualize_chunk(section, whole_document=page_content)
            pages_data.append({"metadata": meta, "content": section})
    return pages_data


def build_freshdesk(contextualize=True):
    content_list, metadata_list = freshdesk.scrape_freshdesk()
    metadata_keys = ["category", "folder", "title"]
    faqs_data = []
    for metadata, content in tqdm(list(zip(metadata_list, content_list)), desc="Processing freshdesk FAQ"):
        if contextualize:
            metadata["contextualized_chunk"] = contextualize_chunk(
                content, metadata_context={k: metadata.get(k) for k in metadata_keys}
            )
        faqs_data.append({"metadata": metadata, "content": content})
    return faqs_data


def build_docs(contextualize=True, include_dirs=None):
    md_file_paths = bdc_docs.get_bdc_gitbook_md_files(GITBOOK_DIR, include_dirs=include_dirs)

    metadata_list, content_list = [], []
    for file_path in tqdm(md_file_paths, desc="Chunking docs"):
        chunks_metadata, chunks_content = bdc_docs.chunk_docs_md_by_headers(file_path)
        metadata_list.extend(chunks_metadata)
        content_list.extend(chunks_content)

    gitbook_root_len = len(GITBOOK_DIR)
    docs_data = []
    for meta, chunk in tqdm(list(zip(metadata_list, content_list)), desc="Contextualizing chunks"):
        if contextualize:
            meta["contextualized_chunk"] = contextualize_chunk(chunk, whole_document=meta["whole_document"])
        meta["page_url"] = GITBOOK_BASE_URL + meta["source"][gitbook_root_len:-3].replace("\\", "/")
        del meta["whole_document"]  # too large for chroma metadata
        docs_data.append({"metadata": meta, "content": chunk})
    return docs_data


def build_vids(contextualize=True):
    all_text, all_metadata = vids.proc_BDC_vids_Google_Sheet()
    upload_dates = {}  # one fetch per unique video, not per chunk
    vids_data = []
    for i in tqdm(range(len(all_text)), desc="Processing videos"):
        for j in range(len(all_text[i])):
            meta = dict(all_metadata[i][j])
            content = all_text[i][j]
            if contextualize:
                content = contextualize_chunk(content, whole_document=meta.get("summary"), is_doc_summary=True)
            url = meta.get("video_url", "")
            video_id = re.search(r"v=([^&]+)", url)
            if video_id:
                meta["timestamp_url"] = f"https://youtu.be/{video_id.group(1)}?t={int(meta['start_seconds'])}"
            if url not in upload_dates:
                upload_dates[url] = youtube_upload_date(url)
            if upload_dates[url]:
                meta["datetime"] = upload_dates[url]
            vids_data.append({"metadata": meta, "content": content})
    return vids_data


BUILDERS = {
    "fellows": lambda ctx: build_fellows(),
    "latest_updates": lambda ctx: build_latest_updates(),
    "events": lambda ctx: build_events(),
    "pages": build_pages,
    "freshdesk": build_freshdesk,
    "docs": build_docs,
    "vids": build_vids,
}


def add_args(parser):
    """Preprocessing options. Defined once here and attached to both this module's CLI and
    `ingest --build`, which then hands its parsed args straight to build()."""
    parser.add_argument("--sources", nargs="+", default=["all"], choices=["all", *BUILDERS], metavar="SOURCE",
                        help=f"any of: all, {', '.join(BUILDERS)} (default: all)")
    parser.add_argument("--no-contextualize", action="store_true",
                        help="skip the LLM contextualizer (much faster/cheaper, weaker retrieval)")
    parser.add_argument("--pull", action="store_true",
                        help="git pull the website/gitbook source repos before preprocessing "
                             "(--ff-only: fails instead of merging if a repo has local commits)")
    parser.add_argument("--data-dir", default=DATA_DIR, help="where the .pkl files go (default: %(default)s)")


def build(args) -> list[Path]:
    """Preprocess the selected sources into <data-dir>/<source>.pkl; returns the paths written.
    `args` is a namespace parsed through add_args()."""
    sources = list(BUILDERS) if "all" in args.sources else args.sources

    if args.pull:
        for repo in (WEBSITE_DIR, GITBOOK_DIR):
            print(f"pulling {repo}")
            subprocess.run(["git", "-C", repo, "pull", "--ff-only"], check=True)

    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    contextualize = not args.no_contextualize

    paths = []
    for name in sources:
        print(f"\n=== {name} ===")
        rows = BUILDERS[name](contextualize)
        path = out_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(rows, f)
        print(f"{name}: {len(rows)} records -> {path}")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BDC sources into .pkl files without pushing them "
                    "(`python -m bdc_doc_builder.ingest --build` does both)")
    add_args(parser)
    paths = build(parser.parse_args())
    print(f"\npush with: python -m bdc_doc_builder.ingest {' '.join(str(p) for p in paths)} [--reset]")


if __name__ == "__main__":
    main()
