import os
import re

GITBOOK_INCLUDE_DIRS = []  # empty = whole gitbook; GITBOOK_EXCLUDE still applies
GITBOOK_EXCLUDE = [
    "summary.md",
    "nih-recover-release-notes.md",  # tables, too large for the context window
]


def get_bdc_docs_md_files(root_dir="../bdc-docs/docs/docs"):
    md_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".md") and filename not in ["index.md", "glossary.md"]:
                md_files.append(os.path.join(dirpath, filename))
    return md_files


def get_bdc_gitbook_md_files(root_dir="../bdc-gitbook", include_dirs=None):
    """Only files under GITBOOK_INCLUDE_DIRS; pass include_dirs=[] to take everything."""
    include_dirs = GITBOOK_INCLUDE_DIRS if include_dirs is None else include_dirs
    md_files = []
    include_abs = [os.path.abspath(os.path.join(root_dir, d)) for d in include_dirs]

    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
        abs_dirpath = os.path.abspath(dirpath)
        if include_abs:
            dirnames[:] = [
                d for d in dirnames
                if any(os.path.commonpath([os.path.join(abs_dirpath, d), inc]) == inc for inc in include_abs)
            ]
            if not any(os.path.commonpath([abs_dirpath, inc]) == inc for inc in include_abs):
                continue
        for filename in filenames:
            if filename.endswith(".md") and filename.lower() not in GITBOOK_EXCLUDE:
                md_files.append(os.path.join(dirpath, filename))

    print(f"Kept {len(md_files)} markdown files from {root_dir}")
    return md_files


def load_docs_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
    content = re.sub(r"!\[.*?\]\[.*?\]", "", content)      # ![][image1]
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content)      # ![](url)
    content = re.sub(r"^\[.*?\]:\s*.*$", "", content, flags=re.MULTILINE)  # [image1]: url
    return content


def chunk_docs_md_by_headers(file_path):
    """One chunk per markdown header section; metadata keeps the header hierarchy
    and the whole document (used later as contextualizer input)."""
    content = load_docs_md(file_path)
    file_name = os.path.basename(file_path)
    header_pattern = r"^(#{1,6})\s+(.+)$"

    chunks_content, chunks_metadata = [], []
    current_chunk, current_headers = [], []

    def flush():
        chunk_content = "\n".join(current_chunk).strip()
        if chunk_content:
            chunks_content.append(chunk_content)
            chunks_metadata.append({
                "source": os.path.relpath(file_path),
                "file_name": file_name,
                "hierarchy": ", ".join(current_headers),
                "whole_document": content,
            })

    for line in content.split("\n"):
        header_match = re.match(header_pattern, line, re.MULTILINE)
        if header_match:
            if current_chunk:
                flush()
            current_chunk = []
            header_level = len(header_match.group(1))
            current_headers = current_headers[: header_level - 1]
            current_headers.append(header_match.group(2))
        else:
            current_chunk.append(line)

    if current_chunk:
        flush()

    return chunks_metadata, chunks_content
