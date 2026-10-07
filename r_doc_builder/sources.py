"""One sources.yaml entry -> chunked documents. Reading and chunking live together because
both depend on what the source is (markdown, html, pdf, transcript).

A document is {"text": whole document text,
               "chunks": [{"content": str, ...extra metadata for that chunk}],
               "metadata": {"source", "page_url", "title"?, "date"? ("YYYY-MM-DD")}}.
"""
import json
import re
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .config import (CHUNK_OVERLAP, CHUNK_SIZE, HTML_DROP_TAGS, HTML_MAIN_TAGS, REPOS_DIR, REQUEST_TIMEOUT,
                     USER_AGENT, YOUTUBE_LANG)

SUPPORTED = {".md", ".mdx", ".txt", ".pdf"}

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.S)
_MDX_IMPORTS = re.compile(r"^\s*(?:import|export)\s.*$", re.M)
_JSX_TAGS = re.compile(r"</?[A-Z][^>]*>")
_MD_IMAGES = re.compile(r"!\[.*?\]\[.*?\]|!\[.*?\]\(.*?\)")
_MD_REFLINKS = re.compile(r"^\[.*?\]:\s*.*$", re.M)
_HEADER = re.compile(r"^(#{1,6})\s+(.+)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FENCED_BLOCK = re.compile(r"^ {0,3}(`{3,}|~{3,}).*?^ {0,3}\1", re.M | re.S)


def _strip_mdx(text):
    """MDX imports/JSX, images and reference links, outside fenced code only: in a fence
    `import os`, `export PATH=...` and List<T> are code."""
    def strip(s):
        for pattern in (_MDX_IMPORTS, _JSX_TAGS, _MD_IMAGES, _MD_REFLINKS):
            s = pattern.sub("", s)
        return s
    out, pos = [], 0
    for m in _FENCED_BLOCK.finditer(text):
        out += [strip(text[pos:m.start()]), m.group(0)]
        pos = m.end()
    return "".join(out + [strip(text[pos:])])


def warn(msg):
    print(f"warning: {msg}", file=sys.stderr)


# --- chunkers ---

def chunk_markdown(text):
    """One chunk per header section, 'hierarchy' = the header path; a section longer than
    CHUNK_SIZE is split further, every piece keeping its hierarchy.
    Returns (cleaned text, chunks); MDX imports/JSX, images and reference links are stripped
    outside fenced code, and a "# comment" inside a fence is not a header."""
    text = _strip_mdx(text)
    chunks, current, headers, in_fence = [], [], [], False

    def flush():  # closure: reads the *current* bindings of current/headers when called
        body = "\n".join(current).strip()
        if body:
            pieces = [body] if len(body) <= CHUNK_SIZE else _splitter.split_text(body)
            chunks.extend({"content": p, "hierarchy": ", ".join(headers)} for p in pieces)

    for line in text.split("\n"):
        in_fence ^= bool(_FENCE.match(line))
        m = not in_fence and _HEADER.match(line)
        if m:
            flush()
            current = []
            headers = headers[: len(m.group(1)) - 1] + [m.group(2).strip()]
        else:
            current.append(line)
    flush()
    return text, chunks


def chunk_text(text):
    return [{"content": c} for c in _splitter.split_text(text)]


def read_pdf(src):
    """src: path or file-like. -> (all pages' text, chunks tagged with their 1-based page)."""
    from pypdf import PdfReader
    pages = [(n, (p.extract_text() or "").strip()) for n, p in enumerate(PdfReader(src).pages, start=1)]
    chunks = [{"content": c, "page": n} for n, t in pages if t for c in _splitter.split_text(t)]
    return "\n\n".join(t for _, t in pages if t), chunks


def read_html(html):
    """-> (<title> or None, main text). HTML_DROP_TAGS thrown away; the first HTML_MAIN_TAGS match wins."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None
    for tag in soup(HTML_DROP_TAGS):
        tag.decompose()
    body = next((t for t in (soup.find(name) for name in HTML_MAIN_TAGS) if t), soup)
    return title or None, re.sub(r"\n{3,}", "\n\n", body.get_text("\n", strip=True))


def _split_frontmatter(text):
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        meta = None
    return (meta if isinstance(meta, dict) else {}), text[m.end():]


def _set_title_date(meta, title, date):
    if title:
        meta["title"] = str(title)
    if hasattr(date, "strftime"):  # yaml parses bare dates
        date = date.strftime("%Y-%m-%d")
    if date and re.match(r"\d{4}-\d{2}-\d{2}", str(date)):
        meta["date"] = str(date)[:10]


def _frontmatter_text(fm, prefix=""):
    """Frontmatter mapping -> 'key: value' lines of its string values; nested keys are prefixed
    ('project title: ...'). Used when a markdown file is only frontmatter (a data record)."""
    lines = []
    for k, v in fm.items():
        if isinstance(v, dict):
            lines.extend(_frontmatter_text(v, f"{prefix}{k} ").splitlines())
        elif isinstance(v, str) and v.strip():
            lines.append(f"{prefix}{k}: {v.strip()}")
    return "\n".join(lines)


# --- files ---

def load_file(path, root=None, page_url=None):
    """One local file -> document, or None when it yields no chunks.
    source = path relative to root (or the path itself); page_url defaults to source."""
    path = Path(path)
    source = path.relative_to(root).as_posix() if root else str(path)
    meta = {"source": source, "page_url": page_url or source}
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text, chunks = read_pdf(path)
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if suffix in (".md", ".mdx"):
            fm, body = _split_frontmatter(raw)
            _set_title_date(meta, fm.get("title") or fm.get("name"), fm.get("date"))
            if fm and not body.strip():
                body = _frontmatter_text(fm)  # frontmatter-only files (people, projects): the fields are the document
            text, chunks = chunk_markdown(body)
        else:
            text, chunks = raw, chunk_text(raw)
    return {"text": text, "chunks": chunks, "metadata": meta} if chunks else None


def load_path(link, page_url_for=None):
    """A file, or a directory searched recursively for SUPPORTED files. Dot-directories and dot-files
    (.git, .gitbook/assets, .github, ...) are skipped: assets and config, not content.
    page_url_for(relative Path) -> str overrides the page_url (git sources use it)."""
    p = Path(link)
    if p.is_dir():
        root = p
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED
                       and not any(part.startswith(".") for part in f.relative_to(p).parts))
    else:
        files, root = [p], None
    docs = []
    for f in tqdm(files, desc=f"reading {p.name}", unit="file", leave=False, disable=len(files) < 2):
        try:
            url = page_url_for(f.relative_to(root)) if page_url_for and root else None
            doc = load_file(f, root, url)
        except Exception as e:
            warn(f"{f}: {type(e).__name__}: {e}")
            continue
        if doc:
            docs.append(doc)
    return docs


# --- url ---

def load_url(link):
    """One web page. A .pdf/.md/.mdx/.txt URL is read as that file type; anything else as HTML."""
    resp = requests.get(link, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    meta = {"source": link, "page_url": link}
    ext = Path(urlparse(link).path).suffix.lower()
    if ext == ".pdf":
        text, chunks = read_pdf(BytesIO(resp.content))
    elif ext in (".md", ".mdx"):
        fm, body = _split_frontmatter(resp.text)
        _set_title_date(meta, fm.get("title"), fm.get("date"))
        text, chunks = chunk_markdown(body)
    elif ext == ".txt":
        text, chunks = resp.text, chunk_text(resp.text)
    else:
        title, text = read_html(resp.text)
        if title:
            meta["title"] = title
        chunks = chunk_text(text)
    return [{"text": text, "chunks": chunks, "metadata": meta}] if chunks else []


# --- git ---

def _repo_slug(url):
    # https://github.com/org/repo.git -> github.com_org_repo
    return re.sub(r"[^A-Za-z0-9._-]+", "_", re.sub(r"^\w+://|^git@|\.git$", "", url)).strip("_")


def load_git(link, pull=False):
    """Clone (depth 1) into REPOS_DIR/<slug>, or reuse the clone (pull only with --pull), then
    load it like a path. link may end in #subdir. A local directory as link is used as the clone."""
    url, _, subdir = link.partition("#")
    if Path(url).is_dir():
        clone = Path(url)
    else:
        clone = Path(REPOS_DIR) / _repo_slug(url)
        if not clone.exists():
            clone.parent.mkdir(parents=True, exist_ok=True)
            try:  # core.longpaths: Windows' 260-char path limit otherwise breaks deep doc trees at checkout
                subprocess.run(["git", "clone", "-c", "core.longpaths=true", "--depth", "1", url, str(clone)], check=True)
            except subprocess.CalledProcessError:
                shutil.rmtree(clone, ignore_errors=True)  # a half-checked-out clone would be reused silently next run
                raise
        elif pull:
            subprocess.run(["git", "-C", str(clone), "-c", "core.longpaths=true", "pull", "--ff-only"], check=True)
    root = clone / subdir if subdir else clone
    if not root.is_dir():
        raise ValueError(f"subdir not found in repo: {subdir or clone}")
    m = re.match(r"https?://github\.com/([^/]+/[^/#]+?)(?:\.git)?/?$", url)
    page_url_for = None
    if m:
        def page_url_for(rel):
            return f"https://github.com/{m.group(1)}/blob/HEAD/{(Path(subdir) / rel).as_posix() if subdir else rel.as_posix()}"
    return load_path(str(root), page_url_for)


# --- youtube ---

def is_youtube(link):
    host = urlparse(link).netloc.lower()
    return any(host == h or host.endswith("." + h) for h in ("youtube.com", "youtu.be"))


def _ytdl(url, **opts):
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, **opts}) as ydl:
        return ydl.extract_info(url, download=False)


def list_videos(link):
    """A video URL -> [itself]; a playlist/channel -> its video URLs (flat listing, one request)."""
    if re.search(r"youtube\.com/(@[^/?#]+|channel/[^/?#]+|c/[^/?#]+|user/[^/?#]+)/?$", link):
        link = link.rstrip("/") + "/videos"  # a bare channel URL lists tabs, not videos
    info = _ytdl(link, extract_flat=True)
    entries = info.get("entries")
    if entries is None:
        return [info.get("webpage_url") or link]
    return [e.get("url") or f"https://www.youtube.com/watch?v={e['id']}" for e in entries if e]


def parse_json3(text):
    """YouTube json3 caption format -> [(start_seconds, text)], empty cues dropped."""
    cues = []
    for ev in json.loads(text).get("events", []):
        t = "".join(seg.get("utf8", "") for seg in ev.get("segs") or []).replace("\n", " ").strip()
        if t:
            cues.append((ev.get("tStartMs", 0) / 1000, t))
    return cues


def fetch_transcript(video_url):
    """-> (yt-dlp info dict, cues). Caption track: manual YOUTUBE_LANG, then automatic YOUTUBE_LANG, then anything."""
    info = _ytdl(video_url)
    lang = YOUTUBE_LANG
    subs, auto = info.get("subtitles") or {}, info.get("automatic_captions") or {}

    def in_lang(pool):  # exact key first, then regional/variant keys like en-US, en-orig
        return next((pool[k] for k in [lang, *pool] if pool.get(k) and (k == lang or k.startswith(lang + "-"))), None)

    tracks = in_lang(subs) or in_lang(auto) or next((t for t in [*subs.values(), *auto.values()] if t), None)
    if not tracks:
        raise ValueError("no captions available")
    fmt = next((t for t in tracks if t.get("ext") == "json3"), None)
    if not fmt:
        raise ValueError("no json3 caption format")
    resp = requests.get(fmt["url"], timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return info, parse_json3(resp.text)


def chunk_transcript(cues, video_id):
    """Concatenate cues into ~CHUNK_SIZE windows, splitting only at cue boundaries."""
    chunks, buf, size, start = [], [], 0, 0.0

    def flush():
        chunks.append({"content": " ".join(buf), "start_seconds": start,
                       "timestamp_url": f"https://youtu.be/{video_id}?t={int(start)}"})

    for t, text in cues:
        if buf and size + len(text) > CHUNK_SIZE:
            flush()
            buf, size = [], 0
        if not buf:
            start = t
        buf.append(text)
        size += len(text) + 1
    if buf:
        flush()
    return chunks


def load_youtube(link):
    docs = []
    for url in list_videos(link):
        try:
            info, cues = fetch_transcript(url)
            vid = info["id"]
        except Exception as e:
            warn(f"{url}: {type(e).__name__}: {e}")
            continue
        page_url = f"https://www.youtube.com/watch?v={vid}"
        meta = {"source": page_url, "page_url": page_url}
        if info.get("title"):
            meta["title"] = info["title"]
        d = info.get("upload_date") or ""  # YYYYMMDD
        if len(d) == 8:
            meta["date"] = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        chunks = chunk_transcript(cues, vid)
        if chunks:
            docs.append({"text": " ".join(t for _, t in cues), "chunks": chunks, "metadata": meta})
    return docs


# --- transcripts ---

_GDRIVE = re.compile(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:export=download&)?id=)([A-Za-z0-9_-]+)")
_YT_ID = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")
_SRT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->")


def _fetch_text(url):
    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.content.decode("utf-8-sig")


def parse_srt(text):
    """SRT -> [(start_seconds, text)]; blocks without a timing line or without text are dropped."""
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines()]
        timing = next((i for i, line in enumerate(lines) if _SRT_TIME.match(line)), None)
        if timing is None:
            continue
        h, m, s, ms = _SRT_TIME.match(lines[timing]).groups()
        body = " ".join(line for line in lines[timing + 1:] if line)
        if body:
            cues.append((int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000, body))
    return cues


def _srt_or_raise(text, ref):
    if text.lstrip()[:15].lower().startswith(("<!doctype html", "<html")):
        raise ValueError(f"HTML page instead of a transcript (Drive sign-in or download quota?): {ref}")
    return text


def read_transcript(ref, list_dir):
    """A transcript entry -> SRT text: a Google Drive share link, any http(s) URL, or a path relative
    to the list's folder (local lists only)."""
    m = _GDRIVE.search(ref)
    if m:
        return _srt_or_raise(_fetch_text(f"https://drive.google.com/uc?export=download&id={m.group(1)}"), ref)
    if ref.startswith(("http://", "https://")):
        return _srt_or_raise(_fetch_text(ref), ref)
    if list_dir is None:
        raise ValueError(f"relative transcript path needs a local list: {ref}")
    return Path(list_dir, ref).read_text(encoding="utf-8-sig")


def load_transcripts(link):
    """A YAML list of {video_url, transcript} — a local .yaml or a YAML URL -> one document per YouTube
    video, chunked like the youtube source; title/date from YouTube when reachable."""
    if link.startswith(("http://", "https://")):
        text, list_dir = _fetch_text(link), None
    else:
        text, list_dir = Path(link).read_text(encoding="utf-8-sig"), Path(link).parent
    entries = yaml.safe_load(text) or []
    if not isinstance(entries, list):
        raise ValueError(f"expected a list of {{video_url, transcript}} entries, got {type(entries).__name__}")
    docs = []
    for n, entry in enumerate(entries, start=1):
        where = f"{link}[{n}]"
        if not isinstance(entry, dict):
            warn(f"{where}: expected a mapping with video_url, transcript")
            continue
        video_url = str(entry.get("video_url") or "").strip()
        ref = str(entry.get("transcript") or "").strip()
        vid = _YT_ID.search(video_url) if is_youtube(video_url) else None
        if not vid:
            warn(f"{where}: not a YouTube video URL: {video_url}")
            continue
        if not ref:
            warn(f"{where}: no transcript for {video_url}")
            continue
        try:
            cues = parse_srt(read_transcript(ref, list_dir))
        except Exception as e:
            warn(f"{where}: {type(e).__name__}: {e}")
            continue
        chunks = chunk_transcript(cues, vid.group(1))
        if not chunks:
            warn(f"{where}: empty transcript: {ref}")
            continue
        page_url = f"https://www.youtube.com/watch?v={vid.group(1)}"
        meta = {"source": page_url, "page_url": page_url}
        try:
            info = _ytdl(video_url)
            if info.get("title"):
                meta["title"] = info["title"]
            d = info.get("upload_date") or ""  # YYYYMMDD
            if len(d) == 8:
                meta["date"] = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        except Exception as e:
            warn(f"{video_url}: no title/date ({type(e).__name__}: {e})")
        docs.append({"text": " ".join(t for _, t in cues), "chunks": chunks, "metadata": meta})
    return docs


# --- freshdesk ---

_FD_PATH = re.compile(r"/support/solutions/(folders/)?\d+(-[^/]*)?/?$")


def is_freshdesk(link):
    """A help-center category (…/support/solutions/<id>) or folder (…/support/solutions/folders/<id>) URL."""
    return link.startswith(("http://", "https://")) and bool(_FD_PATH.search(urlparse(link).path))


def _fd_get(url):
    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def _fd_heading(soup):
    """h2.heading names the category, folder or article; only its direct text (icons are nested)."""
    h = soup.find("h2", class_="heading")
    if not h:
        return None
    direct = h.find(string=True, recursive=False)
    return (direct.strip() if direct and direct.strip() else h.get_text(" ", strip=True)) or None


def load_freshdesk(link):
    """A Freshdesk help-center category or folder URL -> one document per article (public portal
    markup, no API key: category page lists folders, folder page lists articles)."""
    parsed = urlparse(link)
    base = f"{parsed.scheme}://{parsed.netloc}"
    page = _fd_get(link)
    if "/folders/" in parsed.path:
        category, folders = None, [(link, _fd_heading(page))]
    else:
        category = _fd_heading(page)
        folders = [(urljoin(base, a["href"]), a.get("title") or a.get_text(strip=True) or None)
                   for a in page.select("section.cs-g.article-list div.list-lead a[href]")]
        if not folders:
            warn(f"{link}: no folders found (portal markup may have changed)")
    docs = []
    for folder_url, folder in folders:
        try:
            fpage = _fd_get(folder_url)
        except Exception as e:
            warn(f"{folder_url}: {type(e).__name__}: {e}")
            continue
        links = fpage.select("section.article-list.c-list div.c-row.c-article-row a.c-link[href]")
        if not links:
            warn(f"{folder_url}: no articles found (portal markup may have changed)")
        for a in links:
            url = urljoin(base, a["href"])
            try:
                apage = _fd_get(url)
            except Exception as e:
                warn(f"{url}: {type(e).__name__}: {e}")
                continue
            body = apage.find("article", class_="article-body")
            text = re.sub(r"\n{3,}", "\n\n", body.get_text("\n", strip=True)) if body else ""
            chunks = chunk_text(text)
            if not chunks:
                continue
            meta = {"source": url, "page_url": url}
            for key, value in (("title", _fd_heading(apage)), ("category", category), ("folder", folder)):
                if value:
                    meta[key] = value
            docs.append({"text": text, "chunks": chunks, "metadata": meta})
    return docs


# --- dispatch ---

def load_row(source_type, link, pull=False):
    """One sources entry -> documents. ValueError when the link doesn't fit the source_type."""
    is_url = link.startswith(("http://", "https://"))
    if source_type == "path":
        if is_url or not Path(link).exists():
            raise ValueError(f"path does not exist: {link}")
        return load_path(link)
    if source_type == "url":
        if not is_url:
            raise ValueError(f"not an http(s) URL: {link}")
        return load_url(link)
    if source_type == "git":
        if not (is_url or link.startswith(("ssh://", "git@")) or Path(link.partition("#")[0]).is_dir()):
            raise ValueError(f"not a git URL or local repo: {link}")
        return load_git(link, pull)
    if source_type == "youtube":
        if not is_youtube(link):
            raise ValueError(f"not a YouTube URL: {link}")
        return load_youtube(link)
    if source_type == "transcripts":
        if not (is_url or (link.lower().endswith((".yaml", ".yml")) and Path(link).is_file())):
            raise ValueError(f"not a .yaml file or YAML URL: {link}")
        return load_transcripts(link)
    if source_type == "freshdesk":
        if not is_freshdesk(link):
            raise ValueError(f"not a Freshdesk solutions URL (…/support/solutions/<id> or …/folders/<id>): {link}")
        return load_freshdesk(link)
    raise ValueError(f"unknown source_type: {source_type} (path, url, git, youtube, transcripts, freshdesk, yaml)")
