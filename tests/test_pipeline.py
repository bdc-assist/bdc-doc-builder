"""Offline self-checks for sources.py + pipeline.py. No network: requests/git/yt-dlp are stubbed."""
import argparse
import contextlib
import io
import os
import pickle
import re
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # not installed: package sits at the repo root

from r_doc_builder import contextualize
from r_doc_builder import pipeline
from r_doc_builder import sources


def test_chunk_settings_come_from_config():
    from r_doc_builder import config

    assert sources._splitter._chunk_size == config.CHUNK_SIZE
    assert sources._splitter._chunk_overlap == config.CHUNK_OVERLAP
    assert sources.REPOS_DIR is config.REPOS_DIR

    import subprocess
    import yaml

    assert config.BUILD_FILE == os.path.join(config.CONFIG_DIR, "build.yaml")
    with open(config.BUILD_FILE, encoding="utf-8") as f:
        build = yaml.safe_load(f)
    assert [config.CHUNK_SIZE, config.CHUNK_OVERLAP, config.HTML_DROP_TAGS, config.HTML_MAIN_TAGS, config.YOUTUBE_LANG] == \
        [build["chunk_size"], build["chunk_overlap"], build["html_drop_tags"], build["html_main_tags"], build["youtube_lang"]]
    # the old env vars no longer apply, and say so instead of silently building a different corpus
    out = subprocess.run([sys.executable, "-c", "from r_doc_builder import config; print(config.CHUNK_SIZE)"],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
                         env={**os.environ, "CHUNK_SIZE": "7"})
    assert out.stdout.strip() == str(build["chunk_size"]) and "CHUNK_SIZE in the environment is ignored" in out.stderr, out.stderr


def test_data_dir_follows_config_dir():
    """Each CONFIG_DIR builds into its own data/<name>/, so examples never overwrite each other's .pkl files."""
    import subprocess

    out = subprocess.run([sys.executable, "-c", "from r_doc_builder import config; print(config.PREPROC_DATA_DIR)"],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
                         env={**os.environ, "CONFIG_DIR": "examples/bdc/", "PREPROC_DATA_DIR": ""})  # empty = unset, and beats .env
    assert Path(out.stdout.strip()) == Path("data/bdc"), out.stdout + out.stderr


def test_tunables_come_from_config():
    import inspect

    from r_doc_builder import config

    # which client get_llm builds and that the temperature reaches it: tests/test_contextualize.py
    assert inspect.signature(contextualize._invoke_llm).parameters["attempts"].default is config.RETRIES

    assert sources.HTML_DROP_TAGS is config.HTML_DROP_TAGS
    assert sources.HTML_MAIN_TAGS is config.HTML_MAIN_TAGS

    kwargs = {}
    original = sources.requests.get
    sources.requests.get = lambda url, **kw: kwargs.update(kw) or FakeResp(
        "<html><body><main>hello page</main></body></html>")
    try:
        sources.load_url("https://ex.org/p")
    finally:
        sources.requests.get = original
    assert kwargs["timeout"] is config.REQUEST_TIMEOUT, "url downloads use the configured timeout"
    assert kwargs["headers"]["User-Agent"] is config.USER_AGENT, "url downloads send the configured User-Agent"
    assert inspect.signature(contextualize.get_summary).parameters["min_text"].default is config.SUMMARY_MIN_CHARS


def test_prompts_come_from_prompts_yaml():
    from r_doc_builder import config

    assert contextualize.PROMPTS_FILE is config.PROMPTS_FILE
    assert config.PROMPTS_FILE == os.path.join(config.CONFIG_DIR, "prompts.yaml")
    assert pipeline.SOURCES_FILE is config.SOURCES_FILE == os.path.join(config.CONFIG_DIR, "sources.yaml")
    assert set(contextualize.PROMPTS) == {"contextualize_chunk", "summarize"}
    seen = []
    original = contextualize._invoke_llm
    contextualize._invoke_llm = lambda prompt: seen.append(prompt) or "Situating sentence."
    try:
        contextualize.contextualize_chunk("the chunk", "whole doc")
        contextualize.get_summary("t" * 400)
    finally:
        contextualize._invoke_llm = original
    assert seen[0] == contextualize.PROMPTS["contextualize_chunk"].format(
        context="whole doc", chunk_content="the chunk"), "the yaml text is what the LLM gets"
    assert "whole doc" in seen[0] and "the chunk" in seen[0], "{context}/{chunk_content} survived"
    assert seen[1] == contextualize.PROMPTS["summarize"].format(text="t" * 400)


def test_contextualize_prepends_context_and_falls_back_to_raw():
    original = contextualize._invoke_llm
    contextualize._invoke_llm = lambda prompt: "Situating sentence."
    try:
        assert contextualize.contextualize_chunk("the chunk", "whole doc") == "Situating sentence. the chunk"
        contextualize._invoke_llm = lambda prompt: None  # LLM gave up after retries
        assert contextualize.contextualize_chunk("the chunk", "whole doc") == "the chunk"
    finally:
        contextualize._invoke_llm = original


def test_invoke_llm_retries_then_gives_up_and_summary_passthrough():
    from r_doc_builder import config

    class Flaky:
        """Raises for the first fail_times calls, then answers."""
        def __init__(self, fail_times):
            self.fail_times, self.calls = fail_times, 0

        def invoke(self, prompt):
            self.calls += 1
            if self.calls <= self.fail_times:
                raise ConnectionError("503")
            return type("Resp", (), {"content": " ok "})()

    originals = (contextualize.get_llm, contextualize.time.sleep)
    contextualize.time.sleep = lambda seconds: None
    try:
        flaky = Flaky(2)
        contextualize.get_llm = lambda: flaky
        assert contextualize._invoke_llm("p") == "ok" and flaky.calls == 3, "retries transient failures, strips the answer"
        dead = Flaky(999)
        contextualize.get_llm = lambda: dead
        assert contextualize._invoke_llm("p") is None and dead.calls == config.RETRIES, "gives up after the RETRIES budget and returns None"
        assert contextualize.get_summary("short") == "short", "below min_text: returned as-is, no LLM call"
        assert dead.calls == config.RETRIES, "get_summary on short text made no LLM call"
    finally:
        contextualize.get_llm, contextualize.time.sleep = originals


def test_chunk_markdown_by_headers_keeps_hierarchy():
    md = "intro line\n# Title\ntext a\n## Sub\ntext b\n![img](x.png)\n### Deep\ntext c\n## Sub2\ntext d\n"
    text, chunks = sources.chunk_markdown(md)
    assert [c["content"] for c in chunks] == ["intro line", "text a", "text b", "text c", "text d"]
    assert [c["hierarchy"] for c in chunks] == ["", "Title", "Title, Sub", "Title, Sub, Deep", "Title, Sub2"]
    assert "![img]" not in text, "images stripped from the document text too"


def test_chunk_markdown_splits_long_sections():
    from r_doc_builder import config

    words = " ".join(f"word{i}" for i in range(config.CHUNK_SIZE // 3))  # ~2.5x CHUNK_SIZE
    _, chunks = sources.chunk_markdown(f"# Big\n{words}\n## Small\nshort\n")
    big = [c for c in chunks if c["hierarchy"] == "Big"]
    assert len(big) > 1 and all(len(c["content"]) <= config.CHUNK_SIZE for c in big), "long section split to CHUNK_SIZE"
    assert chunks[-1] == {"content": "short", "hierarchy": "Big, Small"}, "short sections stay whole"


def test_chunk_markdown_strips_mdx():
    md = "import X from 'y'\n\n<Hero title=\"x\">\n## A\n<Card>inner</Card> text\n"
    _, chunks = sources.chunk_markdown(md)
    assert chunks == [{"content": "inner text", "hierarchy": "A"}]


def test_load_file_markdown_frontmatter_title_and_date():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "post.md"
        p.write_text("---\ntitle: Hello\ndate: 2025-01-15\n---\n# H\nbody\n", encoding="utf-8")
        doc = sources.load_file(p, Path(d))
    assert doc["metadata"] == {"source": "post.md", "page_url": "post.md", "title": "Hello", "date": "2025-01-15"}
    assert doc["chunks"] == [{"content": "body", "hierarchy": "H"}]
    assert doc["text"].startswith("# H"), "frontmatter is not part of the document text"


def test_load_file_txt_and_empty():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "a.txt").write_text("plain words", encoding="utf-8")
        (Path(d) / "empty.md").write_text("", encoding="utf-8")
        doc = sources.load_file(Path(d) / "a.txt")
        assert doc["chunks"] == [{"content": "plain words"}]
        assert doc["metadata"]["source"].endswith("a.txt"), "single file: source is the given path"
        assert sources.load_file(Path(d) / "empty.md") is None


def test_load_file_frontmatter_only_markdown_uses_its_fields():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fellow.md"
        p.write_text("---\nname: Ada Lovelace\nphoto: ada.jpg\ncohort: II\nproject:\n  title: Engines\n"
                     "  abstract: Analytical engines.\nbio: First programmer.\n---\n", encoding="utf-8")
        doc = sources.load_file(p, Path(d))
    assert doc["metadata"] == {"source": "fellow.md", "page_url": "fellow.md", "title": "Ada Lovelace"}
    assert doc["text"].startswith("name: Ada Lovelace")
    assert "project title: Engines" in doc["text"] and "project abstract: Analytical engines." in doc["text"]
    assert "bio: First programmer." in doc["text"]
    assert len(doc["chunks"]) == 1 and doc["chunks"][0]["content"] == doc["text"]


def test_load_path_walks_dir_and_uses_page_url_callback():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "sub").mkdir()
        (Path(d) / "sub" / "b.md").write_text("# B\nbee", encoding="utf-8")
        (Path(d) / "a.md").write_text("# A\nay", encoding="utf-8")
        (Path(d) / "skip.json").write_text("{}", encoding="utf-8")
        (Path(d) / ".hidden").mkdir()
        (Path(d) / ".hidden" / "h.md").write_text("# H\nhidden", encoding="utf-8")
        (Path(d) / ".dot.md").write_text("# D\ndot", encoding="utf-8")
        docs = sources.load_path(d, page_url_for=lambda rel: f"https://x/{rel.as_posix()}")
    assert [doc["metadata"]["source"] for doc in docs] == ["a.md", "sub/b.md"]
    assert docs[1]["metadata"]["page_url"] == "https://x/sub/b.md"


def test_read_html_main_text_and_title():
    html = ("<html><head><title>Page T</title><script>x()</script></head><body><nav>menu</nav>"
            "<main><h1>Head</h1><p>para one</p></main><footer>foot</footer></body></html>")
    title, text = sources.read_html(html)
    assert title == "Page T"
    assert "para one" in text and "Head" in text
    assert "menu" not in text and "foot" not in text and "x()" not in text


class FakeResp:
    def __init__(self, text="", content=b"", status=200):
        self.text, self.content, self.ok = text, content, status < 400

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError("http error")


def test_load_url_html_and_markdown_by_extension():
    original = sources.requests.get
    pages = {"https://ex.org/p": FakeResp("<html><head><title>T</title></head><body><main>hello page</main></body></html>"),
             "https://ex.org/d.md": FakeResp("# Doc\nmd body")}
    sources.requests.get = lambda url, **kw: pages[url]
    try:
        (doc,) = sources.load_url("https://ex.org/p")
        assert doc["metadata"] == {"source": "https://ex.org/p", "page_url": "https://ex.org/p", "title": "T"}
        assert doc["chunks"] == [{"content": "hello page"}]
        (doc,) = sources.load_url("https://ex.org/d.md")
        assert doc["chunks"] == [{"content": "md body", "hierarchy": "Doc"}]
    finally:
        sources.requests.get = original


def test_load_git_clones_once_pulls_on_request_and_links_to_github():
    calls = []

    def fake_run(cmd, check=True, **kw):
        calls.append(cmd)
        if cmd[1] == "clone":  # simulate the clone: create the repo dir with a file
            dest = Path(cmd[-1]); (dest / "docs").mkdir(parents=True)
            (dest / "docs" / "g.md").write_text("# G\ngee", encoding="utf-8")

    original = sources.subprocess.run
    sources.subprocess.run = fake_run
    original_repos_dir = sources.REPOS_DIR
    with tempfile.TemporaryDirectory() as d:
        sources.REPOS_DIR = d  # config.REPOS_DIR is read once at import; patch the module-level name directly
        try:
            docs = sources.load_git("https://github.com/org/repo#docs")
            assert [c[:2] for c in calls] == [["git", "clone"]]
            assert docs[0]["metadata"] == {"source": "g.md", "page_url": "https://github.com/org/repo/blob/HEAD/docs/g.md"}
            sources.load_git("https://github.com/org/repo#docs")
            assert len(calls) == 1, "existing clone reused without --pull"
            sources.load_git("https://github.com/org/repo#docs", pull=True)
            assert calls[-1][:3] == ["git", "-C", str(Path(d) / "github.com_org_repo")] and "pull" in calls[-1]
            try:
                sources.load_git("https://github.com/org/repo#nope")
                assert False, "missing subdir should raise ValueError"
            except ValueError:
                pass
        finally:
            sources.subprocess.run = original
            sources.REPOS_DIR = original_repos_dir


def test_load_git_uses_longpaths_and_removes_a_failed_clone():
    calls = []

    def failing_run(cmd, check=True, **kw):
        calls.append(cmd)
        Path(cmd[-1]).mkdir(parents=True)  # git creates the directory before the checkout fails
        raise sources.subprocess.CalledProcessError(128, cmd)

    original, original_repos_dir = sources.subprocess.run, sources.REPOS_DIR
    with tempfile.TemporaryDirectory() as d:
        sources.subprocess.run, sources.REPOS_DIR = failing_run, d
        try:
            try:
                sources.load_git("https://github.com/org/broken")
                assert False, "a failed clone must raise"
            except sources.subprocess.CalledProcessError:
                pass
        finally:
            sources.subprocess.run, sources.REPOS_DIR = original, original_repos_dir
        assert calls[0][:6] == ["git", "clone", "-c", "core.longpaths=true", "--depth", "1"], calls[0]
        assert not (Path(d) / "github.com_org_broken").exists(), "a half-checked-out clone is removed so the next run retries"


def test_chunk_transcript_windows_at_cue_boundaries():
    old = sources.CHUNK_SIZE
    sources.CHUNK_SIZE = 1500  # the fixture's cue lengths assume this window
    try:
        cues = [(0.0, "a" * 800), (10.5, "b" * 800), (20.0, "c" * 100)]
        chunks = sources.chunk_transcript(cues, "VID")
        assert [c["start_seconds"] for c in chunks] == [0.0, 10.5]
        assert chunks[0]["content"] == "a" * 800 and chunks[1]["content"] == "b" * 800 + " " + "c" * 100
        assert chunks[1]["timestamp_url"] == "https://youtu.be/VID?t=10"
    finally:
        sources.CHUNK_SIZE = old


def test_parse_json3():
    raw = '{"events":[{"tStartMs":1500,"segs":[{"utf8":"hi "},{"utf8":"there\\n"}]},{"tStartMs":3000},{"tStartMs":4000,"segs":[{"utf8":"next"}]}]}'
    assert sources.parse_json3(raw) == [(1.5, "hi there"), (4.0, "next")]


def test_load_youtube_expands_playlist_and_builds_docs():
    def fake_ytdl(url, **opts):
        if "list=" in url:
            return {"entries": [{"id": "v1", "url": "https://www.youtube.com/watch?v=v1"}, None]}
        return {"id": "v1", "title": "Vid One", "upload_date": "20240302",
                "subtitles": {"fr": [{"ext": "json3", "url": "u.fr"}]},
                "automatic_captions": {"en": [{"ext": "vtt", "url": "u.vtt"}, {"ext": "json3", "url": "u.json3"}]}}
    original = (sources._ytdl, sources.requests.get)
    sources._ytdl = fake_ytdl
    sources.requests.get = lambda url, **kw: FakeResp('{"events":[{"tStartMs":0,"segs":[{"utf8":"words"}]}]}') if url == "u.json3" else FakeResp(status=404)
    try:
        (doc,) = sources.load_youtube("https://www.youtube.com/playlist?list=PL1")
    finally:
        sources._ytdl, sources.requests.get = original
    assert doc["metadata"] == {"source": "https://www.youtube.com/watch?v=v1", "page_url": "https://www.youtube.com/watch?v=v1",
                               "title": "Vid One", "date": "2024-03-02"}
    assert doc["chunks"] == [{"content": "words", "start_seconds": 0.0, "timestamp_url": "https://youtu.be/v1?t=0"}]
    assert doc["text"] == "words"

    original = (sources._ytdl, sources.requests.get)
    sources._ytdl = lambda url, **opts: {"subtitles": {}, "automatic_captions": {"en": [{"ext": "json3", "url": "u.json3"}]}}
    sources.requests.get = lambda url, **kw: FakeResp('{"events":[{"tStartMs":0,"segs":[{"utf8":"words"}]}]}')
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            assert sources.load_youtube("https://www.youtube.com/watch?v=x") == [], "video with no id is skipped, not fatal"
    finally:
        sources._ytdl, sources.requests.get = original
    assert "warning:" in stderr.getvalue()


def test_load_freshdesk_walks_category_folders_and_articles():
    base = "https://help.example.com"
    cat = ('<html><head><title>FAQs :</title></head><body><h2 class="heading">Project FAQs</h2>'
           '<section class="cs-g article-list mt-2"><div class="list-lead"><a href="/support/solutions/folders/1" title="General">General</a></div>'
           '<div class="list-lead"><a href="/support/solutions/folders/2" title="Broken">Broken</a></div></section></body></html>')
    folder = ('<html><body><h2 class="heading">General</h2><section class="c-list article-list">'
              '<div class="c-row c-article-row"><a class="c-link" href="/support/solutions/articles/11-what">What?</a></div>'
              '<div class="c-row c-article-row"><a class="c-link" href="/support/solutions/articles/12-gone">Gone</a></div>'
              '<div class="c-row c-article-row"><a class="c-link" href="/support/solutions/articles/13-empty">Empty</a></div></section></body></html>')
    article = ('<html><body><h2 class="heading">What can it offer me? <span>Print</span></h2>'
               '<article class="article-body"><p>It offers</p><p>things.</p></article></body></html>')
    pages = {f"{base}/support/solutions/9": FakeResp(cat),
             f"{base}/support/solutions/folders/1": FakeResp(folder),
             f"{base}/support/solutions/folders/2": FakeResp(status=404),
             f"{base}/support/solutions/articles/11-what": FakeResp(article),
             f"{base}/support/solutions/articles/12-gone": FakeResp(status=500),
             f"{base}/support/solutions/articles/13-empty": FakeResp('<html><body><h2 class="heading">E</h2></body></html>'),
             f"{base}/support/solutions/8": FakeResp('<html><body><h2 class="heading">Empty</h2></body></html>')}
    original = sources.requests.get
    sources.requests.get = lambda url, **kw: pages[url]
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            docs = sources.load_row("freshdesk", f"{base}/support/solutions/9")
        with redirect_stderr(io.StringIO()):
            folder_docs = sources.load_row("freshdesk", f"{base}/support/solutions/folders/1")
        empty_err = io.StringIO()
        with redirect_stderr(empty_err):
            empty_docs = sources.load_row("freshdesk", f"{base}/support/solutions/8")
    finally:
        sources.requests.get = original
    assert len(docs) == 1, docs
    assert docs[0]["text"] == "It offers\nthings."
    assert docs[0]["chunks"] == [{"content": "It offers\nthings."}]
    assert docs[0]["metadata"] == {"source": f"{base}/support/solutions/articles/11-what",
                                   "page_url": f"{base}/support/solutions/articles/11-what",
                                   "title": "What can it offer me?", "category": "Project FAQs", "folder": "General"}
    assert err.getvalue().count("warning:") == 2, err.getvalue()  # the 404 folder and the 500 article; the empty article is skipped silently
    assert len(folder_docs) == 1 and "category" not in folder_docs[0]["metadata"] and folder_docs[0]["metadata"]["folder"] == "General"
    assert empty_docs == []
    assert "no folders found" in empty_err.getvalue(), empty_err.getvalue()


def test_load_row_rejects_mismatched_links():
    for source_type, link in [("path", "https://x"), ("path", "no/such/dir"), ("url", "local.md"),
                              ("git", "not-a-url"), ("youtube", "https://vimeo.com/1"), ("ftp", "x"),
                              ("freshdesk", "https://help.example.com/about"),
                              ("freshdesk", "https://help.example.com/support/solutions/articles/11-what"),
                              ("freshdesk", "https://help.example.com/support/solutions"),
                              ("transcripts", "no-such.yaml")]:
        try:
            sources.load_row(source_type, link)
            assert False, f"{source_type} {link} should be rejected"
        except ValueError:
            pass
    assert sources.is_freshdesk("https://help.example.com/support/solutions/60000157358-bdc-faqs")


def _write(path, text):
    Path(path).write_text(text, encoding="utf-8")


def test_read_rows_comments_relative_paths_and_inheritance():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "sub").mkdir()
        _write(d / "sources.yaml",
               "# a comment\n"
               "- {source_type: yaml, doc_type: faq, link: sub/faq.yaml}\n"
               "- {source_type: url, doc_type: ' ', link: https://x/a}\n"
               "- {source_type: URL, doc_type: none, link: https://x/b}\n"
               "- {source_type: path, doc_type: docs, link: local}\n"
               "- {source_type: url, link: https://x/c}\n")
        _write(d / "sub" / "faq.yaml",
               "- {source_type: url, link: https://x/f1}\n"
               "- {source_type: url, doc_type: other, link: https://x/f2}\n"
               "- {source_type: yaml, link: ../sources.yaml}\n")
        err = io.StringIO()
        with redirect_stderr(err):
            rows = pipeline.read_rows(d / "sources.yaml")
    assert [(t, dt, link) for t, dt, link, _ in rows] == [
        ("url", "faq", "https://x/f1"),          # inherited from the yaml entry
        ("url", "other", "https://x/f2"),        # included file's own doc_type wins
        ("url", None, "https://x/a"),            # blank -> none
        ("url", None, "https://x/b"),            # 'none' -> none, type lowercased
        ("path", "docs", str(d / "local")),      # relative to the file's folder
        ("url", None, "https://x/c"),            # doc_type omitted -> none
    ]
    assert rows[0][3].endswith("faq.yaml[1]"), "where = file[entry] for warnings"
    assert re.search(r"faq\.yaml\[3\]: .*already included", err.getvalue()), err.getvalue()


def test_read_rows_warns_and_skips_bad_entries():
    with tempfile.TemporaryDirectory() as d:
        _write(Path(d) / "s.yaml",
               "- {source_type: yaml, doc_type: x, link: missing.yaml}\n"
               "- {source_type: yaml, doc_type: x, link: notyaml.txt}\n"
               "- {source_type: url, doc_type: x}\n"
               "- just a string\n")
        _write(Path(d) / "notyaml.txt", "")
        _write(Path(d) / "map.yaml", "source_type: url\nlink: https://x\n")
        err = io.StringIO()
        with redirect_stderr(err):
            assert pipeline.read_rows(Path(d) / "s.yaml") == []
            assert pipeline.read_rows(Path(d) / "map.yaml") == []
    assert err.getvalue().count("warning:") == 5


def test_to_records_merges_metadata_and_contextualizes():
    doc = {"text": "whole", "chunks": [{"content": "c1", "hierarchy": "H"}, {"content": "c2"}],
           "metadata": {"source": "s", "page_url": "u", "date": "2025-01-15"}}
    original = pipeline.contextualize_chunk
    pipeline.contextualize_chunk = lambda chunk, whole: f"ctx({whole}) {chunk}"
    try:
        with redirect_stderr(io.StringIO()):
            recs = pipeline.to_records(doc, "docs", contextualize=True)
    finally:
        pipeline.contextualize_chunk = original
    assert recs[0] == {"content": "c1", "metadata": {"source": "s", "page_url": "u", "date": "2025-01-15", "date_num": 20250115,
                                                     "doc_type": "docs", "hierarchy": "H", "contextualized_chunk": "ctx(whole) c1",
                                                     "context_hash": contextualize.context_hash("whole")}}
    assert "hierarchy" not in recs[1]["metadata"]
    untyped = pipeline.to_records(doc, None, contextualize=False)
    assert "doc_type" not in untyped[0]["metadata"] and "contextualized_chunk" not in untyped[0]["metadata"] and "context_hash" not in untyped[0]["metadata"]


def _args(**kw):
    parser = argparse.ArgumentParser()
    pipeline.add_args(parser)
    args = parser.parse_args([])
    vars(args).update(kw)
    return args


def test_build_groups_by_doc_type_filters_sources_and_skips_bad_rows():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "docs").mkdir()
        _write(d / "docs" / "a.md", "# A\nay")
        _write(d / "sources.yaml",
               "- {source_type: path, doc_type: docs, link: docs}\n"
               "- {source_type: path, link: docs/a.md}\n"
               "- {source_type: path, doc_type: docs, link: missing}\n"
               "- {source_type: youtube, doc_type: video, link: https://vimeo.com/1}\n")
        err = io.StringIO()
        with redirect_stderr(err):
            paths = pipeline.build(_args(yaml=str(d / "sources.yaml"), data_dir=str(d / "out"), no_contextualize=True))
        assert sorted(p.name for p in paths) == ["docs.pkl", "untyped.pkl"]
        with open(d / "out" / "docs.pkl", "rb") as f:
            docs_recs = pickle.load(f)
        with open(d / "out" / "untyped.pkl", "rb") as f:
            untyped_recs = pickle.load(f)
        assert docs_recs[0]["metadata"]["doc_type"] == "docs" and "doc_type" not in untyped_recs[0]["metadata"]
        assert err.getvalue().count("warning:") == 2, "missing path and non-YouTube link are warned, build continues"

        with redirect_stderr(io.StringIO()):
            paths = pipeline.build(_args(yaml=str(d / "sources.yaml"), data_dir=str(d / "out2"), no_contextualize=True, sources=["untyped"]))
        assert [p.name for p in paths] == ["untyped.pkl"]

        _write(d / "empty.yaml", "- {source_type: path, doc_type: docs, link: missing}\n")
        try:
            with redirect_stderr(io.StringIO()):
                pipeline.build(_args(yaml=str(d / "empty.yaml"), data_dir=str(d / "out3"), no_contextualize=True))
            assert False, "a build with zero records must exit non-zero"
        except SystemExit as e:
            assert e.code != 0


def test_load_transcripts_reads_yaml_list():
    srt = ("1\n00:00:01,000 --> 00:00:03,000\nHello there\n\n"
           "2\n00:00:04,500 --> 00:00:06,000\nsecond cue\nstill second\n\n"
           "3\n00:00:07,000 --> 00:00:08,000\n\n")
    list_url = "https://ex.org/videos.yaml"
    yaml_text = ("- video_url: https://www.youtube.com/watch?v=abcdefghijk\n"
                 "  transcript: https://drive.google.com/file/d/FILE1/view?usp=drive_link\n"
                 "- video_url: https://youtu.be/lmnopqrstuv\n  transcript: https://ex.org/t.srt\n"
                 "- video_url: https://vimeo.com/1\n  transcript: https://ex.org/t.srt\n"
                 "- video_url: https://www.youtube.com/watch?v=abcdefghijk\n"
                 "- video_url: https://www.youtube.com/watch?v=zzzzzzzzzzz\n  transcript: https://ex.org/missing.srt\n"
                 "- video_url: https://www.youtube.com/watch?v=yyyyyyyyyyy\n  transcript: https://ex.org/login.srt\n"
                 "- not a mapping\n")
    pages = {list_url: FakeResp(content=yaml_text.encode()),
             "https://drive.google.com/uc?export=download&id=FILE1": FakeResp(content=srt.encode()),
             "https://ex.org/t.srt": FakeResp(content=srt.encode()),
             "https://ex.org/missing.srt": FakeResp(status=404),
             "https://ex.org/login.srt": FakeResp(content=b"<!doctype html><html><body>Sign in</body></html>")}

    def fake_ytdl(url, **opts):
        if "abcdefghijk" in url:
            return {"title": "Vid", "upload_date": "20240302"}
        raise RuntimeError("offline")

    original = (sources._ytdl, sources.requests.get)
    sources._ytdl, sources.requests.get = fake_ytdl, lambda url, **kw: pages[url]
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            docs = sources.load_row("transcripts", list_url)
    finally:
        sources._ytdl, sources.requests.get = original
    assert [d["metadata"] for d in docs] == [
        {"source": "https://www.youtube.com/watch?v=abcdefghijk", "page_url": "https://www.youtube.com/watch?v=abcdefghijk",
         "title": "Vid", "date": "2024-03-02"},
        {"source": "https://www.youtube.com/watch?v=lmnopqrstuv", "page_url": "https://www.youtube.com/watch?v=lmnopqrstuv"}]
    assert docs[0]["chunks"] == [{"content": "Hello there second cue still second", "start_seconds": 1.0,
                                  "timestamp_url": "https://youtu.be/abcdefghijk?t=1"}]
    assert docs[0]["text"] == "Hello there second cue still second"
    assert err.getvalue().count("warning:") == 6, err.getvalue()  # vimeo entry, no transcript, 404 srt, no title/date for video 2, HTML page instead of srt, non-mapping entry
    assert "HTML page instead of a transcript" in err.getvalue()

    def no_net(url, **kw):
        raise AssertionError("a local list with a relative SRT path must not touch the network")

    with tempfile.TemporaryDirectory() as d:
        _write(Path(d) / "v.srt", srt)
        _write(Path(d) / "videos.yaml", "# comment\n- video_url: https://www.youtube.com/watch?v=abcdefghijk\n  transcript: v.srt\n")
        original = (sources._ytdl, sources.requests.get)
        sources._ytdl, sources.requests.get = fake_ytdl, no_net
        try:
            with redirect_stderr(io.StringIO()):
                (doc,) = sources.load_row("transcripts", str(Path(d) / "videos.yaml"))
        finally:
            sources._ytdl, sources.requests.get = original
        assert doc["metadata"]["title"] == "Vid" and doc["chunks"][0]["start_seconds"] == 1.0


def test_fixture_corpus_builds_every_doc_type_offline():
    """examples/fixture (fake, BDC-shaped) builds offline into all seven doc_types with the metadata
    shapes real BDC data has, and extras.yaml adds the shapes only networked sources produce. Guards the
    corpus the fixture rehearsal and the e2e test push."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fixture_corpus import DOC_TYPES, FIXTURE, write_extras

    def offline(url, **opts):
        raise RuntimeError("offline")  # the transcripts source asks yt-dlp for a video's title and date

    original = sources._ytdl
    sources._ytdl = offline
    try:
        with tempfile.TemporaryDirectory() as d:
            with redirect_stderr(io.StringIO()):
                paths = pipeline.build(_args(yaml=str(FIXTURE / "sources.yaml"), data_dir=d, no_contextualize=True))
            recs = {}
            for path in paths:
                with open(path, "rb") as f:
                    recs[path.stem] = pickle.load(f)
            with open(write_extras(d), "rb") as f:
                extras = pickle.load(f)
    finally:
        sources._ytdl = original

    def keys(rows):
        return set().union(*(r["metadata"] for r in rows))

    assert set(recs) == DOC_TYPES, sorted(recs)
    assert all(r["metadata"]["doc_type"] == t for t, rows in recs.items() for r in rows)
    assert "hierarchy" in keys(recs["docs"]) and "title" not in keys(recs["docs"])
    for t in ("page", "fellow", "faq"):
        assert "title" in keys(recs[t]), t
    for t in ("event", "update"):
        assert all({"title", "date", "date_num"} <= set(r["metadata"]) for r in recs[t]), t
    assert {"start_seconds", "timestamp_url"} <= keys(recs["video"])
    assert any(r["metadata"]["start_seconds"] > 0 for r in recs["video"]), "a transcript long enough for two chunks"
    assert len(recs["docs"]) > len({r["metadata"]["source"] for r in recs["docs"]}), "a section long enough to split"
    owners = {}
    for t, rows in recs.items():
        for r in rows:
            owners.setdefault((r["metadata"]["source"], r["metadata"]["page_url"]), set()).add(t)
    assert all(len(ts) == 1 for ts in owners.values()), "no document in two doc_types: ingest refuses that"
    assert {"category", "folder", "title", "date", "date_num", "start_seconds", "timestamp_url"} <= keys(extras)
    assert any(r["metadata"].get("start_seconds") == 12.345 for r in extras), "a float float32 can't hold exactly"
    assert not {(r["metadata"]["source"], r["metadata"]["page_url"]) for r in extras} & set(owners), \
        "extras never share a document with the built files"

    with tempfile.TemporaryDirectory() as d:
        unquoted = Path(d) / "extras.yaml"
        unquoted.write_text("- content: c\n  metadata: {source: s, date: 2025-05-14}\n", encoding="utf-8")
        try:
            write_extras(d, unquoted)
            raise AssertionError("an unquoted date must be refused, not silently dropped at push time")
        except ValueError as e:
            assert "quote dates" in str(e), e


def test_build_warns_on_a_source_with_no_documents():
    """A row that loads nothing (wrong git subdir, JS-only page, scanned PDF) is warned like a row
    that fails; it used to pass with only a progress line while the build carried on without it."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d / "docs").mkdir(); (d / "empty").mkdir()
        _write(d / "docs" / "a.md", "# A\nay")
        _write(d / "sources.yaml", "- {source_type: path, doc_type: docs, link: docs}\n"
                                   "- {source_type: path, doc_type: docs, link: empty}\n")
        err = io.StringIO()
        with redirect_stderr(err):
            paths = pipeline.build(_args(yaml=str(d / "sources.yaml"), data_dir=str(d / "out"), no_contextualize=True))
        assert [p.name for p in paths] == ["docs.pkl"], "the other rows still build"
        # tqdm's bar control codes can precede it on the line
        warnings = [line.split("warning:", 1)[1] for line in err.getvalue().splitlines() if "warning:" in line]
        assert len(warnings) == 1 and "empty" in warnings[0] and "0 documents" in warnings[0], warnings


def _contexts(*paths):
    found = {}
    for path in paths:
        with open(path, "rb") as f:
            found.update({r["content"]: r["metadata"]["contextualized_chunk"] for r in pickle.load(f)})
    return found


def test_build_reuses_context_of_unchanged_documents():
    """A rebuild calls the LLM only for documents whose text changed and keeps the exact sentence
    for the rest, so ingest's embed_hash matches and nothing unchanged is re-embedded. The fake LLM
    never answers the same way twice, like a real one."""
    calls = []
    original = pipeline.contextualize_chunk
    pipeline.contextualize_chunk = lambda chunk, whole: calls.append(chunk) or f"ctx{len(calls)} {chunk}"
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            _write(d / "a.md", "# A\nay\n## A2\nay two")
            _write(d / "b.md", "# B\nbee")
            _write(d / "sources.yaml", "- {source_type: path, doc_type: docs, link: a.md}\n"
                                       "- {source_type: path, doc_type: faq, link: b.md}\n")
            args = _args(yaml=str(d / "sources.yaml"), data_dir=str(d / "out"))
            out = (d / "out" / "docs.pkl", d / "out" / "faq.pkl")
            with redirect_stderr(io.StringIO()):
                pipeline.build(args)
            assert len(calls) == 3
            first = _contexts(*out)

            calls.clear()
            with redirect_stderr(io.StringIO()):
                pipeline.build(args)
            assert calls == [] and _contexts(*out) == first, "nothing changed: no LLM call, the same sentences"

            _write(d / "a.md", "# A\nay\n## A2\nay two, edited")
            calls.clear()
            with redirect_stderr(io.StringIO()):
                pipeline.build(args)
            assert sorted(calls) == ["ay", "ay two, edited"], "an edited document is re-situated whole; b.md is not"

            _write(d / "sources.yaml", "- {source_type: path, doc_type: docs, link: a.md}\n"
                                       "- {source_type: path, doc_type: docs, link: b.md}\n")
            calls.clear()
            with redirect_stderr(io.StringIO()):
                pipeline.build(args)
            assert calls == [], "a source moved to another doc_type keeps its context"
    finally:
        pipeline.contextualize_chunk = original


def test_context_reuse_misses_when_prompt_or_model_changes():
    doc = {"text": "whole", "chunks": [{"content": "c1"}], "metadata": {"source": "s", "page_url": "u"}}
    previous = {("s", "u", "c1", contextualize.context_hash("whole")): "old ctx c1"}
    original, prompt = pipeline.contextualize_chunk, contextualize.PROMPTS["contextualize_chunk"]
    saved_model = os.environ.get("COMPLETION_MODEL")
    pipeline.contextualize_chunk = lambda chunk, whole: f"new ctx {chunk}"

    def context():
        with redirect_stderr(io.StringIO()):
            return pipeline.to_records(doc, None, True, previous)[0]["metadata"]["contextualized_chunk"]

    try:
        assert context() == "old ctx c1"
        assert contextualize.context_hash("whole") != contextualize.context_hash("whole, edited")
        contextualize.PROMPTS["contextualize_chunk"] = prompt + " Be brief."
        assert context() == "new ctx c1", "a new prompt re-situates every chunk"
        contextualize.PROMPTS["contextualize_chunk"] = prompt
        os.environ["COMPLETION_MODEL"] = "another-model"
        assert context() == "new ctx c1", "so does another completion model"
    finally:
        pipeline.contextualize_chunk, contextualize.PROMPTS["contextualize_chunk"] = original, prompt
        if saved_model is None:
            os.environ.pop("COMPLETION_MODEL", None)
        else:
            os.environ["COMPLETION_MODEL"] = saved_model


def test_unreadable_previous_pkl_is_warned_and_ignored():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "docs.pkl").write_bytes(b"not a pickle")
        err = io.StringIO()
        with redirect_stderr(err):
            assert pipeline.previous_contexts(d) == {}
        assert "warning:" in err.getvalue() and "docs.pkl" in err.getvalue(), err.getvalue()


def test_previous_contexts_skips_the_raw_text_fallback_of_a_failed_call():
    rows = [{"content": "raw", "metadata": {"source": "s", "page_url": "u", "contextualized_chunk": "raw", "context_hash": "h"}},
            {"content": "real", "metadata": {"source": "s", "page_url": "u", "contextualized_chunk": "ctx real", "context_hash": "h"}}]
    with tempfile.TemporaryDirectory() as d:
        with open(Path(d) / "docs.pkl", "wb") as f:
            pickle.dump(rows, f)
        assert pipeline.previous_contexts(d) == {("s", "u", "real", "h"): "ctx real"}


if __name__ == "__main__":
    test_chunk_settings_come_from_config()
    test_data_dir_follows_config_dir()
    test_tunables_come_from_config()
    test_prompts_come_from_prompts_yaml()
    test_contextualize_prepends_context_and_falls_back_to_raw()
    test_invoke_llm_retries_then_gives_up_and_summary_passthrough()
    test_chunk_markdown_by_headers_keeps_hierarchy()
    test_chunk_markdown_splits_long_sections()
    test_chunk_markdown_strips_mdx()
    test_load_file_markdown_frontmatter_title_and_date()
    test_load_file_txt_and_empty()
    test_load_file_frontmatter_only_markdown_uses_its_fields()
    test_load_path_walks_dir_and_uses_page_url_callback()
    test_read_html_main_text_and_title()
    test_load_url_html_and_markdown_by_extension()
    test_load_git_clones_once_pulls_on_request_and_links_to_github()
    test_load_git_uses_longpaths_and_removes_a_failed_clone()
    test_chunk_transcript_windows_at_cue_boundaries()
    test_parse_json3()
    test_load_youtube_expands_playlist_and_builds_docs()
    test_load_transcripts_reads_yaml_list()
    test_load_freshdesk_walks_category_folders_and_articles()
    test_load_row_rejects_mismatched_links()
    test_read_rows_comments_relative_paths_and_inheritance()
    test_read_rows_warns_and_skips_bad_entries()
    test_to_records_merges_metadata_and_contextualizes()
    test_build_groups_by_doc_type_filters_sources_and_skips_bad_rows()
    test_build_warns_on_a_source_with_no_documents()
    test_fixture_corpus_builds_every_doc_type_offline()
    test_build_reuses_context_of_unchanged_documents()
    test_context_reuse_misses_when_prompt_or_model_changes()
    test_unreadable_previous_pkl_is_warned_and_ignored()
    test_previous_contexts_skips_the_raw_text_fallback_of_a_failed_call()
    print("pipeline self-check passed")
