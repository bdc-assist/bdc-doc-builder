"""The fixture rehearsal: the real ingest CLI, run as subprocesses the way an operator runs it, against a
real r-doc-mcp serving the fake "fixture" collection from a temporary DB. Builds and pushes the fake
corpus (examples/fixture), rebuilds and pushes again (nothing to do), edits three files and checks that
--dry-run predicts the push exactly, checks search, then compares the DB with a fresh push into an
empty one. A copy of the corpus is edited, never the committed files.

Offline by default: both sides use EMBEDDING_MODEL_PROVIDER=fake. FIXTURE_EMBEDDINGS=real uses the
embedding settings of this repo's .env instead (on both sides), to rehearse against a real endpoint.
Needs r-doc-mcp next to this repo (or DOC_MCP_DIR) with `uv sync` done; skipped otherwise.

    uv run python tests/test_fixture_cli.py
    FIXTURE_EMBEDDINGS=real uv run python tests/test_fixture_cli.py
"""
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

from fixture_corpus import DOC_TYPES, FIXTURE, write_extras
from test_e2e import MCP_DIR, REPO, TOKEN, _mcp_python, server

REAL = os.getenv("FIXTURE_EMBEDDINGS") == "real"
FAKE = {} if REAL else {"EMBEDDING_MODEL_PROVIDER": "fake"}
PLAN = re.compile(r"^(.+?): (\d+) to embed, (\d+) to update, (\d+) unchanged, (\d+) to delete$", re.M)


def _ingest(url, config_dir, *args):
    """python -m r_doc_builder.ingest *args -> {file name: (embed, update, unchanged, delete)} from its
    per-file lines; any non-zero exit fails the test with the CLI's output."""
    env = {**os.environ, "CONFIG_DIR": str(config_dir), "DOC_MCP_URL": url, "INGEST_TOKEN": TOKEN, **FAKE}
    out = subprocess.run([sys.executable, "-m", "r_doc_builder.ingest", *map(str, args)],
                         cwd=REPO, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return {Path(m.group(1)).name: tuple(map(int, m.groups()[1:])) for m in PLAN.finditer(out.stdout)}


def _replace(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{path} no longer contains {old!r}: update the edits here"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _edit(cfg):
    """Three changes an operator might make between pushes, each to a one-chunk part of its file:
    a docs section edited, an event retitled, a page section removed."""
    _replace(cfg / "docs" / "getting-started.md", "directly from the browser.",
             "directly from the browser, or with the command-line uploader for bigger ones.")
    _replace(cfg / "events" / "event-2024-11-cohort-webinar.md", "title: Cohort building webinar",
             "title: Cohort building webinar (recording available)")
    page = cfg / "pages" / "usage-costs.md"
    text = page.read_text(encoding="utf-8")
    page.write_text(text[:text.index("## Egress")].rstrip() + "\n", encoding="utf-8")


def _search(url, **body):
    res = requests.post(url + "/search", json=body)
    assert res.ok, res.text
    return res.json()


def test_fixture_rehearsal():
    python = _mcp_python()
    if not python:
        print(f"skipped: no r-doc-mcp at {MCP_DIR} (set DOC_MCP_DIR, run `uv sync` there)")
        return False
    root = Path(tempfile.mkdtemp(prefix="r_fixture_"))  # kept on failure, for inspection
    cfg, data = root / "fixture", root / "data"
    shutil.copytree(FIXTURE, cfg)
    data.mkdir()
    write_extras(data)
    mcp_env = {"CONFIG_DIR": "examples/fixture", "COLLECTION_NAME": "fixture", **FAKE}
    build = ("--build", "--no-contextualize", "--data-dir", data)
    with server(python, root / "db", **mcp_env) as url:
        # 1. the first push embeds everything. --build pushes what it wrote and data/ adds extras.pkl, so
        #    each built file is named twice: it is still pushed once
        first = _ingest(url, cfg, *build, data)
        assert set(first) == {f"{t}.pkl" for t in DOC_TYPES} | {"extras.pkl"}, first
        assert all(embed > 0 and (update, same, delete) == (0, 0, 0)
                   for embed, update, same, delete in first.values()), first
        total = sum(embed for embed, *_ in first.values())
        assert requests.get(url + "/health").json()["documents"] == total

        # 2. rebuilding unchanged sources and pushing again costs nothing
        again = _ingest(url, cfg, *build, data)
        assert again == {f: (0, 0, embed, 0) for f, (embed, *_) in first.items()}, again

        # 3. after three edits, --dry-run predicts the push exactly and the push touches only those chunks
        _edit(cfg)
        predicted = _ingest(url, cfg, *build, "--dry-run", data)
        pushed = _ingest(url, cfg, data)
        assert pushed == predicted, (predicted, pushed)
        changed = {"docs.pkl": (1, 0, 1), "event.pkl": (0, 1, 0), "page.pkl": (0, 0, 1)}  # (embed, update, delete)
        for f, (embed, update, _, delete) in pushed.items():
            assert (embed, update, delete) == changed.get(f, (0, 0, 0)), (f, pushed[f])
        assert requests.get(url + "/health").json()["documents"] == total - 1, "one page section fewer"

        # 4. search: a chunk's own text finds it; keyword mode and the date filter work on real shapes
        with open(data / "docs.pkl", "rb") as f:
            chunk = pickle.load(f)[0]["content"]
        assert _search(url, query=chunk)[0]["content"] == chunk
        hits = _search(url, query="PIC-SURE", mode="keyword")
        assert hits and all(h["metadata"]["doc_type"] in {"docs", "page", "faq", "video"} for h in hits), hits
        assert any("PIC-SURE" in h["content"] for h in hits), hits
        dated = _search(url, query="workshop webinar", mode="keyword", date_from="2025-01-01")
        assert dated and all(h["metadata"]["date_num"] >= 20250101 for h in dated), dated

    # 5. a fresh push of the same files into an empty DB must hold exactly what the incremental one does
    with server(python, root / "fresh", **mcp_env) as url:
        _ingest(url, cfg, data)
    out = subprocess.run([python, str(MCP_DIR / "tests" / "compare_db.py"), str(root / "db"), str(root / "fresh"),
                          "--collection", "fixture", *(["--atol", "1e-4"] if REAL else [])],
                         cwd=MCP_DIR, capture_output=True, text=True)
    assert out.returncode == 0, "incremental DB differs from a fresh push:\n" + out.stdout + out.stderr
    shutil.rmtree(root, ignore_errors=True)
    return True


if __name__ == "__main__":
    if test_fixture_rehearsal():
        print("fixture rehearsal passed")
