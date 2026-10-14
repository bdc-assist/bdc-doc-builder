"""End-to-end check of incremental ingest against the real r-doc-mcp (bdc-assist#12): seeded random
rounds of source edits, re-labels, chunking and model changes, partial pushes and injected
failures. After every round:
  1. the incremental DB equals a fresh push of the same inputs (r-doc-mcp's tests/compare_db.py);
  2. only what changed cost an LLM call or an embedding;
  3. a round with no change costs nothing and writes nothing;
  4. documents outside a push are untouched.

Needs the r-doc-mcp repo next to this one (or DOC_MCP_DIR) with `uv sync` done; skipped otherwise.
No network: the LLM and the embedder are fakes, and the server runs on a free local port.

    uv run python tests/test_e2e.py                      # 12 rounds, fixed seed, about 80 s
    E2E_ROUNDS=60 E2E_SEED=7 uv run python tests/test_e2e.py
"""
import argparse
import contextlib
import hashlib
import io
import os
import pickle
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # not installed: package sits at the repo root

from r_doc_builder import ingest, pipeline, sources
from fixture_corpus import write_extras

MCP_DIR = Path(os.getenv("DOC_MCP_DIR") or REPO.parent / "bdc-doc-mcp")
TOKEN, COLLECTION = "e2e-token", "e2e"
TYPES = ["docs", "faq", "page"]
WORDS = "data study cohort access token portal workspace analysis variable dataset genome billing".split()
KINDS = ["edit", "add", "remove", "retitle", "move", "chunk_size", "model", "adhoc", "none"]
# each fault is planted, at call k, in the round kind sure to make at least k such calls, so every
# one fires; the lookup and upsert faults land mid-run, after earlier files or slices are stored
PLANTED = {"/ingest/upsert": ("model", 3), "embed": ("add", 1), "/ingest/delete": ("remove", 1),
           "/ingest/update": ("retitle", 1), "/ingest/lookup": ("none", 2)}
REAL_API_POST = ingest._api_post
REAL_SLEEP = time.sleep  # the test no-ops ingest.time.sleep, which is this same module's: keep a real one


def _mcp_python():
    """r-doc-mcp's venv interpreter, to run directly: terminating `uv run` on Windows leaves the
    server it started running."""
    if not (MCP_DIR / "r_doc_mcp").is_dir():
        return None
    out = subprocess.run(["uv", "run", "--project", str(MCP_DIR), "python", "-c", "import sys; print(sys.executable)"],
                         cwd=MCP_DIR, capture_output=True, text=True)
    return out.stdout.strip() or None


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def server(python, db_dir, **overrides):
    """r-doc-mcp's API on db_dir, with ingest pointed at it for the duration; yields its URL.
    overrides: more server environment (another CONFIG_DIR, COLLECTION_NAME, EMBEDDING_MODEL_PROVIDER)."""
    port = _free_port()
    env = {**os.environ, "DB_PATH": str(db_dir), "COLLECTION_NAME": COLLECTION, "INGEST_TOKEN": TOKEN,
           "CONFIG_DIR": "config", **overrides}
    proc = subprocess.Popen([python, "-m", "uvicorn", "r_doc_mcp.api:app", "--port", str(port)],
                            cwd=MCP_DIR, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url, saved = f"http://127.0.0.1:{port}", (ingest.DOC_MCP_URL, ingest.INGEST_TOKEN)
    try:
        for _ in range(150):
            try:
                requests.get(url + "/health", timeout=1)
                break
            except (requests.ConnectionError, requests.Timeout):  # Timeout: chroma still loading on the first request
                if proc.poll() is not None:
                    raise RuntimeError(f"r-doc-mcp exited with {proc.returncode}")
                REAL_SLEEP(0.2)
        else:
            raise RuntimeError("r-doc-mcp did not start within 30s")
        ingest.DOC_MCP_URL, ingest.INGEST_TOKEN = url, TOKEN
        yield url
    finally:
        ingest.DOC_MCP_URL, ingest.INGEST_TOKEN = saved
        proc.terminate()  # safe: chroma persists every request
        proc.wait(timeout=30)


def _embed_hash(model, text):
    """The spec's formula, computed here independently of ingest.py."""
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()[:16]


class FakeEmb:
    """Vector = deterministic function of (model, text), so a fresh push reproduces it exactly.
    Counts texts embedded; from call fail_at on, every call raises like an endpoint that went away."""

    def __init__(self, model):
        self.model, self.texts, self.calls, self.fail_at = model, 0, 0, None

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise ConnectionError("injected: embedding endpoint gone")
        self.texts += len(texts)
        return [[b / 255 for b in hashlib.sha256(f"{self.model}\n{t}".encode()).digest()[:8]] for t in texts]


class ApiSpy:
    """Stands in for ingest._api_post: counts calls per path; fault=(path, k) fails the k-th call to
    that path the way a dying server or a killed process would."""

    def __init__(self):
        self.counts, self.fault, self.fired = {}, None, False

    def __call__(self, path, payload):
        self.counts[path] = self.counts.get(path, 0) + 1
        if self.fault and self.fault[0] == path and self.counts[path] == self.fault[1]:
            self.fired = True
            raise RuntimeError(f"injected: {path} call {self.fault[1]} failed")
        return REAL_API_POST(path, payload)

    def writes(self):
        return sum(n for path, n in self.counts.items() if path != "/ingest/lookup")


class Workspace:
    """A small corpus on disk: markdown docs mapped to doc_types by sources.yaml, an ad-hoc folder
    pushed with --doc-type, and the fixture corpus's extras.pkl: real BDC shapes the markdown can't make
    (faq category/folder, video title/date, a start_seconds float32 can't hold exactly)."""

    def __init__(self, root, rng):
        self.root, self.rng, self.next_part = Path(root), rng, 0
        self.docs, self.adhoc, self.data = self.root / "docs", self.root / "adhoc", self.root / "data"
        self.docs.mkdir()
        self.adhoc.mkdir()
        self.content, self.types = {}, {}
        for n in range(10):
            name = f"doc{n}.md"
            self.types[name] = TYPES[n % len(TYPES)]
            self.content[name] = {"title": f"Doc {n}", "date": f"2025-01-{n + 1:02d}",
                                  "sections": [self.section() for _ in range(rng.randint(2, 4))]}
            self.write(name)
        self.write_sources()
        for n in range(2):
            (self.adhoc / f"note{n}.md").write_text(f"# Note {n}\n{self.paragraph()}", encoding="utf-8")
        self.extras = write_extras(self.root)

    def paragraph(self):
        rng = self.rng
        return " ".join(" ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 14))).capitalize() + "."
                        for _ in range(rng.randint(1, 12)))

    def section(self):
        self.next_part += 1
        return (f"Part {self.next_part}", self.paragraph())

    def write(self, name):
        doc = self.content[name]
        body = "\n".join(f"## {heading}\n{text}\n" for heading, text in doc["sections"])
        date = f"date: {doc['date']}\n" if doc["date"] else ""
        (self.docs / name).write_text(f"---\ntitle: {doc['title']}\n{date}---\n{body}", encoding="utf-8")

    def write_sources(self):
        (self.root / "sources.yaml").write_text(
            "".join(f"- {{source_type: path, doc_type: {self.types[n]}, link: docs/{n}}}\n" for n in sorted(self.types)),
            encoding="utf-8")

    def mutate(self, kind):
        """One edit to a random doc; returns its file name."""
        rng = self.rng
        name = rng.choice(sorted(self.content))
        doc = self.content[name]
        sections = doc["sections"]
        if kind == "edit":
            i = rng.randrange(len(sections))
            sections[i] = (sections[i][0], sections[i][1] + " " + self.paragraph())
        elif kind == "add":
            sections.insert(rng.randint(0, len(sections)), self.section())
        elif kind == "remove":
            if len(sections) > 1:
                sections.pop(rng.randrange(len(sections)))
            else:
                sections[0] = self.section()
        elif kind == "retitle":  # metadata only; dropping the date drops the date and date_num keys
            doc["title"] += " (rev)"
            doc["date"] = None if doc["date"] else f"2025-02-{rng.randint(1, 28):02d}"
        elif kind == "move":
            self.types[name] = rng.choice([t for t in TYPES if t != self.types[name]])
            self.write_sources()
        self.write(name)
        return name


def _rows(pkl_paths):
    for path in pkl_paths:
        with open(path, "rb") as f:
            for row in pickle.load(f):
                yield row, row["metadata"]


def _records(pushes):
    """(ids, embedded texts, sources) of what these pushes send, through ingest's own loaders and
    id function (tested in test_ingest.py)."""
    ids, texts, srcs = [], [], set()
    for paths, doc_type in pushes:
        for f in ingest.iter_files(paths):
            contents, metas, embed_texts = (ingest.load_pkl(f, doc_type) if f.suffix == ".pkl"
                                            else ingest.load_file(f, doc_type))
            ids += ingest._chunk_ids(contents, metas)
            texts += embed_texts
            srcs |= {m["source"] for m in metas}
    return ids, texts, srcs


def _build_args(ws, doc_type):
    parser = argparse.ArgumentParser()
    pipeline.add_args(parser)
    return parser.parse_args(["--yaml", str(ws.root / "sources.yaml"), "--data-dir", str(ws.data)]
                             + (["--sources", doc_type] if doc_type else []))


def _round(ws, kind, fault, state, python, llm_calls):
    """Mutate, build, push (failing once at the injected fault, then rerunning), and check
    everything. Returns whether the fault fired."""
    edited, partial = set(), None
    if kind in ("edit", "add", "remove", "retitle", "move"):
        name = ws.mutate(kind)
        if kind in ("edit", "add", "remove"):
            edited = {name}
        if kind != "move" and ws.rng.random() < 0.5:
            partial = ws.types[name]  # like `--build --sources faq`: only that doc_type is rebuilt and pushed
    elif kind == "chunk_size":
        size = 300 if sources.CHUNK_SIZE != 300 else 1500
        sources.CHUNK_SIZE = size
        sources._splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=50)
    elif kind == "model":
        state["model"] = "m2" if state["model"] == "m1" else "m1"
    elif kind == "adhoc":
        state["adhoc_type"] = "memo" if state["adhoc_type"] == "note" else "note"

    # build: the LLM runs only for chunks no previous .pkl holds with the same text and context_hash
    def key(row, meta):
        return meta["source"], meta.get("page_url"), row["content"], meta.get("context_hash")

    before = {key(r, m) for r, m in _rows(sorted(ws.data.glob("*.pkl")))}
    llm_calls.clear()
    built = [] if kind == "adhoc" else pipeline.build(_build_args(ws, partial))
    built_rows = list(_rows(built))
    expected_llm = sum(1 for r, m in built_rows if key(r, m) not in before)
    assert len(llm_calls) == expected_llm, f"LLM calls: {len(llm_calls)}, expected {expected_llm}"
    # and context_hash must really follow the document: every chunk of an edited one is re-situated
    resituated = {r["content"] for r, m in built_rows if Path(m["source"]).name in edited}
    assert resituated <= set(llm_calls), f"edited document kept stale context: {resituated - set(llm_calls)}"

    adhoc = [([ws.adhoc], state["adhoc_type"])] if state["adhoc_type"] else []
    if kind == "adhoc":
        pushes = adhoc
    elif partial:
        pushes = [(built, None)]
    else:
        pushes = [(built + [ws.extras], None)] + adhoc
    ids, texts, pushed = _records(pushes)

    emb, spy = FakeEmb(state["model"]), ApiSpy()
    ingest.get_emb = lambda: emb
    with server(python, ws.root / "inc"):
        stored = REAL_API_POST("/ingest/lookup", {"sources": sorted(state["sources"] | pushed)})
        expected_embed = sum(1 for cid, text in zip(ids, texts)
                             if stored.get(cid, {}).get("embed_hash") != _embed_hash(state["model"], text))
        others = sorted(state["sources"] - pushed)
        untouched = REAL_API_POST("/ingest/lookup", {"sources": others}) if others else {}

        def push():
            for paths, doc_type in pushes:
                ingest.ingest_paths(paths, doc_type)

        ingest._api_post = spy
        if fault and fault[0] == "embed":
            emb.fail_at = fault[1]
        else:
            spy.fault = fault
        try:
            push()
            failed = False
        except (ConnectionError, RuntimeError) as e:
            failed = True
            print(f"injected failure: {e}")
        fired = spy.fired or (emb.fail_at is not None and emb.calls >= emb.fail_at)
        assert failed == fired, f"run failed={failed}, but the fault fired={fired}"
        if failed:
            first, emb.fail_at, spy.fault = emb.texts, None, None
            push()  # the rerun carries on from what the failed run stored
            assert emb.texts - first <= expected_embed, f"rerun embedded {emb.texts - first}, only {expected_embed} needed"
            assert emb.texts <= expected_embed + ingest.PUSH_BATCH, \
                f"embedded {emb.texts} for {expected_embed} needed: more than one batch wasted"
        else:
            assert emb.texts == expected_embed, f"embedded {emb.texts}, expected {expected_embed}"
        if kind == "none":
            assert emb.texts == 0 and spy.writes() == 0 and not llm_calls, "an unchanged corpus costs nothing"
        ingest._api_post = REAL_API_POST
        if others:
            assert REAL_API_POST("/ingest/lookup", {"sources": others}) == untouched, "documents outside the push changed"

    # the oracle: a fresh push, into an empty DB, of every input as it was last pushed
    snap = ws.root / "snap"
    snap.mkdir(exist_ok=True)
    if kind != "adhoc":
        if not partial:  # a full build is the whole corpus: a doc_type it no longer produces is gone
            for old in snap.glob("*.pkl"):
                old.unlink()
        for path in built:
            shutil.copy(path, snap / path.name)
    state["sources"] |= pushed
    oracle = Path(tempfile.mkdtemp(dir=ws.root, prefix="oracle_"))
    ingest.get_emb = lambda: FakeEmb(state["model"])
    with server(python, oracle):
        for paths, doc_type in [(sorted(snap.glob("*.pkl")) + [ws.extras], None)] + adhoc:
            ingest.ingest_paths(paths, doc_type)
    out = subprocess.run([python, str(MCP_DIR / "tests" / "compare_db.py"), str(ws.root / "inc"), str(oracle),
                          "--collection", COLLECTION], cwd=MCP_DIR, capture_output=True, text=True)
    assert out.returncode == 0, "incremental DB differs from a fresh push of the same inputs:\n" + out.stdout + out.stderr
    return fired


def test_incremental_pushes_match_a_fresh_push():
    python = _mcp_python()
    if not python:
        print(f"skipped: no r-doc-mcp at {MCP_DIR} (set DOC_MCP_DIR, run `uv sync` there)")
        return False
    rounds, seed = int(os.getenv("E2E_ROUNDS", "12")), int(os.getenv("E2E_SEED", "12"))
    rng = random.Random(seed)
    schedule = (rng.sample(KINDS, len(KINDS)) + [rng.choice(KINDS) for _ in range(rounds)])[:rounds]
    faults = {schedule.index(kind): (path, k) for path, (kind, k) in PLANTED.items() if kind in schedule}
    for r in range(len(KINDS), rounds):
        if rng.random() < 0.5:
            faults[r] = (rng.choice(list(PLANTED)), rng.randint(1, 2))

    saved = (ingest.PUSH_BATCH, ingest.time.sleep, ingest.get_emb, ingest._api_post,
             pipeline.contextualize_chunk, sources.CHUNK_SIZE, sources._splitter)
    llm_calls = []
    ingest.PUSH_BATCH, ingest.time.sleep = 4, (lambda seconds: None)  # several slices per file; no backoff waits
    # a real LLM rarely answers twice the same way: neither does this one
    pipeline.contextualize_chunk = lambda chunk, whole: (
        llm_calls.append(chunk) or f"[{uuid.uuid4().hex[:8]}] situates {chunk[:24]!r}. {chunk}")
    root = tempfile.mkdtemp(prefix="r_e2e_")  # kept on failure, for inspection
    fired = set()
    try:
        ws = Workspace(root, rng)
        state = {"model": "m1", "adhoc_type": None, "sources": set()}
        for r, kind in enumerate(["initial"] + schedule):
            fault = faults.get(r - 1) if r else None
            log = io.StringIO()
            try:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    if _round(ws, kind, fault, state, python, llm_calls):
                        fired.add(fault[0])
            except Exception:  # not only assertions: a crashed rerun needs the log and the replay line too
                print(log.getvalue()[-6000:])
                print(f"FAILED round {r} ({kind}, fault {fault}): replay with E2E_SEED={seed} E2E_ROUNDS={rounds}; "
                      f"workspace {root}")
                raise
            print(f"round {r:2d} {kind:10} fault={fault}: ok")
        if rounds >= len(KINDS):
            assert fired == set(PLANTED), f"planted faults that never fired: {set(PLANTED) - fired}"
        shutil.rmtree(root, ignore_errors=True)
    finally:
        (ingest.PUSH_BATCH, ingest.time.sleep, ingest.get_emb, ingest._api_post,
         pipeline.contextualize_chunk, sources.CHUNK_SIZE, sources._splitter) = saved
    return True


if __name__ == "__main__":
    if test_incremental_pushes_match_a_fresh_push():
        print("e2e self-check passed")
