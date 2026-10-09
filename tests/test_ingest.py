import io
import pickle
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # not installed: package sits at the repo root

from r_doc_builder import ingest
from r_doc_builder.ingest import _chunk_ids, _embed_batched, push_chunks


class FakeEmb:
    """Records each request so we can assert on how texts were split."""

    def __init__(self):
        self.requests = []

    def embed_documents(self, texts):
        self.requests.append(list(texts))
        return [[0.0] for _ in texts]


class FlakyEmb:
    """Fails the first `fail_times` calls, then behaves like FakeEmb."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("tunnel dropped")
        return [[0.0] for _ in texts]


def _merged(old, new):
    return {k: v for k, v in {**old, **new}.items() if v is not None}


class FakeServer:
    """In-memory r-doc-mcp ingest API with Chroma's semantics (verified on chromadb 1.5.9): metadata
    merges on upsert and update, and a None value deletes the key. Logs every (path, payload)."""

    def __init__(self):
        self.rows, self.calls = {}, []

    def post(self, path, payload):
        self.calls.append((path, payload))
        if path == "/ingest/lookup":
            return {cid: dict(r["metadata"]) for cid, r in self.rows.items()
                    if r["metadata"].get("source") in payload["sources"]}
        if path == "/ingest/upsert":
            for c in payload:
                old = self.rows.get(c["id"], {}).get("metadata", {})
                self.rows[c["id"]] = {"content": c["content"], "embedding": c["embedding"],
                                      "metadata": _merged(old, c["metadata"])}
        elif path == "/ingest/update":
            for u in payload:
                self.rows[u["id"]]["metadata"] = _merged(self.rows[u["id"]]["metadata"], u["metadata"])
        elif path == "/ingest/delete":
            for cid in payload:
                self.rows.pop(cid, None)
        return {}

    def writes(self):
        return [path for path, _ in self.calls if path != "/ingest/lookup"]


class CountingEmb:
    """Deterministic vectors per (model, text); counts texts embedded. From call fail_at on, every
    call raises like a dropped tunnel."""

    def __init__(self, model="m1", fail_at=None):
        self.model, self.fail_at, self.calls, self.texts = model, fail_at, 0, 0

    def embed_documents(self, texts):
        self.calls += 1
        if self.fail_at is not None and self.calls >= self.fail_at:
            raise ConnectionError("tunnel dropped")
        self.texts += len(texts)
        return [[float(len(t)), float(len(self.model))] for t in texts]


def _row(content, source="doc.md", **meta):
    return {"content": content, "metadata": {"source": source, "page_url": f"https://x/{source}", **meta}}


def _push(server, emb, rows, name="docs.pkl", **kw):
    """Write rows to a .pkl and run ingest_paths on it against the fake server."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / name
        with open(path, "wb") as f:
            pickle.dump(rows, f)
        originals = (ingest._api_post, ingest.get_emb, ingest.time.sleep)
        ingest._api_post, ingest.get_emb, ingest.time.sleep = server.post, (lambda: emb), (lambda s: None)
        try:
            with redirect_stderr(io.StringIO()):
                return ingest.ingest_paths([path], **kw)
        finally:
            ingest._api_post, ingest.get_emb, ingest.time.sleep = originals


def _run_paths(server, emb, paths):
    originals = (ingest._api_post, ingest.get_emb, ingest.time.sleep)
    ingest._api_post, ingest.get_emb, ingest.time.sleep = server.post, (lambda: emb), (lambda s: None)
    try:
        with redirect_stderr(io.StringIO()):
            return ingest.ingest_paths(paths)
    finally:
        ingest._api_post, ingest.get_emb, ingest.time.sleep = originals


def test_a_document_in_two_files_is_refused_before_anything_happens():
    server, emb = FakeServer(), CountingEmb()
    with tempfile.TemporaryDirectory() as d:
        for name, doctype in (("a.pkl", "old"), ("b.pkl", "new")):
            with open(Path(d) / name, "wb") as f:
                pickle.dump([_row("same doc", doc_type=doctype)], f)
        for dry in (False, True):
            try:
                originals = (ingest._api_post, ingest.get_emb)
                ingest._api_post, ingest.get_emb = server.post, (lambda: emb)
                try:
                    with redirect_stderr(io.StringIO()):
                        ingest.ingest_paths([Path(d)], dry_run=dry)
                finally:
                    ingest._api_post, ingest.get_emb = originals
                raise AssertionError("a document in two files must be refused")
            except SystemExit as e:
                msg = str(e)
                assert "doc.md" in msg and "a.pkl" in msg and "b.pkl" in msg, msg
            assert server.calls == [] and emb.texts == 0, "nothing may happen before the check"


def test_the_same_file_listed_twice_is_pushed_once():
    server, emb = FakeServer(), CountingEmb()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "docs.pkl"
        with open(path, "wb") as f:
            pickle.dump([_row("one"), _row("two")], f)
        assert _run_paths(server, emb, [Path(d), path]) == 2
    assert emb.texts == 2 and len(server.rows) == 2


def test_reset_remote_checks_the_server_before_dropping_anything():
    calls, original = [], ingest._api_post

    def old_server(path, payload):
        calls.append(path)
        raise RuntimeError("POST /ingest/lookup -> 404: Not Found")

    ingest._api_post = old_server
    try:
        try:
            ingest.reset_remote()
            raise AssertionError("an old server must stop the reset")
        except RuntimeError as e:
            assert "404" in str(e)
    finally:
        ingest._api_post = original
    assert calls == ["/ingest/lookup"], calls


def test_reset_remote_drops_the_collection_on_a_current_server():
    calls, original = [], ingest._api_post
    ingest._api_post = lambda path, payload: calls.append((path, payload)) or {}
    try:
        with redirect_stdout(io.StringIO()):
            ingest.reset_remote()
    finally:
        ingest._api_post = original
    assert calls == [("/ingest/lookup", {"sources": []}), ("/ingest/reset", None)], calls


def test_files_without_chunks_are_skipped():
    """An empty .pkl or a blank markdown file yields no chunks: no line, no lookup, no write for it,
    and the other files still go through."""
    server, emb, out = FakeServer(), CountingEmb(), io.StringIO()
    with tempfile.TemporaryDirectory() as d:
        with open(Path(d) / "empty.pkl", "wb") as f:
            pickle.dump([], f)
        (Path(d) / "blank.md").write_text("", encoding="utf-8")
        with open(Path(d) / "docs.pkl", "wb") as f:
            pickle.dump([_row("one")], f)
        with redirect_stdout(out):
            assert _run_paths(server, emb, [Path(d)]) == 1
    assert [path for path, _ in server.calls] == ["/ingest/lookup", "/ingest/upsert"], server.calls
    assert "empty.pkl" not in out.getvalue() and "blank.md" not in out.getvalue(), out.getvalue()


def test_lookup_is_split_into_push_batch_sized_requests():
    """A file with more sources than PUSH_BATCH asks in several lookups and must still see every
    stored chunk: a batch whose answer got lost would re-embed unchanged documents."""
    server, rows = FakeServer(), [_row("text", source=f"doc{n}.md") for n in range(5)]
    original_batch, ingest.PUSH_BATCH = ingest.PUSH_BATCH, 2
    try:
        _push(server, CountingEmb(), rows)
        server.calls.clear()
        emb = CountingEmb()
        assert _push(server, emb, rows) == 0 and emb.texts == 0, "every batch's stored chunks were found"
    finally:
        ingest.PUSH_BATCH = original_batch
    lookups = [payload["sources"] for path, payload in server.calls if path == "/ingest/lookup"]
    assert [len(s) for s in lookups] == [2, 2, 1], lookups
    assert sorted(s for batch in lookups for s in batch) == [f"doc{n}.md" for n in range(5)]


def test_ingest_tunables_come_from_config():
    import inspect

    from r_doc_builder import config

    assert inspect.signature(ingest._embed_with_retry).parameters["attempts"].default is config.RETRIES

    kwargs = {}
    resp = type("Resp", (), {"ok": True, "json": lambda self: {}})()
    original = ingest.requests.post
    ingest.requests.post = lambda url, **kw: kwargs.update(kw) or resp
    try:
        ingest._api_post("/ingest/upsert", [])
    finally:
        ingest.requests.post = original
    assert kwargs["timeout"] is config.PUSH_TIMEOUT, "the ingest push uses the configured timeout"


def test_batching_respects_token_budget():
    budget_chars = 6000 * 4  # EMBEDDING_BATCH_TOKENS default, ~4 chars/token
    emb = FakeEmb()
    texts = ["x" * 10_000] * 5

    vectors = _embed_batched(emb, texts)

    assert len(vectors) == 5, "one vector per input"
    assert len(emb.requests) > 1, "should split into several requests"
    for req in emb.requests:
        assert sum(len(t) for t in req) <= budget_chars, "request exceeded the budget"


def test_oversized_single_text_is_truncated():
    emb = FakeEmb()
    _embed_batched(emb, ["x" * 500_000])
    assert len(emb.requests) == 1
    assert len(emb.requests[0][0]) == 6000 * 4, "a lone huge chunk must still fit one request"


def test_embed_batched_retries_transient_failure():
    original_sleep = ingest.time.sleep
    ingest.time.sleep = lambda seconds: None
    try:
        emb = FlakyEmb(fail_times=2)
        vectors = _embed_batched(emb, ["a", "b", "c"])
        assert len(vectors) == 3
        assert emb.calls == 3, "should have retried twice before succeeding"
    finally:
        ingest.time.sleep = original_sleep


def test_embed_batched_reraises_after_exhausting_retries():
    from r_doc_builder import config

    original_sleep = ingest.time.sleep
    ingest.time.sleep = lambda seconds: None
    try:
        emb = FlakyEmb(fail_times=999)
        try:
            _embed_batched(emb, ["a", "b", "c"])
            assert False, "expected the persistent failure to propagate"
        except ConnectionError:
            pass
        assert emb.calls == config.RETRIES, "should give up after the RETRIES budget"
    finally:
        ingest.time.sleep = original_sleep


def test_ids_are_stable_and_unique():
    contents = ["same text", "same text", "other"]
    metas = [{"source": "a.pkl"}, {"source": "a.pkl"}, {"source": "a.pkl"}]

    ids = _chunk_ids(contents, metas)
    assert len(set(ids)) == 3, "duplicate content in one file must not collide"
    assert ids == _chunk_ids(contents, metas), "ids must be stable across runs (upsert, not duplicate)"

    # different source => different id, so files don't overwrite each other
    other = _chunk_ids(["same text"], [{"source": "b.pkl"}])
    assert other[0] != ids[0]


def test_push_chunks_batches_requests():
    calls = []
    original = ingest._api_post
    ingest._api_post = lambda path, payload: calls.append((path, payload))
    try:
        n = 450  # PUSH_BATCH default 200 -> 3 requests
        push_chunks([str(i) for i in range(n)], ["c"] * n, [[0.0]] * n, [{}] * n)
    finally:
        ingest._api_post = original

    assert [len(p) for _, p in calls] == [200, 200, 50], "should split into PUSH_BATCH-sized requests"
    assert all(path == "/ingest/upsert" for path, _ in calls)
    sent = [row["id"] for _, p in calls for row in p]
    assert sent == [str(i) for i in range(n)], "every chunk pushed exactly once, in order"
    assert set(calls[0][1][0]) == {"id", "content", "embedding", "metadata"}, "ingest API contract"


def test_ingest_api_errors_stop_the_push():
    """A non-2xx from r-doc-mcp (bad token, server down, wrong-size embeddings) must raise, never
    pass for a push or a --reset that did not happen."""
    resp = type("Resp", (), {"ok": False, "status_code": 500, "text": "boom"})()
    original = ingest.requests.post
    ingest.requests.post = lambda url, **kw: resp
    try:
        for call in (lambda: ingest.push_chunks(["1"], ["c"], [[0.0]], [{}]), ingest.reset_remote):
            try:
                call()
                raise AssertionError("an ingest API error must raise")
            except RuntimeError as e:
                assert "500" in str(e) and "boom" in str(e), e
    finally:
        ingest.requests.post = original


def test_api_post_retries_dropped_connections_but_not_http_errors():
    """A dropped port-forward to r-doc-mcp is retried like the embedding endpoint's; an HTTP error
    (bad token, server bug) is an answer, and raises at once."""
    import requests

    from r_doc_builder import config

    calls = []
    ok = type("Resp", (), {"ok": True, "json": lambda self: {"fine": 1}})()

    def flaky(url, **kw):
        calls.append(url)
        if len(calls) < 3:
            raise requests.ConnectionError("tunnel dropped")
        return ok

    def dead(url, **kw):
        calls.append(url)
        raise requests.Timeout("no answer")

    bad = type("Resp", (), {"ok": False, "status_code": 401, "text": "bad ingest token"})()
    original_post, original_sleep = ingest.requests.post, ingest.time.sleep
    ingest.time.sleep = lambda seconds: None
    try:
        ingest.requests.post = flaky
        assert ingest._api_post("/ingest/upsert", []) == {"fine": 1} and len(calls) == 3, "retried until it answered"

        calls.clear()
        ingest.requests.post = lambda url, **kw: (calls.append(url), (_ for _ in ()).throw(
            requests.exceptions.ChunkedEncodingError("dropped mid-body")) if len(calls) < 2 else ok)[1]
        assert ingest._api_post("/ingest/upsert", []) == {"fine": 1} and len(calls) == 2, "mid-body drop retried"

        calls.clear()
        ingest.requests.post = dead
        try:
            ingest._api_post("/ingest/upsert", [])
            raise AssertionError("a connection that never comes back must raise")
        except requests.Timeout:
            pass
        assert len(calls) == config.RETRIES, "gives up after the RETRIES budget"

        calls.clear()
        ingest.requests.post = lambda url, **kw: calls.append(url) or bad
        try:
            ingest._api_post("/ingest/upsert", [])
            raise AssertionError("an HTTP error must raise")
        except RuntimeError as e:
            assert "401" in str(e), e
        assert len(calls) == 1, "HTTP errors are not retried"
    finally:
        ingest.requests.post, ingest.time.sleep = original_post, original_sleep


def test_embed_batched_rejects_missing_vectors():
    """An embedder returning fewer vectors than texts used to be zipped away: chunks silently
    missing from the push. Stop before pushing instead."""
    class ShortEmb:
        def embed_documents(self, texts):
            return [[0.0] for _ in texts[1:]]
    try:
        ingest._embed_batched(ShortEmb(), ["a", "b", "c"])
        raise AssertionError("missing embeddings must raise")
    except RuntimeError as e:
        assert "2 vectors for 3 texts" in str(e), e


def test_iter_files_skips_git_clones():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "repos" / "x").mkdir(parents=True)
        (d / "repos" / "x" / "clone.md").write_text("# c", encoding="utf-8")
        (d / "keep.md").write_text("# k", encoding="utf-8")
        original_repos_dir = ingest.REPOS_DIR
        ingest.REPOS_DIR = str(d / "repos")  # config.REPOS_DIR is read once at import; patch the module-level name directly
        try:
            assert [f.name for f in ingest.iter_files([d])] == ["keep.md"], "clones under REPOS_DIR are never pushed as ad-hoc files"
        finally:
            ingest.REPOS_DIR = original_repos_dir


def test_ingest_paths_keeps_record_doc_type_unless_overridden():
    pushed = []
    originals = (ingest.get_emb, ingest._embed_batched, ingest.push_chunks, ingest._api_post)
    ingest.get_emb = lambda: FakeEmb()
    ingest._embed_batched = lambda emb, texts, desc="": [[0.0] for _ in texts]
    ingest.push_chunks = lambda ids, contents, embeddings, metas, desc="": pushed.extend(metas)
    ingest._api_post = lambda path, payload: {}  # an empty DB: every chunk is new
    try:
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d) / "faq.pkl", "wb") as f:
                pickle.dump([{"content": "c", "metadata": {"doc_type": "faq"}}], f)
            with open(Path(d) / "untyped.pkl", "wb") as f:
                pickle.dump([{"content": "c", "metadata": {}}], f)
            (Path(d) / "adhoc.md").write_text("---\ndate: 2025-01-15\n---\n# T\nbody", encoding="utf-8")

            with redirect_stderr(io.StringIO()):
                ingest.ingest_paths([d])
            by_source = {m["source"]: m.get("doc_type") for m in pushed}
            assert by_source == {"faq.pkl": "faq", "untyped.pkl": None, str(Path(d) / "adhoc.md"): None}, \
                "records keep the doc_type they carry; nothing is invented from file names"
            adhoc = next(m for m in pushed if m["source"].endswith("adhoc.md"))
            assert adhoc["hierarchy"] == "T", "ad-hoc files go through sources.load_file"
            assert adhoc["date_num"] == 20250115, "ad-hoc files get date_num like pipeline records"

            pushed.clear()
            with redirect_stderr(io.StringIO()):
                ingest.ingest_paths([d], doc_type="docs")
            assert {m["doc_type"] for m in pushed} == {"docs"}, "--doc-type overrides everything"
    finally:
        ingest.get_emb, ingest._embed_batched, ingest.push_chunks, ingest._api_post = originals


def test_push_skips_what_is_already_stored():
    server, emb = FakeServer(), CountingEmb()
    rows = [_row("one"), _row("two"), _row("three")]
    assert _push(server, emb, rows) == 3 and emb.texts == 3
    server.calls.clear()
    assert _push(server, emb, rows) == 0, "nothing changed: nothing embedded"
    assert emb.texts == 3 and server.writes() == [], "and nothing written"


def test_metadata_only_change_updates_in_place():
    server, emb = FakeServer(), CountingEmb()
    _push(server, emb, [_row("one", title="Old", date="2025-01-01")])
    vector = next(iter(server.rows.values()))["embedding"]
    server.calls.clear()
    assert _push(server, emb, [_row("one", title="New")]) == 0, "metadata alone never re-embeds"
    assert server.writes() == ["/ingest/update"]
    (row,) = server.rows.values()
    assert row["metadata"]["title"] == "New" and "date" not in row["metadata"], "a dropped key is deleted, not kept"
    assert row["embedding"] == vector


def test_changed_embedded_text_or_model_reembeds():
    server = FakeServer()
    _push(server, CountingEmb(), [_row("one", contextualized_chunk="ctx A one"), _row("two")])
    assert _push(server, CountingEmb(), [_row("one", contextualized_chunk="ctx B one"), _row("two")]) == 1, \
        "a new context sentence is new embedded text"
    assert _push(server, CountingEmb(model="m2"), [_row("one", contextualized_chunk="ctx B one"), _row("two")]) == 2, \
        "another embedding model re-embeds everything"


def test_chunks_a_document_dropped_are_deleted_and_nothing_else():
    server, emb = FakeServer(), CountingEmb()
    _push(server, emb, [_row("one"), _row("two"), _row("three")])
    # same source, another page_url: another git repo's README.md, pushed from another file
    _push(server, emb, [{"content": "other", "metadata": {"source": "doc.md", "page_url": "https://y/doc.md"}}],
          name="other.pkl")
    _push(server, emb, [_row("one"), _row("three 3")])
    assert sorted(r["content"] for r in server.rows.values()) == ["one", "other", "three 3"]
    assert server.writes()[-1] == "/ingest/delete", "deletes go last, after the replacements are stored"


def test_killed_push_resumes_without_reembedding_what_was_stored():
    server, rows = FakeServer(), [_row(f"chunk {n}") for n in range(5)]
    original_batch, ingest.PUSH_BATCH = ingest.PUSH_BATCH, 2
    try:
        try:
            _push(server, CountingEmb(fail_at=2), rows)  # the second slice's embedding never succeeds
            raise AssertionError("the dead embedding endpoint must stop the run")
        except ConnectionError:
            pass
        assert len(server.rows) == 2, "the first slice was stored before the second was embedded"
        healthy = CountingEmb()
        assert _push(server, healthy, rows) == 3 and healthy.texts == 3, "the rerun embeds only the rest"
        assert len(server.rows) == 5
    finally:
        ingest.PUSH_BATCH = original_batch


def test_main_build_runs_pipeline_then_reset_then_push():
    from r_doc_builder import pipeline
    calls = []
    originals = (pipeline.build, ingest.reset_remote, ingest.ingest_paths)
    pipeline.build = lambda args: calls.append(("build", args.sources, args.no_contextualize)) or [Path("data/docs.pkl")]
    ingest.reset_remote = lambda: calls.append(("reset",))
    ingest.ingest_paths = lambda paths, doc_type=None, use_summary=False, dry_run=False: calls.append(("push", [str(p) for p in paths])) or 0
    try:
        ingest.main(["--build", "--sources", "docs", "--no-contextualize", "--reset", "extra.md"])
    finally:
        pipeline.build, ingest.reset_remote, ingest.ingest_paths = originals

    assert calls == [("build", ["docs"], True), ("reset",), ("push", [str(Path("data/docs.pkl")), "extra.md"])], \
        "pipeline args pass through; build before reset (a failed build must not empty the DB); push built + given"


def test_dry_run_reports_and_writes_nothing():
    server, emb = FakeServer(), CountingEmb()
    _push(server, emb, [_row("one", title="T"), _row("two"), _row("three")])
    before, server.calls = {k: dict(v) for k, v in server.rows.items()}, []
    out = io.StringIO()
    with redirect_stdout(out):
        assert _push(server, emb, [_row("one", title="T2"), _row("two 2")], dry_run=True) == 0
    assert emb.texts == 3, "a dry run embeds nothing"
    assert server.writes() == [] and server.rows == before, "and writes nothing"
    text = out.getvalue()
    assert "1 to embed, 1 to update, 0 unchanged, 2 to delete" in text, text
    assert "would delete 2 chunks of doc.md" in text, text


def test_main_dry_run_passes_through_and_refuses_reset():
    seen = []
    original = ingest.ingest_paths
    ingest.ingest_paths = lambda paths, doc_type=None, use_summary=False, dry_run=False: seen.append(dry_run) or 0
    try:
        ingest.main(["x.pkl", "--dry-run"])
    finally:
        ingest.ingest_paths = original
    assert seen == [True]
    try:
        with redirect_stderr(io.StringIO()):
            ingest.main(["x.pkl", "--dry-run", "--reset"])
        raise AssertionError("--dry-run --reset must be refused")
    except SystemExit as e:
        assert e.code == 2


def test_main_requires_paths_or_build():
    try:
        ingest.main([])
        assert False, "expected an argparse error"
    except SystemExit as e:
        assert e.code == 2


if __name__ == "__main__":
    test_ingest_tunables_come_from_config()
    test_batching_respects_token_budget()
    test_oversized_single_text_is_truncated()
    test_embed_batched_retries_transient_failure()
    test_embed_batched_reraises_after_exhausting_retries()
    test_ids_are_stable_and_unique()
    test_push_chunks_batches_requests()
    test_ingest_api_errors_stop_the_push()
    test_a_document_in_two_files_is_refused_before_anything_happens()
    test_the_same_file_listed_twice_is_pushed_once()
    test_reset_remote_checks_the_server_before_dropping_anything()
    test_reset_remote_drops_the_collection_on_a_current_server()
    test_files_without_chunks_are_skipped()
    test_lookup_is_split_into_push_batch_sized_requests()
    test_api_post_retries_dropped_connections_but_not_http_errors()
    test_embed_batched_rejects_missing_vectors()
    test_iter_files_skips_git_clones()
    test_ingest_paths_keeps_record_doc_type_unless_overridden()
    test_push_skips_what_is_already_stored()
    test_metadata_only_change_updates_in_place()
    test_changed_embedded_text_or_model_reembeds()
    test_chunks_a_document_dropped_are_deleted_and_nothing_else()
    test_killed_push_resumes_without_reembedding_what_was_stored()
    test_dry_run_reports_and_writes_nothing()
    test_main_dry_run_passes_through_and_refuses_reset()
    test_main_build_runs_pipeline_then_reset_then_push()
    test_main_requires_paths_or_build()
    print("ingest self-check passed")
