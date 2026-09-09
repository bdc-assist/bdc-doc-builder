import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # not installed: package sits at the repo root

from bdc_doc_builder import ingest
from bdc_doc_builder.ingest import _chunk_ids, _embed_batched, push_chunks


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
    original_sleep = ingest.time.sleep
    ingest.time.sleep = lambda seconds: None
    try:
        emb = FlakyEmb(fail_times=999)
        try:
            _embed_batched(emb, ["a", "b", "c"])
            assert False, "expected the persistent failure to propagate"
        except ConnectionError:
            pass
        assert emb.calls == 5, "should give up after the default attempt budget"
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


if __name__ == "__main__":
    test_batching_respects_token_budget()
    test_oversized_single_text_is_truncated()
    test_embed_batched_retries_transient_failure()
    test_embed_batched_reraises_after_exhausting_retries()
    test_ids_are_stable_and_unique()
    test_push_chunks_batches_requests()
    print("ingest self-check passed")
