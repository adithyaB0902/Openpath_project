import json
from datetime import datetime, timezone

import pytest
from huggingface_hub import DatasetInfo, ModelInfo
from requests.structures import CaseInsensitiveDict

from openpath.db import init_db
from openpath.ingest import github as gh
from openpath.ingest import huggingface as hf

NOW = datetime(2026, 11, 24, tzinfo=timezone.utc)
CARD = "---\nlicense: apache-2.0\n---\n# Model A\n\nThis is a small language model tuned for customer support dialogue.\n"


def model(repo, downloads, **extra):
    data = {
        "id": repo, "downloads": downloads, "likes": 10,
        "tags": ["license:apache-2.0", "transformers", "text-generation", "region:us", "arxiv:2307.09288"],
        "pipeline_tag": "text-generation", "library_name": "transformers",
        "createdAt": "2025-01-01T00:00:00.000Z", "lastModified": "2026-10-01T00:00:00.000Z",
        "safetensors": {"total": 7_000_000_000, "parameters": {"BF16": 7_000_000_000}},
    }
    data.update(extra)
    return ModelInfo(**data)


def dataset(repo, downloads):
    return DatasetInfo(**{
        "id": repo, "downloads": downloads, "likes": 3, "tags": ["license:cc-by-4.0", "task_categories:question-answering"],
        "createdAt": "2025-01-01T00:00:00.000Z", "lastModified": "2026-10-01T00:00:00.000Z",
    })


class FakeApi:
    def __init__(self, models=None, datasets=None, reject_expand=False):
        self.models, self.datasets, self.reject_expand = models or [], datasets or [], reject_expand
        self.calls = []

    def list_models(self, filter=None, pipeline_tag=None, sort=None, limit=None, expand=None, full=None,
                    cardData=None, direction=None):
        self.calls.append(("models", {"pipeline_tag": pipeline_tag, "expand": expand, "full": full}))
        if expand is not None and self.reject_expand:
            raise TypeError("expand not supported")
        return iter(self.models[:limit])

    def list_datasets(self, filter=None, sort=None, limit=None, expand=None, full=None, direction=None):
        self.calls.append(("datasets", {"filter": filter}))
        return iter(self.datasets[:limit])


class FakeResp:
    def __init__(self, status=200, text="", headers=None):
        self.status_code, self.text = status, text
        self.headers = CaseInsensitiveDict(headers or {})


class FakeSession:
    def __init__(self, responder):
        self.responder, self.calls = responder, []

    def get(self, url, headers=None, timeout=None):
        self.calls.append(url)
        return self.responder(url)


def card_ok(url):
    return FakeResp(200, CARD)


def make(tmp_path, api, responder=card_ok):
    conn = init_db(str(tmp_path / "t.db"))
    sess = FakeSession(responder)
    fetcher = hf.CardFetcher("tok", session=sess, sleep=lambda s: None)
    return conn, api, fetcher, sess


def test_ingest_models(tmp_path):
    api = FakeApi(models=[model("a/big", 5000), model("b/small", 2000), model("c/tiny", 10)])
    conn, api, fetcher, sess = make(tmp_path, api)
    stats = hf.ingest(conn, api, fetcher, tasks=["text-generation"], min_downloads=1000, now_fn=lambda: NOW)
    assert (stats.processed, stats.skipped, stats.errors) == (2, 0, 0)
    assert conn.execute("SELECT COUNT(*) FROM projects WHERE source_id = 'c/tiny'").fetchone()[0] == 0
    row = conn.execute("SELECT * FROM projects WHERE source_id = 'a/big'").fetchone()
    assert (row["source"], row["kind"], row["name"]) == ("huggingface", "model", "big")
    assert row["url"] == "https://huggingface.co/a/big"
    assert row["license_spdx"] == "apache-2.0"
    assert row["param_count"] == 7_000_000_000
    assert (row["downloads"], row["likes"]) == (5000, 10)
    assert row["last_commit_at"].startswith("2026-10-01")
    assert row["description"] == "This is a small language model tuned for customer support dialogue."
    assert row["readme_excerpt"].startswith("# Model A")
    topics = json.loads(row["topics"])
    assert topics[:2] == ["text-generation", "transformers"]
    assert not any(t.startswith(("region:", "arxiv:", "license:")) for t in topics)
    assert sess.calls[0] == "https://huggingface.co/a/big/raw/main/README.md"


def test_rerun_skips_fresh(tmp_path):
    conn, api, fetcher, sess = make(tmp_path, FakeApi(models=[model("a/big", 5000)]))
    hf.ingest(conn, api, fetcher, now_fn=lambda: NOW)
    n = len(sess.calls)
    stats = hf.ingest(conn, api, fetcher, now_fn=lambda: NOW)
    assert (stats.processed, stats.skipped) == (0, 1)
    assert len(sess.calls) == n


def test_limit_respected(tmp_path):
    conn, api, fetcher, _ = make(tmp_path, FakeApi(models=[model("a/1", 5000), model("a/2", 4000), model("a/3", 3000)]))
    stats = hf.ingest(conn, api, fetcher, limit=2, now_fn=lambda: NOW)
    assert stats.processed == 2


def test_datasets_get_their_own_namespace(tmp_path):
    api = FakeApi(models=[model("a/x", 5000)], datasets=[dataset("a/x", 4000)])
    conn, api, fetcher, _ = make(tmp_path, api)
    stats = hf.ingest(conn, api, fetcher, dataset_filters=["task_categories:question-answering"], now_fn=lambda: NOW)
    assert stats.processed == 2  # same repo id as a model and a dataset: both kept
    ds = conn.execute("SELECT * FROM projects WHERE kind = 'dataset'").fetchone()
    assert ds["source_id"] == "datasets/a/x"
    assert ds["url"] == "https://huggingface.co/datasets/a/x"
    assert ds["param_count"] is None
    assert ("datasets", {"filter": "task_categories:question-answering"}) in api.calls


def test_light_skips_card_requests(tmp_path):
    conn, api, fetcher, sess = make(tmp_path, FakeApi(models=[model("a/big", 5000)]))
    hf.ingest(conn, api, fetcher, light=True, now_fn=lambda: NOW)
    assert sess.calls == []
    row = conn.execute("SELECT description, readme_excerpt FROM projects").fetchone()
    assert row["description"] == "text-generation model by a"
    assert row["readme_excerpt"] is None


def test_private_and_disabled_skipped(tmp_path):
    api = FakeApi(models=[model("a/priv", 5000, private=True), model("a/off", 5000, disabled=True), model("a/ok", 5000)])
    conn, api, fetcher, _ = make(tmp_path, api)
    stats = hf.ingest(conn, api, fetcher, now_fn=lambda: NOW)
    assert stats.processed == 1
    assert conn.execute("SELECT source_id FROM projects").fetchone()[0] == "a/ok"


def test_expand_fallback_to_full(tmp_path):
    api = FakeApi(models=[model("a/big", 5000)], reject_expand=True)
    conn, api, fetcher, _ = make(tmp_path, api)
    stats = hf.ingest(conn, api, fetcher, now_fn=lambda: NOW)
    assert stats.processed == 1
    assert api.calls[0][1]["expand"] is not None and api.calls[1][1]["full"] is True


def test_missing_card_still_stores(tmp_path):
    conn, api, fetcher, _ = make(tmp_path, FakeApi(models=[model("a/big", 5000)]), lambda url: FakeResp(404))
    stats = hf.ingest(conn, api, fetcher, now_fn=lambda: NOW)
    assert stats.processed == 1
    assert conn.execute("SELECT readme_excerpt FROM projects").fetchone()[0] is None


def test_card_fetch_retries_on_429():
    sleeps = []
    responses = iter([FakeResp(429, headers={"Retry-After": "3"}), FakeResp(200, "hello card")])
    fetcher = hf.CardFetcher("", session=FakeSession(lambda url: next(responses)), sleep=sleeps.append)
    assert fetcher.fetch("a/b", "model") == "hello card"
    assert sleeps == [4]


def test_card_text_helpers():
    assert hf.strip_front_matter(CARD).startswith("# Model A")
    assert hf.summarize_card("# Title\n\n![img](x.png)\n\nShort.\n\nA real descriptive paragraph about the model.") \
        == "A real descriptive paragraph about the model."
    assert hf.summarize_card("# only a heading") is None


def test_unified_view_spans_sources(tmp_path):
    conn, api, fetcher, _ = make(tmp_path, FakeApi(models=[model("a/big", 5000)]))
    hf.ingest(conn, api, fetcher, now_fn=lambda: NOW)
    gh.upsert_project(conn, {
        "source": "github", "source_id": "o/r", "kind": "software", "name": "r", "url": "https://github.com/o/r",
        "stars": 900, "last_commit_at": "2026-11-01T00:00:00Z", "fetched_at": NOW.isoformat(),
    })
    rows = conn.execute(
        "SELECT source, kind, popularity, last_activity_at FROM projects_unified ORDER BY source").fetchall()
    assert [(r["source"], r["kind"], r["popularity"]) for r in rows] == [("github", "software", 900), ("huggingface", "model", 10)]
    assert all(r["last_activity_at"] for r in rows)
