import json
from datetime import datetime, timezone

import pytest
from requests.structures import CaseInsensitiveDict

from openpath.db import init_db
from openpath.ingest import github as gh


class FakeResp:
    def __init__(self, status=200, body=None, headers=None, text=None):
        self.status_code = status
        self.headers = CaseInsensitiveDict(headers or {})
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")

    def json(self):
        return json.loads(self.text)


class FakeSession:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers or {}))
        return self.handler(url, params, headers or {}, len(self.calls))


def make_item(full, stars):
    return {
        "full_name": full, "name": full.split("/")[1], "html_url": f"https://github.com/{full}",
        "description": f"desc {full}", "license": {"spdx_id": "MIT"}, "language": "Python",
        "topics": ["machine-learning"], "stargazers_count": stars, "forks_count": 5,
        "open_issues_count": 3, "created_at": "2020-01-01T00:00:00Z", "pushed_at": "2026-10-01T00:00:00Z",
    }


ITEMS = [make_item("a/one", 500), make_item("a/two", 300)]
NOW = datetime(2026, 11, 24, tzinfo=timezone.utc)


def etagged(headers, make):
    if headers.get("If-None-Match") == '"v1"':
        return FakeResp(304)
    r = make()
    r.headers["ETag"] = '"v1"'
    return r


def backend(url, params, headers, n):
    path = url.replace(gh.API, "")
    if path == "/search/repositories":
        return FakeResp(body={"items": ITEMS if params["page"] == 1 else []})
    if path.endswith("/readme"):
        return etagged(headers, lambda: FakeResp(text="# Readme for " + path))
    if path.endswith("/commits"):
        return etagged(headers, lambda: FakeResp(body=[{"commit": {"committer": {"date": "2026-11-01T00:00:00Z"}}}]))
    if path.endswith("/contributors"):
        link = '<https://api.github.com/repositories/1/contributors?per_page=1&anon=true&page=42>; rel="last"'
        return etagged(headers, lambda: FakeResp(body=[{"login": "x"}], headers={"Link": link}))
    if path.endswith("/releases/latest"):
        if "/a/one/" in path:
            return FakeResp(404, body={"message": "Not Found"})
        return etagged(headers, lambda: FakeResp(body={"published_at": "2026-09-01T00:00:00Z"}))
    raise AssertionError("unexpected url " + url)


@pytest.fixture
def env(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    sess = FakeSession(backend)
    sleeps = []
    client = gh.GitHubClient("tok", conn, session=sess, sleep=sleeps.append, now=lambda: 1000.0, search_interval=0)
    return conn, sess, client, sleeps


def readme_calls(sess):
    return sum(1 for c in sess.calls if c[0].endswith("/readme"))


def test_ingest_stores_projects(env):
    conn, sess, client, _ = env
    stats = gh.ingest(conn, client, ["machine-learning"], limit=10, now_fn=lambda: NOW)
    assert (stats.processed, stats.skipped, stats.errors) == (2, 0, 0)
    one = conn.execute("SELECT * FROM projects WHERE source_id = 'a/one'").fetchone()
    assert one["license_spdx"] == "MIT"
    assert one["contributors"] == 42
    assert one["last_commit_at"] == "2026-11-01T00:00:00Z"
    assert one["last_release_at"] is None
    assert one["readme_excerpt"].startswith("# Readme")
    assert json.loads(one["topics"]) == ["machine-learning"]
    two = conn.execute("SELECT * FROM projects WHERE source_id = 'a/two'").fetchone()
    assert two["last_release_at"] == "2026-09-01T00:00:00Z"


def test_rerun_skips_fresh_repos(env):
    conn, sess, client, _ = env
    gh.ingest(conn, client, ["machine-learning"], now_fn=lambda: NOW)
    before = readme_calls(sess)
    stats = gh.ingest(conn, client, ["machine-learning"], now_fn=lambda: NOW)
    assert (stats.processed, stats.skipped) == (0, 2)
    assert readme_calls(sess) == before


def test_stale_rerun_uses_etag_cache(env):
    conn, sess, client, _ = env
    gh.ingest(conn, client, ["machine-learning"], now_fn=lambda: NOW)
    stats = gh.ingest(conn, client, ["machine-learning"], max_age_days=0, now_fn=lambda: NOW)
    assert stats.processed == 2
    second_readme = [c for c in sess.calls if c[0].endswith("/readme")][-1]
    assert second_readme[2].get("If-None-Match") == '"v1"'
    row = conn.execute("SELECT readme_excerpt, contributors FROM projects WHERE source_id = 'a/one'").fetchone()
    assert row["readme_excerpt"].startswith("# Readme")  # served from cache on 304
    assert row["contributors"] == 42


def test_limit_is_respected(env):
    conn, _, client, _ = env
    stats = gh.ingest(conn, client, ["machine-learning"], limit=1, now_fn=lambda: NOW)
    assert stats.processed == 1
    assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1


def test_light_mode_skips_extra_calls(env):
    conn, sess, client, _ = env
    gh.ingest(conn, client, ["machine-learning"], light=True, now_fn=lambda: NOW)
    assert not any(c[0].endswith(("/commits", "/contributors", "/releases/latest")) for c in sess.calls)
    row = conn.execute("SELECT last_commit_at FROM projects WHERE source_id = 'a/one'").fetchone()
    assert row["last_commit_at"] == "2026-10-01T00:00:00Z"  # pushed_at fallback


def test_optional_failure_does_not_abort(tmp_path):
    def handler(url, params, headers, n):
        if url.endswith("/contributors"):
            return FakeResp(403, text="The history or contributor list is too large",
                            headers={"X-RateLimit-Remaining": "4000"})
        return backend(url, params, headers, n)

    conn = init_db(str(tmp_path / "t.db"))
    client = gh.GitHubClient("tok", conn, session=FakeSession(handler), sleep=lambda s: None,
                             now=lambda: 1000.0, search_interval=0)
    stats = gh.ingest(conn, client, ["machine-learning"], now_fn=lambda: NOW)
    assert stats.processed == 2 and stats.errors == 0
    row = conn.execute("SELECT contributors FROM projects WHERE source_id = 'a/one'").fetchone()
    assert row["contributors"] is None


def make_client(tmp_path, responses):
    conn = init_db(str(tmp_path / "t.db"))
    it = iter(responses)
    sess = FakeSession(lambda *a: next(it))
    sleeps = []
    client = gh.GitHubClient("tok", conn, session=sess, sleep=sleeps.append, now=lambda: 1000.0, search_interval=0)
    return client, sleeps


def test_rate_limit_waits_until_reset(tmp_path):
    limited = FakeResp(403, text="API rate limit exceeded",
                       headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1030"})
    client, sleeps = make_client(tmp_path, [limited, FakeResp(body={"ok": 1})])
    assert client.request("/rate").ok
    assert sleeps == [31]


def test_forbidden_without_rate_limit_raises(tmp_path):
    client, _ = make_client(tmp_path, [FakeResp(403, text="nope", headers={"X-RateLimit-Remaining": "4000"})])
    with pytest.raises(gh.GitHubError):
        client.request("/private")


def test_server_error_backoff(tmp_path):
    client, sleeps = make_client(tmp_path, [FakeResp(500, text="oops"), FakeResp(body={"ok": 1})])
    assert client.request("/flaky").ok
    assert sleeps == [2]


def test_last_page_from_link():
    link = '<https://api.github.com/x?per_page=1&page=2>; rel="next", <https://api.github.com/x?per_page=1&page=17>; rel="last"'
    assert gh.last_page_from_link(link) == 17
    assert gh.last_page_from_link("") is None
