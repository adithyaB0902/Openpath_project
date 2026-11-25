"""GitHub ingestion: search repositories and store normalized metadata.

Usage:
    python -m openpath.ingest.github --topics machine-learning,self-hosted --limit 500
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import requests

from openpath.config import get_settings
from openpath.db import init_db
from openpath.ingest.common import is_fresh, upsert_project

log = logging.getLogger("openpath.ingest.github")

API = "https://api.github.com"
README_CHARS = 3000
CACHE_BODY_CHARS = 60000
MAX_WAIT_SECONDS = 3700
RAW_ACCEPT = "application/vnd.github.raw+json"


class GitHubError(Exception):
    """Raised for unrecoverable API errors."""


@dataclass
class Result:
    status: int
    data: Any = None
    link: str = ""
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return self.status == 200


@dataclass
class Stats:
    processed: int = 0
    skipped: int = 0
    errors: int = 0


class GitHubClient:
    """Thin GitHub REST client with ETag caching, rate-limit waits and retries."""

    def __init__(
        self,
        token: str,
        conn: sqlite3.Connection,
        session: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.time,
        max_retries: int = 5,
        search_interval: Optional[float] = None,
    ):
        self.conn = conn
        self.session = session or requests.Session()
        self.sleep = sleep
        self.now = now
        self.max_retries = max_retries
        # Search API limits: 30/min authenticated, 10/min anonymous.
        self.search_interval = search_interval if search_interval is not None else (2.2 if token else 6.5)
        self._last_search = 0.0
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "openpath-ingest",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    # -- cache helpers -------------------------------------------------
    @staticmethod
    def _cache_key(url: str, params: Optional[dict], accept: Optional[str]) -> str:
        return json.dumps([url, sorted((params or {}).items()), accept])

    def _cache_get(self, key: str):
        return self.conn.execute(
            "SELECT etag, body, link FROM http_cache WHERE cache_key = ?", (key,)
        ).fetchone()

    def _cache_put(self, key: str, etag: str, body: str, link: str) -> None:
        self.conn.execute(
            """INSERT INTO http_cache (cache_key, etag, body, link, fetched_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(cache_key) DO UPDATE SET
                 etag = excluded.etag, body = excluded.body,
                 link = excluded.link, fetched_at = excluded.fetched_at""",
            (key, etag, body, link, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()

    @staticmethod
    def _decode(text: str, accept: Optional[str]) -> Any:
        if accept == RAW_ACCEPT:
            return text
        return json.loads(text) if text else None

    # -- rate limits ---------------------------------------------------
    def _throttle_search(self) -> None:
        wait = self._last_search + self.search_interval - self.now()
        if wait > 0:
            self.sleep(wait)
        self._last_search = self.now()

    def _rate_limit_wait(self, resp: Any) -> Optional[float]:
        """Seconds to wait if the response is a rate limit, else None."""
        h = resp.headers
        retry_after = h.get("Retry-After")
        if retry_after and str(retry_after).isdigit():
            return min(int(retry_after) + 1, MAX_WAIT_SECONDS)
        if h.get("X-RateLimit-Remaining") == "0":
            reset = h.get("X-RateLimit-Reset")
            if reset and str(reset).isdigit():
                return min(max(int(reset) - self.now(), 1) + 1, MAX_WAIT_SECONDS)
            return 60
        if "secondary rate limit" in (resp.text or "").lower():
            return 60
        return None

    # -- main request --------------------------------------------------
    def request(
        self,
        path: str,
        params: Optional[dict] = None,
        accept: Optional[str] = None,
        search: bool = False,
    ) -> Result:
        url = path if path.startswith("http") else API + path
        key = self._cache_key(url, params, accept)
        cached = None if search else self._cache_get(key)
        headers: dict = {}
        if accept:
            headers["Accept"] = accept
        if cached and cached["etag"]:
            headers["If-None-Match"] = cached["etag"]

        attempt = 0
        while True:
            if search:
                self._throttle_search()
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=30)
            except requests.RequestException as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise GitHubError(f"network error for {url}: {exc}") from exc
                self.sleep(min(2 ** attempt, 60))
                continue

            sc = resp.status_code
            if sc == 304 and cached:
                return Result(200, self._decode(cached["body"], accept), cached["link"] or "", True)
            if sc == 200:
                text = resp.text
                link = resp.headers.get("Link", "") or ""
                etag = resp.headers.get("ETag")
                if etag and not search:
                    body = text[:CACHE_BODY_CHARS] if accept == RAW_ACCEPT else text
                    self._cache_put(key, etag, body, link)
                return Result(200, self._decode(text, accept), link)
            if sc in (204, 404, 409, 410, 422, 451):
                return Result(sc)
            if sc in (403, 429):
                wait = self._rate_limit_wait(resp)
                if wait is None:
                    raise GitHubError(f"HTTP {sc} for {url}: {(resp.text or '')[:200]}")
                attempt += 1
                if attempt > self.max_retries:
                    raise GitHubError(f"rate limit retries exhausted for {url}")
                log.warning("Rate limited (HTTP %s). Sleeping %.0fs", sc, wait)
                self.sleep(wait)
                continue
            if sc >= 500:
                attempt += 1
                if attempt > self.max_retries:
                    raise GitHubError(f"HTTP {sc} for {url} after retries")
                self.sleep(min(2 ** attempt, 60))
                continue
            raise GitHubError(f"HTTP {sc} for {url}: {(resp.text or '')[:200]}")


# -- parsing helpers ---------------------------------------------------
def last_page_from_link(link: str) -> Optional[int]:
    m = re.search(r'<[^>]*[?&]page=(\d+)[^>]*>;\s*rel="last"', link or "")
    return int(m.group(1)) if m else None


def search_page(client: GitHubClient, topic: str, language: Optional[str], min_stars: int, page: int) -> list:
    q = f"topic:{topic} stars:>={min_stars} archived:false fork:false"
    if language:
        q += f" language:{language}"
    r = client.request(
        "/search/repositories",
        params={"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page},
        search=True,
    )
    if not r.ok or not r.data:
        return []
    return r.data.get("items", [])


def fetch_readme(client: GitHubClient, full: str) -> Optional[str]:
    r = client.request(f"/repos/{full}/readme", accept=RAW_ACCEPT)
    if r.ok and isinstance(r.data, str):
        return r.data[:README_CHARS]
    return None


def fetch_last_commit(client: GitHubClient, full: str) -> Optional[str]:
    r = client.request(f"/repos/{full}/commits", params={"per_page": 1})
    if r.ok and r.data:
        commit = r.data[0].get("commit", {})
        who = commit.get("committer") or commit.get("author") or {}
        return who.get("date")
    return None


def fetch_last_release(client: GitHubClient, full: str) -> Optional[str]:
    r = client.request(f"/repos/{full}/releases/latest")
    if r.ok and r.data:
        return r.data.get("published_at") or r.data.get("created_at")
    return None


def fetch_contributors(client: GitHubClient, full: str) -> Optional[int]:
    r = client.request(f"/repos/{full}/contributors", params={"per_page": 1, "anon": "true"})
    if r.status == 204:
        return 0
    if r.ok:
        last = last_page_from_link(r.link)
        return last if last else len(r.data or [])
    return None


def _safe(fn: Callable, *args):
    """Optional enrichment must never abort a repo."""
    try:
        return fn(*args)
    except GitHubError as exc:
        log.warning("optional fetch %s failed: %s", fn.__name__, exc)
        return None


def enrich(client: GitHubClient, item: dict, light: bool) -> dict:
    full = item["full_name"]
    extra = {"readme": _safe(fetch_readme, client, full)}
    if light:
        extra.update(last_commit=item.get("pushed_at"), last_release=None, contributors=None)
    else:
        extra["last_commit"] = _safe(fetch_last_commit, client, full) or item.get("pushed_at")
        extra["last_release"] = _safe(fetch_last_release, client, full)
        extra["contributors"] = _safe(fetch_contributors, client, full)
    return extra


def build_record(item: dict, extra: dict, fetched_at: str) -> dict:
    return {
        "source": "github",
        "source_id": item["full_name"],
        "kind": "software",
        "name": item.get("name") or item["full_name"].split("/")[-1],
        "url": item["html_url"],
        "description": item.get("description"),
        "readme_excerpt": extra.get("readme"),
        "license_spdx": (item.get("license") or {}).get("spdx_id"),
        "language": item.get("language"),
        "topics": json.dumps(item.get("topics") or []),
        "stars": item.get("stargazers_count"),
        "forks": item.get("forks_count"),
        "open_issues": item.get("open_issues_count"),
        "contributors": extra.get("contributors"),
        "created_at": item.get("created_at"),
        "last_commit_at": extra.get("last_commit"),
        "last_release_at": extra.get("last_release"),
        "fetched_at": fetched_at,
    }


def ingest(
    conn: sqlite3.Connection,
    client: GitHubClient,
    topics: list[str],
    languages: Optional[list[str]] = None,
    limit: int = 500,
    min_stars: int = 50,
    max_age_days: int = 7,
    light: bool = False,
    now_fn: Optional[Callable[[], datetime]] = None,
) -> Stats:
    """Crawl topics (x languages). `limit` caps repos fetched this run; fresh repos are skipped.

    Each repo is committed as soon as it is fetched, so an interrupted run loses nothing
    and a rerun continues where it stopped.
    """
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    stats, seen = Stats(), set()
    for topic in topics:
        for lang in languages or [None]:
            for page in range(1, 11):  # search API exposes at most 1000 results per query
                if stats.processed >= limit:
                    return stats
                items = search_page(client, topic, lang, min_stars, page)
                if not items:
                    break
                for item in items:
                    if stats.processed >= limit:
                        return stats
                    full = item["full_name"]
                    if full in seen:
                        continue
                    seen.add(full)
                    if is_fresh(conn, "github", full, max_age_days, now_fn()):
                        stats.skipped += 1
                        continue
                    try:
                        extra = enrich(client, item, light)
                        upsert_project(conn, build_record(item, extra, now_fn().isoformat()))
                        stats.processed += 1
                        log.info("[%d/%d] stored %s", stats.processed, limit, full)
                    except GitHubError as exc:
                        stats.errors += 1
                        log.error("failed %s: %s", full, exc)
                if len(items) < 100:
                    break
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ingest GitHub repositories into the OpenPath database.")
    p.add_argument("--topics", required=True, help="comma-separated GitHub topics")
    p.add_argument("--languages", default="", help="optional comma-separated languages")
    p.add_argument("--limit", type=int, default=500, help="max repos to fetch this run")
    p.add_argument("--min-stars", type=int, default=50)
    p.add_argument("--max-age-days", type=int, default=7, help="skip repos fetched more recently than this")
    p.add_argument("--light", action="store_true", help="skip commits, releases, contributors (fewer API calls)")
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    if not settings.github_token:
        log.warning("No GITHUB_TOKEN set: anonymous limits are very low (60 requests/hour).")
    conn = init_db(args.db or settings.db_path)
    client = GitHubClient(settings.github_token, conn)
    stats = ingest(
        conn,
        client,
        [t.strip() for t in args.topics.split(",") if t.strip()],
        [l.strip() for l in args.languages.split(",") if l.strip()] or None,
        limit=args.limit,
        min_stars=args.min_stars,
        max_age_days=args.max_age_days,
        light=args.light,
    )
    log.info("Done. stored=%d skipped=%d errors=%d", stats.processed, stats.skipped, stats.errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
