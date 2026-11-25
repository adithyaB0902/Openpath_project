"""Hugging Face ingestion: models (and optionally datasets) into the unified projects table.

Usage:
    python -m openpath.ingest.huggingface --tasks text-generation,text-classification --limit 500
    python -m openpath.ingest.huggingface --datasets-filter task_categories:question-answering --limit 100
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional

import requests
from huggingface_hub import HfApi

from openpath.config import get_settings
from openpath.db import init_db
from openpath.ingest.common import is_fresh, upsert_project

log = logging.getLogger("openpath.ingest.huggingface")

HF = "https://huggingface.co"
CARD_CHARS = 3000
SOURCE = "huggingface"
MODEL_EXPAND = [
    "downloads", "likes", "tags", "pipeline_tag", "library_name", "createdAt",
    "lastModified", "safetensors", "cardData", "private", "disabled", "gated",
]
DATASET_EXPAND = [
    "downloads", "likes", "tags", "createdAt", "lastModified", "cardData",
    "private", "disabled", "gated",
]
NOISE_TAG_PREFIXES = ("license:", "region:", "arxiv:", "doi:", "base_model:", "dataset:")
_SENTINEL = object()


@dataclass
class Stats:
    processed: int = 0
    skipped: int = 0
    errors: int = 0


# -- extracting fields from huggingface_hub objects --------------------
def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def license_from(info: Any) -> Optional[str]:
    """Hub license slug (e.g. apache-2.0, mit, llama3, other). Not always an SPDX id."""
    for tag in _get(info, "tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1].lower()
    lic = _get(_get(info, "card_data"), "license")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    return str(lic).lower() if lic else None


def param_count_from(info: Any) -> Optional[int]:
    total = _get(_get(info, "safetensors"), "total")
    return int(total) if total else None


def topics_from(info: Any) -> list[str]:
    out: list[str] = []
    for value in (_get(info, "pipeline_tag"), _get(info, "library_name")):
        if value and value not in out:
            out.append(value)
    for tag in _get(info, "tags") or []:
        if isinstance(tag, str) and not tag.startswith(NOISE_TAG_PREFIXES) and tag not in out:
            out.append(tag)
    return out[:15]


# -- model card text ---------------------------------------------------
def strip_front_matter(text: str) -> str:
    return re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.S)


def summarize_card(text: str) -> Optional[str]:
    """First real prose paragraph of a card, up to 300 chars."""
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if len(para) < 30 or para.startswith(("#", "<", "!", "[", "|", "```", "-", "*", ">")):
            continue
        return para[:300]
    return None


class CardFetcher:
    """Fetch README.md for a model or dataset; failures return None (card is optional)."""

    def __init__(self, token: str = "", session: Any = None, sleep: Callable[[float], None] = time.sleep,
                 max_retries: int = 3):
        self.session = session or requests.Session()
        self.sleep = sleep
        self.max_retries = max_retries
        self.headers = {"User-Agent": "openpath-ingest"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    def fetch(self, repo_id: str, kind: str) -> Optional[str]:
        prefix = "" if kind == "model" else "datasets/"
        url = f"{HF}/{prefix}{repo_id}/raw/main/README.md"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, headers=self.headers, timeout=30)
            except requests.RequestException:
                self.sleep(min(2 ** attempt, 60))
                continue
            sc = resp.status_code
            if sc == 200:
                return resp.text
            if sc == 429:
                ra = resp.headers.get("Retry-After")
                self.sleep(min(int(ra) + 1, 300) if ra and str(ra).isdigit() else min(2 ** attempt, 60))
                continue
            if sc >= 500:
                self.sleep(min(2 ** attempt, 60))
                continue
            return None  # 404 (no card), 401/403 (gated/private), anything else
        log.warning("giving up on card for %s", repo_id)
        return None


# -- listing -----------------------------------------------------------
def list_infos(api: Any, kind: str, selector: Optional[str], scan_limit: int) -> Iterator[Any]:
    """Yield infos sorted by downloads (highest first).

    Uses `expand` (lean responses) when the installed huggingface_hub supports it,
    otherwise falls back to full=True.
    """
    fn = api.list_models if kind == "model" else api.list_datasets
    params = inspect.signature(fn).parameters

    def kwargs(use_expand: bool) -> dict:
        kw: dict = {"sort": "downloads", "limit": scan_limit}
        if "direction" in params:  # older versions need an explicit descending order
            kw["direction"] = -1
        if use_expand and "expand" in params:
            kw["expand"] = MODEL_EXPAND if kind == "model" else DATASET_EXPAND
        else:
            kw["full"] = True
            if kind == "model" and "cardData" in params:
                kw["cardData"] = True
        if selector:
            kw["pipeline_tag" if kind == "model" else "filter"] = selector
        return kw

    try:
        it = iter(fn(**kwargs(True)))
        first = next(it, _SENTINEL)
    except Exception as exc:  # unsupported expand value on this version, etc.
        log.warning("expand listing failed (%s); retrying with full=True", exc)
        it = iter(fn(**kwargs(False)))
        first = next(it, _SENTINEL)
    if first is not _SENTINEL:
        yield first
        yield from it


# -- record building ---------------------------------------------------
def build_record(info: Any, kind: str, card_text: Optional[str], fetched_at: str) -> dict:
    repo_id = _get(info, "id")
    author = _get(info, "author") or (repo_id.split("/")[0] if "/" in repo_id else None)
    pipeline = _get(info, "pipeline_tag")
    excerpt = strip_front_matter(card_text)[:CARD_CHARS] if card_text else None
    description = summarize_card(excerpt) if excerpt else None
    if not description:
        description = f"{pipeline or 'general'} {kind}" + (f" by {author}" if author else "")
    is_model = kind == "model"
    return {
        "source": SOURCE,
        "source_id": repo_id if is_model else f"datasets/{repo_id}",
        "kind": kind,
        "name": repo_id.split("/")[-1],
        "url": f"{HF}/{repo_id}" if is_model else f"{HF}/datasets/{repo_id}",
        "description": description,
        "readme_excerpt": excerpt,
        "license_spdx": license_from(info),
        "language": None,
        "topics": json.dumps(topics_from(info)),
        "stars": None,
        "forks": None,
        "open_issues": None,
        "contributors": None,
        "downloads": _get(info, "downloads"),
        "likes": _get(info, "likes"),
        "param_count": param_count_from(info) if is_model else None,
        "created_at": _iso(_get(info, "created_at")),
        "last_commit_at": _iso(_get(info, "last_modified")),
        "last_release_at": None,
        "fetched_at": fetched_at,
    }


def ingest(
    conn: sqlite3.Connection,
    api: Any,
    fetcher: CardFetcher,
    tasks: Optional[list[str]] = None,
    dataset_filters: Optional[list[str]] = None,
    include_models: bool = True,
    limit: int = 500,
    min_downloads: int = 1000,
    max_age_days: int = 7,
    scan_limit: int = 5000,
    light: bool = False,
    now_fn: Optional[Callable[[], datetime]] = None,
) -> Stats:
    """Ingest models (per pipeline task) and datasets (per filter), most downloaded first.

    `limit` caps items fetched this run; fresh items are skipped and do not count toward it.
    """
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    jobs: list[tuple[str, Optional[str]]] = []
    if include_models:
        jobs += [("model", t) for t in (tasks or [None])]
    jobs += [("dataset", f) for f in (dataset_filters or [])]

    stats, seen = Stats(), set()
    for kind, selector in jobs:
        for info in list_infos(api, kind, selector, scan_limit):
            if stats.processed >= limit:
                return stats
            repo_id = _get(info, "id")
            if not repo_id or (kind, repo_id) in seen:
                continue
            seen.add((kind, repo_id))
            if _get(info, "private") or _get(info, "disabled"):
                continue
            if (_get(info, "downloads") or 0) < min_downloads:
                break  # sorted by downloads, so the rest of this job is below the threshold
            source_id = repo_id if kind == "model" else f"datasets/{repo_id}"
            if is_fresh(conn, SOURCE, source_id, max_age_days, now_fn()):
                stats.skipped += 1
                continue
            try:
                card = None if light else fetcher.fetch(repo_id, kind)
                upsert_project(conn, build_record(info, kind, card, now_fn().isoformat()))
                stats.processed += 1
                log.info("[%d/%d] stored %s %s", stats.processed, limit, kind, repo_id)
            except sqlite3.Error as exc:
                stats.errors += 1
                log.error("failed %s: %s", repo_id, exc)
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ingest Hugging Face models/datasets into the OpenPath database.")
    p.add_argument("--tasks", default="", help="comma-separated pipeline tags, e.g. text-generation,text-classification")
    p.add_argument("--datasets-filter", default="", help="comma-separated dataset filters, e.g. task_categories:question-answering")
    p.add_argument("--no-models", action="store_true", help="only ingest datasets")
    p.add_argument("--limit", type=int, default=500, help="max items to fetch this run")
    p.add_argument("--min-downloads", type=int, default=1000, help="stop a listing below this download count")
    p.add_argument("--max-age-days", type=int, default=7)
    p.add_argument("--scan-limit", type=int, default=5000, help="max listing entries to scan per task")
    p.add_argument("--light", action="store_true", help="skip model card fetch (faster, shorter descriptions)")
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    if not settings.hf_token:
        log.warning("No HF_TOKEN set: anonymous requests have lower rate limits.")
    conn = init_db(args.db or settings.db_path)
    api = HfApi(token=settings.hf_token or None)
    fetcher = CardFetcher(settings.hf_token)
    stats = ingest(
        conn, api, fetcher,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()] or None,
        dataset_filters=[f.strip() for f in args.datasets_filter.split(",") if f.strip()] or None,
        include_models=not args.no_models,
        limit=args.limit,
        min_downloads=args.min_downloads,
        max_age_days=args.max_age_days,
        scan_limit=args.scan_limit,
        light=args.light,
    )
    log.info("Done. stored=%d skipped=%d errors=%d", stats.processed, stats.skipped, stats.errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
