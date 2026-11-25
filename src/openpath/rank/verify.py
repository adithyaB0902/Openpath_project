"""Live verification: re-check a candidate against its source right before showing it."""
from __future__ import annotations

import inspect
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from openpath.ingest.github import GitHubClient, GitHubError
from openpath.ingest.huggingface import DATASET_EXPAND, MODEL_EXPAND, _get, _iso, license_from, param_count_from
from openpath.scoring.confidence import parse_dt
from openpath.scoring.health import compute_health, store_health

log = logging.getLogger("openpath.rank.verify")


@dataclass
class Verification:
    status: str                       # verified | gone | archived | error
    project: Optional[dict] = None    # refreshed project row (when verified)
    changes: list = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status, "changes": list(self.changes), "reason": self.reason}


class LiveVerifier:
    """GitHub projects are re-fetched through GitHubClient, Hugging Face ones through an HfApi-like object.

    verified: data refreshed (and stored), changes listed.  gone / archived: should be dropped.
    error: the source could not be reached, so the cached data is used and labelled as cached.
    """

    def __init__(self, conn: sqlite3.Connection, github: Optional[GitHubClient] = None, hf_api: Any = None):
        self.conn, self.github, self.hf_api = conn, github, hf_api

    def verify(self, project: dict, now: datetime) -> Verification:
        try:
            if project["source"] == "github" and self.github is not None:
                return self._github(project, now)
            if project["source"] == "huggingface" and self.hf_api is not None:
                return self._huggingface(project, now)
        except Exception as exc:  # network trouble must never break a search
            log.warning("could not verify %s: %s", project.get("source_id"), exc)
            return Verification("error", reason=f"Could not reach the source to verify ({exc.__class__.__name__}); showing saved data.")
        return Verification("error", reason="No live checker is configured for this source; showing saved data.")

    # -- GitHub --------------------------------------------------------
    def _github(self, p: dict, now: datetime) -> Verification:
        try:
            r = self.github.request(f"/repos/{p['source_id']}")
        except GitHubError as exc:
            return Verification("error", reason=f"GitHub could not be checked ({exc}); showing saved data.")
        if r.status in (404, 410, 451):
            return Verification("gone", reason="This repository was not found on GitHub (deleted, renamed, or made private).")
        if not r.ok or not isinstance(r.data, dict):
            return Verification("error", reason=f"GitHub returned an unexpected response (HTTP {r.status}); showing saved data.")
        d = r.data
        if d.get("archived") or d.get("disabled"):
            return Verification("archived", reason="The repository is archived or disabled, so it is read-only and no longer maintained.")
        new_license = (d.get("license") or {}).get("spdx_id")
        updates = {
            "description": d.get("description"), "language": d.get("language"),
            "stars": d.get("stargazers_count"), "forks": d.get("forks_count"), "open_issues": d.get("open_issues_count"),
            "license_spdx": new_license,
        }
        changes = []
        if (new_license or None) != (p["license_spdx"] or None):
            changes.append(f"License changed from {p['license_spdx'] or 'none'} to {new_license or 'none'} since our last crawl.")
        new_url = d.get("html_url")
        if new_url and new_url.rstrip("/").lower() != p["url"].rstrip("/").lower():
            changes.append(f"The repository moved to {d.get('full_name')}.")
            updates["url"] = new_url
        pushed, stored = parse_dt(d.get("pushed_at")), parse_dt(p["last_commit_at"])
        if pushed and (not stored or pushed > stored):
            updates["last_commit_at"] = d["pushed_at"]
        return self._store(p, updates, changes, now)

    # -- Hugging Face --------------------------------------------------
    def _huggingface(self, p: dict, now: datetime) -> Verification:
        is_dataset = p["kind"] == "dataset"
        repo_id = p["source_id"].removeprefix("datasets/")
        fn = self.hf_api.dataset_info if is_dataset else self.hf_api.model_info
        kwargs = {"expand": DATASET_EXPAND if is_dataset else MODEL_EXPAND} if "expand" in inspect.signature(fn).parameters else {}
        try:
            info = fn(repo_id, **kwargs)
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) in (401, 404) or exc.__class__.__name__ == "RepositoryNotFoundError":
                return Verification("gone", reason="This repository was not found on Hugging Face (deleted, renamed, or made private).")
            raise
        if _get(info, "private") or _get(info, "disabled"):
            return Verification("gone", reason="This repository is no longer publicly available on Hugging Face.")
        new_license = license_from(info)
        updates = {"downloads": _get(info, "downloads"), "likes": _get(info, "likes"),
                   "last_commit_at": _iso(_get(info, "last_modified")) or p["last_commit_at"],
                   "license_spdx": new_license or p["license_spdx"]}
        if not is_dataset and param_count_from(info):
            updates["param_count"] = param_count_from(info)
        changes = []
        if new_license and new_license != (p["license_spdx"] or "").lower():
            changes.append(f"License changed from {p['license_spdx'] or 'none'} to {new_license} since our last crawl.")
        return self._store(p, updates, changes, now)

    # -- shared --------------------------------------------------------
    def _store(self, p: dict, updates: dict, changes: list, now: datetime) -> Verification:
        updates = {k: v for k, v in updates.items() if v is not None or k in ("language",)}
        fresh = {**p, **updates, "fetched_at": now.isoformat()}
        cols = list(updates) + ["fetched_at"]
        self.conn.execute(f"UPDATE projects SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
                          [fresh[c] for c in cols] + [p["id"]])
        store_health(self.conn, p["id"], compute_health(fresh, now), now)  # stored as 'cached'; live status is per request
        self.conn.commit()
        fresh["verified_at"] = now.isoformat()
        return Verification("verified", fresh, changes)
