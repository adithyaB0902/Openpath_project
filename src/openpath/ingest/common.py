"""Helpers shared by all ingestion modules."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta


def upsert_project(conn: sqlite3.Connection, rec: dict) -> None:
    """Insert or update one row keyed by (source, source_id). Commits immediately."""
    cols = list(rec)
    placeholders = ", ".join(f":{c}" for c in cols)
    keep_old = {"readme_excerpt", "contributors", "last_release_at", "param_count"}
    updates = ", ".join(
        f"{c} = COALESCE(excluded.{c}, projects.{c})" if c in keep_old else f"{c} = excluded.{c}"
        for c in cols
        if c not in ("source", "source_id")
    )
    conn.execute(
        f"INSERT INTO projects ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT(source, source_id) DO UPDATE SET {updates}",
        rec,
    )
    conn.commit()


def is_fresh(conn: sqlite3.Connection, source: str, source_id: str, max_age_days: int, now: datetime) -> bool:
    """True if the project was fetched within max_age_days (so it can be skipped)."""
    if max_age_days <= 0:
        return False
    row = conn.execute(
        "SELECT fetched_at FROM projects WHERE source = ? AND source_id = ?", (source, source_id)
    ).fetchone()
    if not row:
        return False
    return datetime.fromisoformat(row["fetched_at"]) > now - timedelta(days=max_age_days)
