"""Build or update the embedding index.

Usage:
    python -m openpath.index.build                    # default embedder from OPENPATH_EMBEDDER
    python -m openpath.index.build --embedder hashing # offline baseline
    python -m openpath.index.build --rebuild          # re-embed everything
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from openpath.config import get_settings
from openpath.db import init_db
from openpath.index.embedder import get_embedder

log = logging.getLogger("openpath.index.build")


@dataclass
class BuildStats:
    total: int = 0
    embedded: int = 0
    skipped: int = 0


def project_text(row: Any) -> str:
    """The text that represents a project in the vector space."""
    topics = json.loads(row["topics"]) if row["topics"] else []
    parts = [row["name"]]
    if row["description"]:
        parts.append(row["description"])
    parts.append(f"Type: {row['kind']}")
    if row["language"]:
        parts.append(f"Language: {row['language']}")
    if topics:
        parts.append("Topics: " + ", ".join(topics[:10]))
    if row["readme_excerpt"]:
        parts.append(" ".join(row["readme_excerpt"].split())[:1200])
    return "\n".join(parts)


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_index(conn: sqlite3.Connection, embedder: Any, batch_size: int = 64, rebuild: bool = False) -> BuildStats:
    """Embed projects that are new, changed, or embedded with a different model."""
    rows = conn.execute(
        "SELECT p.*, e.model AS e_model, e.text_hash AS e_hash "
        "FROM projects p LEFT JOIN embeddings e ON e.project_id = p.id"
    ).fetchall()
    stats = BuildStats(total=len(rows))
    todo = []
    for row in rows:
        text = project_text(row)
        digest = _hash(text)
        if not rebuild and row["e_model"] == embedder.name and row["e_hash"] == digest:
            stats.skipped += 1
        else:
            todo.append((row["id"], text, digest))

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        vectors = embedder.encode([t for _, t, _ in batch])
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            """INSERT INTO embeddings (project_id, model, vector, dim, text_hash, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(project_id) DO UPDATE SET
                 model = excluded.model, vector = excluded.vector, dim = excluded.dim,
                 text_hash = excluded.text_hash, updated_at = excluded.updated_at""",
            [(pid, embedder.name, vec.astype("float32").tobytes(), int(vec.shape[0]), digest, now)
             for (pid, _, digest), vec in zip(batch, vectors)],
        )
        conn.commit()
        stats.embedded += len(batch)
        log.info("embedded %d/%d", stats.embedded, len(todo))
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build the OpenPath embedding index.")
    p.add_argument("--embedder", default=None, help="'hashing' or a sentence-transformers model name")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = get_settings()
    conn = init_db(args.db or settings.db_path)
    embedder = get_embedder(args.embedder or settings.embedder)
    stats = build_index(conn, embedder, args.batch_size, args.rebuild)
    log.info("Done (%s). total=%d embedded=%d skipped=%d", embedder.name, stats.total, stats.embedded, stats.skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
