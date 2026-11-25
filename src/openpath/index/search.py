"""Hybrid search: metadata filters -> semantic similarity -> keyword boost.

Usage:
    python -m openpath.index.search "extract text from PDF files" --kind software --language python -k 10
    python -m openpath.index.search --queries-file data/sample_queries.txt -k 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from openpath.config import get_settings
from openpath.db import init_db
from openpath.index.embedder import get_embedder, tokenize

log = logging.getLogger("openpath.index.search")

UNCLEAR_LICENSES = ("noassertion", "other", "none", "unknown")


@dataclass
class Filters:
    """Hard filters applied before scoring (empty / None means no restriction).

    languages applies to software only: models and datasets have no programming language and pass through.
    max_age_months drops projects whose last activity is older, and those with unknown activity.
    """
    kind: Optional[str] = None                    # software | model | dataset
    sources: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    licenses: list[str] = field(default_factory=list)  # allowlist, case-insensitive
    require_license: bool = False                 # drop missing / NOASSERTION / 'other'
    max_age_months: Optional[int] = None


@dataclass
class Hit:
    project: dict
    semantic: float
    keyword: float
    score: float


def _in(column: str, values: list[str]) -> tuple[str, list[str]]:
    return f"{column} IN ({', '.join('?' * len(values))})", [v.lower() for v in values]


def build_where(f: Filters, now: datetime) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if f.kind:
        clauses.append("p.kind = ?")
        params.append(f.kind)
    if f.sources:
        c, v = _in("lower(p.source)", f.sources)
        clauses.append(c); params += v
    if f.languages:
        c, v = _in("lower(p.language)", f.languages)
        clauses.append(f"(p.kind != 'software' OR {c})"); params += v
    if f.licenses:
        c, v = _in("lower(p.license_spdx)", f.licenses)
        clauses.append(c); params += v
    if f.require_license:
        c, v = _in("lower(p.license_spdx)", list(UNCLEAR_LICENSES))
        clauses.append(f"p.license_spdx IS NOT NULL AND NOT ({c})"); params += v
    if f.max_age_months:
        cutoff = (now - timedelta(days=30.44 * f.max_age_months)).date().isoformat()
        clauses.append("date(p.last_commit_at) >= date(?)")
        params.append(cutoff)
    return (" AND ".join(clauses) if clauses else "1=1"), params


def keyword_score(query_tokens: list[str], row: Any) -> float:
    """0..1: how many query words appear in the name (1.0), topics (0.7) or description (0.4)."""
    if not query_tokens:
        return 0.0
    topics = json.loads(row["topics"]) if row["topics"] else []
    name_t = set(tokenize(row["name"]))
    topic_t = set(tokenize(" ".join(topics)))
    desc_t = set(tokenize(row["description"] or ""))
    total = 0.0
    for t in query_tokens:
        total += 1.0 if t in name_t else 0.7 if t in topic_t else 0.4 if t in desc_t else 0.0
    return total / len(query_tokens)


class Searcher:
    def __init__(self, conn: sqlite3.Connection, embedder: Any, keyword_weight: float = 0.15):
        self.conn = conn
        self.embedder = embedder
        self.keyword_weight = keyword_weight

    def search(self, query: str, filters: Optional[Filters] = None, k: int = 10,
               now: Optional[datetime] = None) -> list[Hit]:
        where, params = build_where(filters or Filters(), now or datetime.now(timezone.utc))
        rows = self.conn.execute(
            "SELECT p.*, e.vector AS _vec FROM projects p "
            "JOIN embeddings e ON e.project_id = p.id AND e.model = ? "
            f"WHERE {where}",
            [self.embedder.name, *params],
        ).fetchall()
        if not rows:
            if self.conn.execute("SELECT 1 FROM embeddings WHERE model = ? LIMIT 1", (self.embedder.name,)).fetchone() is None:
                log.warning("No embeddings for model %r. Run: python -m openpath.index.build", self.embedder.name)
            return []
        matrix = np.vstack([np.frombuffer(r["_vec"], dtype=np.float32) for r in rows])
        semantic = matrix @ self.embedder.encode([query])[0]
        q_tokens = tokenize(query)
        keyword = np.array([keyword_score(q_tokens, r) for r in rows], dtype=np.float32)
        score = semantic + self.keyword_weight * keyword
        hits = []
        for i in np.argsort(-score)[:k]:
            project = {key: rows[i][key] for key in rows[i].keys() if key != "_vec"}
            hits.append(Hit(project, float(semantic[i]), float(keyword[i]), float(score[i])))
        return hits


def _print_hits(query: str, hits: list[Hit]) -> None:
    print(f"\n== {query}")
    if not hits:
        print("   (no results)")
    for h in hits:
        p = h.project
        desc = (p["description"] or "")[:70]
        print(f"   {h.score:.3f} (sem {h.semantic:.2f}, kw {h.keyword:.2f}) [{p['kind']}/{p['source']}] "
              f"{p['source_id']} | {p['license_spdx'] or '?'} | {desc}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Search the OpenPath index.")
    p.add_argument("query", nargs="?", help="search text")
    p.add_argument("--queries-file", help="run every non-empty line as a query")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--kind", choices=["software", "model", "dataset"])
    p.add_argument("--source", action="append", default=[], help="github / huggingface (repeatable)")
    p.add_argument("--language", action="append", default=[], help="software language (repeatable)")
    p.add_argument("--license", action="append", default=[], help="allowed license id (repeatable)")
    p.add_argument("--require-license", action="store_true")
    p.add_argument("--max-age-months", type=int)
    p.add_argument("--keyword-weight", type=float, default=0.15)
    p.add_argument("--embedder", default=None)
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)
    if not args.query and not args.queries_file:
        p.error("give a query or --queries-file")

    settings = get_settings()
    conn = init_db(args.db or settings.db_path)
    searcher = Searcher(conn, get_embedder(args.embedder or settings.embedder), args.keyword_weight)
    filters = Filters(args.kind, args.source, args.language, args.license, args.require_license, args.max_age_months)
    queries = [args.query] if args.query else []
    if args.queries_file:
        with open(args.queries_file, encoding="utf-8") as fh:
            queries += [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    for q in queries:
        _print_hits(q, searcher.search(q, filters, args.k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
