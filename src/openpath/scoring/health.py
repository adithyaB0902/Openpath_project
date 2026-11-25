"""Health score (0-100) with visible evidence, license class and confidence label.

Usage:
    python -m openpath.scoring.health                 # score new or changed projects
    python -m openpath.scoring.health --rebuild       # rescore everything
    python -m openpath.scoring.health --show owner/repo
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from openpath.config import get_settings
from openpath.db import init_db
from openpath.scoring.confidence import Confidence, confidence_label, parse_dt
from openpath.scoring.config import load_config
from openpath.scoring.licenses import LicenseInfo, classify

log = logging.getLogger("openpath.scoring.health")

REQUIRED = {
    "software": ["recency", "issue_load", "contributor_spread", "documentation", "adoption"],
    "model": ["recency", "documentation", "adoption"],
    "dataset": ["recency", "documentation", "adoption"],
}
LABELS = {
    "recency": "Recent activity", "release_cadence": "Release cadence", "issue_load": "Open issues",
    "contributor_spread": "Contributors", "documentation": "Documentation", "adoption": "Adoption",
}


@dataclass
class Health:
    score_total: Optional[float]
    label: str
    components: dict
    evidence: list
    license: LicenseInfo
    confidence: Confidence
    completeness: float


def interp(x: float, points: list, log_scale: bool = False) -> float:
    """Piecewise-linear score for x over [[x0, y0], [x1, y1], ...] (clamped at both ends)."""
    pts = sorted((math.log10(max(px, 1)) if log_scale else px, py) for px, py in points)
    if log_scale:
        x = math.log10(max(x, 1))
    if x <= pts[0][0]:
        return float(pts[0][1])
    if x >= pts[-1][0]:
        return float(pts[-1][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0)) if x1 != x0 else float(y1)
    return float(pts[-1][1])


def _g(row: Any, key: str) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def health_label(score: Optional[float], cfg: dict) -> str:
    if score is None:
        return "unknown"
    for threshold, name in sorted(cfg["labels"], reverse=True):
        if score >= threshold:
            return name
    return "inactive"


def _evidence_urls(row: Any) -> dict:
    url = (_g(row, "url") or "").rstrip("/")
    if _g(row, "source") == "github":
        return {"recency": f"{url}/commits", "release_cadence": f"{url}/releases", "issue_load": f"{url}/issues",
                "contributor_spread": f"{url}/graphs/contributors", "documentation": f"{url}#readme",
                "adoption": f"{url}/stargazers"}
    return {"recency": f"{url}/commits/main", "documentation": f"{url}/blob/main/README.md", "adoption": url}


def compute_health(row: Any, now: Optional[datetime] = None, cfg: Optional[dict] = None) -> Health:
    cfg = cfg or load_config()
    now = now or datetime.now(timezone.utc)
    weights, pts = cfg["weights"], cfg["points"]
    kind = _g(row, "kind") or "software"
    software = kind == "software"
    urls = _evidence_urls(row)
    comps: dict[str, dict] = {}

    def put(name: str, score: Optional[float], value: Any, detail: str) -> None:
        comps[name] = {"score": None if score is None else round(score, 1), "weight": weights[name],
                       "value": value, "detail": detail, "url": urls.get(name)}

    # recency
    last = parse_dt(_g(row, "last_commit_at"))
    if last:
        age = max(0, (now - last).days)
        verb = "Last commit" if _g(row, "source") == "github" else "Last updated"
        put("recency", interp(age, pts["recency_days"]), age, f"{verb} {age} days ago ({last.date()})")
    else:
        put("recency", None, None, "No activity date available")

    if software:
        rel = parse_dt(_g(row, "last_release_at"))
        if rel:
            age = max(0, (now - rel).days)
            put("release_cadence", interp(age, pts["release_days"]), age, f"Latest release {age} days ago ({rel.date()})")
        else:
            put("release_cadence", None, None, "No releases published (many healthy projects do not use GitHub releases)")

        issues, stars = _g(row, "open_issues"), _g(row, "stars")
        if issues is None or stars is None:
            put("issue_load", None, None, "Issue counts not available")
        else:
            ratio = issues / max(stars, 1)
            put("issue_load", interp(ratio, pts["issues_per_star"]), round(ratio, 4),
                f"{issues:,} open issues and PRs for {stars:,} stars ({ratio * 100:.1f} per 100 stars)")

        n = _g(row, "contributors")
        if n is None:
            put("contributor_spread", None, None, "Contributor count not available")
        else:
            risk = " (single maintainer: higher risk if they stop)" if n <= 1 else ""
            put("contributor_spread", interp(n, pts["contributors"]), n, f"{n:,} contributors{risk}")

    readme = _g(row, "readme_excerpt")
    chars = len(readme) if readme else 0
    put("documentation", interp(chars, pts["readme_chars"]), chars,
        f"README of about {chars:,}+ characters found" if chars else "No README or model card text found")

    if software:
        stars = _g(row, "stars")
        if stars is None:
            put("adoption", None, None, "Star count not available")
        else:
            put("adoption", interp(stars, pts["stars"], log_scale=True), stars, f"{stars:,} GitHub stars")
    else:
        dl = _g(row, "downloads")
        if dl is None:
            put("adoption", None, None, "Download count not available")
        else:
            put("adoption", interp(dl, pts["downloads"], log_scale=True), dl, f"{dl:,} downloads in the last 30 days")

    scored = [(c["weight"], c["score"]) for c in comps.values() if c["score"] is not None]
    total = round(sum(w * s for w, s in scored) / sum(w for w, _ in scored), 1) if scored else None
    required = REQUIRED.get(kind, REQUIRED["software"])
    completeness = sum(1 for name in required if comps[name]["score"] is not None) / len(required)

    lic = classify(_g(row, "license_spdx"))
    conf = confidence_label(completeness, comps["recency"]["score"] is not None, _g(row, "fetched_at"),
                            verified_at=_g(row, "verified_at"), now=now, cfg=cfg)

    evidence = [{"component": name, "label": LABELS[name], "score": c["score"], "detail": c["detail"], "url": c["url"]}
                for name, c in comps.items()]
    evidence.append({"component": "license", "label": "License", "score": None,
                     "detail": f"{lic.summary}. {lic.note}", "url": _g(row, "url")})
    return Health(total, health_label(total, cfg), comps, evidence, lic, conf, round(completeness, 2))


@dataclass
class ScoreStats:
    total: int = 0
    scored: int = 0
    skipped: int = 0


def store_health(conn: sqlite3.Connection, project_id: int, h: Health, now: datetime) -> None:
    conn.execute(
        """INSERT INTO health (project_id, score_total, components, evidence, computed_at,
                               license_class, commercial_use, confidence, completeness, label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(project_id) DO UPDATE SET
             score_total = excluded.score_total, components = excluded.components, evidence = excluded.evidence,
             computed_at = excluded.computed_at, license_class = excluded.license_class,
             commercial_use = excluded.commercial_use, confidence = excluded.confidence,
             completeness = excluded.completeness, label = excluded.label""",
        (project_id, h.score_total, json.dumps(h.components), json.dumps(h.evidence), now.isoformat(),
         h.license.license_class, h.license.commercial_use, h.confidence.label, h.completeness, h.label),
    )


def score_all(conn: sqlite3.Connection, rebuild: bool = False, now: Optional[datetime] = None,
              cfg: Optional[dict] = None) -> ScoreStats:
    """Score projects that have no health row, or whose data was refetched since scoring."""
    now = now or datetime.now(timezone.utc)
    stats = ScoreStats()
    rows = conn.execute(
        "SELECT p.*, h.computed_at AS h_computed FROM projects p LEFT JOIN health h ON h.project_id = p.id"
    ).fetchall()
    stats.total = len(rows)
    for row in rows:
        if not rebuild and row["h_computed"]:
            computed, fetched = parse_dt(row["h_computed"]), parse_dt(row["fetched_at"])
            if computed and fetched and computed >= fetched:
                stats.skipped += 1
                continue
        store_health(conn, row["id"], compute_health(row, now, cfg), now)
        stats.scored += 1
    conn.commit()
    return stats


def format_breakdown(row: Any, h: Health) -> str:
    score = "n/a" if h.score_total is None else f"{h.score_total:.1f}"
    lines = [
        f"{row['source_id']}  [{row['kind']}/{row['source']}]",
        f"Health {score} ({h.label}) | confidence: {h.confidence.label} | completeness {h.completeness:.0%}",
        f"  {h.confidence.message}",
        f"License: {h.license.summary} | commercial use: {h.license.commercial_use}",
        f"  {h.license.note}",
        "Components:",
    ]
    for name, c in h.components.items():
        s = "  -- " if c["score"] is None else f"{c['score']:5.1f}"
        lines.append(f"  {LABELS[name]:<18} {s}  (weight {c['weight']:.2f})  {c['detail']}")
        if c["url"]:
            lines.append(f"  {'':<18}        {c['url']}")
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute OpenPath health scores.")
    p.add_argument("--rebuild", action="store_true", help="rescore everything")
    p.add_argument("--show", metavar="SOURCE_ID", help="print the breakdown for one project, e.g. owner/repo")
    p.add_argument("--config", help="custom scoring config JSON")
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    conn = init_db(args.db or get_settings().db_path)
    cfg = load_config(args.config)
    if args.show:
        row = conn.execute("SELECT * FROM projects WHERE source_id = ?", (args.show,)).fetchone()
        if not row:
            print(f"No project with source_id {args.show!r}", file=sys.stderr)
            return 1
        print(format_breakdown(row, compute_health(row, cfg=cfg)))
        return 0
    stats = score_all(conn, rebuild=args.rebuild, cfg=cfg)
    log.info("Done. total=%d scored=%d skipped=%d", stats.total, stats.scored, stats.skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
