"""The recommendation pipeline: profile -> retrieve -> reject with reasons -> score -> verify live -> explain.

Guarantees:
- Every result comes from the retrieved candidate set (checked at the end, see assert_grounded).
- Every excluded close match is returned with the reasons it was excluded.
- The LLM (optional) never chooses, scores or adds projects; it can only rephrase grounded facts.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from openpath.explain.explain import Explanation, explain
from openpath.index.search import Filters, Searcher
from openpath.profile import ProfileError, validate_profile
from openpath.rank.config import load_rank_config
from openpath.rank.effort import effort_card, effort_fit
from openpath.rank.rules import Reason, evaluate_rules, norm_language
from openpath.rank.verify import Verification
from openpath.scoring.confidence import Confidence
from openpath.scoring.health import Health, compute_health
from openpath.scoring.licenses import CONSTRAINT_CLASSES, LicenseInfo, classify, license_fit

log = logging.getLogger("openpath.rank")

PROJECT_FIELDS = ("id", "source", "source_id", "kind", "name", "url", "description", "language", "license_spdx",
                  "stars", "likes", "downloads", "param_count", "last_commit_at", "verified_at")


class UngroundedResultError(RuntimeError):
    """A result was not part of the retrieved candidate set. This must never happen."""


def _project_dict(p: dict) -> dict:
    return {k: p.get(k) for k in PROJECT_FIELDS}


@dataclass
class Result:
    project: dict
    score: float
    components: dict
    health: Health
    license: LicenseInfo
    confidence: Confidence
    verification: Optional[Verification]
    explanation: Explanation

    def to_dict(self) -> dict:
        return {
            "project": _project_dict(self.project), "score": self.score, "components": self.components,
            "health": {"score": self.health.score_total, "label": self.health.label,
                       "completeness": self.health.completeness, "evidence": self.health.evidence},
            "license": {"raw": self.license.raw, "class": self.license.license_class,
                        "commercial_use": self.license.commercial_use, "note": self.license.note},
            "confidence": {"label": self.confidence.label, "as_of": self.confidence.as_of, "message": self.confidence.message},
            "verification": self.verification.to_dict() if self.verification else None,
            "explanation": self.explanation.to_dict(),
        }


@dataclass
class Rejected:
    project: dict
    reasons: list
    retrieval_rank: int

    def to_dict(self) -> dict:
        return {"project": _project_dict(self.project), "reasons": [r.to_dict() for r in self.reasons],
                "retrieval_rank": self.retrieval_rank}


@dataclass
class RankResult:
    results: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"results": [r.to_dict() for r in self.results], "rejected": [r.to_dict() for r in self.rejected],
                "meta": self.meta}


@dataclass
class _Entry:
    project: dict
    idx: int
    sem: float
    health: Health
    lic: LicenseInfo
    verification: Optional[Verification] = None
    score: float = 0.0
    components: dict = field(default_factory=dict)


def build_query(profile: dict) -> str:
    """The need, plus words from the constraints that help retrieval."""
    parts = [profile["need"]]
    c, p = profile.get("constraints", {}), profile.get("preferences", {})
    if c.get("hosting") == "self-hosted":
        parts.append("self-hosted")
    if p.get("languages") and p.get("kind") != "model":
        parts.append(" ".join(p["languages"]))
    return " ".join(parts)


def _popularity(p: dict) -> int:
    return p.get("stars") or p.get("likes") or 0


def _sort_key(e: _Entry) -> tuple:
    return (-e.score, -_popularity(e.project), e.project["source_id"])


def _score(e: _Entry, profile: dict, cfg: dict) -> None:
    h = e.health
    comps = {
        "semantic": e.sem,
        "health": h.score_total / 100 if h.score_total is not None else cfg["unknown_health_score"],
        "license": cfg["license_class_scores"][e.lic.license_class],
        "effort": effort_fit(h, e.project, profile, cfg),
    }
    e.components = {k: round(v, 3) for k, v in comps.items()}
    e.score = round(100 * sum(cfg["weights"][k] * comps[k] for k in cfg["weights"]), 1)


def _matches(project: dict, profile: dict, lic: LicenseInfo) -> list:
    """Plain-language statements about requirements this project satisfies (inputs to the explanation)."""
    out = []
    c, prefs = profile.get("constraints", {}), profile.get("preferences", {})
    if c.get("license") in CONSTRAINT_CLASSES or c.get("commercial_use") is not None:
        ok, why = license_fit(lic, c.get("license"), c.get("commercial_use"))
        if ok:
            out.append(why)
    wanted = [norm_language(x) for x in prefs.get("languages", [])]
    if project["kind"] == "software" and wanted and norm_language(project.get("language")) in wanted:
        out.append(f"Written in {project['language']}, one of your preferred languages.")
    return out


def assert_grounded(conn: sqlite3.Connection, results: list, retrieved_ids: set) -> None:
    for r in results:
        pid = r.project["id"]
        if pid not in retrieved_ids:
            raise UngroundedResultError(f"result {r.project.get('source_id')!r} was not retrieved from the index")
        if conn.execute("SELECT 1 FROM projects WHERE id = ?", (pid,)).fetchone() is None:
            raise UngroundedResultError(f"result {r.project.get('source_id')!r} is not in the database")


def rank(conn: sqlite3.Connection, profile: dict, searcher: Searcher, *, verifier: Any = None, llm: Any = None,
         cfg: Optional[dict] = None, now: Optional[datetime] = None, k: Optional[int] = None,
         rejected_k: Optional[int] = None) -> RankResult:
    cfg = cfg or load_rank_config()
    now = now or datetime.now(timezone.utc)
    k, rejected_k = k or cfg["k"], rejected_k or cfg["rejected_k"]
    errors = validate_profile(profile)
    if errors:
        raise ProfileError(errors)

    kind = profile["preferences"].get("kind")
    pool = searcher.search(build_query(profile), Filters(kind=kind if kind in ("software", "model") else None),
                           k=cfg["pool_size"], now=now)
    if kind not in ("software", "model"):
        pool = [h for h in pool if h.project["kind"] != "dataset"]  # 'any' means software or models
    meta = {"pool": len(pool), "excluded_by_rules": 0, "verified": 0, "dropped_by_verification": 0,
            "verification_errors": 0, "warnings": []}
    if not pool:
        meta["warnings"].append("No candidates found. Has the index been built (python -m openpath.index.build)?")
        return RankResult([], [], meta)

    retrieved_ids = {h.project["id"] for h in pool}
    scores = [h.score for h in pool]
    lo, hi = min(scores), max(scores)
    rejected: list = []
    accepted: list = []
    for idx, hit in enumerate(pool):
        project = dict(hit.project)
        health = compute_health(project, now)
        reasons = evaluate_rules(project, profile, health, cfg, now)
        if reasons:
            rejected.append(Rejected(project, reasons, idx))
            continue
        entry = _Entry(project, idx, (hit.score - lo) / (hi - lo) if hi > lo else 1.0, health, classify(project["license_spdx"]))
        _score(entry, profile, cfg)
        accepted.append(entry)
    meta["excluded_by_rules"] = len(rejected)
    accepted.sort(key=_sort_key)

    final: list = []
    if verifier is None:
        final = accepted[:k]
    else:
        budget, position = cfg["verify_budget"], 0
        while position < len(accepted) and len(final) < k and budget > 0:
            entry = accepted[position]
            position += 1
            budget -= 1
            v = verifier.verify(entry.project, now)
            if v.status in ("gone", "archived"):
                rejected.append(Rejected(entry.project, [Reason("unavailable", v.reason)], entry.idx))
                meta["dropped_by_verification"] += 1
                continue
            entry.verification = v
            if v.status == "verified":
                meta["verified"] += 1
                entry.project = v.project
                entry.health = compute_health(entry.project, now)
                entry.lic = classify(entry.project["license_spdx"])
                reasons = evaluate_rules(entry.project, profile, entry.health, cfg, now)
                if reasons:  # fresh data may now conflict with the requirements
                    rejected.append(Rejected(entry.project, reasons, entry.idx))
                    meta["dropped_by_verification"] += 1
                    continue
                _score(entry, profile, cfg)
            else:
                meta["verification_errors"] += 1
            final.append(entry)
        final.extend(accepted[position:position + max(0, k - len(final))])  # budget ran out: show saved data
        final.sort(key=_sort_key)
        final = final[:k]
        if meta["verification_errors"]:
            meta["warnings"].append(f"{meta['verification_errors']} result(s) could not be checked live and show saved data.")

    results = []
    for e in final:
        conf = e.health.confidence
        card = effort_card(e.health, e.project, profile, cfg)
        changes = e.verification.changes if e.verification else []
        expl = explain(profile, e.project, e.health, e.lic, conf, _matches(e.project, profile, e.lic), changes, card, cfg, llm)
        results.append(Result(e.project, e.score, e.components, e.health, e.lic, conf, e.verification, expl))
    assert_grounded(conn, results, retrieved_ids)

    rejected.sort(key=lambda r: r.retrieval_rank)
    return RankResult(results, rejected[:rejected_k], meta)


def save_run(conn: sqlite3.Connection, profile_code: Optional[str], result: RankResult) -> int:
    """Store a run. profile_code must already exist in profiles (see interview.store.save_profile)."""
    data = result.to_dict()
    cur = conn.execute(
        "INSERT INTO runs (profile_code, results_json, rejected_json, created_at) VALUES (?, ?, ?, ?)",
        (profile_code, json.dumps(data["results"]), json.dumps(data["rejected"]), datetime.now(timezone.utc).isoformat()))
    conn.commit()
    return cur.lastrowid
