import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from openpath.db import init_db
from openpath.index import build as ib
from openpath.index import search as isr
from openpath.index.embedder import HashingEmbedder
from openpath.ingest.common import upsert_project
from openpath.scoring import health as H
from openpath.scoring.confidence import confidence_label
from openpath.scoring.config import load_config

NOW = datetime(2026, 11, 24, tzinfo=timezone.utc)
CFG = load_config()


def ago(days):
    return (NOW - timedelta(days=days)).isoformat()


def gh_row(**kw):
    base = {"source": "github", "source_id": "o/r", "kind": "software", "name": "r", "url": "https://github.com/o/r",
            "stars": 20000, "open_issues": 200, "contributors": 60, "license_spdx": "MIT",
            "last_commit_at": ago(10), "last_release_at": ago(20), "readme_excerpt": "x" * 3000, "fetched_at": ago(2)}
    base.update(kw)
    return base


def hf_row(**kw):
    base = {"source": "huggingface", "source_id": "a/m", "kind": "model", "name": "m", "url": "https://huggingface.co/a/m",
            "downloads": 50000, "likes": 10, "license_spdx": "apache-2.0", "last_commit_at": ago(20),
            "readme_excerpt": "y" * 1500, "fetched_at": ago(1)}
    base.update(kw)
    return base


# ---------- interpolation ----------
def test_interp():
    pts = [[0, 100], [30, 100], [180, 75]]
    assert H.interp(-5, pts) == 100 and H.interp(15, pts) == 100
    assert H.interp(105, pts) == pytest.approx(87.5)
    assert H.interp(999, pts) == 75
    assert H.interp(100, [[10, 20], [1000, 80]], log_scale=True) == pytest.approx(50.0)


# ---------- scoring ----------
def test_healthy_software():
    h = H.compute_health(gh_row(), NOW)
    assert h.score_total >= 90 and h.label == "healthy"
    assert h.completeness == 1.0 and h.confidence.label == "cached"
    assert h.license.license_class == "permissive"
    assert set(h.components) == {"recency", "release_cadence", "issue_load", "contributor_spread", "documentation", "adoption"}


def test_abandoned_single_maintainer_software():
    h = H.compute_health(gh_row(last_commit_at=ago(1500), last_release_at=None, contributors=1, open_issues=300,
                                stars=500, readme_excerpt="short readme " * 8), NOW)
    assert h.score_total < 25 and h.label == "inactive"
    assert "single maintainer" in h.components["contributor_spread"]["detail"]
    assert h.components["release_cadence"]["score"] is None  # unknown, not punished


def test_missing_components_are_excluded_from_weighted_mean():
    h = H.compute_health(gh_row(contributors=None, last_release_at=None), NOW)
    scored = [(c["weight"], c["score"]) for c in h.components.values() if c["score"] is not None]
    assert len(scored) == 4
    assert h.score_total == pytest.approx(round(sum(w * s for w, s in scored) / sum(w for w, _ in scored), 1))
    assert h.completeness == 0.8  # 4 of 5 required components


def test_model_scoring_uses_applicable_components_only():
    h = H.compute_health(hf_row(), NOW)
    assert set(h.components) == {"recency", "documentation", "adoption"}
    assert h.completeness == 1.0 and h.score_total > 60
    assert "downloads" in h.components["adoption"]["detail"]
    assert h.license.license_class == "permissive"


def test_no_data_at_all_is_low_data_with_no_score():
    h = H.compute_health({"source": "github", "source_id": "x", "kind": "software", "url": "u", "fetched_at": ago(1)}, NOW)
    assert h.confidence.label == "low_data"
    assert h.license.license_class == "restricted_or_unknown"
    assert h.components["documentation"]["score"] == 0  # no README text found


def test_missing_activity_date_forces_low_data():
    h = H.compute_health(gh_row(last_commit_at=None), NOW)
    assert h.confidence.label == "low_data"


def test_evidence_has_links_and_license():
    h = H.compute_health(gh_row(), NOW)
    urls = {e["component"]: e["url"] for e in h.evidence}
    assert urls["recency"] == "https://github.com/o/r/commits"
    assert urls["release_cadence"].endswith("/releases") and urls["adoption"].endswith("/stargazers")
    lic = [e for e in h.evidence if e["component"] == "license"][0]
    assert "MIT (permissive)" in lic["detail"]
    hf = H.compute_health(hf_row(), NOW)
    assert {e["component"]: e["url"] for e in hf.evidence}["documentation"].endswith("/blob/main/README.md")


def test_custom_weights_change_score():
    heavy_doc = json.loads(json.dumps(CFG))
    heavy_doc["weights"] = {k: (1.0 if k == "documentation" else 0.0) for k in CFG["weights"]}
    h = H.compute_health(gh_row(readme_excerpt="x" * 200), NOW, heavy_doc)
    assert h.score_total == 30.0


def test_labels():
    assert [H.health_label(s, CFG) for s in (90, 60, 30, 5)] == ["healthy", "fair", "weak", "inactive"]
    assert H.health_label(None, CFG) == "unknown"


# ---------- confidence ----------
def test_confidence_cached_live_and_low():
    c = confidence_label(1.0, True, ago(5), now=NOW)
    assert c.label == "cached" and c.as_of == (NOW - timedelta(days=5)).date().isoformat()
    assert "out of date" in confidence_label(1.0, True, ago(60), now=NOW).message
    assert confidence_label(1.0, True, ago(60), verified_at=(NOW - timedelta(hours=1)).isoformat(), now=NOW).label == "verified_live"
    assert confidence_label(1.0, True, ago(60), verified_at=(NOW - timedelta(hours=48)).isoformat(), now=NOW).label == "cached"
    low = confidence_label(0.2, True, ago(1), verified_at=(NOW - timedelta(hours=1)).isoformat(), now=NOW)
    assert low.label == "low_data" and "Verified live" in low.message


# ---------- storing ----------
@pytest.fixture
def conn(tmp_path):
    c = init_db(str(tmp_path / "t.db"))
    upsert_project(c, gh_row(source_id="o/mit", name="mitlib", license_spdx="MIT", description="pdf text extraction library"))
    upsert_project(c, gh_row(source_id="o/gpl", name="gpllib", license_spdx="GPL-3.0", description="pdf text extraction tool"))
    upsert_project(c, hf_row(description="text model"))
    return c


def test_score_all_stores_rows_with_evidence_and_class(conn):
    stats = H.score_all(conn, now=NOW)
    assert (stats.total, stats.scored, stats.skipped) == (3, 3, 0)
    rows = {r["source_id"]: r for r in conn.execute(
        "SELECT p.source_id, h.* FROM projects p JOIN health h ON h.project_id = p.id")}
    assert rows["o/mit"]["license_class"] == "permissive" and rows["o/mit"]["commercial_use"] == "yes"
    assert rows["o/gpl"]["license_class"] == "strong_copyleft"
    assert rows["o/mit"]["label"] == "healthy" and rows["o/mit"]["confidence"] == "cached"
    evidence = json.loads(rows["o/mit"]["evidence"])
    assert any(e["component"] == "recency" and e["url"] for e in evidence)
    assert json.loads(rows["o/mit"]["components"])["recency"]["score"] == 100.0


def test_every_project_gets_a_score_and_license_class(conn):
    H.score_all(conn, now=NOW)
    missing = conn.execute("SELECT COUNT(*) FROM projects p LEFT JOIN health h ON h.project_id = p.id "
                           "WHERE h.score_total IS NULL OR h.license_class IS NULL").fetchone()[0]
    assert missing == 0


def test_rescoring_only_changed_projects(conn):
    H.score_all(conn, now=NOW)
    assert H.score_all(conn, now=NOW).scored == 0
    conn.execute("UPDATE projects SET fetched_at = ? WHERE source_id = 'o/mit'", ((NOW + timedelta(days=1)).isoformat(),))
    stats = H.score_all(conn, now=NOW + timedelta(days=2))
    assert (stats.scored, stats.skipped) == (1, 2)
    assert H.score_all(conn, rebuild=True, now=NOW).scored == 3


def test_migration_adds_health_columns(tmp_path):
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE health (project_id INTEGER PRIMARY KEY, score_total REAL, components TEXT, evidence TEXT, computed_at TEXT NOT NULL)")
    old.commit(); old.close()
    cols = {r["name"] for r in init_db(path).execute("PRAGMA table_info(health)")}
    assert {"license_class", "commercial_use", "confidence", "completeness", "label"} <= cols


# ---------- search integration ----------
def test_search_filters_by_license_class_and_returns_health(conn):
    H.score_all(conn, now=NOW)
    emb = HashingEmbedder()
    ib.build_index(conn, emb)
    s = isr.Searcher(conn, emb)
    hits = s.search("pdf text extraction", isr.Filters(license_classes=["permissive"]), k=10)
    ids = {h.project["source_id"] for h in hits}
    assert "o/mit" in ids and "o/gpl" not in ids
    top = s.search("pdf text extraction", k=10)[0].project
    assert top["health_score"] is not None and top["license_class"] in ("permissive", "strong_copyleft")
    assert {"health_label", "commercial_use", "confidence"} <= set(top)
    both = s.search("pdf", isr.Filters(license_classes=["permissive", "strong_copyleft"]), k=10)
    assert {h.project["source_id"] for h in both} >= {"o/mit", "o/gpl"}


def test_unscored_projects_still_searchable_without_class_filter(conn):
    emb = HashingEmbedder()
    ib.build_index(conn, emb)
    hits = isr.Searcher(conn, emb).search("pdf", k=5)
    assert hits and hits[0].project["health_score"] is None


# ---------- CLI ----------
def test_cli_show_and_score(tmp_path, capsys):
    path = str(tmp_path / "cli.db")
    c = init_db(path)
    upsert_project(c, gh_row(fetched_at=datetime.now(timezone.utc).isoformat(), last_commit_at=datetime.now(timezone.utc).isoformat()))
    c.close()
    assert H.main(["--db", path]) == 0
    assert H.main(["--db", path, "--show", "o/r"]) == 0
    out = capsys.readouterr().out
    assert "o/r" in out and "License: MIT (permissive)" in out and "Recent activity" in out
    assert H.main(["--db", path, "--show", "missing/repo"]) == 1
