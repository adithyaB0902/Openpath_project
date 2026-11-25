import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
import requests
from huggingface_hub import ModelInfo
from huggingface_hub.errors import RepositoryNotFoundError
from requests.structures import CaseInsensitiveDict

from openpath.db import init_db
from openpath.explain import explain as X
from openpath.index import build as ib
from openpath.index.embedder import HashingEmbedder
from openpath.index.search import Searcher
from openpath.ingest.common import upsert_project
from openpath.ingest.github import GitHubClient
from openpath.interview.llm import LLMError, ScriptedLLM
from openpath.interview.store import save_profile
from openpath.profile import ProfileError
from openpath.rank import cli as rcli
from openpath.rank import effort as E
from openpath.rank import ranker as R
from openpath.rank import rules as RU
from openpath.rank.config import load_rank_config
from openpath.rank.verify import LiveVerifier, Verification
from openpath.scoring.confidence import confidence_label
from openpath.scoring.health import compute_health
from openpath.scoring.licenses import classify

NOW = datetime(2026, 11, 24, tzinfo=timezone.utc)
CFG = load_rank_config()
EMB = HashingEmbedder()


def ago(days):
    return (NOW - timedelta(days=days)).isoformat()


BASE = dict(source="github", kind="software", language="Python", license_spdx="MIT", stars=3000, open_issues=30,
            contributors=20, last_commit_at=ago(10), last_release_at=ago(30), readme_excerpt="r" * 2000,
            fetched_at=ago(1), topics="[]", url="https://github.com/x/y")


def add(conn, source_id, name, description, **kw):
    row = {**BASE, "source_id": source_id, "name": name, "description": description,
           "url": f"https://github.com/{source_id}", **kw}
    upsert_project(conn, row)


PROFILE = {"mode": "build", "need": "extract text from PDF documents",
           "constraints": {"license": "permissive", "commercial_use": True},
           "preferences": {"languages": ["python"], "kind": "software"}}


@pytest.fixture
def conn(tmp_path):
    c = init_db(str(tmp_path / "r.db"))
    add(c, "o/pdfplumber", "pdfplumber", "Library to extract text and tables from PDF documents", stars=5000)
    add(c, "o/mini", "pdfmini", "Small Python tool to extract text from PDF files", stars=300, contributors=3, readme_excerpt="r" * 800)
    add(c, "o/pdfgpl", "pdfgpl", "PDF text extraction tool", license_spdx="GPL-3.0")
    add(c, "o/pdfold", "pdfold", "PDF text extraction script", last_commit_at=ago(2000), last_release_at=None)
    add(c, "o/pdfjs", "pdfjs", "JavaScript library to extract text from PDF documents", language="JavaScript")
    add(c, "o/pdfnc", "pdfnc", "Extract text from PDF documents with machine learning", license_spdx="cc-by-nc-4.0")
    add(c, "a/pdf-ocr-70b", "pdf-ocr-70b", "Large OCR model that extracts text from PDF documents", source="huggingface",
        kind="model", language=None, license_spdx="apache-2.0", stars=None, likes=100, downloads=50000,
        param_count=70_000_000_000, contributors=None, open_issues=None, last_release_at=None, last_commit_at=ago(20))
    add(c, "datasets/a/pdf-text", "pdf-text", "Dataset of PDF text", source="huggingface", kind="dataset", language=None,
        license_spdx="cc-by-4.0", stars=None, likes=5, downloads=2000, contributors=None, open_issues=None,
        last_release_at=None)
    ib.build_index(c, EMB)
    return c


@pytest.fixture
def searcher(conn):
    return Searcher(conn, EMB)


def ids(res):
    return [r.project["source_id"] for r in res.results]


def rejected(res):
    return {r.project["source_id"]: r for r in res.rejected}


class FakeVerifier:
    def __init__(self, behaviors=None):
        self.behaviors, self.calls = behaviors or {}, []

    def verify(self, project, now):
        self.calls.append(project["source_id"])
        b = self.behaviors.get(project["source_id"], "ok")
        if b == "gone":
            return Verification("gone", reason="Repository not found.")
        if b == "error":
            return Verification("error", reason="offline")
        if callable(b):
            return b(project, now)
        return Verification("verified", {**project, "verified_at": now.isoformat(), "fetched_at": now.isoformat()}, [])


# ---------- rules ----------
def test_parse_min_activity_and_language_aliases():
    assert RU.parse_min_activity("commits in last 6 months") == 6
    assert RU.parse_min_activity("2 years") == 24
    assert RU.parse_min_activity("recent") is None and RU.parse_min_activity(None) is None
    assert RU.norm_language("JS") == "javascript" and RU.norm_language("Golang") == "go" and RU.norm_language(None) == ""


def test_build_query_adds_helpful_words():
    q = R.build_query({**PROFILE, "constraints": {"hosting": "self-hosted"}})
    assert q == "extract text from PDF documents self-hosted python"
    assert R.build_query({**PROFILE, "preferences": {"kind": "model", "languages": ["python"]}}) == "extract text from PDF documents"


# ---------- ranking and rejection ----------
def test_ranks_good_candidates_and_explains_exclusions(conn, searcher):
    res = R.rank(conn, PROFILE, searcher, now=NOW)
    assert ids(res) == ["o/pdfplumber", "o/mini"]
    rej = rejected(res)
    assert {k: [x.code for x in v.reasons] for k, v in rej.items()} == {
        "o/pdfgpl": ["license"], "o/pdfnc": ["license"], "o/pdfold": ["inactive"], "o/pdfjs": ["language"]}
    assert "strong copyleft" in rej["o/pdfgpl"].reasons[0].text
    assert "commercial" in rej["o/pdfnc"].reasons[0].text
    assert "No activity" in rej["o/pdfold"].reasons[0].text
    assert "JavaScript" in rej["o/pdfjs"].reasons[0].text and "python" in rej["o/pdfjs"].reasons[0].text
    assert res.meta["pool"] == 6 and res.meta["excluded_by_rules"] == 4


def test_results_are_complete_and_ordered(conn, searcher):
    res = R.rank(conn, PROFILE, searcher, now=NOW)
    scores = [r.score for r in res.results]
    assert scores == sorted(scores, reverse=True) and all(0 <= s <= 100 for s in scores)
    top = res.results[0]
    assert set(top.components) == {"semantic", "health", "license", "effort"}
    assert all(0 <= v <= 1 for v in top.components.values())
    assert top.explanation.fit and top.explanation.effort["level"] in ("easy", "moderate", "hard")
    assert top.confidence.label == "cached" and top.verification is None
    assert top.license.license_class == "permissive"
    assert top.explanation.source == "template"


def test_ranking_is_deterministic(conn, searcher):
    a, b = R.rank(conn, PROFILE, searcher, now=NOW), R.rank(conn, PROFILE, searcher, now=NOW)
    assert ids(a) == ids(b) and [r.score for r in a.results] == [r.score for r in b.results]


def test_k_and_rejected_k_caps(conn, searcher):
    res = R.rank(conn, PROFILE, searcher, now=NOW, k=1, rejected_k=2)
    assert len(res.results) == 1 and len(res.rejected) == 2
    assert [r.retrieval_rank for r in res.rejected] == sorted(r.retrieval_rank for r in res.rejected)


def test_kind_preferences(conn, searcher):
    models = R.rank(conn, {**PROFILE, "preferences": {"kind": "model"}, "constraints": {}}, searcher, now=NOW)
    assert ids(models) == ["a/pdf-ocr-70b"]
    anything = R.rank(conn, {**PROFILE, "preferences": {"kind": "any"}, "constraints": {}}, searcher, now=NOW, k=10)
    seen = set(ids(anything)) | set(rejected(anything))
    assert "a/pdf-ocr-70b" in seen and "datasets/a/pdf-text" not in seen  # 'any' = software or models


def test_min_activity_preference_tightens_the_rule(conn, searcher):
    add(conn, "o/stale", "pdfstale", "PDF text extraction library for documents", last_commit_at=ago(300))
    ib.build_index(conn, EMB)
    default = R.rank(conn, PROFILE, searcher, now=NOW, k=10)
    strict = R.rank(conn, {**PROFILE, "preferences": {**PROFILE["preferences"], "min_activity": "commits in last 6 months"}},
                    searcher, now=NOW, k=10)
    assert "o/stale" in ids(default) and "o/stale" in rejected(strict)
    assert "6 months" in rejected(strict)["o/stale"].reasons[0].text


def test_low_health_rule(conn, searcher):
    res = R.rank(conn, PROFILE, searcher, now=NOW, cfg={**CFG, "min_health": 99.9})
    assert not res.results and all(r.reasons[0].code in ("low_health", "license", "language", "inactive") for r in res.rejected)
    assert any(r.reasons[0].code == "low_health" for r in res.rejected)


def test_big_model_gets_effort_penalty_and_warning(conn, searcher):
    profile = {**PROFILE, "preferences": {"kind": "model"}, "constraints": {"skill_level": "beginner"}}
    r = R.rank(conn, profile, searcher, now=NOW).results[0]
    assert r.explanation.effort["level"] != "easy"
    assert "70.0B parameters" in r.explanation.effort["signals"]
    assert any("Large model" in t for t in r.explanation.tradeoffs)


def test_empty_index_gives_a_helpful_warning(tmp_path):
    c = init_db(str(tmp_path / "e.db"))
    res = R.rank(c, PROFILE, Searcher(c, EMB), now=NOW)
    assert res.results == [] and "index" in res.meta["warnings"][0]


def test_invalid_profile_is_rejected(conn, searcher):
    with pytest.raises(ProfileError):
        R.rank(conn, {"mode": "build"}, searcher)


# ---------- live verification inside the pipeline ----------
def test_verified_results_are_labelled_live_and_counted(conn, searcher):
    v = FakeVerifier()
    res = R.rank(conn, PROFILE, searcher, verifier=v, now=NOW)
    assert ids(res) == ["o/pdfplumber", "o/mini"] and v.calls == ["o/pdfplumber", "o/mini"]
    assert all(r.confidence.label == "verified_live" for r in res.results) and res.meta["verified"] == 2


def test_gone_candidate_is_dropped_and_explained_then_backfilled(conn, searcher):
    res = R.rank(conn, PROFILE, searcher, verifier=FakeVerifier({"o/pdfplumber": "gone"}), now=NOW)
    assert ids(res) == ["o/mini"]
    assert [x.code for x in rejected(res)["o/pdfplumber"].reasons] == ["unavailable"]
    assert res.meta["dropped_by_verification"] == 1


def test_license_change_found_by_verification_rejects_candidate(conn, searcher):
    def turned_gpl(p, now):
        return Verification("verified", {**p, "license_spdx": "GPL-3.0", "verified_at": now.isoformat()},
                            ["License changed from MIT to GPL-3.0 since our last crawl."])
    res = R.rank(conn, PROFILE, searcher, verifier=FakeVerifier({"o/pdfplumber": turned_gpl}), now=NOW)
    assert ids(res) == ["o/mini"] and rejected(res)["o/pdfplumber"].reasons[0].code == "license"


def test_verification_changes_are_shown_as_tradeoffs(conn, searcher):
    msg = "License changed from MIT to Apache-2.0 since our last crawl."
    def changed(p, now):
        return Verification("verified", {**p, "license_spdx": "Apache-2.0", "verified_at": now.isoformat()}, [msg])
    top = R.rank(conn, PROFILE, searcher, verifier=FakeVerifier({"o/pdfplumber": changed}), now=NOW).results[0]
    assert msg in top.explanation.tradeoffs and top.verification.changes == [msg]


def test_unreachable_source_keeps_cached_data_with_a_warning(conn, searcher):
    res = R.rank(conn, PROFILE, searcher, verifier=FakeVerifier({"o/pdfplumber": "error", "o/mini": "error"}), now=NOW)
    assert ids(res) == ["o/pdfplumber", "o/mini"]
    assert all(r.confidence.label == "cached" for r in res.results)
    assert res.meta["verification_errors"] == 2 and "could not be checked live" in res.meta["warnings"][0]


def test_verify_budget_is_respected(conn, searcher):
    v = FakeVerifier({"o/pdfplumber": "gone"})
    res = R.rank(conn, PROFILE, searcher, verifier=v, cfg={**CFG, "verify_budget": 1}, now=NOW)
    assert v.calls == ["o/pdfplumber"] and ids(res) == ["o/mini"]
    assert res.results[0].confidence.label == "cached"  # shown from saved data once the budget ran out


# ---------- the grounding guard ----------
def test_result_outside_retrieved_set_is_blocked(conn, searcher):
    def impostor(p, now):
        return Verification("verified", {**p, "id": 99999, "verified_at": now.isoformat()}, [])
    with pytest.raises(R.UngroundedResultError, match="not retrieved"):
        R.rank(conn, PROFILE, searcher, verifier=FakeVerifier({"o/pdfplumber": impostor}), now=NOW)


def test_assert_grounded_unit(conn, searcher):
    res = R.rank(conn, PROFILE, searcher, now=NOW)
    pool_ids = {r.project["id"] for r in res.results}
    R.assert_grounded(conn, res.results, pool_ids)
    with pytest.raises(R.UngroundedResultError):
        R.assert_grounded(conn, res.results, set())
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DELETE FROM projects WHERE id = ?", (res.results[0].project["id"],))
    with pytest.raises(R.UngroundedResultError, match="not in the database"):
        R.assert_grounded(conn, res.results, pool_ids)


# ---------- LLM explanations inside the pipeline ----------
def test_llm_explanations_are_used_when_grounded(conn, searcher):
    llm = ScriptedLLM([lambda s, u: {"fit": ["A strong match for what you described."], "tradeoffs": []}] * 4)
    res = R.rank(conn, PROFILE, searcher, llm=llm, now=NOW)
    assert all(r.explanation.source == "llm" for r in res.results)
    assert res.results[0].explanation.fit == ["A strong match for what you described."]
    facts = json.loads(llm.calls[0][1])
    assert {"user_need", "project", "license", "health", "requirements_met"} <= set(facts)


def test_llm_failure_falls_back_to_template(conn, searcher):
    res = R.rank(conn, PROFILE, searcher, llm=ScriptedLLM([LLMError("down")] * 4), now=NOW)
    assert all(r.explanation.source == "template" and r.explanation.fit for r in res.results)


# ---------- explainer ----------
def project_row(**kw):
    return {**BASE, "id": 1, "source_id": "o/p", "name": "pdfplumber", "description": "Extract text from PDFs", **kw}


def explain_for(project, llm=None, profile=PROFILE):
    h = compute_health(project, NOW)
    lic = classify(project["license_spdx"])
    return X.explain(profile, project, h, lic, h.confidence, ["Written in Python, one of your preferred languages."], [],
                     E.effort_card(h, project, profile, CFG), CFG, llm)


def test_is_grounded():
    facts = json.dumps({"health": {"score": 91.2, "out_of": 100}, "stars": "5,000 GitHub stars"})
    assert X.is_grounded("Health score is 91 out of 100.", facts)
    assert X.is_grounded("It has 5,000 GitHub stars.", facts)
    assert not X.is_grounded("It is 2.5x faster than the alternative.", facts)
    assert not X.is_grounded("See https://example.com for details.", facts)
    assert not X.is_grounded("x" * 400, facts)
    assert X.is_grounded("Looks like a good fit.", facts)


def test_template_explanation_without_llm():
    e = explain_for(project_row())
    assert e.source == "template" and e.fit[0].startswith("pdfplumber: Extract text")
    assert any("Health score" in s for s in e.fit) and e.effort["level"] == "easy"


def test_ungrounded_llm_sentences_are_dropped():
    llm = ScriptedLLM([{"fit": ["Trusted by 7,000 companies worldwide.", "See https://x.io for docs.", "A clean fit for your task."],
                        "tradeoffs": ["Nothing major."]}])
    e = explain_for(project_row(), llm)
    assert e.source == "llm" and e.fit == ["A clean fit for your task."]


def test_all_ungrounded_means_template():
    llm = ScriptedLLM([{"fit": ["Used by 123456 teams."], "tradeoffs": []}])
    assert explain_for(project_row(), llm).source == "template"


@pytest.mark.parametrize("bad", ["not json", {"fit": "oops"}, {"nothing": 1}, LLMError("down")])
def test_broken_llm_output_means_template(bad):
    e = explain_for(project_row(), ScriptedLLM([bad]))
    assert e.source == "template" and e.fit


def test_risk_warnings_always_come_from_code_not_the_llm():
    p = project_row(license_spdx="AGPL-3.0", contributors=1, last_commit_at=ago(10), fetched_at=ago(90))
    llm = ScriptedLLM([{"fit": ["Great project."], "tradeoffs": []}])
    e = explain_for(p, llm)
    text = " ".join(e.tradeoffs)
    assert e.source == "llm" and "Network copyleft" in text and "single maintainer" in text and "out of date" in text


def test_thin_docs_and_many_issues_are_flagged():
    p = project_row(readme_excerpt="tiny", open_issues=2000, stars=100, last_release_at=ago(1200))
    text = " ".join(explain_for(p).tradeoffs)
    assert "Documentation looks thin" in text and "Many open issues" in text and "Releases are infrequent" in text


# ---------- effort ----------
def test_effort_levels_and_penalties():
    good = compute_health(project_row(), NOW)
    assert E.effort_card(good, project_row(), PROFILE, CFG)["level"] == "easy"
    thin = compute_health(project_row(readme_excerpt="x", stars=20), NOW)
    card = E.effort_card(thin, project_row(readme_excerpt="x", stars=20), {**PROFILE, "constraints": {"skill_level": "beginner"}}, CFG)
    assert card["level"] == "hard" and "beginner" in card["note"]
    model = project_row(kind="model", source="huggingface", param_count=70_000_000_000, stars=None, downloads=50000,
                        language=None, contributors=None, open_issues=None)
    h = compute_health(model, NOW)
    beginner = {**PROFILE, "constraints": {"skill_level": "beginner"}}
    assert E.effort_fit(h, model, beginner, CFG) < E.effort_fit(h, model, PROFILE, CFG)
    assert "GPU" in E.effort_card(h, model, beginner, CFG)["note"]


# ---------- live verifier against fake sources ----------
class Resp:
    def __init__(self, status=200, body=None, headers=None, text=None):
        self.status_code = status
        self.headers = CaseInsensitiveDict(headers or {})
        self.text = text if text is not None else json.dumps(body)


class Session:
    def __init__(self, handler):
        self.handler, self.headers, self.calls = handler, {}, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        return self.handler(url)


def gh_verifier(conn, handler):
    client = GitHubClient("t", conn, session=Session(handler), sleep=lambda s: None, now=lambda: 1000.0, search_interval=0)
    return LiveVerifier(conn, github=client)


def repo_json(**kw):
    return {"full_name": "o/pdfplumber", "html_url": "https://github.com/o/pdfplumber", "archived": False, "disabled": False,
            "license": {"spdx_id": "MIT"}, "stargazers_count": 5100, "forks_count": 9, "open_issues_count": 20,
            "pushed_at": "2026-11-20T00:00:00Z", "description": "new description", "language": "Python", **kw}


def project_of(conn, sid):
    return dict(conn.execute("SELECT * FROM projects WHERE source_id = ?", (sid,)).fetchone())


def test_github_verification_refreshes_and_stores(conn):
    v = gh_verifier(conn, lambda url: Resp(body=repo_json()))
    out = v.verify(project_of(conn, "o/pdfplumber"), NOW)
    assert out.status == "verified" and out.changes == [] and out.project["verified_at"]
    row = project_of(conn, "o/pdfplumber")
    assert (row["stars"], row["description"], row["last_commit_at"]) == (5100, "new description", "2026-11-20T00:00:00Z")
    assert row["fetched_at"] == NOW.isoformat()
    assert conn.execute("SELECT confidence FROM health WHERE project_id = ?", (row["id"],)).fetchone()[0] == "cached"


def test_github_license_change_and_rename_are_reported(conn):
    v = gh_verifier(conn, lambda url: Resp(body=repo_json(license={"spdx_id": "GPL-3.0"}, full_name="new/name",
                                                         html_url="https://github.com/new/name")))
    out = v.verify(project_of(conn, "o/pdfplumber"), NOW)
    assert any("License changed from MIT to GPL-3.0" in c for c in out.changes)
    assert any("moved to new/name" in c for c in out.changes)
    assert project_of(conn, "o/pdfplumber")["url"] == "https://github.com/new/name"


def test_github_never_moves_activity_backwards(conn):
    v = gh_verifier(conn, lambda url: Resp(body=repo_json(pushed_at="2020-01-01T00:00:00Z")))
    out = v.verify(project_of(conn, "o/pdfplumber"), NOW)
    assert out.project["last_commit_at"] == ago(10)


@pytest.mark.parametrize("handler, status", [
    (lambda url: Resp(404, {"message": "Not Found"}), "gone"),
    (lambda url: Resp(body=repo_json(archived=True)), "archived"),
    (lambda url: Resp(403, text="forbidden", headers={"X-RateLimit-Remaining": "100"}), "error"),
    (lambda url: Resp(body=["not", "a", "dict"]), "error"),
])
def test_github_verification_outcomes(conn, handler, status):
    assert gh_verifier(conn, handler).verify(project_of(conn, "o/pdfplumber"), NOW).status == status


def test_verifier_without_a_checker_reports_error(conn):
    out = LiveVerifier(conn).verify(project_of(conn, "o/pdfplumber"), NOW)
    assert out.status == "error" and "No live checker" in out.reason


def not_found():
    resp = requests.Response()
    resp.status_code, resp._content = 404, b"{}"
    return RepositoryNotFoundError("nope", response=resp)


class FakeHf:
    def __init__(self, result):
        self.result, self.calls = result, []

    def model_info(self, repo_id, expand=None):
        self.calls.append((repo_id, expand))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    dataset_info = model_info


def hf_model(**kw):
    data = {"id": "a/pdf-ocr-70b", "downloads": 90000, "likes": 150, "tags": ["license:apache-2.0"],
            "lastModified": "2026-11-10T00:00:00.000Z", "safetensors": {"total": 70_000_000_000, "parameters": {"BF16": 70_000_000_000}}}
    data.update(kw)
    return ModelInfo(**data)


def test_huggingface_verification(conn):
    api = FakeHf(hf_model())
    out = LiveVerifier(conn, hf_api=api).verify(project_of(conn, "a/pdf-ocr-70b"), NOW)
    assert out.status == "verified" and out.changes == [] and api.calls[0][0] == "a/pdf-ocr-70b" and api.calls[0][1]
    assert project_of(conn, "a/pdf-ocr-70b")["downloads"] == 90000


def test_huggingface_license_change_private_and_missing(conn):
    p = project_of(conn, "a/pdf-ocr-70b")
    changed = LiveVerifier(conn, hf_api=FakeHf(hf_model(tags=["license:cc-by-nc-4.0"]))).verify(p, NOW)
    assert any("License changed from apache-2.0 to cc-by-nc-4.0" in c for c in changed.changes)
    assert LiveVerifier(conn, hf_api=FakeHf(hf_model(private=True))).verify(p, NOW).status == "gone"
    assert LiveVerifier(conn, hf_api=FakeHf(not_found())).verify(p, NOW).status == "gone"
    assert LiveVerifier(conn, hf_api=FakeHf(RuntimeError("network down"))).verify(p, NOW).status == "error"


def test_huggingface_dataset_uses_repo_id_without_prefix(conn):
    api = FakeHf(hf_model(id="a/pdf-text"))
    LiveVerifier(conn, hf_api=api).verify(project_of(conn, "datasets/a/pdf-text"), NOW)
    assert api.calls[0][0] == "a/pdf-text"


# ---------- storing runs ----------
def test_save_run(conn, searcher):
    res = R.rank(conn, PROFILE, searcher, now=NOW)
    code = save_profile(conn, PROFILE)
    run_id = R.save_run(conn, code, res)
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    stored = json.loads(row["results_json"])
    assert [r["project"]["source_id"] for r in stored] == ids(res) and row["profile_code"] == code
    assert stored[0]["explanation"]["fit"] and stored[0]["license"]["class"] == "permissive"
    assert len(json.loads(row["rejected_json"])) == 4
    assert R.save_run(conn, None, res) > run_id
    with pytest.raises(sqlite3.IntegrityError):
        R.save_run(conn, "zzzzzzzz", res)


# ---------- CLI ----------
@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "cli.db")
    c = init_db(path)
    add(c, "o/pdfplumber", "pdfplumber", "Library to extract text and tables from PDF documents", last_commit_at=datetime.now(timezone.utc).isoformat(),
        fetched_at=datetime.now(timezone.utc).isoformat(), last_release_at=datetime.now(timezone.utc).isoformat())
    add(c, "o/pdfgpl", "pdfgpl", "PDF text extraction tool", license_spdx="GPL-3.0", last_commit_at=datetime.now(timezone.utc).isoformat(),
        fetched_at=datetime.now(timezone.utc).isoformat())
    ib.build_index(c, EMB)
    code = save_profile(c, PROFILE)
    c.close()
    return path, code


def test_cli_quick_search(db_path):
    lines = []
    assert rcli.main(["--need", "extract text from PDF documents", "--no-verify", "--no-llm", "--embedder", "hashing",
                      "--db", db_path[0]], out=lines.append) == 0
    text = "\n".join(lines)
    assert "Recommendations for: extract text from PDF documents" in text and "pdfplumber" in text and "Why it fits:" in text


def test_cli_with_code_shows_why_not_and_saves_run(db_path):
    lines = []
    assert rcli.main(["--code", db_path[1], "--no-verify", "--no-llm", "--embedder", "hashing", "--db", db_path[0], "--save-run"],
                     out=lines.append) == 0
    text = "\n".join(lines)
    assert "Why not these?" in text and "pdfgpl" in text and "strong copyleft" in text and "Saved run" in text


def test_cli_json_and_bad_code(db_path):
    lines = []
    assert rcli.main(["--code", db_path[1], "--no-verify", "--no-llm", "--embedder", "hashing", "--db", db_path[0], "--json"],
                     out=lines.append) == 0
    data = json.loads("".join(lines))
    assert data["results"][0]["project"]["source_id"] == "o/pdfplumber" and data["rejected"][0]["reasons"][0]["code"] == "license"
    assert rcli.main(["--code", "abcd2345", "--no-verify", "--no-llm", "--embedder", "hashing", "--db", db_path[0]]) == 1
