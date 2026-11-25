import sqlite3
import sys
import types
from datetime import datetime, timezone

import numpy as np
import pytest

from openpath.db import init_db
from openpath.index import build as ib
from openpath.index import search as isr
from openpath.index.embedder import (EmbedderUnavailable, HashingEmbedder, get_embedder, tokenize)
from openpath.ingest.common import upsert_project

NOW = datetime(2026, 11, 24, tzinfo=timezone.utc)
EMB = HashingEmbedder()


def add(conn, source_id, name, description, kind="software", language="Python", license="MIT",
        topics="[]", last_commit="2026-11-01T00:00:00Z", source="github", readme=None):
    upsert_project(conn, {
        "source": source, "source_id": source_id, "kind": kind, "name": name, "url": f"https://x/{source_id}",
        "description": description, "language": language, "license_spdx": license, "topics": topics,
        "last_commit_at": last_commit, "readme_excerpt": readme, "fetched_at": NOW.isoformat(),
    })


@pytest.fixture
def conn(tmp_path):
    c = init_db(str(tmp_path / "t.db"))
    add(c, "o/pdfplumber", "pdfplumber", "Library to extract text and tables from PDF documents", topics='["pdf","parsing"]')
    add(c, "o/pdfjs", "pdf.js", "JavaScript library to render PDF documents in the browser", language="JavaScript", license="Apache-2.0")
    add(c, "o/easycrm", "easycrm", "Self-hosted CRM for small businesses with appointment reminders", license="GPL-3.0")
    add(c, "o/chatui", "chatui", "Web chat interface for language models", language="TypeScript", license="NOASSERTION")
    add(c, "o/old", "oldtool", "Abandoned PDF text extraction script", last_commit="2020-01-01T00:00:00Z")
    add(c, "org/resnet", "resnet-mobile", "Image classification model optimized for mobile devices",
        kind="model", language=None, license="apache-2.0", source="huggingface")
    ib.build_index(c, EMB)
    return c


# ---------- embedder ----------
def test_hashing_embedder_is_deterministic_and_normalized():
    a = EMB.encode(["extract text from PDF"])[0]
    b = EMB.encode(["extract text from PDF"])[0]
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-5


def test_hashing_embedder_prefers_overlapping_text():
    q, near, far = EMB.encode(["extract text from PDF documents", "library to extract text from PDF files",
                               "image classification for mobile"])
    assert q @ near > q @ far


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("The PDFs and the documents for my clinic") == ["pdf", "document", "clinic"]


def test_sentence_transformer_wrapper_uses_expected_api(monkeypatch):
    class FakeST:
        def __init__(self, name): self.name = name
        def get_sentence_embedding_dimension(self): return 3
        def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar, convert_to_numpy):
            assert normalize_embeddings and convert_to_numpy
            return [[1.0, 0.0, 0.0] for _ in texts]
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    emb = get_embedder("sentence-transformers/fake")
    assert (emb.name, emb.dim) == ("sentence-transformers/fake", 3)
    assert emb.encode(["a", "b"]).shape == (2, 3)


def test_missing_sentence_transformers_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbedderUnavailable, match="pip install"):
        get_embedder("sentence-transformers/all-MiniLM-L6-v2")


# ---------- build ----------
def test_build_embeds_everything_then_skips(conn):
    n = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert n == 6
    stats = ib.build_index(conn, EMB)
    assert (stats.embedded, stats.skipped) == (0, 6)


def test_build_reembeds_only_changed_text(conn):
    conn.execute("UPDATE projects SET description = 'Totally different description' WHERE source_id = 'o/chatui'")
    stats = ib.build_index(conn, EMB)
    assert (stats.embedded, stats.skipped) == (1, 5)


def test_build_reembeds_all_when_embedder_changes(conn):
    other = HashingEmbedder(256)
    stats = ib.build_index(conn, other)
    assert stats.embedded == 6
    assert {r[0] for r in conn.execute("SELECT DISTINCT model FROM embeddings")} == {"hashing-256"}
    assert isr.Searcher(conn, EMB).search("pdf") == []  # nothing embedded with the old model any more


def test_project_text_contains_key_fields(conn):
    row = conn.execute("SELECT * FROM projects WHERE source_id = 'o/pdfplumber'").fetchone()
    text = ib.project_text(row)
    assert "pdfplumber" in text and "Language: Python" in text and "Topics: pdf, parsing" in text


def test_migration_adds_columns_to_old_embeddings_table(tmp_path):
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE embeddings (project_id INTEGER PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL)")
    old.commit(); old.close()
    conn = init_db(path)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(embeddings)")}
    assert {"dim", "text_hash", "updated_at"} <= cols


# ---------- search ----------
def ids(hits):
    return [h.project["source_id"] for h in hits]


def test_semantic_ranking(conn):
    s = isr.Searcher(conn, EMB)
    assert ids(s.search("extract text from PDF documents", k=3))[0] == "o/pdfplumber"
    assert ids(s.search("appointment reminders for a clinic", k=3))[0] == "o/easycrm"
    assert ids(s.search("image classifier for phones", k=3, filters=isr.Filters(kind="model")))[0] == "org/resnet"


def test_kind_and_source_filters(conn):
    s = isr.Searcher(conn, EMB)
    assert set(ids(s.search("anything", isr.Filters(kind="model"), k=50))) == {"org/resnet"}
    assert "org/resnet" not in ids(s.search("anything", isr.Filters(sources=["github"]), k=50))


def test_language_filter_applies_to_software_only(conn):
    got = set(ids(isr.Searcher(conn, EMB).search("anything", isr.Filters(languages=["python"]), k=50)))
    assert "o/pdfjs" not in got and "o/chatui" not in got
    assert {"o/pdfplumber", "org/resnet"} <= got  # python software kept, model passes through


def test_license_filters(conn):
    s = isr.Searcher(conn, EMB)
    assert set(ids(s.search("x", isr.Filters(licenses=["mit"]), k=50))) == {"o/pdfplumber", "o/old"}
    clear = set(ids(s.search("x", isr.Filters(require_license=True), k=50)))
    assert "o/chatui" not in clear and "o/easycrm" in clear


def test_max_age_filter(conn):
    got = ids(isr.Searcher(conn, EMB).search("pdf", isr.Filters(max_age_months=12), k=50, now=NOW))
    assert "o/old" not in got and "o/pdfplumber" in got
    conn.execute("UPDATE projects SET last_commit_at = NULL WHERE source_id = 'o/chatui'")
    got = ids(isr.Searcher(conn, EMB).search("pdf", isr.Filters(max_age_months=12), k=50, now=NOW))
    assert "o/chatui" not in got  # unknown activity is excluded when an age limit is set


def test_filters_combine(conn):
    f = isr.Filters(kind="software", languages=["python"], licenses=["mit"], max_age_months=12)
    assert ids(isr.Searcher(conn, EMB).search("pdf text", f, k=50, now=NOW)) == ["o/pdfplumber"]


def test_keyword_score_weights():
    row = {"name": "pandoc", "topics": '["converter"]', "description": "Convert documents between formats"}
    assert isr.keyword_score(["pandoc"], row) == 1.0
    assert isr.keyword_score(["converter"], row) == 0.7
    assert isr.keyword_score(["format"], row) == 0.4
    assert isr.keyword_score(["zzz"], row) == 0.0
    assert isr.keyword_score([], row) == 0.0


def test_keyword_boost_changes_score(conn):
    hit_off = isr.Searcher(conn, EMB, keyword_weight=0.0).search("pdfplumber", k=1)[0]
    hit_on = isr.Searcher(conn, EMB, keyword_weight=1.0).search("pdfplumber", k=1)[0]
    assert hit_on.keyword == 1.0 and hit_on.score == pytest.approx(hit_off.score + 1.0)


def test_empty_index_returns_nothing(tmp_path):
    assert isr.Searcher(init_db(str(tmp_path / "e.db")), EMB).search("anything") == []


def test_cli_build_then_search(tmp_path, capsys):
    path = str(tmp_path / "cli.db")
    c = init_db(path)
    add(c, "o/pdfplumber", "pdfplumber", "Library to extract text and tables from PDF documents")
    c.close()
    assert ib.main(["--embedder", "hashing", "--db", path]) == 0
    assert isr.main(["extract text from PDF", "--embedder", "hashing", "--db", path, "-k", "3"]) == 0
    assert "o/pdfplumber" in capsys.readouterr().out
