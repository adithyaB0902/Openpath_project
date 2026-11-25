import sqlite3

import pytest

from openpath.db import init_db


def test_tables_created(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    names = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"projects", "health", "embeddings", "profiles", "runs"} <= names


def test_project_unique_per_source(tmp_path):
    conn = init_db(str(tmp_path / "t.db"))
    row = ("github", "a/b", "b", "https://github.com/a/b", "2026-01-01T00:00:00")
    q = "INSERT INTO projects (source, source_id, name, url, fetched_at) VALUES (?,?,?,?,?)"
    conn.execute(q, row)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(q, row)
