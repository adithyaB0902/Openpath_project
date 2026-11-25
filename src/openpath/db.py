"""SQLite helpers."""
import sqlite3
from importlib import resources
from pathlib import Path


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str) -> sqlite3.Connection:
    """Create all tables if they do not exist and return a connection."""
    conn = connect(db_path)
    sql = resources.files("openpath").joinpath("schema.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring databases created by older phases up to date."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(embeddings)")}
    for name, ddl in (("dim", "INTEGER"), ("text_hash", "TEXT"), ("updated_at", "TEXT")):
        if name not in cols:
            conn.execute(f"ALTER TABLE embeddings ADD COLUMN {name} {ddl}")
