CREATE TABLE IF NOT EXISTS projects (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    kind            TEXT NOT NULL DEFAULT 'software',
    name            TEXT NOT NULL,
    url             TEXT NOT NULL,
    description     TEXT,
    readme_excerpt  TEXT,
    license_spdx    TEXT,
    language        TEXT,
    topics          TEXT,
    stars           INTEGER,
    forks           INTEGER,
    open_issues     INTEGER,
    contributors    INTEGER,
    downloads       INTEGER,
    likes           INTEGER,
    param_count     INTEGER,
    created_at      TEXT,
    last_commit_at  TEXT,
    last_release_at TEXT,
    fetched_at      TEXT NOT NULL,
    etag            TEXT,
    UNIQUE (source, source_id)
);

CREATE TABLE IF NOT EXISTS health (
    project_id   INTEGER PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    score_total  REAL,
    components   TEXT,
    evidence     TEXT,                         -- JSON list of facts and links
    computed_at  TEXT NOT NULL,
    license_class TEXT,
    commercial_use TEXT,
    confidence   TEXT,                         -- verified_live | cached | low_data
    completeness REAL,
    label        TEXT                          -- healthy | fair | weak | inactive
);

CREATE TABLE IF NOT EXISTS embeddings (
    project_id INTEGER PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,
    model      TEXT NOT NULL,
    vector     BLOB NOT NULL,          -- float32, L2-normalized
    dim        INTEGER,
    text_hash  TEXT,                   -- hash of the embedded text, to detect stale vectors
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS profiles (
    code         TEXT PRIMARY KEY,
    mode         TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_code  TEXT REFERENCES profiles(code),
    results_json  TEXT,
    rejected_json TEXT,
    created_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_projects_license ON projects(license_spdx);
CREATE INDEX IF NOT EXISTS idx_projects_language ON projects(language);
CREATE INDEX IF NOT EXISTS idx_projects_kind ON projects(kind);

CREATE TABLE IF NOT EXISTS http_cache (
    cache_key  TEXT PRIMARY KEY,
    etag       TEXT,
    body       TEXT,
    link       TEXT,
    fetched_at TEXT NOT NULL
);

-- One query surface across GitHub and Hugging Face.
-- popularity: stars for software, likes for models/datasets (not directly comparable across sources).
CREATE VIEW IF NOT EXISTS projects_unified AS
SELECT id, source, source_id, kind, name, url, description, license_spdx, language, topics,
       COALESCE(stars, likes) AS popularity, downloads, param_count,
       last_commit_at AS last_activity_at, last_release_at, fetched_at
FROM projects;
