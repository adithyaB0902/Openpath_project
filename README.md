# OpenPath

Open source advisor that shows its work. See CHANGELOG.md for current status.

## Setup
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env             # then add your tokens
pytest
python scripts/init_db.py
```

## Phase 1: ingest from GitHub
```
python -m openpath.ingest.github --topics machine-learning,self-hosted --limit 500
python -m openpath.ingest.github --topics rag --languages python --limit 200 --light
```
- `--light` skips commits, releases and contributors (fewer API calls).
- Authenticated limits: 5000 core requests/hour, 30 searches/minute. A full (non-light) repo costs about 4 requests,
  so roughly 1000 repos per hour. The client waits automatically when limited; re-running is safe.
- Check progress: `sqlite3 data/openpath.db "select count(*) from projects"`
