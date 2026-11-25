# OpenPath changelog

## Phase 0 (done)
- Scaffold: src/openpath/{ingest,index,scoring,interview,rank,explain,ui}, tests, data, scripts
- config.py: Settings from env (GITHUB_TOKEN, HF_TOKEN, OPENPATH_DB)
- schema.sql + db.py: init_db(path) creates projects, health, embeddings, profiles, runs
- profile_schema.json + profile.py: validate_profile(dict) -> list of error strings
- Tests: tests/test_db.py, tests/test_profile.py

## Next: Phase 1 (GitHub ingestion)
