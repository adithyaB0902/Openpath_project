# OpenPath changelog

## Phase 0 (done)
- Scaffold: src/openpath/{ingest,index,scoring,interview,rank,explain,ui}, tests, data, scripts
- config.py: Settings from env (GITHUB_TOKEN, HF_TOKEN, OPENPATH_DB)
- schema.sql + db.py: init_db(path) creates projects, health, embeddings, profiles, runs
- profile_schema.json + profile.py: validate_profile(dict) -> list of error strings
- Tests: tests/test_db.py, tests/test_profile.py


## Phase 1 (code done, needs a live check with your token)
- schema.sql: added http_cache table (ETag cache)
- ingest/github.py:
  - GitHubClient: ETag conditional requests (304 served from cache), rate-limit waits
    (Retry-After / X-RateLimit-Reset / secondary limit), exponential backoff on 5xx and network errors,
    search throttling (2.2s with token, 6.5s without)
  - ingest(conn, client, topics, languages, limit, min_stars, max_age_days, light): upserts each repo
    immediately (resumable); repos fetched within max_age_days are skipped and do not count toward limit
  - Stored: description, README excerpt (3000 chars), SPDX license (raw, may be NOASSERTION), stars, forks,
    open issues, contributors, last commit, last release, topics
  - Optional fields failing (e.g. contributor list too large) never abort a repo
  - CLI: python -m openpath.ingest.github --topics a,b --languages python --limit 500 [--light]
- tests/test_github_ingest.py: 12 mocked tests (14 total passing)
- Known gap: not yet verified against the live API (anonymous limit was exhausted in the build sandbox)

## Phase 2 (code done, needs a live check with your token)
- ingest/common.py: shared upsert_project and is_fresh (github.py refactored to use them)
- schema.sql: added view projects_unified (popularity = stars for software, likes for models/datasets;
  last_activity_at = last_commit_at, which holds last_modified for Hugging Face)
- ingest/huggingface.py:
  - Models per pipeline task (and optional datasets per filter), most downloaded first;
    listing stops below --min-downloads; uses `expand` with automatic fallback to full=True
  - Stores: license (HF slug, lowercase, e.g. apache-2.0 / llama3 / other: NOT always SPDX, Phase 4 must map it),
    downloads (last 30 days), likes, parameter count (safetensors total), topics (pipeline tag, library, tags),
    model card excerpt (front matter stripped, 3000 chars), description from first card paragraph
  - Datasets stored with source_id 'datasets/<id>' so they never collide with a model of the same name
  - HF rows have language = NULL (library goes into topics); stars/forks/issues/contributors = NULL
  - Card fetch retries on 429/5xx; missing cards never abort an item; private/disabled repos skipped
  - CLI: python -m openpath.ingest.huggingface --tasks text-generation --limit 500 [--light] [--datasets-filter ...]
- Added dependency: huggingface_hub
- tests/test_hf_ingest.py: 11 tests using real ModelInfo/DatasetInfo objects (25 total passing)
- Known gap: not yet run against the live Hub (huggingface.co is not reachable from the build sandbox)

## Next: Phase 3 (embeddings and hybrid retrieval)
