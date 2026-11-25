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

## Phase 3 (code done, needs a real-embedder check on your machine)
- db.py: _migrate() adds dim, text_hash, updated_at to older embeddings tables (existing databases keep working)
- config.py / .env.example: OPENPATH_EMBEDDER (default sentence-transformers/all-MiniLM-L6-v2; 'hashing' = offline baseline)
- index/embedder.py: HashingEmbedder (offline, word overlap only), SentenceTransformerEmbedder (lazy import,
  clear error if missing), get_embedder(name), tokenize() with stopwords and light plural stemming
- index/build.py: project_text(), build_index() is incremental (new, changed text, or different embedder model only);
  CLI: python -m openpath.index.build [--embedder X] [--rebuild]
- index/search.py: Filters(kind, sources, languages, licenses, require_license, max_age_months) -> semantic cosine
  + keyword boost (name 1.0 / topic 0.7 / description 0.4, weight 0.15); Searcher.search() returns Hit objects with
  semantic, keyword and total scores; CLI: python -m openpath.index.search "query" [filters] / --queries-file
- Filter semantics: language applies to software only (models pass through); max_age_months also drops unknown activity
- Vector store: SQLite + numpy brute force (no Chroma). Synthetic benchmark, 5000 projects with the hashing
  embedder: build 1.3 s, search about 80 ms. Revisit if the index grows beyond roughly 50k projects.
- Optional install for real embeddings: pip install -e ".[embed]"
- data/sample_queries.txt: 10 queries for the manual quality check
- tests/test_index.py: 20 tests (45 total passing). Real model never run in the sandbox (needs Hugging Face download).
- Known gaps: license filters take exact ids; mapping to license classes (permissive etc.) comes in Phase 4

## Phase 4 (code done; weights are judgment calls, review them on real data)
- scoring/licenses.json + licenses.py: ~100 SPDX ids and Hugging Face slugs mapped to 5 classes
  (permissive, weak_copyleft, strong_copyleft, network_copyleft, restricted_or_unknown), each with a commercial_use
  value (yes / yes_with_conditions / no / unclear) and a plain-language note. classify() handles +, -only,
  -or-later, WITH exceptions and simple OR (loosest wins) / AND (strictest wins) expressions. Anything not in
  the table becomes restricted_or_unknown with known=False. license_fit(info, constraint, commercial_use)
  returns (ok, plain-language reason) for Phase 6 rejection reasons. Informational only, not legal advice.
- scoring/config.json + config.py: all weights, score curves, label thresholds, confidence settings (editable;
  CLI accepts --config for experiments)
- scoring/health.py: components recency 0.30, documentation 0.20, release cadence 0.15, contributor spread 0.15,
  issue load 0.10, adoption 0.10. Total = weighted mean of components that have data. Models and datasets use
  recency, documentation, adoption only. Every component stores value, plain-language detail and a link.
  Missing release data is not punished. completeness = share of required components with data.
- scoring/confidence.py: low_data (no activity date or completeness < 0.6) beats verified_live (verified within
  24h, set by the Phase 6 verifier) beats cached (with 'data as of' date and a stale warning after 30 days)
- health table gained license_class, commercial_use, confidence, completeness, label (auto-migrated);
  CLI: python -m openpath.scoring.health [--rebuild] [--show owner/repo] [--config file.json]
- Search: Filters.license_classes / --license-class; hits now carry health_score, health_label, license_class,
  commercial_use, confidence (None until scoring has run)
- tests: tests/test_licenses.py, tests/test_health.py (107 total passing)
- Known gaps: no issue-response-time or bus-factor trend yet (Phase 16); thresholds are untested heuristics;
  the license table is not exhaustive and not legally reviewed

## Phase 5 (code done; real LLM providers untested live)
- profile_schema.json: added constraints.scale (personal | team | production)
- profile.py: sanitize_profile() (drops invalid optional fields, raises only for bad mode/need), apply_edits()
  (dotted keys like 'constraints.hosting', None removes), editable_fields() (allowed values for an edit form),
  profile_summary() (plain-language text the user can check)
- config.py / .env.example: OPENPATH_LLM (none | ollama | openai | anthropic), _MODEL, _BASE_URL, _API_KEY
- interview/llm.py: OllamaClient (local, free), OpenAICompatClient (any OpenAI-style API, retries without
  response_format on HTTP 400), AnthropicClient (paid), ScriptedLLM (tests), get_llm(); errors never include the key
- interview/agent.py: Interview.start(mode, need) / .answer(state, text) -> Turn(state, question, profile, done).
  At most 5 questions. LLM output must be JSON that validates against the profile schema; retried once with the
  validation errors; then falls back to the built-in flow (also on network errors). mode is enforced by us, not the
  LLM; user text is passed as data with an explicit ignore-instructions rule. State is plain JSON (resumable).
  The final answer is always integrated by one more LLM call, whose extra question is ignored.
- interview/fallback.py: deterministic flows (solve 5 questions, build 5, learn 4) with suggested options;
  answers match by number, label, or a unique substring; ambiguous or unclear answers are ignored, never guessed;
  questions already answered are skipped
- interview/store.py: save_profile() / load_profile() / share_url(); codes are 8 characters derived from the profile
  content (same profile = same code), forgiving of case and o/0, i/l/1; collisions extend the code
- interview/cli.py: python -m openpath.interview.cli --mode build [--need ...] [--no-llm] [--save]
- tests: test_llm.py, test_profile_tools.py, test_interview.py (155 total passing); personas for all three modes
  run both with and without an LLM
- Known gaps: free-text answers are only understood by the LLM path (the built-in flow maps options and
  languages); the edit form itself arrives with the UI in Phase 7; profiles are not yet fed into search (Phase 6)

## Next: Phase 6 (ranker, live verifier, explainer, why-not)
