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

## Phase 2: ingest from Hugging Face
```
python -m openpath.ingest.huggingface --tasks text-generation,text-classification,summarization --limit 500
python -m openpath.ingest.huggingface --tasks text-generation --limit 50 --light     # no card fetch
python -m openpath.ingest.huggingface --no-models --datasets-filter task_categories:question-answering --limit 50
```
Query both sources together:
```
sqlite3 data/openpath.db "select source, kind, name, license_spdx, popularity from projects_unified order by popularity desc limit 20"
```

## Phase 3: embeddings and search
```
pip install -e ".[embed]"                        # real embeddings (downloads a ~90 MB model on first use)
python -m openpath.index.build                   # or: --embedder hashing  (offline baseline)
python -m openpath.index.search "self-hosted appointment reminders" -k 5
python -m openpath.index.search "pdf text extraction" --kind software --language python --max-age-months 12 --license mit
python -m openpath.index.search --queries-file data/sample_queries.txt -k 5
```
Re-running build only embeds new or changed projects.

### How to judge result quality
1. Run the sample queries file above (after ingesting a few hundred projects per source).
2. For each query, mark each of the top 5 as relevant or not.
3. Aim for at least 3 relevant results in the top 5 for 7 of the 10 queries.
4. Compare `--embedder hashing` with the default model. If the real model is not clearly better, something is wrong.
5. Save your judgments: they become the labeled test set for Phase 9.

## Phase 4: health, licenses, confidence
```
python -m openpath.scoring.health                       # score everything new or changed
python -m openpath.scoring.health --show owner/repo     # full breakdown with evidence links
python -m openpath.index.search "pdf text extraction" --license-class permissive --license-class weak_copyleft
```
Run the scoring step after each ingest. Weights and curves live in `src/openpath/scoring/config.json`;
the license table is `src/openpath/scoring/licenses.json`. Add any license you see flagged as "not in our table".

Sanity-check the scores on projects you know: the ones you consider well maintained should score
healthy (75+) and abandoned ones weak or inactive. If not, adjust the config and rerun with `--rebuild`.
