# OpenPath

An open source advisor that shows its work. Describe what you need in plain words, answer a few questions, and get a
verified, explained shortlist of open source projects and models from GitHub and Hugging Face, with the reasons the
close matches that were excluded did not make the list.

Three modes share one engine: **Solve** (non-technical users), **Build** (developers), **Learn** (hands-on learners).

## What makes it different
- **Verified results only.** Every recommendation comes from an index of real projects and is re-checked against its
  source right before it is shown. Nothing is invented by an LLM.
- **Shows its work.** Health scores, license classes, and confidence labels all link to their evidence.
- **Says why not.** The closest matches that were excluded come back with a plain-language reason for each.
- **Works without an LLM.** The interview and explanations have built-in fallbacks; an LLM only improves wording.

## Status
Command-line version, phases 0-6 complete. The frontend (Phase 7) is next.

| Phase | What | State |
|---|---|---|
| 0 | Scaffold, schema, requirement profile | done |
| 1 | GitHub ingestion | done |
| 2 | Hugging Face ingestion | done |
| 3 | Embeddings and hybrid search | done |
| 4 | Health score, license classes, confidence | done |
| 5 | Interview, profile editing, share codes | done |
| 6 | Ranking, live verification, explanations, why-not | done |
| 7 | Streamlit frontend | next |
| 8 | Starter kits and Learn mode | planned |
| 9 | Evaluation and hardening | planned |
| 10 | Deployment | planned |

Tests use mocked services. The real GitHub and Hugging Face APIs, the embedding model, and LLM providers have not yet
been exercised end to end, so run the quick start below and report anything odd. See `CHANGELOG.md` for details and
known gaps per phase.

## Quick start
```
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,embed]"        # "embed" installs sentence-transformers (downloads a model on first use)
cp .env.example .env                 # Windows: copy .env.example .env, then add your tokens
pytest
```

## Typical workflow
```
python -m openpath.ingest.github --topics machine-learning,self-hosted --limit 500
python -m openpath.ingest.huggingface --tasks text-generation,text-classification --limit 500
python -m openpath.index.build                       # embeddings (re-run only embeds what changed)
python -m openpath.scoring.health                    # health scores and license classes
python -m openpath.interview.cli --mode build --save # a few questions, prints a share code
python -m openpath.rank.cli --code YOURCODE          # recommendations with explanations and "why not"
```
Useful extras:
```
python -m openpath.index.search "pdf text extraction" --kind software --language python --license-class permissive
python -m openpath.scoring.health --show owner/repo  # full health breakdown with evidence links
python -m openpath.rank.cli --need "pdf text extraction" --no-verify --no-llm   # quick offline test
```
Add `--light` to either ingest command for fewer API calls. Settings live in `.env` (see `.env.example`):
GitHub and Hugging Face tokens, the embedder, and an optional interview LLM (Ollama, any OpenAI-compatible API, or Claude).

## Layout
```
src/openpath/
  ingest/     github.py, huggingface.py, common.py     crawl sources into SQLite
  index/      embedder.py, build.py, search.py         embeddings and hybrid search
  scoring/    health.py, licenses.py, confidence.py    scores, license classes (data in *.json)
  interview/  agent.py, fallback.py, llm.py, store.py  questions, profiles, share codes
  rank/       ranker.py, rules.py, effort.py, verify.py, cli.py   (settings in config.json)
  explain/    explain.py                               grounded fit explanations
  ui/                                                  frontend (Phase 7)
tests/        pytest suite
data/         local database (git-ignored) and sample queries
```

## Important caveats
- Health weights, license mappings, and ranking weights are judgment calls. They live in JSON files so they can be tuned.
- License information is general, not legal advice. Read the actual license before relying on a project.
- Effort estimates come from documentation and popularity signals, not from testing an install.
- Never commit `.env`. It is git-ignored; if a token ever leaks, revoke it and create a new one.

## License
No license has been chosen yet. Until one is added, others have no permission to reuse this code.
Choose one (for example via GitHub's "Add file > Create new file > LICENSE" template picker) before sharing it.
