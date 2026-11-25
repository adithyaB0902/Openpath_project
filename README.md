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
