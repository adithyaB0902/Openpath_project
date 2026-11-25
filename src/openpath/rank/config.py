"""Ranking configuration loaded from rank/config.json."""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Optional


@lru_cache(maxsize=1)
def _default() -> dict:
    return json.loads(resources.files("openpath.rank").joinpath("config.json").read_text(encoding="utf-8"))


def load_rank_config(path: Optional[str] = None) -> dict:
    if path:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return _default()
