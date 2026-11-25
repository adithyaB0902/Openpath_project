"""Hard rejection rules. Every rule returns a plain-language reason so the user can see why a candidate was excluded."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from openpath.scoring.confidence import parse_dt
from openpath.scoring.health import Health
from openpath.scoring.licenses import classify, license_fit

LANGUAGE_ALIASES = {"js": "javascript", "ts": "typescript", "py": "python", "golang": "go", "cpp": "c++",
                    "csharp": "c#", "c sharp": "c#", "node": "javascript", "nodejs": "javascript", "rs": "rust"}
DAYS_PER_MONTH = 30.44


@dataclass(frozen=True)
class Reason:
    code: str   # license | inactive | low_health | language | unavailable
    text: str

    def to_dict(self) -> dict:
        return {"code": self.code, "text": self.text}


def norm_language(name: Optional[str]) -> str:
    n = (name or "").strip().lower()
    return LANGUAGE_ALIASES.get(n, n)


def parse_min_activity(text: Optional[str]) -> Optional[int]:
    """'commits in last 6 months' -> 6; '2 years' -> 24; anything else -> None."""
    m = re.search(r"(\d+)\s*(month|year)", (text or "").lower())
    if not m:
        return None
    return int(m.group(1)) * (12 if m.group(2) == "year" else 1)


def evaluate_rules(project: Any, profile: dict, health: Health, cfg: dict, now: datetime) -> list[Reason]:
    """Reasons this candidate must be excluded (empty list = it passes)."""
    reasons: list[Reason] = []
    constraints, prefs = profile.get("constraints", {}), profile.get("preferences", {})
    kind = project["kind"]

    # 1. license vs the user's license and commercial-use requirements
    ok, why = license_fit(classify(project["license_spdx"]), constraints.get("license"), constraints.get("commercial_use"))
    if not ok:
        reasons.append(Reason("license", why))

    # 2. no recent activity
    limit = parse_min_activity(prefs.get("min_activity")) or (
        cfg["max_inactive_months_model"] if kind != "software" else cfg["max_inactive_months"])
    last = parse_dt(project["last_commit_at"])
    inactive = False
    if last:
        months = (now - last).days / DAYS_PER_MONTH
        if months > limit:
            inactive = True
            reasons.append(Reason("inactive", f"No activity for about {months:.0f} months (last update {last.date()}); "
                                              f"we require activity within {limit} months."))

    # 3. very low overall health (only if not already rejected for inactivity)
    if not inactive and health.score_total is not None and health.score_total < cfg["min_health"]:
        reasons.append(Reason("low_health", f"Health score is only {health.score_total:.0f}/100 ({health.label}), "
                                            f"below our minimum of {cfg['min_health']}."))

    # 4. programming language (software only; unknown language is not held against it)
    wanted = [norm_language(l) for l in prefs.get("languages", [])]
    lang = norm_language(project["language"])
    if kind == "software" and wanted and lang and lang not in wanted:
        reasons.append(Reason("language", f"Written in {project['language']}; you prefer {', '.join(prefs['languages'])}."))
    return reasons
