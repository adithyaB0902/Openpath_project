"""Confidence labels: how much to trust the data behind a result."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from openpath.scoring.config import load_config


def parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Confidence:
    label: str            # verified_live | cached | low_data
    as_of: Optional[str]  # date the data was last checked
    message: str


def confidence_label(
    completeness: float,
    has_activity: bool,
    fetched_at: Optional[str],
    verified_at: Optional[str] = None,
    now: Optional[datetime] = None,
    cfg: Optional[dict] = None,
) -> Confidence:
    """low_data beats verified_live beats cached.

    verified_at is set by the live verifier (Phase 6) when it has just re-checked the source.
    """
    conf = (cfg or load_config())["confidence"]
    now = now or datetime.now(timezone.utc)
    fetched = parse_dt(fetched_at)
    verified = parse_dt(verified_at)
    is_live = bool(verified and (now - verified).total_seconds() <= conf["live_hours"] * 3600)
    as_of = (verified if is_live else fetched)
    as_of_text = as_of.date().isoformat() if as_of else None

    if not has_activity or completeness < conf["min_completeness"]:
        prefix = "Verified live, but some" if is_live else "Some"
        return Confidence("low_data", as_of_text, f"{prefix} key information is missing, so treat this score with caution.")
    if is_live:
        return Confidence("verified_live", as_of_text, "Checked against the source just now.")
    msg = f"Data as of {as_of_text}." if as_of_text else "Data date unknown."
    if fetched and (now - fetched).days > conf["stale_days"]:
        msg += " This is more than %d days old and may be out of date." % conf["stale_days"]
    return Confidence("cached", as_of_text, msg)
