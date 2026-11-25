"""Shareable profile codes: save_profile() -> short code, load_profile(code) -> profile.

Codes are derived from the profile content, so the same profile always gets the same code.
Only profiles the user chooses to share are saved.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from openpath.profile import ProfileError, validate_profile

log = logging.getLogger("openpath.interview.store")

_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"  # Crockford base32 (no i, l, o, u)
_CODE_RE = re.compile(r"^[0-9abcdefghjkmnpqrstvwxyz]{8,16}$")


def canonical(profile: dict) -> str:
    return json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _code(text: str, length: int) -> str:
    n = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")
    chars = []
    for _ in range(length):
        n, r = divmod(n, 32)
        chars.append(_ALPHABET[r])
    return "".join(chars)


def normalize_code(code: str) -> Optional[str]:
    """Lowercase and fix look-alike characters (o->0, i/l->1, u->v). None if not a valid code shape."""
    c = (code or "").strip().lower().translate(str.maketrans("oilu", "011v"))
    return c if _CODE_RE.match(c) else None


def save_profile(conn: sqlite3.Connection, profile: dict) -> str:
    errors = validate_profile(profile)
    if errors:
        raise ProfileError(errors)
    text = canonical(profile)
    for length in range(8, 17):  # extend the code in the (very unlikely) event of a collision
        code = _code(text, length)
        row = conn.execute("SELECT profile_json FROM profiles WHERE code = ?", (code,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO profiles (code, mode, profile_json, created_at) VALUES (?, ?, ?, ?)",
                         (code, profile["mode"], text, datetime.now(timezone.utc).isoformat()))
            conn.commit()
            return code
        if row["profile_json"] == text:
            return code
    raise RuntimeError("could not allocate a profile code")  # pragma: no cover


def load_profile(conn: sqlite3.Connection, code: str) -> Optional[dict]:
    norm = normalize_code(code)
    if not norm:
        return None
    row = conn.execute("SELECT profile_json FROM profiles WHERE code = ?", (norm,)).fetchone()
    if row is None:
        return None
    try:
        profile = json.loads(row["profile_json"])
    except json.JSONDecodeError:
        log.warning("stored profile %s is corrupt", norm)
        return None
    if validate_profile(profile):
        log.warning("stored profile %s no longer matches the schema", norm)
        return None
    return profile


def share_url(code: str, base_url: str = "") -> str:
    return f"{base_url}?p={code}"
