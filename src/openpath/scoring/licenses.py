"""License classification and fit checks. Informational only, not legal advice."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import resources
from typing import Optional

DISCLAIMER = "This is general information, not legal advice. Read the actual license before relying on a project."

CLASS_ORDER = ["permissive", "weak_copyleft", "strong_copyleft", "network_copyleft", "restricted_or_unknown"]
COMMERCIAL_VALUES = ("yes", "yes_with_conditions", "no", "unclear")

CONSTRAINT_CLASSES = {
    "permissive": {"permissive"},
    "permissive-or-weak-copyleft": {"permissive", "weak_copyleft"},
    "copyleft-ok": {"permissive", "weak_copyleft", "strong_copyleft", "network_copyleft"},
}
CONSTRAINT_TEXT = {
    "permissive": "permissive licenses only",
    "permissive-or-weak-copyleft": "permissive or weak-copyleft licenses",
    "copyleft-ok": "an open source license",
}


@dataclass(frozen=True)
class LicenseInfo:
    raw: Optional[str]
    normalized: str
    license_class: str
    commercial_use: str
    note: str
    known: bool

    @property
    def label(self) -> str:
        return _table()["classes"][self.license_class]["label"]

    @property
    def summary(self) -> str:
        return f"{self.raw or 'No license detected'} ({self.label})"


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(resources.files("openpath.scoring").joinpath("licenses.json").read_text(encoding="utf-8"))


def _rank(info: LicenseInfo) -> int:
    return CLASS_ORDER.index(info.license_class)


def _normalize(token: str) -> str:
    t = re.sub(r"\s+", "-", token.strip().strip("()").strip().lower())
    return t[:-1] + "-or-later" if t.endswith("+") else t


def _from_entry(raw: Optional[str], norm: str, entry) -> LicenseInfo:
    cls_defs = _table()["classes"]
    over = entry if isinstance(entry, dict) else {"class": entry}
    cls = over["class"]
    return LicenseInfo(
        raw=raw, normalized=norm, license_class=cls,
        commercial_use=over.get("commercial_use", cls_defs[cls]["commercial_use"]),
        note=over.get("note", cls_defs[cls]["note"]), known=True,
    )


def _single(token: str) -> LicenseInfo:
    exception = None
    parts = re.split(r"\s+with\s+", token, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        token, exception = parts
    norm = _normalize(token)
    licenses = _table()["licenses"]
    for cand in (norm, re.sub(r"-only$", "", norm), re.sub(r"-or-later$", "", norm)):
        if cand in licenses:
            info = _from_entry(token.strip(), norm, licenses[cand])
            if exception:
                info = replace(info, note=f"{info.note} It comes with an exception ({exception.strip()}), so read the exact terms.")
            return info
    entry = {"class": "restricted_or_unknown", "commercial_use": "unclear",
             "note": f"The license '{token.strip()}' is not in our table of known licenses. Read it before relying on the project."}
    info = _from_entry(token.strip(), norm, entry)
    return replace(info, known=False)


def classify(raw: Optional[str]) -> LicenseInfo:
    """Classify a GitHub SPDX id or a Hugging Face license slug (or a simple OR / AND expression)."""
    text = (raw or "").strip()
    if not text:
        return replace(_single("none"), raw=None)
    options = []
    multi = False
    for or_part in re.split(r"\s+or\s+", text, flags=re.I):
        and_parts = re.split(r"\s+and\s+", or_part, flags=re.I)
        multi = multi or len(and_parts) > 1
        options.append(max((_single(p) for p in and_parts), key=_rank))  # AND: strictest applies
    best = min(options, key=_rank)                                       # OR: you may choose the loosest
    if len(options) > 1:
        best = replace(best, note=f"Offered under several licenses; the most permissive option ({best.normalized}) is described here. {best.note}")
    elif multi:
        best = replace(best, note=f"Several licenses apply at once; the strictest is described here. {best.note}")
    return replace(best, raw=text)


def license_fit(info: LicenseInfo, constraint: Optional[str], commercial_use: Optional[bool]) -> tuple[bool, str]:
    """Does this license meet the user's profile? Returns (ok, plain-language reason)."""
    name = info.raw or "No license"
    if commercial_use and info.commercial_use == "no":
        return False, f"{name} does not allow commercial use."
    if constraint not in CONSTRAINT_CLASSES:  # 'any' or unspecified
        if commercial_use and info.commercial_use == "unclear":
            return True, f"{name}: the license is unclear, so check it before using the project commercially."
        return True, f"{name} is {info.label}."
    if info.license_class in CONSTRAINT_CLASSES[constraint]:
        return True, f"{name} is {info.label}, which fits your requirement for {CONSTRAINT_TEXT[constraint]}."
    if info.license_class == "restricted_or_unknown":
        return False, f"The license ({info.raw or 'none stated'}) is unclear or restricted, so we cannot confirm it meets your requirement for {CONSTRAINT_TEXT[constraint]}."
    return False, f"{name} is {info.label}, which does not meet your requirement for {CONSTRAINT_TEXT[constraint]}."
