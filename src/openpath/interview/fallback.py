"""Built-in question flow used when no LLM is configured or the LLM fails. Fully deterministic."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

NO_PREFERENCE = {"any", "anything", "no preference", "none", "doesn't matter", "does not matter", "dont care", "don't care", "skip"}


@dataclass
class Question:
    text: str
    options: list = field(default_factory=list)
    id: Optional[str] = None

    def to_dict(self) -> dict:
        return {"text": self.text, "options": list(self.options), "id": self.id}

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(d["text"], list(d.get("options") or []), d.get("id"))


@dataclass
class _Def:
    id: str
    text: str
    covers: list                                   # dotted fields this question determines
    options: list = field(default_factory=list)    # [(label, {dotted_field: value})]
    free: Optional[Callable[[str], dict]] = None   # parser for free-text answers


def _languages(text: str) -> dict:
    if text.strip().lower() in NO_PREFERENCE:
        return {}
    parts = [p.strip().lower() for p in re.split(r"[,/;]|\band\b|\bor\b", text) if p.strip()]
    return {"preferences.languages": parts[:5]} if parts else {}


def _o(label: str, **fields) -> tuple:
    return label, {k.replace("__", "."): v for k, v in fields.items()}


_DEFS = {d.id: d for d in [
    _Def("solve.hosting", "Would you rather run it yourself (on your own computer or server) or use a service hosted by someone else?",
         ["constraints.hosting"], [_o("Run it myself", constraints__hosting="self-hosted"),
                                   _o("Use a hosted service", constraints__hosting="cloud"),
                                   _o("No preference", constraints__hosting="any")]),
    _Def("solve.budget", "What is your budget for this?", ["constraints.budget"],
         [_o("Free only", constraints__budget="free"), _o("A little", constraints__budget="low"),
          _o("Flexible", constraints__budget="flexible")]),
    _Def("solve.commercial", "Will you use it for a business or to earn money?", ["constraints.commercial_use"],
         [_o("Yes", constraints__commercial_use=True), _o("No", constraints__commercial_use=False)]),
    _Def("solve.skill", "How comfortable are you with setting up technical tools?", ["constraints.skill_level"],
         [_o("Not at all", constraints__skill_level="beginner"), _o("A little", constraints__skill_level="intermediate"),
          _o("Very comfortable", constraints__skill_level="advanced")]),
    _Def("solve.data", "Will it handle private information, such as customer, health or financial data?",
         ["constraints.data_sensitivity"],
         [_o("Yes", constraints__data_sensitivity="high"), _o("Some", constraints__data_sensitivity="medium"),
          _o("No", constraints__data_sensitivity="low")]),

    _Def("build.kind", "Are you looking for a library or tool, or for an AI model?", ["preferences.kind"],
         [_o("Library or tool", preferences__kind="software"), _o("AI model", preferences__kind="model"),
          _o("Either", preferences__kind="any")]),
    _Def("build.languages", "Which programming languages do you prefer? (comma-separated, or say 'any')",
         ["preferences.languages"], free=_languages),
    _Def("build.license", "How will you use it?", ["constraints.license", "constraints.commercial_use"],
         [_o("Commercial product, keep my code private", constraints__commercial_use=True,
             constraints__license="permissive-or-weak-copyleft"),
          _o("Commercial, open source licenses are fine", constraints__commercial_use=True, constraints__license="copyleft-ok"),
          _o("Not commercial", constraints__commercial_use=False, constraints__license="any")]),
    _Def("build.hosting", "Where will it run?", ["constraints.hosting"],
         [_o("Self-hosted", constraints__hosting="self-hosted"), _o("Cloud", constraints__hosting="cloud"),
          _o("Locally only", constraints__hosting="local-only"), _o("Not sure", constraints__hosting="any")]),
    _Def("build.scale", "What scale do you expect?", ["constraints.scale"],
         [_o("Personal project", constraints__scale="personal"), _o("Small team", constraints__scale="team"),
          _o("Production", constraints__scale="production")]),

    _Def("learn.skill", "How would you describe your experience in this area?", ["constraints.skill_level"],
         [_o("Complete beginner", constraints__skill_level="beginner"), _o("Some experience", constraints__skill_level="intermediate"),
          _o("Experienced", constraints__skill_level="advanced")]),
    _Def("learn.languages", "Which programming language do you want to practice with? (or say 'any')",
         ["preferences.languages"], free=_languages),
    _Def("learn.hosting", "Do you want everything to run on your own laptop?", ["constraints.hosting"],
         [_o("Yes, local only", constraints__hosting="local-only"), _o("Cloud is fine", constraints__hosting="any")]),
    _Def("learn.budget", "Do you want to stay on free tools only?", ["constraints.budget"],
         [_o("Yes, free only", constraints__budget="free"), _o("Low cost is okay", constraints__budget="low")]),
]}

FLOWS = {
    "solve": ["solve.hosting", "solve.budget", "solve.commercial", "solve.skill", "solve.data"],
    "build": ["build.kind", "build.languages", "build.license", "build.hosting", "build.scale"],
    "learn": ["learn.skill", "learn.languages", "learn.hosting", "learn.budget"],
}


def _has(profile: dict, dotted: str) -> bool:
    section, name = dotted.split(".")
    return name in profile.get(section, {})


def _set(profile: dict, dotted: str, value) -> None:
    section, name = dotted.split(".")
    profile.setdefault(section, {})[name] = value


def next_question(mode: str, profile: dict, asked_ids: list) -> Optional[Question]:
    """The next unanswered question for this mode, skipping ones already covered by the profile."""
    for qid in FLOWS[mode]:
        d = _DEFS[qid]
        if qid in asked_ids or all(_has(profile, f) for f in d.covers):
            continue
        return Question(d.text, [label for label, _ in d.options], qid)
    return None


def apply_answer(profile: dict, qid: str, text: str) -> bool:
    """Map a free-text or option answer onto the profile. Returns False if it could not be understood."""
    d = _DEFS.get(qid)
    if d is None:
        return False
    answer = text.strip().lower()
    chosen = None
    if d.options:
        m = re.fullmatch(r"(\d+)[.)]?", answer)
        if m and 1 <= int(m.group(1)) <= len(d.options):
            chosen = d.options[int(m.group(1)) - 1][1]
        else:
            for label, fields in d.options:
                if answer == label.lower():
                    chosen = fields
                    break
            else:
                hits = [fields for label, fields in d.options if len(answer) >= 3 and (answer in label.lower() or label.lower() in answer)]
                chosen = hits[0] if len(hits) == 1 else None
    elif d.free:
        chosen = d.free(text)
    if not chosen:
        return False
    for dotted, value in chosen.items():
        _set(profile, dotted, value)
    return True
