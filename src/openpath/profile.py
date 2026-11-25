"""Requirement profile: validation, sanitizing, editing and plain-language summaries."""
import copy
import json
from importlib import resources
from typing import Any, Optional

from jsonschema import Draft202012Validator

_schema = json.loads(
    resources.files("openpath").joinpath("profile_schema.json").read_text(encoding="utf-8")
)
_validator = Draft202012Validator(_schema)


class ProfileError(ValueError):
    def __init__(self, errors: list):
        super().__init__("; ".join(errors))
        self.errors = errors


def _fmt(e) -> str:
    return f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}"


def validate_profile(profile: dict) -> list:
    """Return a list of human-readable errors (empty list means valid)."""
    return [_fmt(e) for e in sorted(_validator.iter_errors(profile), key=lambda e: list(e.path))]


def _walk(obj: Any, path: list) -> Any:
    for key in path:
        obj = obj[key]
    return obj


def sanitize_profile(profile: Any) -> dict:
    """Drop invalid or unknown optional fields so the profile validates.

    Raises ProfileError if the problem is in a required top-level field (mode, need).
    """
    p = copy.deepcopy(profile) if isinstance(profile, dict) else {}
    for section in ("constraints", "preferences"):
        if not isinstance(p.get(section), dict):
            p[section] = {}
    for _ in range(10):
        errors = sorted(_validator.iter_errors(p), key=lambda e: list(e.path))
        if not errors:
            return p
        for e in errors:
            path = list(e.path)
            if e.validator == "additionalProperties":
                obj, allowed = _walk(p, path), set(e.schema.get("properties", {}))
                for key in [k for k in obj if k not in allowed]:
                    del obj[key]
            elif len(path) >= 2:
                p[path[0]].pop(path[1], None)
            else:
                raise ProfileError([_fmt(e)])
    raise ProfileError(validate_profile(p))


def apply_edits(profile: dict, edits: dict) -> dict:
    """Return a new profile with edits applied. Keys: 'need', 'constraints.hosting', ...; None removes a field."""
    new = copy.deepcopy(profile)
    for dotted, value in edits.items():
        parts = dotted.split(".")
        if parts in (["need"], ["mode"]) and value is not None:
            new[parts[0]] = value
        elif len(parts) == 2 and parts[0] in ("constraints", "preferences"):
            section = new.setdefault(parts[0], {})
            if value is None:
                section.pop(parts[1], None)
            else:
                section[parts[1]] = value
        else:
            raise ProfileError([f"cannot edit {dotted!r}"])
    errors = validate_profile(new)
    if errors:
        raise ProfileError(errors)
    return new


def editable_fields() -> dict:
    """Allowed values per editable field, for building an edit form."""
    out: dict = {"mode": list(_schema["properties"]["mode"]["enum"])}
    for section in ("constraints", "preferences"):
        for name, spec in _schema["properties"][section]["properties"].items():
            if "enum" in spec:
                out[f"{section}.{name}"] = list(spec["enum"])
            elif spec.get("type") == "boolean":
                out[f"{section}.{name}"] = [True, False]
            else:
                out[f"{section}.{name}"] = "text" if spec.get("type") == "string" else "list"
    return out


_WORDS = {
    "constraints.license": {
        "any": "any license, as long as it is clear",
        "permissive": "permissive licenses only (most freedom)",
        "permissive-or-weak-copyleft": "permissive or weak-copyleft licenses (your own code can stay private)",
        "copyleft-ok": "any open source license, including copyleft",
    },
    "constraints.hosting": {"any": "no preference", "self-hosted": "run it yourself", "cloud": "hosted or cloud service",
                            "local-only": "runs only on your own machine"},
    "constraints.budget": {"free": "free only", "low": "low cost", "flexible": "flexible"},
    "constraints.skill_level": {"beginner": "beginner", "intermediate": "some experience", "advanced": "experienced"},
    "constraints.data_sensitivity": {"low": "not sensitive", "medium": "somewhat sensitive", "high": "private or sensitive"},
    "constraints.scale": {"personal": "personal use", "team": "a small team", "production": "production use"},
    "preferences.kind": {"software": "software or libraries", "model": "AI models", "any": "software or models"},
}


def profile_summary(profile: dict) -> str:
    """Plain-language summary the user can read and correct."""
    c, pr = profile.get("constraints", {}), profile.get("preferences", {})
    lines = [f"You need: {profile.get('need', '')}"]

    def add(label: str, key: str, value: Any) -> None:
        if value is not None:
            lines.append(f"{label}: {_WORDS.get(key, {}).get(value, value)}")

    add("Looking for", "preferences.kind", pr.get("kind"))
    if "commercial_use" in c:
        lines.append("Commercial use: " + ("yes" if c["commercial_use"] else "no"))
    add("License", "constraints.license", c.get("license"))
    add("Where it runs", "constraints.hosting", c.get("hosting"))
    add("Budget", "constraints.budget", c.get("budget"))
    add("Your experience", "constraints.skill_level", c.get("skill_level"))
    add("Data", "constraints.data_sensitivity", c.get("data_sensitivity"))
    add("Scale", "constraints.scale", c.get("scale"))
    if pr.get("languages"):
        lines.append("Languages: " + ", ".join(pr["languages"]))
    if pr.get("min_activity"):
        lines.append(f"Activity: {pr['min_activity']}")
    return "\n".join(lines)
