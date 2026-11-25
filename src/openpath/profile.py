"""Requirement profile validation."""
import json
from importlib import resources
from jsonschema import Draft202012Validator

_schema = json.loads(
    resources.files("openpath").joinpath("profile_schema.json").read_text(encoding="utf-8")
)
_validator = Draft202012Validator(_schema)


def validate_profile(profile: dict) -> list[str]:
    """Return a list of human-readable errors (empty list means valid)."""
    return [
        f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}"
        for e in sorted(_validator.iter_errors(profile), key=lambda e: list(e.path))
    ]
