"""Effort: how hard is this likely to be to start with? Deterministic and based only on stored facts."""
from __future__ import annotations

from typing import Any

from openpath.scoring.health import Health


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def effort_fit(health: Health, project: Any, profile: dict, cfg: dict) -> float:
    """0..1, higher = easier to get started. Well documented and widely used projects are easier."""
    e = cfg["effort"]
    doc = (health.components["documentation"]["score"] or 0) / 100
    adoption = (health.components["adoption"]["score"] or 0) / 100
    fit = e["doc_weight"] * doc + e["adoption_weight"] * adoption
    c = profile.get("constraints", {})
    params = project.get("param_count")
    if params and params > e["big_model_params"] and (c.get("skill_level") == "beginner" or c.get("hosting") == "local-only"):
        fit -= e["big_model_penalty"]
    return _clamp(fit)


def effort_card(health: Health, project: Any, profile: dict, cfg: dict) -> dict:
    """A short, honest effort summary. Real install steps come from the starter kit (Phase 8)."""
    e = cfg["effort"]
    fit = effort_fit(health, project, profile, cfg)
    level = "easy" if fit >= e["easy_at"] else "moderate" if fit >= e["moderate_at"] else "hard"
    signals = [health.components["documentation"]["detail"], health.components["adoption"]["detail"]]
    contributors = health.components.get("contributor_spread")
    if contributors and contributors["score"] is not None:
        signals.append(contributors["detail"])
    params = project.get("param_count")
    skills = [project["language"]] if project["kind"] == "software" and project.get("language") else []
    note = ""
    if params:
        signals.append(f"{params / 1e9:.1f}B parameters")
        if params > 7e9:
            note = "Models above roughly 7B parameters usually need a GPU with a lot of memory, a quantized version, or a hosted service (rough guide)."
    if level == "hard" and profile.get("constraints", {}).get("skill_level") == "beginner":
        note = (note + " " if note else "") + "This may be difficult for a beginner."
    return {"level": level, "score": round(fit, 2), "skills": skills, "signals": signals, "note": note,
            "basis": "Estimate from documentation and adoption signals, not from testing the install."}
