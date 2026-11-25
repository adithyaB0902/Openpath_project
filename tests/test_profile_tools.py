import pytest

from openpath import profile as P

GOOD = {
    "mode": "build", "need": "self-hosted appointment reminders for a small clinic",
    "constraints": {"license": "permissive-or-weak-copyleft", "commercial_use": True, "hosting": "self-hosted",
                    "budget": "free", "skill_level": "beginner", "scale": "team"},
    "preferences": {"languages": ["python"], "kind": "software"},
}


def test_scale_is_valid_and_bad_scale_is_not():
    assert P.validate_profile(GOOD) == []
    bad = {**GOOD, "constraints": {**GOOD["constraints"], "scale": "huge"}}
    assert any("scale" in e for e in P.validate_profile(bad))


def test_sanitize_drops_invalid_optional_fields():
    messy = {"mode": "build", "need": "a thing", "constraints": {"hosting": "on-prem", "budget": "free", "foo": 1},
             "preferences": {"languages": ["python", 5], "kind": "software", "extra": True}, "unknown_top": 1}
    clean = P.sanitize_profile(messy)
    assert P.validate_profile(clean) == []
    assert clean["constraints"] == {"budget": "free"}
    assert clean["preferences"] == {"kind": "software"}
    assert messy["constraints"]["hosting"] == "on-prem"  # input untouched


def test_sanitize_adds_missing_containers_but_not_missing_need():
    assert P.sanitize_profile({"mode": "learn", "need": "learn rust"})["constraints"] == {}
    with pytest.raises(P.ProfileError):
        P.sanitize_profile({"mode": "learn"})
    with pytest.raises(P.ProfileError):
        P.sanitize_profile({"mode": "hack", "need": "x y z"})


def test_apply_edits():
    new = P.apply_edits(GOOD, {"constraints.hosting": "cloud", "constraints.scale": None, "preferences.languages": ["go"]})
    assert new["constraints"]["hosting"] == "cloud" and "scale" not in new["constraints"]
    assert new["preferences"]["languages"] == ["go"] and GOOD["constraints"]["hosting"] == "self-hosted"
    assert P.apply_edits(GOOD, {"need": "something else"})["need"] == "something else"


def test_apply_edits_rejects_invalid():
    with pytest.raises(P.ProfileError):
        P.apply_edits(GOOD, {"constraints.hosting": "mars"})
    with pytest.raises(P.ProfileError, match="cannot edit"):
        P.apply_edits(GOOD, {"constraints.nonsense.deep": 1})


def test_editable_fields():
    f = P.editable_fields()
    assert f["constraints.hosting"] == ["any", "self-hosted", "cloud", "local-only"]
    assert f["constraints.commercial_use"] == [True, False] and f["preferences.languages"] == "list"
    assert f["mode"] == ["solve", "build", "learn"]


def test_summary_is_plain_language():
    s = P.profile_summary(GOOD)
    assert s.startswith("You need: self-hosted appointment reminders")
    assert "run it yourself" in s and "Budget: free only" in s and "Commercial use: yes" in s
    assert "Languages: python" in s and "a small team" in s
    assert P.profile_summary({"mode": "learn", "need": "learn rust", "constraints": {}, "preferences": {}}) == "You need: learn rust"
