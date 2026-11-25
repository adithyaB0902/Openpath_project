from openpath.profile import validate_profile

GOOD = {
    "mode": "build",
    "need": "self-hosted appointment reminders for a small clinic",
    "constraints": {"license": "permissive-or-weak-copyleft", "commercial_use": True,
                    "hosting": "self-hosted", "budget": "free", "skill_level": "beginner"},
    "preferences": {"languages": ["python"], "kind": "software"},
}


def test_valid_profile():
    assert validate_profile(GOOD) == []


def test_invalid_mode_and_missing_field():
    bad = {**GOOD, "mode": "hack"}
    assert any("mode" in e for e in validate_profile(bad))
    del bad["need"]
    assert any("need" in e for e in validate_profile(bad))
