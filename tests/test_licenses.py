import pytest

from openpath.scoring import licenses as L


@pytest.mark.parametrize("raw, cls", [
    ("MIT", "permissive"), ("Apache-2.0", "permissive"), ("BSD-3-Clause", "permissive"), ("Unlicense", "permissive"),
    ("apache-2.0", "permissive"), ("cc-by-4.0", "permissive"),
    ("LGPL-2.1", "weak_copyleft"), ("MPL-2.0", "weak_copyleft"), ("epl-2.0", "weak_copyleft"),
    ("GPL-3.0", "strong_copyleft"), ("GPL-2.0-only", "strong_copyleft"), ("GPL-3.0-or-later", "strong_copyleft"),
    ("GPL-2.0+", "strong_copyleft"), ("cc-by-sa-4.0", "strong_copyleft"),
    ("AGPL-3.0", "network_copyleft"), ("AGPL-3.0-only", "network_copyleft"), ("SSPL-1.0", "network_copyleft"),
    ("cc-by-nc-4.0", "restricted_or_unknown"), ("llama3", "restricted_or_unknown"),
    ("openrail", "restricted_or_unknown"), ("BUSL-1.1", "restricted_or_unknown"),
])
def test_known_licenses(raw, cls):
    info = L.classify(raw)
    assert info.license_class == cls and info.known


def test_commercial_use_values():
    assert L.classify("MIT").commercial_use == "yes"
    assert L.classify("GPL-3.0").commercial_use == "yes_with_conditions"
    assert L.classify("cc-by-nc-4.0").commercial_use == "no"
    assert L.classify("llama3").commercial_use == "yes_with_conditions"


@pytest.mark.parametrize("raw", [None, "", "  ", "NOASSERTION", "other", "unknown", "Weird-License-1.0"])
def test_missing_or_unrecognized_is_unclear(raw):
    info = L.classify(raw)
    assert info.license_class == "restricted_or_unknown" and info.commercial_use == "unclear"


def test_unrecognized_is_flagged_not_known_and_explained():
    info = L.classify("Weird-License-1.0")
    assert not info.known and "not in our table" in info.note
    assert not L.classify("NOASSERTION").note.startswith("The license '")  # has its own message
    assert "No license was detected" in L.classify(None).note


def test_expressions():
    assert L.classify("MIT OR Apache-2.0").license_class == "permissive"
    assert L.classify("(MIT OR GPL-3.0)").license_class == "permissive"       # you may pick the loosest
    assert L.classify("MIT AND GPL-3.0").license_class == "strong_copyleft"   # the strictest applies
    assert L.classify("MIT OR GPL-3.0").raw == "MIT OR GPL-3.0"
    assert "most permissive option" in L.classify("MIT OR GPL-3.0").note
    exc = L.classify("GPL-2.0-only WITH Classpath-exception-2.0")
    assert exc.license_class == "strong_copyleft" and "exception" in exc.note


def test_summary_text():
    assert L.classify("MIT").summary == "MIT (permissive)"
    assert L.classify(None).summary == "No license detected (restricted or unclear)"


def test_table_is_consistent():
    table = L._table()
    for key, entry in table["licenses"].items():
        assert key == key.lower()
        cls = entry if isinstance(entry, str) else entry["class"]
        assert cls in L.CLASS_ORDER, key
        if isinstance(entry, dict) and "commercial_use" in entry:
            assert entry["commercial_use"] in L.COMMERCIAL_VALUES, key
    assert set(table["classes"]) == set(L.CLASS_ORDER)


@pytest.mark.parametrize("raw, constraint, commercial, ok", [
    ("MIT", "permissive", True, True),
    ("GPL-3.0", "permissive", True, False),
    ("LGPL-3.0", "permissive-or-weak-copyleft", True, True),
    ("GPL-3.0", "permissive-or-weak-copyleft", True, False),
    ("AGPL-3.0", "copyleft-ok", True, True),
    ("cc-by-nc-4.0", "any", True, False),
    ("cc-by-nc-4.0", "any", False, True),
    ("NOASSERTION", "any", True, True),
    ("NOASSERTION", "permissive", True, False),
    ("MIT", None, None, True),
])
def test_license_fit(raw, constraint, commercial, ok):
    assert L.license_fit(L.classify(raw), constraint, commercial)[0] is ok


def test_license_fit_reasons_are_plain_language():
    _, why = L.license_fit(L.classify("GPL-3.0"), "permissive", True)
    assert "strong copyleft" in why and "permissive licenses only" in why
    _, why = L.license_fit(L.classify("cc-by-nc-4.0"), "any", True)
    assert "commercial" in why
    _, why = L.license_fit(L.classify("NOASSERTION"), "permissive", True)
    assert "unclear or restricted" in why
    _, why = L.license_fit(L.classify("NOASSERTION"), "any", True)
    assert "check it before" in why
