from dataclasses import replace

import pytest

from tinytms.adapters import parse_count, parse_seal
from tinytms.verify import Limits, judge
from tinytms.witness import DocWitness, VisionWitness, VoiceWitness

# Hand-built witnesses (FIXTURE provider): unit inputs for the judge, not model outputs.
V = VisionWitness(status="OK", model="FIXTURE", provider="FIXTURE", count=12)
D = DocWitness(status="OK", model="FIXTURE", provider="FIXTURE", pieces=12, weight_kg=8450.5,
               seal="MX-48213", hazmat=False)
A = VoiceWitness(status="OK", model="FIXTURE", provider="FIXTURE", count=12, seal="mx 48213")
L = Limits(max_weight_kg=24000)


def test_accept():
    v = judge(V, D, A, L)
    assert v.decision == "ACCEPT"
    assert v.reasons == ()
    assert set(v.checks.values()) == {"pass"}


def test_count_mismatch():
    v = judge(replace(V, count=11), D, A, L)
    assert v.decision == "HOLD"
    assert v.reasons == ("COUNT_MISMATCH doc=12 vision=11 voice=12",)
    assert v.checks["count"] == "fail"


def test_seal_mismatch():
    v = judge(V, D, replace(A, seal="MX-48231"), L)
    assert v.decision == "HOLD"
    assert v.reasons == ("SEAL_MISMATCH doc=MX48213 voice=MX48231",)


@pytest.mark.parametrize("doc_seal,voice_seal", [(None, "MX-48213"), ("MX-48213", None), ("  ", None)])
def test_seal_missing(doc_seal, voice_seal):
    v = judge(V, replace(D, seal=doc_seal), replace(A, seal=voice_seal), L)
    assert v.decision == "HOLD"
    assert len(v.reasons) == 1 and v.reasons[0].startswith("SEAL_MISSING")


def test_overweight():
    v = judge(V, replace(D, weight_kg=24000.5), A, L)
    assert v.decision == "HOLD"
    assert v.reasons[0].startswith("OVERWEIGHT")
    assert judge(V, replace(D, weight_kg=24000.0), A, L).decision == "ACCEPT"  # limit is inclusive


def test_missing_weight_fails_closed():
    v = judge(V, replace(D, weight_kg=None), A, L)
    assert v.decision == "HOLD" and v.reasons == ("WEIGHT_MISSING",)


@pytest.mark.parametrize("which", ["vision", "doc", "voice"])
@pytest.mark.parametrize("status", ["NO_MODEL", "ERROR"])
def test_any_witness_not_ok_holds(which, status):
    ws = {"vision": V, "doc": D, "voice": A}
    ws[which] = replace(ws[which], status=status)
    v = judge(ws["vision"], ws["doc"], ws["voice"], L)
    assert v.decision == "HOLD"
    assert v.reasons == (f"{status}:{which}",)
    assert v.checks["witnesses"] == "fail" and v.checks["count"] == "skipped"


def test_all_witnesses_down_lists_each():
    v = judge(replace(V, status="NO_MODEL"), replace(D, status="NO_MODEL"), replace(A, status="ERROR"), L)
    assert v.reasons == ("NO_MODEL:vision", "NO_MODEL:doc", "ERROR:voice")


def test_hazmat_review():
    v = judge(V, replace(D, hazmat=True), A, L)
    assert v.decision == "HOLD" and v.reasons == ("HAZMAT_REVIEW",)


def test_voice_without_number():
    v = judge(V, D, replace(A, count=None), L)
    assert v.decision == "HOLD"
    assert v.reasons == ("VOICE_NO_COUNT",)


def test_bool_is_not_a_count():
    assert judge(replace(V, count=True), replace(D, pieces=1), replace(A, count=1), L).decision == "HOLD"


def test_all_discrepancies_reported_in_order():
    v = judge(replace(V, count=10), replace(D, weight_kg=30000.0, hazmat=True), replace(A, seal="X1"), L)
    kinds = [r.split()[0] for r in v.reasons]
    assert kinds == ["COUNT_MISMATCH", "SEAL_MISMATCH", "OVERWEIGHT", "HAZMAT_REVIEW"]


def test_judge_is_deterministic():
    assert judge(V, D, A, L) == judge(V, D, A, L)


# --- transcript parsing feeding the voice witness ---


@pytest.mark.parametrize("text,count", [
    ("Traigo doce tarimas, el sello es MX-48213.", 12),
    ("Son 12 tarimas", 12),
    ("veinte y cuatro cajas", 24),
    ("Twenty-four pallets, seal 991", 24),
    ("Twelve pallets, seal number MX-48213.", 12),
    ("traigo una tarima", 1),
])
def test_parse_count(text, count):
    assert parse_count(text)[0] == count


@pytest.mark.parametrize("text", ["Ya llegué al andén", "doce tarimas y trece cajas"])
def test_parse_count_none_or_ambiguous(text):
    assert parse_count(text)[0] is None


@pytest.mark.parametrize("text,seal", [
    ("Traigo doce tarimas, el sello es MX-48213.", "MX-48213"),
    ("seal number is 4 8 2 1 3", "48213"),
    ("el sello es cuatro ocho dos uno tres", "48213"),
    ("Son doce tarimas", None),
])
def test_parse_seal(text, seal):
    assert parse_seal(text) == seal


def test_count_mismatch_is_reported_even_when_one_count_is_missing():
    # Found live on 2026-10-28: doc=12 vs vision=1 stayed silent because the voice count was None.
    v = judge(replace(V, count=1), D, replace(A, count=None), L)
    assert v.decision == "HOLD"
    assert "VOICE_NO_COUNT" in v.reasons
    assert "COUNT_MISMATCH doc=12 vision=1" in v.reasons
    assert v.checks["count"] == "fail"


# --- found by tests/test_verify_properties.py (2026-10-29), fixed in the judge ---


def test_nan_weight_limit_fails_closed():
    # `--max-weight-kg nan` is a float to argparse; `w > nan` is False, so every weight used to pass.
    v = judge(V, D, A, Limits(max_weight_kg=float("nan")))
    assert v.decision == "HOLD" and v.reasons == ("OVERWEIGHT doc=8450.5kg max=nankg",) and v.checks["weight"] == "fail"
    assert judge(V, D, A, Limits(max_weight_kg=float("inf"))).decision == "ACCEPT"  # no limit is a choice


@pytest.mark.parametrize("doc_seal,voice_seal", [
    ("MXÑ-48213", "MX 48213"),  # a letter the old ASCII filter dropped
    ("MX-48213", "MX-4821З"),  # Cyrillic Ze for a 3: a look-alike is not the seal
    ("МX-48213", "X 48213"),  # Cyrillic Em dropped on one side made them equal
    ("MX-48213", "MX-48213５"),  # full-width digit appended
])
def test_seal_letters_and_digits_beyond_ascii_are_not_dropped(doc_seal, voice_seal):
    v = judge(V, replace(D, seal=doc_seal), replace(A, seal=voice_seal), L)
    assert v.decision == "HOLD" and v.reasons[0].startswith("SEAL_MISMATCH"), v.reasons


def test_seal_normalisation_is_unchanged_for_ascii():
    import random
    import re

    from tinytms.verify import normalize_seal

    rng = random.Random(20261029)
    for _ in range(3000):
        s = "".join(rng.choice("abcXYZ019 -_./#\t") for _ in range(rng.randint(0, 12)))
        old = re.sub(r"[^0-9A-Za-z]", "", s).upper() or None
        assert normalize_seal(s) == old, s
    assert normalize_seal("mx–48 213") == "MX48213"  # en dash
    assert normalize_seal("Ñ-1") == "Ñ1" and normalize_seal("ñ-1") == "Ñ1"  # case-insensitive beyond ASCII
    assert normalize_seal("Ñ-1") == "Ñ1"  # the same letter written decomposed is the same letter
    assert normalize_seal("—") is None and normalize_seal("​") is None
