"""Property tests for the judge (tinytms/verify.py): thousands of generated witness triples, fixed seed.

Standard library only: `random.Random(SEED)` builds the same cases on every run, so a failure reproduces and the
message names the case. The generated witnesses are hand-built test inputs (provider FIXTURE), not model outputs.

The expected outcome of each case comes from how the case was BUILT (which load it describes, which seal it spells,
which field was broken on purpose), not from re-running the judge's own code. Properties, from the lane order:

  - any witness not OK -> HOLD, and the only reasons are NO_MODEL:<w> / ERROR:<w>;
  - a HOLD lists every failing check (never only the first); ACCEPT has zero reasons;
  - seal comparison is invariant to case, spaces and dashes (and to nothing else that is a letter or a digit);
  - `hazmat is True` always holds; None never does;
  - the decision is a pure function: same inputs -> same Verdict, `checks` included.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, replace

import pytest

from tinytms.verify import Limits, judge, normalize_seal
from tinytms.witness import DocWitness, VisionWitness, VoiceWitness

SEED = 20261029
N = 5000

FX = dict(model="FIXTURE", provider="FIXTURE")
CHECKS = ("witnesses", "count", "seal", "weight", "hazmat")
# Reason label -> the check it belongs to (rule order 2..5). Labels are the judge's contract.
FAMILY = {
    "DOC_NO_PIECES": "count", "VISION_NO_COUNT": "count", "VOICE_NO_COUNT": "count", "COUNT_MISMATCH": "count",
    "SEAL_MISSING": "seal", "SEAL_MISMATCH": "seal",
    "WEIGHT_MISSING": "weight", "OVERWEIGHT": "weight",
    "HAZMAT_REVIEW": "hazmat",
}
LABEL_ORDER = list(FAMILY)  # rule order, and inside the count rule: doc, vision, voice, then the mismatch

# Separators a seal may be written or read with: ASCII and typographic dashes, spaces of several kinds.
SEPARATORS = (" ", "  ", "-", " - ", "--", "‐", "‑", "–", "—", "−", " ", "\t", "​")
# Letters and digits a seal may carry. Non-ASCII ones too: a Carta de Porte can read "Ñ", a model can answer with a
# Cyrillic or full-width look-alike. None of these is a case variant of another (checked in _different_char).
ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "ÑÁÉÍÓÚÜñáéíóúü"
    "АМХОЕРС"  # Cyrillic look-alikes of A M X O E P C
    "ΑΜΧ"  # Greek look-alikes of A M X
    "０１２３４５６７８９ＭＸ"  # full-width
)
JUNK_COUNTS = (None, -1, -12, True, False, 12.0, "12", 1.5, float("nan"))
JUNK_WEIGHTS = (None, float("nan"), float("inf"), float("-inf"), -0.5, -1, True, False, "4800")
JUNK_SEALS = (None, "", " ", "-", " - ", "—", "\t")


# --- generation --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    vision: VisionWitness
    doc: DocWitness
    voice: VoiceWitness
    limits: Limits
    # what the case was built to be
    seal_ids: tuple[int | None, int | None]  # (doc, voice): which core each seal spells; None = absent/junk

    def args(self):
        return self.vision, self.doc, self.voice, self.limits


def _core(rng: random.Random) -> str:
    """A seal identity: 0-3 letters then 3-6 digits, ASCII upper case ('MX48213')."""
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(rng.randint(0, 3)))
    return letters + "".join(rng.choice("0123456789") for _ in range(rng.randint(3, 6)))


def spell(rng: random.Random, core: str) -> str:
    """The same seal written another way: random case per letter, random separators between characters."""
    out = [rng.choice(SEPARATORS)] if rng.random() < 0.2 else []
    for ch in core:
        out.append(ch.lower() if rng.random() < 0.5 else ch)
        if rng.random() < 0.3:
            out.append(rng.choice(SEPARATORS))
    return "".join(out)


def _count(rng: random.Random, n: int) -> object:
    r = rng.random()
    if r < 0.75:
        return n
    if r < 0.88:
        return n + rng.choice((-3, -1, 1, 2, 10))
    return rng.choice(JUNK_COUNTS)


def _weight(rng: random.Random, limit: float) -> object:
    r = rng.random()
    if r < 0.70:
        return rng.uniform(0, 24000)
    if r < 0.78 and math.isfinite(limit):
        return limit  # the limit is inclusive
    if r < 0.88:
        return rng.uniform(24000, 60000)
    return rng.choice(JUNK_WEIGHTS)


def _limit(rng: random.Random) -> float:
    r = rng.random()
    if r < 0.80:
        return 24000.0
    if r < 0.93:
        return rng.uniform(0, 40000)
    return rng.choice((float("inf"), float("nan"), -1.0, 0.0))


def _status(rng: random.Random) -> str:
    return "OK" if rng.random() < 0.85 else rng.choice(("NO_MODEL", "ERROR"))


def make_case(rng: random.Random) -> Case:
    n = rng.randint(0, 40)
    cores = [_core(rng), _core(rng)]
    while normalize_seal(cores[1]) == normalize_seal(cores[0]):
        cores[1] = _core(rng)

    def seal() -> tuple[object, int | None]:
        r = rng.random()
        if r < 0.80:
            return spell(rng, cores[0]), 0
        if r < 0.90:
            return spell(rng, cores[1]), 1
        return rng.choice(JUNK_SEALS), None

    limit = _limit(rng)
    (ds, di), (vs, vi) = seal(), seal()
    vision = VisionWitness(status=_status(rng), count=_count(rng, n), **FX)
    doc = DocWitness(status=_status(rng), pieces=_count(rng, n), weight_kg=_weight(rng, limit), seal=ds,
                     hazmat=rng.choice((True, False, None, None, False)), **FX)
    voice = VoiceWitness(status=_status(rng), count=_count(rng, n), seal=vs, **FX)
    return Case(vision, doc, voice, Limits(max_weight_kg=limit), (di, vi))


CASES = [make_case(random.Random(f"{SEED}:{i}")) for i in range(N)]
OK_CASES = [c for c in CASES if all(w.status == "OK" for w in (c.vision, c.doc, c.voice))]


def _who(i: int, c: Case) -> str:
    return f"case {i} (seed {SEED}:{i}): {c!r}"


# --- the spec, written from the README rules, not from verify.py --------------------------------------------------


def is_count(n: object) -> bool:
    return type(n) is int and n >= 0


def expected_failing_checks(c: Case) -> set[str]:
    fails = set()
    counts = (c.doc.pieces, c.vision.count, c.voice.count)
    if not all(is_count(x) for x in counts) or len(set(counts)) > 1:
        fails.add("count")
    di, vi = c.seal_ids
    if di is None or vi is None or di != vi:
        fails.add("seal")
    w = c.doc.weight_kg
    weight_ok = type(w) in (int, float) and math.isfinite(w) and 0 <= w <= c.limits.max_weight_kg
    if not weight_ok:  # a NaN limit compares false: no weight is known to be under it
        fails.add("weight")
    if c.doc.hazmat is True:
        fails.add("hazmat")
    return fails


def expected_count_reasons(c: Case) -> list[str]:
    out = [label for label, x in (("DOC_NO_PIECES", c.doc.pieces), ("VISION_NO_COUNT", c.vision.count),
                                  ("VOICE_NO_COUNT", c.voice.count)) if not is_count(x)]
    seen = [(name, x) for name, x in (("doc", c.doc.pieces), ("vision", c.vision.count), ("voice", c.voice.count))
            if is_count(x)]
    if len({x for _, x in seen}) > 1:
        out.append("COUNT_MISMATCH " + " ".join(f"{name}={x}" for name, x in seen))
    return out


def label(reason: str) -> str:
    return reason.split()[0]


# --- properties ---------------------------------------------------------------------------------------------------


def test_generator_covers_every_outcome():
    """The corpus is only worth something if it reaches every branch: guard against a silent generator change."""
    assert len(CASES) == N and len(OK_CASES) > N // 2
    verdicts = [judge(*c.args()) for c in OK_CASES]
    assert sum(v.decision == "ACCEPT" for v in verdicts) > 100
    seen = {label(r) for v in verdicts for r in v.reasons}
    assert seen == set(FAMILY), f"labels never produced: {set(FAMILY) - seen}"
    not_ok = [c for c in CASES if any(w.status != "OK" for w in (c.vision, c.doc, c.voice))]
    assert len(not_ok) > 500
    assert {len([w for w in (c.vision, c.doc, c.voice) if w.status != "OK"]) for c in not_ok} == {1, 2, 3}


def test_any_witness_not_ok_holds_with_only_witness_reasons():
    for i, c in enumerate(CASES):
        down = [(name, w.status) for name, w in (("vision", c.vision), ("doc", c.doc), ("voice", c.voice))
                if w.status != "OK"]
        if not down:
            continue
        v = judge(*c.args())
        assert v.decision == "HOLD", _who(i, c)
        assert v.reasons == tuple(f"{s}:{name}" for name, s in down), _who(i, c)
        assert all(re.fullmatch(r"(NO_MODEL|ERROR):(vision|doc|voice)", r) for r in v.reasons), _who(i, c)
        assert v.checks == {"witnesses": "fail", "count": "skipped", "seal": "skipped", "weight": "skipped",
                            "hazmat": "skipped"}, _who(i, c)


def test_hold_lists_every_failing_check_and_accept_has_no_reasons():
    for i, c in enumerate(OK_CASES):
        v = judge(*c.args())
        want = expected_failing_checks(c)
        failed = {k for k, s in v.checks.items() if s == "fail"}
        assert tuple(v.checks) == CHECKS and set(v.checks.values()) <= {"pass", "fail"}, _who(i, c)
        assert failed == want, f"{_who(i, c)}\nfailed={failed} want={want} reasons={v.reasons}"
        # every failing check has at least one reason, and no reason belongs to a passing check
        assert {FAMILY[label(r)] for r in v.reasons} == want, _who(i, c)
        assert [label(r) for r in v.reasons if FAMILY[label(r)] == "count"] == \
               [label(r) for r in expected_count_reasons(c)], _who(i, c)
        assert [r for r in v.reasons if FAMILY[label(r)] == "count"] == expected_count_reasons(c), _who(i, c)
        if v.decision == "ACCEPT":
            assert v.reasons == () and not want and set(v.checks.values()) == {"pass"}, _who(i, c)
        else:
            assert v.decision == "HOLD" and v.reasons and want, _who(i, c)


def test_reasons_follow_the_rule_order():
    for i, c in enumerate(OK_CASES):
        idx = [LABEL_ORDER.index(label(r)) for r in judge(*c.args()).reasons]
        assert idx == sorted(idx) and len(idx) == len(set(idx)), _who(i, c)


def test_seal_reason_names_what_is_missing_or_both_normalised_values():
    for i, c in enumerate(OK_CASES):
        (r,) = [r for r in judge(*c.args()).reasons if FAMILY[label(r)] == "seal"] or ["(none)"]
        di, vi = c.seal_ids
        if di is None or vi is None:
            missing = ",".join(n for n, x in (("doc", di), ("voice", vi)) if x is None)
            assert r == f"SEAL_MISSING {missing}", _who(i, c)
        elif di != vi:
            assert r == f"SEAL_MISMATCH doc={normalize_seal(c.doc.seal)} voice={normalize_seal(c.voice.seal)}", \
                _who(i, c)
        else:
            assert r == "(none)", _who(i, c)


def test_seal_comparison_is_invariant_to_case_spaces_and_dashes():
    """Re-spell both seals of every case (new case, new separators): the whole Verdict must not move."""
    rng = random.Random(f"{SEED}:respell")
    for i, c in enumerate(CASES):
        ds = spell(rng, normalize_seal(c.doc.seal)) if c.seal_ids[0] is not None else c.doc.seal
        vs = spell(rng, normalize_seal(c.voice.seal)) if c.seal_ids[1] is not None else c.voice.seal
        again = replace(c, doc=replace(c.doc, seal=ds), voice=replace(c.voice, seal=vs))
        assert judge(*again.args()) == judge(*c.args()), f"{_who(i, c)}\nrespelled doc={ds!r} voice={vs!r}"


def _different_char(rng: random.Random, ch: str) -> str:
    while True:
        x = rng.choice(ALPHABET)
        if x.upper() != ch.upper():
            return x


def test_seal_comparison_is_invariant_to_nothing_else():
    """One letter or digit more, or one changed (not a case change), is a different seal: the judge must HOLD.

    Invariance alone is cheap (a judge that ignores seals is invariant to everything); this is the other half.
    """
    rng = random.Random(f"{SEED}:mutate")
    for i, c in enumerate(OK_CASES):
        if c.seal_ids[0] is None or c.seal_ids[0] != c.seal_ids[1]:
            continue
        s = c.voice.seal
        pos = [k for k, ch in enumerate(s) if ch.isalnum()]
        k = rng.choice(pos)
        x = rng.choice(ALPHABET)
        for mutated in (s[:k] + x + s[k:], s[:k] + _different_char(rng, s[k]) + s[k + 1:]):
            v = judge(c.vision, c.doc, replace(c.voice, seal=mutated), c.limits)
            assert v.checks["seal"] == "fail" and v.decision == "HOLD", \
                f"{_who(i, c)}\ndoc seal {c.doc.seal!r} matched mutated voice seal {mutated!r}"


def test_hazmat_true_always_holds_and_none_never_does():
    for i, c in enumerate(OK_CASES):
        for hz in (True, False, None):
            v = judge(c.vision, replace(c.doc, hazmat=hz), c.voice, c.limits)
            assert ("HAZMAT_REVIEW" in v.reasons) is (hz is True), _who(i, c)
            assert v.checks["hazmat"] == ("fail" if hz is True else "pass"), _who(i, c)
            if hz is True:
                assert v.decision == "HOLD", _who(i, c)


def test_judge_is_a_pure_function():
    """Same inputs -> same Verdict (checks included), whatever was judged before, and equal copies agree."""
    first = [judge(*c.args()) for c in CASES]
    order = list(range(N))
    random.Random(f"{SEED}:order").shuffle(order)
    for i in order:
        c = CASES[i]
        copy = Case(replace(c.vision), replace(c.doc), replace(c.voice), replace(c.limits), c.seal_ids)
        v = judge(*copy.args())
        assert v == first[i] and v.checks == first[i].checks and v.to_dict() == first[i].to_dict(), _who(i, c)
        v.checks["seal"] = "tampered"  # a returned verdict shares no state with the next call
    assert [judge(*c.args()) for c in CASES] == first


@pytest.mark.parametrize("limit", [float("nan")])
def test_a_limit_that_is_not_a_number_lets_no_weight_through(limit):
    """Focused: `--max-weight-kg nan` parses as a float; the weight rule must fail closed on it."""
    c = next(c for c in OK_CASES if expected_failing_checks(c) == set())
    v = judge(c.vision, c.doc, c.voice, Limits(max_weight_kg=limit))
    assert v.decision == "HOLD" and [label(r) for r in v.reasons] == ["OVERWEIGHT"] and v.checks["weight"] == "fail"
