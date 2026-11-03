"""Measure the voice parser on a blind corpus pair written outside this repository (ORDER Nube-10, item 11).

    python -m tests.voice_blind GOOD.jsonl BAD.jsonl ROOT [ROOT ...]

GOOD.jsonl: one note per line, {"id", "lang", "text", "low", "count", "seal"}: a careful checker takes both values.
BAD.jsonl: {"id", "lang", "category", "bad_fields", "text", "low", "count", "seal", "count_said", "seal_said"}: each
field in bad_fields must give None; count_said / seal_said list every value said for it. "text" is the note
punctuated, "low" the same words lowercase with no punctuation. Every checkout at ROOT is run in a fresh interpreter.

Per form and checkout it prints ACCEPT on good notes (the paper says the note's own count and seal; vision agrees),
false ACCEPT on bad notes (ACCEPT against any paper built from the values said for a bad field), and each note's worst
field class as in tests/voice_notes.py: right, none_reason, none_empty, picked (one of two values said), wrong.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

from tests.voice_notes import parse_all

WORST = ("right", "none_reason", "none_empty", "picked", "wrong")

_JUDGE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from tinytms.verify import Limits, judge
from tinytms.witness import DocWitness, VisionWitness, VoiceWitness
out = []
for c, s, vc, vs in json.load(sys.stdin):
    v = VisionWitness(status="OK", model="F", provider="F", count=c)
    d = DocWitness(status="OK", model="F", provider="F", pieces=c, weight_kg=4800.0, seal=s, hazmat=False)
    w = VoiceWitness(status="OK", model="F", provider="F", count=vc, seal=vs)
    out.append(judge(v, d, w, Limits(max_weight_kg=24000)).decision == "ACCEPT")
print(json.dumps(out))
"""


def _norm(seal: str | None) -> str | None:
    return re.sub(r"[\W_]", "", seal).upper() if seal else None


def field(meant, said: list, got, why: str, seal: bool) -> str:
    f = _norm if seal else (lambda x: x)
    if got is None:
        return "none_reason" if why else "none_empty"
    if meant is not None and f(got) == f(meant):
        return "right"
    if len({f(x) for x in said}) > 1 and f(got) in {f(x) for x in said}:
        return "picked"
    return "wrong"


def _accepts(root: Path, jobs: list) -> list[bool]:
    import subprocess

    p = subprocess.run([sys.executable, "-I", "-c", _JUDGE, str(root)], input=json.dumps(jobs), text=True,
                       capture_output=True, check=True)
    return json.loads(p.stdout)


def measure(root: Path, good: list[dict], bad: list[dict]) -> dict:
    out = {}
    for form in ("text", "low"):
        g = parse_all(root, [n[form] for n in good])
        b = parse_all(root, [n[form] for n in bad])
        jobs, owners = [], []
        for k, (n, p) in enumerate(zip(good, g, strict=True)):
            jobs.append([n["count"], n["seal"], p[0], p[2]])
            owners.append(("good", k))
        for k, (n, p) in enumerate(zip(bad, b, strict=True)):
            counts = [n["count"]] if n["count"] is not None else n["count_said"]
            seals = [n["seal"]] if n["seal"] is not None else n["seal_said"]
            for c, s in itertools.product(counts, seals):
                jobs.append([c, s, p[0], p[2]])
                owners.append(("bad", k))
        accepted = {"good": set(), "bad": set()}
        for (kind, k), a in zip(owners, _accepts(root, jobs), strict=True):
            if a:
                accepted[kind].add(k)
        classes = {"good": Counter(), "bad": Counter()}
        for kind, notes, parsed in (("good", good, g), ("bad", bad, b)):
            for n, p in zip(notes, parsed, strict=True):
                said_c = n.get("count_said", [n["count"]])
                said_s = n.get("seal_said", [n["seal"]])
                c = field(n["count"], said_c, p[0], p[1], False)
                s = field(n["seal"], said_s, p[2], p[3], True)
                classes[kind][max(c, s, key=WORST.index)] += 1
        out[form] = {"good_accept": len(accepted["good"]), "good": len(good), "bad_false_accept": len(accepted["bad"]),
                     "bad": len(bad), "good_classes": dict(classes["good"]), "bad_classes": dict(classes["bad"])}
    return out


def main(argv: list[str]) -> None:
    good = [json.loads(line) for line in Path(argv[0]).read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [json.loads(line) for line in Path(argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
    for root in map(Path, argv[2:]):
        for form, r in measure(root, good, bad).items():
            good_c = " ".join(f"{c}={r['good_classes'].get(c, 0)}" for c in WORST)
            bad_c = " ".join(f"{c}={r['bad_classes'].get(c, 0)}" for c in WORST)
            print(f"{root.name} {form}: good ACCEPT {r['good_accept']}/{r['good']}, bad false ACCEPT "
                  f"{r['bad_false_accept']}/{r['bad']}; good {good_c}; bad {bad_c}")


if __name__ == "__main__":
    main(sys.argv[1:])
