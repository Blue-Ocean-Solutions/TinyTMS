"""Live eval: run the three REAL Meta API witnesses over every case in fixtures/ and sign a receipt.

Needs MODEL_API_KEY in the environment (never on the command line). Without it: exit 3, nothing run.

    MODEL_API_KEY=... python scripts/live_eval.py [--fixtures fixtures] [--cases accept,overweight]

Media come from each case directory (load.*, doc.*, voice.*). The FIXTURE placeholder media are 1x1
images and silent audio: real photos/notes must replace them for a meaningful eval. Per case the
receipt stores the live witnesses, the judge's verdict on them, the hand-written expectation
(from *.FIXTURE.json) and whether they agree. Raw API errors are recorded literally.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinytms.adapters import doc_from_spark, from_fixture, vision_from_sam, voice_from_transcript  # noqa: E402
from tinytms.cli import FIXTURE_FILES  # noqa: E402
from tinytms.meta_api import KEY_ENV, ApiResult, MetaClient, NoKey  # noqa: E402
from tinytms.receipt import load_or_create_key, sign_receipt, write_receipt  # noqa: E402
from tinytms.verify import DEFAULT_MAX_WEIGHT_KG, Limits, judge  # noqa: E402
from tinytms.witness import DocWitness, VisionWitness, VoiceWitness, sha256_file  # noqa: E402


def _media(case: Path, stem: str) -> Path | None:
    hits = sorted(p for p in case.glob(f"{stem}.*") if p.suffix != ".json")
    real = [p for p in hits if "FIXTURE" not in p.name]
    return (real or hits or [None])[0]


def _call(fn, *args) -> tuple[ApiResult, float]:
    t0 = time.perf_counter()
    r = fn(*args)
    return r, round(time.perf_counter() - t0, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="fixtures")
    ap.add_argument("--cases", default="", help="comma list; default all")
    ap.add_argument("--max-weight-kg", type=float, default=DEFAULT_MAX_WEIGHT_KG)
    ap.add_argument("--receipts-dir", default="receipts")
    ap.add_argument("--keys-dir", default="keys")
    a = ap.parse_args()

    try:
        client = MetaClient()
    except NoKey:
        print(f"NO_MODEL: {KEY_ENV} is not set; live eval needs a real key. Nothing was run.", file=sys.stderr)
        return 3

    root = Path(a.fixtures)
    names = [c for c in a.cases.split(",") if c] or sorted(p.name for p in root.iterdir() if p.is_dir())
    limits = Limits(a.max_weight_kg)
    now = datetime.now(timezone.utc)
    results = []
    with client:
        for name in names:
            case = root / name
            media = {k: _media(case, k) for k in ("load", "doc", "voice")}
            if None in media.values():
                results.append({"case": name, "status": "ERROR", "error": f"missing media: {media}"})
                continue
            sha = {k: sha256_file(p) for k, p in media.items()}
            rv, tv = _call(client.sam_segment, media["load"], "pallet")
            rd, td = _call(client.spark_extract_document, media["doc"])
            ra, ta = _call(client.voice_transcribe, media["voice"])
            m = dict(provider="meta-api")
            v = (vision_from_sam(rv.data, model=rv.model, source_sha256=sha["load"], **m) if rv.data is not None
                 else VisionWitness(status="ERROR", model=rv.model, source_sha256=sha["load"], detail=rv.detail, **m))
            d = (doc_from_spark(rd.data, model=rd.model, source_sha256=sha["doc"], **m) if rd.data is not None
                 else DocWitness(status="ERROR", model=rd.model, source_sha256=sha["doc"], detail=rd.detail, **m))
            s = (voice_from_transcript(ra.data, model=ra.model, source_sha256=sha["voice"], **m) if ra.data is not None
                 else VoiceWitness(status="ERROR", model=ra.model, source_sha256=sha["voice"], detail=ra.detail, **m))
            live = judge(v, d, s, limits)
            exp = judge(*(from_fixture(case / FIXTURE_FILES[k]) for k in ("vision", "doc", "voice")), limits)  # type: ignore[arg-type]
            results.append({
                "case": name,
                "inputs": {k: {"name": p.name, "sha256": sha[k]} for k, p in media.items()},
                "witnesses": {"vision": v.to_dict(), "doc": d.to_dict(), "voice": s.to_dict()},
                "raw": {"vision": rv.data, "doc": rd.data, "voice": ra.data},
                "wall_s": {"vision": tv, "doc": td, "voice": ta},
                "verdict": live.to_dict(),
                "expected_from_fixture": exp.to_dict(),
                "agrees_with_fixture": live.decision == exp.decision and live.reasons == exp.reasons,
            })

    body = {
        "kind": "live-eval",
        "utc": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "command": " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv]),
        "limits": {"max_weight_kg": limits.max_weight_kg},
        "results": results,
    }
    path = write_receipt(sign_receipt(body, load_or_create_key(a.keys_dir)), a.receipts_dir,
                         "live-eval-" + now.strftime("%Y%m%dT%H%M%SZ"))
    for r in results:
        print(f"{r['case']}: {r.get('verdict', {}).get('decision', r.get('status'))} "
              f"agrees={r.get('agrees_with_fixture')} {r.get('verdict', {}).get('reasons', r.get('error'))}")
    print(f"receipt: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
