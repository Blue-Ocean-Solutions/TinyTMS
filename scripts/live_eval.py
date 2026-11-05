"""Live eval: the three REAL Meta API witnesses over evals/, one signed receipt per call.

Needs MODEL_API_KEY in the environment (never on the command line). Without it: exit 3, nothing run.

    python scripts/live_eval.py [--evals evals] [--keys-dir keys-lane2] [--save-test-data tests/data] [--only sam,spark,voice,voice-es,sam-own] [--concepts television,tablet]

Cases (evals/PROCEDENCIA.md):
  SAM 3.1    evals/public/boxes-moving.jpg  x  concepts "cardboard box", "pallet"
  Spark 1.3  evals/doc/carta_sintetica.png -> JSON fields
  Voice      evals/voice/silencio_16k.wav  -> empty transcript expected            (case voice)
  Voice      evals/voice/es-*.wav          -> a Spanish voice note, if there is one  (case voice-es)
  SAM 3.1    evals/own/*.jpg  x  --concepts -> the author's own photos, EXIF removed (case sam-own, not a default)

boxes-pallet.jpg is no longer a case: it stays in evals/public/ only as the input of the 2026-10-28 receipts
(evals/public/PROCEDENCIA.md). The SYNTHETIC-TTS Spanish note was withdrawn for its license (evals/PROCEDENCIA.md);
with no evals/voice/es-*.wav the voice-es case is skipped and the script says so.

Each receipt holds model, provider meta-api, utc, seconds, sha256 of the input (original file), the request
WITHOUT base64 and the raw answer truncated to 2,000 chars with SAM mask data removed. Every call is also
appended to receipts/llamadas.json (budget caps enforced there). With --save-test-data the real answers
(masks removed, no key) are written to tests/data/real-<witness>-<utc>.json for the offline parser tests.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinytms.adapters import (  # noqa: E402
    call_record,
    doc_from_spark,
    strip_sse,
    vision_from_sam,
    voice_from_transcript,
)
from tinytms.meta_api import KEY_ENV, ApiResult, MetaClient, NoKey  # noqa: E402
from tinytms.receipt import load_or_create_key, sign_receipt, write_receipt  # noqa: E402
from tinytms.witness import DocWitness, VisionWitness, VoiceWitness, sha256_file  # noqa: E402

SAM_IMAGES = ("public/boxes-moving.jpg",)
SAM_CONCEPTS = ("cardboard box", "pallet")
DOC = "doc/carta_sintetica.png"
VOICE = "voice/silencio_16k.wav"  # case voice keeps its original untagged receipt and test-data names
VOICE_ES = "voice/es-*.wav"  # case voice-es: tagged with the note's own name
OWN_PHOTOS = "own/*.jpg"  # case sam-own: photos taken for this repo by its author, EXIF removed
OWN_CONCEPTS = ("television", "tablet")

Job = tuple[str, Path, str, str]  # kind, input, SAM concept, name tag


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in s.lower()).strip("-")


def _stamp(utc: str) -> str:
    return utc.replace("-", "").replace(":", "").replace(".", "")[:15] + "Z" if utc else "nodate"


def plan(root: Path, only: set[str], own_concepts: tuple[str, ...] = OWN_CONCEPTS) -> tuple[list[Job], list[str]]:
    """The calls to make for the selected cases, and a line for each case skipped because its input is absent."""
    jobs: list[Job] = []
    skipped: list[str] = []
    if "sam" in only:
        jobs += [("sam", root / img, c, f"{_slug(Path(img).stem)}-{_slug(c)}-")
                 for img in SAM_IMAGES for c in SAM_CONCEPTS]
    if "spark" in only:
        jobs.append(("spark", root / DOC, "", ""))
    if "voice" in only:
        jobs.append(("voice", root / VOICE, "", ""))
    if "voice-es" in only:
        notes = sorted(root.glob(VOICE_ES))
        jobs += [("voice", n, "", f"{_slug(n.stem)}-") for n in notes]
        if not notes:
            skipped.append(f"voice-es: skipped, no Spanish voice note in {root / Path(VOICE_ES).parent} "
                           "(the SYNTHETIC-TTS one was withdrawn for its license, see evals/PROCEDENCIA.md)")
    if "sam-own" in only:
        photos = sorted(root.glob(OWN_PHOTOS))
        jobs += [("sam", p, c, f"{_slug(p.stem)}-{_slug(c)}-") for p in photos for c in own_concepts]
        if not photos:
            skipped.append(f"sam-own: skipped, no own photo in {root / Path(OWN_PHOTOS).parent} "
                           "(photos taken for this repo, EXIF removed)")
    return jobs, skipped


def _witness(kind: str, r: ApiResult, sha: str, concept: str = "") -> VisionWitness | DocWitness | VoiceWitness:
    m = dict(model=r.model, provider="meta-api", source_sha256=sha)
    if kind == "sam":
        if r.status == "OK" and r.raw_text is not None:
            return vision_from_sam(r.raw_text, concept=concept, **m)
        return VisionWitness(status="ERROR", detail=r.detail, concept=concept, **m)
    if kind == "spark":
        if r.status == "OK" and r.data is not None:
            return doc_from_spark(r.data, **m)
        return DocWitness(status="ERROR", detail=r.detail, **m)
    if r.status == "OK" and r.data is not None:
        return voice_from_transcript(r.data, **m)
    return VoiceWitness(status="ERROR", detail=r.detail, **m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="evals")
    ap.add_argument("--receipts-dir", default="receipts")
    ap.add_argument("--keys-dir", default="keys-lane2")
    ap.add_argument("--calls-log", default=None, help="default: <receipts-dir>/llamadas.json")
    ap.add_argument("--save-test-data", default=None, help="directory for real-<witness>-<utc>.json")
    ap.add_argument("--only", default="sam,spark,voice", help="comma list of sam, spark, voice, voice-es, sam-own")
    ap.add_argument("--concepts", default=",".join(OWN_CONCEPTS), help="comma list of SAM concepts for sam-own")
    a = ap.parse_args()

    log = Path(a.calls_log) if a.calls_log else Path(a.receipts_dir) / "llamadas.json"
    try:
        client = MetaClient(call_log=log, origin="scripts/live_eval.py")
    except NoKey:
        print(f"NO_MODEL: {KEY_ENV} is not set; live eval needs a real key. Nothing was run.", file=sys.stderr)
        return 3

    own_concepts = tuple(c.strip() for c in a.concepts.split(",") if c.strip())
    jobs, skipped = plan(Path(a.evals), set(a.only.split(",")), own_concepts=own_concepts)
    for line in skipped:
        print(line)

    sk = load_or_create_key(a.keys_dir)
    # Receipts are public: record the command relative to the repo, never the machine's interpreter or paths.
    repo = Path(__file__).resolve().parent.parent
    try:
        script = str(Path(sys.argv[0]).resolve().relative_to(repo))
    except ValueError:
        script = Path(sys.argv[0]).name
    command = " ".join(shlex.quote(x) for x in ["python", script, *sys.argv[1:]])
    n_ok = 0
    with client:
        for kind, path, concept, tag in jobs:
            sha = sha256_file(path)
            if kind == "sam":
                r = client.sam_segment(path, concept)
            elif kind == "spark":
                r = client.spark_extract_document(path)
            else:
                r = client.voice_transcribe(path)
            w = _witness(kind, r, sha, concept)
            body: dict[str, Any] = {
                "kind": "live-witness",
                "witness": kind,
                "model": r.model,
                "provider": "meta-api",
                "utc": r.utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "s": r.seconds,
                "http": r.http_status,
                "command": command,
                "input": {"path": str(path), "sha256": sha},
                "call": call_record(r),
                "result": w.to_dict(),
            }
            if concept:
                body["concept"] = concept
            stem = f"live-{kind}-{tag}" + _stamp(r.utc)
            rp = write_receipt(sign_receipt(body, sk), a.receipts_dir, stem)
            n_ok += r.status == "OK"

            if a.save_test_data and r.status == "OK":
                td = Path(a.save_test_data)
                td.mkdir(parents=True, exist_ok=True)
                rec = {
                    "provenance": f"REAL meta-api {r.utc}",
                    "witness": kind, "model": r.model, "utc": r.utc, "http": r.http_status, "s": r.seconds,
                    "input": {"name": path.name, "sha256": sha}, "concept": concept or None,
                    "request": r.request, "media": r.media,
                    "receipt": rp.name,
                    "response_sse": strip_sse(r.raw_text) if kind == "sam" and r.raw_text is not None else None,
                    "response_json": r.data,
                    "witness_at_record_time": w.to_dict(),
                }
                tname = f"real-{kind}-{tag}" + _stamp(r.utc)
                (td / f"{tname}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

            summary = (f"count={w.count}" if isinstance(w, (VisionWitness, VoiceWitness)) else
                       f"pieces={w.pieces} weight_kg={w.weight_kg} seal={w.seal}")
            print(f"{kind:5} {path.name:24} {concept or '-':14} http={r.http_status} s={r.seconds} "
                  f"{w.status} {summary} {('(' + w.detail[:160] + ')') if w.detail else ''}")
            print(f"      receipt: {rp}")
    print(f"{n_ok}/{len(jobs)} calls OK; ledger: {log}")
    return 0 if n_ok == len(jobs) else 1


if __name__ == "__main__":
    sys.exit(main())
