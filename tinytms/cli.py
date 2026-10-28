"""tinytms checkin: three witnesses -> judge -> signed receipt.

Exit codes: 0 ACCEPT, 2 HOLD, 3 ERROR (bad input, I/O, internal failure).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import call_record, doc_from_spark, from_fixture, vision_from_sam, voice_from_transcript
from .meta_api import ApiResult, MetaClient, NoKey
from .receipt import load_or_create_key, sign_receipt, write_receipt
from .verify import DEFAULT_MAX_WEIGHT_KG, Limits, Verdict, judge
from .witness import DocWitness, VisionWitness, VoiceWitness, Witness, sha256_file

EXIT_ACCEPT, EXIT_HOLD, EXIT_ERROR = 0, 2, 3
FIXTURE_FILES = {"vision": "vision.FIXTURE.json", "doc": "doc.FIXTURE.json", "voice": "voice.FIXTURE.json"}
FIXTURE_MEDIA = {"load": "load.FIXTURE.png", "doc": "doc.FIXTURE.png", "voice": "voice.FIXTURE.wav"}

Triple = tuple[VisionWitness, DocWitness, VoiceWitness]


def _fixture_witnesses(fdir: Path, sha: dict[str, str]) -> Triple:
    v = from_fixture(fdir / FIXTURE_FILES["vision"], sha["load"])
    d = from_fixture(fdir / FIXTURE_FILES["doc"], sha["doc"])
    a = from_fixture(fdir / FIXTURE_FILES["voice"], sha["voice"])
    assert isinstance(v, VisionWitness) and isinstance(d, DocWitness) and isinstance(a, VoiceWitness)
    return v, d, a


def _api_witnesses(
    inp: dict[str, Path], sha: dict[str, str], concept: str, calls_log: Path | None = None,
    calls: dict[str, Any] | None = None,
) -> Triple:
    """Real Meta API witnesses. `calls` (if given) is filled with a receipt record per call."""
    try:
        client = MetaClient(call_log=calls_log, origin="cli checkin")
    except NoKey as e:
        why = f"NoKey: {e}"
        return (
            VisionWitness(status="NO_MODEL", provider="meta-api", source_sha256=sha["load"], detail=why, concept=concept),
            DocWitness(status="NO_MODEL", provider="meta-api", source_sha256=sha["doc"], detail=why),
            VoiceWitness(status="NO_MODEL", provider="meta-api", source_sha256=sha["voice"], detail=why),
        )

    def err(cls: type[Witness], r: ApiResult, s: str) -> Any:
        return cls(status="ERROR", model=r.model, provider="meta-api", source_sha256=s, detail=r.detail)

    out = calls if calls is not None else {}
    with client:
        r = client.sam_segment(inp["load"], concept)
        out["vision"] = call_record(r)
        v = (vision_from_sam(r.raw_text, model=r.model, provider="meta-api", source_sha256=sha["load"], concept=concept)
             if r.status == "OK" and r.raw_text is not None else err(VisionWitness, r, sha["load"]))
        r = client.spark_extract_document(inp["doc"])
        out["doc"] = call_record(r)
        d = (doc_from_spark(r.data, model=r.model, provider="meta-api", source_sha256=sha["doc"])
             if r.status == "OK" and r.data is not None else err(DocWitness, r, sha["doc"]))
        r = client.voice_transcribe(inp["voice"])
        out["voice"] = call_record(r)
        a = (voice_from_transcript(r.data, model=r.model, provider="meta-api", source_sha256=sha["voice"])
             if r.status == "OK" and r.data is not None else err(VoiceWitness, r, sha["voice"]))
    return v, d, a


def _local_witnesses(inp: dict[str, Path], sha: dict[str, str], concept: str) -> Triple:
    from ._local import LocalModelError, ModelNotCached
    from .local_vlm import GLIMMER_MODEL_ID, LocalVLM, doc_local, vision_local

    model = f"{GLIMMER_MODEL_ID}@local"
    voice = VoiceWitness(status="NO_MODEL", provider="local-mlx", source_sha256=sha["voice"],
                         detail="no local speech-to-text model wired in this build")
    vlm = LocalVLM()
    try:
        vlm.load()
    except LocalModelError as e:
        status = "NO_MODEL" if isinstance(e, ModelNotCached) else "ERROR"
        why = f"{type(e).__name__}: {e}"
        return (
            VisionWitness(status=status, model=model, provider="local-mlx", source_sha256=sha["load"],
                          detail=why, concept=concept),
            DocWitness(status=status, model=model, provider="local-mlx", source_sha256=sha["doc"], detail=why),
            voice,
        )
    try:
        return vision_local(vlm, inp["load"], concept), doc_local(vlm, inp["doc"]), voice
    finally:
        vlm.close()


def _explain_local(verdict: Verdict, language: str) -> dict[str, Any]:
    from ._local import LocalModelError
    from .local_llm import LLAMA_MODEL_ID, LocalLLM

    out: dict[str, Any] = {"model": f"{LLAMA_MODEL_ID}@local", "provider": "local-mlx", "language": language}
    try:
        with LocalLLM() as llm:
            g = llm.explain(verdict.reasons, language)
        return {**out, "status": "OK", "text": g.text}
    except LocalModelError as e:
        return {**out, "status": "ERROR", "detail": f"{type(e).__name__}: {e}"}


def checkin(args: argparse.Namespace) -> int:
    fdir = Path(args.fixture_dir) if args.fixture_dir else None
    if args.mode == "fixture" and fdir is None:
        print("ERROR: --mode fixture needs --fixture-dir", file=sys.stderr)
        return EXIT_ERROR

    given = {"load": args.load, "doc": args.doc, "voice": args.voice}
    inp: dict[str, Path] = {}
    for k, v in given.items():
        if v is None and fdir is not None:
            v = fdir / FIXTURE_MEDIA[k]
        if v is None:
            print(f"ERROR: --{k} is required in --mode {args.mode}", file=sys.stderr)
            return EXIT_ERROR
        inp[k] = Path(v)
        if not inp[k].is_file():
            print(f"ERROR: input not found: {inp[k]}", file=sys.stderr)
            return EXIT_ERROR
    sha = {k: sha256_file(p) for k, p in inp.items()}

    if args.mode == "fixture":
        assert fdir is not None
        vision, doc, voice = _fixture_witnesses(fdir, sha)
    elif args.mode == "api":
        calls: dict[str, Any] = {}
        log = Path(args.calls_log) if args.calls_log else Path(args.receipts_dir) / "llamadas.json"
        vision, doc, voice = _api_witnesses(inp, sha, args.concept, log, calls)
    else:
        vision, doc, voice = _local_witnesses(inp, sha, args.concept)

    limits = Limits(max_weight_kg=args.max_weight_kg)
    verdict = judge(vision, doc, voice, limits)

    now = datetime.now(timezone.utc)
    witnesses = {"vision": vision, "doc": doc, "voice": voice}
    body: dict[str, Any] = {
        "kind": "checkin",
        "utc": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "mode": args.mode,
        "inputs": {k: {"name": p.name, "sha256": sha[k]} for k, p in inp.items()},
        "models": {k: {"model": w.model, "provider": w.provider} for k, w in witnesses.items()},
        "witnesses": {k: w.to_dict() for k, w in witnesses.items()},
        "limits": {"max_weight_kg": limits.max_weight_kg},
        "verdict": verdict.to_dict(),
    }
    if args.mode == "fixture":
        body["provenance"] = "FIXTURE: witnesses are hand-written fixtures, not model outputs"
    if args.mode == "api" and calls:
        body["calls"] = calls
    if verdict.decision == "HOLD" and args.explain == "local":
        body["explanation"] = _explain_local(verdict, args.language)

    sk = load_or_create_key(args.keys_dir)
    receipt = sign_receipt(body, sk)
    stem = now.strftime("%Y%m%dT%H%M%S%fZ") + f"-{verdict.decision}" + ("-FIXTURE" if args.mode == "fixture" else "")
    path = write_receipt(receipt, args.receipts_dir, stem)

    print(f"{verdict.decision}")
    for r in verdict.reasons:
        print(f"  reason: {r}")
    for k, w in witnesses.items():
        seen = (f" pieces={w.pieces} seal={w.seal}" if isinstance(w, DocWitness)
                else f" count={w.count}" + (f" seal={w.seal}" if isinstance(w, VoiceWitness) else ""))
        print(f"  {k}: {w.status} model={w.model or '-'} provider={w.provider}" + (seen if w.status == "OK" else "")
              + (f" ({w.detail})" if w.detail else ""))
    if "explanation" in body:
        e = body["explanation"]
        print(f"  explanation[{e['status']}]: {e.get('text') or e.get('detail')}")
    print(f"  receipt: {path}")
    return EXIT_ACCEPT if verdict.decision == "ACCEPT" else EXIT_HOLD


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tinytms")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checkin", help="three witnesses -> judge -> signed receipt")
    c.add_argument("--load", help="photo of the load")
    c.add_argument("--doc", help="photo of the BOL / Carta de Porte")
    c.add_argument("--voice", help="driver voice note (audio)")
    c.add_argument("--mode", choices=("fixture", "api", "local"), required=True)
    c.add_argument("--fixture-dir", help="directory with *.FIXTURE.json witnesses")
    c.add_argument("--max-weight-kg", type=float, default=DEFAULT_MAX_WEIGHT_KG)
    c.add_argument("--concept", default="pallet", help="SAM concept prompt")
    c.add_argument("--receipts-dir", default="receipts")
    c.add_argument("--keys-dir", default=os.environ.get("TINYTMS_KEYS_DIR", "keys"))
    c.add_argument("--calls-log", help="API call ledger (default: <receipts-dir>/llamadas.json)")
    c.add_argument("--explain", choices=("none", "local"), default="none",
                   help="on HOLD, explain reasons with the local LLM (never changes the verdict)")
    c.add_argument("--language", default="Spanish")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return checkin(args)
    except Exception as e:  # noqa: BLE001 - operational failure, not a verdict
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
