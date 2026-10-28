"""One check-in, shared by the CLI and the web page: inputs -> three witnesses -> judge -> (explain) -> receipt.

Modes:
  fixture  hand-written *.FIXTURE.json witnesses (receipt says FIXTURE)
  api      Meta Model API for all three witnesses
  local    Muse Glimmer (mlx_vlm worker) for paper and photo; voice is NO_MODEL (no local STT), so it HOLDs
  auto     API first; a witness whose call fails with NoKey or a network/HTTP error is redone locally.
           The receipt names provider and model per witness and why each fallback happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .adapters import call_record, doc_from_spark, from_fixture, vision_from_sam, voice_from_transcript
from .explain import Explanation, explain_hold
from .meta_api import ApiResult, MetaClient, NoKey, is_transport_failure
from .receipt import load_or_create_key, sign_receipt, write_receipt
from .verify import DEFAULT_MAX_WEIGHT_KG, Limits, Verdict, judge
from .witness import DocWitness, VisionWitness, VoiceWitness, Witness, sha256_file

MODES = ("fixture", "api", "local", "auto")
EXPLAIN_MODES = ("none", "api", "local")
FIXTURE_FILES = {"vision": "vision.FIXTURE.json", "doc": "doc.FIXTURE.json", "voice": "voice.FIXTURE.json"}
FIXTURE_MEDIA = {"load": "load.FIXTURE.png", "doc": "doc.FIXTURE.png", "voice": "voice.FIXTURE.wav"}
LOCAL_LANGUAGE = {"es": "Spanish", "en": "English"}

ClientFactory = Callable[..., MetaClient]
Witnesses = dict[str, Witness]


class InputError(ValueError):
    """Bad arguments or missing input files: an operational error, never a verdict."""


@dataclass
class CheckinResult:
    verdict: Verdict
    witnesses: Witnesses
    body: dict[str, Any]
    receipt: dict[str, Any]
    path: Path
    explanation: dict[str, Any] | None = None
    fallback: dict[str, str] = field(default_factory=dict)


def resolve_inputs(mode: str, fixture_dir: Path | None, given: dict[str, str | Path | None]) -> dict[str, Path]:
    if mode not in MODES:
        raise InputError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "fixture" and fixture_dir is None:
        raise InputError("--mode fixture needs --fixture-dir")
    inp: dict[str, Path] = {}
    for k in ("load", "doc", "voice"):
        v = given.get(k)
        if v is None and fixture_dir is not None:
            v = fixture_dir / FIXTURE_MEDIA[k]
        if v is None:
            raise InputError(f"--{k} is required in --mode {mode}")
        inp[k] = Path(v)
        if not inp[k].is_file():
            raise InputError(f"input not found: {inp[k]}")
    return inp


# --- witnesses --------------------------------------------------------------------------------


def fixture_witnesses(fdir: Path, sha: dict[str, str]) -> Witnesses:
    v = from_fixture(fdir / FIXTURE_FILES["vision"], sha["load"])
    d = from_fixture(fdir / FIXTURE_FILES["doc"], sha["doc"])
    a = from_fixture(fdir / FIXTURE_FILES["voice"], sha["voice"])
    assert isinstance(v, VisionWitness) and isinstance(d, DocWitness) and isinstance(a, VoiceWitness)
    return {"vision": v, "doc": d, "voice": a}


def local_witnesses(inp: dict[str, Path], sha: dict[str, str], concept: str,
                    which: tuple[str, ...] = ("vision", "doc", "voice")) -> Witnesses:
    """Glimmer (one worker, loaded once) for vision/doc; voice is always NO_MODEL (no local STT)."""
    from ._local import LocalModelError, ModelNotCached
    from .local_vlm import GLIMMER_MODEL_ID, LocalVLM, doc_local, vision_local

    out: Witnesses = {}
    if "voice" in which:
        out["voice"] = VoiceWitness(status="NO_MODEL", provider="local-mlx", source_sha256=sha["voice"],
                                    detail="no local speech-to-text model wired in this build")
    need = [w for w in ("vision", "doc") if w in which]
    if not need:
        return out
    model = f"{GLIMMER_MODEL_ID}@local"
    vlm = LocalVLM()
    try:
        vlm.load()
    except LocalModelError as e:
        status = "NO_MODEL" if isinstance(e, ModelNotCached) else "ERROR"
        why = f"{type(e).__name__}: {e}"
        if "vision" in need:
            out["vision"] = VisionWitness(status=status, model=model, provider="local-mlx", source_sha256=sha["load"],
                                          detail=why, concept=concept)
        if "doc" in need:
            out["doc"] = DocWitness(status=status, model=model, provider="local-mlx", source_sha256=sha["doc"],
                                    detail=why)
        return out
    try:
        if "vision" in need:
            out["vision"] = vision_local(vlm, inp["load"], concept)
        if "doc" in need:
            out["doc"] = doc_local(vlm, inp["doc"])
    finally:
        vlm.close()
    return out


def api_witnesses(
    inp: dict[str, Path], sha: dict[str, str], concept: str, calls_log: Path | None, calls: dict[str, Any],
    client_factory: ClientFactory | None = None, fallback: dict[str, str] | None = None,
    origin: str = "cli checkin",
) -> Witnesses:
    """Real Meta API witnesses; `calls` is filled with a receipt record per call.

    With `fallback` (a dict, auto mode) a witness lost to NoKey or a network/HTTP failure is not judged as
    ERROR here: its reason goes into `fallback` and the witness is left out for the caller to redo locally.
    """
    try:
        client = (client_factory or MetaClient)(call_log=calls_log, origin=origin)
    except NoKey as e:
        why = f"NoKey: {e}"
        if fallback is not None:
            fallback.update({w: why for w in ("vision", "doc", "voice")})
            return {}
        return {
            "vision": VisionWitness(status="NO_MODEL", provider="meta-api", source_sha256=sha["load"], detail=why,
                                    concept=concept),
            "doc": DocWitness(status="NO_MODEL", provider="meta-api", source_sha256=sha["doc"], detail=why),
            "voice": VoiceWitness(status="NO_MODEL", provider="meta-api", source_sha256=sha["voice"], detail=why),
        }

    def err(cls: type[Witness], r: ApiResult, s: str) -> Any:
        extra = {"concept": concept} if cls is VisionWitness else {}
        return cls(status="ERROR", model=r.model, provider="meta-api", source_sha256=s, detail=r.detail, **extra)

    out: Witnesses = {}

    def keep(name: str, r: ApiResult, w: Witness) -> None:
        calls[name] = call_record(r)
        if fallback is not None and is_transport_failure(r):
            fallback[name] = r.detail[:300] or f"HTTP {r.http_status}"
        else:
            out[name] = w

    with client:
        r = client.sam_segment(inp["load"], concept)
        keep("vision", r, vision_from_sam(r.raw_text, model=r.model, provider="meta-api", source_sha256=sha["load"],
                                          concept=concept)
             if r.status == "OK" and r.raw_text is not None else err(VisionWitness, r, sha["load"]))
        r = client.spark_extract_document(inp["doc"])
        keep("doc", r, doc_from_spark(r.data, model=r.model, provider="meta-api", source_sha256=sha["doc"])
             if r.status == "OK" and r.data is not None else err(DocWitness, r, sha["doc"]))
        r = client.voice_transcribe(inp["voice"])
        keep("voice", r, voice_from_transcript(r.data, model=r.model, provider="meta-api", source_sha256=sha["voice"])
             if r.status == "OK" and r.data is not None else err(VoiceWitness, r, sha["voice"]))
    return out


def _explain_local(verdict: Verdict, lang: str) -> dict[str, Any]:
    from ._local import LocalModelError
    from .local_llm import LLAMA_MODEL_ID, LocalLLM

    out: dict[str, Any] = {"model": f"{LLAMA_MODEL_ID}@local", "provider": "local-mlx", "language": lang}
    try:
        with LocalLLM() as llm:
            g = llm.explain(verdict.reasons, LOCAL_LANGUAGE.get(lang, lang))
        return {**out, "status": "OK", "text": g.text}
    except LocalModelError as e:
        return {**out, "status": "ERROR", "detail": f"{type(e).__name__}: {e}"}


# --- the check-in -----------------------------------------------------------------------------


def run_checkin(
    mode: str,
    *,
    load: str | Path | None = None,
    doc: str | Path | None = None,
    voice: str | Path | None = None,
    fixture_dir: str | Path | None = None,
    concept: str = "pallet",
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG,
    receipts_dir: str | Path = "receipts",
    keys_dir: str | Path = "keys",
    calls_log: str | Path | None = None,
    explain: str = "none",
    lang: str = "es",
    client_factory: ClientFactory | None = None,
    origin: str = "cli",
) -> CheckinResult:
    if explain not in EXPLAIN_MODES:
        raise InputError(f"explain must be one of {EXPLAIN_MODES}, got {explain!r}")
    if lang not in LOCAL_LANGUAGE:
        raise InputError(f"lang must be one of {sorted(LOCAL_LANGUAGE)}, got {lang!r}")
    fdir = Path(fixture_dir) if fixture_dir else None
    inp = resolve_inputs(mode, fdir, {"load": load, "doc": doc, "voice": voice})
    sha = {k: sha256_file(p) for k, p in inp.items()}
    log = Path(calls_log) if calls_log else Path(receipts_dir) / "llamadas.json"

    calls: dict[str, Any] = {}
    fallback: dict[str, str] = {}
    if mode == "fixture":
        assert fdir is not None
        ws = fixture_witnesses(fdir, sha)
    elif mode == "api":
        ws = api_witnesses(inp, sha, concept, log, calls, client_factory, origin=f"{origin} checkin")
    elif mode == "local":
        ws = local_witnesses(inp, sha, concept)
    else:
        ws = api_witnesses(inp, sha, concept, log, calls, client_factory, fallback, f"{origin} checkin")
        if fallback:
            ws.update(local_witnesses(inp, sha, concept, tuple(fallback)))
    vision, doc_w, voice_w = ws["vision"], ws["doc"], ws["voice"]
    assert isinstance(vision, VisionWitness) and isinstance(doc_w, DocWitness) and isinstance(voice_w, VoiceWitness)

    limits = Limits(max_weight_kg=max_weight_kg)
    verdict = judge(vision, doc_w, voice_w, limits)

    now = datetime.now(timezone.utc)
    witnesses: Witnesses = {"vision": vision, "doc": doc_w, "voice": voice_w}
    body: dict[str, Any] = {
        "kind": "checkin",
        "utc": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "mode": mode,
        "origin": origin,
        "inputs": {k: {"name": p.name, "sha256": sha[k]} for k, p in inp.items()},
        "models": {k: {"model": w.model, "provider": w.provider} for k, w in witnesses.items()},
        "witnesses": {k: w.to_dict() for k, w in witnesses.items()},
        "limits": {"max_weight_kg": limits.max_weight_kg},
        "verdict": verdict.to_dict(),
    }
    if mode == "fixture":
        body["provenance"] = "FIXTURE: witnesses are hand-written fixtures, not model outputs"
    if calls:
        body["calls"] = calls
    if fallback:
        body["fallback"] = {k: {"to": "local-mlx", "why": v} for k, v in fallback.items()}

    explanation: dict[str, Any] | None = None
    if verdict.decision == "HOLD" and explain == "api":
        e: Explanation = explain_hold(verdict, vision, doc_w, voice_w, lang, max_weight_kg=max_weight_kg,
                                      calls_log=log, client_factory=client_factory, origin=f"{origin} explain")
        explanation = e.to_dict()
    elif verdict.decision == "HOLD" and explain == "local":
        explanation = _explain_local(verdict, lang)
    if explanation is not None:
        body["explanation"] = explanation

    sk = load_or_create_key(keys_dir)
    receipt = sign_receipt(body, sk)
    stem = now.strftime("%Y%m%dT%H%M%S%fZ") + f"-{verdict.decision}" + ("-FIXTURE" if mode == "fixture" else "")
    path = write_receipt(receipt, receipts_dir, stem)
    return CheckinResult(verdict, witnesses, body, receipt, path, explanation, fallback)
