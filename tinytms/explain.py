"""Explain a HOLD to the driver with Muse Spark 1.3 (text only). The model explains; it never decides.

The prompt is built ONLY from the judge's reasons and the witnesses' structured fields (pieces, count,
seal from paper and voice, weight, weight limit). No image, no audio, no transcript, no hash goes in.
The verdict is fixed before this runs and is never touched: a dead model gives status ERROR / NO_MODEL
and the HOLD stands. ACCEPT is never sent to the model.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .adapters import output_text
from .meta_api import SPARK_MODEL, MetaClient, NoKey
from .verify import DEFAULT_MAX_WEIGHT_KG, Verdict
from .witness import DocWitness, VisionWitness, VoiceWitness

ExplainStatus = Literal["OK", "NO_MODEL", "ERROR"]

# Spark reasons first: this budget covers reasoning + the sentences. 600 was measured too short on 2026-10-28:
# at the default effort "high" 598 tokens went to reasoning (receipts/explain-20261028T160346Z.json), and at
# effort "low" (accepted, echoed back) still 558 (receipts/explain-20261028T160427Z.json); both incomplete.
EXPLAIN_MAX_OUTPUT_TOKENS = int(os.environ.get("TINYTMS_EXPLAIN_MAX_OUTPUT_TOKENS", "1200"))
EXPLAIN_REASONING_EFFORT = "low"
LANGUAGES = {"es": "Spanish (Mexico)", "en": "English"}

EXPLAIN_PROMPT = (
    "A freight check-in at a loading dock was put ON HOLD by a deterministic rule engine. The decision is "
    "final and is not yours to change. Write 2 or 3 short sentences for the truck driver, in {language}, "
    "in plain words: no rule codes (never write things like COUNT_MISMATCH), no jargon. Say what does not "
    "match and what the driver should do next (for example: recount or check the load, or get the bill of "
    "lading / Carta de Porte corrected). Use only the facts below; do not add numbers, causes or facts that "
    "are not listed. Return only the sentences.\n"
    "Reasons from the rule engine:\n{reasons}\n"
    "What each witness reported:\n{facts}"
)


@dataclass(frozen=True)
class Explanation:
    status: ExplainStatus
    text: str = ""
    model: str = ""
    provider: str = "none"
    usage: dict[str, Any] | None = None
    s: float | None = None
    language: str = "es"
    detail: str = ""
    utc: str = ""
    http: int | None = None
    prompt: str = ""
    echo: dict[str, Any] | None = None  # what the API says it ran: status, reasoning, max_output_tokens, id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _v(x: object) -> str:
    if x is None:
        return "not reported"
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def explain_facts(
    vision: VisionWitness, doc: DocWitness, voice: VoiceWitness, max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG
) -> list[str]:
    """One line per witness, from structured fields only (whitelist: nothing else can leak in)."""

    def line(label: str, status: str, fields: str) -> str:
        return f"- {label}: " + (fields if status == "OK" else f"no answer ({status})")

    return [
        line("paper (bill of lading / Carta de Porte)", doc.status,
             f"pieces={_v(doc.pieces)}, seal={_v(doc.seal)}, weight_kg={_v(doc.weight_kg)} "
             f"(limit {_v(float(max_weight_kg))} kg)"),
        line("photo of the load", vision.status, f"{vision.concept} count={_v(vision.count)}"),
        line("driver's voice note", voice.status, f"count={_v(voice.count)}, seal={_v(voice.seal)}"),
    ]


def explain_prompt(
    verdict: Verdict, vision: VisionWitness, doc: DocWitness, voice: VoiceWitness, language: str = "es",
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG,
) -> str:
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {sorted(LANGUAGES)}, got {language!r}")
    return EXPLAIN_PROMPT.format(
        language=LANGUAGES[language],
        reasons="\n".join(f"- {r}" for r in verdict.reasons),
        facts="\n".join(explain_facts(vision, doc, voice, max_weight_kg)),
    )


def build_explain_request(prompt: str, model: str = SPARK_MODEL) -> dict[str, Any]:
    return {"model": model, "max_output_tokens": EXPLAIN_MAX_OUTPUT_TOKENS,
            "reasoning": {"effort": EXPLAIN_REASONING_EFFORT},
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}]}


def explain_hold(
    verdict: Verdict, vision: VisionWitness, doc: DocWitness, voice: VoiceWitness, language: str = "es", *,
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG, calls_log: str | Path | None = None,
    client_factory: Callable[..., MetaClient] | None = None, origin: str = "explain",
) -> Explanation:
    """Spark's 2-3 sentences for the driver about a HOLD. Never raises for a model failure."""
    base = dict(model=SPARK_MODEL, provider="meta-api", language=language)
    if verdict.decision != "HOLD":
        return Explanation(status="NO_MODEL", detail="verdict is not HOLD: nothing to explain, model not called",
                           **{**base, "model": "", "provider": "none"})
    prompt = explain_prompt(verdict, vision, doc, voice, language, max_weight_kg)
    try:
        client = (client_factory or MetaClient)(call_log=calls_log, origin=origin)
    except NoKey as e:
        return Explanation(status="NO_MODEL", detail=f"NoKey: {e}", prompt=prompt, **base)
    with client:
        r = client.responses(build_explain_request(prompt))
    timing = dict(s=r.seconds, utc=r.utc, http=r.http_status, prompt=prompt)
    if r.status != "OK" or r.data is None:
        return Explanation(status="ERROR", detail=r.detail, **timing, **base)
    usage = r.data.get("usage")
    timing["echo"] = {k: r.data.get(k) for k in ("id", "status", "model", "reasoning", "max_output_tokens")
                      if k in r.data}
    if r.data.get("status") == "incomplete":
        why = (r.data.get("incomplete_details") or {}).get("reason", "?")
        return Explanation(status="ERROR", detail=f"response incomplete ({why})", usage=usage, **timing, **base)
    text = (output_text(r.data) or "").strip()
    if not text:
        return Explanation(status="ERROR", detail="no output text", usage=usage, **timing, **base)
    return Explanation(status="OK", text=text, usage=usage, **timing, **base)
