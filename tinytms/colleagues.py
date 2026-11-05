"""Meta models as colleagues, behind a firewall: asked about a check-in the judge already decided. They answer;
they never decide.

What goes out is built only from the decided receipt: the judge's reasons and the witnesses' structured fields (the
same whitelist the driver's explanation uses), plus, for a vision colleague, the receipt's own photo, checked by
sha256. Before any request, the outgoing text passes `firewall()`:

- every entry of a LOCAL list (`TINYTMS_FIREWALL_FILE`, default `~/.config/tinytms/firewall.txt`), which is kept out of
  the repository on purpose, so the names it guards are never published by guarding them;
- key-shaped strings;
- machine paths.

Any hit stops the call: status REFUSED, nothing is sent, and the refused prompt is kept only as its sha256. With no list
there is no call at all: a missing configuration never opens the gate.

The answer is advice. It goes into its own signed receipt beside a copy of the verdict, which it cannot change.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .adapters import output_text
from .explain import explain_facts
from .meta_api import MetaClient, NoKey, data_url, prepare_image
from .receipt import find_machine_paths
from .verify import DEFAULT_MAX_WEIGHT_KG, Verdict
from .witness import DocWitness, VisionWitness, VoiceWitness

FIREWALL_ENV = "TINYTMS_FIREWALL_FILE"
DEFAULT_FIREWALL_FILE = "~/.config/tinytms/firewall.txt"
MAX_OUTPUT_TOKENS = int(os.environ.get("TINYTMS_COLLEAGUE_MAX_OUTPUT_TOKENS", "2000"))

ConsultStatus = Literal["OK", "REFUSED", "NO_MODEL", "ERROR"]


@dataclass(frozen=True)
class Colleague:
    name: str
    model: str
    role: str
    uses_photo: bool = False


COLLEAGUES: dict[str, Colleague] = {
    "wonder": Colleague("wonder", "muse-wonder-1.0", "audit: do the judge's reasons follow from the witnesses?"),
    "spark-1.1": Colleague("spark-1.1", "muse-spark-1.1", "second opinion on the photo's count", uses_photo=True),
}

AUDIT_PROMPT = (
    "A deterministic rule engine already decided this freight check-in. The decision is final and you cannot change "
    "it. Audit it: does each of the engine's reasons follow from what the witnesses reported? Answer CONSISTENT if "
    "they all do; otherwise list each reason that does not follow and why, one per line. Use only the facts below.\n"
    "Decision: {decision}\nReasons from the rule engine:\n{reasons}\nWhat each witness reported:\n{facts}"
)
COUNT_PROMPT = "How many {concept}s are in this photo? Answer with one integer only."

_KEY_SHAPES = re.compile(
    r"sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,}|AIza[0-9A-Za-z_-]{20,}|xox[abp]-[A-Za-z0-9-]{8,}"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY|Bearer\s+[A-Za-z0-9._-]{12,}"
)
_INTEGER = re.compile(r"-?\d+")


class FirewallMissing(RuntimeError):
    """No local firewall list: no colleague may be called."""


def load_firewall(path: str | Path | None = None) -> list[str]:
    """The local list, lower case, comments and blank lines dropped. Missing or empty: FirewallMissing."""
    p = Path(os.path.expanduser(str(path or os.environ.get(FIREWALL_ENV) or DEFAULT_FIREWALL_FILE)))
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise FirewallMissing(f"no firewall list at {p.name}: {type(e).__name__}") from e
    words = [w.strip().lower() for w in lines if w.strip() and not w.strip().startswith("#")]
    if not words:
        raise FirewallMissing(f"firewall list {p.name} has no entries")
    return words


def firewall(text: str, words: list[str]) -> list[str]:
    """Why `text` may not leave, or [] when it may. A list entry is named by its position, never by itself."""
    low = text.lower()
    reasons = [f"firewall list entry {i}" for i, w in enumerate(words, 1) if w and w in low]
    if _KEY_SHAPES.search(text):
        reasons.append("key-shaped string")
    if find_machine_paths(text):
        reasons.append("machine path")
    return reasons


@dataclass(frozen=True)
class Consult:
    colleague: str
    model: str
    role: str
    status: ConsultStatus
    answer: str = ""
    count: int | None = None
    detail: str = ""
    prompt: str = ""  # only when the firewall let it out
    prompt_sha256: str = ""
    firewall: tuple[str, ...] = ()
    photo_sha256: str = ""
    usage: dict[str, Any] | None = None
    s: float | None = None
    http: int | None = None
    utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["firewall"] = list(self.firewall) or ["passed"]
        return d


def audit_prompt(verdict: Verdict, vision: VisionWitness, doc: DocWitness, voice: VoiceWitness,
                 max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG) -> str:
    return AUDIT_PROMPT.format(decision=verdict.decision,
                               reasons="\n".join(f"- {r}" for r in verdict.reasons) or "- none",
                               facts="\n".join(explain_facts(vision, doc, voice, max_weight_kg)))


def _request(model: str, prompt: str, image_url: str | None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    if image_url:
        content.append({"type": "input_image", "image_url": image_url})
    return {"model": model, "max_output_tokens": MAX_OUTPUT_TOKENS,
            "input": [{"type": "message", "role": "user", "content": content}]}


def consult(
    name: str, verdict: Verdict, vision: VisionWitness, doc: DocWitness, voice: VoiceWitness, *,
    firewall_words: list[str] | None, photo: str | Path | None = None, photo_sha256: str | None = None,
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG, calls_log: str | Path | None = None,
    client_factory: Callable[..., MetaClient] | None = None, origin: str = "colleagues",
) -> Consult:
    """One colleague's advice on a decided check-in. Never raises for a model failure; never touches the verdict."""
    if name not in COLLEAGUES:
        raise ValueError(f"unknown colleague {name!r}; known: {sorted(COLLEAGUES)}")
    col = COLLEAGUES[name]
    base = dict(colleague=col.name, model=col.model, role=col.role)
    prompt = (COUNT_PROMPT.format(concept=vision.concept or "object") if col.uses_photo
              else audit_prompt(verdict, vision, doc, voice, max_weight_kg))
    sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if firewall_words is None:
        return Consult(status="REFUSED", detail="no firewall list: no colleague is called", prompt_sha256=sha,
                       firewall=("no firewall list",), **base)
    hits = firewall(prompt, firewall_words)
    if hits:
        return Consult(status="REFUSED", detail="firewall: " + "; ".join(hits), prompt_sha256=sha,
                       firewall=tuple(hits), **base)
    image_url, photo_sha = None, ""
    if col.uses_photo:
        try:
            raw = Path(photo).read_bytes() if photo else b""
        except OSError as e:
            return Consult(status="REFUSED", detail=f"photo unreadable: {type(e).__name__}", prompt_sha256=sha,
                           firewall=("photo",), **base)
        photo_sha = hashlib.sha256(raw).hexdigest() if raw else ""
        if not raw or not photo_sha256 or photo_sha != photo_sha256.lower():
            return Consult(status="REFUSED", detail="photo is not the receipt's own input (sha256)", prompt_sha256=sha,
                           firewall=("photo",), photo_sha256=photo_sha, **base)
        img, mime, _media = prepare_image(photo)
        image_url = data_url(img, mime)
    try:
        client = (client_factory or MetaClient)(call_log=calls_log, origin=origin)
    except NoKey as e:
        return Consult(status="NO_MODEL", detail=f"NoKey: {e}", prompt=prompt, prompt_sha256=sha,
                       photo_sha256=photo_sha, **base)
    with client:
        r = client.responses(_request(col.model, prompt, image_url))
    sent = dict(prompt=prompt, prompt_sha256=sha, photo_sha256=photo_sha, s=r.seconds, http=r.http_status, utc=r.utc)
    if r.status != "OK" or r.data is None:
        return Consult(status="ERROR", detail=r.detail, **sent, **base)
    usage = r.data.get("usage")
    if r.data.get("status") == "incomplete":
        why = (r.data.get("incomplete_details") or {}).get("reason", "?")
        return Consult(status="ERROR", detail=f"response incomplete ({why})", usage=usage, **sent, **base)
    text = (output_text(r.data) or "").strip()
    if not text:
        return Consult(status="ERROR", detail="no output text", usage=usage, **sent, **base)
    count, detail = None, ""
    if col.uses_photo:
        m = _INTEGER.search(text)
        count = int(m.group(0)) if m else None
        detail = "" if m else "no integer in the answer"
    return Consult(status="OK", answer=text, count=count, detail=detail, usage=usage, **sent, **base)


def receipt_body(kind: str, *, source_name: str, source_raw: bytes, source_key: str | None, verdict: Verdict,
                 consult: Consult, command: str, utc: str = "") -> dict[str, Any]:
    """The receipt's body: the source receipt by name and sha256, the verdict copied as it was, and the advice."""
    return {
        "kind": kind,
        "utc": utc,
        "command": command,
        "source_receipt": {"name": source_name, "sha256": hashlib.sha256(source_raw).hexdigest(),
                           "public_key": source_key},
        "verdict": verdict.to_dict(),
        "colleague": consult.to_dict(),
    }
