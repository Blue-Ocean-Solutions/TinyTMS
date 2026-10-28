"""Explain an existing signed HOLD receipt with Muse Spark 1.3 and sign the result.

    python scripts/explain_receipt.py --receipt receipts/<file>-HOLD.json --lang es [--keys-dir keys-lane3]

The source receipt's signature is checked first; its verdict and witness fields are rebuilt from the receipt
(no image or audio is read, none is sent). Writes receipts/explain-<utc>.json holding the source receipt's
name and sha256, the explanation (text, model, usage, seconds, the exact prompt) and the unchanged verdict.
Needs MODEL_API_KEY in the environment; the call goes to receipts/llamadas.json like every other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinytms.explain import LANGUAGES, explain_hold  # noqa: E402
from tinytms.receipt import load_or_create_key, sign_receipt, verify_receipt, write_receipt  # noqa: E402
from tinytms.verify import Verdict  # noqa: E402
from tinytms.witness import DocWitness, VisionWitness, VoiceWitness  # noqa: E402


def _witness(cls: type, d: dict) -> object:
    d = dict(d)
    if "boxes" in d:
        d["boxes"] = tuple(tuple(b) for b in d["boxes"])
    return cls(**d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--lang", choices=sorted(LANGUAGES), default="es")
    ap.add_argument("--receipts-dir", default="receipts")
    ap.add_argument("--keys-dir", default="keys-lane3")
    ap.add_argument("--calls-log", default=None, help="default: <receipts-dir>/llamadas.json")
    a = ap.parse_args()

    src = Path(a.receipt)
    raw = src.read_bytes()
    if not verify_receipt(raw):
        print(f"ERROR: {src.name}: signature does not verify", file=sys.stderr)
        return 3
    r = json.loads(raw)
    v = r["verdict"]
    verdict = Verdict(v["decision"], tuple(v["reasons"]), dict(v.get("checks") or {}))
    ws = r["witnesses"]
    vision = _witness(VisionWitness, ws["vision"])
    doc = _witness(DocWitness, ws["doc"])
    voice = _witness(VoiceWitness, ws["voice"])
    max_kg = float(r.get("limits", {}).get("max_weight_kg", 24000.0))

    log = Path(a.calls_log) if a.calls_log else Path(a.receipts_dir) / "llamadas.json"
    e = explain_hold(verdict, vision, doc, voice, a.lang, max_weight_kg=max_kg, calls_log=log,
                     origin="scripts/explain_receipt.py")

    now = datetime.now(timezone.utc)
    body = {
        "kind": "explain",
        "utc": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "command": " ".join(shlex.quote(x) for x in ["python", "scripts/explain_receipt.py", *sys.argv[1:]]),
        "source_receipt": {"name": src.name, "sha256": hashlib.sha256(raw).hexdigest(),
                           "public_key": r.get("public_key")},
        "verdict": verdict.to_dict(),
        "explanation": e.to_dict(),
    }
    path = write_receipt(sign_receipt(body, load_or_create_key(a.keys_dir)), a.receipts_dir,
                         "explain-" + now.strftime("%Y%m%dT%H%M%SZ"))
    print(f"{verdict.decision} {list(verdict.reasons)}")
    print(f"explanation[{e.status}] {e.model} lang={e.language} s={e.s} usage={e.usage}")
    print(f"  {e.text or e.detail}")
    print(f"  receipt: {path}")
    return 0 if e.status == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
