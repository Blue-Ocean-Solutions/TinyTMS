"""Ask Meta models, as colleagues behind a firewall, about a check-in the judge already decided; sign each answer.

    python scripts/ask_colleagues.py --receipt receipts/<file>-HOLD.json [--colleagues wonder,spark-1.1]
                                     [--photo evals/public/<photo>] [--keys-dir keys] [--firewall-file <local list>]

The source receipt must verify and its signer must be in keys/known_signers.txt; its verdict and witness fields are
rebuilt from it. Each colleague's advice goes through tinytms/colleagues.py (firewall first, nothing sent on a hit,
no call at all without the local firewall list) and is written to receipts/colleague-<name>-<utc>.json, which holds
the source receipt's name and sha256, the verdict copied unchanged, and the advice. Every call goes to
receipts/llamadas.json under its model's cap. Exit 0 when every colleague answered, 1 otherwise, 3 when the source
receipt cannot be trusted.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from receipt_show import KNOWN_SIGNERS, check  # noqa: E402
from tinytms.colleagues import COLLEAGUES, FirewallMissing, consult, load_firewall, receipt_body  # noqa: E402
from tinytms.receipt import load_or_create_key, sign_receipt, write_receipt  # noqa: E402
from tinytms.receipts_index import load_known_signers  # noqa: E402
from tinytms.verify import Verdict  # noqa: E402
from tinytms.witness import DocWitness, VisionWitness, VoiceWitness  # noqa: E402


def _witness(cls: type, d: dict) -> object:
    d = dict(d)
    if "boxes" in d:
        d["boxes"] = tuple(tuple(b) for b in d["boxes"])
    return cls(**d)


def _photo(r: dict, given: str | None) -> tuple[Path | None, str | None]:
    """The receipt's own photo: --photo, or the load input's name under evals/public or evals/own."""
    load = (r.get("inputs") or {}).get("load") or {}
    want = load.get("sha256")
    if given:
        return Path(given), want
    for sub in ("public", "own"):
        p = REPO / "evals" / sub / str(load.get("name", ""))
        if load.get("name") and p.is_file():
            return p, want
    return None, want


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--colleagues", default=",".join(COLLEAGUES), help=f"comma list of {', '.join(COLLEAGUES)}")
    ap.add_argument("--photo", help="the receipt's own photo, when it is not under evals/")
    ap.add_argument("--keys-dir", default="keys")
    ap.add_argument("--receipts-dir", default="receipts")
    ap.add_argument("--calls-log", default=None, help="default: <receipts-dir>/llamadas.json")
    ap.add_argument("--firewall-file", default=None, help="the local list (default: $TINYTMS_FIREWALL_FILE or "
                                                           "~/.config/tinytms/firewall.txt)")
    ap.add_argument("--known-signers", default=str(KNOWN_SIGNERS))
    a = ap.parse_args(argv)
    names = [n.strip() for n in a.colleagues.split(",") if n.strip()]
    unknown = [n for n in names if n not in COLLEAGUES]
    if unknown:
        ap.error(f"unknown colleague(s) {unknown}; known: {sorted(COLLEAGUES)}")

    src = Path(a.receipt)
    try:
        raw = src.read_bytes()
        known = load_known_signers(a.known_signers)
    except (OSError, ValueError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 3
    ok, line = check(raw, known)
    print(f"{src.name}: {line}")
    if not ok:
        return 3
    r = json.loads(raw)
    if r.get("kind") != "checkin":
        print("ERROR: the source is not a check-in receipt", file=sys.stderr)
        return 3
    v = r["verdict"]
    verdict = Verdict(v["decision"], tuple(v["reasons"]), dict(v.get("checks") or {}))
    ws = r["witnesses"]
    vision, doc, voice = (_witness(VisionWitness, ws["vision"]), _witness(DocWitness, ws["doc"]),
                          _witness(VoiceWitness, ws["voice"]))
    max_kg = float(r.get("limits", {}).get("max_weight_kg", 24000.0))
    try:
        words: list[str] | None = load_firewall(a.firewall_file)
    except FirewallMissing as e:
        print(f"firewall: {e}; no colleague will be called")
        words = None
    photo, photo_sha = _photo(r, a.photo)
    log = Path(a.calls_log) if a.calls_log else Path(a.receipts_dir) / "llamadas.json"
    command = " ".join(shlex.quote(x) for x in ["python", "scripts/ask_colleagues.py", *(argv or sys.argv[1:])])

    all_ok = True
    for name in names:
        c = consult(name, verdict, vision, doc, voice, firewall_words=words, photo=photo, photo_sha256=photo_sha,
                    max_weight_kg=max_kg, calls_log=log, origin="scripts/ask_colleagues.py")
        now = datetime.now(timezone.utc)
        body = receipt_body("colleague", source_name=src.name, source_raw=raw, source_key=r.get("public_key"),
                            verdict=verdict, consult=c, command=command,
                            utc=now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
        path = write_receipt(sign_receipt(body, load_or_create_key(a.keys_dir)), a.receipts_dir,
                             f"colleague-{name}-" + now.strftime("%Y%m%dT%H%M%SZ"))
        shown = c.answer if c.status == "OK" else c.detail
        extra = f" count={c.count}" if c.count is not None else ""
        print(f"{name} · {c.model} · {c.status}{extra} · {' '.join(shown.split())[:200]}")
        print(f"  receipt: {path}")
        all_ok = all_ok and c.status == "OK"
    print(f"verdict: {verdict.decision} (unchanged: colleagues advise, the judge decides)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
