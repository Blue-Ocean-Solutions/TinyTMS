"""tinytms checkin: three witnesses -> judge -> signed receipt.

Exit codes: 0 ACCEPT, 2 HOLD, 3 ERROR (bad input, I/O, internal failure).
"""

from __future__ import annotations

import argparse
import os
import sys

from .pipeline import EXPLAIN_MODES, MODES, CheckinResult, InputError, run_checkin
from .verify import DEFAULT_MAX_WEIGHT_KG
from .witness import DocWitness, VoiceWitness

EXIT_ACCEPT, EXIT_HOLD, EXIT_ERROR = 0, 2, 3


def report(res: CheckinResult) -> list[str]:
    """Human summary of one check-in (also used by tests and the web page's text view)."""
    lines = [res.verdict.decision]
    lines += [f"  reason: {r}" for r in res.verdict.reasons]
    for k, w in res.witnesses.items():
        seen = (f" pieces={w.pieces} seal={w.seal}" if isinstance(w, DocWitness)
                else f" count={w.count}" + (f" seal={w.seal}" if isinstance(w, VoiceWitness) else ""))
        lines.append(f"  {k}: {w.status} model={w.model or '-'} provider={w.provider}" + (seen if w.status == "OK" else "")
                     + (f" ({w.detail})" if w.detail else ""))
    for k, f in res.fallback.items():
        lines.append(f"  fallback: {k} -> local-mlx ({f})")
    if res.explanation is not None:
        e = res.explanation
        lines.append(f"  explanation[{e['status']}]: {e.get('text') or e.get('detail')}")
    lines.append(f"  receipt: {res.path}")
    return lines


def checkin(args: argparse.Namespace) -> int:
    try:
        res = run_checkin(
            args.mode, load=args.load, doc=args.doc, voice=args.voice, fixture_dir=args.fixture_dir,
            concept=args.concept, max_weight_kg=args.max_weight_kg, receipts_dir=args.receipts_dir,
            keys_dir=args.keys_dir, calls_log=args.calls_log, explain=args.explain, lang=args.lang,
        )
    except InputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_ERROR
    print("\n".join(report(res)))
    return EXIT_ACCEPT if res.verdict.decision == "ACCEPT" else EXIT_HOLD


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tinytms")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checkin", help="three witnesses -> judge -> signed receipt")
    c.add_argument("--load", help="photo of the load")
    c.add_argument("--doc", help="photo of the BOL / Carta de Porte")
    c.add_argument("--voice", help="driver voice note (audio)")
    c.add_argument("--mode", choices=MODES, required=True,
                   help="auto: API first, a witness lost to NoKey or a network/HTTP error is redone locally")
    c.add_argument("--fixture-dir", help="directory with *.FIXTURE.json witnesses")
    c.add_argument("--max-weight-kg", type=float, default=DEFAULT_MAX_WEIGHT_KG)
    c.add_argument("--concept", default="pallet", help="SAM concept prompt")
    c.add_argument("--receipts-dir", default="receipts")
    c.add_argument("--keys-dir", default=os.environ.get("TINYTMS_KEYS_DIR", "keys"))
    c.add_argument("--calls-log", help="API call ledger (default: <receipts-dir>/llamadas.json)")
    c.add_argument("--explain", choices=EXPLAIN_MODES, default="none",
                   help="on HOLD, explain to the driver: api = Muse Spark 1.3, local = Llama 3.1 8B (never changes the verdict)")
    c.add_argument("--lang", choices=("es", "en"), default="es", help="language of the explanation")
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
