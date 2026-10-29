"""One command for the dock scene: boxes on camera, a printed Carta de Porte, a voice note -> check-in -> ACTA.md.

    # CI / rehearsal: fixtures/<case>, no camera, no model, no MODEL_API_KEY
    python scripts/maqueta.py --dry-run accept
    # Saturday
    python scripts/maqueta.py --load camera:0 --doc camera:0 --voice nota.wav --mode auto --explain api --lang es

`camera:N` grabs one frame with tinytms.capture.capture_frame; files are copied. Either way the input lands in
--out (default maqueta/<utc>/) with its sha256, so the record is self-contained. The check-in is
tinytms.pipeline.run_checkin (receipt in the repo's receipts/); <out>/ACTA.md is the human-readable record next to
it. Exit code is the CLI's: 0 ACCEPT, 2 HOLD, 3 operational error (a failed capture is said in ACTA.md, and no
placeholder image is ever used).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tinytms import capture  # noqa: E402
from tinytms.cli import EXIT_ACCEPT, EXIT_ERROR, EXIT_HOLD, report  # noqa: E402
from tinytms.pipeline import EXPLAIN_MODES, FIXTURE_MEDIA, MODES, CheckinResult, InputError, run_checkin  # noqa: E402
from tinytms.receipt import REDACTED, redact_machine_paths  # noqa: E402
from tinytms.witness import sha256_bytes, sha256_file  # noqa: E402

ROLES = ("load", "doc", "voice")
CAMERA = re.compile(r"camera:(\d+)")
SCRIPT = "python scripts/maqueta.py"
SHOW = {"load": "la carga / the load", "doc": "la Carta de Porte / the paper"}
# Witness fields shown in the record; everything else in a witness is identity (model, sha) or bulk (boxes).
FIELDS = {"vision": ("concept", "count", "image_w", "image_h"),
          "doc": ("pieces", "weight_kg", "seal", "shipper", "consignee", "hazmat"),
          "voice": ("transcript", "language", "count", "seal", "audio_duration_ms")}
TEXT = {
    "es": {"title": "Acta de recepción", "inputs": "Entradas", "role": "papel", "file": "archivo",
           "source": "origen", "mode": "Modo", "witnesses": "Testigos", "verdict": "Veredicto",
           "reasons": "Razones", "explanation": "Explicación al chofer", "receipt": "Recibo firmado",
           "command": "Comando", "error": "Error operativo", "no_receipt": "Sin recibo: el check-in no corrió.",
           "boxes": "cajas detectadas", "none": "ninguna",
           "dry": "ensayo, fixtures/{}: testigos FIXTURE escritos a mano, ningún modelo"},
    "en": {"title": "Check-in record", "inputs": "Inputs", "role": "role", "file": "file", "source": "source",
           "mode": "Mode", "witnesses": "Witnesses", "verdict": "Verdict", "reasons": "Reasons",
           "explanation": "Driver explanation", "receipt": "Signed receipt", "command": "Command",
           "error": "Operational error", "no_receipt": "No receipt: the check-in did not run.",
           "boxes": "boxes detected", "none": "none",
           "dry": "dry run, fixtures/{}: hand-written FIXTURE witnesses, no model"},
}


class Refused(Exception):
    """Arguments that would break the dry run's promise (no camera, no model)."""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def shown(token: str, cwd: Path) -> str:
    """A command-line token as it may appear in the record: repo-relative, never a machine path."""
    if token.startswith("--") and "=" in token:
        k, v = token.split("=", 1)
        return f"{k}={shown(v, cwd)}"
    p = Path(token)
    full = p if p.is_absolute() else cwd / p
    if p.is_absolute() or full.exists():
        try:
            return full.resolve().relative_to(REPO).as_posix()
        except (ValueError, OSError):
            return redact_machine_paths(token) if p.is_absolute() and REDACTED not in token else token
    return token


def repo_relative(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return p.name


def command_line(argv: list[str]) -> str:
    cwd = Path.cwd()
    out = [shown(t, cwd) for t in argv]
    # an absolute path outside the repo with no known machine prefix is still this machine's
    out = [REDACTED if Path(t).is_absolute() else t for t in out]
    return " ".join([SCRIPT, *(shlex.quote(t) for t in out)])


def take(role: str, given: str, out: Path) -> dict[str, Any]:
    """Copy a file or grab a camera frame into `out`; return its record (file name, sha256, source)."""
    m = CAMERA.fullmatch(given)
    if m:
        if role == "voice":
            raise InputError("--voice must be a WAV file, not a camera")
        n = int(m.group(1))
        if sys.stdin.isatty():  # one person, one camera: boxes first, then the paper
            input(f"[{role}] {SHOW[role]} -> camera {n}, Enter: ")
        img = capture.capture_frame(n)  # looked up at call time: tests swap it
        dest = out / f"{role}-camera{n}.jpg"
        dest.write_bytes(img.data)
        sha = sha256_file(dest)
        if sha != sha256_bytes(img.data):
            raise capture.CaptureError(f"{dest.name}: written bytes differ from the captured frame")
        return {"path": dest, "name": dest.name, "sha256": sha, "source": f"camera:{n}"}
    src = Path(given)
    if not src.is_file():
        raise InputError(f"input not found: {src.name}")
    if role != "voice":
        capture.load_image(src)  # JPEG/PNG that cv2 can decode, else CaptureError
    name = src.name if src.name.startswith(role) else f"{role}-{src.name}"
    dest = out / name
    shutil.copyfile(src, dest)
    return {"path": dest, "name": name, "sha256": sha256_file(dest), "source": src.name}


def _value(v: Any) -> str:
    return "—" if v is None or v == "" else f"`{v}`" if not isinstance(v, str) else v


def acta(t: dict[str, str], *, mode: str, inputs: dict[str, dict[str, Any]], command: str,
         res: CheckinResult | None = None, error: str | None = None, dry_run: str | None = None) -> str:
    lines = [f"# {t['title']} · {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", ""]
    if res is not None:
        lines += [f"**{t['verdict']}: {res.verdict.decision}**", ""]
    if error is not None:
        lines += [f"**ERROR** (exit 3): {t['error'].lower()}", ""]
    lines += [f"- {t['mode']}: `{mode}`" + (f" ({t['dry'].format(dry_run)})" if dry_run else ""),
              "", f"## {t['inputs']}", "", f"| {t['role']} | {t['file']} | {t['source']} | sha256 |",
              "|---|---|---|---|"]
    lines += [f"| {k} | {i['name']} | {i['source']} | `{i['sha256']}` |" for k, i in inputs.items()]
    if error is not None:
        lines += ["", f"## {t['error']}", "", f"    {error}", "", t["no_receipt"]]
    if res is not None:
        lines += ["", f"## {t['witnesses']}"]
        for k, w in res.witnesses.items():
            d = w.to_dict()
            lines += ["", f"### {k}", "", f"- status: **{w.status}** · model: {_value(w.model)} · "
                      f"provider: `{w.provider}`"]
            lines += [f"- {f}: {_value(d.get(f))}" for f in FIELDS[k] if f in d]
            if d.get("boxes"):
                lines.append(f"- {t['boxes']}: {len(d['boxes'])}")
            if w.detail:
                lines.append(f"- detail: {w.detail}")
            if k in res.fallback:
                lines.append(f"- fallback → local-mlx: {res.fallback[k]}")
        lines += ["", f"## {t['verdict']}: {res.verdict.decision}", "", f"{t['reasons']}:"]
        lines += [f"- `{r}`" for r in res.verdict.reasons] or [f"- {t['none']}"]
        if res.explanation is not None:
            e = res.explanation
            lines += ["", f"## {t['explanation']}", "",
                      f"- status: **{e.get('status')}** · model: {_value(e.get('model'))} · "
                      f"provider: `{e.get('provider')}`", ""]
            lines += [f"> {x}" for x in str(e.get("text") or e.get("detail") or "").splitlines()]
        lines += ["", f"## {t['receipt']}", "", f"- `{repo_relative(res.path)}`",
                  f"- sha256: `{sha256_file(res.path)}`"]
    lines += ["", f"## {t['command']}", "", "```", command, "```", ""]
    # witness details and model words are free text: the record describes the scene, never this machine
    return redact_machine_paths("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="maqueta", description=__doc__.split("\n")[0])
    p.add_argument("--load", help="photo of the load: jpg|png|camera:N")
    p.add_argument("--doc", help="photo of the Carta de Porte / BOL: jpg|png|camera:N")
    p.add_argument("--voice", help="driver voice note (WAV)")
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--explain", choices=EXPLAIN_MODES, default="none")
    p.add_argument("--lang", choices=("es", "en"), default="es")
    p.add_argument("--out", help="record directory (default maqueta/<utc>/)")
    p.add_argument("--dry-run", nargs="?", const="accept", metavar="CASE",
                   help="fixtures/CASE (default accept): no camera, no model, no key")
    p.add_argument("--concept", default="pallet", help="SAM concept prompt")
    p.add_argument("--receipts-dir", default=str(REPO / "receipts"))
    p.add_argument("--keys-dir", default=os.environ.get("TINYTMS_KEYS_DIR", str(REPO / "keys")))
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.dry_run is None and (args.mode is None or not (args.load and args.doc and args.voice)):
        ap.error("a real run needs --mode, --load, --doc and --voice (or use --dry-run)")
    t = TEXT[args.lang]
    out = Path(args.out) if args.out else Path("maqueta") / utc_stamp()
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_ERROR
    record = out / "ACTA.md"
    if record.exists():
        print(f"ERROR: {record} already exists; a record is never overwritten", file=sys.stderr)
        return EXIT_ERROR

    mode = "fixture" if args.dry_run is not None else args.mode
    command = command_line(argv)
    inputs: dict[str, dict[str, Any]] = {}
    fdir: Path | None = None

    def fail(msg: str) -> int:
        record.write_text(acta(t, mode=mode, inputs=inputs, command=command, error=msg, dry_run=args.dry_run),
                          encoding="utf-8")
        print(f"ERROR: {redact_machine_paths(msg)}\n  record: {record}", file=sys.stderr)
        return EXIT_ERROR

    try:
        if args.dry_run is not None:
            if args.mode not in (None, "fixture") or args.explain != "none":
                raise Refused("--dry-run uses FIXTURE witnesses only: no --mode api|local|auto, no --explain")
            fdir = Path(args.dry_run) if os.sep in args.dry_run else REPO / "fixtures" / args.dry_run
            if not fdir.is_dir():
                raise Refused(f"no fixture case {args.dry_run!r} under fixtures/")
        given = {"load": args.load, "doc": args.doc, "voice": args.voice}
        for k in ROLES:
            v = given[k]
            if fdir is not None:
                if v is not None and CAMERA.fullmatch(v):
                    raise Refused(f"--dry-run never touches a camera (--{k} {v})")
                v = v or str(fdir / FIXTURE_MEDIA[k])
            inputs[k] = take(k, v, out)
    except (Refused, InputError, capture.CaptureError, OSError) as e:
        return fail(f"{type(e).__name__}: {e}")

    try:
        res = run_checkin(
            mode, load=inputs["load"]["path"], doc=inputs["doc"]["path"], voice=inputs["voice"]["path"],
            fixture_dir=fdir, concept=args.concept, receipts_dir=args.receipts_dir, keys_dir=args.keys_dir,
            explain=args.explain, lang=args.lang, origin="maqueta",
        )
    except Exception as e:  # noqa: BLE001 - operational failure, not a verdict
        return fail(f"{type(e).__name__}: {e}")

    record.write_text(acta(t, mode=mode, inputs=inputs, command=command, res=res, dry_run=args.dry_run),
                      encoding="utf-8")
    print("\n".join(report(res)))
    print(f"  record: {record}")
    return EXIT_ACCEPT if res.verdict.decision == "ACCEPT" else EXIT_HOLD


if __name__ == "__main__":
    sys.exit(main())
