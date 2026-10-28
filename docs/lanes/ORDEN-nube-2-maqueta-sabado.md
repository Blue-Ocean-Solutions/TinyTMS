# ORDEN nube-2 · One command for the dock scene (Saturday's live run)

**Lane name:** `Nube-2` · **Model:** Opus 5.5 · **Cap:** 80 turns or $25 · **Branch:** `nube/2-maqueta`

## Goal
On Saturday a person will put real boxes in front of the Mac camera, hold up a printed Carta de Porte and record a
voice note. One command must take those three inputs, run the check-in, and leave a human-readable record next to
the signed receipt. No new model code: reuse `tinytms/pipeline.py`, `tinytms/capture.py` and the CLI exit codes.

## Deliverables
1. `scripts/maqueta.py`:
   - inputs: `--load <jpg|png|camera:N>`, `--doc <jpg|png|camera:N>`, `--voice <wav>`, `--mode fixture|api|local|auto`,
     `--explain none|api|local`, `--lang es|en`, `--out <dir>` (default `maqueta/<utc>/`).
   - `camera:N` grabs one frame with `tinytms.capture.capture_frame` and saves it under `--out` with its sha256;
     files are copied there too, so the record is self-contained.
   - runs `run_checkin` with `receipts_dir=receipts` (the repo's) and writes `<out>/ACTA.md`: inputs (name + sha256),
     mode, three witnesses (status, fields, detail), verdict and reasons, explanation if any, the receipt file name and
     its sha256, and the exact command line (repo-relative, no machine paths).
   - exit code = the pipeline's (0 ACCEPT, 2 HOLD, 3 operational error).
2. A `--dry-run` that uses `fixtures/<case>` and never touches a camera or a key (for CI).
3. README: a "Saturday" section with the two commands (dry run; real run with `--mode auto --explain api --lang es`).

## Tests (first)
- Dry run on `fixtures/accept` → exit 0, ACTA.md exists, lists the three sha256, names the receipt, has no machine
  paths (`tinytms.receipt.find_machine_paths` on the text).
- Dry run on `fixtures/count_mismatch` → exit 2 and the reasons appear in ACTA.md.
- `camera:0` with a fake `capture_frame` (monkeypatch) → the frame is saved and hashed; a `CaptureError` → exit 3
  and ACTA.md says so (never a placeholder image).
- Suite stays green; no sockets, no network in tests (see the autouse fixtures in `tests/test_web.py`).

## Do not
- Do not call any model in tests. Do not add dependencies. Do not write under `evals/`.
