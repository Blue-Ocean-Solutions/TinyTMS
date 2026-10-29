# ORDEN nube-6 · The pull-the-plug script and the calls view

**Lane name:** `F1-nube` · **Model:** Opus 5.5 · **Cap:** 50 turns or $15 · **Branch:** `nube/6-pull-the-plug`
Read first: `docs/lanes/LEEME.md`, `docs/lanes/CARTA-F1-NUBE-2026-10-29.md`, `README.md`, `tinytms/pipeline.py`,
`tinytms/cli.py`, `docs/VIDEO-OUTLINE.md`.

## Goal
Make the video's fourth beat ("pull the plug") a single reproducible command, and make every receipt readable by a
human in one screen: which model answered each witness, how long it took, and whether the API or the Mac answered.

## Deliverables
1. `scripts/pull_the_plug.py`: runs the same check-in twice on the same three inputs — first `--mode api`, then
   `--mode auto` with the API made unreachable **inside the process** (point `TINYTMS_META_BASE_URL` at a closed local
   port; do not touch the network stack) — and prints a side-by-side summary: per witness, provider, model, seconds,
   status; verdict and reasons for both runs; the two receipt paths. The script never fakes a result: if the local
   worker is absent (as in the cloud), the second run shows the local witnesses as ERROR/NO_MODEL and says so.
   You cannot run the local half here; the Mac will. Test the script's control flow with the FIXTURE mode and a fake
   worker, as `tests/test_local_modes.py` already does.
2. `scripts/receipt_show.py <receipt.json>`: verifies the signature and prints the one-screen view (inputs by sha,
   witnesses, calls, verdict, reasons, explanation if present, `redacted_machine_paths`). Exit 1 if the signature
   fails. Use it in `docs/VIDEO-OUTLINE.md` as the way the camera shows a receipt.
3. `docs/VIDEO-OUTLINE.md`: beat 4 rewritten around these two commands, with `[to be measured on Saturday]` where the
   Mac's numbers go. Keep every existing number linked to its receipt.

## Tests (first)
- `tests/test_pull_the_plug.py`: with fixture inputs and the fake worker, the summary has two runs, the second shows
  `fallback` for vision and doc and `NO_MODEL` for voice; with no fake worker it shows ERROR, never a number.
- `tests/test_receipt_show.py`: a real receipt from `receipts/` prints and exits 0; a tampered copy exits 1.

## Do not
- Do not touch `tinytms/web.py` or `tinytms/local_vlm.py`. Do not change the judge.
- No `time.sleep` to imitate latency; no invented seconds.
- Nothing about customs authorities or government filings.
