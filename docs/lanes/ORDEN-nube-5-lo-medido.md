# ORDEN nube-5 · "What is measured" generated from receipts

**Lane name:** `F1-nube` · **Model:** Opus 5.5 · **Cap:** 50 turns or $15 · **Branch:** `nube/5-medido`
Read first: `docs/lanes/LEEME.md`, `docs/lanes/CARTA-F1-NUBE-2026-10-29.md`, `README.md`, `tinytms/receipt.py`, `tests/test_readme_numbers.py`.

## Goal
Make the README's numbers impossible to drift: a script builds the "What is measured" table from the receipts
themselves, CI fails when the committed table is stale.

## Deliverables
1. `scripts/receipts_table.py`: reads every `receipts/*.json`, verifies its signature with the public key embedded in
   the receipt, and writes `docs/MEASURED.md`: one row per receipt — utc, witness or kind, model and provider,
   HTTP status, seconds, verdict or count/fields summary, and the receipt file as a link. Unverifiable receipts are
   listed in a separate "not verified" section, never dropped silently. Deterministic output (sorted by utc).
2. `docs/MEASURED.md` committed, and README's "What is measured" section reduced to a short paragraph that links to it.
3. CI: a step that regenerates the table into a temp file and fails if it differs from the committed one.

## Tests (first)
- `tests/test_receipts_table.py`: a tampered receipt (one byte changed in the signed body) lands in "not verified";
  the table is byte-identical on two runs; a receipt with a machine path is impossible (sign rejects it) — assert the
  existing guard still holds.
- `tests/test_real_data.py`: the real Spark document answers in `tests/data/` still parse to `hazmat=False` (the
  synthetic document says "MATERIAL PELIGROSO: NO").

## Do not
- Do not change the judge. Carlos's word (2026-10-29 15:2x CST, wizard): an unknown hazmat field (`None`) keeps
  ACCEPTing; `HAZMAT_REVIEW` only on `True`. The hazmat change proposed in the first draft of this order is withdrawn.
- Do not edit, re-sign or delete any receipt. Do not add numbers to Markdown by hand.
- Do not touch `tinytms/web.py` or `tinytms/local_vlm.py` (another lane owns them this week).
- Nothing about customs authorities or government filings.
