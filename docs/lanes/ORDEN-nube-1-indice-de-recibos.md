# ORDEN nube-1 · Receipts index and viewer

**Lane name:** `Nube-1` · **Model:** Opus 5.5 · **Cap:** 60 turns or $20 · **Branch:** `nube/1-indice-recibos`

## Goal
Make the evidence legible. Every claim in this repo points at a signed receipt in `receipts/`; today a reader
has to open JSON files by hand. Build one index and one viewer.

## Deliverables
1. `scripts/receipts_index.py`: reads every `receipts/*.json` (skips `llamadas.json` and `.gitkeep`), verifies each
   signature (`tinytms.receipt.verify_receipt`), and writes `receipts/INDEX.md`: one table, one row per receipt
   (file, kind, witness or mode, model/provider, UTC, result in one short phrase, `verify` ✓/✗, signer key prefix).
   Idempotent; deterministic order (by file name). `python scripts/receipts_index.py --check` exits 1 when
   `INDEX.md` is stale or any receipt fails to verify.
2. CI: add a step to `.github/workflows/ci.yml` that runs `python scripts/receipts_index.py --check`.
3. Web: `GET /receipts/` (trailing slash) renders the same index as HTML (no JavaScript, escaped text), with links
   to the existing `GET /receipts/<name>.json`. Same Host/Origin fences as the rest of `tinytms/web.py`
   (read them before touching the app).
4. README: one line under "What is measured" pointing at `receipts/INDEX.md`.

## Tests (write them first)
- Index over a temp dir with two good receipts and one tampered → the tampered row shows ✗ and `--check` exits 1.
- `--check` on a stale INDEX.md exits 1; on a fresh one exits 0.
- Web `GET /receipts/` → 200, contains every receipt name, no `<script`, escaped; foreign `Host` → 403.
- The real tree: `--check` passes on the repository as committed (fix the index, never the receipts).

## Do not
- Do not modify, rename or re-sign any file under `receipts/`.
- Do not add dependencies.
