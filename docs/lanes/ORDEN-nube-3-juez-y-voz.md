# ORDEN nube-3 · Harden the judge and the Spanish voice parser

**Lane name:** `Nube-3` · **Model:** Opus 5.5 · **Cap:** 80 turns or $25 · **Branch:** `nube/3-juez-y-voz`

## Goal
The judge (`tinytms/verify.py`) decides; the voice parser (`tinytms/adapters.py`, `voice_from_transcript`) feeds it
from real transcripts that vary ("doce payets", "pajets", "MX MX guión"). Make both provably robust without
changing their contracts.

## Deliverables
1. **Judge property tests** (`tests/test_verify_properties.py`, standard library `random` with a fixed seed, no new
   deps): for thousands of generated witness triples, assert the invariants:
   - any witness not `OK` → HOLD, and the only reasons are `NO_MODEL:*` / `ERROR:*`;
   - a HOLD lists every failing check (never only the first); ACCEPT has zero reasons;
   - seal comparison is invariant to case, spaces and dashes;
   - `hazmat is True` always holds; `None` never does;
   - the decision is a pure function (same inputs → same `Verdict`, including `checks`).
   If a property fails, fix the judge with a focused test and say so in the PR.
2. **Voice parser corpus:** `tests/data/voice-es-variantes.json`, labelled `"provenance": "SYNTHETIC hand-written"`:
   30+ Spanish and English phrasings of counts (números en palabras, "docena", "quince cajas", "cero") and seals
   (letras deletreadas, "guión", "raya", "MX cuarenta y ocho mil doscientos trece"). Tests that the parser returns the
   right count/seal or `None`, never a wrong number. Keep the three REAL transcripts in `tests/data/real-voice-*`
   passing exactly as recorded.
3. **Fail-closed on ambiguity:** when two different counts appear in one transcript ("doce… no, trece"), the parser
   returns `count=None` and a `detail` that says why. Test it.

## Do not
- Do not change the rule order or the reason labels of the judge.
- Do not touch `tinytms/meta_api.py`, `tinytms/web.py` or anything under `receipts/`.
