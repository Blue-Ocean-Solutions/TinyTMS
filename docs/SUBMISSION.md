# tinyTMS.IO · submission package

The fields a Cerebral Valley / Devpost-style form asks for, filled only from what this repository shows.
Anything the repository cannot prove is a `[bracketed placeholder]` for a human to fill or delete.
Event: Meta Global AI Developer Hackathon 2026, Indie track. Kickoff Monday 2026-10-26; building period 2026-10-27 to
2026-11-04; submissions due Wednesday 2026-11-04, 9:00 PM ET (official participant guide, not stored in this repo).
Any performance figure below cites a signed file in `receipts/` on the same line; `scripts/check_docs_numbers.py`
enforces that in CI for `README.md` and every `docs/*.md`.

## Project name

tinyTMS.IO

## One-line pitch

Three Meta models witness a freight check-in (the load photo, the paper, the driver's voice), a plain deterministic
judge says ACCEPT or HOLD, and every decision is a signed receipt anyone can verify.

*ES:* Tres modelos de Meta atestiguan el check-in de una carga (la foto, el papel y la voz del chofer), un juez
determinista dice ACCEPT o HOLD, y cada decisión queda en un recibo firmado que cualquiera puede verificar.

## The problem

A small carrier backs into the dock with pallets and a paper bill of lading / Carta de Porte. Three things describe
the same load and they do not always agree: the paper (pieces, weight, seal), what is physically on the dock (the
photo), and what the driver says (the voice note). When they disagree, someone has to notice at the dock, and
afterwards nobody can show who said what. tinyTMS turns each of the three into a witness, lets a fixed set of rules
decide, explains a HOLD to the driver in their language, and signs the whole thing.

[A human: add the carrier's own story here if there is one; the repository does not hold any customer anecdote.]

## What it does

- **Three witnesses.** Vision: SAM 3.1 counts pallets in the load photo. Paper: Muse Spark 1.3 reads the document
  into pieces, weight, seal, shipper, consignee, hazmat. Voice: Muse Voice Transcribe turns the driver's note into a
  transcript; count and seal are parsed from it deterministically (`tinytms/adapters.py`).
- **One judge, no model.** `tinytms/verify.py` is plain Python: every witness OK, counts equal, seals equal, weight
  under the limit, hazmat sent to review; otherwise HOLD with its reasons. Fail-closed: a missing key, a dead
  model or an unreadable answer is a HOLD, never an ACCEPT by default.
- **Signed receipt.** Every check-in writes canonical JSON to `receipts/`, signed with Ed25519: input sha256s, model
  and provider per witness, verdict, reasons, public key. Receipts describe the run, never the machine.
- **Explanation for the driver.** On a HOLD, Muse Spark 1.3 writes 2–3 sentences in Spanish or English from the
  judge's reasons and the witnesses' fields only (no image, no audio). It cannot change the verdict.
- **Local mode.** Paper and photo run on Muse Glimmer 30B (open weights, 4-bit) on a Mac through `mlx_vlm`, no
  network. There is no local speech-to-text yet, so a local check-in always HOLDs on `NO_MODEL:voice`.
- **One web page** (`python -m tinytms.web`, standard library only, binds 127.0.0.1) running the same pipeline.

## What was built during the event

Every commit on `main` up to `7166006` is dated 2026-10-28 and those of #7 are dated 2026-10-29
(`git log --format=%ad main`), inside the building period.
By pull request, from `git log --merges`:

| PR | What it added |
|---|---|
| (first commit `9eb8e52`, before any PR) | Day 1: three witnesses, deterministic fail-closed judge, Ed25519 receipts. |
| [#1](https://github.com/Blue-Ocean-Solutions/TinyTMS/pull/1) | Receipts describe the run, never the machine; first on-metal Glimmer receipt (`receipts/local-vlm-20261028T152737Z.json`). |
| [#2](https://github.com/Blue-Ocean-Solutions/TinyTMS/pull/2) | Machine paths are redacted instead of refusing the receipt; one shared redactor. |
| [#3](https://github.com/Blue-Ocean-Solutions/TinyTMS/pull/3) | Day 2: the three witnesses against the real Meta Model API, with receipts (`receipts/live-*.json`, `receipts/llamadas.json`). |
| [#4](https://github.com/Blue-Ocean-Solutions/TinyTMS/pull/4) | Day 3: explanation for the driver, the one-page web, local and auto modes, the real Spanish voice eval. |
| [#5](https://github.com/Blue-Ocean-Solutions/TinyTMS/pull/5) | Web fences: Host allowlist, same-origin POST, `--allow-api` gate, negative Content-Length refused. |
| [#6](https://github.com/Blue-Ocean-Solutions/TinyTMS/pull/6) | Local mode: a Glimmer generation cut in its analysis channel is an ERROR, never a parsed answer; real local check-in receipt (`receipts/20261028T165434299357Z-HOLD.json`); `Sec-Fetch-Site` on POST. |
| [#7](https://github.com/Blue-Ocean-Solutions/TinyTMS/pull/7) | Saturday's dock scene in one command: `scripts/maqueta.py` (camera or file inputs, the same check-in, a human-readable `ACTA.md` next to the signed receipt; `--dry-run` on the fixtures for CI and rehearsal). |
| this package | `docs/SUBMISSION.md`, `docs/VIDEO-OUTLINE.md`, `scripts/check_docs_numbers.py` (numbers guard for docs, in CI). |
| [later PRs] | [A human: add every PR merged after this package, with its number.] |

## Meta tools used

| Tool | How tinyTMS uses it | Code | Evidence |
|---|---|---|---|
| **SAM 3.1** (Meta Model API) | `/v1/responses` with SSE streaming, concept prompt `pallet`; the vision witness's count (+ boxes) | `tinytms/meta_api.py` | `receipts/live-sam-*.json` |
| **Muse Spark 1.3** (Meta Model API) | `/v1/responses` with the document image: the paper witness. Text only: the driver's explanation on a HOLD | `tinytms/meta_api.py`, `tinytms/explain.py` | `receipts/live-spark-20261028T153211Z.json`, `receipts/explain-*.json` |
| **Muse Voice Transcribe** (Meta Model API) | `/v1/asr/transcribe` (multipart, `muse-voice-transcribe-1.0`); the voice witness's transcript | `tinytms/meta_api.py` | `receipts/live-voice-*.json` |
| **Muse Glimmer 30B** (open weights, 4-bit) | `mlx_vlm` 0.7.10 on Apple silicon, in a separate worker process, no network, no download: paper and photo in `--mode local` and as the fallback of `--mode auto` | `tinytms/local_vlm.py`, `tinytms/_local.py`, `tinytms/_mlx_worker.py` | `receipts/local-vlm-20261028T152737Z.json`, `receipts/20261028T165434299357Z-HOLD.json` |
| Llama 3.1 8B Instruct (4-bit, `mlx_lm`) | `--explain local`: the driver's explanation without network | `tinytms/local_llm.py` | none yet: wired; tests cover its prompt and its error paths, no real run |

Every Meta Model API call made so far (model, UTC, HTTP status, seconds) is in `receipts/llamadas.json`, with per-model caps.

## How to run it

From a clone of <https://github.com/Blue-Ocean-Solutions/TinyTMS> with Python 3.12 or newer:

```bash
pip install -e ".[dev]"
pytest -q
python -m tinytms.cli checkin --mode fixture --fixture-dir fixtures/count_mismatch
```

The last one prints HOLD with `COUNT_MISMATCH`, exits 2 and writes a signed receipt to `receipts/` (the inputs are
hand-written fixtures, labelled FIXTURE in the receipt; no model, no key). With a Meta Model API key in
`MODEL_API_KEY`, `--mode api --load <photo> --doc <document photo> --voice <wav>` runs the real witnesses; the README
has the local, auto, explain and web commands.

## Evidence for reviewers

- `receipts/`: every signed run; each one passes `tinytms.receipt.verify_receipt` (snippet in README). Three signer
  keys, by `public_key` prefix: `50be0ced` (the measuring seat: the two local Glimmer receipts), `a100b16d` (the
  day-2 API lane) and `37f9717d` (the day-3 API lane: the Spanish voice, explanation and explained check-in receipts).
  The full hex is in each receipt's `public_key`.
- `tests/data/real-*.json`: the raw Meta API answers (masks removed, no key), and `tests/test_real_data.py` runs the
  parsers on them offline.
- `.github/workflows/ci.yml`: the whole suite with no network and no key, plus the numbers guards.
- `evals/PROCEDENCIA.md`, `evals/public/PROCEDENCIA.md`: where every test input comes from.

## License

Apache-2.0 (`LICENSE`). Test photos from Wikimedia Commons: `boxes-pallet.jpg` public domain, `boxes-moving.jpg` by
brownpau, CC BY 2.0 (`evals/public/PROCEDENCIA.md`). The Carta de Porte and both voice notes are synthetic.

## Team

- [Team name]
- [Members, roles, country, contact]
- What the repository shows: GitHub organization `Blue-Ocean-Solutions`; every pull request merged by `cefuentes82`.
  Code was written with Claude Code sessions ("lanes": L1, F1, Nube-1 to Nube-4), each credited in the
  `Co-Authored-By` trailers of its commits.

## Links

- Repository: <https://github.com/Blue-Ocean-Solutions/TinyTMS>
- Demo video: [link; outline in `docs/VIDEO-OUTLINE.md`]
- Live demo: [none; the web page is a one-machine dock demo and binds 127.0.0.1 by design]

## What is not here yet

Copied from the README section "What is NOT here (yet)". The local-mode bullet was corrected in the same pull
request that added this file: `--mode local` has a real run, the local fallback of `--mode auto` does not.

- `fixtures/` are hand-written and labelled FIXTURE in file name and content. They are not model outputs.
- The `evals/` media are a handful of test inputs: two public photos, one synthetic Carta de Porte, one second
  of silence and one SYNTHETIC-TTS Spanish voice note (macOS voice, not a person). They are not an accuracy
  benchmark. There is no real driver voice note yet, and the voice parser has only seen three transcripts of one note.
- No local speech-to-text: in `--mode local` the voice witness is `NO_MODEL`, so local check-ins HOLD.
- `--mode local` has one real Glimmer run (`receipts/20261028T165434299357Z-HOLD.json`); the local fallback of
  `--mode auto` has only been run against a fake worker in tests, no real run backs it yet.
- Muse Glimmer needs `mlx_vlm` >= 0.7.10 (model type `muse_glimmer`); point `TINYTMS_LOCAL_PYTHON` at an interpreter that has it.
- The voice seal parser collapses a letter group repeated back to back ("MX MX") as an ASR stutter. A driver who
  really says a doubled prefix against a paper with a single one would match; accepted and written down here.
- `tinytms/capture.py` can grab a camera frame (`cv2.VideoCapture`), but the CLI does not use it yet. The web page
  is local and single-user. There are no TMS records (load / stop / document) and no exception tickets yet.
- Nothing about customs or government filings.
