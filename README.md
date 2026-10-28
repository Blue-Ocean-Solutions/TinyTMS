# tinyTMS — verifiable freight check-in

A small carrier backs into the dock with pallets and a paper bill of lading / Carta de Porte.
tinyTMS checks the load in with **three witnesses and a judge**, and signs what it decided.

| Witness | Input | Model | What it reports |
|---|---|---|---|
| **Vision** | photo of the load | SAM 3.1 (Meta Model API), concept prompt `pallet` | count (+ boxes) |
| **Paper** | photo of the document | Muse Spark 1.3 (Meta Model API, vision) | pieces, weight kg, seal, shipper, consignee, hazmat |
| **Voice** | driver's voice note (es/en) | Muse Voice Transcribe (Meta Model API) | transcript → count and seal (parsed deterministically) |

**The judge** (`tinytms/verify.py`) is plain, deterministic Python. Models report; the judge decides.
Rules, in order:

1. every witness `OK`, otherwise **HOLD** with `NO_MODEL:<witness>` / `ERROR:<witness>`;
2. `doc.pieces == vision.count == voice.count`, otherwise `COUNT_MISMATCH` (a missing count gives `VOICE_NO_COUNT`, `VISION_NO_COUNT`, `DOC_NO_PIECES`);
3. `doc.seal` and `voice.seal` present and equal (alphanumerics, case-insensitive), otherwise `SEAL_MISSING` / `SEAL_MISMATCH`;
4. `doc.weight_kg <= --max-weight-kg`, otherwise `OVERWEIGHT` (missing weight: `WEIGHT_MISSING`);
5. `doc.hazmat is True` → `HAZMAT_REVIEW`;
6. no reasons → **ACCEPT**.

It is **fail-closed**: a missing key, a dead model or an unreadable answer is a HOLD, never an ACCEPT by default.

Every check-in writes a **signed receipt** to `receipts/`: canonical JSON (sorted keys, no spaces) signed
with Ed25519 (PyNaCl). It holds the sha256 of each input, model + provider per witness
(`meta-api` | `local-mlx` | `FIXTURE`), the verdict, the reasons, the public key and the signature.
The private key lives in `keys/` (git-ignored, mode 0600) and is never printed.

## Run

```bash
pip install -e ".[dev]"
pytest -q

# 1. fixture mode: hand-written witnesses, no model involved (receipt says FIXTURE)
python -m tinytms.cli checkin --mode fixture --fixture-dir fixtures/accept          # ACCEPT, exit 0
python -m tinytms.cli checkin --mode fixture --fixture-dir fixtures/count_mismatch  # HOLD, exit 2

# 2. api mode: Meta Model API, key from the environment only
MODEL_API_KEY=... python -m tinytms.cli checkin --mode api --load load.jpg --doc bol.jpg --voice note.wav
#    voice must be a RIFF WAV (converted to mono 16 kHz PCM16 if needed); every call is logged to receipts/llamadas.json
#    without MODEL_API_KEY: HOLD with NO_MODEL:vision / NO_MODEL:doc / NO_MODEL:voice, exit 2

# 3. local mode: Muse Glimmer 30B 4-bit on Apple silicon via mlx_vlm (no network, no download); always HOLDs today
TINYTMS_LOCAL_PYTHON=/path/to/venv/bin/python python -m tinytms.cli checkin --mode local --load ... --doc ... --voice ...

# 4. auto mode: API first, a witness lost to a missing key or a network/HTTP error is redone locally
python -m tinytms.cli checkin --mode auto --load ... --doc ... --voice ...

# on HOLD, explain it to the driver (Muse Spark 1.3, text only; the verdict does not change)
python -m tinytms.cli checkin --mode api --explain api --lang es --load ... --doc ... --voice ...

# the same pipeline behind one web page (stdlib wsgiref, binds 127.0.0.1)
python -m tinytms.web --port 8765 --receipts-dir receipts
```

Exit codes: `0` ACCEPT, `2` HOLD, `3` operational error (missing file, bad arguments).

Verify a receipt:

```python
from tinytms.receipt import verify_receipt
verify_receipt(open("receipts/<file>.json", "rb").read(), trusted_public_key=open("keys/ed25519.pub").read().strip())
```

Local models run in a separate worker process (`tinytms/_mlx_worker.py`) so an MLX import or load failure
is reported as an explicit error instead of taking the CLI down. `TINYTMS_LOCAL_PYTHON` selects the
interpreter; `TINYTMS_GLIMMER_PATH` / `TINYTMS_LLAMA_PATH` point at a local snapshot directory.
`--explain local` asks Llama 3.1 8B Instruct (4-bit, `mlx_lm`) to explain a HOLD from the judge's reasons;
the verdict is already fixed and does not change.

## Explicación al chofer (driver explanation)

`tinytms/explain.py`, `--explain api|none|local` (default `none`), `--lang es|en` (default `es`).

- **What it does.** On a HOLD, Muse Spark 1.3 writes 2–3 plain sentences for the driver in their language: what does
  not match and what to do (recount the load, get the Carta de Porte corrected). The prompt is built **only** from the
  judge's reasons and the witnesses' structured fields: paper pieces / seal / weight, the weight limit, photo count,
  voice count / seal. No image, no audio, no transcript, no hash is sent; the exact prompt is stored in the receipt.
  Request: `/v1/responses`, text only, `max_output_tokens` 1200, `reasoning.effort` `low`.
- **What it does NOT do.** It never decides. The verdict is computed before the call and is not touched by it; a
  model that says "ACCEPT" changes nothing. ACCEPT is never sent to the model. If Spark fails (no key, HTTP error,
  `incomplete`), the HOLD stands and the receipt says `explanation.status = NO_MODEL | ERROR`.
- The explanation (text, model, provider, usage, seconds, language, prompt, the API's echo of what it ran) lives
  inside the signed receipt. Re-explain an existing receipt: `python scripts/explain_receipt.py --receipt receipts/<file>-HOLD.json --lang en`.
- Why 1200 and not 600: with 600, Spark spent nearly all of it reasoning and came back `incomplete`, at its default
  effort and at `low` alike: `receipts/explain-20261028T160346Z.json`, `receipts/explain-20261028T160427Z.json`.
  The two completed runs (es, en) are `receipts/explain-20261028T160450Z.json` and `receipts/explain-20261028T160459Z.json`.

## Web

`python -m tinytms.web --port 8765 --receipts-dir receipts [--keys-dir keys] [--host 127.0.0.1]`

- **What it does.** `GET /` is one HTML page (inline CSS, no JavaScript) with the three files (load photo, document
  photo, WAV voice note), the mode (`fixture|api|local|auto`), the FIXTURE case, explain and language.
  `POST /checkin` runs the same pipeline as the CLI (`tinytms/pipeline.py`) and shows the three witnesses, the
  verdict, the reasons, the explanation and a link to the signed receipt. `GET /receipts/<name>.json` serves a
  receipt; only names matching `[A-Za-z0-9._-]+` ending in `.json`, never `..`, never outside `--receipts-dir`.
- **What it does NOT do.** No login, no TLS, no rate limit, no background jobs: it is a dock-side demo for one
  machine and binds `127.0.0.1` by default. Uploads larger than 40 MiB are refused. Uploads live in a temporary
  directory for the request only; the receipt keeps each file's sanitised name and sha256. Standard library only.
- **Fences.** Requests must name this machine (`Host` must be `127.0.0.1`, `localhost`, `::1` or the `--host` given;
  anything else is 403, which stops DNS rebinding). A browser `POST /checkin` must carry an `Origin` or `Referer` from
  this server (a cross-site form gets 403). The page cannot spend Meta API credits unless the server was started with
  `--allow-api`: without it, modes `api`/`auto` and `explain=api` answer 403. A negative `Content-Length` is 400.

## Modo local y auto (local and auto modes)

- **`--mode local`.** Paper and photo go to Muse Glimmer 30B 4-bit through the existing worker
  (`tinytms/_local.py` + `tinytms/_mlx_worker.py`, one process, loaded once). Paper: the same prompt and JSON contract
  as Spark (`doc_from_fields`). Photo: "answer with a single integer"; the answer counts only if it **is** a single
  integer, prose is `count = None` (no guessing). Voice: `NO_MODEL`, there is no local speech-to-text, so
  **local mode always HOLDs today** (`NO_MODEL:voice`).
- **`--mode auto`.** API first. A witness whose call fails with `NoKey` or a network/HTTP error (timeout, transport
  error, non-200) is redone locally; a 200 with an unreadable body, a media error or a budget refusal is an answer,
  not an outage, and stays an API `ERROR`. The receipt names provider and model per witness under `models`, keeps the
  failed API call under `calls`, and says why each witness fell back under `fallback`. A voice witness that falls
  back is `NO_MODEL`, so it HOLDs.
- The pipeline is tested with a FAKE worker that speaks the JSON-lines protocol (`tests/test_local_modes.py`); the real
  model was run once on metal, see «What is measured». Glimmer reasons in an analysis channel before answering, so the
  worker reports `truncated` when the budget runs out and the witness is an ERROR, never a parsed guess
  (`tests/test_local_glimmer_truncation.py`, real outputs as data). The command for a machine that has Glimmer cached
  and an `mlx_vlm` that loads `muse_glimmer`:

```bash
TINYTMS_LOCAL_PYTHON=/path/to/mlx-venv/bin/python python -m tinytms.cli checkin --mode local \
  --load evals/public/boxes-pallet.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/es-doce-pallets-16k.wav
```

## What is measured

The Meta Model API client (`tinytms/meta_api.py`) is built to request/response shapes measured with a live key
on 2026-10-28: SAM 3.1 over `/v1/responses` with SSE streaming, Muse Spark 1.3 over `/v1/responses`, Muse Voice
Transcribe over `/v1/asr/transcribe` (multipart). Real runs, each with a signed receipt holding model, UTC time,
wall-clock seconds, input sha256, the request without base64 and the raw answer (truncated, SAM masks removed):

- `python scripts/live_eval.py --save-test-data tests/data` over `evals/` (sources in `evals/PROCEDENCIA.md`):
  one receipt per call, `receipts/live-sam-*.json`, `receipts/live-spark-*.json`, `receipts/live-voice-*.json`.
- `python -m tinytms.cli checkin --mode api --load evals/public/boxes-pallet.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/silencio_16k.wav`:
  `receipts/20261028T153041947742Z-HOLD.json` (HOLD: `VOICE_NO_COUNT`, `COUNT_MISMATCH doc=12 vision=1`, `SEAL_MISSING voice`;
  the per-call records sit under `calls`).
- Every API call made (model, UTC, HTTP status, seconds): `receipts/llamadas.json`, with per-model caps.
- **Local mode, real** (2026-10-28, Apple M5 Max, Muse Glimmer 30B 4-bit through `mlx_vlm` 0.7.10, no network):
  `python -m tinytms.cli checkin --mode local --load evals/public/boxes-pallet.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/es-doce-pallets-16k.wav`
  → `receipts/20261028T165434299357Z-HOLD.json`: photo count 1 (the boxes sit on one pallet), paper pieces 12 / 4800 kg / seal MX-48213 / hazmat false,
  voice `NO_MODEL` (no local speech-to-text) → HOLD. Wall-clock per witness is in the receipt. The first attempts with small
  token budgets returned Glimmer's analysis channel instead of an answer; those raw outputs are kept as test data in
  `tests/data/real-local-glimmer-truncated-20261028.json` and a truncated generation is now an explicit ERROR. With
  `TINYTMS_LANE` / `TINYTMS_LANE_CAPS` (e.g. `sam-3.1=20,muse-spark-1.3=25`) each record carries the lane and the
  client refuses a call once that lane's own records reach its cap (`lanes` in the ledger).
- 2026-10-28, lane 3: `python scripts/live_eval.py --only voice-es --keys-dir keys-lane3 --save-test-data tests/data`
  over the SYNTHETIC-TTS note `evals/voice/es-doce-pallets-16k.wav`: `receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json`.
  The same audio was transcribed three times that day (that run plus the two check-ins below) and came back three
  ways ("doce payets" / "pajets" / "payettes"; seal "MX guión…", "MX MX guión…", "SMX SMX guión…"). The ASR is not
  deterministic. All three real transcripts are in `tests/data/real-voice-es-*.json` and `tests/test_real_data.py`
  pins the parse of each: count 12 every time, seal `MX-48213` for the first two and `SMX-48213` for the third.
  "SMX" is not guessed back to "MX", so that run HOLDs on `SEAL_MISMATCH`.
- `python -m tinytms.cli checkin --mode api --explain api --lang es --load evals/public/boxes-pallet.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/es-doce-pallets-16k.wav --keys-dir keys-lane3`,
  run twice: `receipts/20261028T160514217793Z-HOLD.json`, `receipts/20261028T160626939074Z-HOLD.json`. Both HOLD
  with a Spanish explanation inside. The witness values in those receipts come from the parser as it stood at the
  time (before "pajets"/"payettes" and the stutter rule), so they show `VOICE_NO_COUNT` and a `SEAL_MISMATCH`.

Any count, timing or token figure lives in those files, not in this README. The real answers (masks removed,
no key) are also in `tests/data/real-*.json`, labelled `"provenance": "REAL meta-api <utc>"`, and
`tests/test_real_data.py` runs the parsers against them offline. Those API-lane receipts are signed by the lane's
key, public half `a100b16dfd814a49` (full hex in each receipt's `public_key`).

## Numbers policy

No performance figure (latency, throughput, accuracy) appears here unless it links to a file in `receipts/`
produced by a command that is written down, with its date. `tests/test_readme_numbers.py` fails CI if this
README quotes a time, frame rate, percentage or token rate on a line without a `receipts/` link.

First receipt (2026-10-28, Apple M5 Max, 128 GB unified memory, `mlx_vlm` 0.7.10), `receipts/local-vlm-20261028T152737Z.json`: Muse Glimmer 30B 4-bit loads in 1.2 s from a warm cache and generates at 30 tok/s on `evals/public/boxes-pallet.jpg` (1,242 prompt tokens, 128 generated, peak 20.7 GB), command `python scripts/measure_local_vlm.py --image evals/public/boxes-pallet.jpg`.
The model's sentence is in the receipt; it describes the door and the boxes and invents nothing.
One run on one image is a measurement, not a benchmark.

Receipts in this repo are signed by the measuring seat's key, public half
`50be0cedb34ab0b93023708d5059408480839afad77d8ef7889ef3b5f53fbb9c`; pass it as `trusted_public_key` to pin origin.
A receipt describes the run, never the machine: `sign_receipt` refuses any body carrying a user, temp or volume path,
and `tests/test_no_absolute_paths.py` fails CI if one slips into `receipts/` or `tests/data/`.

## What is NOT here (yet)

- `fixtures/` are hand-written and labelled FIXTURE in file name and content. They are not model outputs.
- The `evals/` media are a handful of test inputs: two public photos, one synthetic Carta de Porte, one second
  of silence and one SYNTHETIC-TTS Spanish voice note (macOS voice, not a person). They are not an accuracy
  benchmark. There is no real driver voice note yet, and the voice parser has only seen three transcripts of one note.
- No local speech-to-text: in `--mode local` the voice witness is `NO_MODEL`, so local check-ins HOLD.
- `--mode local` and the local fallback of `--mode auto` have only been run against a fake worker in tests; no real
  Glimmer run backs them yet.
- Muse Glimmer needs `mlx_vlm` >= 0.7.10 (model type `muse_glimmer`); point `TINYTMS_LOCAL_PYTHON` at an interpreter that has it.
- The voice seal parser collapses a letter group repeated back to back ("MX MX") as an ASR stutter. A driver who
  really says a doubled prefix against a paper with a single one would match; accepted and written down here.
- `tinytms/capture.py` can grab a camera frame (`cv2.VideoCapture`), but the CLI does not use it yet. The web page
  is local and single-user. There are no TMS records (load / stop / document) and no exception tickets yet.
- Nothing about customs or government filings.

## License

Apache-2.0, see `LICENSE`.
