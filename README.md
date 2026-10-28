# tinyTMS — verifiable freight check-in

A small carrier backs into the dock with pallets and a paper bill of lading / Carta de Porte.
tinyTMS checks the load in with **three witnesses and a judge**, and signs what it decided.

| Witness | Input | Model (planned) | What it reports |
|---|---|---|---|
| **Vision** | photo of the load | SAM 3.1 (Meta Model API), concept prompt `pallet` | count (+ boxes) |
| **Paper** | photo of the document | Muse Spark (Meta Model API, vision) | pieces, weight kg, seal, shipper, consignee, hazmat |
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
MODEL_API_KEY=... python -m tinytms.cli checkin --mode api --load load.jpg --doc bol.jpg --voice note.m4a
#    without MODEL_API_KEY: HOLD with NO_MODEL:vision / NO_MODEL:doc / NO_MODEL:voice, exit 2

# 3. local mode: Muse Glimmer 30B 4-bit on Apple silicon via mlx_vlm (no network, no download)
TINYTMS_LOCAL_PYTHON=/path/to/venv/bin/python python -m tinytms.cli checkin --mode local --load ... --doc ... --voice ...
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

## Numbers policy

No performance figure (latency, throughput, accuracy) appears here unless it links to a file in `receipts/`
produced by a command that is written down, with its date. **Today there are none**: no real model has
been run through tinyTMS yet. `tests/test_readme_numbers.py` fails CI if this README quotes a time,
frame rate, percentage or token rate on a line without a `receipts/` link.

## What is NOT here (yet)

- No real model output. The Meta API request bodies and model ids are assumptions until checked with a
  live key; `fixtures/` are hand-written and labelled FIXTURE in file name and content.
- No local speech-to-text: in `--mode local` the voice witness is `NO_MODEL`, so local check-ins HOLD.
- The house `mlx_vlm` does not load Muse Glimmer today; local mode needs an interpreter whose `mlx_vlm` does.
- No camera capture, web page, TMS records (load / stop / document) or exception tickets yet.
- Nothing about customs or government filings.

## License

Apache-2.0, see `LICENSE`.
