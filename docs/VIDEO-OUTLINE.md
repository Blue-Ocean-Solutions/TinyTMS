# tinyTMS.IO · 3-minute video outline

Five beats. Rules for whoever records it:

- **No receipt, no number.** Every figure in a row links to the signed receipt it comes from, in that same row. Where
  nothing has been measured yet the row says `[recibo nuevo]`; replace it only with a figure that has its own receipt.
- Timestamps are the script's pacing, not measurements.
- **Length.** The guide's limit is 3:00. The voice-over below is **378 spoken words**, counted as they are said
  (numbers and acronyms as pronounced: "SAM three point one", "A P I", "M X four eight two one three"); the count is in
  each row and per beat. Target: 420 or fewer. Whoever narrates does one **timed read-through** of the final script
  before recording and cuts again if it runs over 3:00: `say -r 150` does not read at 150 words a minute, so it is
  not a timing.
- Say what each input is: `evals/public/boxes-moving.jpg` is a public photo (CC BY 2.0, `evals/public/PROCEDENCIA.md`),
  `evals/doc/carta_sintetica.png` is a **synthetic** Carta de Porte, `evals/voice/silencio_16k.wav` is one second of
  silence. The voice transcript at 0:56 is shown from its signed receipt, as text; no audio plays.
- The photo is a placeholder: on Monday Carlos can replace it with his own photos of boxes, each with a new receipt.
- Voice-over in English, Spanish on screen where the driver reads it. [A human decides the final language mix.]
- No accuracy claims, no benchmark, no "real driver". Nothing about customs or government filings.

## Beat 1 · The dock problem (0:00–0:30) · 40 words

| Time | On screen | Voice-over | Words | Receipt |
|---|---|---|---|---|
| 0:00 | A load at the dock door, a paper Carta de Porte handed over. [Placeholder: the public photo `evals/public/boxes-moving.jpg`; Carlos's own photos replace it if he takes them.] | "A small carrier backs into the dock with a load and a paper Carta de Porte." | 16 | — |
| 0:10 | Three cards side by side: PAPER · PHOTO · VOICE. | "Paper, load and driver can disagree, and afterwards nobody can show who said what." | 14 | — |
| 0:22 | Title card: **tinyTMS.IO** · three witnesses, one judge, one signed receipt. | "tinyTMS makes each of them a witness." | 10 | — |

## Beat 2 · Three witnesses (0:30–1:10) · 102 words

| Time | On screen | Voice-over | Words | Receipt |
|---|---|---|---|---|
| 0:30 | `evals/public/boxes-moving.jpg`, concept `pallet`: no box drawn, count 0. | "Vision: SAM 3.1, through the Meta Model API, counts pallets. On this public photo of moving boxes: none." | 22 | [`receipts/live-sam-boxes-moving-pallet-20261028T153209Z.json`](../receipts/live-sam-boxes-moving-pallet-20261028T153209Z.json) |
| 0:38 | Same photo, concept `cardboard box`: 4 boxes outlined. | "Ask it for cardboard boxes and it finds 4. The vision witness asks for pallets." | 15 | [`receipts/live-sam-boxes-moving-cardboard-box-20261028T153208Z.json`](../receipts/live-sam-boxes-moving-cardboard-box-20261028T153208Z.json) |
| 0:46 | The synthetic Carta de Porte, then the JSON Spark returned. Label: SYNTHETIC. | "Paper: Muse Spark 1.3 reads the Carta de Porte into fields: 12 pieces, 4800 kg, seal MX-48213, no hazmat." | 30 | [`receipts/live-spark-20261028T153211Z.json`](../receipts/live-spark-20261028T153211Z.json) |
| 0:56 | No waveform, no audio. An excerpt of the transcript in the signed receipt, as text: «…Traigo doce payets de madera. El sello es MX guión cuatro ocho dos uno tres.…» Label: SYNTHETIC-TTS, not a driver. | "Voice: Muse Voice Transcribe turns the note into words." | 9 | [`receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json`](../receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json) |
| 1:04 | The parsed result: count 12, seal MX-48213. Caption: "parser pinned offline in `tests/test_real_data.py`; this receipt predates the fix and says `count: null`". | "Plain code reads the count from the words: «doce payets» is 12." | 12 | [`receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json`](../receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json) |
| 1:08 | Cut on 2026-11-04: there is no receipt yet of real boxes or of a driver's own voice. | — | 0 | — |

## Beat 3 · The judge says HOLD and the driver hears why (1:10–1:55) · 100 words

| Time | On screen | Voice-over | Words | Receipt |
|---|---|---|---|---|
| 1:10 | `tinytms/verify.py`, the six rules highlighted one by one. | "Models report; the judge decides. Plain Python, no model inside, and fail-closed: a missing answer is a HOLD, never an ACCEPT." | 22 | — |
| 1:20 | The check-in on the API, shown with `python scripts/receipt_show.py receipts/<file>.json`: verdict **HOLD** and every reason, run on `boxes-moving.jpg`, the synthetic Carta de Porte and `silencio_16k.wav` (recorded on screen for the video on 2026-11-04 CST; the day's two earlier takes are [`20261104T192306594488Z-HOLD.json`](../receipts/20261104T192306594488Z-HOLD.json) and [`20261104T192407727942Z-HOLD.json`](../receipts/20261104T192407727942Z-HOLD.json)). | "Here the voice note was one second of silence. Paper, photo and voice disagree: HOLD, with every reason listed." | 19 | [`receipts/20261104T235651889142Z-HOLD.json`](../receipts/20261104T235651889142Z-HOLD.json) |
| 1:32 | The driver's explanation of that check-in, first sentence, large: «Su carta de porte dice 12 piezas, pero en la foto se ven 0 tarimas.» (the sentence follows the receipt), subtitled «Your Carta de Porte says 12 pieces, but the photo shows 0 pallets.» | [Read aloud in Spanish: «Su carta de porte dice doce piezas, pero en la foto se ven cero tarimas.»] Then: "Muse Spark 1.3 wrote that from the judge's reasons and the witnesses' fields: no image, no audio." | 14 + 19 | [`receipts/20261104T235651889142Z-HOLD.json`](../receipts/20261104T235651889142Z-HOLD.json) |
| 1:42 | The same explanation in English, small. | "The same in English, for the dispatcher." | 7 | [`receipts/explain-20261105T000046Z.json`](../receipts/explain-20261105T000046Z.json) |
| 1:46 | Card: the explanation's ERROR beside the verdict, HOLD, unchanged (what those receipts record; no model said ACCEPT). | "It cannot change the verdict: when Spark ran out of budget, the HOLD stood and the receipt says ERROR." | 19 | [`receipts/explain-20261028T160346Z.json`](../receipts/explain-20261028T160346Z.json), [`receipts/explain-20261028T160427Z.json`](../receipts/explain-20261028T160427Z.json) |

## Beat 4 · Pull the plug (1:55–2:35) · variant A 72 words, variant B 65 words

Two variants for Monday's session on the Mac; record **A**, and **B** only if A fails. Every `[recibo nuevo]` is
filled from the receipts that run writes, each with its link in the same row.

**Variant A · `scripts/pull_the_plug.py`: the Meta API, then the API unreachable**

| Time | On screen | Voice-over | Words | Receipt |
|---|---|---|---|---|
| 1:55 | Terminal: `python scripts/pull_the_plug.py --load evals/public/boxes-moving.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/silencio_16k.wav`. | "Now pull the plug. Same three inputs, two runs: first through the Meta API, then with the API unreachable." | 23 | — |
| 2:02 | Side by side. Left, run 1 `--mode api`: each witness `meta-api` and its model. Right, run 2 `--mode auto`, API on a closed local port: vision and doc on `local-mlx`, voice `NO_MODEL`. | "Paper and photo fall back to Muse Glimmer 30B, Meta's open weights, on the Mac, offline." | 17 | [`receipts/20261105T000458332283Z-HOLD.json`](../receipts/20261105T000458332283Z-HOLD.json) (run 1), [`receipts/20261105T000538292980Z-HOLD.json`](../receipts/20261105T000538292980Z-HOLD.json) (run 2) |
| 2:10 | Run 2's paper fields and photo count, provider `local-mlx`. | "Glimmer reads the same paper and photo; the receipt names each model." | 12 | [`receipts/20261105T000538292980Z-HOLD.json`](../receipts/20261105T000538292980Z-HOLD.json) |
| 2:18 | `python scripts/receipt_show.py receipts/20261105T000538292980Z-HOLD.json`: «signature OK», provider and model per witness, verdict **HOLD** `NO_MODEL:voice`. | "And it holds: no local speech-to-text yet, so the voice witness is NO_MODEL. Fail-closed, even offline." | 20 | [`receipts/20261105T000538292980Z-HOLD.json`](../receipts/20261105T000538292980Z-HOLD.json) |

**Variant B · if A fails: `--mode local` with no key, one new local receipt** (not recorded: variant A worked)

| Time | On screen | Voice-over | Words | Receipt |
|---|---|---|---|---|
| 1:55 | Terminal, `MODEL_API_KEY` unset: `python -m tinytms.cli checkin --mode local --load evals/public/boxes-moving.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/silencio_16k.wav`. | "Now with no API key at all: the same three inputs, on the Mac alone." | 17 | — |
| 2:02 | The run's witnesses: vision and doc `local-mlx`, voice `NO_MODEL`. `[recibo nuevo]` | "Paper and photo go to Muse Glimmer 30B, Meta's open weights, on the Mac, offline." | 16 | `[recibo nuevo]` |
| 2:10 | Paper fields and photo count, provider `local-mlx`. `[recibo nuevo]` | "Glimmer reads the same paper and photo; the receipt names each model." | 12 | `[recibo nuevo]` |
| 2:18 | `python scripts/receipt_show.py receipts/<new local>.json`: «signature OK», verdict **HOLD** `NO_MODEL:voice`. `[recibo nuevo]` | "And it holds: no local speech-to-text yet, so the voice witness is NO_MODEL. Fail-closed, even offline." | 20 | `[recibo nuevo]` |

**Both variants**

| Time | On screen | Voice-over | Words | Receipt |
|---|---|---|---|---|
| 2:27 | Text card, no voice: Muse Glimmer 30B 4-bit on the Mac · loads in 1.2 s · 30 tok/s · peak 20.7 GB · one run on one image. | — (on-screen text only) | 0 | [`receipts/local-vlm-20261028T152737Z.json`](../receipts/local-vlm-20261028T152737Z.json) |

## Beat 5 · What is real and what is next (2:35–3:00) · 64 words

| Time | On screen | Voice-over | Words | Receipt |
|---|---|---|---|---|
| 2:35 | `python scripts/receipt_show.py receipts/live-sam-boxes-moving-cardboard-box-20261028T153208Z.json` → «signature OK · known signer»; the same file with one byte changed → «signature FAILED», exit 1; the same body re-signed by a key not in `keys/known_signers.txt` → «✗ unknown signer», exit 1 (`tests/test_receipt_show.py`, `tests/test_receipt.py`). | "Every run you saw is a signed receipt from a known seat. Change one byte, or re-sign it with another key, and it no longer passes." | 27 | [`receipts/live-sam-boxes-moving-cardboard-box-20261028T153208Z.json`](../receipts/live-sam-boxes-moving-cardboard-box-20261028T153208Z.json) |
| 2:42 | Text card, no voice: every Meta API call is in the ledger · 71 calls (2026-10-28 to 2026-11-04 CST) · 65 answered HTTP 200 · 6 refused when the plug was pulled (two runs) · each model under its cap. | — (on-screen text only) | 0 | [`receipts/llamadas.json`](../receipts/llamadas.json) |
| 2:48 | Two columns. REAL: the Meta API answers, the local Glimmer runs. SYNTHETIC: the Carta de Porte, the voice note's TTS, the FIXTURE cases. | "What is real: the model answers and the local run. What is synthetic is labelled synthetic." | 16 | [`receipts/live-spark-20261028T153211Z.json`](../receipts/live-spark-20261028T153211Z.json), [`receipts/local-vlm-20261028T152737Z.json`](../receipts/local-vlm-20261028T152737Z.json) |
| 2:53 | The author's own photo (`evals/own/devices.jpg`, EXIF removed), one frame per SAM 3.1 answer drawn by `scripts/receipt_boxes.py` from its receipt: "VR headset", "stylus" and "smartphone" box the right device; "projector" boxed the headset. | "Next, off the screen: SAM 3.1 finds the headset, the wand and the phone. Asked for the projector, it picked the headset. Every answer keeps its receipt." | 29 | [`vr-headset`](../receipts/live-sam-devices-vr-headset-20261104T200907Z.json), [`stylus`](../receipts/live-sam-devices-stylus-20261104T200908Z.json), [`smartphone`](../receipts/live-sam-devices-smartphone-20261104T200909Z.json), [`projector`](../receipts/live-sam-devices-projector-20261104T200906Z.json) |
| 2:58 | End card: **tinyTMS.IO** · Apache-2.0 · github.com/Blue-Ocean-Solutions/TinyTMS · Built during the hackathon (repository created 2026-10-28) · Photo: brownpau, CC BY 2.0 · Built with Claude Code sessions (seat names in the commits) | — | 0 | — |

**Total spoken, as recorded on 2026-11-04: 372 words** (Beat 1: 40 · Beat 2: 88, with 1:08 cut · Beat 3: 100 · Beat 4,
variant A: 72 · Beat 5: 72, with the 2:53 bridge; 2:27 and 2:42 are on-screen text). The narration is a generated voice
(Kokoro-82M, Apache-2.0 weights, run on the Mac), not a person.

## Shot list: commands behind the screens

All of these exist today; the ones marked with a key need `MODEL_API_KEY` in the environment and spend credits.

```bash
python -m tinytms.cli checkin --mode fixture --fixture-dir fixtures/count_mismatch   # no key: a FIXTURE HOLD, for rehearsal
python -m tinytms.cli checkin --mode api --explain api --lang es --load evals/public/boxes-moving.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/silencio_16k.wav   # key: beat 3, the 1:20 receipt
python scripts/pull_the_plug.py --load evals/public/boxes-moving.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/silencio_16k.wav   # key + Mac with Glimmer: beat 4 variant A, two receipts
python -m tinytms.cli checkin --mode local --load evals/public/boxes-moving.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/silencio_16k.wav   # no key, Mac with Glimmer: beat 4 variant B
python scripts/receipt_show.py receipts/<file>.json                                  # no key: any receipt on one screen, signature first
python scripts/receipt_boxes.py receipts/<live-sam-file>.json --out <frame>.png       # no key: 0:30 and 0:38, a SAM receipt's boxes on its own photo
python -m tinytms.web --port 8765 --receipts-dir receipts                              # the one-page web, if a beat uses it
python scripts/maqueta.py --dry-run count_mismatch                                     # no key: rehearse the dock-scene command
```

Re-running a command writes a **new** receipt; quote the new file, never edit an old one.

## Open for a human

- [Who narrates, in which language, and who reads the Spanish sentence aloud in beat 3; then the timed read-through.]
- [Monday, on the Mac: beat 3's 1:20 check-in and beat 4 (variant A, or B if A fails) on `boxes-moving.jpg`,
  the synthetic Carta de Porte and `silencio_16k.wav`; fill every `[recibo nuevo]` from the receipts they write.]
- [Monday: Carlos may replace `boxes-moving.jpg` with his own photos of boxes; 0:30 and 0:38 then need new SAM receipts.]
- [Tuesday: if 1:08 and 2:53 still have no receipt behind them, cut both rows.]
- [Any receipt put on screen with `receipt_show` prints its input names: show only receipts whose inputs are in the
  video (2:35 uses the `boxes-moving.jpg` SAM receipt for that reason).]
- [The explanation receipts at 1:32–1:46 explain a check-in made on 2026-10-28 with the photo that left the video;
  only their text is shown. A fresh `--explain api` run on Monday's 1:20 receipt can replace them.]
