# tinyTMS.IO · 3-minute video outline

Five beats. Rules for whoever records it:

- **No receipt, no number.** Every figure in a row links to the signed receipt it comes from, in that same row. Where
  nothing has been measured yet (Saturday's real boxes, the driver's own voice) the row says
  `[to be measured on Saturday]`; replace it only with a figure that has its own receipt.
- Timestamps are the script's pacing, not measurements.
- Say what each input is: `evals/public/boxes-pallet.jpg` is a public photo, `evals/doc/carta_sintetica.png` is a
  **synthetic** Carta de Porte, `evals/voice/es-doce-pallets-16k.wav` is **SYNTHETIC-TTS** (a macOS voice, not a driver),
  `evals/voice/silencio_16k.wav` is one second of silence (`evals/PROCEDENCIA.md`).
- Voice-over in English, Spanish on screen where the driver reads it. [A human decides the final language mix.]
- No accuracy claims, no benchmark, no "real driver" until Saturday. Nothing about customs or government filings.

## Beat 1 · The dock problem (0:00–0:30)

| Time | On screen | Voice-over | Receipt |
|---|---|---|---|
| 0:00 | A trailer at the dock door, pallets, the driver hands over a paper Carta de Porte. [Saturday footage; until then the public photo `evals/public/boxes-pallet.jpg`.] | "A small carrier backs into the dock with pallets and a paper Carta de Porte." | — |
| 0:10 | Three cards side by side: PAPER · PHOTO · VOICE. | "The paper says one thing, the load says another, the driver says a third. When they disagree, someone has to catch it at the dock, and afterwards nobody can show who said what." | — |
| 0:22 | Title card: **tinyTMS.IO** · three witnesses, one judge, one signed receipt. | "tinyTMS makes each of them a witness." | — |

## Beat 2 · Three witnesses (0:30–1:10)

| Time | On screen | Voice-over | Receipt |
|---|---|---|---|
| 0:30 | The load photo with SAM 3.1's box drawn on it, concept `pallet`. | "Vision: SAM 3.1, through the Meta Model API, counts pallets. On this public photo: 1 pallet, in 0.47 s." | [`receipts/live-sam-boxes-pallet-pallet-20261028T153211Z.json`](../receipts/live-sam-boxes-pallet-pallet-20261028T153211Z.json) |
| 0:38 | Same photo, concept `cardboard box`: 22 boxes outlined. | "Ask it for cardboard boxes and it finds 22. The vision witness asks for pallets." | [`receipts/live-sam-boxes-pallet-cardboard-box-20261028T153210Z.json`](../receipts/live-sam-boxes-pallet-cardboard-box-20261028T153210Z.json) |
| 0:46 | The synthetic Carta de Porte, then the JSON Spark returned. Label: SYNTHETIC. | "Paper: Muse Spark 1.3 reads the Carta de Porte into fields: 12 pieces, 4800 kg, seal MX-48213, no hazmat, in 2.234 s." | [`receipts/live-spark-20261028T153211Z.json`](../receipts/live-spark-20261028T153211Z.json) |
| 0:56 | Waveform, then the transcript. Label: SYNTHETIC-TTS, not a driver. | "Voice: Muse Voice Transcribe hears the note: «Traigo doce payets de madera. El sello es MX guión cuatro ocho dos uno tres.» 11440 ms of audio, transcribed in 2.98 s." | [`receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json`](../receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json) |
| 1:04 | The parsed result: count 12, seal MX-48213. Small caption: "parser pinned offline in `tests/test_real_data.py`". | "Count and seal are parsed from the words by plain code: «doce payets» is 12. That receipt predates the parser fix and still says `count: null`; we show it as it was signed." | [`receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json`](../receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json) |
| 1:08 | Saturday: the same three witnesses on real boxes and a real driver's note. | "On Saturday: real boxes `[to be measured on Saturday]` and the driver's own voice `[to be measured on Saturday]`." | — |

## Beat 3 · The judge says HOLD and the driver hears why (1:10–1:55)

| Time | On screen | Voice-over | Receipt |
|---|---|---|---|
| 1:10 | `tinytms/verify.py`, the six rules highlighted one by one. | "Models report; the judge decides. Plain Python, no model inside, and fail-closed: a missing answer is a HOLD, never an ACCEPT." | — |
| 1:20 | The check-in on the API, shown with `python scripts/receipt_show.py receipts/20261028T153041947742Z-HOLD.json`: verdict **HOLD**, reasons `VOICE_NO_COUNT` · `COUNT_MISMATCH doc=12 vision=1` · `SEAL_MISSING voice`. | "Here the voice note was one second of silence. The paper says 12, the photo shows 1 pallet, the voice gave no count and no seal: HOLD, with every reason listed." | [`receipts/20261028T153041947742Z-HOLD.json`](../receipts/20261028T153041947742Z-HOLD.json) |
| 1:32 | The driver's explanation, in Spanish, large: «Su carga quedó en espera. El documento dice 12 piezas pero en la foto se ve 1 tarima, y en su mensaje no dijo cuántas piezas trae ni el sello. Vuelva a contar la carga, revise el sello MX-48213 y pida que le corrijan el documento si hace falta.» | [A human reads it aloud in Spanish: the repo writes text, it has no text-to-speech.] Then: "Muse Spark 1.3 wrote that in 4.307 s, from the judge's reasons and the witnesses' fields only: no image, no audio." | [`receipts/explain-20261028T160450Z.json`](../receipts/explain-20261028T160450Z.json) |
| 1:42 | The same explanation in English, small. | "The same in English, for the dispatcher." | [`receipts/explain-20261028T160459Z.json`](../receipts/explain-20261028T160459Z.json) |
| 1:46 | A model saying "ACCEPT" crossed out; the verdict line unchanged. | "It cannot change the verdict. When Spark ran out of budget at 600 output tokens, the HOLD stood and the receipt says ERROR." | [`receipts/explain-20261028T160346Z.json`](../receipts/explain-20261028T160346Z.json), [`receipts/explain-20261028T160427Z.json`](../receipts/explain-20261028T160427Z.json) |
| 1:51 | Three transcripts of the same synthetic note, one under the other: «payets» / «pajets» / «payettes», seal «MX» / «MX MX» / «SMX SMX». | "And the ear is not deterministic: same audio, three spellings. Nothing guesses «SMX» back to «MX»: the seals differ, so the judge holds." | [`receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json`](../receipts/live-voice-es-doce-pallets-16k-20261028T160244Z.json), [`receipts/20261028T160514217793Z-HOLD.json`](../receipts/20261028T160514217793Z-HOLD.json), [`receipts/20261028T160626939074Z-HOLD.json`](../receipts/20261028T160626939074Z-HOLD.json) |

## Beat 4 · Pull the plug: the same check-in with and without the API (1:55–2:35)

| Time | On screen | Voice-over | Receipt |
|---|---|---|---|
| 1:55 | Terminal: `python scripts/pull_the_plug.py --load evals/public/boxes-pallet.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/es-doce-pallets-16k.wav` [Saturday: the real photo, paper and voice note]. | "Now pull the plug. Same three inputs, two runs: first through the Meta API, then with the API unreachable." | — |
| 2:02 | The side-by-side summary. Left, run 1 `--mode api`: each witness `meta-api`, its model and its seconds `[to be measured on Saturday]`. Right, run 2 `--mode auto` with `TINYTMS_META_BASE_URL` on a closed port of the Mac itself: vision and doc `fallback` to `local-mlx`, voice `NO_MODEL`. | "On the right the API cannot be reached, so paper and photo fall back to Muse Glimmer 30B, Meta's open weights, 4-bit, on the Mac through mlx_vlm, with no network and no download. Seconds on the Mac: `[to be measured on Saturday]`." | — |
| 2:10 | Paper fields and photo count, provider `local-mlx` [the local run we have until Saturday's run-2 receipt replaces it]. | "Glimmer reads the same paper: 12 pieces, 4800 kg, seal MX-48213, no hazmat. It sees 1 pallet; that answer took 9.924 s." | [`receipts/20261028T165434299357Z-HOLD.json`](../receipts/20261028T165434299357Z-HOLD.json) |
| 2:18 | `python scripts/receipt_show.py receipts/<run-2 receipt>.json`: «signature OK», then per witness its provider, model and seconds, the fallback reason, and the verdict **HOLD** `NO_MODEL:voice`. | "And it holds: there is no local speech-to-text yet, so the voice witness is NO_MODEL. Fail-closed, even offline, and the receipt says which model answered what." | [`receipts/20261028T165434299357Z-HOLD.json`](../receipts/20261028T165434299357Z-HOLD.json) |
| 2:27 | Measurement card: load, speed, memory. | "The model loads in 1.2 s from a warm cache and generates at 30 tok/s, peak 20.7 GB, one run on one image." | [`receipts/local-vlm-20261028T152737Z.json`](../receipts/local-vlm-20261028T152737Z.json) |

## Beat 5 · What is real and what is next (2:35–3:00)

| Time | On screen | Voice-over | Receipt |
|---|---|---|---|
| 2:35 | `python scripts/receipt_show.py receipts/20261028T153041947742Z-HOLD.json` → «signature OK»; the same file with one byte changed → «signature FAILED», exit 1 (`tests/test_receipt_show.py`, `tests/test_receipt.py`). | "Every run you saw is a signed receipt. Change one byte and it no longer verifies." | [`receipts/20261028T153041947742Z-HOLD.json`](../receipts/20261028T153041947742Z-HOLD.json) |
| 2:42 | The call ledger scrolling. | "Every Meta API call is in the ledger: 39 calls on 2026-10-28, all HTTP 200, under per-model caps." | [`receipts/llamadas.json`](../receipts/llamadas.json) |
| 2:48 | Two columns. REAL: the Meta API answers, one local Glimmer check-in. SYNTHETIC: the Carta de Porte, the voice notes, the FIXTURE cases. | "What is real: the model answers and the local run. What is synthetic is labelled synthetic." | [`receipts/live-spark-20261028T153211Z.json`](../receipts/live-spark-20261028T153211Z.json), [`receipts/20261028T165434299357Z-HOLD.json`](../receipts/20261028T165434299357Z-HOLD.json) |
| 2:53 | Next: real boxes · a driver's own voice · local speech-to-text. | "Next: real boxes `[to be measured on Saturday]`, the driver's own voice `[to be measured on Saturday]`, and a local ear so the Mac can accept a load on its own." | — |
| 2:58 | End card: **tinyTMS.IO** · Apache-2.0 · github.com/Blue-Ocean-Solutions/TinyTMS | — | — |

## Shot list: commands behind the screens

All of these exist today; the ones marked with a key need `MODEL_API_KEY` in the environment and spend credits.

```bash
python -m tinytms.cli checkin --mode fixture --fixture-dir fixtures/count_mismatch   # no key: a FIXTURE HOLD, for rehearsal
python scripts/live_eval.py --save-test-data tests/data                                # key: the beat-2 witness receipts
python -m tinytms.cli checkin --mode api --explain api --lang es --load evals/public/boxes-pallet.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/es-doce-pallets-16k.wav   # key: beat 3
python -m tinytms.cli checkin --mode local --load evals/public/boxes-pallet.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/es-doce-pallets-16k.wav   # Mac with Glimmer cached: the local run alone
python scripts/pull_the_plug.py --load evals/public/boxes-pallet.jpg --doc evals/doc/carta_sintetica.png --voice evals/voice/es-doce-pallets-16k.wav   # key + Mac with Glimmer: beat 4, two receipts side by side
python scripts/receipt_show.py receipts/<file>.json                                  # no key: any receipt on one screen, signature first
python -m tinytms.web --port 8765 --receipts-dir receipts                              # the one-page web, if a beat uses it
python scripts/maqueta.py --dry-run count_mismatch                                     # no key: rehearse the Saturday command
python scripts/maqueta.py --load camera:0 --doc camera:0 --voice nota.wav --mode auto --explain api --lang es   # key + camera: Saturday's real scene, ACTA.md + receipt
```

Re-running a command writes a **new** receipt; quote the new file, never edit an old one.

## Open for a human

- [Who narrates, in which language, and who reads the Spanish explanation aloud in beat 3.]
- [Saturday: footage of real boxes and a real driver's note, each with its own receipt, replacing every `[to be measured on Saturday]`.]
- [Saturday, beat 4: run `scripts/pull_the_plug.py` on the Mac with the real inputs and quote each run's seconds from its two new receipts; until then rows 2:10 and 2:18 show the 2026-10-28 local run.]
- [Whether beat 3 should be re-shot with a fresh `--explain api` run on the SYNTHETIC-TTS note now that the parser reads «doce payets» as 12; the two explained check-ins in `receipts/` predate that fix.]
