# Procedencia de `evals/` (copiado el 2026-10-28 por la lane F1-api-tinytms-2 desde `../evals/` de F1)

| fichero | qué es | origen |
|---|---|---|
| `public/boxes-moving.jpg` | foto real de cajas de mudanza | Wikimedia Commons, CC BY 2.0 — detalle, URL y atribución en `public/PROCEDENCIA.md`; sumas en `public/SHA256SUMS` |
| `public/boxes-pallet.jpg` | foto real de cajas apiladas sobre un pallet en un almacén. Por su procedencia no se usa en el demo ni en comandos nuevos; se queda como entrada de los recibos del 2026-10-28 | Wikimedia Commons, dominio público — `public/PROCEDENCIA.md` |
| `doc/carta_sintetica.png` | Carta de Porte **sintética** dibujada por F1 (datos inventados: 12 piezas, 4800 kg, sello MX-48213) | F1, 2026-10-28; ver `FORMAS-MEDIDAS.md` de F1 |
| `voice/silencio_16k.wav` | 1 s de silencio, WAV mono PCM16 16 kHz | F1, 2026-10-28 (sintético) |
| nota de voz en español, sha256 `556f30380504357582c87bbc52f08a92a5a631d12139e1d77508d4df29f2e1ea` (**retirada del árbol el 2026-11-01**) | **SYNTHETIC-TTS**: TTS de macOS, voz Paulina (es-MX), leyendo un texto escrito por F1; no era un chofer real. Se retiró porque la licencia de esa voz sólo permite uso personal y no comercial. Sus recibos (`receipts/live-voice-es-*.json` y los check-ins que la nombran) y sus tres transcripciones reales (`tests/data/real-voice-es-*.json`) se quedan como historia. | F1, 2026-10-28; era el caso `voice-es` de `scripts/live_eval.py`, que ahora toma `voice/es-*.wav` y se omite si no hay ninguna |

Son entradas de prueba. Las salidas de modelo sobre ellas están en `receipts/live-*.json` (firmados) y en `tests/data/real-*.json`.
