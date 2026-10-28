# Procedencia de `evals/` (copiado el 2026-10-28 por la lane F1-api-tinytms-2 desde `../evals/` de F1)

| fichero | qué es | origen |
|---|---|---|
| `public/boxes-moving.jpg` | foto real de cajas de mudanza | Wikimedia Commons, CC BY 2.0 — detalle, URL y atribución en `public/PROCEDENCIA.md`; sumas en `public/SHA256SUMS` |
| `public/boxes-pallet.jpg` | foto real de cajas apiladas sobre un pallet en un almacén | Wikimedia Commons, dominio público — `public/PROCEDENCIA.md` |
| `doc/carta_sintetica.png` | Carta de Porte **sintética** dibujada por F1 (datos inventados: 12 piezas, 4800 kg, sello MX-48213) | F1, 2026-10-28; ver `FORMAS-MEDIDAS.md` de F1 |
| `voice/silencio_16k.wav` | 1 s de silencio, WAV mono PCM16 16 kHz | F1, 2026-10-28 (sintético) |
| `voice/es-doce-pallets-16k.wav` | **SYNTHETIC-TTS**: nota de voz sintética, TTS de macOS, voz Paulina (es-MX); texto: «Buenas, aquí Ramón, del tráiler cuatro. Traigo doce pallets de madera. El sello es M X guión cuatro ocho dos uno tres. Peso cuatro mil ochocientos kilos.»; 11.36 s, WAV mono PCM16 16 kHz; sha256 `556f30380504357582c87bbc52f08a92a5a631d12139e1d77508d4df29f2e1ea`. No es un chofer real. | F1, 2026-10-28; caso `voice-es` de `scripts/live_eval.py` |

Son entradas de prueba. Las salidas de modelo sobre ellas están en `receipts/live-*.json` (firmados) y en `tests/data/real-*.json`.
