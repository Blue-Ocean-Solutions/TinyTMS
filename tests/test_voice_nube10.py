"""ORDER Nube-10 (2026-11-03): close #24's gaps, stop holding good loads.

House rule: the voice witness returns the right value, or None with a non-empty reason. It never picks one of two
readings and never returns a wrong value, with or without a reason. Every phrase of the order is pinned here inside an
otherwise good note, with its exact count, seal, reason and the judge's decision against the paper (12 pieces, 4800 kg,
seal MX-48213, no hazmat; vision 12). Then the order's own word lists (items 2, 4, 5), each in a note; and, for every
scan, a list word at the far end of a long sentence: no scan has a fixed word-count window, each ends at the end of
its sentence, or of the note for a damaged, missing or retracted seal (item 9). tests/test_voice_words.py pins every
word of every list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinytms.adapters import from_fixture, voice_from_transcript
from tinytms.verify import Limits, judge
from tinytms.witness import DocWitness, VisionWitness

V = VisionWitness(status="OK", model="FIXTURE", provider="FIXTURE", count=12)
D = DocWitness(status="OK", model="FIXTURE", provider="FIXTURE", pieces=12, weight_kg=4800.0, seal="MX-48213",
               hazmat=False)
L = Limits(max_weight_kg=24000)
ROOT = Path(__file__).resolve().parent.parent


def check(text: str, count, seal, detail: str, decision: str) -> None:
    w = voice_from_transcript({"transcript": text}, model="prueba", provider="none", source_sha256="")
    assert (w.count, w.seal, w.detail) == (count, seal, detail), text
    assert judge(V, D, w, L).decision == decision, text
    # the house rule: a field said and not read always says why
    assert (count is not None and seal is not None) or detail, text


# (item, note, count, seal, detail, decision). "keep": gave None on f8caa74 and must keep it; "good": must read.
ORDER = [
    ("1a", "Traigo doce tarimas en la caja, no trece. Sello MX-48213.", None, "MX-48213",
     "counts disagree in transcript: [12, 13] (self-correction: 'no')", "HOLD"),
    ("1a", "Twelve pallets on the truck, no thirteen. Seal MX-48213.", None, "MX-48213",
     "counts disagree in transcript: [12, 13] (self-correction: 'no')", "HOLD"),
    ("1a", "Doce tarimas en el andén, no trece. Sello MX-48213.", None, "MX-48213",
     "counts disagree in transcript: [12, 13] (self-correction: 'no')", "HOLD"),
    ("1a", "Doce tarimas, andén no cuatro. Sello MX-48213.", None, "MX-48213",
     "counts disagree in transcript: [4, 12] (self-correction: 'no')", "HOLD"),
    ("1b", "Twelve pallets, sorry, one short. Seal MX-48213.", None, "MX-48213",
     "unclear count in transcript: 'one short'", "HOLD"),
    ("1b", "Doce tarimas, perdón, un par más. Sello MX-48213.", None, "MX-48213",
     "unclear count in transcript: 'un par'", "HOLD"),
    ("1b", "Doce tarimas, no, una faltante. Sello MX-48213.", None, "MX-48213",
     "unclear count in transcript: 'una faltante'", "HOLD"),
    ("1b keep", "Doce tarimas, perdón, una menos. Sello MX-48213.", None, "MX-48213",
     "unclear count in transcript: 'una menos'", "HOLD"),
    ("1b keep", "Twelve pallets, wait, one less. Seal MX-48213.", None, "MX-48213",
     "unclear count in transcript: 'one less'", "HOLD"),
    ("1b good", "Doce tarimas. Sello MX-48213. Bueno, un saludo.", 12, "MX-48213", "", "ACCEPT"),
    ("1c", "Doce tarimas, sello MX-48213, no me coincide.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 , no me coincide'", "HOLD"),
    ("1c", "Twelve pallets, seal MX-48213, it doesn't even match.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'seal mx-48213 , it doesn t even match'", "HOLD"),
    ("1c", "Twelve pallets, seal MX-48213, it doesn't quite match.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'seal mx-48213 , it doesn t quite match'", "HOLD"),
    ("1c", "Doce tarimas, sello MX-48213, no coincide con el material peligroso de la hoja.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 , no coincide'", "HOLD"),
    ("1c", "Twelve pallets, seal MX-48213 doesn't match the hazmat paperwork.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'seal mx-48213 doesn t match'", "HOLD"),
    ("2", "Son doce tarimas. Sello MX-48213. Está roto.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . está roto'", "HOLD"),
    ("2", "Twelve pallets. Seal MX-48213. It's broken.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'seal mx-48213 . it s broken'", "HOLD"),
    ("2", "Doce tarimas. Sello MX-48213. No coincide.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . no coincide'", "HOLD"),
    ("2", "Traigo doce tarimas, el sello MX-48213, no está.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 , no está'", "HOLD"),
    ("2", "Twelve pallets, seal MX-48213, don't have it.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'seal mx-48213 , don t have it'", "HOLD"),
    ("2", "Twelve pallets, they broke the seal MX-48213.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'broke the seal'", "HOLD"),
    ("2", "Traigo doce tarimas, le quitaron el sello MX-48213.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'quitaron el sello'", "HOLD"),
    ("2", "Traigo doce tarimas, el sello es MX-48213 pero cuando lo revisé ya estaba roto.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello es mx-48213 pero cuando lo revisé ya "
     "estaba roto'", "HOLD"),
    ("2 retract", "Doce tarimas. Sello MX-48213. No, ese es el de la otra caja.", None, None,
     "count taken back in transcript: 'ese es el de la otra caja .'; seal taken back in transcript: 'ese es el de la "
     "otra caja .'", "HOLD"),
    ("2 retract", "Twelve pallets, seal MX-48213. Wait, no, that's from my last load.", None, None,
     "count taken back in transcript: 'that s from my last load .'; seal taken back in transcript: 'that s from my "
     "last load .'", "HOLD"),
    ("2 retract", "Doce tarimas. Sello MX-48213, ese era el de ayer.", None, None,
     "count taken back in transcript: 'ese era el de ayer .'; seal taken back in transcript: 'ese era el de ayer .'",
     "HOLD"),
    ("2 retract", "Doce tarimas. Sello MX-48213. Mentira.", None, None,
     "count taken back in transcript: 'mentira .'; seal taken back in transcript: 'mentira .'", "HOLD"),
    ("2 good", "Doce tarimas. Sello MX-48213, nada roto.", 12, "MX-48213", "", "ACCEPT"),
    ("2 good", "Twelve pallets. Seal MX-48213, nothing broken.", 12, "MX-48213", "", "ACCEPT"),
    ("2 good", "Twelve pallets. Seal MX-48213. Seal's intact, nothing wrong with it.", 12, "MX-48213", "", "ACCEPT"),
    ("2 good", "Twelve pallets, nothing missing, seal MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("2 good", "Doce tarimas. Sello MX-48213, no está roto.", 12, "MX-48213", "", "ACCEPT"),
    ("2 good", "Doce tarimas. Sello MX-48213, no hazmat.", 12, "MX-48213", "", "ACCEPT"),
    ("2 good", "Doce tarimas. Sello MX-48213, sin material peligroso.", 12, "MX-48213", "", "ACCEPT"),
    ("3", "Twelve pallets. Seal MX-48213. Sorry, the number is MX-48231.", 12, None,
     "seals disagree in transcript: ['MX48213', 'MX48231']", "HOLD"),
    ("3", "Doce tarimas. Perdón, el número es trece. Sello MX-48213.", None, "MX-48213",
     "counts disagree in transcript: [12, 13]", "HOLD"),
    ("3", "Twelve pallets. Thirteen honestly. Seal MX-48213.", None, "MX-48213",
     "counts disagree in transcript: [12, 13]", "HOLD"),
    ("4", "Doce tarimas. La oficina me pasó el sello MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'oficina me pasó el sello'", "HOLD"),
    ("4", "Twelve pallets. Per the docs, the seal is MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'per the docs , the seal'", "HOLD"),
    ("4", "The sheet has twelve pallets. Seal MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'sheet has twelve pallets'", "HOLD"),
    ("4", "Doce tarimas. En la remisión viene el sello MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'remisión viene el sello'", "HOLD"),
    ("4 good", "Vengo de la planta con doce tarimas. Sello MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("4 good", "Traigo doce tarimas del cliente. Sello MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("4 good", "Twelve pallets. Hey, I came straight from the shipper and the seal is MX-48213.", 12, "MX-48213", "",
     "ACCEPT"),
    ("4 good", "Doce tarimas. Ya pasé la caseta y el sello es MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("4 keep", "En la planta me contaron doce tarimas. Sello MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'planta me contaron doce tarimas'", "HOLD"),
    ("4 keep", "El cliente contó doce tarimas. Sello MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'cliente contó doce tarimas'", "HOLD"),
    ("5", "Doce tarimas. No sé si el sello es MX-48213.", 12, None, "hedged seal in transcript: 'no sé si el sello'",
     "HOLD"),
    ("5", "Ponle que doce tarimas. Sello MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'ponle que doce tarimas'", "HOLD"),
    ("5", "I'd say twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'i d say twelve pallets'", "HOLD"),
    ("5", "Twelve pallets. Seal MX-48213, if I'm not wrong.", 12, None,
     "hedged seal in transcript: 'seal mx-48213 , if i m not wrong'", "HOLD"),
    ("5", "Creo que lo que traigo son doce tarimas. Sello MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'creo que lo que traigo son doce tarimas'", "HOLD"),
    ("6", "Twelve pallets. The seal is el em ex four eight two one three.", 12, None,
     "unclear seal letters: 'seal is el em ex'", "HOLD"),
    ("6", "Doce tarimas. Sello ese eme equis cuatro ocho dos uno tres.", 12, None,
     "unclear seal letters: 'sello ese eme'", "HOLD"),
    ("6 either", "Twelve pallets. The seal is, see, MX-48213.", 12, None,
     "unclear seal letters: 'seal is , see , mx-48213'", "HOLD"),
    ("6 either", "Twelve pallets, seal MX-48213, PO number 55123.", 12, "MX-48213", "", "ACCEPT"),
    ("6 either", "Twelve pallets, seal MX-48213, BOL number 77812.", 12, "MX-48213", "", "ACCEPT"),
    ("6 good", "Twelve pallets. Seal, mmh, MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Twelve pallets. Seal MX-48213. Forty-eight hundred kilos.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Twelve pallets. Seal MX-48213. 4,800 kilograms.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Doce tarimas. Sello MX-48213. Y pesa cuatro mil ochocientos kilos.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Twelve pallets. Seal MX-48213. Have a good one.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Doce tarimas. Sello MX-48213. Fue un viaje largo.", 12, "MX-48213", "", "ACCEPT"),
    ("7 count", "Twelve pallets, one more. Seal MX-48213.", None, "MX-48213", "unclear count in transcript: 'one more'",
     "HOLD"),
    ("7 count", "Doce tarimas, una más. Sello MX-48213.", None, "MX-48213", "unclear count in transcript: 'una más'",
     "HOLD"),
    ("7 count", "Twelve pallets, one short. Seal MX-48213.", None, "MX-48213",
     "unclear count in transcript: 'one short'", "HOLD"),
    ("7 count", "Doce tarimas, una menos. Sello MX-48213.", None, "MX-48213",
     "unclear count in transcript: 'una menos'", "HOLD"),
    ("7", "Twelve pallets. Seal MX-48213. Got here at 1:45.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Twelve pallets. Seal MX-48213. PO 55123.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Twelve pallets. Seal MX-48213. My cell is 956-555-0142.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Twelve pallets. Seal MX-48213. It's a 53-foot dry van.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Doce tarimas. Sello MX-48213. Las placas son 45-AB-7C.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Doce tarimas. Sello MX-48213. El resto está bien.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Twelve pallets. Seal MX-48213. The rest is fine.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Twelve pallets. Seal MX-48213. Going to get some rest.", 12, "MX-48213", "", "ACCEPT"),
    ("7", "Twelve pallets. Seal MX-48213. Stopping at a rest area.", 12, "MX-48213", "", "ACCEPT"),
    ("7 keep", "Doce tarimas. Sello MX-482, el resto no se ve.", 12, None,
     "seal not read in full in transcript: 'sello mx-482 , el resto'", "HOLD"),
    ("8", "Doce tarimas. El sello pues es MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("8", "Doce tarimas. El sello, eh, es MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("8", "Doce tarimas. El sello este es MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("8", "Doce tarimas. El sello a ver es MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("8", "Twelve pallets. The seal, um, is MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("8", "Twelve pallets. The seal, let me see, is MX-48213.", 12, "MX-48213", "", "ACCEPT"),
]


@pytest.mark.parametrize("item,text,count,seal,detail,decision", ORDER, ids=[f"{r[0]}:{r[1][:40]}" for r in ORDER])
def test_every_phrase_of_the_order(item, text, count, seal, detail, decision):
    check(text, count, seal, detail, decision)


# Item 1a as the order writes them, with no seal: the count is None and says why.
@pytest.mark.parametrize("text,detail", [
    ("Traigo doce tarimas en la caja, no trece.", "counts disagree in transcript: [12, 13] (self-correction: 'no')"),
    ("Twelve pallets on the truck, no thirteen.", "counts disagree in transcript: [12, 13] (self-correction: 'no')"),
    ("Doce tarimas en el andén, no trece.", "counts disagree in transcript: [12, 13] (self-correction: 'no')"),
    ("Doce tarimas, andén no cuatro.", "counts disagree in transcript: [4, 12] (self-correction: 'no')"),
    ("Doce tarimas, andén No. 4.", ""),  # the abbreviation before digits names the dock: count 12
])
def test_1a_a_correction_word_is_never_the_abbreviation(text, detail):
    w = voice_from_transcript({"transcript": text}, model="prueba", provider="none", source_sha256="")
    assert (w.count, w.detail) == ((12 if not detail else None), detail), text


# Item 9: every scan ends at the end of the sentence (or of the note, item 2), never after a fixed number of words.
LONG = [
    ("count hedge after",
     "Traigo doce tarimas que cargué esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo "
     "con mucho calor, creo. Sello MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'doce tarimas que cargué … viejo con mucho calor , creo'", "HOLD"),
    ("count hedge before",
     "Creo que esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo con mucho calor "
     "traigo doce tarimas. Sello MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'creo que esta mañana … con mucho calor traigo doce tarimas'", "HOLD"),
    ("count source after",
     "Traigo doce tarimas que cargué esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo "
     "con mucho calor, según la hoja. Sello MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'doce tarimas que cargué … viejo con mucho calor , según'",
     "HOLD"),
    ("count source before",
     "Según la hoja esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo con mucho calor "
     "son doce tarimas. Sello MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'según la hoja esta … con mucho calor son doce tarimas'", "HOLD"),
    ("count negation before",
     "No esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo con mucho calor traigo doce "
     "tarimas. Sello MX-48213.", None, "MX-48213",
     "negated count in transcript: 'no esta mañana muy … con mucho calor traigo doce tarimas'", "HOLD"),
    ("count more after",
     "Traigo doce tarimas que cargué esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo "
     "con mucho calor y una suelta. Sello MX-48213.", None, "MX-48213", "unclear count in transcript: 'una suelta'",
     "HOLD"),
    ("count weight before",
     "Pesa esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo con mucho calor veinte "
     "mil doce tarimas. Sello MX-48213.", None, "MX-48213",
     "count runs into a weight: 'pesa esta mañana muy … mucho calor veinte mil doce tarimas'", "HOLD"),
    ("count retraction later",
     "Doce tarimas. Sello MX-48213. Lo de las tarimas esta mañana muy temprano antes de salir rumbo a la frontera "
     "por el camino viejo con mucho calor, mentira.", None, None,
     "count taken back in transcript: 'mentira .'; seal taken back in transcript: 'mentira .'", "HOLD"),
    ("count doubt later",
     "Doce tarimas. Sello MX-48213. This morning before heading south through the hills with the windows down and "
     "the radio on, I didn't count.", None, "MX-48213", "count not checked in transcript: 'doce tarimas'", "HOLD"),
    ("count hedge after en",
     "I have twelve pallets that I picked up early this morning before heading south through the hills with the "
     "windows down and the radio on, I think. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'twelve pallets that i … the radio on , i think'", "HOLD"),
    ("seal hedge before",
     "Creo que esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo con mucho calor el "
     "sello es MX-48213. Doce tarimas.", 12, None,
     "hedged seal in transcript: 'creo que esta mañana … viejo con mucho calor el sello'", "HOLD"),
    ("seal hedge after",
     "El sello es MX-48213 que cargué esta mañana muy temprano antes de salir rumbo a la frontera por el camino "
     "viejo con mucho calor, creo. Doce tarimas.", 12, None,
     "hedged seal in transcript: 'sello es mx-48213 que … viejo con mucho calor , creo'", "HOLD"),
    ("seal source before",
     "Según la hoja esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo con mucho calor "
     "el sello es MX-48213. Doce tarimas.", 12, None,
     "seal said from the paperwork or someone else: 'según la hoja esta … viejo con mucho calor el sello'", "HOLD"),
    ("seal source after",
     "Sello MX-48213 que cargué esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo con "
     "mucho calor, eso dice la hoja. Doce tarimas.", 12, None,
     "seal said from the paperwork or someone else: 'sello mx-48213 que cargué … calor , eso dice la hoja'", "HOLD"),
    ("seal damage before",
     "Rompieron esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo con mucho calor el "
     "sello MX-48213. Doce tarimas.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'rompieron esta mañana muy … viejo con mucho "
     "calor el sello'", "HOLD"),
    ("seal damage later",
     "Doce tarimas. Sello MX-48213. Lo revisé esta mañana muy temprano antes de salir rumbo a la frontera por el "
     "camino viejo con mucho calor y está roto.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo … con mucho calor y está roto'",
     "HOLD"),
    ("seal damage later en",
     "Twelve pallets. Seal MX-48213. I checked it this morning before heading south through the hills with the "
     "windows down and the radio on and it was cracked.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'seal mx-48213 . i … radio on and it was cracked'",
     "HOLD"),
    ("seal match denied",
     "Doce tarimas. Sello MX-48213, no esta mañana muy temprano antes de salir rumbo a la frontera por el camino "
     "viejo con mucho calor coincide.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 , no … camino viejo con mucho "
     "calor coincide'", "HOLD"),
    ("seal missing later",
     "Doce tarimas. Sello MX-48213. Lo busqué esta mañana muy temprano antes de salir rumbo a la frontera por el "
     "camino viejo con mucho calor, no está.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo … con mucho calor , no está'",
     "HOLD"),
    ("seal retraction later",
     "Twelve pallets. Seal MX-48213. I read that this morning before heading south through the hills with the "
     "windows down and the radio on off my last load.", None, None,
     "count taken back in transcript: 'that this morning before heading south through the'; seal taken back in "
     "transcript: 'that this morning before heading south through the'", "HOLD"),
    ("seal partial after",
     "Doce tarimas. Sello MX-48213 esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo "
     "con mucho calor los últimos.", 12, None,
     "part of a seal said in transcript: 'sello mx-48213 esta mañana … viejo con mucho calor los últimos'", "HOLD"),
    ("seal letter after",
     "Doce tarimas. Sello MX-48213 esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo "
     "con mucho calor con letra.", 12, None,
     "seal letter said after the digits: 'sello mx-48213 esta mañana … viejo con mucho calor con letra'", "HOLD"),
    ("seal unread later",
     "Doce tarimas. Sello MX-482. Lo revisé esta mañana muy temprano antes de salir rumbo a la frontera por el "
     "camino viejo con mucho calor y no se ve el resto.", 12, None,
     "seal not read in full in transcript: 'no se ve el resto'", "HOLD"),
    ("trailer damage",
     "La caja esta mañana muy temprano antes de salir rumbo a la frontera por el camino viejo con mucho calor viene "
     "abierta. Doce tarimas. Sello MX-48213.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'la caja esta mañana … viejo con mucho calor "
     "viene abierta'", "HOLD"),
]


@pytest.mark.parametrize("scan,text,count,seal,detail,decision", LONG, ids=[r[0] for r in LONG])
def test_every_scan_reaches_the_end_of_its_sentence(scan, text, count, seal, detail, decision):
    assert len(text.split()) >= 18
    check(text, count, seal, detail, decision)


# Item 2: the order's damage words, each the last word of a later sentence.
ITEM2 = [
    ("cracked", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y cracked.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y cracked'", "HOLD"),
    ("cut", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y cut.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y cut'", "HOLD"),
    ("busted", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y busted.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y busted'", "HOLD"),
    ("snapped", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y snapped.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y snapped'", "HOLD"),
    ("ripped", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y ripped.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y ripped'", "HOLD"),
    ("torn", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y torn.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y torn'", "HOLD"),
    ("bent", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y bent.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y bent'", "HOLD"),
    ("popped", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y popped.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y popped'", "HOLD"),
    ("twisted", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y twisted.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y twisted'", "HOLD"),
    ("pulled", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y pulled.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y pulled'", "HOLD"),
    ("sheared", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y sheared.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y sheared'", "HOLD"),
    ("quitaron", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y quitaron.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y quitaron'", "HOLD"),
    ("arrancaron", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y arrancaron.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y arrancaron'", "HOLD"),
    ("reventado", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y reventado.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y reventado'", "HOLD"),
    ("chueco", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y chueco.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y chueco'", "HOLD"),
    ("flojo", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y flojo.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y flojo'", "HOLD"),
    ("doblado", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y doblado.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y doblado'", "HOLD"),
    ("colgando", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y colgando.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y colgando'", "HOLD"),
    ("desprendido", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y desprendido.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y desprendido'", "HOLD"),
    ("cambiaron", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y cambiaron.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo revisé esta mañana antes de "
     "salir y cambiaron'", "HOLD"),
    ("se cayó", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y se cayó.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo … antes de salir y se cayó'",
     "HOLD"),
    ("se lo llevaron", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y se lo llevaron.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo … de salir y se lo llevaron'",
     "HOLD"),
    ("no sirve", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y no sirve.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo … antes de salir y no sirve'",
     "HOLD"),
    ("se zafó", "Doce tarimas. Sello MX-48213. Lo revisé esta mañana antes de salir y se zafó.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'sello mx-48213 . lo … antes de salir y se zafó'",
     "HOLD"),
]


@pytest.mark.parametrize("word,text,count,seal,detail,decision", ITEM2, ids=[r[0] for r in ITEM2])
def test_2_a_damaged_seal_said_in_a_later_sentence(word, text, count, seal, detail, decision):
    check(text, count, seal, detail, decision)


# Item 4: the order's verbs of telling or giving, and documents with has/shows/trae/viene/aparece/anotó/wrote down.
ITEM4 = [
    ("dice", "Doce tarimas. La oficina dice el sello MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'oficina dice el sello'", "HOLD"),
    ("says", "Twelve pallets. The office says the seal MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'office says the seal'", "HOLD"),
    ("told", "Twelve pallets. The office told the seal MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'office told the seal'", "HOLD"),
    ("me dijeron", "Doce tarimas. La oficina me dijeron el sello MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'oficina me dijeron el sello'", "HOLD"),
    ("me pasó", "Doce tarimas. La oficina me pasó el sello MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'oficina me pasó el sello'", "HOLD"),
    ("me mandó", "Doce tarimas. La oficina me mandó el sello MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'oficina me mandó el sello'", "HOLD"),
    ("me dieron", "Doce tarimas. La oficina me dieron el sello MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'oficina me dieron el sello'", "HOLD"),
    ("gave me", "Twelve pallets. The office gave me the seal MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'office gave me the seal'", "HOLD"),
    ("sent me", "Twelve pallets. The office sent me the seal MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'office sent me the seal'", "HOLD"),
    ("docs has", "The docs has twelve pallets. Seal MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'docs has twelve pallets'", "HOLD"),
    ("sheet shows", "The sheet shows twelve pallets. Seal MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'sheet shows twelve pallets'", "HOLD"),
    ("paperwork has", "The paperwork has twelve pallets. Seal MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'paperwork has twelve pallets'", "HOLD"),
    ("carta trae", "The carta trae twelve pallets. Seal MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'carta trae twelve pallets'", "HOLD"),
    ("guía viene", "The guía viene twelve pallets. Seal MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'guía viene twelve pallets'", "HOLD"),
    ("remisión aparece", "The remisión aparece twelve pallets. Seal MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'remisión aparece twelve pallets'", "HOLD"),
    ("sistema trae", "The sistema trae twelve pallets. Seal MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'sistema trae twelve pallets'", "HOLD"),
    ("app shows", "The app shows twelve pallets. Seal MX-48213.", None, "MX-48213",
     "count said from the paperwork or someone else: 'app shows twelve pallets'", "HOLD"),
    ("la hoja anotó", "Doce tarimas. La hoja anotó el sello MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'hoja anotó el sello'", "HOLD"),
    ("the sheet wrote down", "Doce tarimas. The sheet wrote down el sello MX-48213.", 12, None,
     "seal said from the paperwork or someone else: 'sheet wrote down el sello'", "HOLD"),
]


@pytest.mark.parametrize("words,text,count,seal,detail,decision", ITEM4, ids=[r[0] for r in ITEM4])
def test_4_a_second_hand_value(words, text, count, seal, detail, decision):
    check(text, count, seal, detail, decision)


# Item 5: the order's hedges, before the seal and before the count.
ITEM5 = [
    ("no sé si / seal", "Doce tarimas. No sé si el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'no sé si el sello'", "HOLD"),
    ("no sé si / count", "No sé si twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'no sé si twelve pallets'", "HOLD"),
    ("que yo sepa / seal", "Doce tarimas. Que yo sepa el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'que yo sepa el sello'", "HOLD"),
    ("que yo sepa / count", "Que yo sepa twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'que yo sepa twelve pallets'", "HOLD"),
    ("hasta donde sé / seal", "Doce tarimas. Hasta donde sé el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'hasta donde sé el sello'", "HOLD"),
    ("hasta donde sé / count", "Hasta donde sé twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'hasta donde sé twelve pallets'", "HOLD"),
    ("casi seguro / seal", "Doce tarimas. Casi seguro el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'casi seguro el sello'", "HOLD"),
    ("casi seguro / count", "Casi seguro twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'casi seguro twelve pallets'", "HOLD"),
    ("igual y / seal", "Doce tarimas. Igual y el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'igual y el sello'", "HOLD"),
    ("igual y / count", "Igual y twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'igual y twelve pallets'", "HOLD"),
    ("a lo mejor / seal", "Doce tarimas. A lo mejor el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'a lo mejor el sello'", "HOLD"),
    ("a lo mejor / count", "A lo mejor twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'a lo mejor twelve pallets'", "HOLD"),
    ("se supone / seal", "Doce tarimas. Se supone el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'se supone el sello'", "HOLD"),
    ("se supone / count", "Se supone twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'se supone twelve pallets'", "HOLD"),
    ("ponle que / seal", "Doce tarimas. Ponle que el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'ponle que el sello'", "HOLD"),
    ("ponle que / count", "Ponle que twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'ponle que twelve pallets'", "HOLD"),
    ("yo diría / seal", "Doce tarimas. Yo diría el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'diría el sello'", "HOLD"),
    ("yo diría / count", "Yo diría twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'diría twelve pallets'", "HOLD"),
    ("me late / seal", "Doce tarimas. Me late el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'me late el sello'", "HOLD"),
    ("me late / count", "Me late twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'me late twelve pallets'", "HOLD"),
    ("digamos / seal", "Doce tarimas. Digamos el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'digamos el sello'", "HOLD"),
    ("digamos / count", "Digamos twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'digamos twelve pallets'", "HOLD"),
    ("I'd say / seal", "Doce tarimas. I'd say el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'i d say el sello'", "HOLD"),
    ("I'd say / count", "I'd say twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'i d say twelve pallets'", "HOLD"),
    ("I reckon / seal", "Doce tarimas. I reckon el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'reckon el sello'", "HOLD"),
    ("I reckon / count", "I reckon twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'reckon twelve pallets'", "HOLD"),
    ("I suppose / seal", "Doce tarimas. I suppose el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'suppose el sello'", "HOLD"),
    ("I suppose / count", "I suppose twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'suppose twelve pallets'", "HOLD"),
    ("as far as I know / seal", "Doce tarimas. As far as I know el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'as far as i know el sello'", "HOLD"),
    ("as far as I know / count", "As far as I know twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'as far as i know twelve pallets'", "HOLD"),
    ("almost sure / seal", "Doce tarimas. Almost sure el sello es MX-48213.", 12, None,
     "hedged seal in transcript: 'almost sure el sello'", "HOLD"),
    ("almost sure / count", "Almost sure twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'almost sure twelve pallets'", "HOLD"),
    ("if I'm not wrong / seal", "Doce tarimas. If I'm not wrong el sello es MX-48213.", 12, None,
     "seal reported broken, missing or not matching in transcript: 'wrong el sello'", "HOLD"),
    ("if I'm not wrong / count", "If I'm not wrong twelve pallets. Seal MX-48213.", None, "MX-48213",
     "hedged count in transcript: 'if i m not wrong twelve pallets'", "HOLD"),
]


@pytest.mark.parametrize("hedge,text,count,seal,detail,decision", ITEM5, ids=[r[0] for r in ITEM5])
def test_5_a_hedge(hedge, text, count, seal, detail, decision):
    check(text, count, seal, detail, decision)


# Item 5: "-ish" is a hedge wherever it is glued: never part of a count or of an id; "finish", "Spanish" stay words.
ISH = [
    ("Twelve-ish pallets. Seal MX-48213.", None, "MX-48213", "hedged count in transcript: 'twelve ish'", "HOLD"),
    ("I have 12ish pallets. Seal MX-48213.", None, "MX-48213", "hedged count in transcript: '12 ish'", "HOLD"),
    ("Twelve pallets. Seal MX-48213-ish.", 12, None, "hedged seal in transcript: 'seal mx-48213 ish'", "HOLD"),
    ("Twelve pallets, I finish at five. Seal MX-48213.", 12, "MX-48213", "", "ACCEPT"),
    ("Twelve pallets of Spanish tile. Seal MX-48213.", 12, "MX-48213", "", "ACCEPT"),
]


@pytest.mark.parametrize("text,count,seal,detail,decision", ISH)
def test_5_ish_is_a_hedge(text, count, seal, detail, decision):
    check(text, count, seal, detail, decision)


# Item 9: the 3 pinned real transcripts and the demo lines give exactly what they gave on f8caa74.
REAL = {
    "Traigo doce payets de madera. El sello es MX guión cuatro ocho dos uno tres.": (12, "MX-48213", ""),
    "Traigo doce pajets de madera. El sello es MX MX guión cuatro ocho dos uno tres.": (12, "MX-48213", ""),
    "Traigo doce payettes de madera. El sello SMX SMX guión cuatro ocho dos uno tres.": (12, "SMX-48213", ""),
}
DEMO = {"accept": (12, "MX-48213"), "count_mismatch": (12, "MX-48213"), "seal_mismatch": (12, "MX-48231"),
        "overweight": (12, "MX-48213")}


def test_the_real_transcripts_read_as_on_f8caa74():
    seen = set()
    for f in sorted((ROOT / "tests" / "data").glob("real-voice-es-*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))["response_json"]["transcript"]
        (key,) = [k for k in REAL if k in t]
        seen.add(key)
        w = voice_from_transcript({"transcript": t}, model="prueba", provider="none", source_sha256="")
        assert (w.count, w.seal, w.detail) == REAL[key], t
    assert seen == set(REAL)


@pytest.mark.parametrize("name", sorted(DEMO))
def test_the_demo_lines_read_as_on_f8caa74(name):
    w = from_fixture(ROOT / "fixtures" / name / "voice.FIXTURE.json")
    assert (w.status, w.count, w.seal) == ("OK", *DEMO[name])
    assert w.detail.startswith("fixture voice.FIXTURE.json sha256=") and ";" not in w.detail, w.detail
