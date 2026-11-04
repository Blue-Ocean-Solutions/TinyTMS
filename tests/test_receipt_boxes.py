"""scripts/receipt_boxes.py: a SAM receipt's boxes on its own photo, drawn only when the receipt verifies."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tinytms.receipt import load_or_create_key, public_key_hex, sign_receipt

ROOT = Path(__file__).resolve().parent.parent
CARDBOARD = ROOT / "receipts" / "live-sam-boxes-moving-cardboard-box-20261028T153208Z.json"
PALLET = ROOT / "receipts" / "live-sam-boxes-moving-pallet-20261028T153209Z.json"
SPARK = ROOT / "receipts" / "live-spark-20261028T153211Z.json"
PHOTO = ROOT / "evals" / "public" / "boxes-moving.jpg"
OTHER_PHOTO = ROOT / "evals" / "doc" / "carta_sintetica.png"


def _script():
    spec = importlib.util.spec_from_file_location("receipt_boxes", ROOT / "scripts" / "receipt_boxes.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _body(source: Path, **changes) -> dict:
    """A real receipt's body, unsigned, with `changes` applied ("result__image_w" reaches into result)."""
    body = {k: v for k, v in json.loads(source.read_text(encoding="utf-8")).items()
            if k not in ("signature", "public_key")}
    for key, value in changes.items():
        *path, last = key.split("__")
        node = body
        for part in path:
            node = node[part]
        node[last] = value
    return body


def _signed(tmp_path: Path, body: dict) -> tuple[Path, Path]:
    """`body` signed by a scratch key, and a signer list naming only that key."""
    sk = load_or_create_key(tmp_path / "seat")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(sign_receipt(body, sk)), encoding="utf-8")
    signers = tmp_path / "known_signers.txt"
    signers.write_text(f"{public_key_hex(sk)}  # scratch seat for this test\n", encoding="utf-8")
    return receipt, signers


def test_the_four_cardboard_boxes_come_from_the_signed_receipt(tmp_path, capsys):
    m, out = _script(), tmp_path / "frame.png"
    assert m.main([str(CARDBOARD), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "signature OK · known signer" in printed and "drew 4 boxes" in printed
    frame, photo = cv2.imread(str(out)), cv2.imread(str(PHOTO))
    band = frame.shape[0] - photo.shape[0]
    assert frame.shape[1] == photo.shape[1] and band > 0  # the caption sits above the photo, never on it
    for x1, y1, x2, y2, _score in json.loads(CARDBOARD.read_text(encoding="utf-8"))["result"]["boxes"]:
        ys = (band + int(y1), band + int((y1 + y2) / 2), band + int(y2))
        on_the_outline = [frame[y, int(x)] for y in ys for x in (x1, x2)]
        assert all(tuple(int(c) for c in px) == m.BOX_BGR for px in on_the_outline), (x1, y1, x2, y2)


def test_the_pallet_receipt_draws_no_box_and_leaves_the_photo_untouched(tmp_path, capsys):
    out = tmp_path / "frame.png"
    assert _script().main([str(PALLET), "--out", str(out)]) == 0
    assert "drew 0 boxes" in capsys.readouterr().out
    frame, photo = cv2.imread(str(out)), cv2.imread(str(PHOTO))
    assert np.array_equal(frame[frame.shape[0] - photo.shape[0]:], photo)  # count 0: not one pixel changes


def test_one_changed_digit_draws_nothing(tmp_path, capsys):
    raw = CARDBOARD.read_text(encoding="utf-8")
    assert raw.count("780.0") == 1  # control: the coordinate we move is there, once
    tampered = tmp_path / CARDBOARD.name
    tampered.write_text(raw.replace("780.0", "781.0"), encoding="utf-8")
    out = tmp_path / "frame.png"
    assert _script().main([str(tampered), "--out", str(out)]) == 1
    assert "signature FAILED" in capsys.readouterr().out and not out.exists()


def test_the_same_body_signed_by_an_unknown_key_draws_nothing(tmp_path, capsys):
    receipt, _listed_only_in_tmp = _signed(tmp_path, _body(CARDBOARD))
    out = tmp_path / "frame.png"
    assert _script().main([str(receipt), "--out", str(out)]) == 1  # default list: keys/known_signers.txt
    assert "unknown signer" in capsys.readouterr().out and not out.exists()


def test_control_the_same_body_signed_by_a_listed_key_draws(tmp_path):
    receipt, signers = _signed(tmp_path, _body(CARDBOARD))
    out = tmp_path / "frame.png"
    assert _script().main([str(receipt), "--known-signers", str(signers), "--out", str(out)]) == 0
    assert out.exists()


def test_another_photo_draws_nothing(tmp_path, capsys):
    out = tmp_path / "frame.png"
    assert _script().main([str(CARDBOARD), "--image", str(OTHER_PHOTO), "--out", str(out)]) == 1
    assert "sha256" in capsys.readouterr().out and not out.exists()


def test_a_receipt_that_is_not_sam_draws_nothing(tmp_path, capsys):
    out = tmp_path / "frame.png"
    assert _script().main([str(SPARK), "--out", str(out)]) == 1
    assert "not a SAM" in capsys.readouterr().out and not out.exists()


def test_a_size_the_photo_does_not_have_draws_nothing(tmp_path, capsys):
    receipt, signers = _signed(tmp_path, _body(CARDBOARD, result__image_w=1000))
    out = tmp_path / "frame.png"
    assert _script().main([str(receipt), "--known-signers", str(signers), "--out", str(out)]) == 1
    assert "size" in capsys.readouterr().out and not out.exists()


@pytest.mark.parametrize("path", ["../outside.jpg", "/etc/hosts"])
def test_an_input_outside_the_repo_is_never_read(tmp_path, capsys, path):
    receipt, signers = _signed(tmp_path, _body(CARDBOARD, input__path=path))
    assert json.loads(receipt.read_text(encoding="utf-8"))["input"]["path"] == path  # control: signing kept it
    out = tmp_path / "frame.png"
    assert _script().main([str(receipt), "--known-signers", str(signers), "--out", str(out)]) == 1
    assert "outside the repo" in capsys.readouterr().out and not out.exists()


def test_the_frame_never_overwrites_the_photo(tmp_path):
    photo = tmp_path / "photo.png"
    photo.write_bytes(PHOTO.read_bytes())  # the same bytes, so the same sha256 the receipt signed
    with pytest.raises(SystemExit) as e:
        _script().main([str(CARDBOARD), "--image", str(photo), "--out", str(photo)])
    assert e.value.code == 2 and photo.read_bytes() == PHOTO.read_bytes()


def _jpeg(orientation: int | None) -> bytes:
    """A small photo that is not symmetric under a 180° turn, with an EXIF orientation tag when one is given."""
    img = np.zeros((40, 60, 3), np.uint8)
    img[:, :30] = (0, 0, 255)
    raw = cv2.imencode(".jpg", img)[1].tobytes()
    if orientation is None:
        return raw
    tiff = (b"MM\x00\x2a\x00\x00\x00\x08" + b"\x00\x01"
            + b"\x01\x12\x00\x03\x00\x00\x00\x01" + orientation.to_bytes(2, "big") + b"\x00\x00" + b"\x00\x00\x00\x00")
    payload = b"Exif\x00\x00" + tiff
    return raw[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + raw[2:]


def _sam_receipt_for(tmp_path: Path, photo: bytes) -> tuple[Path, Path, Path]:
    """The photo on disk and a one-box SAM receipt for exactly those bytes, signed by a listed scratch key."""
    import hashlib

    p = tmp_path / "photo.jpg"
    p.write_bytes(photo)
    sha = hashlib.sha256(photo).hexdigest()
    body = {"kind": "live-witness", "witness": "sam", "model": "sam-3.1", "concept": "box",
            "input": {"path": "evals/own/photo.jpg", "sha256": sha},
            "result": {"status": "OK", "count": 1, "image_w": 60, "image_h": 40, "source_sha256": sha,
                       "boxes": [[2.0, 2.0, 20.0, 20.0, 0.9]]}}
    receipt, signers = _signed(tmp_path, body)
    return p, receipt, signers


def test_a_photo_with_an_exif_rotation_draws_nothing(tmp_path, capsys):
    # Turned 180° the photo keeps its size, so only this check stops boxes drawn on the wrong side of it.
    photo, receipt, signers = _sam_receipt_for(tmp_path, _jpeg(orientation=3))
    out = tmp_path / "frame.png"
    assert _script().main([str(receipt), "--known-signers", str(signers), "--image", str(photo),
                           "--out", str(out)]) == 1
    assert "rotation" in capsys.readouterr().out and not out.exists()


def test_control_the_same_photo_without_the_tag_draws(tmp_path):
    photo, receipt, signers = _sam_receipt_for(tmp_path, _jpeg(orientation=None))
    out = tmp_path / "frame.png"
    assert _script().main([str(receipt), "--known-signers", str(signers), "--image", str(photo),
                           "--out", str(out)]) == 0
    assert out.exists()


def test_out_must_be_a_png(tmp_path):
    with pytest.raises(SystemExit) as e:
        _script().main([str(CARDBOARD), "--out", str(tmp_path / "frame.jpg")])
    assert e.value.code == 2


def test_a_missing_receipt_exits_3(tmp_path):
    assert _script().main([str(tmp_path / "nope.json"), "--out", str(tmp_path / "frame.png")]) == 3


GOOD = {"kind": "live-witness", "witness": "sam",
        "result": {"status": "OK", "count": 1, "image_w": 100, "image_h": 50, "boxes": [[10.0, 5.0, 60.0, 45.0, 0.9]]}}


def test_boxes_from_returns_the_receipts_own_boxes():
    assert _script().boxes_from(json.loads(json.dumps(GOOD))) == [(10.0, 5.0, 60.0, 45.0, 0.9)]


def test_zero_boxes_need_no_image_size():
    # SAM reports the image size inside each box record, so a count-0 answer has none (the pallet receipt).
    nothing = {**GOOD, "result": {**GOOD["result"], "count": 0, "boxes": [], "image_w": None, "image_h": None}}
    assert _script().boxes_from(nothing) == []


@pytest.mark.parametrize("change, why", [
    ({"count": 2}, "count"),
    ({"count": True}, "count"),  # a bool is not a count, even though True == 1
    ({"status": "ERROR"}, "status"),
    ({"image_w": None}, "image size"),  # boxes with no size: there is no frame to place them in
    ({"boxes": [[10.0, 5.0, 160.0, 45.0, 0.9]]}, "outside"),
    ({"boxes": [[60.0, 5.0, 10.0, 45.0, 0.9]]}, "empty"),
    ({"boxes": [["10", 5.0, 60.0, 45.0]]}, "not a number"),
    ({"boxes": [[10.0, 5.0, float("nan"), 45.0]]}, "not a number"),
    ({"boxes": [[10.0, 5.0, 60.0]]}, "four corners"),
])
def test_boxes_the_receipt_cannot_back_are_refused(change, why):
    m = _script()
    receipt = json.loads(json.dumps(GOOD))
    receipt["result"].update(change)
    with pytest.raises(m.Refused, match=why):
        m.boxes_from(receipt)
