import json
import os
import stat

from nacl.signing import SigningKey

from tinytms.receipt import (
    PRIVATE_KEY_FILE,
    PUBLIC_KEY_FILE,
    canonical_json,
    load_or_create_key,
    public_key_hex,
    sign_receipt,
    verify_receipt,
)

BODY = {"kind": "checkin", "utc": "2026-10-28T15:00:00.000000Z", "verdict": {"decision": "ACCEPT", "reasons": []},
        "inputs": {"load": {"sha256": "00" * 32}}, "note": "año"}


def test_canonical_json_sorted_compact():
    assert canonical_json({"b": 1, "a": [1, 2], "ñ": "é"}) == '{"a":[1,2],"b":1,"ñ":"é"}'.encode()


def test_sign_and_verify():
    sk = SigningKey.generate()
    r = sign_receipt(BODY, sk)
    assert r["public_key"] == public_key_hex(sk)
    assert len(bytes.fromhex(r["signature"])) == 64
    assert verify_receipt(r)
    assert verify_receipt(json.dumps(r))
    assert verify_receipt(r, trusted_public_key=public_key_hex(sk))


def test_flipping_any_byte_breaks_verification():
    r = sign_receipt(BODY, SigningKey.generate())
    raw = canonical_json(r)
    assert verify_receipt(raw)
    for i in range(len(raw)):
        b = bytearray(raw)
        b[i] ^= 0x01
        assert not verify_receipt(bytes(b)), f"byte {i} flipped but receipt still verified"


def test_tampered_field_fails():
    r = sign_receipt(BODY, SigningKey.generate())
    r["verdict"] = {"decision": "ACCEPT", "reasons": ["x"]}
    assert not verify_receipt(r)


def test_untrusted_key_rejected():
    r = sign_receipt(BODY, SigningKey.generate())
    other = public_key_hex(SigningKey.generate())
    assert not verify_receipt(r, trusted_public_key=other)


def test_swapped_key_rejected():
    r = sign_receipt(BODY, SigningKey.generate())
    r["public_key"] = public_key_hex(SigningKey.generate())
    assert not verify_receipt(r)


def test_garbage_is_false_not_exception():
    assert not verify_receipt(b"not json")
    assert not verify_receipt({"signature": "zz", "public_key": "yy"})
    assert not verify_receipt([])  # type: ignore[arg-type]


def test_key_files(tmp_path, capsys):
    sk1 = load_or_create_key(tmp_path / "keys")
    sk2 = load_or_create_key(tmp_path / "keys")
    assert bytes(sk1) == bytes(sk2)
    priv = tmp_path / "keys" / PRIVATE_KEY_FILE
    assert stat.S_IMODE(os.stat(priv).st_mode) == 0o600
    assert (tmp_path / "keys" / PUBLIC_KEY_FILE).read_text().strip() == public_key_hex(sk1)
    out = capsys.readouterr()
    assert bytes(sk1).hex() not in out.out + out.err  # private key never printed
