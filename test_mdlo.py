"""Checks the .mdlo packing: Mira's container layout, checksum, and round trip."""
import json
import os
import struct
import tempfile

import mdlo
from neo_eliza import NeoEliza
from slm import SLM

with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "test.mdlo")
    header = mdlo.pack(path, SLM(seed=0))

    # Same outer layout as Mira's MDLO v2.
    with open(path, "rb") as fh:
        raw = fh.read()
    assert raw[:4] == b"MDLO"
    assert struct.unpack("<I", raw[4:8])[0] == 2
    hlen = struct.unpack("<Q", raw[8:16])[0]
    on_disk = json.loads(raw[16:16 + hlen])
    for key in ("format", "format_version", "model_name", "engine", "payload_sha256",
                "payload_bytes", "chat", "provenance"):
        assert key in on_disk, key
    assert on_disk["payload_bytes"] == len(raw) - 16 - hlen

    # A loaded model writes exactly what the trained one does.
    trained, (loaded, _) = SLM(seed=7), mdlo.load(path, seed=7)
    for intent, slots in [("sad", {"describer": "down"}), ("no-time", {"follow": "do your homework"}),
                          ("detail", {"d": "coding tasks"}), ("open", {})]:
        for _ in range(5):
            assert trained.write(intent, slots) == loaded.write(intent, slots), intent

    # The chat runs from the file.
    bot = NeoEliza(seed=1, model=path)
    assert bot.respond("i feel really lonely lately").endswith("?")

    # Corruption and other engines are refused.
    bad = os.path.join(td, "bad.mdlo")
    with open(bad, "wb") as fh:
        fh.write(raw[:-10] + bytes(10))
    try:
        mdlo.load(bad)
        raise AssertionError("corrupt file was accepted")
    except ValueError as e:
        assert "checksum" in str(e)
    gguf = os.path.join(td, "mira.mdlo")
    hb = json.dumps({**on_disk, "engine": "gguf"}).encode()
    with open(gguf, "wb") as fh:
        fh.write(b"MDLO" + struct.pack("<I", 2) + struct.pack("<Q", len(hb)) + hb + raw[16 + hlen:])
    try:
        mdlo.load(gguf)
        raise AssertionError("a GGUF model was accepted")
    except ValueError as e:
        assert "gguf" in str(e)

# The committed elizilla.mdlo must match corpus.txt and rules.py.
if os.path.exists(mdlo.DEFAULT_PATH):
    header, _ = mdlo.read(mdlo.DEFAULT_PATH)
    assert not mdlo.is_stale(header), "elizilla.mdlo is out of date: run `python3 mdlo.py pack`"

print("ok: .mdlo layout, checksum, engine check, and round trip")
