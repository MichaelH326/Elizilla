#!/usr/bin/env python3
"""
Pack Elizilla's small language model into a single `.mdlo` file, the same
container Mira uses (see MDLO_SPEC.md in the Mira repo), and load it back.

Layout (identical to Mira's MDLO v2):

    offset 0    4 bytes   magic "MDLO"
    offset 4    4 bytes   version, uint32 little-endian (2)
    offset 8    8 bytes   header length, uint64 little-endian
    offset 16   header    UTF-8 JSON
    then        payload   the model; header["payload_sha256"] is its SHA-256

Mira's v2 payload is a GGUF file (header "engine": "gguf"). Elizilla's is the
trained n-gram model as gzipped JSON (header "engine": "ngram-kn"), so it
needs no installs to read. A loader picks the engine from the header.

    python3 mdlo.py pack                   # train from corpus.txt + rules.py -> elizilla.mdlo
    python3 mdlo.py info elizilla.mdlo     # print the header
    python3 neo_eliza.py --model elizilla.mdlo
"""

import gzip
import hashlib
import json
import os
import struct
import sys
from datetime import datetime, timezone

from slm import SLM, HERE

MDLO_MAGIC = b"MDLO"
MDLO_VERSION = 2
ENGINE = "ngram-kn"
DEFAULT_PATH = os.path.join(HERE, "elizilla.mdlo")

# Elizilla's counterpart to Mira's system prompt: how it talks. Unlike Mira,
# this isn't fed to the model; it records the behaviour the rules and the
# corpus were written for.
PERSONA = (
    "You are NeoEliza, a warm listener. Reply in one or two short, casual spoken "
    "sentences: at most a short reaction, then one open-ended question about what "
    "the person just said, using their own words. Never give advice or do tasks. "
    "Never guess: if you can't know something (time, weather, news), say so; if "
    "you don't know a fact, say so."
)


def training_fingerprint():
    """SHA-256 of what the model is trained from, so a loader can tell when
    the .mdlo is older than corpus.txt / rules.py."""
    h = hashlib.sha256()
    for name in ("corpus.txt", "rules.py"):
        with open(os.path.join(HERE, name), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def pack(path=DEFAULT_PATH, slm=None):
    slm = slm or SLM()
    payload = gzip.compress(json.dumps(slm.state(), separators=(",", ":")).encode("utf-8"), mtime=0)
    header = {
        "format": "mdlo",
        "format_version": MDLO_VERSION,
        "model_name": "elizilla",
        "engine": ENGINE,
        "payload_encoding": "gzip+json",
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chat": {
            "system_prompt": PERSONA,
            "style": "listener",
            "greeting": "Hello, I'm NeoEliza. How are you feeling today?",
            "temperature": slm.temperature,
            "candidates_per_reply": slm.samples,
            "min_confidence": slm.min_confidence,
            "min_word_confidence": slm.min_word_confidence,
        },
        "provenance": {
            "base_model": None,                      # trained from scratch
            "order": slm.order,
            "training_sentences": len(slm.seen_sentences),
            "intents": len(slm.intent_slots),
            "vocab_size": len(slm.global_model.vocab),
            "trained_on": ["corpus.txt", "rules.py"],
            "training_sha256": training_fingerprint(),
        },
    }
    hb = json.dumps(header, indent=1).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(MDLO_MAGIC)
        fh.write(struct.pack("<I", MDLO_VERSION))
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(payload)
    return header


def read(path):
    """Return (header, payload bytes), checking magic, version and checksum."""
    with open(path, "rb") as fh:
        head = fh.read(16)
        if head[:4] != MDLO_MAGIC:
            raise ValueError(f"{path} is not an MDLO file")
        version = struct.unpack("<I", head[4:8])[0]
        if version != MDLO_VERSION:
            raise ValueError(f"unsupported MDLO version {version}")
        header_len = struct.unpack("<Q", head[8:16])[0]
        header = json.loads(fh.read(header_len).decode("utf-8"))
        payload = fh.read()
    if hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
        raise ValueError(f"{path}: payload checksum mismatch, the file is corrupt")
    return header, payload


def load(path=DEFAULT_PATH, seed=None):
    """Load an Elizilla .mdlo into a ready-to-use SLM. Returns (slm, header)."""
    header, payload = read(path)
    if header.get("engine") != ENGINE:
        raise ValueError(f"{path} holds a {header.get('engine')!r} model "
                         f"(a Mira GGUF model runs with Mira's run_mira.py)")
    state = json.loads(gzip.decompress(payload).decode("utf-8"))
    return SLM(seed=seed, state=state), header


def is_stale(header):
    """True if corpus.txt or rules.py changed since the file was packed."""
    try:
        return header["provenance"]["training_sha256"] != training_fingerprint()
    except (KeyError, OSError):
        return False


def main():
    args = sys.argv[1:]
    if not args or args[0] == "pack":
        path = args[1] if len(args) > 1 else DEFAULT_PATH
        header = pack(path)
        size = os.path.getsize(path)
        print(f"wrote {path} ({size / 1024:.0f} KB, {header['provenance']['training_sentences']} "
              f"training sentences, {header['provenance']['intents']} intents)")
    elif args[0] == "info":
        header, _ = read(args[1] if len(args) > 1 else DEFAULT_PATH)
        print(json.dumps(header, indent=2))
        if is_stale(header):
            print("note: corpus.txt or rules.py changed since this was packed; run `python3 mdlo.py pack`")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
