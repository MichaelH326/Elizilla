"""
Finish an .mdlo made by Mira's package_mdlo.py for Elizilla: rename it,
declare the crisis lock, and record where it came from. The GGUF payload and
its checksum are untouched.

    python3 neural/tag_mdlo.py elizilla-neural.mdlo --card dataset_card.json --eval report.json
"""

import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdlo import PERSONA  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mdlo")
    p.add_argument("--card", default=None, help="dataset_card.json from make_dataset.py")
    p.add_argument("--eval", default=None, help="report from eval_neural.py --json")
    p.add_argument("--name", default="elizilla-neural")
    args = p.parse_args()

    with open(args.mdlo, "rb") as fh:
        head = fh.read(16)
        assert head[:4] == b"MDLO" and struct.unpack("<I", head[4:8])[0] == 2, "not an MDLO v2 file"
        header = json.loads(fh.read(struct.unpack("<Q", head[8:16])[0]))
        payload_start = fh.tell()

    header["model_name"] = args.name
    header["safety"] = {"crisis_lock": True, "guard": "safety.py"}
    header.setdefault("chat", {})
    if header["chat"].get("system_prompt") != PERSONA:
        raise SystemExit("system prompt in the file isn't Elizilla's persona from mdlo.py; "
                         "package with --system_prompt set to it")
    header["chat"]["style"] = "listener"
    prov = header.setdefault("provenance", {})
    if args.card:
        with open(args.card) as fh:
            card = json.load(fh)
        prov["dataset"] = {k: card.get(k) for k in ("conversations_out", "turns", "unique_reply_pct",
                                                    "turn_sources_pct", "teacher", "seed")}
    if args.eval:
        with open(args.eval) as fh:
            rep = json.load(fh)
        prov["eval"] = {name: {k: r.get(k) for k in ("known", "held_out", "adapt", "hard_pct")}
                        for name, r in rep.items()}

    hb = json.dumps(header, indent=1).encode()
    tmp = args.mdlo + ".tmp"
    with open(args.mdlo, "rb") as src, open(tmp, "wb") as out:
        out.write(b"MDLO" + struct.pack("<I", 2) + struct.pack("<Q", len(hb)) + hb)
        src.seek(payload_start)
        while chunk := src.read(1 << 20):
            out.write(chunk)
    os.replace(tmp, args.mdlo)
    print(f"tagged {args.mdlo}: model_name={args.name}, crisis_lock=True")


if __name__ == "__main__":
    main()
