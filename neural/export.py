"""
Write the cleaned conversations in Mira's training format, with a
validation split by profile (the same invented person never appears in both).

    {"messages": [{"role": "system", "content": PERSONA},
                  {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

The system prompt is the persona in mdlo.py, word for word, so training and
packaging use exactly the same text.
"""

import argparse
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mdlo import PERSONA  # noqa: E402


def to_messages(conv):
    msgs = [{"role": "system", "content": PERSONA}]
    for t in conv["turns"]:
        msgs.append({"role": "user", "content": t["user"]})
        msgs.append({"role": "assistant", "content": t["assistant"]})
    return {"messages": msgs}


def is_val(conv, share=0.05):
    return zlib.crc32(conv["profile"].encode()) % 1000 < share * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("clean")
    p.add_argument("--out", required=True)
    p.add_argument("--val-out", required=True)
    args = p.parse_args()
    n = v = 0
    with open(args.clean) as src, open(args.out, "w") as tr, open(args.val_out, "w") as va:
        for line in src:
            c = json.loads(line)
            if is_val(c):
                va.write(json.dumps(to_messages(c)) + "\n")
                v += 1
            else:
                tr.write(json.dumps(to_messages(c)) + "\n")
                n += 1
    print(f"{n} training + {v} validation conversations -> {args.out}, {args.val_out}")


if __name__ == "__main__":
    main()
