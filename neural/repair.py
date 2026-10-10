"""
Repair built conversations without calling the teacher again.

Every Elizilla-side turn is checked once more, with the current checks:
  - fails a hard check (checks.py)
  - generic ("What else comes to mind?") although the person said something real
  - a shaky Elizilla rule that ignores every word the person said
  - a teacher reply that coaches, runs long, or brings in details nobody said
A failing turn is replaced by a mirror of the person's own words when that
passes every check; otherwise the conversation is cut just before it (kept if
at least 3 turns remain). Finally no single reply may exceed --max-reply-pct
of all turns: conversations are cut at the over-used turn until it doesn't.

    python3 neural/repair.py built-*.jsonl --out repaired.jsonl
"""

import argparse
import json
import os
import random
import sys
import zlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from build import COACHING, DIALOGUE_ACTS, SHAKY_RULES, invented  # noqa: E402
from checks import content_words, copy_score, problems  # noqa: E402
from mirror import mirror  # noqa: E402

GENERIC = {"What else comes to mind?", "How did that feel?", "How are you feeling about it now?",
           "What happened next?", "What do you think is behind it?", "How are you feeling right now?",
           "What's on your mind?", "What do you make of that?", "How do you feel about that?",
           "What's that like for you?", "How did that come about?"}
MIN_TURNS = 3


def reason(turn, history, earlier):
    u, r, rule, src = turn["user"], turn["assistant"], turn["rule"], turn["source"]
    p = problems(u, r, earlier)
    if p:
        return p[0]
    has_words = bool(content_words(u))
    if src != "teacher" and r in GENERIC and len(content_words(u)) >= 2 and rule not in DIALOGUE_ACTS:
        return "generic"
    if src == "elizilla" and rule in SHAKY_RULES and has_words and not copy_score(u, r):
        return "ignores-them"
    if src == "teacher" and (COACHING.search(r) or len(r.split()) > 18 or len(invented(history, r)) > 1):
        return "teacher-style"
    return None


def repair(conv, stats, rng):
    turns, earlier, history = [], [], []
    for t in conv["turns"]:
        why = reason(t, history + [(t["user"], None)], earlier)
        if why:
            m = mirror(t["user"], rng)
            if not problems(t["user"], m, earlier) and (copy_score(t["user"], m) or 0) > 0:
                t = {**t, "assistant": m, "source": "mirror", "repaired": why}
                stats["mirrored:" + why] += 1
            else:
                stats["cut:" + why] += 1
                break
        turns.append(t)
        earlier.append(t["assistant"])
        history.append((t["user"], t["assistant"]))
    return {**conv, "turns": turns} if len(turns) >= MIN_TURNS else None


def cap(convs, max_pct, stats):
    """Cut conversations at over-used replies until none is above max_pct."""
    while True:
        counts = Counter(t["assistant"] for c in convs for t in c["turns"])
        total = sum(counts.values())
        reply, n = counts.most_common(1)[0]
        limit = int(max_pct / 100 * total)
        if n <= limit:
            return convs
        excess = n - limit
        out = []
        for c in convs:
            idx = next((i for i, t in enumerate(c["turns"]) if t["assistant"] == reply), None)
            if excess > 0 and idx is not None:
                excess -= sum(1 for t in c["turns"] if t["assistant"] == reply)
                stats["capped"] += 1
                if idx >= MIN_TURNS:
                    out.append({**c, "turns": c["turns"][:idx]})
                continue
            out.append(c)
        convs = out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("built", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--max-reply-pct", type=float, default=1.5)
    args = p.parse_args()
    convs = []
    for path in args.built:
        with open(path) as fh:
            convs += [json.loads(l) for l in fh if l.strip()]
    stats = Counter()
    before = sum(len(c["turns"]) for c in convs)
    fixed = []
    for c in convs:
        r = repair(c, stats, random.Random(zlib.crc32(c["id"].encode())))
        if r:
            fixed.append(r)
        else:
            stats["dropped-too-short"] += 1
    fixed = cap(fixed, args.max_reply_pct, stats)
    after = sum(len(c["turns"]) for c in fixed)
    with open(args.out, "w") as fh:
        for c in fixed:
            fh.write(json.dumps(c) + "\n")
    print(f"{len(convs)} -> {len(fixed)} conversations, {before} -> {after} turns -> {args.out}")
    print("repairs: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))


if __name__ == "__main__":
    main()
