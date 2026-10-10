"""
Print statistics and random samples from pipeline files, for checking data
by eye (profiles.jsonl, built-*.jsonl, clean.jsonl).

    python3 neural/inspect_data.py parts/*.jsonl --samples 40
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from checks import copy_score, problems  # noqa: E402


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def show_profiles(rows, n, rng):
    print(f"== {len(rows)} profiles, {len({r['situation'] for r in rows})} situations")
    lens = [len(r["lines"]) for r in rows]
    words = [len(l.split()) for r in rows for l in r["lines"]]
    print(f"lines per profile: min {min(lens)}, avg {sum(lens) / len(lens):.1f}, max {max(lens)}")
    print(f"words per line: avg {sum(words) / len(words):.1f}, max {max(words)}")
    pats = Counter()
    for r in rows:
        for l in r["lines"]:
            low = l.lower()
            if re.search(r"\b(?:you|your)\b", low):
                pats["talks to the listener (you/your)"] += 1
            if re.search(r"\b(?:help me|advice|tips|should i|can you|could you)\b", low):
                pats["asks for help/advice"] += 1
            if l[:1].isupper():
                pats["starts with a capital"] += 1
            if "!" in l:
                pats["has !"] += 1
    total = sum(lens)
    print("patterns: " + ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in pats.most_common()))
    dup = Counter(l.lower() for r in rows for l in r["lines"])
    print("most repeated lines: " + "; ".join(f"{k!r} x{v}" for k, v in dup.most_common(8)))
    for r in rng.sample(rows, min(n, len(rows))):
        print(f"\n[{r['situation']}] ({r['mood']})")
        for l in r["lines"]:
            print(f"   {l}")


def show_convs(rows, n, rng):
    turns = [t for c in rows for t in c["turns"]]
    print(f"== {len(rows)} conversations, {len(turns)} turns, {len(turns) / max(len(rows), 1):.1f} turns each")
    src = Counter(t["source"] for t in turns)
    print("turn sources: " + ", ".join(f"{k} {100 * v / len(turns):.0f}%" for k, v in src.most_common()))
    print("rules: " + ", ".join(f"{k} {v}" for k, v in Counter(t["rule"] for t in turns).most_common(25)))
    for s in src:
        rs = [t["assistant"] for t in turns if t["source"] == s]
        cs = [copy_score(t["user"], t["assistant"]) for t in turns if t["source"] == s]
        cs = [c for c in cs if c is not None]
        print(f"  {s:<9} replies: {len(rs)}, unique {100 * len(set(rs)) / len(rs):.0f}%, avg words "
              f"{sum(len(r.split()) for r in rs) / len(rs):.1f}, reuse of their words {sum(cs) / max(len(cs), 1):.2f}")
    # Re-run the hard checks on everything as a second line of defence.
    fails = Counter()
    for c in rows:
        earlier = []
        for t in c["turns"]:
            for p in problems(t["user"], t["assistant"], earlier):
                fails[(t["source"], p)] += 1
            earlier.append(t["assistant"])
    print("hard-check failures in the file: " + (", ".join(f"{k} {v}" for k, v in fails.most_common()) or "none"))
    starts = Counter(" ".join(t["assistant"].split()[:3]) for t in turns)
    print("most common reply openings: " + "; ".join(f"{k!r} {100 * v / len(turns):.1f}%" for k, v in starts.most_common(12)))
    rep = Counter(t["assistant"] for t in turns)
    print("most repeated replies: " + "; ".join(f"{k!r} x{v}" for k, v in rep.most_common(10)))
    for s in src:
        pool = [t for t in turns if t["source"] == s]
        print(f"\n-- {min(n, len(pool))} random {s} turns")
        for t in rng.sample(pool, min(n, len(pool))):
            print(f"  P: {t['user']}\n  E: {t['assistant']}   [{t['rule']}]")
    print("\n-- 3 whole conversations")
    for c in rng.sample(rows, min(3, len(rows))):
        print(f"\n[{c['situation']}] behaviours: {c['behaviours']}")
        for t in c["turns"]:
            print(f"  P: {t['user']}\n  E: {t['assistant']}   [{t['source']} {t['rule']}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rows = load(args.files)
    rng = random.Random(args.seed)
    if rows and "turns" in rows[0]:
        show_convs(rows, args.samples, rng)
    else:
        show_profiles(rows, args.samples, rng)


if __name__ == "__main__":
    main()
