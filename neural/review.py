"""
Human review: the part code can't check.

    python3 neural/review.py sheet clean.jsonl --n 200 --out review.tsv
        writes 200 random conversations, one row per Elizilla turn, stratified
        by area and by who wrote the turn (elizilla / teacher / mirror)
    (fill in the "rating" column: good / ok / bad, and a note for bad ones)
    python3 neural/review.py score review.tsv
        pass: <= 5% of conversations contain a bad turn, and teacher turns are
        rated no worse than Elizilla's own
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict


def sheet(path, n, out, seed=0):
    rng = random.Random(seed)
    with open(path) as fh:
        convs = [json.loads(l) for l in fh if l.strip()]
    by_area = defaultdict(list)
    for c in convs:
        by_area[c["area"]].append(c)
    picked = []
    while len(picked) < min(n, len(convs)):
        for group in by_area.values():
            if group and len(picked) < n:
                picked.append(group.pop(rng.randrange(len(group))))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["conversation", "turn", "source", "person", "elizilla", "rating", "note"])
        for c in picked:
            for i, t in enumerate(c["turns"]):
                w.writerow([c["id"], i, t["source"], t["user"], t["assistant"], "", ""])
    print(f"{len(picked)} conversations -> {out}. Fill in 'rating' (good / ok / bad) and 'note'.")


def score(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    rated = [r for r in rows if r["rating"].strip()]
    if not rated:
        sys.exit("no ratings yet")
    bad_convs = {r["conversation"] for r in rated if r["rating"].strip().lower() == "bad"}
    convs = {r["conversation"] for r in rated}
    by_source = defaultdict(Counter)
    for r in rated:
        by_source[r["source"]][r["rating"].strip().lower()] += 1
    bad_pct = 100 * len(bad_convs) / len(convs)
    print(f"{len(rated)} turns rated in {len(convs)} conversations; {bad_pct:.1f}% contain a bad turn")
    share = {}
    for src, c in by_source.items():
        total = sum(c.values())
        share[src] = 100 * c["bad"] / total
        print(f"  {src:<9} {total:>4} turns: good {c['good']}, ok {c['ok']}, bad {c['bad']} ({share[src]:.1f}% bad)")
    notes = Counter(r["note"].strip() for r in rated if r["rating"].strip().lower() == "bad" and r["note"].strip())
    if notes:
        print("most common notes on bad turns:")
        for note, k in notes.most_common(10):
            print(f"  {k}x {note}")
    ok = bad_pct <= 5 and share.get("teacher", 0) <= share.get("elizilla", 0) + 1
    print("PASS" if ok else "FAIL: turn each bad pattern into a check or an Elizilla fix, then rebuild")
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheet")
    s.add_argument("clean")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--out", default="review.tsv")
    sc = sub.add_parser("score")
    sc.add_argument("sheet")
    args = p.parse_args()
    if args.cmd == "sheet":
        sheet(args.clean, args.n, args.out)
    else:
        score(args.sheet)


if __name__ == "__main__":
    main()
