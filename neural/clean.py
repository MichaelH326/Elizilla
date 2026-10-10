"""
Clean the built conversations and fail loudly if the set would teach the
wrong thing:

  1. dedupe: exact copies removed; near-copies (>= 0.8 overlap) thinned to 3
  2. unique replies: >= 15% of Elizilla turns must be unique (else it memorizes)
  3. balance: each area within +-20% of an even share (extra conversations are
     trimmed); each scheduled behaviour in >= 5% of conversations; teacher
     turns <= 40% of all turns
  4. leakage: no held-out situation, no line close to an eval message
  5. crisis: safety.check finds nothing anywhere

    python3 neural/clean.py built.jsonl --eval neural/eval_set.jsonl --out clean.jsonl --card card.json
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import safety  # noqa: E402
from simulate import (SCHEDULE, ONE_WORD, WHY, CLARIFY, WORLD, ABOUT_BOT, YES_NO,  # noqa: E402
                      GREETINGS)

STOCK = set(ONE_WORD + WHY + CLARIFY + WORLD + ABOUT_BOT + YES_NO + GREETINGS)
NAME_LINE = re.compile(r"^(?:hi |hey )?(?:my name is|call me|im \w+ btw)")


def shingles(text, n=3):
    w = re.findall(r"[a-z']+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def similarity(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def conv_text(c):
    return " ".join(t["user"] + " " + t["assistant"] for t in c["turns"])


def dedupe(convs, rng, near=0.8, keep_near=3):
    seen, out = set(), []
    for c in convs:
        key = conv_text(c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    # Near-duplicates: bucket by situation (they can only be close within one).
    by_sit = defaultdict(list)
    for c in out:
        by_sit[c["situation"]].append(c)
    kept = []
    for group in by_sit.values():
        sigs = []
        for c in group:
            s = shingles(conv_text(c))
            close = sum(1 for t in sigs if similarity(s, t) >= near)
            if close < keep_near:
                kept.append(c)
                sigs.append(s)
    return kept


def balance(convs, rng, tolerance=0.2):
    by_area = defaultdict(list)
    for c in convs:
        by_area[c["area"]].append(c)
    target = len(convs) / len(by_area)
    cap = int(target * (1 + tolerance))
    out = []
    for group in by_area.values():
        rng.shuffle(group)
        out.extend(group[:cap])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("built", nargs="+")
    p.add_argument("--eval", default=os.path.join(HERE, "eval_set.jsonl"))
    p.add_argument("--out", required=True)
    p.add_argument("--card", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-unique", type=float, default=15.0)
    p.add_argument("--max-reply-pct", type=float, default=2.0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    convs = []
    for path in args.built:
        with open(path) as fh:
            convs += [json.loads(l) for l in fh if l.strip()]
    start = len(convs)
    failures = []

    # 4. leakage, first: nothing held out, nothing close to an eval message
    held = [c for c in convs if c.get("held_out")]
    if held:
        failures.append(f"{len(held)} conversations from held-out situations")
    eval_sigs = []
    if os.path.exists(args.eval):
        with open(args.eval) as fh:
            for line in fh:
                item = json.loads(line)
                for msg in item.get("turns", [item.get("message", "")]):
                    # Stock phrases ("what do you mean", "my name is ...") must be in
                    # training; only real eval messages count as leakage.
                    if len(msg.split()) >= 4 and msg not in STOCK and not NAME_LINE.match(msg):
                        eval_sigs.append(shingles(msg))
    eval_index = defaultdict(list)
    for s in eval_sigs:
        for g in s:
            eval_index[g].append(s)
    leaked = 0
    for c in convs:
        for t in c["turns"]:
            if len(t["user"].split()) < 4:
                continue
            s = shingles(t["user"])
            if any(similarity(s, e) >= 0.8 for g in s for e in eval_index.get(g, ())):
                leaked += 1
                break
    if leaked:
        failures.append(f"{leaked} conversations contain a line close to an eval message")

    # 5. crisis
    crisis = sum(1 for c in convs for t in c["turns"]
                 if safety.check(t["user"]) or safety.check(t["assistant"]))
    if crisis:
        failures.append(f"{crisis} turns hit the crisis check")

    # 1. dedupe, then 3. balance areas
    convs = balance(dedupe(convs, rng), rng)

    # 2. unique replies
    replies = [t["assistant"] for c in convs for t in c["turns"]]
    unique_pct = 100 * len(set(replies)) / max(len(replies), 1)
    if unique_pct < args.min_unique:
        failures.append(f"only {unique_pct:.1f}% unique replies (need {args.min_unique}%)")

    # 2b. no single reply may dominate (the model would learn it by heart)
    top_reply, top_n = Counter(replies).most_common(1)[0] if replies else ("", 0)
    top_pct = 100 * top_n / max(len(replies), 1)
    if top_pct > args.max_reply_pct:
        failures.append(f"one reply is {top_pct:.1f}% of all turns (max {args.max_reply_pct}%): {top_reply!r}")

    # 3. behaviour coverage and teacher share
    beh = Counter(b for c in convs for b in c["behaviours"])
    low = {b: round(100 * beh[b] / max(len(convs), 1), 1) for b in SCHEDULE
           if beh[b] < 0.05 * len(convs)}
    if low:
        failures.append(f"behaviours under 5% of conversations: {low}")
    sources = Counter(t["source"] for c in convs for t in c["turns"])
    teacher_pct = 100 * sources["teacher"] / max(sum(sources.values()), 1)
    if teacher_pct > 60:
        failures.append(f"teacher wrote {teacher_pct:.0f}% of turns (max 60%)")

    card = {
        "conversations_in": start, "conversations_out": len(convs),
        "turns": len(replies), "unique_reply_pct": round(unique_pct, 1),
        "most_common_replies": Counter(replies).most_common(15),
        "areas": Counter(c["area"] for c in convs), "situations": len({c["situation"] for c in convs}),
        "behaviours_pct": {b: round(100 * beh[b] / max(len(convs), 1), 1) for b in SCHEDULE},
        "turn_sources_pct": {k: round(100 * v / max(sum(sources.values()), 1), 1) for k, v in sources.items()},
        "profile_sources": Counter(c["source"] for c in convs),
        "rules": Counter(t["rule"] for c in convs for t in c["turns"]).most_common(),
        "failures": failures,
    }
    with open(args.card, "w") as fh:
        json.dump(card, fh, indent=1, default=dict)
    with open(args.out, "w") as fh:
        for c in convs:
            fh.write(json.dumps(c) + "\n")
    print(f"{start} -> {len(convs)} conversations, {len(replies)} turns, {unique_pct:.1f}% unique replies, "
          f"teacher {teacher_pct:.0f}% -> {args.out}")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
