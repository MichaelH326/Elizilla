#!/usr/bin/env python3
"""
Score conversation systems on the eval set (eval_set.py): today's rule-based
Elizilla and any trained neural model, side by side.

    python3 neural/eval_neural.py                                   # rule-based Elizilla (baseline)
    python3 neural/eval_neural.py --system elizilla --system elizilla-neural.mdlo --json report.json
    python3 neural/eval_neural.py --system model.gguf --show fail

For each behaviour (B2..B14) and split (known / held-out): % of items whose
scored reply meets its expectation AND every reply in the item passes the
hard checks (open question, no "I don't know" or advice, no repeats, no
answers to world questions). "hard" is the % of all replies passing those
checks. "adapt" is held-out score / known score: how well it carries over to
situations it never saw.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from checks import copy_score, problems, sentences  # noqa: E402
from eval_set import items  # noqa: E402

GENERIC = {"what else comes to mind?", "what's on your mind?", "how do you feel about that?",
           "what do you think?", "how are you feeling right now?"}
BEHAVIOURS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B14"]


class Elizilla:
    name = "elizilla (rules)"

    def __init__(self):
        from neo_eliza import NeoEliza
        self.cls = NeoEliza
        self.lock_dir = tempfile.mkdtemp(prefix="elizilla-eval-")

    def run(self, turns, seed):
        bot = self.cls(seed=seed, lock_path=os.path.join(self.lock_dir, f"{seed}.lock"))
        return [bot.respond(t) for t in turns]


class Neural:
    def __init__(self, path):
        from chat_model import ChatModel
        self.name = os.path.basename(path)
        self.lock_dir = tempfile.mkdtemp(prefix="elizilla-eval-")
        self.model = ChatModel(path, lock_path=os.path.join(self.lock_dir, "eval.lock"))

    def run(self, turns, seed):
        replies = []
        for i in range(len(turns)):
            replies.append(self.model.reply(turns[:i + 1], replies))
        return replies


def meets(expect, message, reply):
    low = reply.lower()
    if "copy" in expect and (copy_score(message, reply) or 0) < expect["copy"]:
        return False
    if expect.get("reaction") and len(sentences(reply)) < 2:
        return False
    if "any" in expect and not any(w in low for w in expect["any"]):
        return False
    if "both" in expect and not all(any(w in low for w in group) for group in expect["both"]):
        return False
    if expect.get("not_generic") and (low.strip() in GENERIC or len(reply.split()) < 5):
        return False
    return True


def score(system, show):
    per = defaultdict(lambda: [0, 0])
    hard_ok = hard_all = 0
    rows = []
    for n, it in enumerate(items()):
        replies = system.run(it["turns"], seed=n)
        hard = []
        for i, (m, r) in enumerate(zip(it["turns"], replies)):
            p = problems(m, r, replies[:i])
            hard.append(p)
            hard_ok += not p
            hard_all += 1
        k = it.get("check", len(it["turns"]) - 1)
        ok = meets(it["expect"], it["turns"][k], replies[k]) and not any(hard)
        key = (it["behaviour"], it["split"])
        per[key][0] += ok
        per[key][1] += 1
        rows.append({"id": it["id"], "ok": ok, "turns": it["turns"], "replies": replies,
                     "problems": [p for p in hard if p]})
        if show == "all" or (show == "fail" and not ok):
            print(f"  {'ok  ' if ok else 'MISS'} {it['id']}: {it['turns'][k]!r} -> {replies[k]!r}"
                  + (f"  {[p for p in hard if p]}" if any(hard) else ""))
    return per, hard_ok / max(hard_all, 1), rows


def pct(p):
    return f"{100 * p[0] / p[1]:.0f}%" if p[1] else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", action="append", default=[],
                    help='"elizilla" or a path to a .gguf / .mdlo model (repeatable)')
    ap.add_argument("--show", choices=["fail", "all", "none"], default="none")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    systems = [Elizilla() if s == "elizilla" else Neural(s) for s in (args.system or ["elizilla"])]

    report = {}
    for s in systems:
        print(f"== {s.name}")
        per, hard, rows = score(s, args.show)
        known = [sum(per[(b, "known")][i] for b in BEHAVIOURS) for i in (0, 1)]
        held = [sum(per[(b, "held-out")][i] for b in BEHAVIOURS) for i in (0, 1)]
        report[s.name] = {"hard_pct": round(100 * hard, 1), "rows": rows,
                          "by_behaviour": {f"{b}/{sp}": per[(b, sp)] for b in BEHAVIOURS
                                           for sp in ("known", "held-out") if per[(b, sp)][1]}}
        print(f"{'':6}{'known':>8}{'held-out':>10}")
        for b in BEHAVIOURS:
            if per[(b, "known")][1] or per[(b, "held-out")][1]:
                print(f"{b:<6}{pct(per[(b, 'known')]):>8}{pct(per[(b, 'held-out')]):>10}")
        adapt = (held[0] / held[1]) / (known[0] / known[1]) if known[0] and held[1] else 0
        print(f"{'all':<6}{pct(known):>8}{pct(held):>10}   hard checks {100 * hard:.0f}% of replies, "
              f"adapt {adapt:.2f}")
        report[s.name].update(known=known, held_out=held, adapt=round(adapt, 2))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
