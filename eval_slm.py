#!/usr/bin/env python3
"""
Measure the SLM writer, intent by intent.

For each intent it generates N replies with sample slot values and reports:
  written   % of tries where a reply passed every gate (otherwise the chat
            falls back to a hand-written template)
  new       % of written replies whose wording isn't in the training data
  copies    % of written replies that are exact training sentences
  unique    distinct replies / written replies
It also prints why candidates were rejected, and writes a sheet for rating
replies by hand (eval_slm_sheet.tsv: intent, reply, new?, rating).

    python3 eval_slm.py            # 50 replies per intent
    python3 eval_slm.py 20 sad     # 20 replies, only the "sad" intent
"""

import re
import sys
from collections import Counter

from slm import SLM

SAMPLE_SLOTS = {
    "describer": "lonely", "follow": "finish the project", "who": "your mom",
    "whom": "them", "action": "like", "acting": "playing", "what": "music",
    "rest": "a student", "says": "is always on your case", "topic": "mom",
    "n1": "school", "n2": "work", "before": "it was fun", "clause": "you are tired",
    "when": "yesterday", "d": "coding tasks", "x": "day", "prev": "I'm tired",
}


# What {topic} / {follow} / {action} look like for each kind of rule.
INTENT_SLOTS = {
    "work": {"topic": "job"}, "work-low": {"topic": "job"}, "work-good": {"topic": "job"},
    "task": {"topic": "homework"}, "activity": {"topic": "guitar"}, "friend": {"topic": "best friend"},
    "boss": {"topic": "boss"}, "pet": {"topic": "dog"}, "family": {"topic": "dad"},
    "anxious": {"describer": "stressed", "follow": "your exams"},
    "sad": {"describer": "down", "follow": "the move"}, "happy": {"describer": "happy", "follow": "the trip"},
    "angry": {"describer": "mad"}, "busy-thing": {"describer": "crazy"}, "busy-me": {"describer": "busy"},
    "good-thing": {"describer": "great", "who": "it", "whom": "it"},
    "bad-thing": {"describer": "awful", "who": "it", "whom": "it"},
    "want-to": {"action": "want", "follow": "quit school"}, "want": {"action": "want", "what": "a new job"},
    "like-doing": {"action": "love", "follow": "playing guitar"}, "like": {"action": "love", "what": "music"},
    "like-that": {"follow": "people listen to you"}, "hate-when": {"follow": "no one shows up"},
    "hate": {"what": "mornings"}, "belief": {"follow": "your boss hates you"},
    "cannot": {"action": "sleep"}, "worry": {"action": "worry"}, "feel-like": {"follow": "nobody cares"},
    "feel-not": {"describer": "happy"}, "ordinary": {"describer": "normal"},
    "everyone": {"who": "everyone"}, "loss": {"whom": "your grandpa"},
}


def slots_for(slm, intent):
    """Give the writer the slots that this intent's own replies use."""
    used = set(slm.intent_slots.get(intent, ()))
    sample = {**SAMPLE_SLOTS, **INTENT_SLOTS.get(intent, {})}
    return {k: sample[k] for k in used if k in sample}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    only = sys.argv[2:]
    slm = SLM(seed=1)
    intents = sorted(i for i in slm.intent_models
                     if (not only or i in only) and not i.startswith("ack-"))
    rows, totals, reasons = [], Counter(), Counter()
    print(f"{'intent':<16}{'written':>8}{'new':>7}{'copies':>8}{'unique':>8}")
    for intent in intents:
        slots = slots_for(slm, intent)
        out = []
        for _ in range(n):
            reply, info = slm.write(intent, slots)
            reasons.update(info.get("rejected", {}))
            if reply:
                out.append((reply, info["new_sentence"]))
        new = sum(1 for _, nw in out if nw)
        written = len(out)
        totals.update(tries=n, written=written, new=new, unique=len({r for r, _ in out}))
        pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"
        print(f"{intent:<16}{pct(written, n):>8}{pct(new, written):>7}"
              f"{pct(written - new, written):>8}{pct(len({r for r, _ in out}), written):>8}")
        for reply, nw in sorted(set(out)):
            rows.append(f"{intent}\t{reply}\t{'new' if nw else 'copy'}\t")
    t = totals
    print(f"\n{'all':<16}{100 * t['written'] / t['tries']:>7.0f}%"
          f"{100 * t['new'] / max(t['written'], 1):>6.0f}%"
          f"{100 * (t['written'] - t['new']) / max(t['written'], 1):>7.0f}%"
          f"{100 * t['unique'] / max(t['written'], 1):>7.0f}%")
    print("\nwhy candidates were rejected: " +
          ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    with open("eval_slm_sheet.tsv", "w") as fh:
        fh.write("intent\treply\tnew?\trating (good / ok / bad)\n")
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} distinct replies to eval_slm_sheet.tsv for rating by hand")


if __name__ == "__main__":
    main()
