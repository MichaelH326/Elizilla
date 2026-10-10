"""
Situation profiles: one invented person in one situation, and the lines they
type over a short conversation. Written by the teacher model, or from
templates when there is no teacher.

    python3 neural/profiles.py --teacher fake --per-situation 2 --out profiles.jsonl
    python3 neural/profiles.py --teacher models/teacher.gguf --shard 3 --shards 20 --out p3.jsonl
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import safety  # noqa: E402
from situations import load as load_situations  # noqa: E402
import teacher as teachers  # noqa: E402

MOOD_LINES = {
    "anxious": ["i keep worrying about {about}", "what if it goes badly", "i cant stop thinking about {about}",
                "my stomach hurts just thinking about it"],
    "sad": ["it just really sucks", "i feel kind of down about {about}", "i didnt expect it to hit me this hard"],
    "angry": ["it makes me so mad", "its so unfair", "i honestly cant believe it"],
    "happy": ["im really happy about {about}", "it was such a good day", "i keep smiling about it"],
    "tired": ["im so tired", "i barely have energy for anything", "{about} is wearing me out"],
    "lonely": ["i feel kind of alone", "nobody really gets it", "i wish i had someone to talk to about {about}"],
    "bored": ["theres nothing to do", "everything feels the same", "i just sit around all day"],
    "proud": ["im actually really proud of myself", "i worked really hard for {about}", "i didnt think i could do it"],
    "guilty": ["i feel bad about it", "i should have done better", "i keep beating myself up about {about}"],
    "confused": ["i dont know what to do about {about}", "i keep going back and forth", "its all so confusing"],
    "excited": ["im so excited about {about}", "i cant wait", "its all i can think about"],
    "nervous": ["im kind of nervous about {about}", "what if i mess it up", "i keep picturing it going wrong"],
    "frustrated": ["its so annoying", "this keeps happening", "{about} is driving me crazy"],
    "calm": ["its actually kind of nice", "i feel pretty calm about {about}", "it helps me relax"],
    "mixed": ["i have mixed feelings about {about}", "part of me is happy but part of me isnt",
              "its good and bad at the same time"],
}
OTHER_LINES = ["my friend says im overthinking it", "my mom doesnt really get it",
               "i told my best friend about {about}", "nobody else knows yet", "it started last week",
               "its been like this for a while", "i guess thats whats going on", "anyway thats my day",
               "i didnt really expect it", "everyone keeps asking me about {about}"]
AI_TALK = re.compile(r"\b(?:as an ai|language model|chatbot|assistant|bot)\b", re.I)


def template_profile(situation, rng):
    fill = lambda s: s.format(about=situation["about"])
    lines = [situation["event"]] + [fill(l) for l in rng.sample(MOOD_LINES[situation["mood"]], 3)]
    lines += [fill(l) for l in rng.sample(OTHER_LINES, 3)]
    rest = lines[1:]
    rng.shuffle(rest)
    return {"facts": [situation["event"]], "lines": [lines[0]] + rest}


def screen(profile):
    """Drop single bad lines (too long, talk about AI/chatbots); reject the
    whole profile only for crisis content or too few lines left.
    Returns (profile or None, reason)."""
    if not profile:
        return None, "unparseable"
    for line in profile["lines"] + profile["facts"]:
        if safety.check(line):
            return None, "crisis"
    lines = [l for l in profile["lines"] if not AI_TALK.search(l) and 2 <= len(l.split()) <= 30]
    if len(lines) < 4:
        return None, "too-few-lines"
    return {**profile, "lines": lines}, "ok"


def build(teacher, situations, per_situation, rng, show_rejected=False):
    out, rejected = [], Counter()
    for sit in situations:
        for k in range(per_situation):
            prof, why = screen(teacher.profile(sit) if teacher else template_profile(sit, rng))
            if not prof:
                rejected[why] += 1
                if show_rejected:
                    print(f"--- rejected ({why}) {sit['id']}:\n{getattr(teacher, 'last_raw', '')[:1500]}\n---")
                continue
            out.append({"id": f"{sit['id']}#{k}", "situation": sit["id"], "area": sit["area"],
                        "mood": sit["mood"], "about": sit["about"], "held_out": sit["held_out"], **prof,
                        "source": teacher.name if teacher else "template"})
    return out, rejected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="none", help='"none" (templates), "fake", or a GGUF path')
    p.add_argument("--per-situation", type=int, default=2)
    p.add_argument("--held-out", action="store_true", help="only held-out situations (for the eval)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="only the first N situations (benchmarks)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show-rejected", action="store_true", help="print the teacher's raw text for rejects")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sits = [s for s in load_situations() if s["held_out"] == args.held_out]
    sits = sits[args.shard::args.shards]
    if args.limit:
        sits = sits[:args.limit]
    t = teachers.load(args.teacher, seed=args.seed)
    start = time.time()
    profs, rejected = build(t, sits, args.per_situation, random.Random(args.seed + args.shard),
                            args.show_rejected)
    with open(args.out, "w") as fh:
        for pr in profs:
            fh.write(json.dumps(pr) + "\n")
    tps = f", {t.tokens_per_second:.1f} tokens/s" if t and t.tokens_per_second else ""
    print(f"{len(profs)} profiles from {len(sits)} situations, rejected {dict(rejected)}, "
          f"{time.time() - start:.0f}s{tps} -> {args.out}")


if __name__ == "__main__":
    main()
