"""
Give each simulated conversation its Elizilla side.

The real NeoEliza engine replies turn by turn, so memory, names, one-word
answers, repeats and "why?" come out exactly as Elizilla does them. Where
its reply is weak (a generic fallback, or it ignores everything the person
said), the teacher writes candidates; the best one that passes every check is
used, otherwise a mirror reply. A conversation with any turn that can't be
fixed, or any crisis line, is dropped whole.

    python3 neural/build.py people.jsonl --teacher fake --out built.jsonl
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
import zlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import safety  # noqa: E402
from checks import content_words, copy_score, problems  # noqa: E402
from mirror import mirror  # noqa: E402
from neo_eliza import NeoEliza  # noqa: E402
from slm import SLM  # noqa: E402
import teacher as teachers  # noqa: E402

# Elizilla's own handling of these moments is exactly what the model must
# learn, so a passing reply is never replaced.
DIALOGUE_ACTS = {"name", "clarify", "why", "short-answers", "yes", "no", "repeated", "empty",
                 "memory", "reflect-question", "reflect-request", "greeting", "how-are-you",
                 "what-about-you", "your-mind", "thanks", "not-much"}


def is_weak(rule, message, reply):
    if rule in DIALOGUE_ACTS or not content_words(message):
        return False             # nothing to reflect: "hiya", "k"; Elizilla's own handling stays
    if rule == "fallback":
        return True
    # Ignores every word the person said: the teacher may do better.
    return copy_score(message, reply) == 0


def best_candidate(message, candidates, earlier, floor=0.0, avoid=()):
    """The candidate that passes every check and reuses the most of the
    person's words, if it reuses more than `floor` (what Elizilla managed).
    floor=None accepts any passing candidate (nothing to reuse: "hi")."""
    # Elizilla asks one short question at a time; so must the teacher.
    ok = [c for c in candidates if not problems(message, c, earlier) and c not in avoid
          and c.count("?") == 1 and len(c.split()) <= 30]
    if not ok:
        return None
    best = max(ok, key=lambda c: copy_score(message, c) or 0)
    if floor is None:
        return best
    return best if (copy_score(message, best) or 0) > floor else None


class Builder:
    def __init__(self, teacher=None, seed=0, max_uses=4):
        self.teacher = teacher
        # A reply used this many times already gets rewritten, so the model
        # doesn't learn a handful of stock questions by heart.
        self.max_uses = max_uses
        self.used = Counter()
        self.slm = SLM(seed=seed)
        self.lock_dir = tempfile.mkdtemp(prefix="elizilla-build-")
        self.stats = Counter()

    def build(self, conv):
        seed = zlib.crc32(conv["id"].encode())
        lock = os.path.join(self.lock_dir, f"{seed}.lock")
        bot = NeoEliza(seed=seed, use_slm=False, lock_path=lock)
        bot.writer = self.slm
        self.slm.rng.seed(seed)
        rng = random.Random(seed)
        turns, earlier, history = [], [], []
        for line in conv["user"]:
            if safety.check(line):
                self.stats["dropped:crisis"] += 1
                return None
            reply = bot.respond(line)
            rule = bot.last_rule
            source = "elizilla"
            bad = problems(line, reply, earlier)
            overused = self.max_uses and self.used[reply] >= self.max_uses and rule not in ("name",)
            if bad or overused or is_weak(rule, line, reply):
                fixed = None
                if self.teacher:
                    cands = self.teacher.replies(history + [(line, None)], n=3)
                    if overused and not bad:
                        floor = None if not content_words(line) else (copy_score(line, reply) or 0) - 0.01
                    else:
                        floor = 0.0 if bad else (copy_score(line, reply) or 0)
                    fixed = best_candidate(line, cands, earlier, floor=floor,
                                           avoid={r for r, k in self.used.items() if k >= self.max_uses})
                    source = "teacher" if fixed else source
                    if overused:
                        self.stats["overused:" + ("rewritten" if fixed else "kept")] += 1
                if not fixed:
                    # Broken, or ignores everything they said ("How are things
                    # with your bestie?"): a mirror of their own words is better.
                    m = mirror(line, rng)
                    if not problems(line, m, earlier) and (bad or (copy_score(line, m) or 0) > 0):
                        fixed, source = m, "mirror"
                if fixed:
                    reply = fixed
                    bot.last_bot = reply                 # keep the engine's state in step
                    bot.asked.add(reply)
                elif bad:
                    self.stats["dropped:" + bad[0]] += 1
                    return None
            self.used[reply] += 1
            self.stats["turns:" + source] += 1
            turns.append({"user": line, "assistant": reply, "source": source, "rule": rule})
            earlier.append(reply)
            history.append((line, reply))
        self.stats["kept"] += 1
        return {**{k: v for k, v in conv.items() if k != "user"}, "turns": turns}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("people")
    p.add_argument("--teacher", default="none", help='"none", "fake", or a GGUF path')
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-uses", type=int, default=4,
                   help="rewrite a reply once it has been used this often in this shard (0 = off)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.people) as fh:
        convs = [json.loads(l) for l in fh if l.strip()][args.shard::args.shards]
    t = teachers.load(args.teacher, seed=args.seed)
    b = Builder(t, seed=args.seed, max_uses=args.max_uses)
    start = time.time()
    with open(args.out, "w") as fh:
        for c in convs:
            built = b.build(c)
            if built:
                fh.write(json.dumps(built) + "\n")
    s = b.stats
    turns = sum(v for k, v in s.items() if k.startswith("turns:"))
    print(f"{s['kept']}/{len(convs)} conversations kept, {turns} turns "
          f"({', '.join(f'{k[6:]} {100 * v / max(turns, 1):.0f}%' for k, v in s.items() if k.startswith('turns:'))}), "
          f"{time.time() - start:.0f}s -> {args.out}")
    over = {k[9:]: v for k, v in s.items() if k.startswith("overused:")}
    if over:
        print("overused replies:", over)
    drops = {k[8:]: v for k, v in s.items() if k.startswith("dropped:")}
    if drops:
        print("dropped:", drops)


if __name__ == "__main__":
    main()
