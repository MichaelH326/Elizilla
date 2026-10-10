"""
Simulated people: turn each situation profile into several conversations,
adding the moments that make Elizilla's behaviours happen (a name, one-word
answers, a repeated line, "why?", a question about the world, a request, a
topic change, coming back to something earlier) and casual texting noise.

    python3 neural/simulate.py profiles.jsonl --per-profile 4 --out people.jsonl
"""

import argparse
import math
import json
import random
import re

NAMES = """sam alex jordan priya maya leo noah emma liam ava zoe kai mateo sofia lucas mia omar
aisha chen wei hana yuki ravi arjun nina ella jack ruby theo isla finn luna diego elena
marcus jada tyler kayla dev nora owen lily ezra ines amir sara ben chloe max rosa""".split()
GREETINGS = ["hi", "hey", "hello", "hey there", "hiya", "yo", "hi there"]
ONE_WORD = ["idk", "ok", "nah", "maybe", "meh", "sure", "dunno", "eh", "fine", "k"]
WHY = ["why?", "why do you ask", "why"]
CLARIFY = ["what do you mean", "huh", "what do you mean?", "i dont understand"]
WORLD = ["what time is it?", "whats the weather like tomorrow", "what is the capital of japan?",
         "how far away is the moon?", "who invented the telephone?", "what is the biggest ocean?",
         "how tall is the eiffel tower?", "whats the news today", "who is the president of brazil?",
         "what is the population of canada?"]
REQUEST = ["can you help me with {about}", "can you write my essay for me", "could you help me with my homework",
           "can you look up a recipe for me", "can you calculate my budget", "please write a poem for me",
           "can you fix my resume"]
ABOUT_BOT = ["are you a robot", "what about you", "how are you", "are you real", "what do you think"]
CALLBACK = ["i keep thinking about {about}", "but yeah {about} is still on my mind",
            "still thinking about {about} honestly"]
YES_NO = ["yeah", "yes", "no", "nope", "yep"]
SLANG = ["tbh", "ngl", "lol", "honestly", "like", "i mean", "fr"]

# How often each behaviour is placed (share of conversations). Every one of
# B3-B15 needs at least 5% of conversations, so nothing is left to chance.
SCHEDULE = {"greet": 0.35, "name": 0.20, "short": 0.12, "one-word": 0.15, "repeat": 0.08,
            "why": 0.08, "clarify": 0.08, "world": 0.12, "request": 0.08, "about-bot": 0.08,
            "topic-change": 0.12, "callback": 0.15, "yes-no": 0.10}


def typo(word, rng):
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.random()
    if kind < 0.4:
        return word[:i] + word[i + 1:]                     # dropped letter
    if kind < 0.8:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]   # swapped letters
    return word[:i] + word[i] + word[i:]                   # doubled letter


def voice(line, rng, style):
    """Texting noise, with a per-person style so a conversation stays consistent."""
    words = line.split()
    if style["typos"]:
        words = [typo(w, rng) if rng.random() < 0.06 else w for w in words]
    text = " ".join(words)
    if style["no_apostrophes"]:
        text = text.replace("'", "")
    if style["slang"] and rng.random() < 0.25:
        s = rng.choice(SLANG)
        text = f"{s} {text}" if rng.random() < 0.5 else f"{text} {s}"
    if style["caps"]:
        text = text[:1].upper() + text[1:]
    if style["punct"] and not text.endswith(("?", "!", ".")) and rng.random() < 0.5:
        text += rng.choice([".", "!", "..."])
    return text


def conversation(profile, others, rng, behaviours):
    style = {"typos": rng.random() < 0.4, "no_apostrophes": rng.random() < 0.6,
             "slang": rng.random() < 0.4, "caps": rng.random() < 0.25, "punct": rng.random() < 0.3}
    about = profile.get("about") or "that"
    lines = profile["lines"][:]
    keep = rng.randint(min(4, len(lines)), min(10, len(lines)))
    body = [lines[0]] + sorted(rng.sample(lines[1:], keep - 1), key=lines.index)

    def at(pos_lo=1):
        return rng.randint(min(pos_lo, len(body)), len(body))

    for b in behaviours:
        if b == "name":
            body.insert(rng.randint(0, min(2, len(body))), rng.choice(["my name is {n}", "call me {n}"])
                        .format(n=rng.choice(NAMES)))
        elif b == "short":
            i = at(2)
            body[i:i] = rng.sample(ONE_WORD, 2)
        elif b == "one-word":
            body.insert(at(2), rng.choice(ONE_WORD))
        elif b == "repeat":
            earlier = [l for l in body if len(l.split()) > 3]
            if earlier:
                body.insert(at(3), rng.choice(earlier))
        elif b == "why":
            body.insert(at(2), rng.choice(WHY))
        elif b == "clarify":
            body.insert(at(2), rng.choice(CLARIFY))
        elif b == "world":
            body.insert(at(2), rng.choice(WORLD))
        elif b == "request":
            body.insert(at(2), rng.choice(REQUEST).format(about=about))
        elif b == "about-bot":
            body.insert(at(2), rng.choice(ABOUT_BOT))
        elif b == "topic-change" and others:
            body.insert(at(3), rng.choice(others)["lines"][0])
        elif b == "callback":
            body.append(rng.choice(CALLBACK).format(about=about))
        elif b == "yes-no":
            body.insert(at(2), rng.choice(YES_NO))
    if "greet" in behaviours:
        body.insert(0, rng.choice(GREETINGS))
    # Lines that came from the schedule keep their exact wording, so the
    # engine recognizes them; everything else gets the person's voice.
    fixed = set(ONE_WORD + WHY + CLARIFY + WORLD + ABOUT_BOT + YES_NO + GREETINGS)
    user = [l if l in fixed or re.match(r"(?:my name is|call me) ", l) else voice(l, rng, style)
            for l in body]
    return user, behaviours


def simulate(profiles, per_profile, seed=0):
    rng = random.Random(seed)
    total = len(profiles) * per_profile
    # Exact quotas, not coin flips: each behaviour lands in exactly its share
    # of conversations, so even a small build covers every one.
    plan = [[] for _ in range(total)]
    for b, share in SCHEDULE.items():
        for i in rng.sample(range(total), min(total, max(1, math.ceil(share * total)))):
            plan[i].append(b)
    out, n = [], 0
    for prof in profiles:
        others = [p for p in rng.sample(profiles, min(5, len(profiles))) if p["situation"] != prof["situation"]]
        for k in range(per_profile):
            user, behaviours = conversation(prof, others, rng, plan[n])
            n += 1
            out.append({"id": f"{prof['id']}/{k}", "profile": prof["id"], "situation": prof["situation"],
                        "area": prof["area"], "held_out": prof["held_out"], "source": prof["source"],
                        "behaviours": behaviours, "user": user})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("profiles")
    p.add_argument("--per-profile", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    with open(args.profiles) as fh:
        profiles = [json.loads(l) for l in fh if l.strip()]
    convs = simulate(profiles, args.per_profile, args.seed)
    with open(args.out, "w") as fh:
        for c in convs:
            fh.write(json.dumps(c) + "\n")
    print(f"{len(convs)} conversations from {len(profiles)} profiles -> {args.out}")


if __name__ == "__main__":
    main()
