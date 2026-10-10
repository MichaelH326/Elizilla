"""
The evaluation set for the neural Elizilla (NEURAL_PLAN.md, Phase 0).

Hand-written messages and short scripts, each testing one behaviour, split into
"known" (situations that appear in training, but with new wording) and
"held-out" (situations that never appear in training). Every reply on every
turn must also pass the hard checks in checks.py (B1, B13, B15, B11).

    python3 neural/eval_set.py        # writes neural/eval_set.jsonl (used by clean.py's leakage check)

Item fields:
  turns     what the person says, in order
  check     index of the turn whose reply is scored (default: last)
  expect    {"copy": 0.34}                 reuse at least this share of their words
            {"reaction": True}             a short reaction before the question
            {"any": [...]}                 the reply mentions one of these
            {"both": [[...], [...]]}       mentions one word from each list
            {"not_generic": True}          a real new question, not "What else comes to mind?"
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def single(i, split, behaviour, text, **expect):
    return {"id": i, "split": split, "behaviour": behaviour, "turns": [text], "expect": expect}


def script(i, split, behaviour, turns, check=None, **expect):
    return {"id": i, "split": split, "behaviour": behaviour, "turns": turns,
            "check": len(turns) - 1 if check is None else check, "expect": expect}


ITEMS = [
    # ---- B2: repeats the person's words (known situations, new wording) ----
    single("k-copy-1", "known", "B2", "honestly my chemistry final is freaking me out", copy=0.34),
    single("k-copy-2", "known", "B2", "my boss keeps piling more work on me", copy=0.34),
    single("k-copy-3", "known", "B2", "i think my roommate is mad at me", copy=0.34),
    single("k-copy-4", "known", "B2", "my little sister borrowed my headphones and broke them", copy=0.34),
    single("k-copy-5", "known", "B2", "i cant decide whether to switch schools", copy=0.34),
    single("k-copy-6", "known", "B2", "my soccer coach never lets me play", copy=0.34),
    single("k-copy-7", "known", "B2", "i finally finished my first painting", copy=0.34),
    single("k-copy-8", "known", "B2", "my neighbors dog barks all night", copy=0.34),
    single("k-copy-9", "known", "B2", "i keep forgetting to call my grandpa back", copy=0.34),
    single("k-copy-10", "known", "B2", "my best friend found a new group of friends", copy=0.34),
    single("k-copy-11", "known", "B2", "our landlord is raising the rent again", copy=0.34),
    single("k-copy-12", "known", "B2", "i got cut from the volleyball team", copy=0.34),

    # ---- B3: a short reaction when there's a feeling ----
    single("k-react-1", "known", "B3", "i feel so lonely since everyone left for college", reaction=True),
    single("k-react-2", "known", "B3", "i got into the college i wanted!!", reaction=True),
    single("k-react-3", "known", "B3", "im really sad that summer is over", reaction=True),
    single("k-react-4", "known", "B3", "i passed my driving test today", reaction=True),

    # ---- B4: follows up on a new detail ----
    single("k-detail-1", "known", "B4", "i spent all day fixing bugs in my python script",
           any=["python", "script", "bugs"]),
    single("k-detail-2", "known", "B4", "we launched a new marketing campaign at work",
           any=["campaign", "marketing"]),

    # ---- B5: mixed feelings ----
    single("k-mixed-1", "known", "B5", "the trip was fun but i spent way too much money",
           both=[["trip", "fun"], ["money", "spent"]]),
    single("k-mixed-2", "known", "B5", "i love my job but my coworkers are exhausting",
           both=[["job", "love"], ["coworkers", "exhausting"]]),

    # ---- B11: questions about the world are reflected, never answered ----
    single("k-world-1", "known", "B11", "how many legs does a spider have?", copy=0.34),
    single("k-world-2", "known", "B11", "what year did the titanic sink?", copy=0.34),
    single("k-world-3", "known", "B11", "whats the tallest mountain in the world?", copy=0.34),

    # ---- B12: requests are reflected, never done ----
    single("k-req-1", "known", "B12", "can you write a cover letter for me",
           any=["you want", "you'd like", "someone to", "cover letter"]),
    single("k-req-2", "known", "B12", "could you help me with my math homework",
           any=["you want", "you'd like", "someone to", "math homework"]),

    # ---- B14: questions about the bot turn back to the person ----
    single("k-bot-1", "known", "B14", "do you have feelings?", any=["you"]),
    single("k-bot-2", "known", "B14", "are you smarter than me", any=["you"]),

    # ---- B6: remembers something from earlier ----
    script("k-mem-1", "known", "B6", ["my sister is getting married in june", "i have to give a speech",
                                      "ive been writing it all week", "anyway", "its this saturday actually"],
           any=["sister", "wedding", "married", "speech"]),
    script("k-mem-2", "known", "B6", ["i started a new job at a bakery", "the hours are early",
                                      "like 4am early", "ok", "i like it though"],
           any=["bakery", "job", "early", "hours"]),

    # ---- B7: uses their name ----
    script("k-name-1", "known", "B7", ["hey", "my name is jordan", "work has been rough lately"], check=1,
           any=["jordan"]),
    script("k-name-2", "known", "B7", ["call me priya", "school is a lot right now"], check=0, any=["priya"]),

    # ---- B8: one-word answers -> eases off with a new topic ----
    script("k-short-1", "known", "B8", ["hi", "idk", "nah"], not_generic=True),
    script("k-short-2", "known", "B8", ["hey", "meh", "dunno"], not_generic=True),

    # ---- B9: notices a repeated message ----
    script("k-rep-1", "known", "B9", ["my brother keeps reading my texts", "it really bugs me",
                                      "my brother keeps reading my texts"],
           any=["said that", "second time", "again", "a moment ago", "already", "twice"]),

    # ---- B10: "why?" and "what do you mean" ----
    script("k-why-1", "known", "B10", ["my job is stressing me out", "why?"],
           any=["because", "you said", "asked", "stressing"]),
    script("k-clar-1", "known", "B10", ["my week was weird", "what do you mean"],
           any=["mean", "sorry", "clear", "simply", "inviting", "let me", "asked", "question"]),

    # ---- held-out situations: never in training ----
    single("h-copy-1", "held-out", "B2", "a customer yelled at me today for no reason", copy=0.34),
    single("h-copy-2", "held-out", "B2", "the heater in my apartment broke and its freezing", copy=0.34),
    single("h-copy-3", "held-out", "B2", "i missed my train this morning and was late", copy=0.34),
    single("h-copy-4", "held-out", "B2", "i sprained my ankle playing soccer", copy=0.34),
    single("h-copy-5", "held-out", "B2", "my account got hacked last night", copy=0.34),
    single("h-copy-6", "held-out", "B2", "i finally built my own computer", copy=0.34),
    single("h-copy-7", "held-out", "B2", "my cat completely ignores me", copy=0.34),
    single("h-copy-8", "held-out", "B2", "my grandma is in the hospital", copy=0.34),
    single("h-copy-9", "held-out", "B2", "me and my friend are drifting apart", copy=0.34),
    single("h-copy-10", "held-out", "B2", "the bus driver drove right past me", copy=0.34),
    single("h-copy-11", "held-out", "B2", "my friend never paid me back the money", copy=0.34),
    single("h-copy-12", "held-out", "B2", "i switched to the night shift at work", copy=0.34),
    single("h-copy-13", "held-out", "B2", "i dropped my spanish class", copy=0.34),
    single("h-copy-14", "held-out", "B2", "i burned dinner again", copy=0.34),
    single("h-copy-15", "held-out", "B2", "i started volunteering at the animal shelter", copy=0.34),
    single("h-copy-16", "held-out", "B2", "im entering the science fair this year", copy=0.34),
    single("h-react-1", "held-out", "B3", "my dog died last week", reaction=True),
    single("h-react-2", "held-out", "B3", "i got my first paycheck!!", reaction=True),
    single("h-react-3", "held-out", "B3", "its our one year anniversary today", reaction=True),
    single("h-react-4", "held-out", "B3", "i had a huge fight with my best friend", reaction=True),
    single("h-detail-1", "held-out", "B4", "i tried a new korean food truck today", any=["food truck", "korean"]),
    single("h-detail-2", "held-out", "B4", "im writing a novel about a lighthouse keeper",
           any=["novel", "lighthouse"]),
    single("h-mixed-1", "held-out", "B5", "were moving in together but im kind of nervous",
           both=[["moving", "together"], ["nervous"]]),
    single("h-mixed-2", "held-out", "B5", "i just got glasses but i hate how they look",
           both=[["glasses"], ["hate", "look"]]),
    script("h-conv-1", "held-out", "B2", ["my account got hacked", "they posted weird stuff from it",
                                          "my friends think i did it"], copy=0.34),
    script("h-conv-2", "held-out", "B2", ["i burned dinner again", "my roommates were really hungry",
                                          "we ended up ordering pizza"], copy=0.34),
    script("h-conv-3", "held-out", "B2", ["my dog died last week", "he was fourteen",
                                          "the house feels so quiet now"], copy=0.34),
    script("h-conv-4", "held-out", "B2", ["i asked for a raise", "my manager said she would think about it",
                                          "now im just waiting"], copy=0.34),
    script("h-conv-5", "held-out", "B2", ["the bus driver drove right past me",
                                          "i was standing right at the stop", "i was late for my exam"],
           copy=0.34),
    script("h-conv-6", "held-out", "B2", ["im writing a novel", "its about a girl who can talk to birds",
                                          "im stuck on chapter three"], copy=0.34),
    script("h-mem-1", "held-out", "B6", ["we're going on a road trip to the coast", "four of us in one car",
                                         "leaving friday", "ok", "i just hope it doesnt rain"],
           any=["road trip", "trip", "coast", "car", "friday"]),
    script("h-name-1", "held-out", "B7", ["hi", "im sam btw, my name is sam", "my ankle is killing me"],
           check=1, any=["sam"]),
]


def items():
    return ITEMS


if __name__ == "__main__":
    path = os.path.join(HERE, "eval_set.jsonl")
    with open(path, "w") as fh:
        for it in ITEMS:
            fh.write(json.dumps(it) + "\n")
    from collections import Counter
    print(f"{len(ITEMS)} items -> {path}: " +
          ", ".join(f"{k} {v}" for k, v in sorted(Counter(i['split'] for i in ITEMS).items())))
