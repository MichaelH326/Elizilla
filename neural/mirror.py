"""
Mirror replies: the always-grammatical backup. Turn the person's words
around ("my" -> "your") and wrap them in a question frame.

    mirror("i think my boss hates me", rng) -> "What makes you say you think your boss hates you?"
"""

import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neo_eliza import FILLER_WORDS, question_topic, WORLD_QUESTION, WH_START  # noqa: E402
from sentence import normalize, reflect  # noqa: E402
from slm import pos_of  # noqa: E402
from checks import content_words  # noqa: E402

CLAUSE_FRAMES = [
    "What makes you say {c}?",
    "How do you feel about the fact that {c}?",
    "What does it mean to you that {c}?",
    "When you think about how {c}, what comes up for you?",
    "What's the hardest part about the fact that {c}?",
]
TOPIC_FRAMES = [
    "What's on your mind about {t}?",
    "How are you feeling about {t}?",
    "What's {t} been like for you?",
]
QUESTION_FRAMES = [
    "So you're asking about {t}. What made you think of it?",
    "What's got you curious about {t}?",
]
OPEN = ["What's on your mind?", "What else comes to mind?", "How are you feeling right now?"]


def mirror(message, rng=None):
    rng = rng or random.Random()
    words = [w for w in normalize(message) if re.match(r"[a-z0-9']+$", w)]
    sent = " ".join(words)
    if words and (message.strip().endswith("?") or words[0] in WH_START):
        topic = question_topic(words)
        if topic and (WORLD_QUESTION.search(sent) or len(topic.split()) <= 6):
            return rng.choice(QUESTION_FRAMES).format(t=topic)
    while words and words[0] in FILLER_WORDS | {"just", "honestly", "ugh", "lol", "idk"}:
        words.pop(0)
    if not words:
        return rng.choice(OPEN)
    clause = reflect(" ".join(words))
    if not content_words(message):
        return rng.choice(OPEN)                      # "idk", "ok": nothing to mirror
    has_verb = any(pos_of(w) in ("VERB", "AUX") for w in words[1:])
    if has_verb and 2 <= len(words) <= 14 and words[0] in (
            "i", "my", "we", "our", "everyone", "everything", "nobody", "it", "things",
            "people", "they", "the", "this", "life", "work", "school"):
        return rng.choice(CLAUSE_FRAMES).format(c=clause)
    if len(words) <= 4:
        return rng.choice(TOPIC_FRAMES).format(t=clause)
    return rng.choice(OPEN)


if __name__ == "__main__":
    r = random.Random(0)
    for m in ["i think my boss hates me", "my sourdough starter died", "ugh chem exam tomorrow",
              "what is the capital of japan?", "idk", "the puppy", "honestly everything is a mess right now"]:
        print(f"{m!r:45} -> {mirror(m, r)}")
