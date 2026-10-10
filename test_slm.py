"""Checks the SLM writer: its gates, its replies, and reflected questions."""
import re
from neo_eliza import NeoEliza, CLOSED_QUESTION
from slm import SLM, grammar_problem

# Sentences that came out of the generator before each gate existed.
BAD = ["how do you and your <topic> ?",
       "what role does your <topic> ?",
       "when you feel <describer> ?",
       "what part of feeling <describer> ?",
       "how long have you been feeling <describer> look like on a normal day ?",
       "how are things between you and your <topic> usually get along ?",
       "what do you want to <follow> ?",
       "what do you wish your <topic> right now ?",
       "what do you usually handle it when <follow> ?"]
GOOD = ["what usually helps when you feel <describer> ?",
        "what do you wish your <topic> understood ?",
        "how does <topic> make you feel ?",
        "what does feeling <describer> look like on a normal day ?",
        "what do you and your <topic> usually do together ?",
        "how do you get along with your <topic> right now ?",
        "what makes you want to <follow> ?"]

slm = SLM(seed=0)
allowed = {"topic", "describer", "follow"}
for s in BAD:
    assert slm.check(s.split(), allowed) is not None, f"should be rejected: {s}"
for s in GOOD:
    assert grammar_problem(s.split(), False) is None, f"should pass: {s}"

# Every generated reply is a question, never left with an unfilled slot.
for intent, slots in [("sad", {"describer": "down"}), ("no-time", {"follow": "do your homework"}),
                      ("family-who", {"topic": "mom", "who": "your mom", "says": "is always on your case"}),
                      ("detail", {"d": "coding tasks"}), ("open", {})]:
    for _ in range(20):
        reply, info = slm.write(intent, slots)
        assert reply and reply.endswith("?") and "<" not in reply, (intent, reply, info)

# Questions and requests are reflected back, never answered or refused.
bot = NeoEliza(seed=1)
for text, rule, words in [("what time is it?", "reflect-question", None),
                          ("whats the weather like tomorrow", "reflect-question", "weather"),
                          ("what is the capital of france?", "reflect-question", "capital of france"),
                          ("can you write my essay for me", "reflect-request", "your essay")]:
    reply = bot.respond(text)
    assert bot.last_rule == rule, (text, bot.last_rule, reply)
    assert not re.search(r"\b(?:i don't know|i can't|i'm not able)\b", reply.lower()), (text, reply)
    assert words is None or words in reply.lower() or reply.count("?") == 1, (text, reply)
    last = re.split(r"(?<=[.!?])\s+", reply.strip())[-1]
    assert reply.endswith("?") and not CLOSED_QUESTION.match(last), (text, reply)

# The template-only mode still works.
plain = NeoEliza(seed=1, use_slm=False)
assert plain.respond("i feel lonely").endswith("?")
assert plain.last_source.startswith("template")

print("ok: SLM gates, generated replies, and reflected questions")
