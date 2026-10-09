"""Checks that every NeoEliza reply ends with an open-ended question."""
import re
from neo_eliza import NeoEliza, CLOSED_QUESTION

SCRIPT = ["hi eliza", "pretty good", "not much", "not much", "no", "nothing",
          "my work is great", "i hate it when no one shows up", "i dont know",
          "what do you mean", "why?", "yeah", "i feel kinda tierd",
          "my mom is always on my case", "I want to quit school",
          "are you a robot", "whats on your mind", "everyone ignores me",
          "i had a dream about falling", "ok", "sure", "i am a student"]

for seed in range(20):
    bot = NeoEliza(seed=seed)
    for line in SCRIPT:
        reply = bot.respond(line)
        last = re.split(r"(?<=[.!?])\s+", reply.strip())[-1]
        assert reply.endswith("?") and not CLOSED_QUESTION.match(last), (seed, line, reply)
print("ok: every reply ends with an open-ended question")
