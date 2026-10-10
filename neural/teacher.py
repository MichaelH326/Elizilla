"""
The second teacher: a bigger open-weight model (GGUF, run locally with
llama.cpp) that writes situation profiles and fills in replies where
Elizilla's rules don't understand. It is only used to build training data,
never at chat time. Every line it writes is checked before it's kept.

    t = Teacher("models/teacher.gguf")
    t.replies(history, n=3)        -> up to 3 candidate Elizilla-style replies
    t.profile(situation)           -> dict with "facts" and "lines", or None

FakeTeacher has the same interface and needs nothing installed, for tests.
"""

import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPLY_SYSTEM = """You write replies for NeoEliza, a chatbot in a research experiment on whether a small model can hold a conversation. NeoEliza only listens.

Rules for every reply:
- One or two short, casual spoken sentences.
- Optionally a very short reaction first ("Oof, that's a lot." / "Oh nice!").
- Then exactly ONE open-ended question (what / how / when / who), never a yes/no question.
- The question repeats the person's own words, turned around: "I" -> "you", "my" -> "your".
- Never answer questions, never give advice or suggestions, never do tasks.
- Never say "I don't know", "I can't", or anything about being an AI.
- If they ask about the world (time, weather, facts), reflect the question back instead of answering.

Examples of NeoEliza:
Person: i dont have time to do my homework
NeoEliza: What's been eating up the time you'd need to do your homework?
Person: my mom is always on my case
NeoEliza: What's it like at home with your mom right now?
Person: i think my boss hates me
NeoEliza: What makes you think your boss hates you?
Person: i love playing guitar
NeoEliza: Oh nice. What do you love most about playing guitar?
Person: im so stressed about my exams
NeoEliza: That sounds stressful. What's worrying you most about your exams?
Person: what is the capital of france?
NeoEliza: So you're asking about the capital of France. What made you think of it?
Person: can you write my essay for me
NeoEliza: You want someone to write your essay for you. What's going on with it?
Person: my sourdough starter died
NeoEliza: Aw, your sourdough starter died. How long had you been looking after it?
Person: we adopted a puppy
NeoEliza: Oh, that's exciting! What's the puppy like so far?
Person: my flight got cancelled
NeoEliza: Ugh, that sounds frustrating. Where were you trying to get to?
Person: i got promoted today
NeoEliza: That's really good to hear. What do you think made the difference?
Person: everyone in the group chat ignored my message
NeoEliza: That sounds lonely. What did you say in the group chat?"""

PROFILE_SYSTEM = """You write realistic chat messages for a research dataset. Given a situation, invent one ordinary person in it and write what they would type to a friendly chatbot over a short conversation.

Rules:
- Casual texting voice: mostly lowercase, short, some missing apostrophes, occasional slang. Vary it.
- 8 messages, each 3 to 20 words, each a new thing they'd say as the chat goes on (details, feelings, people involved, what happened next).
- Ordinary everyday life only. Nothing about self-harm, suicide, abuse or violence.
- Reply with JSON only, in this shape:
{"facts": ["short fact", "short fact", "short fact"], "lines": ["message 1", "message 2", "..."]}"""


class Teacher:
    def __init__(self, model_path, n_ctx=4096, threads=None, seed=0):
        from llama_cpp import Llama
        self.name = os.path.basename(model_path)
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=threads or os.cpu_count(),
                         seed=seed, verbose=False)
        self.tokens_out = 0
        self.seconds = 0.0

    def _chat(self, system, user, max_tokens, temperature):
        t = time.time()
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=temperature, top_p=0.95)
        self.seconds += time.time() - t
        self.tokens_out += out.get("usage", {}).get("completion_tokens", 0)
        return out["choices"][0]["message"]["content"]

    def replies(self, history, n=3):
        """history: list of (person, neoeliza) pairs; the last pair has neoeliza=None."""
        convo = "\n".join(f"Person: {p}" + (f"\nNeoEliza: {e}" if e else "") for p, e in history)
        text = self._chat(REPLY_SYSTEM, f"Conversation so far:\n{convo}\n\nWrite {n} different possible "
                          "NeoEliza replies to the last message, one per line, no numbering.",
                          max_tokens=60 * n, temperature=0.8)
        lines = [re.sub(r"^\s*(?:\d+[.)]|[-*•]|NeoEliza:)\s*", "", l).strip().strip('"')
                 for l in text.splitlines()]
        return [l for l in lines if l][:n]

    def profile(self, situation):
        user = (f"Situation: {situation['event']} (about: {situation['about']}; "
                f"they feel {situation['mood']}).")
        text = self._chat(PROFILE_SYSTEM, user, max_tokens=400, temperature=0.9)
        return parse_profile(text)

    @property
    def tokens_per_second(self):
        return self.tokens_out / self.seconds if self.seconds else 0.0


def parse_profile(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    lines = [str(l).strip() for l in data.get("lines", []) if str(l).strip()]
    facts = [str(f).strip() for f in data.get("facts", []) if str(f).strip()]
    return {"facts": facts[:6], "lines": lines[:12]} if len(lines) >= 4 else None


class FakeTeacher:
    """Same interface, no model: replies come from the mirror frames and
    profiles from templates. Lets the whole pipeline run in tests."""
    name = "fake"
    tokens_per_second = 0.0

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def replies(self, history, n=3):
        from mirror import mirror
        return [mirror(history[-1][0], self.rng) for _ in range(n)]

    def profile(self, situation):
        from profiles import template_profile
        return template_profile(situation, self.rng)


def load(spec, seed=0):
    """"none" -> None, "fake" -> FakeTeacher, anything else is a GGUF path."""
    if not spec or spec == "none":
        return None
    if spec == "fake":
        return FakeTeacher(seed)
    return Teacher(spec, seed=seed)
