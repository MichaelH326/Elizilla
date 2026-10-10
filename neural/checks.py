"""
Checks for one Elizilla-side reply. The dataset builder and the eval both
use this file, so the training data and the scoring can never disagree.

    problems(message, reply, earlier_replies) -> list of failed hard checks
    copy_score(message, reply)                -> share of the person's words reused, or None

Hard checks (any failure means the reply can't be used or scores as a miss):
  not-question     doesn't end in "?"
  closed-question  the final question can be answered yes/no
  cut-off          the question stops mid-phrase ("... to do the?")
  too-long         more than 3 sentences or 40 words
  banned           "I don't know", "as an AI", advice ("you should") ...
  format           lists, markdown, emoji, links
  point-of-view    repeats the person's "my X" without turning it into "your X"
  repeat           the same reply as earlier in the conversation, or a doubled word
  answered         a question about the world got an answer (new numbers or names)
  mood-mismatch    an upbeat question ("best part", "enjoy most") about something bad
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo_eliza import CLOSED_QUESTION, FILLER_WORDS, STOPWORDS, WH_START  # noqa: E402
from sentence import normalize, reflect  # noqa: E402

BANNED = re.compile(
    r"\b(?:i (?:do not|don't|dont) know|i (?:can't|cannot|can not|am not able|'m not able)"
    r"|i'm unable|as an ai|an ai (?:model|assistant)|language model|i'm just a (?:program|bot|machine)"
    r"|(?<!say )(?<!think )(?<!that )(?<!like )(?<!sounds like )you should|(?<!say )(?<!that )you shouldn't|you could try|(?:^|[.!?] )try to|have you (?:considered|tried)|i recommend"
    r"|i suggest|my advice|make sure (?:you|to)|it's important to)\b", re.I)
FORMAT = re.compile(r"[*_`#\[\]{}|<>]|^\s*(?:\d+[.)]|-)\s|https?://|www\.|"
                    r"[\U0001F000-\U0001FAFF☀-➿]", re.M)
CUT_OFF = {"the", "a", "an", "your", "my", "and", "or", "but", "to", "of", "with", "for", "in", "on",
           "at", "so", "do", "does", "did", "is", "are", "was", "be", "been", "very", "really", "than"}
HIDDEN_YES_NO = re.compile(r",\s*(?:do|does|did|is|are|was|were|have|has|can|could|would|will|should)\s+"
                           r"(?:you|they|he|she|it|we|there)\b", re.I)
SENTENCES = re.compile(r"(?<=[.!?])\s+")
UPBEAT = re.compile(r"\b(?:best part|enjoy|love most|like most|highlight|going well|so great|"
                    r"that's great|nice!|oh nice|love to hear|good to hear|glad)\b", re.I)
POSITIVE = re.compile(r"\b(?:happy|excited|great|awesome|amazing|love|loved|proud|glad|won|promoted|"
                      r"passed|finally|yay|best|fun|good news|got into|got an a)\b", re.I)
DOWNBEAT = re.compile(r"\b(?:sounds (?:rough|hard|heavy|draining|exhausting|stressful|frustrating|lonely)|"
                      r"i'm sorry|so sorry|oof|ugh|aw\b|that's a lot|tough spot|not easy|hardest part|"
                      r"weighing on you|holding up|wearing you out)", re.I)
NEGATIVE = re.compile(r"\b(?:hurts?|hurting|sick|died|dead|lost|lose|losing|failed|fail|broke|broken|"
                      r"sad|awful|terrible|horrible|worst|hate|hates|mad|angry|scared|worried|worry|"
                      r"worrying|stressed|anxious|nervous|lonely|alone|cried|crying|sucks?|unfair|"
                      r"cancelled|canceled|fired|laid off|dumped|broke up|rejected|ignored|yelled|should have|"
                      r"shouldnt have|shouldn't have|guilty|bad about|regret|messed up|mess up|my fault)\b", re.I)
EXTRA_STOP = STOPWORDS | FILLER_WORDS | {"im", "dont", "cant", "ive", "gonna", "wanna", "got",
                                         "get", "really", "kinda", "lol", "idk", "ngl", "tbh", "?",
                                         "know", "think", "guess", "mean", "maybe", "stuff", "thing",
                                         "dunno", "fine", "meh", "nope", "yep", "yeah", "sure", "okay",
                                         "whatever", "nah", "hmm", "cool", "lol", "haha", "fr", "btw"}


def sentences(text):
    return [s for s in SENTENCES.split(text.strip()) if s]


def content_words(message):
    """The person's own meaningful words, already turned around ("my" -> "your")."""
    words = reflect(" ".join(normalize(message))).split()
    return [w for w in words if w not in EXTRA_STOP and len(w) > 2 and w.isalpha()]


def _stem(w):
    return w[:5]


def copy_score(message, reply):
    words = content_words(message)
    if not words:
        return None
    have = {_stem(w) for w in re.findall(r"[a-z']+", reply.lower())}
    return sum(1 for w in words if _stem(w) in have) / len(words)


_KNOWN = None


def known_words():
    """Ordinary words (lexicon + corpus), so a capitalized word outside them
    in a reply to a question looks like a name the reply brought in."""
    global _KNOWN
    if _KNOWN is None:
        from lexicon import all_words
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        corpus = set()
        for name in ("corpus.txt", "rules.py", "neo_eliza.py"):
            with open(os.path.join(here, name)) as fh:
                corpus |= set(re.findall(r"[a-z']+", fh.read().lower()))
        _KNOWN = all_words() | corpus | STOPWORDS
    return _KNOWN


def is_world_question(message):
    """A question about something out in the world ("what is the capital of
    france?"), not "why?" or a question about the person themselves."""
    toks = normalize(message)
    if not toks or not (message.strip().endswith("?") or toks[0] in WH_START):
        return False
    return bool(content_words(message)) and not {"i", "my", "me", "you", "your"} & set(toks)


def problems(message, reply, earlier=()):
    out = []
    r = reply.strip()
    sents = sentences(r)
    if not r.endswith("?"):
        out.append("not-question")
    elif sents and (CLOSED_QUESTION.match(sents[-1]) or HIDDEN_YES_NO.search(sents[-1])):
        out.append("closed-question")            # also "Hoping for a raise, have you asked ...?"
    last_words = re.findall(r"[a-z']+", sents[-1].lower()) if sents else []
    if last_words and last_words[-1] in CUT_OFF:
        out.append("cut-off")                    # "Why do you want to do the?"
    if len(sents) > 3 or len(r.split()) > 40:
        out.append("too-long")
    if BANNED.search(r):
        out.append("banned")
    if FORMAT.search(r):
        out.append("format")
    low = r.lower()
    for a, b in zip(normalize(message), normalize(message)[1:]):
        if a == "my" and b not in EXTRA_STOP and re.search(rf"\bmy {re.escape(b)}\b", low):
            out.append("point-of-view")
            break
    if r in earlier or re.search(r"\b(\w+) \1\b", low):
        out.append("repeat")
    if (UPBEAT.search(r) and NEGATIVE.search(message)) or (DOWNBEAT.search(r) and POSITIVE.search(message)
                                                          and not NEGATIVE.search(message)):
        out.append("mood-mismatch")
    if is_world_question(message):
        said = set(re.findall(r"[a-z0-9']+", message.lower()))
        new_numbers = [n for n in re.findall(r"\d+", r) if n not in said]
        new_names = [w for w in re.findall(r"\b[A-Z][a-z]+", r)
                     if w.lower() not in said and w.lower() not in known_words()]
        if new_numbers or new_names:
            out.append("answered")
    return out


if __name__ == "__main__":
    for msg, rep in [("i think my boss hates me", "What makes you think your boss hates you?"),
                     ("i think my boss hates me", "You should talk to my boss. Do you like him?"),
                     ("what is the capital of france?", "Paris is the capital. What made you ask?"),
                     ("what is the capital of france?", "So you're asking about the capital of france. What made you think of it?")]:
        print(problems(msg, rep), round(copy_score(msg, rep) or 0, 2), "|", rep)
