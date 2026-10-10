#!/usr/bin/env python3
"""
A small language model (SLM) in pure Python, for writing NeoEliza's replies.

Model:   word-level 4-gram language model with interpolated Kneser-Ney
         smoothing, trained on corpus.txt plus every template in rules.py.
Slots:   topic words are replaced by placeholders (<topic>, <describer>, ...)
         so the model learns the *shape* of a listening reply, and the parser
         fills in the person's own words afterwards.
Intents: one model per intent (rule name) is mixed with one global model,
         so replies stay on-topic but can borrow phrasing from elsewhere.

Generation: sample ~20 candidates, throw out bad ones (unfillable slots,
yes/no questions, things already said), rank the rest by fluency and by how
much of the person's own words they use, and return the best.

Try it on its own:
    python3 slm.py sample sad describer=lonely
    python3 slm.py stats
"""

import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BOS, EOS = "<s>", "</s>"
SEP = "\x1f"                     # joins a word history into one JSON key
SLOT_RX = re.compile(r"<(\w+)>")
CLOSED_START = re.compile(r"^(?:do|does|did|is|are|was|were|have|has|can|could|would|will|should|am)\b", re.I)


# ---------------------------------------------------------------------------
# Tokenizing
# ---------------------------------------------------------------------------

def tokenize(text):
    text = text.lower().replace("’", "'")
    text = re.sub(r"\{(\w+)\}", r"<\1>", text)          # rule templates use {slot}
    text = re.sub(r"([.,!?;\"])", r" \1 ", text)
    return text.split()


def detokenize(tokens):
    text = " ".join(tokens)
    text = re.sub(r" ([.,!?;])", r"\1", text)
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\bi'(m|ve|d|ll)\b", r"I'\1", text)
    # capitalize the start of each sentence
    return re.sub(r"(^|[.!?] )([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


# ---------------------------------------------------------------------------
# Kneser-Ney trigram model
# ---------------------------------------------------------------------------

class KNModel:
    """Interpolated Kneser-Ney language model of a given order."""

    def __init__(self, order=3, discount=0.75):
        self.order = order
        self.d = discount
        self.counts = [defaultdict(Counter) for _ in range(order)]   # [n][history][word]
        self.vocab = set()

    def state(self):
        """Trained counts as plain JSON-ready data (for the .mdlo file)."""
        return {"order": self.order, "discount": self.d, "vocab": sorted(self.vocab),
                "counts": [{SEP.join(h): dict(nxt) for h, nxt in table.items()} for table in self.counts]}

    @classmethod
    def from_state(cls, st):
        m = cls(st["order"], st["discount"])
        m.vocab = set(st["vocab"])
        for k, table in enumerate(st["counts"]):
            for h, nxt in table.items():
                m.counts[k][tuple(h.split(SEP)) if h else ()] = Counter(nxt)
        return m

    def train(self, sentences):
        n = self.order
        for toks in sentences:
            seq = [BOS] * (n - 1) + toks + [EOS]
            self.vocab.update(toks + [EOS])
            for i in range(n - 1, len(seq)):
                w = seq[i]
                self.counts[n - 1][tuple(seq[i - n + 1:i])][w] += 1
        # Lower orders use continuation counts: how many different histories
        # a word follows, not how often it occurs.
        for k in range(n - 1, 0, -1):
            for hist, nxt in self.counts[k].items():
                for w in nxt:
                    self.counts[k - 1][hist[1:]][w] += 1
        return self

    def prob(self, hist, w, k=None):
        """P(w | hist) with interpolated Kneser-Ney."""
        if k is None:
            k = self.order - 1
            hist = tuple(hist[-k:]) if k else ()
        if k < 0:
            return 1.0 / max(len(self.vocab), 1)
        table = self.counts[k].get(hist)
        lower = self.prob(hist[1:], w, k - 1) if k > 0 else 1.0 / max(len(self.vocab), 1)
        if not table:
            return lower
        total = sum(table.values())
        c = table.get(w, 0)
        gamma = self.d * len(table) / total
        return max(c - self.d, 0) / total + gamma * lower

    def sample(self, hist, rng, k=None):
        """Draw the next word exactly from the interpolated distribution by
        either taking an observed continuation or backing off a level."""
        if k is None:
            k = self.order - 1
            hist = tuple(hist[-k:]) if k else ()
        table = self.counts[k].get(hist)
        if table:
            total = sum(table.values())
            gamma = self.d * len(table) / total
            if rng.random() >= gamma or k == 0:
                words = list(table)
                weights = [max(table[w] - self.d, 0) or 1e-9 for w in words]
                return rng.choices(words, weights)[0]
        if k == 0:
            return rng.choice(sorted(self.vocab))
        return self.sample(hist[1:], rng, k - 1)


# ---------------------------------------------------------------------------
# The writer: intent-aware generation + reranking
# ---------------------------------------------------------------------------

def load_corpus(path=os.path.join(HERE, "corpus.txt")):
    data = defaultdict(list)
    if not os.path.exists(path):
        return data
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            intent, text = line.split(":", 1)
            data[intent.strip()].append(tokenize(text))
    return data


def rule_templates():
    """Every reply template in rules.py, grouped by rule name."""
    from rules import RULES, PHRASES
    data = defaultdict(list)
    for r in RULES + PHRASES:
        for t in r["say"]:
            data[r["name"]].append(tokenize(t))
    return data


WH = {"what", "what's", "how", "how's", "when", "who", "who's", "where", "which", "why"}
DO = {"do", "does", "did"}
# Words a question can't stop on: "How do you and your mom?"
BAD_LAST = {"the", "a", "an", "your", "my", "and", "or", "but", "to", "of", "about", "with",
            "for", "in", "on", "at", "so", "do", "does", "did", "is", "are", "was", "be",
            "been", "very", "really", "that", "if", "when", "than"}


SUBJECTS = {"i", "you", "we", "they", "he", "she", "it", "things", "<who>", "<topic>",
            "<whom>", "<what>", "<n1>", "<n2>", "<d>", "<describer>"}
CLAUSE_START = {"when", "if", "that", "because", "while", "how", "what", "wish", "think",
                "hope", "guess", "know"}
INVERTED = {"do", "does", "did", "has", "have", "had", "is", "are", "was", "were", "can",
            "could", "would", "will", "should"}
NO_GAP = {"<follow>", "<what>", "<rest>", "<topic>", "<d>"}


def pos_of(tok):
    """Coarse part of speech from the lexicon; slots carry their own role."""
    if SLOT_RX.fullmatch(tok):
        return "VERB" if tok in ("<action>", "<acting>") else "SLOT"
    if tok.endswith(("'s", "'re", "'d", "'ve", "'ll")) or tok in ("what's", "how's", "who's"):
        return "AUX"
    from lexicon import lookup
    tags = [w.pos for w in lookup(tok)]
    for p in ("AUX", "VERB"):
        if p in tags:
            return p
    return tags[0]


def is_verb(tok):
    return tok in ("<follow>", "<rest>", "<says>") or pos_of(tok) in ("VERB", "AUX")


def grammar_problem(q, known):
    """Small grammar checks for one question (a list of tokens ending in "?").
    These catch the usual ways two spliced replies fail to fit together."""
    if q[0] not in WH and not (q[0] in ("so", "and") and len(q) > 1 and q[1] in WH):
        if not known:
            return "no-wh-start"
    if q[0] in ("when", "where") and pos_of(q[1]) != "AUX":
        return "fragment"                            # "When you feel lonely?"
    if not known and not any(is_verb(t) and not t.endswith("ing") for t in q):
        return "no-verb"                             # "What part of feeling lonely?"
    for i, t in enumerate(q):                        # "What role does your mom?"
        if t in DO and i and q[i - 1] in WH | {"role", "part", "kind"}:
            if not any(is_verb(x) for x in q[i + 2:i + 7]):
                return "no-verb"
    # "How long have you been feeling lonely look like ...": a verb right after
    # the person's words only fits if a helper verb or a clause word comes
    # just before them ("How does <topic> make you feel?", "when <who> is ...").
    for i in range(2, len(q) - 1):
        prev = q[i - 1] if q[i - 1] not in ("usually", "really", "always", "still", "just") else q[i - 2]
        if is_verb(q[i]) and not q[i].endswith("ing") and prev in ("<describer>", "<topic>", "<who>"):
            j = q.index(prev, 0, i) - 1
            while j > 0 and q[j] in ("your", "the", "my", "feeling", "and", "you"):
                j -= 1
            if q[j] not in INVERTED | CLAUSE_START:
                return "two-verbs"
    # "What do you wish your mom right now?": the clause after "wish" or
    # "think" needs its own verb, and a subject before it ("wish would").
    for i, t in enumerate(q[:-1]):
        if t in ("wish", "hope", "think", "guess"):
            if t in ("wish", "hope") and pos_of(q[i + 1]) == "AUX" and q[i + 1] != "you'd":
                return "no-subject"
            rest = q[i + 1:q.index(",", i) if "," in q[i:] else len(q)]
            if not any(is_verb(x) and not x.endswith("ing") and x != "like" for x in rest):
                return "no-verb"
    # "... about talking to a program can understand": a verb can't follow a
    # noun that sits inside "to a program" / "of the dream".
    for i in range(3, len(q) - 1):
        if (is_verb(q[i]) and not q[i].endswith("ing") and q[i] != "like"
                and pos_of(q[i - 1]) == "NOUN" and pos_of(q[i - 2]) in ("DET", "POSS")
                and pos_of(q[i - 3]) == "PREP"):
            return "two-verbs"
    # "How often do you like most about mom?": "<verb> most about" needs "what".
    if q[0] != "what" and any(a == "most" and b == "about" for a, b in zip(q, q[1:])) \
            and any(t in DO for t in q):
        return "no-gap"
    # "What do you want to quit school?": "what" already stands for the
    # object, so the verb can't have another one.
    if q[0] in ("what", "who") and len(q) > 3 and q[1] in DO:
        for a, b in zip(q[3:], q[4:]):
            if (pos_of(a) == "VERB" or a == "to") and b in NO_GAP:
                return "double-object"
        # The word after "what do you (usually)" is the verb, whatever the
        # lexicon says: "What do you usually handle it when ..." fills the gap twice.
        k = 3
        while k < len(q) - 1 and q[k] in ("usually", "really", "always", "still", "just", "most"):
            k += 1
        if k < len(q) - 1 and q[k] not in CLAUSE_START and q[k + 1] in ("it", "them", "that"):
            return "double-object"
    return None


class SLM:
    """Writes a listening reply for a plan chosen by the rules.

    The rules decide *what* to listen for (the intent, e.g. "no-time") and
    hand over the person's own words as slots. The SLM decides *how to say
    it*: it walks a 4-gram chain and, now and then, backs off to a 3-gram so
    it can cross from one known reply into another. That is where new
    sentences come from.

    Like Mira, it would rather not answer than answer badly. Every candidate
    has to pass grammar gates and a confidence gate (mean log-probability per
    word); if none do, write() returns None and the caller uses a hand-written
    template instead.
    """

    SETTINGS = ("order", "mix", "samples", "splice", "frame_len", "temperature",
                "min_confidence", "min_word_confidence")

    def __init__(self, order=4, mix=0.7, samples=40, splice=0.35, frame_len=4, temperature=0.8,
                 min_confidence=-1.5, min_word_confidence=-2.4, seed=None, state=None):
        self.order, self.mix, self.samples, self.frame_len = order, mix, samples, frame_len
        self.temperature = temperature
        self.splice, self.min_confidence = splice, min_confidence
        self.min_word_confidence = min_word_confidence
        self.rng = random.Random(seed)
        if state is not None:
            self._load(state)
            return
        data = load_corpus()
        for intent, sents in rule_templates().items():
            data[intent].extend(sents)
        self.data = data
        everything = [s for sents in data.values() for s in sents]
        self.seen_sentences = {" ".join(s) for s in everything}
        # Known question endings: the last three words before "?".
        self.endings = set()
        for s in everything:
            for i, t in enumerate(s):
                if t == "?" and i >= 3:
                    self.endings.add(tuple(s[i - 3:i]))
        self.global_model = KNModel(order).train(everything)
        self.intent_models = {i: KNModel(order).train(s) for i, s in data.items() if len(s) >= 3}
        self.intent_slots = {i: sorted({m for s in sents for m in SLOT_RX.findall(" ".join(s))})
                             for i, sents in data.items()}

    # --- saving and loading (see mdlo.py) ---
    def state(self):
        """Everything needed to write replies, without the training corpus."""
        return {
            "settings": {k: getattr(self, k) for k in self.SETTINGS},
            "global_model": self.global_model.state(),
            "intent_models": {i: m.state() for i, m in sorted(self.intent_models.items())},
            "intent_slots": self.intent_slots,
            "seen_sentences": sorted(self.seen_sentences),
            "endings": sorted(SEP.join(e) for e in self.endings),
        }

    def _load(self, st):
        for k, v in st["settings"].items():
            setattr(self, k, v)
        self.data = None                             # a loaded model carries no corpus
        self.global_model = KNModel.from_state(st["global_model"])
        self.intent_models = {i: KNModel.from_state(m) for i, m in st["intent_models"].items()}
        self.intent_slots = st["intent_slots"]
        self.seen_sentences = set(st["seen_sentences"])
        self.endings = {tuple(e.split(SEP)) for e in st["endings"]}

    # --- scoring ---
    def logprob(self, intent, toks, weakest=False):
        """Mean log-probability per word: the model's confidence in a reply.
        With weakest=True, also return the log-probability of the least
        likely word, which is where a bad join shows up."""
        im = self.intent_models.get(intent)
        hist = [BOS] * (self.order - 1)
        logps = []
        for w in toks + [EOS]:
            g = self.global_model.prob(hist, w)
            p = self.mix * im.prob(hist, w) + (1 - self.mix) * g if im else g
            logps.append(math.log(max(p, 1e-12)))
            hist = hist[1:] + [w]
        mean = sum(logps) / len(logps)
        return (mean, min(logps)) if weakest else mean

    # --- sampling ---
    def _table(self, model, hist, k, allowed):
        t = model.counts[k].get(hist[len(hist) - k:] if k else ())
        return {w: c for w, c in (t or {}).items()
                if not ((m := SLOT_RX.fullmatch(w)) and m.group(1) not in allowed)}

    def _sample_one(self, intent, allowed, max_len=30):
        """Walk the chain using only word sequences seen in training. Most
        steps follow a known 4-gram; with probability `splice` a step uses
        only the last two words, which lets the walk jump into a different
        training reply that shares those two words."""
        im = self.intent_models.get(intent)
        top = self.order - 1
        hist = (BOS,) * top
        out, splices = [], []
        for _ in range(max_len):
            first = im if im and self.rng.random() < self.mix else self.global_model
            models = [m for m in (first, self.global_model if first is im else im) if m]
            table = full = joined_from = None
            for model in models:
                full = self._table(model, hist, top, allowed)
                # The opening words of a question set its frame ("what do you
                # <verb>", "how does <topic> <verb>"), so they always come from
                # one known reply. Joins can only happen after that.
                head = len(out) - (len(out) - 1 - out[::-1].index(".") if "." in out else -1) - 1
                if head < self.frame_len or (full and self.rng.random() >= self.splice):
                    if full:
                        table = full
                        break
                    continue
                short = self._table(model, hist, 2, allowed)
                if short:
                    table, joined_from = short, full
                    break
            if not table:
                return None, splices
            words = list(table)
            w = self.rng.choices(words, [table[x] ** (1 / self.temperature) for x in words])[0]
            if table is not full and w not in (joined_from or ()):
                splices.append(len(out))             # this word starts a different reply
            if w == EOS:
                return out, splices
            out.append(w)
            hist = hist[1:] + (w,)
        return None, splices

    def check(self, toks, allowed, splices=()):
        """Return why a candidate is rejected, or None if it passes."""
        for p in splices:                            # "that feeling | is trying to tell you"
            if p < len(toks) and (toks[p] == "like" or is_verb(toks[p]) and not toks[p].endswith("ing")):
                return "verb-join"
        if not toks or toks[-1] != "?" or toks.count("?") > 1:
            return "no-question"
        if len(toks) < 4 or len(toks) > 28:
            return "length"
        used = SLOT_RX.findall(" ".join(toks))
        if any(u not in allowed for u in used) or len(used) != len(set(used)):
            return "slots"
        if any(a == b for a, b in zip(toks, toks[1:])):
            return "repeat"
        grams = list(zip(toks, toks[1:], toks[2:]))
        if len(grams) != len(set(grams)):
            return "repeat"                          # "what usually helps ..., what usually helps"
        known = " ".join(toks) in self.seen_sentences
        if not known and len(toks) > 14:
            return "too-long"                        # long new sentences are where splices go wrong
        sents = " ".join(toks).split(" . ")
        q = sents[-1].split()
        if len(sents) > 2 or any(s.split()[-1] == "?" for s in sents[:-1]):
            return "shape"
        if CLOSED_START.match(sents[-1]):
            return "yes-no"                          # "Do you ...?" can be answered with "no"
        if not any(t in WH for t in q):
            return "no-wh"
        if q[-2] in BAD_LAST:
            return "cut-off"                         # "How do you and your mom?"
        if tuple(q[-4:-1]) not in self.endings:
            return "unknown-ending"                  # last three words never ended a question
        return grammar_problem(q, known)

    def write(self, intent, slots, avoid=()):
        """Generate a reply for `intent`, filled with `slots`.
        Returns (reply or None, info)."""
        allowed = {k for k, v in slots.items() if v}
        best, best_score, best_info = None, -1e9, {}
        rejected = Counter()
        for _ in range(self.samples):
            toks, splices = self._sample_one(intent, allowed)
            if toks is None:
                rejected["dead-end"] += 1
                continue
            why = self.check(toks, allowed, splices)
            if why:
                rejected[why] += 1
                continue
            conf, weakest = self.logprob(intent, toks, weakest=True)
            is_new = " ".join(toks) not in self.seen_sentences
            if conf < self.min_confidence or (is_new and weakest < self.min_word_confidence):
                rejected["low-confidence"] += 1
                continue
            filled = detokenize([slots[m.group(1)] if (m := SLOT_RX.fullmatch(t)) else t for t in toks])
            if filled in avoid or any(s in avoid for s in re.split(r"(?<=[.!?])\s+", filled)):
                rejected["already-said"] += 1
                continue
            score = conf
            score -= 0.08 * max(0, len(toks) - 18)                   # rambling
            score += 0.5 * len(SLOT_RX.findall(" ".join(toks)))      # uses the person's words
            score += 0.3 if is_new else 0                            # some credit for new wording
            score += 0.4 * self.rng.random()                         # variety between turns
            if score > best_score:
                best, best_score = filled, score
                best_info = {"confidence": round(conf, 2), "new_sentence": is_new}
        best_info["rejected"] = dict(rejected)
        return best, best_info

    def react(self, mood, avoid=()):
        """A short reaction ("aw, that sounds rough.") to put before a question."""
        intent = "ack-" + mood
        for _ in range(10):
            toks, _ = self._sample_one(intent, set())
            if toks and toks[-1] in (".", "!") and "?" not in toks and len(toks) <= 10:
                text = detokenize(toks)
                if text not in avoid:
                    return text
        return None

# ---------------------------------------------------------------------------
# Command line: inspect the model on its own
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    slm = SLM()
    if not args or args[0] == "stats":
        print(f"training sentences: {len(slm.seen_sentences)} distinct")
        print(f"intents: {len(slm.intent_slots)}  vocabulary: {len(slm.global_model.vocab)} words")
        print(f"word histories: {sum(len(t) for t in slm.global_model.counts)}")
    elif args[0] == "sample":
        intent = args[1] if len(args) > 1 else "open"
        slots = dict(a.split("=", 1) for a in args[2:])
        for _ in range(8):
            reply, info = slm.write(intent, slots)
            print(f"{reply}   {info}")
