#!/usr/bin/env python3
"""
A small language model (SLM) in pure Python, for writing NeoEliza's replies.

Model:   word-level trigram language model with interpolated Kneser-Ney
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


class SLM:
    def __init__(self, order=3, mix=0.7, samples=30, seed=None):
        self.order, self.mix, self.samples = order, mix, samples
        self.rng = random.Random(seed)
        data = load_corpus()
        for intent, sents in rule_templates().items():
            data[intent].extend(sents)
        self.data = data
        self.seen_sentences = {" ".join(s) for sents in data.values() for s in sents}
        self.global_model = KNModel(order).train([s for sents in data.values() for s in sents])
        self.intent_models = {i: KNModel(order).train(s) for i, s in data.items() if len(s) >= 3}

    # --- scoring ---
    def logprob(self, intent, toks):
        im = self.intent_models.get(intent)
        hist = [BOS] * (self.order - 1)
        total = 0.0
        for w in toks + [EOS]:
            p = (1 - self.mix) * self.global_model.prob(hist, w)
            if im:
                p += self.mix * im.prob(hist, w)
            else:
                p /= (1 - self.mix)
            total += math.log(max(p, 1e-12))
            hist = hist[1:] + [w]
        return total / (len(toks) + 1)

    # --- sampling ---
    def _sample_one(self, intent, allowed, max_len=28, temperature=0.8):
        """Walk the trigram chain using only three-word sequences seen in
        training. New sentences appear where two training sentences share a
        two-word overlap and the walk crosses from one to the other."""
        im = self.intent_models.get(intent)
        hist = (BOS,) * (self.order - 1)
        out = []
        for _ in range(max_len):
            first = im if im and self.rng.random() < self.mix else self.global_model
            second = self.global_model if first is im else im
            table = None
            for model in (first, second):
                if model:
                    t = model.counts[self.order - 1].get(hist)
                    t = {w: c for w, c in (t or {}).items()
                         if not (SLOT_RX.fullmatch(w) and SLOT_RX.fullmatch(w).group(1) not in allowed)}
                    if t:
                        table = t
                        break
            if not table:
                return None
            words = list(table)
            w = self.rng.choices(words, [table[x] ** (1 / temperature) for x in words])[0]
            if w == EOS:
                return out
            out.append(w)
            hist = hist[1:] + (w,)
        return None

    def valid(self, toks, slots):
        if not toks or toks[-1] != "?":
            return False
        used = SLOT_RX.findall(" ".join(toks))
        if any(u not in slots for u in used) or len(used) != len(set(used)):
            return False
        if toks.count("?") > 1 or len(toks) < 4:
            return False
        last = " ".join(toks).split(" . ")[-1]
        if CLOSED_START.match(last):
            return False                           # yes/no question
        for a, b in zip(toks, toks[1:]):
            if a == b:
                return False                       # "the the"
        return True

    def write(self, intent, slots, avoid=()):
        """Return (reply, info) or (None, info) if nothing good was generated."""
        allowed = {k for k, v in slots.items() if v}
        best, best_score, novel = None, -1e9, False
        tried = 0
        for _ in range(self.samples):
            toks = self._sample_one(intent, allowed)
            tried += 1
            if toks is None or not self.valid(toks, allowed):
                continue
            filled = detokenize([slots[m.group(1)] if (m := SLOT_RX.fullmatch(t)) else t for t in toks])
            if filled in avoid or any(s in avoid for s in re.split(r"(?<=[.!?])\s+", filled)):
                continue
            score = self.logprob(intent, toks)
            score -= 0.08 * max(0, len(toks) - 18)                   # rambling
            score -= 0.5 * max(0, toks.count(".") - 1)               # more than two sentences
            score += 0.6 * len(SLOT_RX.findall(" ".join(toks)))      # uses the person's words
            is_new = " ".join(toks) not in self.seen_sentences
            score += 0.25 if is_new else 0                           # a little credit for new wording
            if score > best_score:
                best, best_score, novel = filled, score, is_new
        return best, {"score": round(best_score, 2), "new_sentence": novel, "tried": tried}


# ---------------------------------------------------------------------------
# Command line: inspect the model on its own
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    slm = SLM()
    if not args or args[0] == "stats":
        n = sum(len(s) for s in slm.data.values())
        print(f"training sentences: {n}")
        print(f"intents: {len(slm.data)}  vocabulary: {len(slm.global_model.vocab)} words")
        print(f"trigram histories: {len(slm.global_model.counts[2])}")
    elif args[0] == "sample":
        intent = args[1] if len(args) > 1 else "open"
        slots = dict(a.split("=", 1) for a in args[2:])
        for _ in range(5):
            reply, info = slm.write(intent, slots)
            print(f"{reply}   {info}")
