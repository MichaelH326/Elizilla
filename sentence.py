"""
Parser: turns a sentence into "frames", one per clause.

    "its exciting but i dont have the time to do my homework"

    clause 1  who=it(@it)  action=be  describer=exciting(@good)
    clause 2  (joined by "but")
              who=i(@self)  action=have(@have)  negated  what=time(@timeamt)
              follow-on: to -> "do my homework"

Rules (rules.py) match on these sections instead of on raw text patterns.
"""

import re
from functools import lru_cache

from lexicon import (Word, lookup, gerund, NOUNS, WORD_TO_CLASS, all_words,
                     PRONOUNS)

# ---------------------------------------------------------------------------
# Normalization and spelling
# ---------------------------------------------------------------------------

CONTRACTIONS = {
    "i'm": "i am", "im": "i am", "you're": "you are", "youre": "you are",
    "can't": "cannot", "cant": "cannot", "won't": "will not", "wont": "will not",
    "don't": "do not", "dont": "do not", "doesn't": "does not", "doesnt": "does not",
    "didn't": "did not", "didnt": "did not", "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not", "wasn't": "was not", "wasnt": "was not",
    "weren't": "were not", "i've": "i have", "ive": "i have", "i'll": "i will",
    "i'd": "i would", "you've": "you have", "youve": "you have",
    "you'll": "you will", "you'd": "you would", "it's": "it is", "its": "it is",
    "that's": "that is", "thats": "that is", "what's": "what is", "whats": "what is",
    "there's": "there is", "theres": "there is", "let's": "let us",
    "couldn't": "could not", "couldnt": "could not", "shouldn't": "should not",
    "shouldnt": "should not", "wouldn't": "would not", "wouldnt": "would not",
    "haven't": "have not", "havent": "have not", "hasn't": "has not",
    "they've": "they have", "theyve": "they have", "they're": "they are",
    "theyre": "they are", "we've": "we have", "weve": "we have",
    "how's": "how is", "hows": "how is", "where's": "where is", "wheres": "where is",
    "who's": "who is", "whos": "who is", "he's": "he is", "she's": "she is",
    "gonna": "going to", "wanna": "want to", "gotta": "have to",
    "alot": "a lot", "u": "you", "ur": "your", "r": "are", "pls": "please",
    "plz": "please", "idk": "i do not know", "tbh": "to be honest",
    "bc": "because", "cuz": "because", "coz": "because", "no one": "noone",
}

COMMON_WORDS = """day week month year today tomorrow yesterday morning night time
really actually probably maybe something anything everything nothing someone
people person thing things life lunch dinner breakfast food sleep weekend
interesting different little lot much many always never sometimes often usually
because though although about again already still since until while before
after relationship feeling feelings thinking thought happen happened problem
problems trouble important difficult easy better worse please thanks thank sorry
honest mind guess mean case piling keeps keep shows""".split()

VOCAB = all_words() | set(COMMON_WORDS)


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"\bno one\b", "noone", text)
    text = re.sub(r"([.!?,;])", r" \1 ", text)
    words = []
    for tok in text.split():
        words.extend(CONTRACTIONS.get(tok, tok).split())
    return [correct_spelling(w) for w in words]


def _edit_distance(a, b, limit):
    """Optimal string alignment distance (insert, delete, substitute, swap = 1)."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    rows = [list(range(len(b) + 1))]
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(rows[-1][j] + 1, cur[j - 1] + 1, rows[-1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], rows[-2][j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        rows.append(cur)
    return rows[-1][-1]


@lru_cache(maxsize=None)
def correct_spelling(word):
    if len(word) < 5 or word in VOCAB or not word.isalpha():
        return word
    for suffix in ("s", "es", "ed", "d", "ing", "ly", "er"):
        stem = word[:-len(suffix)]
        if word.endswith(suffix) and (stem in VOCAB or stem + "e" in VOCAB):
            return word
    # Conservative on purpose: real words outside the lexicon ("kitchen",
    # "annoying") must survive, so only fix one-edit typos of longer words
    # that keep their first letter ("abuot" stays, "freind" -> "friend").
    if len(word) < 6:
        return word
    limit = 1
    best, best_d, tie = word, limit + 1, False
    for v in VOCAB:
        if v[0] != word[0]:
            continue
        d = _edit_distance(word, v, limit)
        if d < best_d:
            best, best_d, tie = v, d, False
        elif d == best_d:
            tie = True
    return best if best_d <= limit and not tie else word


REFLECTIONS = {
    "i": "you", "me": "you", "my": "your", "mine": "yours", "am": "are",
    "myself": "yourself", "was": "were", "your": "my", "yours": "mine",
    "yourself": "myself", "we": "you", "us": "you", "our": "your",
}


def reflect(text):
    out = []
    words = text.split()
    for i, w in enumerate(words):
        if w == "you":
            out.append("I" if i == 0 else "me")
        elif w == "are" and i > 0 and words[i - 1] == "you":
            out.append("am")
        elif w == "was" and i > 0 and words[i - 1] not in ("i",):
            out.append("was")
        else:
            out.append(REFLECTIONS.get(w, w))
    return " ".join(out).strip(" .!?,;")


# ---------------------------------------------------------------------------
# Tagging: choose one tag per word using its neighbours
# ---------------------------------------------------------------------------

SUBJECT_PRONOUNS = {"i", "you", "we", "they", "he", "she"}


def tag(words):
    tagged = []
    for i, w in enumerate(words):
        options = lookup(w)
        prev = tagged[-1] if tagged else None
        nxt = lookup(words[i + 1]) if i + 1 < len(words) else []
        nxt_pos = {o.pos for o in nxt}
        by_pos = {o.pos: o for o in options}

        def choose(*order):
            for p in order:
                if p in by_pos:
                    return by_pos[p]
            return None

        pick = None
        prev_pos = prev.pos if prev else None
        prev_text = prev.text if prev else None
        # nearest earlier word that isn't an adverb or negation
        anchor = next((t for t in reversed(tagged) if t.pos not in ("ADV", "NEG")), None)

        if "AUX" in by_pos and "VERB" in by_pos:
            pick = by_pos["AUX"] if nxt_pos & {"NEG", "VERB", "AUX"} and not nxt_pos & {"NOUN", "DET"} else by_pos["VERB"]
        elif "ADJ" in by_pos and anchor is not None and (
                (anchor.pos == "AUX" and anchor.lemma == "be") or
                (anchor.pos == "VERB" and anchor.lemma in ("feel", "seem", "look", "get", "become", "sound"))):
            pick = by_pos["ADJ"]                         # "i am really stressed"
        elif w in ("that", "this") and prev_pos in ("VERB", "ADJ") and prev.cat not in (None,) and nxt_pos & {"PRON", "POSS", "NOUN", "DET"}:
            pick = by_pos.get("COMP")                    # "i like that i get to..."
        elif w == "that" and prev_pos == "PRON" and prev_text == "it":
            pick = by_pos.get("COMP")
        elif w == "her":
            pick = choose("POSS") if nxt_pos & {"NOUN", "ADJ"} else choose("PRON")
        elif w == "no":
            pick = choose("DET") if nxt_pos & {"NOUN", "ADJ"} and "INTJ" not in nxt_pos else choose("NEG")
        elif w == "so":
            pick = choose("ADV") if nxt_pos & {"ADJ", "ADV"} or not prev else choose("CONJ")
        elif w == "like":
            if prev_pos in ("PRON", "AUX", "NEG") or (prev_pos == "ADV" and len(tagged) > 1 and tagged[-2].pos in ("PRON", "AUX", "NEG")) or prev_text == "to":
                pick = choose("VERB")
            elif prev_pos == "VERB" and prev.lemma in ("feel", "look", "seem", "sound"):
                pick = Word("like", "PREP")
            else:
                pick = choose("INTJ")
        elif w == "to" and prev is not None and prev.lemma in ("go", "come", "get", "drive", "walk") and words[i + 1:i + 2] and words[i + 1] in NOUNS:
            pick = Word("to", "PREP")                    # "going to work" = a place
        elif w == "to" and nxt_pos & {"VERB", "AUX"} and not (nxt_pos & {"NOUN"} and words[i + 1] in NOUNS and words[i + 1] not in ("work",)):
            pick = Word("to", "INF")                     # infinitive marker
        elif "VERB" in by_pos and (prev_pos in ("PRON", "AUX", "NEG", "INF") or
                                   (prev_pos == "ADV" and len(tagged) > 1 and tagged[-2].pos in ("PRON", "AUX", "NEG"))):
            pick = choose("VERB")
        elif "NOUN" in by_pos and prev_pos in ("DET", "POSS", "ADJ", "PREP", "NUM"):
            pick = choose("NOUN")
        elif "ADJ" in by_pos and (prev_pos in ("AUX", "ADV", "NEG") or (prev_pos == "VERB" and prev.lemma in ("feel", "seem", "look", "get", "become", "sound"))):
            pick = choose("ADJ")

        if pick is None:
            pick = options[0]
            # An unknown word right after "i/you/we/they" is probably a verb.
            if pick.pos == "NOUN" and pick.cat in ("@thing", "@activity") and prev_text in SUBJECT_PRONOUNS:
                pick = Word(w, "VERB", "@act", w)
            # unknown word right after "is/was/were" describes something ("they were rotten")
            if pick.pos == "NOUN" and pick.cat == "@thing" and anchor is not None and anchor.pos == "AUX" and anchor.lemma == "be":
                pick = Word(w, "ADJ", "@desc")
            # "-ing" word after a be-verb is a verb ("i am coding")
            if pick.pos == "NOUN" and w.endswith("ing") and prev_pos == "AUX":
                pick = Word(w, "VERB", "@act", w[:-3])
        tagged.append(pick)
    return tagged


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

class Frame:
    """The sections of one clause."""

    def __init__(self, words, conj=None, question=False):
        self.words = words
        self.conj = conj                 # the conjunction that introduced this clause
        self.question = question
        self.subj = self.subj_text = self.subj_owner = None
        self.neg = False
        self.modal = None
        self.verb = self.vcat = self.verb_text = None
        self.obj = self.ocat = self.obj_head = None
        self.adj = self.acat = None
        self.comp_type = self.comp_text = None
        self.comp = None                 # parsed follow-on clause
        self.pred = ""                   # everything after the main verb
        self.says = ""                   # everything after the subject
        self.nouns = []                  # (phrase, head, category) for every noun phrase
        self.cats = set()                # every category mentioned in the clause
        self.time = None

    @property
    def text(self):
        return " ".join(w.text for w in self.words)

    @property
    def fragment(self):
        return self.verb is None

    def describe(self):
        parts = []
        if self.conj:
            parts.append(f"[{self.conj}]")
        if self.subj:
            parts.append(f"who={self.subj_text}({self.subj})")
        if self.verb:
            parts.append(f"action={self.verb}({self.vcat})")
        if self.modal:
            parts.append(f"modal={self.modal}")
        if self.neg:
            parts.append("NOT")
        if self.adj:
            parts.append(f"describer={self.adj}({self.acat})")
        if self.obj:
            parts.append(f"what={self.obj}({self.ocat})")
        if self.comp_type:
            parts.append(f"follow-on={self.comp_type}:'{self.comp_text}'")
        if self.time:
            parts.append(f"when={self.time}")
        if self.fragment:
            parts.append("fragment")
            if self.nouns:
                parts.append("nouns=" + ",".join(f"{n}({c})" for n, _, c in self.nouns))
        if self.question:
            parts.append("question")
        return " ".join(parts)


BREAK_CONJ = {"but", "because", "although", "though", "however", "so", "while", "or"}
HEDGES = [("i", "guess"), ("i", "mean"), ("i", "suppose"), ("to", "be", "honest")]


def split_clauses(tagged):
    """Split on punctuation and conjunctions. "and" only splits when both
    sides have their own verb ("school and work" stays together)."""
    clauses, cur, conj = [], [], None
    for i, w in enumerate(tagged):
        if w.text in ".!?,;":
            if cur:
                clauses.append((conj, cur))
            cur, conj = [], None
            continue
        if w.pos == "CONJ" and (w.text in BREAK_CONJ or w.text == "and"):
            rest = tagged[i + 1:]
            has_verb_after = any(t.pos in ("VERB", "AUX") for t in rest[:6])
            has_verb_before = any(t.pos in ("VERB", "AUX") for t in cur)
            if w.text != "and" or (has_verb_after and has_verb_before):
                if cur:
                    clauses.append((conj, cur))
                cur, conj = [], w.text
                continue
        cur.append(w)
    if cur:
        clauses.append((conj, cur))
    return clauses


def noun_phrases(words):
    """Find [det/poss] [adj]* noun+ groups."""
    out, i = [], 0
    while i < len(words):
        j = i
        while j < len(words) and (words[j].pos in ("DET", "POSS") or words[j].text == "of" and j > i):
            j += 1
        k = j
        while k < len(words) and words[k].pos == "ADJ":
            k += 1
        m = k
        while m < len(words) and words[m].pos in ("NOUN", "NUM"):
            m += 1
        if m > k:
            phrase = words[i:m]
            head = words[m - 1]
            out.append((" ".join(w.text for w in phrase), head.text, head.cat))
            i = m
        else:
            i += 1
    return out


def build_frame(words, conj=None, question=False, inherit_subj=None):
    f = Frame(words, conj, question)
    for w in words:
        if w.cat:
            f.cats.add(w.cat)
        if w.cat == "@time" and w.pos in ("NOUN", "NUM"):
            f.time = w.text
    f.nouns = [np for np in noun_phrases(words)]

    # Strip leading fillers, hedges and stray adverbs.
    body = list(words)
    while body and (body[0].pos in ("INTJ", "CONJ") or (body[0].pos == "ADV" and body[0].cat in ("@hedge", "@just"))):
        body.pop(0)
    for h in HEDGES:
        if tuple(w.text for w in body[:len(h)]) == h and len(body) > len(h) + 1:
            body = body[len(h):]

    # Find the first verb or helper verb.
    v = next((i for i, w in enumerate(body) if w.pos in ("VERB", "AUX")), None)
    if v is None:
        for w in body:
            if w.pos == "ADJ" and not f.adj:
                f.adj, f.acat = w.text, w.cat
            if w.pos == "NEG":
                f.neg = True
        return f

    # --- who: the subject before the verb ---
    pre = body[:v]
    if question and v == 0:
        pre = [w for w in body[1:3] if w.pos in ("PRON", "POSS", "NOUN")][:1]
    if question and v == 1 and body[0].text in ("what", "how", "why", "who", "where", "when"):
        pre = [w for w in body[v + 1:v + 3] if w.pos in ("PRON", "POSS", "NOUN")][:2]
    pron = next((w for w in pre if w.pos == "PRON"), None)
    nps = noun_phrases(pre)
    if nps:
        phrase, head, cat = nps[-1]
        f.subj, f.subj_text = cat, phrase
        owner = next((w for w in pre if w.pos == "POSS"), None)
        f.subj_owner = owner.cat if owner else None
    elif pron:
        f.subj, f.subj_text = pron.cat, pron.text
    elif inherit_subj:
        f.subj, f.subj_text = inherit_subj

    f.says = " ".join(w.text for w in body[v:])

    # --- action: skip helper verbs, negation and adverbs to the main verb ---
    i = v
    last_aux = None
    while i < len(body) and body[i].pos in ("AUX", "NEG", "ADV"):
        w = body[i]
        if w.pos == "NEG" or w.text == "cannot":
            f.neg = True
        if w.pos == "AUX":
            last_aux = w
            if w.lemma in ("can", "will", "should", "may", "must"):
                f.modal = w.lemma
            # "have" / "do" followed by a noun is the main verb, not a helper
            nxt = body[i + 1] if i + 1 < len(body) else None
            if w.lemma in ("have", "do") and (nxt is None or nxt.pos in ("DET", "NOUN", "POSS", "PRON", "NUM", "ADJ", "INF", "PREP")):
                break
        i += 1
    if i < len(body) and body[i].pos == "VERB":
        main, i = body[i], i + 1
    elif last_aux is not None and i < len(body) and body[i] is last_aux:
        main, i = last_aux, i + 1
    else:
        main = last_aux
    f.verb, f.verb_text = main.lemma, main.text
    f.vcat = main.cat if main.pos == "VERB" else ("@be" if main.lemma == "be" else "@" + main.lemma)
    if main.lemma == "have":
        f.vcat = "@have"
    if main.lemma == "do":
        f.vcat = "@do"
    if f.modal and main is last_aux and main.lemma == f.modal:
        f.vcat = "@" + f.modal
    rest = body[i:]
    if question and v == 0 and pre and rest and rest[0] is pre[0]:
        rest = rest[1:]                  # "are YOU a robot": skip the subject

    # "going to X", "have to X", "get to X": the real action is X
    if f.verb in ("go", "have", "get") and len(rest) >= 2 and rest[0].pos == "INF":
        inner = build_frame(rest[1:], inherit_subj=(f.subj, f.subj_text))
        inner.neg = inner.neg or f.neg
        inner.conj, inner.question, inner.cats = conj, question, f.cats
        inner.nouns, inner.time = f.nouns, f.time
        return inner

    f.pred = " ".join(w.text for w in rest)
    j = 0
    while j < len(rest) and rest[j].pos in ("ADV", "NEG"):
        if rest[j].pos == "NEG":
            f.neg = True
        j += 1
    rest = rest[j:]
    if not rest:
        return f
    first = rest[0]

    def follow_on(kind, words_after):
        f.comp_type = kind
        f.comp_text = " ".join(w.text for w in words_after)
        if words_after:
            f.comp = build_frame(words_after, inherit_subj=(f.subj, f.subj_text))

    # --- describer: "i am tired", "they have been crazy", "i feel happy"
    if first.pos == "ADJ" and f.vcat in ("@be", "@feel", "@get"):
        f.adj, f.acat = first.text, first.cat
        after = rest[1:]
        if after and after[0].pos == "CONJ" and after[0].text == "but":
            after = after[1:]
        if after and (after[0].pos == "COMP" or after[0].text in ("about", "because", "with", "at")):
            follow_on(after[0].text, after[1:])
        return f
    # --- follow-on clause: "that ...", "when ...", "to ...", "like ..."
    if first.pos == "COMP" or first.text in ("when", "how", "if", "what", "that"):
        follow_on(first.text, rest[1:])
        return f
    if first.pos == "INF":
        follow_on("to", rest[1:])
        return f
    if first.text == "like" and f.vcat == "@feel":
        follow_on("like", rest[1:])
        return f
    if first.pos in ("NOUN", "VERB") and first.text.endswith("ing") and f.vcat in ("@like", "@dislike"):
        follow_on("ing", rest)
        return f
    if f.vcat in ("@belief", "@say", "@know", "@remember", "@feel", "@worry") and \
            any(w.pos in ("VERB", "AUX") for w in rest[1:]) and first.pos in ("PRON", "POSS", "DET", "NOUN"):
        follow_on("that", rest)
        return f
    # --- what: the object noun phrase (or pronoun)
    if first.pos == "PRON":
        f.obj, f.ocat, f.obj_head = first.text, first.cat, first.text
        after = rest[1:]
    else:
        nps = noun_phrases(rest)
        lead = []
        for w in rest:
            if w.pos in ("DET", "POSS", "ADJ", "NOUN", "NUM") or (w.text == "of" and lead):
                lead.append(w)
            else:
                break
        nps = noun_phrases(lead)
        if nps:
            f.obj, f.obj_head, f.ocat = nps[-1]
            f.obj = " ".join(w.text for w in lead)
        after = rest[len(lead):]
    # "i hate it when ...", "i have no time to ..."
    if after and (after[0].pos == "COMP" or after[0].text in ("when", "that")):
        follow_on(after[0].text, after[1:])
    elif after and after[0].pos == "INF":
        follow_on("to", after[1:])
    elif after and after[0].text == "for" and f.ocat == "@timeamt":
        follow_on("to", after[1:])
    return f


def parse(words, raw=""):
    """Return the list of frames for a normalized word list."""
    tagged = tag([w for w in words])
    question = raw.strip().endswith("?") or bool(words and words[0] in ("what", "why", "how", "who", "where", "when", "do", "does", "can", "are", "is"))
    frames, prev_subj = [], None
    for conj, clause in split_clauses(tagged):
        f = build_frame(clause, conj, question, inherit_subj=prev_subj)
        if f.subj:
            prev_subj = (f.subj, f.subj_text)
        frames.append(f)
    return frames, tagged


def main_frame(frames):
    """The clause that carries the point: whatever follows "but", otherwise the
    most complete clause."""
    for f in frames:
        if f.conj == "but" and f.verb:
            return f

    def richness(f):
        return (2 if f.verb else 0) + (1 if f.subj else 0) + (1 if f.adj or f.obj or f.comp_type else 0)
    return max(frames, key=richness) if frames else None
