#!/usr/bin/env python3
"""
NeoEliza - an ELIZA-style chatbot driven by sentence sections and word categories.

Pipeline for every message:
  1. normalize    contractions, slang, spelling correction        (sentence.py)
  2. tag          each word gets a part of speech + category      (lexicon.py)
  3. parse        split into clauses; each clause into sections:  (sentence.py)
                  who / action / not / what / describer / follow-on / when
  4. match        rules say which sections + categories they need (rules.py)
  5. score        rule weight + how many sections matched + mood fit
                  - novelty penalty (topics already discussed)
                  - recently used rules - rules whose replies are used up
  6. reply        the winning rule is a listening plan: what to ask about,
                  plus the person's own words as slots. The SLM (slm.py)
                  writes new wording for it. If no generated reply passes
                  the SLM's grammar and confidence gates, a hand-written
                  template is used. Every reply ends in an open-ended question.

It never answers questions or does tasks. Like a listener, it reflects them
back with the person's own words ("So you're asking about the capital of
France. What made you think of it?").

Any talk of suicide, self-harm or harming others gets the crisis route
professionals advise (988, Crisis Text Line, 911) and locks the bot for good;
see safety.py. Clear the lock deliberately with --reset-lock.

Run with --debug to see the sections, the winning rule, and where each reply
came from. Run with --no-slm to use only the hand-written templates, or with
--model elizilla.mdlo to load a packed model (see mdlo.py).
"""

import math
import random
import re
import sys
from collections import Counter, defaultdict, deque

from lexicon import WORD_TO_CLASS, gerund
from rules import RULES, PHRASES
from sentence import normalize, parse, main_frame, reflect
from slm import SLM
import mdlo
import safety

# ---------------------------------------------------------------------------
# Emotion model
# ---------------------------------------------------------------------------

EMOTION_CATS = {"@sad": (-0.7, 0.4), "@happy": (0.7, 0.6), "@anxious": (-0.6, 0.8),
                "@angry": (-0.7, 0.9), "@tired": (-0.4, 0.1), "@lonely": (-0.6, 0.3),
                "@good": (0.5, 0.4), "@bad": (-0.5, 0.4), "@busy": (-0.1, 0.8),
                "@confused": (-0.3, 0.5), "@guilty": (-0.6, 0.5)}
EMOTION_WORDS = {"love": (0.8, 0.6), "hate": (-0.8, 0.8), "hurt": (-0.7, 0.5),
                 "cry": (-0.7, 0.6), "crying": (-0.7, 0.6), "fail": (-0.6, 0.5),
                 "failed": (-0.6, 0.5), "failure": (-0.7, 0.5), "worthless": (-0.9, 0.4),
                 "calm": (0.4, 0.1), "relaxed": (0.5, 0.1), "hope": (0.5, 0.4),
                 "thanks": (0.5, 0.3), "thank": (0.5, 0.3), "sorry": (-0.3, 0.3),
                 "enjoy": (0.6, 0.4), "like": (0.3, 0.3), "stressed": (-0.6, 0.8),
                 "worried": (-0.6, 0.7), "excited": (0.7, 0.8), "ok": (0.1, 0.2),
                 "okay": (0.1, 0.2), "fine": (0.2, 0.2)}
NEGATORS = {"not", "no", "never", "cannot", "hardly", "barely", "nothing"}
INTENSIFIERS = {"very": 1.5, "really": 1.4, "so": 1.4, "extremely": 1.8,
                "super": 1.5, "totally": 1.4, "incredibly": 1.7,
                "slightly": 0.6, "kinda": 0.7, "somewhat": 0.7, "bit": 0.7}


# Questions about the world and requests for help aren't answered: like a
# listener, Elizilla reflects them back with the person's own words.
WORLD_QUESTION = re.compile(r"\b(?:what time|what day is|what is the date|today s date|todays date"
                            r"|weather|forecast|news|stock price|the score)\b"
                            r"|^(?:what|who|when|where|how many|how much|how far|how old|how tall)\b.*\b"
                            r"(?:capital|president|population|invented|discovered|distance|tallest"
                            r"|largest|biggest|fastest|square root|plus|minus|times|meaning of|define"
                            r"|definition|located|born)\b")
REQUEST = re.compile(r"^(?:can|could|will|would) you (?:please )?(?P<req>(?:help me (?:with|write|do|fix)"
                     r"|write|code|fix|calculate|search|look up|google|book|order|send|make me|do my)\b.*)"
                     r"|^(?:please )?(?P<req2>(?:write|calculate|search|look up|google|translate) .*)")
QUESTION_AUX = {"is", "are", "was", "were", "do", "does", "did", "will", "can", "would", "should", "could"}
REFLECT_QUESTION = ["So you're asking about {topic}. What made you think of it?",
                    "What's on your mind about {topic}?",
                    "What got you wondering about that?"]
REFLECT_REQUEST = ["You want someone to {follow}. What's going on with it?",
                   "So you'd like someone to {follow}. How's it going so far?"]


def question_topic(words):
    """The thing a question is about, in the person's words:
    "what is the capital of france" -> "the capital of france"."""
    w = [x for x in words if x != "?"]
    while w and w[0] in WH_START | {"many", "much", "far", "old", "tall"}:
        w.pop(0)
    while w and w[0] in QUESTION_AUX:
        w.pop(0)
    out = []
    for x in w:
        if x in QUESTION_AUX or x in ("like", "it"):
            break
        out.append(x)
    return reflect(" ".join(out)) if out else None


def emotion_value(tagged_word):
    if tagged_word.text in EMOTION_WORDS:
        return EMOTION_WORDS[tagged_word.text]
    if tagged_word.pos == "ADJ" and tagged_word.cat in EMOTION_CATS:
        return EMOTION_CATS[tagged_word.cat]
    return None


class EmotionModel:
    def __init__(self, smoothing=0.6):
        self.valence = 0.0
        self.arousal = 0.3
        self.smoothing = smoothing
        self.last_turn = (0.0, 0.0, 0)  # valence, arousal, hits

    def score(self, tagged):
        v_sum = a_sum = 0.0
        hits = 0
        for i, w in enumerate(tagged):
            val = emotion_value(w)
            if not val:
                continue
            v, a = val
            window = [t.text for t in tagged[max(0, i - 3):i]]
            mult = 1.0
            for p in window:
                mult *= INTENSIFIERS.get(p, 1.0)
            if any(p in NEGATORS for p in window):
                v = -v * 0.6  # "not happy" is negative, but milder than "sad"
            v_sum += v * mult
            a_sum += a * min(mult, 1.5)
            hits += 1
        if hits:
            v_t = max(-1, min(1, v_sum / hits))
            a_t = max(0, min(1, a_sum / hits))
            self.valence = self.smoothing * v_t + (1 - self.smoothing) * self.valence
            self.arousal = self.smoothing * a_t + (1 - self.smoothing) * self.arousal
            self.last_turn = (v_t, a_t, hits)
        else:
            self.valence *= 0.85
            self.last_turn = (0.0, 0.0, 0)
        return self.last_turn

    @property
    def label(self):
        if self.valence < -0.35:
            return "low"
        if self.valence > 0.35:
            return "positive"
        return "neutral"


# ---------------------------------------------------------------------------
# Memory: facts come from the parsed sections
# ---------------------------------------------------------------------------

JOBS = set("""student teacher doctor nurse engineer developer programmer artist
writer lawyer designer manager cashier chef cook driver mechanic accountant
scientist researcher intern""".split())
STOPWORDS = set("""a an the and or but if then so to of in on at for with by from
is are was were be been being am do does did have has had i you me my your it
this that these those what which who whom how why when where not no yes just
very really can could would should will shall may might must about as into
than too also there here they them their we us our he she him her his its
""".split())


def strip_det(phrase):
    words = phrase.split()
    while words and words[0] in ("a", "an", "the", "my", "your", "our", "his", "her",
                                 "their", "some", "any", "lot", "of", "lots"):
        words.pop(0)
    return " ".join(words)


class Memory:
    def __init__(self):
        self.facts = defaultdict(list)
        self.history = []               # (normalized string, bag of words)
        self.pending = deque(maxlen=6)  # (turn, key, value) to bring up later

    def add(self, key, val):
        if val and val not in self.facts[key]:
            self.facts[key].append(val)
            return True
        return False

    def extract(self, sent, frames):
        new = []
        for pat, key in ((r"\bmy name is (\w+)", "name"), (r"\bcall me (\w+)", "name"),
                         (r"\bi am (\d+) years? old", "age"), (r"\bi live in ([\w ]+)", "home")):
            m = re.search(pat, sent)
            if m and self.add(key, m.group(1).strip()):
                new.append((key, m.group(1).strip()))
        for f in frames:
            if f.subj == "@self" and f.vcat in ("@like", "@dislike"):
                key = "likes" if f.vcat == "@like" else "dislikes"
                val = f.comp_text if f.comp_type in ("ing", "to") else (f.obj if f.ocat not in ("@it", None) else None)
                if val and self.add(key, " ".join(val.split()[:5])):
                    new.append((key, val))
            if f.subj_owner == "@self" and f.subj not in ("@name",) and f.verb == "be" and (f.adj or f.pred):
                topic = strip_det(f.subj_text)
                val = (topic, f.adj or " ".join(f.pred.split()[:5]))
                if self.add("about", val):
                    new.append(("about", val))
            if f.subj == "@self" and f.verb == "be" and f.obj and strip_det(f.obj) in JOBS:
                if self.add("job", strip_det(f.obj)):
                    new.append(("job", strip_det(f.obj)))
        return new

    def remember_turn(self, sent, words):
        self.history.append((sent, Counter(w for w in words if re.match(r"[a-z0-9']+$", w))))

    def most_similar(self, words, exclude_last=1):
        """Cosine similarity over bags of words, ignoring stopwords."""
        q = Counter(w for w in words if w not in STOPWORDS and re.match(r"[a-z]+$", w))
        if not q:
            return None, 0.0
        best, best_s = None, 0.0
        pool = self.history[:-exclude_last] if exclude_last else self.history
        for sent, bag in pool:
            b = {w: c for w, c in bag.items() if w not in STOPWORDS}
            dot = sum(q[w] * b.get(w, 0) for w in q)
            if not dot:
                continue
            s = dot / (math.sqrt(sum(v * v for v in q.values())) *
                       math.sqrt(sum(v * v for v in b.values())))
            if s > best_s:
                best, best_s = sent, s
        return best, best_s


# ---------------------------------------------------------------------------
# Rule matching on sections
# ---------------------------------------------------------------------------

def _opts(v):
    return v if isinstance(v, tuple) else (v,)


def match(rule, frame):
    """Return the list of categories/words the rule matched, or None."""
    hits = []
    for key, want in rule["when"].items():
        opts = _opts(want)
        if key == "who":
            ok = frame.subj in opts
            hits.append(frame.subj)
        elif key == "mine":
            ok = frame.subj_owner == "@self"
        elif key == "action":
            ok = frame.vcat in opts or frame.verb in opts
            hits.append(frame.verb)
        elif key == "not":
            ok = frame.neg == want
        elif key == "modal":
            ok = frame.modal in opts
        elif key == "describer":
            ok = frame.acat is not None and ("any" in opts or frame.acat in opts)
            hits.append(frame.acat)
        elif key == "what":
            ok = frame.ocat is not None and ("any" in opts or frame.ocat in opts or frame.obj_head in opts)
        elif key == "follow":
            ok = frame.comp_type is not None and ("any" in opts or frame.comp_type in opts)
        elif key == "joined":
            ok = frame.conj in opts
        elif key == "has":
            found = [c for c in opts if c in frame.cats]
            ok = bool(found)
            hits.extend(found)
        elif key == "fragment":
            ok = frame.fragment == want
        elif key == "pair":
            ok = frame.fragment and len(frame.nouns) >= 2
        elif key == "question":
            ok = frame.question == want
        else:
            raise ValueError(f"unknown rule key {key!r} in rule {rule['name']}")
        if not ok:
            return None
    return [h for h in hits if h]


OBJECT_FORM = {"i": "you", "you": "me", "they": "them", "he": "him", "she": "her",
               "we": "you", "it": "it", "everyone": "everyone"}


def slots_for(rule, frame, frames):
    s = {}
    if frame.subj_text:
        s["who"] = reflect(frame.subj_text)
        s["whom"] = OBJECT_FORM.get(frame.subj_text, s["who"])
    if frame.verb and frame.verb != "be":
        s["action"] = frame.verb
        s["acting"] = gerund(frame.verb)
    if frame.obj and frame.ocat not in ("@self",):
        s["what"] = reflect(frame.obj)
    if frame.adj:
        s["describer"] = frame.adj
    if frame.comp_text:
        s["follow"] = reflect(frame.comp_text)
    if frame.pred:
        s["rest"] = reflect(frame.pred)
    if frame.says:
        s["says"] = reflect(frame.says)
    if frame.time:
        s["when"] = frame.time
    # {topic}: the noun phrase that made a "has"/"who" category match
    wanted = set(_opts(rule["when"].get("has", ()))) | set(_opts(rule["when"].get("who", ())))
    for phrase, head, cat in frame.nouns:
        if cat in wanted:
            s["topic"] = reflect(strip_det(phrase))
            break
    nouns = [strip_det(p) for p, _, _ in frame.nouns]
    if len(nouns) >= 2:
        s["n1"], s["n2"] = nouns[0], nouns[1]
    i = frames.index(frame)
    s["clause"] = reflect(frame.text)
    if i > 0:
        s["before"] = reflect(frames[i - 1].text)
    return {k: v for k, v in s.items() if v}


# ---------------------------------------------------------------------------
# Canned replies for dialogue acts
# ---------------------------------------------------------------------------

WHY_FOLLOWUP = [
    "I asked because you said \"{prev}\", and that seemed important. What's behind it?",
    "Mostly because of what you said: \"{prev}\". How important does that feel to you?",
]
CLARIFY = [
    "Sorry, I wasn't very clear. I'd like to hear more about you. What's been on your mind today?",
    "Fair question. I was inviting you to keep talking. What's been on your mind lately?",
    "I phrased that badly. Let me ask more simply: how are you feeling right now?",
]
YES = ["Okay.", "Mm-hm.", "I see."]
NO = ["That's okay.", "Alright, no problem.", "Fair enough."]
OPEN = ["What else comes to mind?", "How do you feel about that?",
        "What's that like for you?", "What happened next?",
        "How did that come about?", "What do you make of that?"]
STARTERS = ["What's been taking up most of your time lately?",
            "What was the best part of your week so far?",
            "What's something you're looking forward to?",
            "Who have you been spending the most time with lately?",
            "What's something you've been thinking about more than usual?",
            "How do you usually spend your evenings?",
            "What's something you've been meaning to do but haven't yet?",
            "What did you do last weekend?",
            "What's something that made you laugh recently?",
            "If you could change one thing about this week, what would it be?"]
ACK_POS = ["Glad to hear it.", "That's nice to hear."]
ACK_NEG = ["That sounds hard.", "I'm sorry to hear that.", "That doesn't sound easy."]
FOLLOW_POS = ["What's been going well?", "What's been the highlight so far?",
              "What do you think is behind that?"]
FOLLOW_NEG = ["What's been the hardest part?", "What's weighing on you most?",
              "How long has it been like that?"]
NO_PRESSURE = ["That's okay, we'll find something.", "No problem, I'll pick something.",
               "That's fine, let me ask you something else."]
QUESTION = ["What do you think?", "What got you wondering about that?",
            "What answer would help you most?"]
DETAIL_Q = ["What kind of {d}?", "What goes into the {d}?", "How do you feel about the {d}?",
            "What's the most interesting part of the {d}?"]
REPEATED = ["You said that a moment ago, so it seems important. What makes it stand out for you?",
            "That's the second time you've said that. What's behind it?"]
EMPATHY_PREFIX = {"low": ["That sounds really hard. ", "I'm sorry. ", ""],
                  "positive": ["", "That's great! "], "neutral": [""]}
MEMORY_CALLBACKS = {
    "about": ["Earlier you mentioned your {0} was {1}. How does that connect to this?",
              "You told me your {0} was {1}. How do you feel about that now?"],
    "likes": ["You said you enjoy {0}. How has that been going lately?"],
    "dislikes": ["You mentioned you don't like {0}. How does this relate?"],
    "job": ["How does being a {0} play into this?"],
}
FILLER_WORDS = {"no", "yes", "yeah", "ok", "okay", "well", "um", "uh", "so", "oh", "hmm",
                "honestly", "nah", "yep", "and", "but", "like", "nope", "sure", "not"}
# Pronouns and helper verbs show up in almost every sentence, so they never
# count toward "we've already talked about this".
GENERIC_HITS = {"@self", "@you", "@it", "@they", "@other", "be", "have", "do", "get", "go"}

WH_START = {"what", "who", "when", "where", "how", "why", "which"}
CLOSED_QUESTION = re.compile(
    r"(?:do|does|did|is|are|was|were|have|has|can|could|would|will|should|am)\b", re.I)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

class NeoEliza:
    def __init__(self, seed=None, debug=False, use_slm=True, model=None, lock_path=safety.DEFAULT_LOCK):
        """model: path to an .mdlo file. Without one, the SLM trains from
        corpus.txt and rules.py at startup (about 0.05 seconds).
        lock_path: where the crisis lock is kept (see safety.py)."""
        self.rng = random.Random(seed)
        self.lock = safety.Lock(lock_path)
        self.model_header = None
        if model and use_slm:
            self.writer, self.model_header = mdlo.load(model, seed=seed)
        else:
            self.writer = SLM(seed=seed) if use_slm else None
        self.last_source = None
        self.emotion = EmotionModel()
        self.memory = Memory()
        self.template_uses = Counter()
        self.recent_rules = deque(maxlen=5)
        self.seen = Counter()            # how many turns each word/category has come up
        self.asked = set()               # every sentence Eliza has said
        self.last_bot = ""
        self.last_user = ""
        self.last_template = None
        self.short_streak = 0
        self.starters_left = []
        self.turn = 0
        self.debug = debug
        self.last_rule = None

    # --- template choice ---
    @staticmethod
    def _fill(t, slots):
        try:
            return t.format(**slots)
        except (KeyError, IndexError):
            return t

    def _usable(self, templates, slots):
        return [t for t in templates if all(slots.get(n) for n in re.findall(r"{(\w+)}", t))]

    def pick(self, templates, slots=None):
        """Pick a template whose slots can be filled, avoiding anything already said."""
        slots = slots or {}
        usable = self._usable(templates, slots) or [t for t in templates if "{" not in t] or templates
        if len(usable) > 1:
            usable = [t for t in usable if t != self.last_template] or usable
        usable = [t for t in usable if self._fill(t, slots) not in self.asked] or usable
        weights = [1.0 / (1 + self.template_uses[t]) ** 2 for t in usable]
        t = self.rng.choices(usable, weights)[0]
        self.template_uses[t] += 1
        self.last_template = t
        return self._fill(t, slots)

    def write(self, intent, templates, slots=None):
        """Generated wording for a listening plan, or a template if the SLM
        has nothing that passes its gates."""
        slots = slots or {}
        if self.writer:
            text, info = self.writer.write(intent, slots, avoid=self.asked)
            if text:
                self.last_source = f"slm:{intent} conf={info['confidence']}" + (" new" if info["new_sentence"] else "")
                return text
        self.last_source = f"template:{intent}"
        return self.pick(templates, slots)

    def react(self, mood, templates):
        """A short reaction for the front of a reply ("Aw, that sounds rough.")."""
        if self.writer:
            text = self.writer.react(mood, avoid=self.asked)
            if text:
                return text + " "
        return self.pick(templates)

    def exhausted(self, templates, slots):
        filled = [self._fill(t, slots) for t in self._usable(templates, slots)]
        return not filled or all(f in self.asked for f in filled)

    # --- scoring ---
    def candidates(self, sent, frames):
        mood = self.emotion.label
        main = main_frame(frames)
        out = []
        for rule in PHRASES:
            m = re.search(rule["pattern"], sent)
            if m:
                slots = {k: v for k, v in m.groupdict().items() if v}
                score = rule["weight"] + 4 - 4 * self.recent_rules.count(rule["name"])
                if self.exhausted(rule["say"], slots):
                    score -= 8
                out.append((score, rule, slots))
        for frame in frames:
            for rule in RULES:
                hits = match(rule, frame)
                if hits is None:
                    continue
                slots = slots_for(rule, frame, frames)
                score = rule["weight"] + 2 * len(rule["when"])
                if frame is main:
                    score += 1
                if rule.get("mood"):
                    score += 3 if rule["mood"] == mood else -3
                topical = [h for h in hits if h not in GENERIC_HITS]
                score -= 2.5 * max((min(self.seen[h], 3) for h in topical), default=0)
                score -= 3 * self.recent_rules.count(rule["name"])
                if self.exhausted(rule["say"], slots):
                    score -= 8
                out.append((score, rule, slots))
        out.sort(key=lambda c: -c[0])
        return out

    def detail_followup(self, frames):
        """A new, specific noun phrase (like "coding tasks") worth asking about."""
        best = None
        for f in frames:
            for phrase, head, cat in f.nouns:
                d = strip_det(phrase)
                parts = d.split()
                if cat not in ("@thing", "@activity", "@task") or len(head) < 4:
                    continue
                if any(self.seen[p] or p in STOPWORDS for p in parts):
                    continue
                if cat != "@thing" and len(parts) < 2:
                    continue      # single known words have their own rules
                if best is None or len(d) > len(best):
                    best = d
        return best

    # --- main entry point ---
    def respond(self, text):
        if self.lock.locked:
            self.last_rule = "locked"
            return safety.LOCKED_MESSAGE
        self.turn += 1
        self.last_source = None
        raw = text.strip()
        words = normalize(raw)
        sent = " ".join(words)
        toks = [w for w in words if re.match(r"[a-z0-9':]+$", w)]

        # Crisis: the route crisis professionals advise, then the bot locks
        # for good (see safety.py). Nothing else runs first.
        kind = safety.check(raw, sent)
        if kind:
            self.lock.engage(kind)
            self.last_rule = "crisis-" + kind
            return safety.message_for(kind)

        frames, tagged = parse(words, raw)
        self.emotion.score(tagged)
        new_facts = self.memory.extract(sent, frames)
        self.memory.remember_turn(sent, toks)
        for k, v in new_facts:
            self.memory.pending.append((self.turn, k, v))

        if self.debug:
            print("  words:  " + " ".join(f"{w.text}/{w.pos}" + (f"({w.cat})" if w.cat else "") for w in tagged))
            for f in frames:
                print("  clause: " + f.describe())
            print(f"  mood={self.emotion.label} v={self.emotion.valence:+.2f}")

        if not toks:
            return self._say(self.pick(OPEN), raw)

        for k, v in new_facts:
            if k == "name":
                return self._say(f"Nice to meet you, {v.capitalize()}. What brings you here today?", raw)

        # Questions about the world and requests: reflect them back.
        if (raw.endswith("?") or toks[0] in WH_START) and WORLD_QUESTION.search(sent):
            self.last_rule = "reflect-question"
            topic = question_topic(words)
            return self._say(self.write("reflect-question", REFLECT_QUESTION, {"topic": topic} if topic else {}), raw)
        m = REQUEST.search(sent)
        if m:
            self.last_rule = "reflect-request"
            follow = reflect(m.group("req") or m.group("req2"))
            return self._say(self.write("reflect-request", REFLECT_REQUEST, {"follow": follow}), raw)

        # Dialogue acts
        if (toks in (["what", "do", "you", "mean"], ["huh"], ["what"])
                or sent.startswith(("i do not understand", "i am confused", "what are you talking about"))) and self.last_bot:
            return self._say(self.write("clarify", CLARIFY), raw, keep_last=True)
        if toks in (["why"], ["why", "do", "you", "ask"]) and self.last_user:
            return self._say(self.pick(WHY_FOLLOWUP, {"prev": self.last_user}), raw, keep_last=True)

        content = [t for t in toks if t not in FILLER_WORDS]
        if toks[0] in ("hi", "hello", "hey", "greetings", "yo", "sup"):
            self.short_streak = 0
        else:
            self.short_streak = self.short_streak + 1 if len(content) <= 1 and len(toks) <= 3 else 0
        if self.short_streak >= 2:
            self.short_streak = 0
            return self._say(self.pick(NO_PRESSURE) + " " + self.next_starter(), raw)
        if toks[0] in ("yes", "yeah", "yep", "sure") and len(toks) <= 2:
            return self._say(self.pick(YES) + " " + self.pick(OPEN), raw)
        if toks[0] in ("no", "nope", "nah") and len(toks) <= 2:
            return self._say(self.pick(NO) + " " + self.next_starter(), raw)

        if sum(1 for prev, _ in self.memory.history[:-1] if prev == sent) and len(toks) > 3:
            self._mark_seen(frames)
            return self._say(self.pick(REPEATED), raw)

        cands = self.candidates(sent, frames)
        detail = self.detail_followup(frames)
        self._mark_seen(frames)

        if detail and (not cands or cands[0][0] < 9.5):
            self.last_rule = "detail"
            return self._say(self.write("detail", DETAIL_Q, {"d": detail}), raw)

        if cands and cands[0][0] > 4:
            score, rule, slots = cands[0]
            self.last_rule = rule["name"]
            self.recent_rules.append(rule["name"])
            if self.debug:
                print("  top rules: " + ", ".join(f"{r['name']}={s:.1f}" for s, r, _ in cands[:4]))
            reply = self.write(rule["name"], rule["say"], slots)
            mood = self.emotion.label
            if (rule.get("mood") == mood != "neutral" and self.emotion.last_turn[2]
                    and not re.search(r"[.!] ", reply)):     # no reaction in front yet
                reply = self.react("low" if mood == "low" else "positive", EMPATHY_PREFIX[mood]) + reply
            return self._say(reply, raw)

        self.last_rule = "fallback"
        return self._say(self._fallback(words, toks, frames, raw), raw)

    def _fallback(self, words, toks, frames, raw):
        roll = self.rng.random()
        if raw.endswith("?"):
            topic = question_topic(words)
            if topic:
                self.last_rule = "reflect-question"
                return self.write("reflect-question", REFLECT_QUESTION, {"topic": topic})
            return self.pick(QUESTION)

        # Bring back a remembered fact if it relates to what they just said.
        content = {t for t in toks if t not in STOPWORDS}
        for item in list(self.memory.pending):
            t, key, val = item
            fact_words = set(" ".join(val if isinstance(val, tuple) else (val,)).split())
            if self.turn - t >= 4 and key in MEMORY_CALLBACKS and content & fact_words:
                self.memory.pending.remove(item)
                args = tuple(reflect(a) for a in (val if isinstance(val, tuple) else (val,)))
                return self.rng.choice(MEMORY_CALLBACKS[key]).format(*args)

        prev, sim = self.memory.most_similar(toks)
        if prev and prev != " ".join(words) and len(content) >= 3 and sim > 0.45 and roll < 0.5:
            return f"That reminds me of when you said \"{reflect(prev)}\". How are the two connected?"

        # Reflective listening: mirror, acknowledge the feeling, then ask.
        v_turn, _, hits = self.emotion.last_turn
        ack, follow = "", None
        if hits and v_turn > 0.2:
            ack, follow = self.react("positive", ACK_POS).strip(), self.write("follow-positive", FOLLOW_POS)
        elif hits and v_turn < -0.2:
            ack, follow = self.react("low", ACK_NEG).strip(), self.write("follow-low", FOLLOW_NEG)
        frag = list(toks)
        while frag and frag[0] in FILLER_WORDS | {"just"}:
            frag.pop(0)
        if frag[:2] in (["i", "guess"], ["i", "suppose"], ["i", "mean"]):
            frag = frag[2:]
        if not frag:
            return " ".join(x for x in (ack, follow or self.next_starter()) if x)
        mirror = reflect(" ".join(frag))
        if len(frag) <= 4:
            echo = mirror[0].upper() + mirror[1:] + "."
        elif frag[0] in ("i", "it", "my", "we", "things", "that", "this", "everything", "life"):
            echo = "It sounds like " + mirror + "."
        else:
            echo = ""
        return " ".join(x for x in (echo, ack, follow or self.write("open", OPEN)) if x)

    def _mark_seen(self, frames):
        keys = set()
        for f in frames:
            keys |= {w.text for w in f.words if w.text not in STOPWORDS}
            keys |= {w.lemma for w in f.words if w.pos == "VERB"}
            keys |= f.cats
        for k in keys:
            self.seen[k] += 1

    def next_starter(self):
        """Next unused starter that doesn't echo the question Eliza just asked."""
        if not self.starters_left:
            self.starters_left = STARTERS[:]
            self.rng.shuffle(self.starters_left)
        asked = set(re.findall(r"[a-z]+", self.last_bot.lower())) - STOPWORDS - {"what", "something"}
        for i in range(len(self.starters_left) - 1, -1, -1):
            words = set(re.findall(r"[a-z]+", self.starters_left[i].lower())) - STOPWORDS
            if not words & asked:
                return self.starters_left.pop(i)
        return self.starters_left.pop()

    def _say(self, reply, raw, keep_last=False):
        reply = reply[:1].upper() + reply[1:]
        # Every turn ends with an open-ended question so the conversation keeps moving.
        last = re.split(r"(?<=[.!?])\s+", reply.strip())[-1]
        if not reply.rstrip().endswith("?") or CLOSED_QUESTION.match(last):
            source = self.last_source
            reply = reply.rstrip() + " " + self.write("open", OPEN)
            self.last_source = source
        self.asked.add(reply)
        self.asked.update(re.split(r"(?<=[.!?])\s+", reply.strip()))
        self.last_bot = reply
        if not keep_last:
            self.last_user = raw.strip(" .!")
        return reply


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if "--reset-lock" in sys.argv:
        cleared = safety.Lock().reset()
        print("Crisis lock cleared." if cleared else "No crisis lock was set.")
        return
    if safety.Lock().locked:
        print("NeoEliza:", safety.LOCKED_MESSAGE)
        return
    debug = "--debug" in sys.argv
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv[:-1] else None
    bot = NeoEliza(debug=debug, use_slm="--no-slm" not in sys.argv, model=model)
    if bot.model_header:
        p = bot.model_header["provenance"]
        print(f"(model: {model}, {p['training_sentences']} training sentences, {p['intents']} intents)")
        if mdlo.is_stale(bot.model_header):
            print("(note: corpus.txt or rules.py changed since this model was packed; "
                  "run `python3 mdlo.py pack` to rebuild it)")
    print("NeoEliza: Hello, I'm NeoEliza. How are you feeling today?")
    print("          (type 'quit' to leave, '/memory' to see what I remember)\n")
    while True:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        t = text.strip().lower()
        if t in ("quit", "exit", "bye", "goodbye"):
            name = bot.memory.facts.get("name")
            print(f"NeoEliza: Goodbye{', ' + name[0].capitalize() if name else ''}. Take care of yourself.")
            break
        if t == "/memory":
            print("NeoEliza remembers:")
            for k, vs in bot.memory.facts.items():
                print(f"  {k}: {vs}")
            print(f"  mood: {bot.emotion.label}")
            continue
        bot.last_source = None
        print("NeoEliza:", bot.respond(text))
        if bot.lock.locked:
            break                                   # closed after a crisis message
        if debug:
            print(f"  (rule: {bot.last_rule}, words: {bot.last_source or 'fixed'})")


if __name__ == "__main__":
    main()
