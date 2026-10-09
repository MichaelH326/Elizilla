"""
Lexicon: every word gets a part of speech (what section of a sentence it can
fill) and a meaning category (what it is about).

    "mom"   -> NOUN  @family
    "hate"  -> VERB  @dislike
    "crazy" -> ADJ   @busy
    "i"     -> PRON  @self

Words that aren't listed are guessed from their shape (see guess()).
"""

# --- Parts of speech -------------------------------------------------------

PRONOUNS = {
    "i": "@self", "me": "@self", "myself": "@self",
    "we": "@self", "us": "@self",
    "you": "@you", "yourself": "@you",
    "he": "@other", "him": "@other", "she": "@other", "her": "@other",
    "someone": "@other", "somebody": "@other",
    "they": "@they", "them": "@they",
    "it": "@it", "that": "@it", "this": "@it", "things": "@it", "everything": "@it",
    "everyone": "@everyone", "everybody": "@everyone", "people": "@everyone",
    "nobody": "@nobody", "noone": "@nobody",
    "nothing": "@nothing", "something": "@it", "anything": "@it",
}
POSSESSIVES = {"my": "@self", "our": "@self", "your": "@you", "his": "@other",
               "her": "@other", "their": "@they", "its": "@it"}
DETERMINERS = {"a", "an", "the", "some", "any", "these", "those", "every",
               "each", "another", "no", "all", "both", "much", "many", "more",
               "most", "few", "lot", "lots", "bunch", "couple"}
NEGATIONS = {"not", "never", "no", "hardly", "barely"}
PREPOSITIONS = {"in", "on", "at", "to", "for", "with", "about", "from", "of",
                "into", "by", "over", "after", "before", "during", "around",
                "without", "through", "since", "until"}
CONJUNCTIONS = {"and", "but", "or", "so", "because", "although", "though",
                "while", "however"}
COMPLEMENTIZERS = {"that", "when", "how", "if", "what", "why", "where", "who",
                   "whether"}
WH_WORDS = {"what", "why", "how", "who", "where", "when", "which"}
FILLERS = {"yeah", "yes", "yep", "ok", "okay", "well", "um", "uh", "oh", "hmm",
           "honestly", "lol", "haha", "nah", "nope", "like", "hi", "hello",
           "hey", "anyway", "sure", "basically", "literally"}

# Adverbs: frequency / intensity / hedges
ADVERBS = {
    "always": "@always", "constantly": "@always", "forever": "@always",
    "never": "@never",
    "sometimes": "@freq", "often": "@freq", "usually": "@freq", "rarely": "@freq",
    "really": "@intense", "very": "@intense", "so": "@intense", "extremely": "@intense",
    "super": "@intense", "totally": "@intense", "too": "@intense", "pretty": "@mild",
    "kinda": "@mild", "sorta": "@mild", "somewhat": "@mild", "bit": "@mild",
    "just": "@just", "still": "@still", "actually": "@hedge", "probably": "@hedge",
    "maybe": "@hedge", "also": "@also", "already": "@time", "again": "@time",
    "lately": "@time", "recently": "@time", "now": "@time", "ever": "@time",
    "even": "@intense", "all": "@intense",
}

# Auxiliaries (helper verbs). Each maps to its lemma.
AUX = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
    "be": "be", "been": "be", "being": "be",
    "do": "do", "does": "do", "did": "do",
    "have": "have", "has": "have", "had": "have",
    "will": "will", "would": "will", "can": "can", "could": "can",
    "cannot": "can", "should": "should", "might": "may", "may": "may",
    "must": "must", "shall": "will",
}

# --- Verbs by meaning ------------------------------------------------------

VERB_CATS = {
    "@desire": "want need wish crave long hope",
    "@like": "like love enjoy adore",
    "@dislike": "hate dislike detest despise",
    "@belief": "think believe suppose reckon",
    "@feel": "feel seem",
    "@know": "know understand realize",
    "@have": "have own",
    "@go": "go come leave move travel",
    "@get": "get receive",
    "@do": "do make",
    "@say": "say tell talk speak mention",
    "@remember": "remember recall forget",
    "@dream": "dream",
    "@labor": "work study code program build write practice learn",
    "@play": "play watch read game draw sing paint cook",
    "@fight": "fight argue yell scream",
    "@lose": "lose fail miss quit",
    "@try": "try attempt",
    "@help": "help support",
    "@spend": "spend waste",
    "@see": "see meet visit",
    "@sleep": "sleep rest nap",
    "@worry": "worry stress panic overthink",
    "@cry": "cry",
    "@ignore": "ignore avoid",
    "@stand": "stand",
    "@hurt": "hurt",
    "@care": "care",
    "@death": "die pass",
}

IRREGULAR = {
    "went": "go", "gone": "go", "got": "get", "gotten": "get", "made": "make",
    "done": "do", "thought": "think", "felt": "feel", "knew": "know",
    "known": "know", "said": "say", "told": "tell", "saw": "see", "seen": "see",
    "met": "meet", "lost": "lose", "left": "leave", "came": "come",
    "fought": "fight", "slept": "sleep", "wrote": "write", "written": "write",
    "meant": "mean", "spent": "spend", "built": "build", "understood": "understand",
    "taught": "teach", "bought": "buy", "brought": "bring", "began": "begin",
    "ran": "run", "took": "take", "taken": "take", "gave": "give", "given": "give",
    "found": "find", "dreamt": "dream", "stood": "stand", "drew": "draw",
    "sang": "sing", "forgot": "forget",
}
# Plain verbs with no special meaning category
OTHER_VERBS = """take give find keep start stop begin end finish turn call ask
answer show feel look mean let put run walk drive eat drink buy pay bring sit
stay wait hear listen use happen change open close win teach pick send follow
need show grow live kill join wake cook clean visit break fix share
""".split()

# --- Nouns by meaning ------------------------------------------------------

NOUN_CATS = {
    "@family": """mother mom mum father dad sister brother parents family wife
                  husband son daughter grandma grandpa grandmother grandfather
                  aunt uncle cousin kids children sibling siblings stepdad stepmom""",
    "@friend": "friend friends buddy girlfriend boyfriend partner roommate bestie crush",
    "@boss": """boss coworker coworkers manager teacher teachers professor
                supervisor classmates colleague colleagues""",
    "@work": """job work career office school class classes college university
                internship shift company""",
    "@task": """homework exam exams assignment assignments project projects
                deadline deadlines test tests essay essays studying tasks task
                meeting meetings chores""",
    "@activity": """coding programming guitar music games gaming sports soccer
                    basketball football reading drawing art movies movie tv
                    show shows gym running swimming cooking hiking piano
                    painting writing dancing singing baseball tennis videogames""",
    "@time": """today tomorrow yesterday morning night evening afternoon week
                weekend month year day lately tonight monday tuesday wednesday
                thursday friday saturday sunday summer winter spring fall""",
    "@place": "home house city town room apartment country place outside",
    "@pet": "dog cat pet puppy kitten",
    "@money": "money rent bills debt paycheck",
    "@health": "doctor sick illness hospital therapy therapist medicine",
    "@computer": "computer bot robot ai machine program chatbot",
    "@dreamthing": "dream dreams nightmare nightmares",
    "@timeamt": "time",
    "@name": "name",
}

# --- Adjectives by meaning -------------------------------------------------

ADJ_CATS = {
    "@sad": """sad unhappy depressed miserable down upset hopeless empty
               heartbroken gloomy""",
    "@happy": "happy glad elated excited cheerful joyful thrilled content proud grateful",
    "@anxious": """anxious nervous worried scared afraid stressed overwhelmed
                   panicked tense frightened""",
    "@angry": "angry mad furious annoyed irritated frustrated pissed resentful",
    "@busy": "crazy hectic busy insane wild chaotic nuts intense nonstop stressful",
    "@tired": "tired exhausted sleepy drained burnt worn",
    "@lonely": "lonely alone isolated",
    "@good": """good great fine okay ok nice amazing awesome wonderful fun cool
                exciting interesting better best fantastic perfect alright""",
    "@bad": """bad terrible awful horrible rough hard tough difficult worse worst
               boring rotten broken ruined""",
    "@ordinary": "regular normal ordinary usual typical average same",
    "@confused": "confused lost stuck unsure",
    "@guilty": "guilty ashamed embarrassed",
}

# --- Build the lookup tables -----------------------------------------------


def _split(cats):
    return {w: c for c, words in cats.items() for w in words.split()}


NOUNS = _split(NOUN_CATS)
ADJS = _split(ADJ_CATS)
VERB_LEMMAS = _split(VERB_CATS)
for v in OTHER_VERBS:
    VERB_LEMMAS.setdefault(v, "@act")


def _inflect(lemma):
    """Regular verb forms: likes, liked, liking."""
    forms = {lemma, lemma + "s", lemma + "es"}
    if lemma.endswith("e"):
        forms |= {lemma + "d", lemma[:-1] + "ing"}
    elif lemma.endswith("y") and lemma[-2] not in "aeiou":
        forms |= {lemma[:-1] + "ies", lemma[:-1] + "ied", lemma + "ing"}
    else:
        forms |= {lemma + "ed", lemma + "ing"}
        if len(lemma) <= 4 and lemma[-1] not in "aeiouwxy" and lemma[-2] in "aeiou" and lemma[-3:-2] not in ("a", "e", "i", "o", "u", ""):
            forms |= {lemma + lemma[-1] + "ed", lemma + lemma[-1] + "ing"}
    return forms


VERB_FORMS = {}       # surface form -> lemma
for lemma in VERB_LEMMAS:
    for f in _inflect(lemma):
        VERB_FORMS.setdefault(f, lemma)
for form, lemma in IRREGULAR.items():
    VERB_FORMS[form] = lemma
    VERB_LEMMAS.setdefault(lemma, "@act")

# Every word that has a topic category (used for topic tracking).
WORD_TO_CLASS = {**NOUNS, **ADJS}


def gerund(lemma):
    """like -> liking, get -> getting, play -> playing."""
    for form, lem in VERB_FORMS.items():
        if lem == lemma and form.endswith("ing"):
            return form
    return lemma + "ing"


class Word:
    """A tagged word: surface text, part of speech, meaning category, lemma."""
    __slots__ = ("text", "pos", "cat", "lemma")

    def __init__(self, text, pos, cat=None, lemma=None):
        self.text, self.pos, self.cat, self.lemma = text, pos, cat, lemma or text

    def __repr__(self):
        return f"{self.text}/{self.pos}{'(' + self.cat + ')' if self.cat else ''}"


def lookup(word):
    """All the tags a word could have, most likely first."""
    tags = []
    if word in FILLERS and word not in ("like",):
        tags.append(Word(word, "INTJ"))
    if word in PRONOUNS:
        tags.append(Word(word, "PRON", PRONOUNS[word]))
    if word in POSSESSIVES:
        tags.append(Word(word, "POSS", POSSESSIVES[word]))
    if word in NEGATIONS:
        tags.append(Word(word, "NEG"))
    if word in AUX:
        tags.append(Word(word, "AUX", None, AUX[word]))
    if word in ADJS:
        tags.append(Word(word, "ADJ", ADJS[word]))
    if word in NOUNS:
        tags.append(Word(word, "NOUN", NOUNS[word]))
    if word in VERB_FORMS:
        lemma = VERB_FORMS[word]
        tags.append(Word(word, "VERB", VERB_LEMMAS.get(lemma, "@act"), lemma))
    if word in ADVERBS:
        tags.append(Word(word, "ADV", ADVERBS[word]))
    if word in DETERMINERS:
        tags.append(Word(word, "DET"))
    if word in PREPOSITIONS:
        tags.append(Word(word, "PREP"))
    if word in CONJUNCTIONS:
        tags.append(Word(word, "CONJ"))
    if word in COMPLEMENTIZERS:
        tags.append(Word(word, "COMP"))
    if word == "like":
        tags.append(Word(word, "INTJ"))
    return tags or [guess(word)]


def guess(word):
    """Guess an unknown word's role from its shape."""
    if word.isdigit() or ":" in word:
        return Word(word, "NUM", "@time" if ":" in word else None)
    if word.endswith("ly") and len(word) > 4:
        return Word(word, "ADV")
    if word.endswith(("ful", "ous", "ive", "able", "ible", "less", "ish", "ic")):
        return Word(word, "ADJ", "@desc")
    if word.endswith("ing") and len(word) > 5:
        return Word(word, "NOUN", "@activity")      # may be re-tagged as a verb in context
    if word.endswith("ed") and len(word) > 4:
        return Word(word, "VERB", "@act", word[:-2])
    return Word(word, "NOUN", "@thing")


def all_words():
    """Every word the lexicon knows (used by the spelling corrector)."""
    words = set(PRONOUNS) | set(POSSESSIVES) | DETERMINERS | NEGATIONS | PREPOSITIONS
    words |= CONJUNCTIONS | COMPLEMENTIZERS | FILLERS | set(ADVERBS) | set(AUX)
    words |= set(VERB_FORMS) | set(NOUNS) | set(ADJS)
    return words
