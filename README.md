# Elizilla

NeoEliza, the chatbot behind Elizilla. See [PLAN.md](PLAN.md) for the roadmap.

**What this is:** an experiment to see whether a small language model (SLM),
with no API, no GPU and no installs, can have and hold a conversation with a
person. It keeps the person talking by reflecting what they say and asking
open-ended questions.

**What this isn't:** a therapist, a counsellor, a support service, or a
source of advice. It doesn't understand people the way a person does, and
it shouldn't be used as if it did. The crisis handling below is there only
because anyone might type something serious into a chat box.

An ELIZA-style chatbot in pure Python (no dependencies). Instead of matching
fixed text patterns, it breaks every sentence into **sections** and labels every
word with a **category**, then picks rules based on those.

The rules decide **what to listen for**. A small language model (`slm.py`)
decides **how to say it**: it writes the reply instead of picking a prebuilt
one. Like [Mira](https://github.com/MichaelH326/Mira), it would rather not
answer than answer badly. Generated replies have to pass grammar and
confidence gates, and if none do, a hand-written template is used.

It only listens. It never answers questions, does tasks, or says "I don't
know": it reflects them back ("what is the capital of france?" gets "So
you're asking about the capital of France. What made you think of it?").

```bash
python3 neo_eliza.py           # chat
python3 neo_eliza.py --debug   # show word tags, sections, the winning rule, and where the wording came from
python3 neo_eliza.py --no-slm  # hand-written templates only
python3 neo_eliza.py --model elizilla.mdlo   # load the packed model instead of training at startup
```

Type `/memory` in the chat to see what it remembers, `quit` to leave.

## Crisis safety

Elizilla is a conversation experiment, not a support tool, so it never tries
to handle a crisis itself. Any message about suicide, self-harm or harming others skips everything else.
Elizilla answers with the route crisis professionals advise: it says it cares,
points to a real person right away (988 call or text, Crisis Text Line at
741741, 911 or the emergency room if anyone is in danger, findahelpline.com
outside the US), and encourages reaching out to someone trusted. It doesn't
ask a listening question or try to counsel.

Then it **locks for good**: every later message, and every later start, gets
only the help resources. The lock is a file (`~/.elizilla/crisis.lock`) that
records only when and which kind of message triggered it, never what was
said. A person has to clear it on purpose:

```bash
python3 neo_eliza.py --reset-lock
```

The detection errs on the side of locking, so a figure of speech like "I could
kill him" locks it too. The check and the lock live in `safety.py`, plain
Python with no other dependencies, so any runner (including a future neural
one, see [NEURAL_PLAN.md](NEURAL_PLAN.md)) uses the same ones.

## How a message is understood

```
"its exciting but i dont have the time to do my homework"

words:   it/PRON(@it) is/AUX exciting/ADJ(@good) but/CONJ i/PRON(@self) do/AUX not/NEG
         have/VERB(@have) the/DET time/NOUN(@timeamt) to/INF do/VERB my/POSS homework/NOUN(@task)

clause 1: who=it  action=be  describer=exciting(@good)
clause 2: [but] who=i(@self)  action=have  NOT  what=time(@timeamt)  follow-on=to:'do my homework'

rule "no-time" matches clause 2:
  who=@self, action=@have, not=True, what=@timeamt
plan:  intent=no-time  follow="do your homework"
SLM:   "What's been eating up the time you'd need to do your homework?"
```

| File | Job |
|---|---|
| `lexicon.py` | Word lists: part of speech + category (`@family`, `@dislike`, `@busy`...). Guesses unknown words from their endings |
| `sentence.py` | Normalizing, spelling correction, tagging each word using its neighbours, splitting into clauses, and filling the sections: **who / action / not / what / describer / follow-on / when** |
| `rules.py` | Rules written as sections + categories, plus a few fixed idioms ("what's on your mind") |
| `neo_eliza.py` | Scoring, mood, memory, dialogue handling, and the chat loop |
| `slm.py`, `corpus.txt` | Small language model (4-gram, Kneser-Ney) that writes each reply from the rule's plan, with grammar and confidence gates. Try it alone: `python3 slm.py sample sad describer=lonely` |
| `mdlo.py`, `elizilla.mdlo` | Packs the trained SLM into one `.mdlo` file, the same container Mira uses, and loads it back |
| `safety.py` | Crisis detection, the advised crisis replies, and the lock |
| `eval_slm.py` | Measures the SLM per intent (% written, % new wording, % copies) and writes a sheet for rating replies by hand |

## Writing a rule

```python
dict(name="hate-when", weight=8,
     when=dict(who="@self", action="@dislike", follow="when"),
     say=["What bothers you most when {follow}?",
          "How often does that happen?"]),
```

This matches "I hate it when no one shows up", "I really can't stand when people
are late", and so on, because it looks at the sentence's structure rather than
its exact words.

## Scoring

Every rule that matches any clause is scored:

- base weight + 2 per section it requires (more specific wins)
- +1 if it's the main clause (the part after "but", or the most complete one)
- ±3 if the rule's mood fits the person's current mood
- minus a penalty for topics already discussed (novelty)
- minus a penalty for rules used recently or whose replies are all used up

New details ("coding tasks") can outrank weak rules, and every reply ends with
an open-ended question.

## How a reply is written

The winning rule is a plan: an intent (its name, like `no-time`) plus the
person's words as slots. The SLM is trained on `corpus.txt` plus every
template in `rules.py`, with the person's words replaced by slots
(`<follow>`, `<topic>`...), so it learns the *shapes* of listening replies.

1. **Generate.** Walk a 4-gram chain for that intent (mixed with a global
   model). Now and then a step uses only the last two words, which lets the
   walk cross into a different known reply. That is where new wording comes
   from. The opening words of a question always come from one reply, so its
   grammatical frame stays intact.
2. **Gate.** Reject candidates that are cut off ("How do you and your
   mom?"), have no verb, glue two verbs together, fill the "what" gap twice,
   are yes/no questions, end in a way no training reply ends, or join two
   replies right before a verb.
3. **Confidence gate (from Mira).** Reject replies whose average word
   probability, or weakest word, is too unlikely.
4. **Rank** the rest by fluency, use of the person's words and new wording,
   and avoid anything already said. If nothing passes, use a template.

A short reaction ("Aw, that sounds rough.") is generated the same way and put
in front when the rule's mood matches the person's mood.

```bash
python3 eval_slm.py        # per-intent report + eval_slm_sheet.tsv
```

## The .mdlo file

`elizilla.mdlo` is the trained SLM in one file, in the same container as
[Mira's](https://github.com/MichaelH326/Mira/blob/main/MDLO_SPEC.md) `mira.mdlo`:

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `MDLO` |
| 4 | 4 | version, uint32 little-endian: `2` |
| 8 | 8 | header length, uint64 little-endian |
| 16 | header length | JSON header |
| after the header | rest of file | payload, checked against `payload_sha256` |

The header has the same fields as Mira's v2 header (`format`, `format_version`,
`model_name`, `engine`, `payload_sha256`, `payload_bytes`, `chat`, `provenance`).
The difference is the engine: Mira's payload is a GGUF neural network
(`"engine": "gguf"`). Elizilla's is the n-gram model as gzipped JSON
(`"engine": "ngram-kn"`), so reading it needs nothing but Python.
`chat.system_prompt` describes the persona the rules and corpus were written
for, and `chat` holds the generation settings the model loads with.

```bash
python3 mdlo.py pack                  # rebuild elizilla.mdlo (do this after editing corpus.txt or rules.py)
python3 mdlo.py info elizilla.mdlo    # print the header
```

`provenance.training_sha256` records which `corpus.txt` and `rules.py` the
file was built from. The chat warns when the file is out of date, and
`test_mdlo.py` fails until you repack.

## Tests

```bash
python3 test_sections.py        # sentences split into the right sections
python3 test_open_questions.py  # every reply ends with an open-ended question
python3 test_slm.py             # SLM gates, generated replies, reflected questions
python3 test_mdlo.py            # .mdlo layout, checksum, round trip, and that elizilla.mdlo is up to date
python3 test_safety.py          # crisis messages caught, ordinary ones not, the lock holds
```
