# Elizilla

NeoEliza, the listening chatbot behind Elizilla. See [PLAN.md](PLAN.md) for the roadmap.

An ELIZA-style chatbot in pure Python (no dependencies). Instead of matching
fixed text patterns, it breaks every sentence into **sections** and labels every
word with a **category**, then picks rules based on those.

```bash
python3 neo_eliza.py          # chat
python3 neo_eliza.py --debug  # show word tags, sections, and the winning rule
```

Type `/memory` in the chat to see what it remembers, `quit` to leave.

## How a message is understood

```
"its exciting but i dont have the time to do my homework"

words:   it/PRON(@it) is/AUX exciting/ADJ(@good) but/CONJ i/PRON(@self) do/AUX not/NEG
         have/VERB(@have) the/DET time/NOUN(@timeamt) to/INF do/VERB my/POSS homework/NOUN(@task)

clause 1: who=it  action=be  describer=exciting(@good)
clause 2: [but] who=i(@self)  action=have  NOT  what=time(@timeamt)  follow-on=to:'do my homework'

rule "no-time" matches clause 2:
  who=@self, action=@have, not=True, what=@timeamt
reply: "When do you usually try to do your homework, and what gets in the way?"
```

| File | Job |
|---|---|
| `lexicon.py` | Word lists: part of speech + category (`@family`, `@dislike`, `@busy`...). Guesses unknown words from their endings |
| `sentence.py` | Normalizing, spelling correction, tagging each word using its neighbours, splitting into clauses, and filling the sections: **who / action / not / what / describer / follow-on / when** |
| `rules.py` | Rules written as sections + categories, plus a few fixed idioms ("what's on your mind") |
| `neo_eliza.py` | Scoring, mood, memory, dialogue handling, and the chat loop |
| `slm.py`, `corpus.txt` | Experimental small language model (trigram) for writing reply wording. Not connected yet: `python3 slm.py sample sad describer=lonely` |

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

## Tests

```bash
python3 test_sections.py        # sentences split into the right sections
python3 test_open_questions.py  # every reply ends with an open-ended question
```
