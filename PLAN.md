# Elizilla Plan

Goal: an ELIZA-style listener that holds a real conversation, keeps the person
talking with open-ended questions, and runs anywhere Python runs: no API, no
installs, no GPU. It only listens and talks. It doesn't do tasks.

## Where it stands (October 2026)

| Piece | File | Status |
|---|---|---|
| Word labels: part of speech + category (`@family`, `@dislike`, `@busy`...) | `lexicon.py` | Done |
| Parser: clauses → who / action / not / what / describer / follow-on / when | `sentence.py` | Done, tested (`test_sections.py`) |
| Rules written as sections + categories (about 50) | `rules.py` | Done |
| Engine: scoring, mood, memory, novelty, no-repeat, open-question check | `neo_eliza.py` | Done, tested (`test_open_questions.py`) |
| Small language model (trigram, Kneser-Ney) that writes reply wording | `slm.py`, `corpus.txt` | **Experimental, not connected to the chat yet** |

### What the SLM does today
- Trained on 311 sentences (`corpus.txt` plus every template in `rules.py`), and trains in about 0.03 seconds.
- Topic words are replaced with slots (`<topic>`, `<describer>`...), so it learns reply *shapes*. The parser fills in the person's words.
- It only uses three-word sequences seen in training, so new sentences come from joining two known replies where they overlap.
- Good output: "It sounds like a lot is going on. What's been making them so crazy lately?"
- Bad output still slips through: "What's the coding tasks?", "How do you and your mom?"

## Roadmap

### Phase 1: Make the SLM good enough to switch on
1. Grow `corpus.txt` from about 150 hand-written lines to 1,000 or more, with at least 15 per intent.
2. Add quality gates before a generated reply can be used:
   - each slot must sit in a position seen in training (e.g. `<topic>` after "your")
   - the reply must contain a question word (what / how / when / who)
   - reject sentences cut off mid-phrase ("How do you and your mom?")
3. Write `eval_slm.py`: generate 50 replies per intent and report % valid, % new wording, and % that are exact copies. It should also write a sheet for rating replies by hand.
4. Connect it to the chat with a `--no-slm` flag. Use the SLM's reply only if it passes the gates; otherwise use the hand-written template.

### Phase 2: Modular brain
Turn the engine into a router over plug-in modules, each with one small interface:

```python
class Module:
    def propose(self, context) -> list[Candidate]   # each with a score and a reply
```

- `chat_rules`: today's section-based rules
- `slm_writer`: rewrites the winning rule's reply with the SLM
- `memory`: facts about the person, and callbacks to them
- `small_talk`: starters and handling for low-effort answers

The router picks the best candidate, the same way rules are scored now. A new
behaviour is a new file, not an edit to the engine.

### Phase 3: Learning without an API
- Save memory between sessions (local JSON): the bot greets returning people by name and remembers what they told it.
- Developer feedback commands, `/good` and `/bad`: good replies are added to the corpus and bad ones are blocked, so the SLM improves with use.
- Learn new vocabulary: unknown words that come up repeatedly get offered as category candidates for review.

### Phase 4 (experiment): Tiny neural reranker
- Train a tiny pure-Python neural model (no numpy) only to *rank* the SLM's candidates, not to generate them.
- Keep it only if it beats the trigram ranking on the Phase 1 eval.

## Principles
- Pure Python standard library only.
- Every reply ends with an open-ended question.
- The hand-written templates stay as the fallback, so quality never drops below today's.
- Crisis language always gets the safety message, no matter which module is active.
