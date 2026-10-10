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
| Small language model (4-gram, Kneser-Ney) that writes every reply from the rule's plan | `slm.py`, `corpus.txt` | **Connected** (`--no-slm` to turn off), tested (`test_slm.py`) |
| Mira-style honesty: can't know time/weather, won't guess facts, can't do tasks | `neo_eliza.py` | Done |
| Packed model in Mira's `.mdlo` container (v2 layout, `"engine": "ngram-kn"`) | `mdlo.py`, `elizilla.mdlo` | Done, tested (`test_mdlo.py`) |

### What the SLM does today
- Trained on about 600 sentences (`corpus.txt` plus every template in `rules.py`), and trains in about 0.05 seconds. A reply takes about 10 ms.
- The rules pick the intent and slots; the SLM writes the wording; templates are the fallback.
- New sentences come from joining two known replies where they share two words, but never inside a question's opening words.
- Gates: cut-off endings, missing verbs, two verbs glued together, double-filled "what" gaps, yes/no questions, joins right before a verb, plus Mira's confidence gate (average and weakest word).
- `eval_slm.py`, 30 tries per intent: every intent produced a passing reply each time; about a third of distinct replies are new wording.
- Bad output still slips through sometimes ("How long have you known your best friend lately?"). More corpus lines fix most of these.

## Roadmap

### Phase 1: Make the SLM good enough to switch on (done)
1. ~~Grow `corpus.txt`~~: 153 → about 440 lines, in Mira's short, warm voice. Still short of 1,000; keep adding, at least 15 per intent.
2. ~~Quality gates~~ plus Mira's confidence gate.
3. ~~`eval_slm.py`~~: % written, % new, % copies, and a rating sheet.
4. ~~Connected to the chat~~ with `--no-slm`.
5. Next: rate `eval_slm_sheet.tsv` by hand and turn the bad patterns into gates or corpus lines.

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

### Phase 5: Neural Elizilla (standalone model)
Distill the rule-based Elizilla into a small fine-tuned model packaged as a
GGUF `.mdlo`, like Mira, that runs without Elizilla's code. Full plan, with
gates, risks and decisions: [NEURAL_PLAN.md](NEURAL_PLAN.md).

### Phase 4 (experiment): Tiny neural reranker
- Train a tiny pure-Python neural model (no numpy) only to *rank* the SLM's candidates, not to generate them.
- Keep it only if it beats the trigram ranking on the Phase 1 eval.

## Principles
- Pure Python standard library only.
- Every reply ends with an open-ended question.
- The hand-written templates stay as the fallback, so quality never drops below today's.
- Crisis language always gets the safety message, no matter which module is active.
