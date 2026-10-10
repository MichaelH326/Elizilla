# Neural Plan: an SLM that turns what you said into a question

**Research question:** can a small language model, trained on a laptop CPU or
a free CI runner, listen to what someone says and repeat it back as a good
question, well enough to hold a conversation?

Elizilla is a conversation experiment, not a therapist or a support service.
Nothing here changes that.

---

## 1. The task, defined exactly

**Input:** what the person just said (plus, in later stages, the last two
turns).

**Output:** one short, open-ended question that **repeats the person's own
words**, with the point of view flipped (I → you, my → your), and adds
nothing of its own: no answers, no advice, no "I don't know".

| Person says | Good output | Why |
|---|---|---|
| i dont have time to do my homework | What's keeping you from having time to do your homework? | Their words, flipped, open question |
| my mom is always on my case | What makes it feel like your mom is always on your case? | Repeats the whole clause |
| i think my boss hates me | What makes you think your boss hates you? | |
| i love playing guitar | What do you love most about playing guitar? | |
| im so stressed about my exams | What about your exams is making you so stressed? | |
| what is the capital of france? | What made you think about the capital of France? | Reflected, not answered |

| Bad output | Problem |
|---|---|
| Do you not have time for homework? | Yes/no question; conversation stops at "yes" |
| Why don't you make a schedule? | Advice |
| What's keeping you from having time to do my homework? | Point of view not flipped |
| How is school going? | Doesn't repeat what they said |
| Paris. What made you ask? | Answered the question |

This narrow task is the heart of Elizilla, and it's mostly **copying with a
twist** (flip pronouns, wrap in a question frame). That makes it a fair test
of what a *very* small model can learn.

## 2. Two models, one dataset, one eval

The research is more useful if it compares sizes, so the plan trains two
models on the same data and scores them with the same eval:

| | Model A: tiny, from scratch | Model B: small, fine-tuned |
|---|---|---|
| Recipe | Mira v1 (`mira/train.py`): character-level transformer trained from nothing | Mira v2 (`voice/finetune_mira.py`): LoRA fine-tune of a pretrained model |
| Size | ~0.8M–2.7M parameters (`tiny`/`small` presets) | `SmolLM2-135M-Instruct` first, `SmolLM2-360M-Instruct` if needed |
| Knows English beforehand | No: everything comes from our data | Yes |
| Training | ~40 min–3 h on a laptop CPU | A few hours on a GitHub Actions CPU runner |
| File | MDLO v1, about 3–11 MB | MDLO v2 GGUF, about 100–400 MB |
| Runs with | PyTorch (`mira/assistant.py`) | llama.cpp / Ollama / `run_mira.py` |
| Answers | Can a model *this* small do it at all? | How good does it get with real English underneath? |

Both are compared against a third, free baseline: **the rule-based mirror**
(Elizilla's parser + `reflect()` + a question frame, no learning at all). If
neither model beats the rules on new phrasings, that's a result too.

Why character-level for Model A: copying an unfamiliar word ("my
*ukulele* teacher") is spelling it out letter by letter, so a char model can
in principle copy words it has never seen. That's exactly the
generalization this research is about.

## 3. Phase 0: the eval comes first

Write `reflect/eval_reflect.py` and `reflect/eval_set.jsonl` (about 400
hand-written messages) **before** any training, and run it on the
rule-based mirror to get baseline numbers.

| # | Metric | How it's measured | Target |
|---|---|---|---|
| R1 | Open question | ends in "?", one question, not yes/no | ≥ 98% |
| R2 | Repeats their words | share of the message's content words (reflected) found in the output | mean ≥ 0.6 |
| R3 | Point of view flipped | no "I / my / me" carried over from the person's side | ≥ 97% |
| R4 | Grammatical | hand-rated sample of 100 | ≥ 90% |
| R5 | Adds nothing | no answers, advice, or "I don't know / I can't" | ≥ 99% |
| R6 | **New words** | R2 + R3 on messages full of words in no training example (rare hobbies, names, slang) | ≥ 85% of R2/R3 scores on known words |
| R7 | **Holds a conversation** | sessions with real people: median turns before they stop, % of turns where they had to repeat themselves | Reported, compared across models |

R6 and R7 answer the research question. R1–R5 make sure the outputs are the
right kind of thing.

The eval set is split so its messages, and the special words used in R6,
**never** appear in training (checked automatically).

**Gate 0:** eval runs on the rule-based mirror and produces a baseline report.

## 4. Phase 1: the dataset (`reflect/make_pairs.py`, plain Python)

Training pairs: *(what the person said → reflected question)*.

1. **Messages.** Build 30,000+ distinct messages: sentence frames × every word
   in `lexicon.py` × casual variation (lowercase, missing apostrophes, typos,
   slang, run-ons, filler like "idk", "honestly"). Hand-write ~300 more that
   sound like real people, to check the frames aren't too uniform.
2. **Targets, three sources, all checked:**
   - **Mirror frames** (always correct): flip the clause with `reflect()` and
     wrap it: "What makes you say {clause}?", "What's it like when
     {clause}?", "How long have you felt that {clause}?". There are about 30
     frames, chosen by clause type (feeling, wanting, liking, a person doing
     something, a question about the world).
   - **Elizilla's rules + SLM**, kept only when the output passes R1, R3, R5
     and copies enough (R2 ≥ 0.6).
   - **Hand-written** targets for the ~300 hand-written messages.
3. **2–3 targets per message**, so the model learns a style, not one fixed
   sentence per input.
4. **Formats**, both from the same pairs:
   - Model A: Mira v1 stream: `<|u|>message<|m|>question<|e|>`.
   - Model B: Mira v2 JSONL: `{"messages": [system, user, assistant]}`.
5. **Kept out:** anything about suicide, self-harm or harming others (that
   goes through `safety.py`, never the model); every eval message and R6 word.
6. **`dataset_card.json`:** counts per frame and source, the seed, and the
   SHA-256 of `lexicon.py`, `rules.py`, `corpus.txt` and the generator.

**Gate 1:**
- every target passes R1, R3, R5 automatically;
- ≥ 15% unique targets (Mira's `validate_dataset.py` rule);
- zero overlap with the eval set;
- hand rating of 200 random pairs: ≤ 5% bad. Each bad pattern becomes a
  filter, then regenerate.

## 5. Phase 2: train Model A (tiny, from scratch)

```bash
python -m mira.train --data reflect_stream.txt --preset tiny \
    --max-iters 4000 --out checkpoints/reflect-tiny
```

- Start with `tiny` (~0.8M); try `small` (~2.7M) only if R2/R3 fall short.
- Keep 5% of the pairs as a validation set; stop when validation loss stops
  improving (memorizing).
- Greedy decoding plus Mira's confidence gate: a low-confidence reply falls
  back to the rule-based mirror, and the fallback rate is reported.
- Package as MDLO v1 (raw tensors + char tokenizer).

**Gate 2:** eval report for Model A next to the baseline.

## 6. Phase 3: train Model B (fine-tuned)

```bash
python mira/voice/finetune_mira.py --base HuggingFaceTB/SmolLM2-135M-Instruct \
    --data reflect_pairs.jsonl --out reflect-hf --steps 400 --loss_on assistant
python mira/voice/package_mdlo.py --model reflect-hf --out elizilla-reflect.mdlo \
    --quant Q8_0 --system_prompt "<persona from mdlo.py>"
```

- 135M first: it's the smallest model that knows English, and the task is
  narrow. Move to 360M only if Gate 3 fails.
- Same system prompt for training and packaging.
- A GitHub Actions workflow, adapted from Mira's `train-mira-voice.yml`,
  checks out Mira at a pinned commit (currently `b4ce45c`), builds the data,
  runs Gate 1, trains, packages and runs the eval. It uploads the result as
  an artifact; it only publishes a release after Gate 3.

**Gate 3:** eval report for Model B next to Model A and the baseline.

## 7. Phase 4: conversations with people (R7)

1. Wire both models into the chat as `neo_eliza.py --reflect-model FILE`: the
   rules still pick *when* to reflect, and the model writes the question.
   `safety.py` runs first on every message, as it does now.
2. Run short sessions (10–15 minutes) with a handful of people who know it's
   an experiment. Log only turn counts and breakdown markers, not what they
   said, unless they explicitly agree.
3. Report median turns per session, % of turns where the person repeated
   themselves or gave a one-word answer, and how often the model's question
   was rated "made sense" (asked after the session).

## 8. Phase 5: write it up

`reflect/RESULTS.md`: all three systems side by side on R1–R7, examples of
good and bad outputs, what failed and why, and the honest answer to the
research question.

---

## Safety (fixed, not optional)

- `safety.py` checks every message **before** any model sees it. A crisis
  message gets the advised route and locks the bot. This works the same
  with or without a model.
- No crisis content in any training data, so no model learns a weaker version.
- Every `.mdlo` header declares `"safety": {"crisis_lock": true}`. A runner
  that can't honour it must refuse to load the file.

## Risks and what catches them

| Risk | What catches it |
|---|---|
| Model memorizes frames, can't copy new words | R6 with held-out words; char-level Model A; validation-loss stopping |
| Model drops or garbles their words | R2 and R3 in every eval; hand rating |
| Model B's base habits leak (answers, advice, "as an AI") | R5; examples of reflected questions about the world in the data |
| Eval leaks into training | Automatic overlap check in Gate 1 |
| Rule-generated targets have mistakes | Mirror frames are correct by construction; other sources filtered; hand rating |
| Mira's scripts change | Pinned commit |
| CI time limit | Mira's checkpoint cache resumes training |
| Results look better than they are | Same eval for all three systems; baseline reported; R7 with real people |

## Order of work

| Step | Needs | Where |
|---|---|---|
| Phase 0: eval + baseline | Plain Python | This repo |
| Phase 1: dataset + Gate 1 | Plain Python + your 200-pair rating | This repo |
| Phase 2: Model A | PyTorch, laptop CPU | Locally or CI |
| Phase 3: Model B | GitHub Actions | CI |
| Phase 4: sessions with people | Both models + a few volunteers | Locally |
| Phase 5: write-up | Results from all phases | This repo |

Phases 0 and 1 are most of the work and need no installs. Start there.
