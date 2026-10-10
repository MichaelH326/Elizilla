# Neural Plan: a small model that talks like Elizilla, in any situation

**Research question:** can a small language model learn to hold a conversation
the way Elizilla does (listening, reflecting the person's words back, and
asking open questions) and keep doing it in situations nobody wrote rules for?

Elizilla is a conversation experiment, not a therapist or a support service.
Nothing here changes that.

The result is one `.mdlo` file (MDLO v2, GGUF, like `mira.mdlo`). It runs
without Elizilla's rules, with only `safety.py` in front of it.

---

## 1. What "work like the current Elizilla" means

Every behaviour below exists in `neo_eliza.py` today. The model has to do all
of them, and each one gets its own check in the eval.

| # | Behaviour | Example |
|---|---|---|
| B1 | Every reply ends in **one open-ended question** | never "Do you like it?" |
| B2 | **Repeats the person's own words**, point of view flipped | "i think my boss hates me" → "What makes you think your boss hates you?" |
| B3 | **Short reaction** when the mood calls for it, then the question | "Oof, that's a lot. What's taking up most of your time?" |
| B4 | Follows up on **new details** | "...my coding tasks" → "What kind of coding tasks?" |
| B5 | Handles **mixed feelings** | "it's exciting but I don't have time" → asks about both sides |
| B6 | **Remembers** earlier turns and brings them back | "Earlier you mentioned your job was stressful. How does that connect?" |
| B7 | Uses the person's **name** once they give it | "Nice to meet you, Sam. What brings you here today?" |
| B8 | **One-word answers** ("idk", "nah", "ok"): eases off, offers a new topic | "No problem, I'll pick something. What did you do last weekend?" |
| B9 | **Repeated message**: notices it | "That's the second time you've said that. What's behind it?" |
| B10 | "**what do you mean?**" / "**why?**": clarifies, says why it asked | "I asked because you said..." |
| B11 | **Questions about the world**: reflects, never answers | "So you're asking about the capital of France. What made you think of it?" |
| B12 | **Requests**: reflects, never does them | "You want someone to write your essay for you. What's going on with it?" |
| B13 | **Never** says "I don't know", "I can't", "as an AI", or gives advice | |
| B14 | Questions **about itself**: turns it back to the person | "We were talking about you. What were you about to say?" |
| B15 | **Doesn't repeat itself** within a conversation | |

**Crisis messages are not a model behaviour.** `safety.py` handles them
before the model sees anything, and locks the bot.

## 2. What "adapt to any situation" means, and why it's the hard part

Today Elizilla only understands situations its word lists and rules cover. A
model trained only on Elizilla's replies would learn the same limits: it
can't learn to handle a situation its teacher never showed it.

So "any situation" comes from three things working together:

1. **A pretrained base model.** It already knows English and the world, so it
   can recognize "my sourdough starter died" as a situation even though no
   Elizilla rule mentions bread. That's why the plan fine-tunes a pretrained
   model rather than training one from scratch.
2. **Training data that covers far more situations than Elizilla's rules**
   (see Decision 1 below).
3. **An eval that holds out whole situations**, so "adapts" is measured on
   things the model has never seen, not assumed.

### Situation map

The dataset and the eval both use this map, so coverage is measured, not
guessed. Each area gets many specific situations (e.g. *school* → exams,
group projects, a teacher who's unfair, changing schools, graduation...).

| Everyday life | People | Feelings | Conversation itself |
|---|---|---|---|
| school, work, money, home, moving, travel, food, sleep, health, tech, hobbies, sport, pets, weather, news, chores, plans | family, friends, partners, crushes, breakups, coworkers, bosses, neighbours, strangers, online friends | stress, boredom, excitement, pride, guilt, loneliness, anger, nervousness, nostalgia, mixed feelings | greetings, small talk, one-word answers, sarcasm, jokes, rambling, typos and slang, very long messages, questions about the world, requests, questions about the bot, changing the subject, saying goodbye |

**Held out:** about 15% of specific situations, picked at random from every
area (for example "a pet hamster escaped", "a cancelled flight"). They never
appear in training and are only used to test adaptation.

## 3. Decisions

**Decision 1: where training data for new situations comes from.**

| Option | How | Coverage | Trade-off |
|---|---|---|---|
| **A. Elizilla only** | Elizilla's rules + SLM + "mirror" question frames around the person's own words | Only what the rules and word lists understand, plus generic mirroring | No new tools. The model can't adapt much beyond Elizilla. |
| **B. Elizilla + a larger open model as a second teacher (recommended)** | A bigger open-weight model writes varied situation scripts for the person's side and fills in replies only where Elizilla's rules don't understand. **Every** reply must pass Elizilla's checks or it's thrown away. Full design, teacher choice and CPU budget: [DATA_PLAN.md](DATA_PLAN.md). | Any situation on the map, and phrasings no rule would produce | The big model is used only to build data, never at chat time. No API, no internet at runtime. About 4–5 hours of parallel CI to build. |

B is the only option that can really deliver "adapt to any situation". A
gives a model that is roughly as limited as today's Elizilla.

**Decision 2: base model size.** Start with `SmolLM2-360M-Instruct`
(Apache-2.0, what Mira uses). Multi-turn memory (B6) and odd situations need
more than the smallest model. Try `Qwen2.5-0.5B-Instruct` only if the eval
falls short. The file will be roughly 250–400 MB.

**Decision 3: model alone, or model + rules at chat time?** Recommended:
**model alone** (only `safety.py` in front), so the file is standalone like
Mira. The rule-based Elizilla stays available as `--no-model` and as the
comparison.

## 4. Phase 0: the eval first (`neural/eval_neural.py`)

Built before any data or training, then run on today's Elizilla to get the
baseline.

1. **Single-turn checks** for B1–B5 and B11–B14: about 500 messages spread
   over the situation map.
2. **Multi-turn scripts** for B6–B10 and B15: about 100 short conversations,
   each with a check on a specific turn ("turn 6 must refer back to the job
   mentioned in turn 2").
3. **Held-out situations:** about 200 messages and 40 conversations from the
   held-out list.
4. **Side-by-side rating:** 100 conversations, rated blind (the rater doesn't
   know which system wrote what).

**Targets for shipping:**

| | Target |
|---|---|
| B1, B13 (open question; no "I don't know" or advice) | ≥ 99% |
| B2 (repeats their words) on known situations | ≥ Elizilla's score |
| B3–B12, B14, B15 | each ≥ 90%, and no more than 2 points below Elizilla |
| Same checks on **held-out situations** | ≥ 85% of the model's own known-situation scores |
| Blind side-by-side, situations Elizilla handles | model preferred or tied ≥ 50% |
| Blind side-by-side, **held-out situations** | model preferred ≥ 60% (this is the adaptation result) |

**Gate 0:** the eval runs on today's Elizilla and produces a baseline report.
Wherever Elizilla itself scores badly, fix Elizilla first.

## 5. Phase 1: the dataset (`neural/make_dataset.py`)

Full design: **[DATA_PLAN.md](DATA_PLAN.md)**. In short:

1. **Situation map** (~2,500 situations, 15% held out for the eval, nothing
   crisis-related).
2. **Person side:** the teacher writes ~5,000 situation profiles once; plain
   Python multiplies them into ~20,000 conversations of 4–12 turns, adding
   names, typos, slang and a **behaviour schedule** that puts every behaviour
   B3–B15 into at least 5% of conversations.
3. **Elizilla side:** the **real `NeoEliza` engine** replies turn by turn, so
   memory, names, one-word answers, repeats and "why?" come out exactly as
   Elizilla does them. The teacher fills in only the weak turns (generic
   fallbacks, replies that don't use the person's words), and mirror frames
   back it up.
4. **Checks** (`neural/checks.py`, shared with the eval) on every turn; any
   failing turn drops the whole conversation.
5. **Clean:** dedupe, balance, leakage and crisis checks, all of which fail
   the build on any hit.
6. **Human review** of 200 conversations, repeated until ≤ 5% contain a bad turn.
7. **Output** in Mira's JSONL format, with a validation split by profile and
   a `dataset_card.json`.

**Gate 1:**
- Mira's `validate_dataset.py` passes (English, length, ≥ 15% unique replies);
- every reply passes the automatic checks; zero held-out leakage;
- coverage and behaviour balance within targets;
- your hand rating of 200 random conversations: ≤ 5% contain a bad turn.
  Each bad pattern becomes a filter, then regenerate.

## 6. Phase 2: train and package (GitHub Actions)

A workflow adapted from Mira's `train-mira-voice.yml`, with Mira checked out
at a pinned commit (currently `b4ce45c`):

1. build the dataset and run Gate 1 (stop on failure);
2. fine-tune:
   ```bash
   python mira/voice/finetune_mira.py --base HuggingFaceTB/SmolLM2-360M-Instruct \
       --data neural_data.jsonl --out elizilla-hf --steps 400 --loss_on assistant
   ```
   Loss only on Elizilla's replies. Hold out 5% and stop when its loss rises.
3. package as `elizilla-neural.mdlo` with Elizilla's persona as the system
   prompt (the same text used in training) and `"safety": {"crisis_lock": true}`;
4. run the full eval and upload the model + report as an artifact.

A smoke run (200 conversations, 2 steps) must go end to end in under 15
minutes before any real run.

## 7. Phase 3: compare, fix, repeat

Read the eval report. For each check that misses its target, add or fix data
for exactly that behaviour or situation area, and retrain. Change one thing
per run (data, steps, or base model) so the cause of any change is clear.

## 8. Phase 4: conversations with people

- `neural/run_neural.py`: a small runner that calls `safety.py` on every
  message first, refuses to start while the crisis lock is set, and otherwise
  passes the conversation to the model.
- Short sessions with a handful of volunteers who know it's an experiment,
  half with the model and half with today's Elizilla, without telling them
  which. Log only turn counts and breakdown markers (repeating themselves,
  one-word answers, leaving early), not what they said, unless they agree.

## 9. Phase 5: release and write-up

- Publish `elizilla-neural.mdlo` as a GitHub Release only after every target
  in Phase 0 is met.
- `neural/RESULTS.md`: the model vs today's Elizilla on every behaviour, known
  vs held-out situations, the people sessions, good and bad examples, and
  the honest answer to the research question.

---

## Risks and what catches them

| Risk | What catches it |
|---|---|
| Model only handles what Elizilla handles | Second teacher (Decision 1); held-out situations in the eval |
| Second teacher slips in advice, answers or "as an AI" | Every reply runs through B1/B2/B13 checks; failing conversations dropped |
| Model loses the thread in long conversations | Multi-turn data and multi-turn eval scripts (B6, B15) |
| Model memorizes instead of generalizing | ≥ 15% unique replies; held-out loss; held-out situations |
| Eval looks good because it leaked into training | Automatic leakage check in Gate 1 |
| Crisis handled badly | Never the model's job: `safety.py` first, crisis kept out of training |
| Mira's scripts change | Pinned commit |
| CI time limit | Mira's checkpoint cache resumes training |

## Order of work

| Step | Needs | Where |
|---|---|---|
| Phase 0: eval + baseline on today's Elizilla | Plain Python | This repo |
| Phase 1: situation map, simulated people, dataset | Plain Python; the second teacher runs in CI | This repo + CI |
| Phase 2: training workflow + smoke run | GitHub Actions | CI |
| Phase 3: fix and retrain | CI | CI |
| Phase 4: sessions with people | Volunteers | Locally |
| Phase 5: release + write-up | Everything above | This repo |
