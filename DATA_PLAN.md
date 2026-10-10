# Training Data Plan

How to build the conversations the neural Elizilla learns from
([NEURAL_PLAN.md](NEURAL_PLAN.md), Phase 1): data that teaches **everything
Elizilla does** (B1–B15) across **far more situations than its rules cover**,
with no bad turns, no leakage into the eval, and no crisis content.

Elizilla is a conversation experiment, not a therapist. The data is built to
teach conversation, not support.

---

## The core idea: split the job by who is best at it

A conversation has two sides, and they have different problems:

| Side | What's hard | Who writes it |
|---|---|---|
| **The person** (user turns) | Variety: thousands of situations, voices, typos, moods, arcs | A bigger open model writes situation scripts once; cheap code multiplies them |
| **Elizilla** (assistant turns) | Behaving exactly like Elizilla, every turn | **The real Elizilla engine** replies turn by turn; the bigger model fills in only where Elizilla's rules don't understand, and every reply is checked |

Running the real `NeoEliza` engine through each conversation means memory
callbacks, name use, one-word-answer handling, repeat detection, "why?"
answers and the rest (B6–B10, B15) come out exactly as Elizilla does them, for
free. The bigger model is only needed for what Elizilla can't do: understand
new situations.

```
situation map ──► teacher writes        ──► code multiplies        ──► simulated person
(areas, ~2,500       ~5,000 profiles +        (names, typos, slang,      plays the script,
 situations)          user lines               behaviour schedule)        turn by turn
                                                                              │
                                                                              ▼
            filters + checks ◄── teacher fills weak turns ◄── NeoEliza engine replies
                   │
                   ▼
   dedupe ► balance ► leakage check ► human review ► neural_data.jsonl + dataset_card.json
```

---

## Step 1: the situation map (`neural/situations.json`)

A hand-curated tree: **area → specific situation**, with tags.

```json
{"id": "school/exam-tomorrow", "area": "everyday/school",
 "moods": ["anxious", "tired"], "people": ["teacher", "friend"],
 "held_out": false}
```

- About 12 areas × ~200 situations each, roughly 2,500 situations in total.
  Draft with the teacher model, then edit by hand: delete duplicates, add
  missing everyday ones, remove anything crisis-related.
- **Held out:** 15% of situations in every area, chosen at random with a fixed
  seed and marked `held_out: true`. Nothing from them is ever generated for
  training. They're used only by the eval.
- **Crisis-adjacent situations are excluded** (self-harm, suicide, abuse,
  violence). Those are `safety.py`'s job, not the model's.

## Step 2: user side

### 2a. Situation profiles (teacher, once per profile)

For each training situation, the teacher writes **2 profiles**: a short JSON
description of one person in that situation, plus their lines for a
conversation.

```json
{"situation": "school/exam-tomorrow",
 "person": {"age_band": "teen", "voice": "casual, lowercase, short"},
 "facts": ["chemistry exam tomorrow", "studied with a friend named Priya",
           "mom expects good grades"],
 "mood_arc": ["anxious", "frustrated", "a bit calmer"],
 "lines": ["ugh chem exam tomorrow", "i studied but it didnt stick",
           "priya says im overthinking", "my mom is gonna freak if i fail",
           "idk maybe ill just sleep early", ...]}
```

About 5,000 profiles × ~250 output tokens ≈ **1.25M teacher tokens**.

### 2b. Multiply in code (no teacher calls)

Each profile becomes about 4 conversations by varying, in plain Python:

| Variation | Examples |
|---|---|
| Names and details | Priya → Sam, chemistry → history, mom → dad |
| Voice | lowercase, missing apostrophes, common typos (Elizilla's spelling corrector, in reverse), slang ("ngl", "tbh"), run-ons |
| Length | trim to 3 lines, or merge two lines into one long message |
| **Behaviour schedule** | Insert, at chosen turns: giving a name, one-word answers ("idk", "nah"), repeating a line, "why?", "what do you mean?", a fact question, a request, a question about the bot, a topic change, coming back to an earlier fact, "bye" |

The schedule guarantees every behaviour B3–B15 is exercised: each one is
placed in at least 5% of conversations.

Result: **~20,000 conversations of 4–12 turns** from ~5,000 profiles.

### 2c. Extra user-side sources

| Source | Use | Notes |
|---|---|---|
| Elizilla's sentence frames × `lexicon.py` | Single opening messages | Free; covers what the rules know |
| ~300 hand-written messages | Real-sounding people, checks the teacher isn't too uniform | Write them yourself or with friends |
| An open dialogue corpus such as Ai2's **SODA** | Person-side lines only, for more voices | Verify the licence first (CC BY 4.0 at last check); keep only person turns, never its replies; run `safety.check` and drop crisis content |

## Step 3: Elizilla side

For each conversation, run a **fresh `NeoEliza`** (fixed seed, crisis lock
pointed at a temporary file) through the person's lines, turn by turn. For
each turn, keep or replace its reply:

| Engine result | What happens |
|---|---|
| A specific rule matched (sad, no-time, family-who, detail, reflect-question...) and the reply passes the checks | **Keep** (most turns) |
| Generic fallback ("What else comes to mind?") **or** a rule matched but the reply doesn't use the person's words (B2 below threshold) | **Teacher writes 3 candidates** (below); keep the best one that passes the checks, otherwise keep a mirror-frame reply |
| Dialogue acts (name, one-word streak, repeat, why, clarify) | **Keep**: this is Elizilla's own handling, which the model must copy exactly |
| `safety.check` fires on any person line | **Drop the whole conversation** (shouldn't happen after Step 1; this is a backstop) |

Expect the teacher to fill roughly **30–40%** of turns: ~60,000 turns × 3
candidates × ~30 tokens ≈ **5.4M tokens** at most. Cut it to ~1.8M by asking
for the 3 candidates in one call (one shared prompt).

### The teacher prompt (one per weak turn)

A fixed prefix (persona, rules, 12 real Elizilla exchanges as examples)
followed by the conversation so far. The fixed prefix lets llama.cpp reuse it
across calls, so only the conversation part costs time.

```
You are writing replies for NeoEliza, a listener in a conversation experiment.
Rules: 1-2 short spoken sentences. At most a short reaction, then ONE open
question that repeats the person's own words with "I/my" turned into
"you/your". Never answer questions, give advice, or say you don't know or
can't. Never mention being an AI.
[12 example exchanges taken from real Elizilla output]
Conversation so far:
...
Write 3 different possible replies, one per line.
```

### Mirror frames (always-correct backup)

When no teacher candidate passes, use Elizilla's own reflection: flip the
person's clause with `reflect()` and wrap it in one of ~30 question frames
chosen by clause type ("What makes you say {clause}?", "What's it like when
{clause}?"). It's grammatical by construction, so no turn is ever left empty.

## Step 4: checks every Elizilla turn must pass

All automatic, in `neural/checks.py`. **The eval uses the exact same file**,
so the data and the scoring can't disagree.

| Check | Rule |
|---|---|
| B1 open question | ends with one "?", the last sentence isn't yes/no (Elizilla's `CLOSED_QUESTION`) |
| B2 uses their words | ≥ 50% of the person's content words (reflected) appear, or ≥ 1 for very short messages |
| Point of view | no "I/my/me" referring to the person (checked against the reflected words) |
| B13 banned | none of: "I don't know", "I can't", "I'm not able", "as an AI", "language model", "you should", "try to", "have you considered", "I recommend" |
| Length | ≤ 2 sentences, ≤ 35 words, no lists, markdown or emoji |
| Grammar | passes `slm.py`'s grammar gates (cut-off endings, glued verbs, missing verbs) |
| B15 no repeats | not identical to any earlier Elizilla turn in the conversation |
| Answers nothing | for B11 turns: no answer to the fact question (simple check: the reply contains no numbers or proper nouns that weren't in the person's message) |

**Rule:** any failing turn the teacher can't fix drops the whole
conversation. A conversation is never "mostly good".

## Step 5: clean, balance, guard

1. **Dedupe:** exact duplicates removed. Near-duplicates (≥ 0.8 word-trigram
   overlap) thinned to at most 3 copies. Plain-Python MinHash.
2. **Unique replies:** ≥ 15% of Elizilla turns must be unique across the set
   (Mira's `validate_dataset.py` rule). Below that, the model memorizes.
3. **Balance:** each area within ±20% of its target share. Each behaviour
   B3–B15 in ≥ 5% of conversations. Teacher-written turns ≤ 40% of all
   Elizilla turns, so the style stays Elizilla's.
4. **Leakage:** no conversation from a held-out situation. No person line
   within 0.8 overlap of any eval message. Checked automatically; the build
   fails on any hit.
5. **Crisis:** `safety.check` on every line of the final file must find
   nothing. The build fails on any hit.

## Step 6: human review (the part code can't do)

- A simple rating sheet (`neural/review.tsv`, or a small web page) shows 200
  random conversations, stratified by area and by who wrote each turn
  (Elizilla / teacher / mirror).
- You mark each turn **good / ok / bad**, with a short note on bad ones.
- **Pass:** ≤ 5% of conversations contain a bad turn, and teacher-written
  turns are rated no worse than Elizilla's own.
- Every "bad" note becomes a new check in `checks.py` or a fix to Elizilla,
  then the data is rebuilt. Repeat until it passes.

About an hour per round. Plan for two or three rounds.

## Step 7: output

- `neural_data.jsonl`: Mira's format, one conversation per line:
  `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]}`.
  The system prompt is the persona in `mdlo.py`, word for word.
- `neural_data_val.jsonl`: 5%, split **by profile** (never the same person
  in both).
- `dataset_card.json`: counts per area, situation, behaviour and source;
  % teacher-written; check failure counts; review results; seeds; the
  teacher model and its file hash; SHA-256 of every input (`situations.json`,
  `lexicon.py`, `rules.py`, `corpus.txt`, the generator).

Same inputs and seeds → the same file. Any change shows up in the card.

---

## Choosing and running the teacher

**Requirements:** open weights, a licence that allows using its outputs to
train another model, runs in llama.cpp on a CPU.

| Candidate | Size | Licence (check before use) | Notes |
|---|---|---|---|
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | MIT | Good writing for its size; recommended starting point |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Apache-2.0 | Better, about twice as slow |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | Apache-2.0 | Fast fallback if time is tight |

**Step 0 is a benchmark, not a guess.** Run 50 profile calls and 200 reply
calls on a GitHub Actions runner with each candidate. Record tokens per
second and the share of replies that pass the checks. Pick the teacher with
the best passing replies per hour.

**Budget:** about 1.25M tokens (profiles) + ~1.8M (replies) ≈ **3M tokens**.
At an assumed ~8–12 tokens per second for a 3.8B model on a 4-core runner
(to be measured in Step 0), that's ~70–100 runner-hours. Spread across a
**matrix of 20 parallel jobs** (each takes a slice of the situations), it's
about **4–5 hours** of wall-clock time, inside one workflow run. Public
repositories get these runners free. Each job writes its slice; a final job
merges, cleans, balances and checks.

If the benchmark is slower than assumed: use the 1.7B teacher for profiles
(simpler job) and the 3.8B one for replies only, or reduce to 3,000 profiles.

## Where it lives

```
neural/
  situations.json      # Step 1, hand-edited
  profiles.py          # Step 2a: teacher → profiles (CI)
  simulate.py          # Step 2b: multiply + behaviour schedule
  build.py             # Step 3: run NeoEliza, fill weak turns, mirror backup
  checks.py            # Step 4: shared with the eval
  clean.py             # Step 5: dedupe, balance, leakage, crisis
  review.py            # Step 6: makes and reads the rating sheet
  make_dataset.py      # runs 2b → 7 end to end
.github/workflows/build-data.yml   # benchmark, matrix generation, merge
```

Everything except the teacher calls is plain Python and runs locally in
minutes, so it can be built and tested before spending any CI time.

## If you'd rather not use a teacher model

The same pipeline works without one: hand-write situation profiles (aim for
500+), let Step 2b multiply them, and use mirror frames wherever Elizilla's
rules don't understand. The data is smaller and plainer, and the model will
adapt much less to new situations. The eval will show how much less.

## Order of work

1. `checks.py` + run it on today's Elizilla output (also feeds the eval).
2. `situations.json` draft (teacher or by hand) + hand edit + held-out split.
3. `simulate.py` + `build.py` with hand-written profiles only: a first small
   dataset, no teacher yet.
4. Teacher benchmark in CI (Step 0), then `profiles.py` and teacher-filled
   turns.
5. `clean.py`, then a full build in CI.
6. Review round(s), fixes, rebuild until Step 6 passes.
