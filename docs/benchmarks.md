# Benchmarks

Two benchmarks back the model choices. Both run on the device with one command,
so the numbers can be re-measured on the Raspberry Pi.

- [Language model](#language-model): which small LLM reasons best on this assistant's job
- [Knowledge retrieval](#knowledge-retrieval): finding the right personal record for natural spoken questions
- [Memory retrieval](#memory-retrieval): keywords vs embeddings for finding remembered notes

---

## Language model

```bash
.venv/bin/python scripts/eval_llm.py --threads 4 [--models qwen3-0.6b qwen3-1.7b] [--save results.json]
```

`tests/data/llm_eval.json` holds 32 cases in 8 categories, each rendered with the
**app's own prompts** (system prompt, retrieved records from `personal_data.json`,
memory, previous question, online facts) and scored automatically by required and
forbidden patterns:

| Category | Cases | What it checks |
|---|---|---|
| grounded | 8 | Answer a question from retrieved records (income, EMI, insurance date, family…) |
| reasoning | 8 | Arithmetic and comparison over records (left after expenses, total invested, extra per year…) |
| honesty | 4 | Say "not in the records" instead of inventing (blood group, passport expiry, wife's BP); defer medical decisions |
| attribution | 2 | Use the right person's record when both are present |
| follow-up | 2 | Resolve "and my wife's?" using the previous question |
| memory | 2 | Answer from a remembered note or a past answer |
| weather-advice | 2 | Umbrella advice from 80% vs 5% chance of rain |
| general | 4 | Simple general knowledge (capital, leap year, percentage, boiling point) |

"Style" checks every answer is at most three sentences, leaks no field labels or
JSON, uses no foreign currency, and speaks to the user ("Your…", not "My…").

### Results

Q4_K_M quantisation, 4 threads (as on a Pi 4), greedy decoding, measured on an Apple
M2 Pro laptop on 9 Oct 2026. **Times are laptop times**; RAM on the laptop includes a
second copy of the weights that llama.cpp makes only on CPUs with newer matrix
instructions, so expect less on the Pi.

| Model | Licence | File | Accuracy | grounded | reasoning | honesty | attribution | follow-up | memory | weather | general | Style | Median tokens | Median reply (laptop) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3-0.6B** (current) | Apache-2.0 | 397 MB | 69% | 8/8 | 2/8 | 2/4 | 2/2 | 2/2 | 2/2 | 2/2 | 2/4 | 94% | 19 | 0.4 s |
| Qwen3-0.6B + thinking | Apache-2.0 | 397 MB | 88% | 8/8 | 6/8 | 3/4 | 2/2 | 2/2 | 2/2 | 1/2 | 4/4 | 97% | 172 | 1.8 s |
| Qwen2.5-0.5B-Instruct | Apache-2.0 | 491 MB | 62% | 7/8 | 2/8 | 1/4 | 1/2 | 2/2 | 2/2 | 1/2 | 4/4 | 100% | 15 | 0.3 s |
| Llama-3.2-1B-Instruct | Llama 3.2 Community ⚠️ | 808 MB | 59% | 4/8 | 2/8 | 4/4 | 1/2 | 1/2 | 2/2 | 1/2 | 4/4 | 38% | 17 | 0.4 s |
| Gemma-3-1B-it | Gemma Terms ⚠️ | 806 MB | 62% | 8/8 | 3/8 | 1/4 | 2/2 | 1/2 | 2/2 | 1/2 | 2/4 | 84% | 34 | 0.8 s |
| Qwen2.5-1.5B-Instruct | Apache-2.0 | 1,117 MB | 84% | 8/8 | 5/8 | 4/4 | 2/2 | 2/2 | 2/2 | 1/2 | 3/4 | 84% | 17 | 0.6 s |
| SmolLM2-1.7B-Instruct | Apache-2.0 | 1,056 MB | 41% | 6/8 | 1/8 | 0/4 | 1/2 | 0/2 | 1/2 | 2/2 | 2/4 | 100% | 14 | 0.7 s |
| **Qwen3-1.7B** | Apache-2.0 | 1,107 MB | 78% | 8/8 | 3/8 | 4/4 | 2/2 | 2/2 | 1/2 | 1/2 | 4/4 | 97% | 19 | 0.8 s |
| Qwen3-1.7B + thinking | Apache-2.0 | 1,107 MB | **100%** | 8/8 | 8/8 | 4/4 | 2/2 | 2/2 | 2/2 | 2/2 | 4/4 | 100% | 181 | 3.6 s |

⚠️ Not an OSI open-source licence (usage restrictions), which the problem statement asks for.

### Findings

1. **Retrieval-grounded answers are solved; reasoning is not.** Every Qwen model gets
   8/8 grounded answers, thanks to the retrieval work. Without a reasoning trace,
   even 1.5–1.7B models get arithmetic wrong: "42,800" / "42,100" for 17,500 + 25,000,
   "₹30" profit for 1,880 − 1,750, "₹8,400" extra for (10,000 − 9,200) × 12.
2. **Thinking fixes reasoning but costs ~9× the tokens.** Qwen3-1.7B goes from 78% to
   100%, Qwen3-0.6B from 69% to 88%, but median answers grow from ~19 to ~180 tokens
   (up to ~740). On a Pi, where generation is several times slower than on a laptop,
   that is the difference between a few seconds and tens of seconds per answer.
3. **Bigger isn't automatically better.** SmolLM2-1.7B scored lowest (41%, honesty 0/4:
   it invented a blood group and a passport expiry). Qwen2.5-1.5B is the strongest
   non-thinking model (84%) but speaks as the user in 5 answers.
4. **Honesty improves with size within Qwen.** Qwen3-0.6B said "Your wife's blood
   pressure is 120/80" (John's reading) and "your passport expires on the date specified
   in the record"; Qwen3-1.7B and Qwen2.5-1.5B declined correctly (4/4).
5. **Llama 3.2 1B speaks as the user** ("My monthly income is…") in 20 of 32 answers,
   refused a simple family question, and gave a wrong insurance date (4/8 grounded).
6. **Gemma has no system role, and llama-cpp-python's `gemma` chat format silently drops
   the system prompt.** The benchmark merges it into the first user turn; the app's
   `LLM_CHAT_FORMAT=gemma` would lose all instructions.
7. **Our own system prompt over-restricts general questions.** "Use only the supplied
   knowledge" made several models refuse "how many days in a leap year" (general 2/4
   for Qwen3-0.6B and Gemma).
8. **Weather advice from probabilities is unreliable** for every model without thinking
   ("likely to rain" or "carry an umbrella just in case" at 5%).

### System prompt variants

The system prompt describes the assistant's role (keeper of the user's and family's
financial, insurance, medical, health and identity records, answering from the knowledge
base). Wording matters a lot for small models; four variants were measured on the same 32
cases:

| Prompt | Qwen3-0.6B | Style | Qwen3-1.7B | Style | Notes |
|---|---|---|---|---|---|
| Original | 69% | 94% | 78% | 97% | Refused general questions ("leap year": "not in the knowledge") |
| A: persona ("You are Sam…") + scope rules | 69% | 84% | 75% | 94% | 0.6B spoke as the user ("I am John."); 1.7B refused "capital of France" |
| B: persona, user identity first, separate rules for personal vs general | 62% | 81% | **84%** | 94% | Best for 1.7B; 0.6B kept speaking as the user |
| **C: original structure + role, record types, family, general-knowledge and identity-number rules (shipped)** | **69%** | **97%** | 75% | 100% | General 4/4 for 0.6B; no persona name (the name is in the welcome message) |

C is shipped because it suits the default Qwen3-0.6B; if the device moves to Qwen3-1.7B,
variant B scores higher. Differences of one or two cases (3–6%) are within noise for a
32-case set. The two honesty cases the 0.6B model still fails (passport expiry, wife's
blood pressure) are caught by the app before the model is asked: retrieval finds no such
record and the app answers "I couldn't find that in your personal records".

### What this suggests

- Stay with the **Qwen3 family** (Apache-2.0, best accuracy and honesty per MB).
- Don't buy reasoning with model size or thinking time on a Pi; do arithmetic **in
  code**, as the portfolio feature already does: the model picks the numbers or writes
  an expression, Python computes it.
- Choose between **Qwen3-0.6B** (fast, weaker honesty) and **Qwen3-1.7B** (honest 4/4,
  ~3× slower per token) once Pi timings are measured with this script.
- Fix the general-knowledge prompt restriction, the Gemma system prompt, and make
  weather advice rule-based.

---

## Knowledge retrieval

```bash
.venv/bin/python -m unittest tests.test_privacy_and_control.RetrievalBenchmarkTests
```

`tests/data/knowledge_retrieval_eval.json`: natural spoken questions over the records,
each with the record type the top result must have, or `null` when retrieval must find
nothing (so the assistant says it isn't recorded rather than guessing).

A live session showed retrieval failing on ordinary speech ("what are the medications I
am taking daily?" → "couldn't find that"; "about my medications" → the blood-test report).
Measured, then fixed:

| Version | Main set (40 answerable) | Must find nothing (6) | Unseen set A (20) | Unseen set B (15) |
|---|---|---|---|---|
| Before | 15/40 | 6/6 | — | — |
| + filler words, synonyms, own-word ranking | 40/40 | 6/6 | 12/20 | — |
| + one record must cover the question, acronyms, no type filter | 39/40 | 6/6 | 18/20 | 13/15 *(first run, untuned)* |
| + named entities, category-collision rule (shipped) | 40/40 | 6/6 | 19/20 | 15/15 |

Sets A and B were written after the main set and scored before tuning on them; later
fixes did use them, so the final numbers are optimistic. The first untuned scores (12/20,
13/15) are the honest estimate for brand-new phrasings at each step.

Lessons:
- **Synonyms alone don't generalise**: the first version scored 40/40 on its own questions
  but 12/20 on new ones. The structural rules (one record must cover the whole question;
  names may point elsewhere; acronyms; stem collisions) fixed whole classes of misses.
- **Word-level embeddings did not help**: MiniLM similarity between a question word and a
  record field was as high for wrong pairs ("expire" ↔ a bond's maturity date, 0.50) as for
  right ones ("SBI" ↔ State Bank of India, 0.35), so no threshold separates them.
- **Known miss**: "which medicine my **doctor** prescribed" finds nothing: no record names
  a doctor, and mapping "doctor" to the prescription would make "who is my family doctor?"
  answer wrongly.
- With the right record in context, Qwen3-0.6B still refused "what are the medications I am
  taking daily?" ("not recorded"); Qwen3-1.7B answered it correctly.

## Memory retrieval

```bash
.venv/bin/python scripts/eval_memory_retrieval.py --models minilm-int8 [minilm bge-small]
```

16 notes, 37 questions that should find a note and 15 that must find nothing
(`tests/data/memory_retrieval_eval.json`; the last 5 come from a Raspberry Pi session).
Laptop, 2 threads. The table below was measured on the first 15 notes and 47 questions.

| Method | Finds | Rejects | Download | RAM | Per question |
|---|---|---|---|---|---|
| Keywords only (`MEMORY_EMBEDDINGS=0`) | 79% | 62% | — | — | <1 ms |
| **Keywords + MiniLM int8 (default)** | **94%** | 62% | 23 MB | ~90 MB | ~1 ms |
| Keywords + MiniLM full precision | 94% | 62% | 90 MB | ~185 MB | ~2 ms |
| Keywords + bge-small (threshold 0.65) | 79% | 85% | 133 MB | ~230 MB | ~7 ms |

On the extended set (Oct 2026: notes stored with resolved dates, memory-only synonyms such
as park → level), the shipped hybrid finds 92% and rejects 47%; 9/9 wrong answers are
hedged. A lower similarity bar for notes sharing some of the question's words (0.25) was
tried and dropped: it added a wrong match ("who borrowed my ladder?" → the drill) and found
nothing new.

No method rejects near-misses ("wife's birthday" when only mom's is stored), so answers
found only by meaning are hedged; in the benchmark all wrong matches were hedged. See
configuration.md, *Choosing the embedding model*.
