# Configuration Reference

All settings are environment variables read at startup, so you can change models
and tuning without editing code. Set them per run:

```bash
WHISPER_MODEL_SIZE=base.en LLM_MAX_TOKENS=96 scripts/run.sh
```

or persistently in `~/.bashrc` (`export MIC_DEVICE=USB`).

Defaults marked **Pi / other** differ by platform. A Raspberry Pi is detected from
`/proc/device-tree/model`. "Cores" means the number of CPU cores (4 on a Pi 4).

## Language model

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL_PATH` | `models/llm/Qwen3-0.6B-Q4_K_M.gguf` | Path to the GGUF model file |
| `LLM_MODEL` | `Qwen3-0.6B-Q4_K_M` | Display name used in logs |
| `LLM_CHAT_FORMAT` | `chatml` | llama-cpp-python chat format. Qwen uses `chatml`; use `llama-3` for Llama 3.x and `gemma` for Gemma |
| `LLM_CONTEXT` | `2048` | Context window in tokens. Each 1024 tokens costs about 115 MB of RAM with Qwen3-0.6B |
| `LLM_THREADS` | cores | CPU threads for generation |
| `LLM_BATCH` | `256` | Prompt-processing batch size |
| `LLM_MAX_TOKENS` | `128` | Maximum reply length. The main lever on reply time on a Pi |
| `LLM_TEMPERATURE` | `0` | 0 = greedy: the same question always gets the same answer. At 0.2 the Pi answered "monthly income" with the gross figure once and the net figure once |

## Speech recognition (faster-whisper)

| Variable | Default (Pi / other) | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base.en` | `tiny.en`, `base.en`, `small.en`, ... On the Pi 4, `base.en` takes ~4.2 s per short question vs ~2.2 s for `tiny.en`, but `tiny.en` misheard live questions ("what is my EMI" → "what is mine"). Cache a new size with setup first (see [setup.md §8](setup.md#8-updating-and-changing-models)) |
| `WHISPER_LANGUAGE` | `en` | Language code. Use a multilingual size (no `.en`) for other languages |
| `WHISPER_BEAM_SIZE` | `1` / `5` | `1` is greedy decoding (fastest); `5` is more accurate |
| `WHISPER_HOTWORDS` | `EMI PAN Aadhaar` | Words Whisper should favour; `DEFAULT_PLACE` is added automatically |
| `WHISPER_THREADS` | cores | CPU threads for transcription |

## Microphone capture

| Variable | Default | Description |
|---|---|---|
| `MIC_DEVICE` | *(automatic)* | Index or part of the name of the input device (`scripts/run.sh --list-mics`). Unset: the system default input, else the first USB/ReSpeaker/"mic" device that isn't a loopback or monitor |
| `SPEECH_ONSET_FRAMES` | `3` | Consecutive 80 ms frames above the threshold that count as speech starting, so clicks and bumps don't start a recording. Only 0.5 s before that is kept |
| `SPEECH_RMS_THRESHOLD` | `450` | Loudness that counts as speech (int16 RMS). Lower for quiet mics, higher for noisy rooms |
| `SILENCE_SECONDS` | `1.2` | Silence that ends an utterance (0.8 split "remember that … meeting tomorrow" in two on the Pi) |
| `MAX_RECORD_SECONDS` | `15` | Hard cap on one utterance |
| `MAX_WAIT_FOR_SPEECH_SECONDS` | `30` | How long one listening window waits for speech before restarting |

Audio is captured at 16 kHz when the device supports it; otherwise it is captured at
the device's native rate and resampled.

### Tuning the threshold

1. Run `scripts/run.sh --mic-test` in a quiet room, then while speaking normally.
2. If speech is not detected, halve `SPEECH_RMS_THRESHOLD`; if room noise triggers
   recording, double it.

## Speech output

| Variable | Default | Description |
|---|---|---|
| `TTS_ENABLED` | `1` | `0` prints replies instead of speaking (same as `--no-tts`) |
| `TTS_RATE` | `150` | Words per minute |
| `TTS_VOLUME` | `1.0` | 0.0 to 1.0 |
| `TTS_VOICE` | *(system default)* | Voice name, e.g. `en-us` or `en-gb` for espeak-ng (`espeak-ng --voices=en` lists them); a voice id for pyttsx3 |
| `TTS_ENGINE` | `auto` | `auto` uses espeak-ng + aplay when both are installed (Linux/Pi), otherwise pyttsx3 (macOS/Windows). Force with `espeak` or `pyttsx3` |
| `AUDIO_OUTPUT_DEVICE` | *(automatic)* | ALSA device for `aplay`. Unset: the microphone's own USB device (e.g. a headset), then the Pi's 3.5 mm jack (`plughw:CARD=Headphones,DEV=0`), then `default`; a device that fails is skipped and the first that works is kept (start-up prints `Speaker: …`). On a Pi running PipeWire, `default` fails with error 524 outside the desktop session. List with `scripts/run.sh --list-speakers` |

## Online lookups

| Variable | Default | Description |
|---|---|---|
| `ONLINE_LOOKUPS` | `1` | `0` disables the online gateway entirely (same as `--offline`) |
| `ONLINE_WEATHER` | `1` | `0` switches off weather lookups |
| `ONLINE_MARKET` | `1` | `0` switches off share prices and fund NAVs; portfolio questions are then answered from the saved records |
| `ONLINE_NEWS` | `1` | `0` switches off news headlines |
| `ASSISTANT_NAME` | `Sam` | The assistant's name in the welcome message |
| `CONTEXT_DEBUG` | `0` | `1` prints, for each question, the records and memory sent to the model, to check what it was given |
| `HOME_COUNTRY` | `IN` | Preferred country (ISO code) when a place name exists in several countries; chosen locally, never sent. Old city names (Bangalore, Bombay, Madras, Calcutta, Mysore, Trivandrum, Cochin, Pondicherry) are mapped to current ones first |
| `DEFAULT_PLACE` | `Bengaluru` | Place used for weather questions that don't name one ("will it rain today?") |
| `MARKET_SYMBOLS_FILE` | `knowledge_base/market_symbols.json` | Company and index names the assistant may look up, mapped to symbols |

Startup prints which are on, e.g. `Online lookups: market, news, weather`. No source needs an
API key. What each sends:

| Feature | Source | Sent | Kept on the device |
|---|---|---|---|
| Weather | Open-Meteo | Place name, then its coordinates | The question |
| Market | Yahoo Finance (prices; **unofficial endpoint**, may change), mfapi.in (AMFI's official NAVs) | The **whole watchlist** of symbols and fund codes, whatever was asked | Quantities, prices paid, all values and gains, the question |
| News | The Hindu (business, national, technology), BBC News (world) RSS | Nothing but the fixed feed address | The topic: headlines are filtered on the device |

Results are cached (prices 5 min, NAVs 6 h, news 15 min, weather 10 min). If a refresh fails,
the cached copy is used and the reply says when it was fetched.

### Market data

Holdings come from `knowledge_base/personal_data.json`: a `stock` record needs `ticker`
(Yahoo symbol, e.g. `INFY.NS`), `quantity` and `purchase_price`; a `mutual_fund` record needs
`scheme_code` (AMFI code, e.g. `120377`, findable at `https://api.mfapi.in/mf/search?q=<name>`),
`units` and `investment_amount`. Only the primary user's holdings are used.

`knowledge_base/market_symbols.json` maps spoken names to symbols for companies you don't
hold ("TCS" → `TCS.NS`) and lists the `indices` always fetched (Nifty 50, Sensex). Add
entries to ask about more companies; the longest name is the one spoken in replies.

## Memory

| Variable | Default | Description |
|---|---|---|
| `MEMORY` | `1` | `0` keeps memory in RAM for the current session only; nothing is written to disk |
| `MEMORY_FILE` | `data/memory.sqlite3` | Where notes and past conversations are stored (git-ignored) |
| `MEMORY_RETENTION_DAYS` | `30` | Past questions and answers older than this are deleted at startup; notes and reminders are kept until deleted |
| `FOLLOW_UP_MINUTES` | `10` | How recent the previous question must be for "and my wife's?" to build on it |
| `MEMORY_TOP_K` | `3` | Remembered items given to the model per question |
| `MEMORY_EMBEDDINGS` | `1` | `0` uses keyword search only (no embedding model loaded, ~90 MB less RAM) |
| `EMBEDDING_MODEL` | `minilm-int8` | `minilm-int8`, `minilm` or `bge-small` (see *Choosing the embedding model*) |
| `EMBEDDING_DIR` | `models/embedding` | Folder with the embedding models |
| `MEMORY_MIN_SIMILARITY` | `0.35` | Cosine similarity a note needs to match by meaning alone. Lower finds more paraphrases but more wrong notes |

### Choosing the embedding model

Measured with `scripts/eval_memory_retrieval.py` on 15 notes, 34 questions that should
find a note and 13 that must not (on a laptop, 2 threads):

| Method | Finds | Rejects | Download | RAM | Per question |
|---|---|---|---|---|---|
| Keywords only (`MEMORY_EMBEDDINGS=0`) | 79% | 62% | — | — | <1 ms |
| **Keywords + `minilm-int8` (default)** | **94%** | 62% | 23 MB | ~90 MB | ~1 ms |
| Keywords + `minilm` | 94% | 62% | 90 MB | ~185 MB | ~2 ms |
| Keywords + `bge-small` (threshold 0.65) | 79% | 85% | 133 MB | ~230 MB | ~7 ms |

`bge-small` rejects near-misses better only in a narrow threshold band (0.60 → 54%,
0.65 → 85%), at 2.5× the RAM. No method reliably rejects near-misses ("wife's
birthday" when only mom's is stored), so those answers are hedged instead. To compare
models on your device, download `minilm` or `bge-small` into `models/embedding/` and run
`.venv/bin/python scripts/eval_memory_retrieval.py --models minilm-int8 bge-small`.

Voice commands:

| Say | Effect |
|---|---|
| "Remember (that) …", "Note (that) …", "Make a note …", "Don't forget …", "Save a note …" | Saves a note; confirmed as "Okay, I'll remember that you …". Relative dates are resolved ("tomorrow at 7am" → "on Saturday 10 October at 7:00 AM") |
| "Remember that." / "Remind me." with nothing after it (a pause) | Asks "What should I remember?" and saves the next sentence |
| "Remind me to call mom at 6 pm", "Remind me in 10 minutes to …" | Saves "you need to call mom …" and announces it when due (checked every `MAX_WAIT_FOR_SPEECH_SECONDS` while listening, and at startup) |
| A statement with a date, time, place or task: "I have a meeting with Jay tomorrow", "I parked on B2", "My wife asked me to buy vegetables" | Asks "Do you want me to remember that …?"; "yes" saves it, "no" or another question drops it |
| "What's my schedule tomorrow?", "Do I have anything on Monday?", "What are my reminders?" | Lists the dated notes for that day, or the coming week, without the model |
| "Forget that", "Forget the last thing", "Delete that" | Deletes the most recent note or exchange |
| "Forget everything", "Clear your memory" | Deletes all memory |

The model never saves anything itself; if it replies "I'll keep that in mind", the app
replaces that with "I haven't saved that. Do you want me to remember …?".

At startup the assistant reads out reminders missed in the last 12 hours and what is still
ahead today.

To inspect memory on the device: `sqlite3 data/memory.sqlite3 "SELECT created, kind, question, answer FROM memory"`
and `sqlite3 data/memory.sqlite3 "SELECT * FROM schedule"`.

## Tracing

| Variable | Default | Meaning |
|---|---|---|
| `TRACING` | `1` | Trace every turn (see [observability.md](observability.md)); `0` turns it off |
| `TRACE_FILE` | `data/traces.sqlite3` | Trace store; RAM only when `MEMORY=0` |
| `TRACE_CONTENT` | `1` | `0` keeps timings and routes but not the question and reply text |
| `TRACE_CONSOLE` | `1` | `0` hides the `[TRACE]` line printed after each turn |
| `TRACE_RETENTION_DAYS` | `30` | Older traces are deleted at startup |

## Set by `scripts/run.sh`

| Variable | Value | Why |
|---|---|---|
| `HF_HUB_OFFLINE` | `1` | Model libraries may only read the local cache |
| `TRANSFORMERS_OFFLINE` | `1` | Same, for any transformers-based component |
| `HF_HUB_DISABLE_TELEMETRY` | `1` | No usage reporting |

## Personal data

The assistant answers personal questions from `knowledge_base/personal_data.json`,
which has four lists: `financial`, `medical`, `documents`, `history`. Each entry is a
JSON object with an `owner`; `source` is never shown to the model. Edit the file and
restart the assistant to pick up changes. Use synthetic data only for demos.

| Variable | Default | Description |
|---|---|---|
| `KNOWLEDGE_FILE` | `knowledge_base/personal_data.json` | Path to the records file |
| `PRIMARY_USER` | most frequent `owner` | Whose records "I", "me" and "my" refer to |
| `KNOWLEDGE_TOP_K` | `4` | Records sent to the model per question; each adds prompt time on a Pi |
| `CURRENCY` | `INR` | Currency the model uses for money amounts |

### Writing records that answer well

- **Owners:** name other people relative to the primary user, e.g. `"John's wife"`.
  A `family_member` record with `name` and `relationship` lets questions use the name
  ("how much does Jane earn"). Owners containing "family" are shared by everyone.
- **`record_type`:** use the word people will say (`passport`, `bank_loan`,
  `vehicle`). Naming a type in a question narrows the answer to that type.
- **Field names:** descriptive snake_case (`expiry_date`, `monthly_installment`); they
  become labels the search and the model both use.
- **Missing facts:** leave them out rather than writing placeholders; the assistant
  then says the fact is not in the records.
- **New vocabulary:** if people ask with words your records don't use, add a mapping
  to `SYNONYMS` in `src/companion/brain/knowledge.py`.

## Recommended Pi 4 profiles

All profiles fit comfortably in a 4 GB Pi 4.

| Goal | Settings | Approx. peak memory |
|---|---|---|
| Fastest replies | `WHISPER_MODEL_SIZE=tiny.en LLM_MAX_TOKENS=80 LLM_CONTEXT=1024` | ~1.0 GB |
| Balanced *(default)* | no overrides | ~1.1 GB |
| Better transcription | `WHISPER_MODEL_SIZE=base.en` (more accurate, slower to transcribe) | ~1.2 GB |
