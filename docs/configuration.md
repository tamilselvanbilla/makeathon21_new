# Configuration Reference

All settings are environment variables read at startup, so you can change models
and tuning without editing code. Set them per run:

```bash
WHISPER_MODEL_SIZE=base.en LLM_MAX_TOKENS=96 scripts/run.sh
```

or persistently in `~/.bashrc` (`export MIC_DEVICE=1`).

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
| `LLM_TEMPERATURE` | `0.2` | Lower means more factual and repeatable |

## Speech recognition (faster-whisper)

| Variable | Default (Pi / other) | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `tiny.en` / `base.en` | `tiny.en`, `base.en`, `small.en`, ... Larger is more accurate and slower. Cache a new size with setup first (see [setup.md §8](setup.md#8-updating-and-changing-models)) |
| `WHISPER_LANGUAGE` | `en` | Language code. Use a multilingual size (no `.en`) for other languages |
| `WHISPER_BEAM_SIZE` | `1` / `5` | `1` is greedy decoding (fastest); `5` is more accurate |
| `WHISPER_THREADS` | cores | CPU threads for transcription |

## Microphone capture

| Variable | Default | Description |
|---|---|---|
| `MIC_DEVICE` | *(prompt, or system default when headless)* | Input device index from `scripts/run.sh --list-mics` |
| `SPEECH_RMS_THRESHOLD` | `450` | Loudness that counts as speech (int16 RMS). Lower for quiet mics, higher for noisy rooms |
| `SILENCE_SECONDS` | `0.8` | Silence that ends an utterance |
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
| `AUDIO_OUTPUT_DEVICE` | `default` | ALSA device for `aplay`. Pi 4 3.5 mm jack: `plughw:CARD=Headphones`. List with `scripts/run.sh --list-speakers` |

## Online lookups

| Variable | Default | Description |
|---|---|---|
| `ONLINE_LOOKUPS` | `1` | `0` disables the online gateway entirely (same as `--offline`) |
| `DEFAULT_PLACE` | `Bengaluru` | Place used for weather questions that don't name one ("will it rain today?") |

Only weather is supported (Open-Meteo, no API key). The question is parsed on the
device and only the place name and the day ("today"/"tomorrow") are sent.

## Wake phrase and conversation

The wake phrase is spotted in Whisper's own transcripts, so no wake-word model is
downloaded and there is no extra licence (Whisper and faster-whisper are MIT).

| Variable | Default | Description |
|---|---|---|
| `WAKE_WORD` | `1` | `0` treats any speech as a request (same as `--no-wake-word`) |
| `WAKE_PHRASE` | `hey jarvis` | Any phrase. The greeting is flexible: with `hey jarvis`, "Jarvis", "Hi Jarvis" and "OK Jarvis" also work |
| `CONVERSATION_TIMEOUT` | `30` | Seconds of silence after a reply before it needs the wake phrase again |
| `WHISPER_HOTWORDS` | `EMI PAN Aadhaar` | Words Whisper should favour. The wake name and `DEFAULT_PLACE` are added automatically |

**Choosing a wake phrase:** use a name Whisper spells consistently, i.e. real words
or common names ("hey computer", "hey jarvis", "hello friday"). Invented names get
spelled differently each time and won't match. Check yours with
`scripts/run.sh --mic-test`, then
`cd src && ../.venv/bin/python -m companion.audio.stt ../audio/test.wav`.

**Ending a conversation:** "that's all", "stop listening", "go to sleep", "goodbye",
"thank you" or "no thanks" put it back to sleep, as do 30 s of silence and the mute
switch. By voice it never shuts down; `Ctrl+C` stops the program.

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
