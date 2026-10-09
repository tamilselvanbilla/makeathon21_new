# Offline AI Companion

A private, always-available voice assistant that runs **entirely on a Raspberry Pi 4**.
Speech recognition, reasoning, and speech output all happen on the device. The only
network traffic the design allows is a narrow, allowlisted channel for real-time
facts (weather, news, search), and the reasoning about those facts still happens locally.

Built for Makeathon problem statement 4: *Personal, Offline, Always-On AI Companion*.

```
 Microphone ─► Whisper STT ─► Router ─┬─► Local knowledge ─► Qwen3-0.6B (llama.cpp) ─► Voice
   (RAM only)    (on device)          │                          ▲
                                      └─► Online gateway ────────┘  (facts only, text only)
```

---

## Contents

- [Why this exists](#why-this-exists)
- [Status against the challenge requirements](#status-against-the-challenge-requirements)
- [Hardware](#hardware)
- [Quick start](#quick-start)
- [Using the assistant](#using-the-assistant)
- [Project layout](#project-layout)
- [Privacy guarantees](#privacy-guarantees)
- [Documentation](#documentation)
- [Roadmap](#roadmap)

---

## Why this exists

Cloud-dependent AI wearables stop working when their backend changes or shuts down,
and always-listening devices raise real privacy concerns. This companion takes the
opposite approach: the models live on the device, raw audio never leaves it, and it
keeps working with the network cable unplugged.

## Status against the challenge requirements

| Requirement | Status | Where |
|---|---|---|
| All reasoning on-device, no cloud LLM | ✅ Done | Qwen3-0.6B Q4 via llama.cpp: `brain/llm.py` |
| Raw audio never leaves the device | ✅ Done | Audio held in RAM only; the gateway accepts text only, and a test fails if any other module imports a network library |
| Online calls limited to factual lookups | ✅ Boundary built | `online_gateway.py` (no providers yet, so lookups are refused honestly) |
| Honest fallback instead of guessing | 🟡 Basic | `brain/policy.py`; signal-based fallback is planned |
| Working demo in one human-potential domain | 🟡 Basic | Personal records recall over `knowledge_base/` |
| Continuous sensing with **wake word** | ⏳ Next | Currently listens for any speech; openWakeWord is planned |
| **Physical mute switch** | ⏳ Next | Interface and software switch done; GPIO driver planned |
| **Visible listening light** | ⏳ Next | All states implemented, printed to the console; LED driver planned |

## Hardware

| Part | Recommended | Notes |
|---|---|---|
| Board | Raspberry Pi 4 Model B, **4 GB** | 64-bit Raspberry Pi OS **Lite** recommended (the desktop uses several hundred MB of RAM) |
| Storage | 32 GB microSD (A1/A2) | Models and dependencies use about 2 GB |
| Memory use | about 1.1 GB at peak | Qwen3 weights ~370 MB, KV cache 224 MB, compute 150 MB, Whisper tiny.en ~225 MB, leaving over 2 GB free on a 4 GB Pi |
| Cooling | Heatsink + fan case | The Pi 4 throttles at about 80 °C under sustained inference |
| Microphone | ReSpeaker 2-Mic Pi HAT, or any USB mic | The HAT also provides RGB LEDs and a button |
| Speaker | 3.5 mm or USB speaker | For spoken replies |
| Power | Official 5 V / 3 A USB-C supply | Under-voltage causes throttling |

Development also works on macOS or Linux laptops; the same code detects the platform
and adjusts its defaults.

## Quick start

```bash
git clone <this repo> && cd makeathon21_new
scripts/setup.sh                 # one-time install; nothing is compiled on the Pi
scripts/run.sh --text            # type to it first, to check the language model
scripts/run.sh --list-mics       # find your microphone index
MIC_DEVICE=1 scripts/run.sh      # talk to it
```

`setup.sh` installs system packages, creates `.venv`, installs Python dependencies,
downloads and checksum-verifies the Qwen3 model, caches the Whisper model, and runs
the tests. For each step explained, manual installation, and troubleshooting, see
**[docs/setup.md](docs/setup.md)**.

## Using the assistant

| Command | What it does |
|---|---|
| `scripts/run.sh` | Voice mode: microphone → Whisper → local LLM → spoken reply |
| `scripts/run.sh --text` | Type requests; ideal over SSH or without a microphone |
| `scripts/run.sh --no-tts` | Voice in, printed replies out |
| `scripts/run.sh --offline` | Disable every online lookup |
| `scripts/run.sh --list-mics` | List audio devices and their indexes |
| `scripts/run.sh --mic-test` | Record 5 seconds to `audio/test.wav` |
| `scripts/run.sh --list-speakers` | List audio outputs (Pi 4 aux jack: `plughw:CARD=Headphones`) |
| `scripts/run.sh --speaker-test` | Speak a test phrase through `AUDIO_OUTPUT_DEVICE` |

Try asking:

- "What is my monthly income?" (answered locally from `knowledge_base/personal_data.json`)
- "What's the weather in Bengaluru?" (routed to the online gateway; currently refused honestly)
- "Exit" (stops the session)

The `[LED] ...` lines in the console show the indicator state: `IDLE`, `LISTENING`,
`THINKING`, `ONLINE`, `SPEAKING`, `MUTED`. The physical LED driver will show the
same states.

All settings are environment variables (model path, Whisper size, thresholds, and so on).
See **[docs/configuration.md](docs/configuration.md)**.

## Project layout

```
README.md                         this file
docs/
  setup.md                        detailed installation, Pi notes, troubleshooting
  configuration.md                every environment variable
  architecture.md                 pipeline, modules, privacy boundary, extension points
scripts/
  setup.sh                        one-time setup (Pi OS / Debian / macOS)
  run.sh                          start the assistant with model hubs forced offline
  mic_test.py                     record a test clip
src/
  main.py                         entry point
  companion/
    app.py                        builds components from configuration
    config.py                     settings with Raspberry Pi 4 defaults
    pipeline.py                   turn loop, microphone and text input
    online_gateway.py             the ONLY module allowed network access
    telemetry.py                  JSON timing logs (no prompts or audio)
    audio/   capture.py, stt.py
    brain/   llm.py, router.py, prompts.py, policy.py, knowledge.py
    device/  indicator.py, mute.py, tts.py
knowledge_base/personal_data.json synthetic personal records used in the demo
tests/                            unit tests (run without models or audio hardware)
```

## Privacy guarantees

1. **Reasoning is local.** Every answer comes from the on-device model.
2. **Audio stays in memory.** Recordings are never written to disk or sent anywhere,
   and audio captured while the mute switch is flipped is discarded.
3. **One network exit.** Only `online_gateway.py` may import networking code. It accepts
   plain text only, refuses requests mentioning private data (financial, medical,
   documents, recordings), and only contacts allowlisted hosts.
4. **Models are offline at runtime.** `run.sh` sets `HF_HUB_OFFLINE=1`, so model libraries
   cannot download or report anything.
5. **Logs hold timings, not content.** `logs/assistant.log` records events and durations,
   never prompts, transcripts, or audio.

Points 2 and 3 are enforced by tests in `tests/test_privacy_and_control.py`.

## Documentation

| Document | Read it when you want to |
|---|---|
| [docs/setup.md](docs/setup.md) | Install on a Pi or laptop, step by step, and fix problems |
| [docs/configuration.md](docs/configuration.md) | Tune models, microphone sensitivity, or performance |
| [docs/architecture.md](docs/architecture.md) | Understand the pipeline or add a feature |

## Roadmap

1. **Must-haves:** openWakeWord wake word, GPIO mute switch, LED driver, weather lookup via Open-Meteo.
2. **Core use case:** remember/recall notes, reminders, grammar-constrained intent routing.
3. **Polish:** signal-based honest fallback, Piper TTS, local web dashboard showing LOCAL vs ONLINE steps, systemd service.

## Running the tests

```bash
.venv/bin/python -m unittest discover -s tests
```

The tests use fakes, so they need no models, microphone, or network.
