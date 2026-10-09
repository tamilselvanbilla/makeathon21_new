# Setup Guide

This guide covers a full install on a **Raspberry Pi 4** (the target device), on a
macOS or Linux laptop for development, and manually on Windows. It ends with a
troubleshooting section.

- [1. Prepare the Raspberry Pi](#1-prepare-the-raspberry-pi)
- [2. Get the code](#2-get-the-code)
- [3. Run the setup script](#3-run-the-setup-script)
- [4. Verify each component](#4-verify-each-component)
- [5. Run the assistant](#5-run-the-assistant)
- [6. Development on macOS / Linux](#6-development-on-macos--linux)
- [7. Manual installation (no script / Windows)](#7-manual-installation-no-script--windows)
- [8. Updating and changing models](#8-updating-and-changing-models)
- [9. Troubleshooting](#9-troubleshooting)
- [10. What is not committed to Git](#10-what-is-not-committed-to-git)

---

## 1. Prepare the Raspberry Pi

### 1.1 Flash the OS

1. Install **Raspberry Pi Imager** on your laptop.
2. Choose **Raspberry Pi OS Lite (64-bit)**. Lite has no desktop, which leaves
   several hundred MB more RAM for the models on a 4 GB Pi. The 32-bit image will
   not work, because llama.cpp and faster-whisper need `aarch64`. Bookworm
   (Python 3.11) and Trixie (Python 3.13) are both supported.
3. In the Imager's settings (gear icon), set a hostname (e.g. `companion`), enable
   SSH, set a username/password, and configure Wi-Fi.
4. Flash the SD card, insert it, and power on the Pi.

### 1.2 First login

```bash
ssh <user>@companion.local
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

Check that you are on 64-bit:

```bash
uname -m          # must print: aarch64
```

### 1.3 Memory budget (4 GB Pi)

Nothing is compiled on the Pi, so the default swap is enough. At runtime the
companion needs about 1.1 GB:

| Component | Memory |
|---|---|
| Qwen3-0.6B Q4_K_M weights (memory-mapped) | ~370 MB |
| LLM KV cache (`LLM_CONTEXT=2048`) | 224 MB |
| LLM compute buffers | ~150 MB |
| Whisper `tiny.en` (int8) + runtime | ~225 MB |
| Python, NumPy, audio | ~100 MB |

Raspberry Pi OS Lite itself uses about 250 MB, leaving over 2 GB free for the
planned Piper TTS voice. Check with `free -h` while the assistant
is running.

### 1.4 Connect audio hardware

- **USB microphone:** plug it in; it is detected automatically.
- **ReSpeaker 2-Mic Pi HAT:** seat it on the GPIO header and install its driver
  following the vendor's instructions for your OS release, then reboot.
- **Speaker:** 3.5 mm jack or USB.

Confirm the microphone is visible:

```bash
arecord -l        # lists capture devices
```

### 1.5 Cooling

Fit a heatsink and fan. Check the temperature while the assistant is answering:

```bash
vcgencmd measure_temp
vcgencmd get_throttled    # 0x0 means no throttling or under-voltage
```

---

## 2. Get the code

```bash
sudo apt install -y git
git clone <repository-url> ~/makeathon21_new
cd ~/makeathon21_new
```

---

## 3. Run the setup script

```bash
scripts/setup.sh
```

It asks for your password once (for `apt`). Nothing is compiled on the Pi: every
Python package, including `llama-cpp-python`, installs from a prebuilt aarch64 wheel,
so the time is mostly downloads (about 130 MB of packages and 470 MB of models).

### What the script does

| Step | Detail |
|---|---|
| Detect platform | Prints OS, CPU, and whether it is a Raspberry Pi |
| System packages (Linux) | `python3-venv python3-dev build-essential cmake git curl portaudio19-dev libportaudio2 espeak-ng alsa-utils` (the build tools are only a fallback) |
| System check (macOS) | Requires Xcode command-line tools for compiling |
| Python check | Requires Python 3.11 or newer, and a 64-bit Python (refuses `armv7l`) |
| LLM runtime (Pi) | Installs the prebuilt `llama-cpp-python` 0.3.35 aarch64 wheel; pip verifies its SHA-256. It uses baseline ARMv8 NEON only, so it runs on the Pi 4's Cortex-A72. On macOS/x86 it is compiled instead (a few minutes) |
| Virtual environment | Creates `.venv/` in the project root (reused if present) |
| Python packages | Installs `requirements.txt` (faster-whisper, llama-cpp-python, sounddevice, numpy, scipy, pyttsx3) |
| Language model | Downloads `Qwen3-0.6B-Q4_K_M.gguf` (~397 MB) from `huggingface.co/unsloth/Qwen3-0.6B-GGUF` into `models/llm/`, resumes interrupted downloads, and verifies the SHA-256 checksum |
| Speech model | Caches the Whisper model: `tiny.en` on a Pi, `base.en` elsewhere |
| Tests | Runs the unit test suite |

### Options

| Option | Use it when |
|---|---|
| `--skip-system` | You have no `sudo`, or system packages are already installed |
| `--skip-models` | You only want to reinstall Python packages |
| `--skip-tests` | You want a faster re-run |

The script is safe to run again: it reuses `.venv`, skips the model download if the
file exists (but still verifies it), and reinstalls only missing packages.

---

## 4. Verify each component

Work through these in order; each one isolates one part of the pipeline.

### 4.1 Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

Expected: `OK`. The tests need no models or hardware.

### 4.2 Language model (text mode)

```bash
scripts/run.sh --text
```

Expected output:

```
Device: Raspberry Pi
Loading local model 'Qwen3-0.6B-Q4_K_M' (4 threads)...
Ready in ...s. Type 'exit' to stop.
[LED] IDLE
You: what is my monthly income
[LED] THINKING
[LED] SPEAKING
Assistant: Your monthly income is INR 62,000. ...
```

Type `exit` to quit. Check `logs/assistant.log` for the `llm_generation` duration.

### 4.3 Microphone

```bash
scripts/run.sh --list-mics      # note the index of a device with "1 in" or more (">" marks the default)
scripts/run.sh --mic-test       # records 5 s to audio/test.wav
aplay audio/test.wav            # play it back (Linux)
```

### 4.4 Speech recognition

```bash
cd src && HF_HUB_OFFLINE=1 ../.venv/bin/python -m companion.audio.stt ../audio/test.wav; cd ..
```

Expected: `Transcription: <what you said>`.

### 4.5 Speaker (3.5 mm aux)

On Linux the companion renders speech with `espeak-ng` and plays it with `aplay`
straight to an ALSA device. It does not use espeak-ng's own playback, which needs a
desktop sound server and fails on Pi OS Lite or over SSH with
`audio open error: Unknown error 524`.

```bash
scripts/run.sh --list-speakers                 # find the output device
amixer -c Headphones sset PCM 90%              # raise the jack's volume (often low)
AUDIO_OUTPUT_DEVICE=plughw:CARD=Headphones scripts/run.sh --speaker-test
```

On a Pi 4, the 3.5 mm jack is the card named `Headphones`. If you hear nothing with
`default`, the audio is probably going to HDMI; set the device explicitly and keep it:

```bash
echo 'export AUDIO_OUTPUT_DEVICE=plughw:CARD=Headphones' >> ~/.bashrc
```

Use `plughw:` rather than `hw:` so ALSA converts espeak-ng's 22 kHz mono output to
whatever the card needs.

---

## 5. Run the assistant

```bash
MIC_DEVICE=<index> scripts/run.sh
```

| Option | Effect |
|---|---|
| *(none)* | Voice in, voice out |
| `--text` | Type requests instead of speaking |
| `--no-tts` | Print replies instead of speaking |
| `--offline` | Disable all online lookups |
| `--no-wake-word` | Treat any speech as a request |

### Wake phrase and conversation

The console shows `[LED] IDLE` while the device waits for the wake phrase. Speech
that doesn't start with it is dropped (`[IDLE] speech ignored`), and its text is never
shown or kept.

1. Say **"Hey Jarvis, what is my EMI?"** in one breath (or "Hey Jarvis", then the
   question). You'll see `[WAKE] 'hey jarvis' heard` and `[LED] LISTENING`.
2. Ask follow-ups **without** the wake phrase: "And when does my car insurance expire?"
3. Say **"That's all, thanks"**, or stay quiet for 30 s: `[SLEEP] ...` and back to `IDLE`.

If it doesn't wake, check what Whisper heard: run with `--no-wake-word` and look at the
`You:` line. Pick a phrase it spells consistently (configuration.md, *Wake phrase*).

### Online lookup

Ask "what's the weather in Mysore tomorrow?". The console shows exactly what left
the device and where the reasoning happened:

```
[LED] ONLINE
[ONLINE] sent only place='Mysore', day='tomorrow' to Open-Meteo
[ONLINE] Open-Meteo returned: In Mysore, India tomorrow: ...
[LED] THINKING
[LOCAL] reasoning about the result on-device
```

Unplug the network and ask again: the assistant says it couldn't reach the weather
service and won't guess, while personal questions keep working.

Stop with `Ctrl+C` (or type `exit` in `--text` mode). Spoken "exit" or "goodbye" only ends the current conversation.

To avoid passing `MIC_DEVICE` each time, add it to your shell profile:

```bash
echo 'export MIC_DEVICE=1' >> ~/.bashrc
```

If no `MIC_DEVICE` is set and the program runs without a terminal (e.g. as a
service), it uses the system default microphone instead of prompting.

All tunable settings are listed in [configuration.md](configuration.md).

---

## 6. Development on macOS / Linux

The same scripts work on a laptop:

```bash
xcode-select --install     # macOS only, once
scripts/setup.sh
scripts/run.sh --text
```

Differences from the Pi:

- Whisper defaults to `base.en` with beam size 5 (more accurate, needs more CPU).
- The LLM uses every CPU core.
- On macOS, grant your terminal microphone access in
  **System Settings → Privacy & Security → Microphone** the first time you use voice mode.

---

## 7. Manual installation (no script / Windows)

### Linux / macOS / Raspberry Pi

```bash
# System packages (Debian / Raspberry Pi OS)
sudo apt install -y python3-venv python3-dev build-essential cmake curl \
                    portaudio19-dev libportaudio2 espeak-ng alsa-utils

# Python environment
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -r requirements.txt

# Language model
mkdir -p models/llm
curl -fL -o models/llm/Qwen3-0.6B-Q4_K_M.gguf \
  https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf
sha256sum models/llm/Qwen3-0.6B-Q4_K_M.gguf     # macOS: shasum -a 256
# expected: ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a

# Run (the first voice run downloads the Whisper model, so leave HF_HUB_OFFLINE
# unset for that one run, or use setup.sh which caches it)
.venv/bin/python src/main.py --text
```

### Windows (PowerShell)

Windows is supported for development only; the scripts are bash, so install manually:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Download Qwen3-0.6B-Q4_K_M.gguf (URL above) into models\llm\
.\.venv\Scripts\python.exe src\main.py --text
$env:MIC_DEVICE = "0"; .\.venv\Scripts\python.exe src\main.py
```

Building `llama-cpp-python` on Windows needs the Visual Studio Build Tools
("Desktop development with C++").

---

## 8. Updating and changing models

### Pull new code

```bash
git pull
scripts/setup.sh --skip-system --skip-models    # picks up new Python dependencies
```

### Use a different Whisper size

Whisper models are cached by `setup.sh`, and `run.sh` blocks downloads, so cache the
new size first:

```bash
WHISPER_MODEL_SIZE=base.en scripts/setup.sh --skip-system
WHISPER_MODEL_SIZE=base.en scripts/run.sh
```

### Use a different language model

Any GGUF chat model works. Point to it and set its chat format:

```bash
LLM_MODEL_PATH=models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
LLM_MODEL=Llama-3.2-1B-Instruct-Q4_K_M \
LLM_CHAT_FORMAT=llama-3 \
scripts/run.sh --text
```

On a 4 GB Pi 4, Qwen3-1.7B at Q4_K_M (~1.1 GB) still fits in memory, but replies
take roughly three times as long, in proportion to its size. Larger models are
not recommended.

---

## 9. Troubleshooting

### Installation

| Symptom | Fix |
|---|---|
| `llama-cpp-python` builds from source on the Pi | The prebuilt wheel was skipped: check `python3 -c "import platform; print(platform.machine())"` prints `aarch64`, then re-run setup |
| `llama-cpp-python` build fails on a laptop with "cmake not found" or compiler errors | `sudo apt install -y build-essential cmake python3-dev` (Linux) or `xcode-select --install` (macOS), then re-run setup |
| `Illegal instruction` when the model loads | A llama.cpp build for a newer ARM CPU got installed. Re-run setup so the baseline wheel is used |
| `This Python is 32-bit` / `uname -m` prints `armv7l` | You installed 32-bit Raspberry Pi OS; re-flash with the 64-bit image |
| `Checksum mismatch` | The download was corrupted: `rm models/llm/*.gguf` and re-run setup |
| `Python 3.11 or newer is required` | Use Raspberry Pi OS Bookworm or newer, or run `PYTHON=python3.11 scripts/setup.sh` |
| `OSError: PortAudio library not found` | `sudo apt install -y libportaudio2 portaudio19-dev` |

### Running

| Symptom | Fix |
|---|---|
| `No .venv found` / `Local model missing` | Run `scripts/setup.sh` from the project root |
| `Could not find the project root` | The scripts locate the repo from their own path (symlinks are followed). If you moved them out of the repo, set `COMPANION_ROOT=/path/to/makeathon21_new` |
| `set: Illegal option -o pipefail` | Old scripts run with `sh` (dash). Pull the latest version, which re-runs itself under bash, or start it with `bash scripts/run.sh` |
| `bash\r: No such file or directory` / `$'\r': command not found` | The scripts have Windows line endings. Pull again (`.gitattributes` now forces LF) or fix in place: `sed -i 's/\r$//' scripts/*.sh` |
| `Do not run setup with sudo` | Run `scripts/setup.sh` as your normal user; it calls `sudo` only for `apt`. If a root-owned `.venv` exists from an earlier sudo run, remove it with `sudo rm -r .venv` and re-run setup |
| Error mentioning `HF_HUB_OFFLINE` or "cannot find the requested files in the local cache" | That Whisper size was never cached: run setup with the same `WHISPER_MODEL_SIZE` (section 8) |
| `No microphone input devices were found` | Check `arecord -l`; add your user to the audio group: `sudo usermod -aG audio $USER`, then log out and in |
| `Device N is not an available microphone` | `MIC_DEVICE` points at an output-only device; pick one whose `--list-mics` entry shows at least `1 in` |
| It never starts recording | Your mic is quiet: lower `SPEECH_RMS_THRESHOLD` (e.g. `200`) |
| Every recording runs the full 15 s | Background noise never drops below the threshold: raise `SPEECH_RMS_THRESHOLD` (e.g. `800`) |
| `microphone input overflowed` warnings | The CPU is overloaded: close other programs, check `vcgencmd get_throttled` |
| `audio open error: Unknown error 524` | espeak-ng's own playback found no sound server. Update to this version (it plays through `aplay` instead) and set `AUDIO_OUTPUT_DEVICE=plughw:CARD=Headphones` for the 3.5 mm jack (section 4.5) |
| `Speech output failed: aplay: ... No such file or directory` / `Device or resource busy` | Wrong device name, or a desktop sound server holds the card: check `scripts/run.sh --list-speakers`; on Pi OS Desktop use `AUDIO_OUTPUT_DEVICE=default` |
| No spoken reply, no error | Audio is going to HDMI or is muted: set `AUDIO_OUTPUT_DEVICE` (section 4.5) and raise volume with `alsamixer` |
| Replies are slow | Use `WHISPER_MODEL_SIZE=tiny.en`, lower `LLM_MAX_TOKENS`, check cooling |
| "I couldn't find that in your personal records" although the data has it | The question uses a word your records don't. Add it to `SYNONYMS` in `src/companion/brain/knowledge.py`, or use the record's own wording (see configuration.md, *Writing records that answer well*) |

### Logs

`logs/assistant.log` holds one JSON line per event:

| Event | Meaning |
|---|---|
| `startup` | Time to load all models |
| `stt` | Speech-to-text duration |
| `llm_generation` | Language model duration |
| `online_lookup_refused` | A lookup was blocked or no provider exists |
| `turn_failed` | An error occurred; the assistant recovered and kept running |

```bash
tail -f logs/assistant.log
```

---

## 10. What is not committed to Git

These are machine-specific or large, and are recreated by `setup.sh`:

- `.venv/`: Python environment
- `models/llm/*.gguf`: language model
- Hugging Face cache (`~/.cache/huggingface`): Whisper model
- `audio/*.wav`: test recordings
- `logs/*.log`: runtime logs
