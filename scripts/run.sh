#!/usr/bin/env bash
# Start the offline AI companion.
#
# Usage:
#   scripts/run.sh                 voice mode (microphone + Whisper)
#   scripts/run.sh --text          type requests instead of speaking
#   scripts/run.sh --no-tts        print replies instead of speaking
#   scripts/run.sh --offline       disable all online lookups
#   scripts/run.sh --no-wake-word  treat any speech as a request (no wake phrase)
#   scripts/run.sh --list-mics     show microphone indexes for MIC_DEVICE
#   scripts/run.sh --mic-test      record a 5-second clip to audio/test.wav
#   scripts/run.sh --list-speakers show ALSA output devices for AUDIO_OUTPUT_DEVICE
#   scripts/run.sh --speaker-test  speak a test phrase through the configured output
#
# Settings can be overridden per run, e.g. MIC_DEVICE=1 scripts/run.sh
# (see docs/configuration.md for all variables).

# Re-run under bash when started as `sh scripts/<name>.sh` (/bin/sh is dash
# on Raspberry Pi OS and does not support the bash features used below).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

set -euo pipefail
# A CDPATH from the user's profile makes `cd` print paths into $(...) results.
unset CDPATH

# Repository root: independent of the current directory and of symlinks that
# point at this script. Set COMPANION_ROOT to override.
resolve_root() {
  local source="${BASH_SOURCE[0]}" dir
  while [ -L "$source" ]; do
    dir="$(cd -P "$(dirname "$source")" >/dev/null && pwd)"
    source="$(readlink "$source")"
    case "$source" in /*) ;; *) source="$dir/$source" ;; esac
  done
  cd -P "$(dirname "$source")/.." >/dev/null && pwd
}
ROOT="${COMPANION_ROOT:-$(resolve_root)}"
if [ ! -f "$ROOT/src/main.py" ]; then
  echo "Could not find the project root (looked in: $ROOT)." >&2
  echo "Run the script from inside the repository, or set COMPANION_ROOT=/path/to/repo." >&2
  exit 1
fi
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

if [ "$(id -u)" = 0 ]; then
  echo "Warning: running as root; the Whisper model cached by setup.sh for your user will not be found." >&2
fi

if [ ! -x "$PY" ]; then
  echo "No .venv found. Run scripts/setup.sh first." >&2
  exit 1
fi
# Models were cached by setup.sh; never let a model library reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Audio checks do not need the language model.
case "${1:-}" in
  --list-mics)
    exec "$PY" -c "import sounddevice as sd; print(sd.query_devices())"
    ;;
  --mic-test)
    exec "$PY" "$ROOT/scripts/mic_test.py"
    ;;
  --list-speakers)
    aplay -l || true
    echo
    echo "Set AUDIO_OUTPUT_DEVICE to one of these (Pi 4 3.5 mm jack: plughw:CARD=Headphones):"
    aplay -L | grep -E '^(default|plughw:)' || true
    exit 0
    ;;
  --speaker-test)
    shift
    cd "$ROOT/src"
    exec "$PY" -m companion.device.tts "$@"
    ;;
esac

if [ ! -f "${LLM_MODEL_PATH:-$ROOT/models/llm/Qwen3-0.6B-Q4_K_M.gguf}" ]; then
  echo "Local model missing. Run scripts/setup.sh (or set LLM_MODEL_PATH)." >&2
  exit 1
fi

cd "$ROOT"
exec "$PY" src/main.py "$@"
