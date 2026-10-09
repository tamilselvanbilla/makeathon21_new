#!/usr/bin/env bash
# Start the offline AI companion.
#
# Usage:
#   scripts/run.sh                 voice mode (microphone + Whisper)
#   scripts/run.sh --text          type requests instead of speaking
#   scripts/run.sh --no-tts        print replies instead of speaking
#   scripts/run.sh --offline       disable all online lookups
#   scripts/run.sh --list-mics     show microphone indexes for MIC_DEVICE
#   scripts/run.sh --mic-test      record a 5-second clip to audio/test.wav
#
# Settings can be overridden per run, e.g. MIC_DEVICE=1 scripts/run.sh
# (see docs/configuration.md for all variables).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "No .venv found. Run scripts/setup.sh first." >&2
  exit 1
fi
if [ ! -f "${LLM_MODEL_PATH:-$ROOT/models/llm/Qwen3-0.6B-Q4_K_M.gguf}" ]; then
  echo "Local model missing. Run scripts/setup.sh (or set LLM_MODEL_PATH)." >&2
  exit 1
fi

# Models were cached by setup.sh; never let a model library reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

case "${1:-}" in
  --list-mics)
    exec "$PY" -c "import sounddevice as sd; print(sd.query_devices())"
    ;;
  --mic-test)
    exec "$PY" "$ROOT/scripts/mic_test.py"
    ;;
esac

cd "$ROOT"
exec "$PY" src/main.py "$@"
