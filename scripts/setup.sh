#!/usr/bin/env bash
# One-time setup for the offline AI companion on Raspberry Pi OS (64-bit),
# Debian/Ubuntu, or macOS.
#
# Usage: scripts/setup.sh [--skip-system] [--skip-models] [--skip-tests]
#
# After this finishes, every model is cached locally and scripts/run.sh
# starts the assistant with all model hubs forced offline.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
MODEL_DIR="$ROOT/models/llm"
MODEL_FILE="Qwen3-0.6B-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/$MODEL_FILE"
MODEL_SHA256="ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a"

# PyPI only ships llama-cpp-python as source, which takes 20-40 minutes to
# compile on a Pi 4. This prebuilt wheel (same version as requirements.txt)
# uses baseline ARMv8 NEON only, so it runs on the Pi 4's Cortex-A72.
LLAMA_WHEEL_URL="https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35/llama_cpp_python-0.3.35-py3-none-manylinux2014_aarch64.manylinux_2_17_aarch64.whl"
LLAMA_WHEEL_SHA256="b5a4abd4d1d506d6f06b997c21d0532670a3f5350c9e6cfb3e54c33bf1322584"

SKIP_SYSTEM=0
SKIP_MODELS=0
SKIP_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --skip-system) SKIP_SYSTEM=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$1"; }

IS_PI=0
if grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
  IS_PI=1
fi
OS="$(uname -s)"
echo "Platform: $OS $(uname -m)$([ "$IS_PI" = 1 ] && echo ' (Raspberry Pi)')"
if [ -r /proc/meminfo ]; then
  MEM_MB=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1024 ))
  echo "Memory: ${MEM_MB} MB (the companion needs about 1.1 GB at peak)"
fi

# --- System packages ---------------------------------------------------------
if [ "$SKIP_SYSTEM" = 0 ]; then
  if [ "$OS" = "Linux" ] && command -v apt-get >/dev/null; then
    step "Installing system packages (sudo)"
    # build-essential + cmake: llama-cpp-python compiles from source on ARM.
    # portaudio: microphone access. espeak-ng: offline voice for pyttsx3.
    sudo apt-get update
    sudo apt-get install -y \
      python3 python3-venv python3-dev python3-pip \
      build-essential cmake git curl \
      portaudio19-dev libportaudio2 \
      espeak-ng \
      alsa-utils
  elif [ "$OS" = "Darwin" ]; then
    step "macOS: no system packages required (sounddevice bundles PortAudio)"
    xcode-select -p >/dev/null 2>&1 || {
      echo "Xcode command line tools are needed to build llama-cpp-python:"
      echo "  xcode-select --install"
      exit 1
    }
  else
    echo "Unsupported system; install Python 3.11+, a C/C++ toolchain, cmake, and PortAudio manually."
  fi
fi

# --- Python environment ------------------------------------------------------
step "Creating Python virtual environment in .venv"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || {
  echo "Python 3.11 or newer is required (found $("$PYTHON" --version 2>&1))." >&2
  exit 1
}
PY_ARCH="$("$PYTHON" -c 'import platform; print(platform.machine())')"
case "$PY_ARCH" in
  armv7l|armv6l)
    echo "This Python is 32-bit ($PY_ARCH). Use 64-bit Raspberry Pi OS; the models need aarch64." >&2
    exit 1 ;;
esac
[ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel

step "Installing Python dependencies"
if [ "$OS" = "Linux" ] && [ "$PY_ARCH" = "aarch64" ]; then
  echo "Using the prebuilt llama-cpp-python wheel for aarch64 (no compile)."
  # pip verifies the #sha256 fragment and refuses a mismatching download.
  "$VENV/bin/python" -m pip install "llama-cpp-python @ $LLAMA_WHEEL_URL#sha256=$LLAMA_WHEEL_SHA256"
else
  echo "llama-cpp-python will be compiled for this machine (a few minutes)."
fi
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"

# --- Models ------------------------------------------------------------------
if [ "$SKIP_MODELS" = 0 ]; then
  step "Downloading local LLM ($MODEL_FILE, ~400 MB)"
  mkdir -p "$MODEL_DIR"
  if [ ! -f "$MODEL_DIR/$MODEL_FILE" ]; then
    curl -fL --retry 3 -C - -o "$MODEL_DIR/$MODEL_FILE.part" "$MODEL_URL"
    mv "$MODEL_DIR/$MODEL_FILE.part" "$MODEL_DIR/$MODEL_FILE"
  else
    echo "Already present."
  fi

  if command -v sha256sum >/dev/null; then
    ACTUAL="$(sha256sum "$MODEL_DIR/$MODEL_FILE" | cut -d' ' -f1)"
  else
    ACTUAL="$(shasum -a 256 "$MODEL_DIR/$MODEL_FILE" | cut -d' ' -f1)"
  fi
  if [ "$ACTUAL" != "$MODEL_SHA256" ]; then
    echo "Checksum mismatch for $MODEL_FILE; delete it and re-run setup." >&2
    exit 1
  fi
  echo "Checksum OK."

  step "Caching Whisper speech-to-text model"
  # Uses the same Pi-aware default as the app (tiny.en on a Pi, base.en elsewhere).
  (cd "$ROOT/src" && "$VENV/bin/python" - <<'EOF'
from faster_whisper import WhisperModel
from companion.config import STTConfig

size = STTConfig.from_env().model_size
WhisperModel(size, device="cpu", compute_type="int8")
print(f"Whisper '{size}' cached.")
EOF
  )
fi

# --- Verification ------------------------------------------------------------
mkdir -p "$ROOT/logs" "$ROOT/audio"
if [ "$SKIP_TESTS" = 0 ]; then
  step "Running tests"
  (cd "$ROOT" && "$VENV/bin/python" -m unittest discover -s tests)
fi

step "Setup complete"
cat <<EOF
Next:
  scripts/run.sh --text      type to the assistant (checks the LLM)
  scripts/run.sh             speak to it (list microphones with: scripts/run.sh --list-mics)
EOF
