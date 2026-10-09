"""Record a 5-second test clip to audio/test.wav with the automatically selected
microphone (set MIC_DEVICE to an index or part of a name to choose another)."""

import os
import sys
from pathlib import Path

import sounddevice as sd
from scipy.io.wavfile import write

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir / "src"))
from companion.audio.capture import choose_input_device  # noqa: E402

output_path = project_dir / "audio" / "test.wav"
output_path.parent.mkdir(parents=True, exist_ok=True)

duration = 5

device = choose_input_device(os.getenv("MIC_DEVICE") or None)
device_info = sd.query_devices(device, "input")
sample_rate = int(device_info["default_samplerate"])
channels = 1

sd.check_input_settings(
    device=device,
    channels=channels,
    samplerate=sample_rate,
    dtype="int16",
)

print(f"Recording from {device_info['name']} at {sample_rate} Hz...")
audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    device=device,
    channels=channels,
    dtype="int16",
)

sd.wait()

write(output_path, sample_rate, audio)

print(f"Saved: {output_path}")