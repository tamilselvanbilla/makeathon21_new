import sounddevice as sd
from scipy.io.wavfile import write
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
output_path = project_dir / "audio" / "test.wav"
output_path.parent.mkdir(parents=True, exist_ok=True)

duration = 5

input_devices = [
    (index, info)
    for index, info in enumerate(sd.query_devices())
    if info["max_input_channels"] > 0
]
default_device = sd.default.device[0]

print("Available microphones:")
for index, info in input_devices:
    default_label = " (default)" if index == default_device else ""
    print(f"  {index}: {info['name']}{default_label}")

choice = input(f"Microphone index [Enter for {default_device}]: ").strip()
device = default_device if not choice else int(choice)
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