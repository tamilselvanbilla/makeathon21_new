from pathlib import Path

from faster_whisper import WhisperModel

MODEL_SIZE = "base"
LANGUAGE = "en"
AUDIO_PATH = Path(__file__).resolve().parents[1] / "audio" / "test.wav"

print("Loading Whisper...")
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8",
)

print("Model loaded.")

segments, info = model.transcribe(
    str(AUDIO_PATH),
    beam_size=5,
    language=LANGUAGE,
    vad_filter=True,
)

print("\nDetected language:", info.language)

print("\nTranscription:")

for segment in segments:
    print(segment.text)