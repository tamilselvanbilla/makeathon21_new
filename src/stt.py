from faster_whisper import WhisperModel

MODEL_SIZE = "tiny"
LANGUAGE = "en"

print("Loading Whisper...")
model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)

print("Model loaded.")

segments, info = model.transcribe(
    "audio/test.wav",
    beam_size=5,
    language=LANGUAGE,
)

print("\nDetected language:", info.language)

print("\nTranscription:")

for segment in segments:
    print(segment.text)