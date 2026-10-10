"""Microphone capture. Audio is held in memory only and never written to disk.

The microphone is read as a stream of 80 ms frames of 16 kHz int16 audio.
`record_command` works on any iterator of such frames, so it is testable
without hardware.
"""

import re
from collections import deque
from contextlib import closing
from typing import Callable, Iterable, Iterator

import numpy as np

from ..config import CaptureConfig

TARGET_RATE = 16_000
FRAME_SAMPLES = 1280
FRAME_SECONDS = FRAME_SAMPLES / TARGET_RATE  # 0.08 s
PRE_ROLL_SECONDS = 0.5  # audio kept from just before speech starts


def record_command(
    frames: Iterable[np.ndarray],
    config: CaptureConfig,
    max_wait_seconds: float,
) -> np.ndarray:
    """Record until speech is followed by silence.

    Only the last `PRE_ROLL_SECONDS` before speech starts are kept (so the first
    syllable isn't clipped); earlier silence or room noise never reaches Whisper,
    which on a Pi made transcription take 8-16 s. Speech must be louder than
    SPEECH_RMS_THRESHOLD for SPEECH_ONSET_FRAMES frames in a row, so a click or
    bump doesn't start a recording. MAX_RECORD_SECONDS counts from speech start.

    Returns float32 mono 16 kHz audio, or an empty array when no speech started
    within `max_wait_seconds` or the frames ran out (muted; audio is discarded).
    """
    silence_frames_needed = max(1, round(config.silence_seconds / FRAME_SECONDS))
    onset_frames = max(1, config.speech_onset_frames)
    pre_roll: deque[np.ndarray] = deque(maxlen=onset_frames + round(PRE_ROLL_SECONDS / FRAME_SECONDS))
    chunks: list[np.ndarray] = []
    loud_run = silent_frames = 0
    waited = spoken = 0.0

    for frame in frames:
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        loud = rms >= config.speech_rms_threshold
        if not chunks:  # waiting for speech
            waited += FRAME_SECONDS
            pre_roll.append(frame)
            loud_run = loud_run + 1 if loud else 0
            if loud_run >= onset_frames:
                chunks = list(pre_roll)
            elif waited >= max_wait_seconds:
                return np.empty(0, dtype=np.float32)
            continue

        chunks.append(frame)
        spoken += FRAME_SECONDS
        silent_frames = 0 if loud else silent_frames + 1
        if silent_frames >= silence_frames_needed or spoken >= config.max_record_seconds:
            break
    else:
        return np.empty(0, dtype=np.float32)  # stream ended: muted mid-recording

    return np.concatenate(chunks).astype(np.float32) / 32768.0


# Names that suggest a real microphone, and names of inputs that are not one.
LIKELY_MIC = re.compile(r"usb|respeaker|seeed|mic|headset|webcam", re.IGNORECASE)
NOT_A_MIC = re.compile(r"monitor|loopback|virtual|blackhole|soundflower|hdmi", re.IGNORECASE)


def pick_microphone(inputs: list[tuple[int, str]], default: int | None, configured: str | None) -> int:
    """Choose an input without asking.

    1. MIC_DEVICE, as an index ("1") or part of the name ("USB", "ReSpeaker");
    2. the system default input;
    3. the first input that looks like a real microphone (USB, ReSpeaker, "mic"),
       skipping loopback/monitor devices; a Pi has no built-in microphone, so its
       default is often unset or wrong;
    4. the first input.
    """
    indexes = [index for index, _ in inputs]
    if configured:
        if configured.strip().isdigit():
            if int(configured) in indexes:
                return int(configured)
            raise RuntimeError(f"MIC_DEVICE={configured} is not an available microphone.")
        matches = [index for index, name in inputs if configured.casefold() in name.casefold()]
        if matches:
            return matches[0]
        raise RuntimeError(f"No microphone name contains MIC_DEVICE={configured!r}.")
    if default is not None and default in indexes:
        return default
    likely = [index for index, name in inputs if LIKELY_MIC.search(name) and not NOT_A_MIC.search(name)]
    real = [index for index, name in inputs if not NOT_A_MIC.search(name)]
    return (likely or real or indexes)[0]


def choose_input_device(configured: str | None) -> int:
    """List microphones and select one automatically (see pick_microphone)."""
    import sounddevice as sd

    inputs = [
        (index, info["name"])
        for index, info in enumerate(sd.query_devices())
        if info["max_input_channels"] > 0
    ]
    if not inputs:
        raise RuntimeError("No microphone input devices were found.")

    default = sd.default.device[0]
    selected = pick_microphone(inputs, default if default is not None and default >= 0 else None, configured)
    print("Available microphones:")
    for index, name in inputs:
        marks = [mark for mark, on in (("default", index == default), ("selected", index == selected)) if on]
        print(f"  {index}: {name}" + (f" ({', '.join(marks)})" if marks else ""))
    return selected


class MicrophoneCapture:
    def __init__(self, config: CaptureConfig, device: int):
        import sounddevice as sd

        self._sd = sd
        self.config = config
        self.device = device
        self.device_name = sd.query_devices(device, "input")["name"]
        self.sample_rate = self._pick_sample_rate()
        self.block_size = round(self.sample_rate * FRAME_SECONDS)

    def _pick_sample_rate(self) -> int:
        """Prefer capturing at 16 kHz natively to skip resampling on the Pi."""
        try:
            self._sd.check_input_settings(
                device=self.device,
                channels=1,
                samplerate=TARGET_RATE,
                dtype="int16",
            )
            return TARGET_RATE
        except (self._sd.PortAudioError, ValueError):
            return int(self._sd.query_devices(self.device, "input")["default_samplerate"])

    def frames(self, should_abort: Callable[[], bool] = lambda: False) -> Iterator[np.ndarray]:
        """Yield 80 ms int16 frames at 16 kHz until `should_abort` returns true."""
        with self._sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            dtype="int16",
            blocksize=self.block_size,
        ) as stream:
            while not should_abort():
                chunk, overflowed = stream.read(self.block_size)
                if overflowed:
                    print("Warning: microphone input overflowed; audio may be incomplete.")
                yield self._to_16k(chunk.reshape(-1))

    def _to_16k(self, chunk: np.ndarray) -> np.ndarray:
        if self.sample_rate == TARGET_RATE:
            return chunk.copy()
        from scipy.signal import resample_poly

        divisor = np.gcd(self.sample_rate, TARGET_RATE)
        audio = resample_poly(chunk.astype(np.float32), TARGET_RATE // divisor, self.sample_rate // divisor)
        return np.clip(audio, -32768, 32767).astype(np.int16)[:FRAME_SAMPLES]

    def record_utterance(
        self,
        should_abort: Callable[[], bool] = lambda: False,
        max_wait_seconds: float | None = None,
    ) -> np.ndarray:
        """Record one utterance: wait up to `max_wait_seconds` for speech to start
        (default MAX_WAIT_FOR_SPEECH_SECONDS), then stop after silence."""
        wait = self.config.max_wait_for_speech_seconds if max_wait_seconds is None else max_wait_seconds
        with closing(self.frames(should_abort)) as frames:
            return record_command(frames, self.config, wait)
