"""Microphone capture. Audio is held in memory only and never written to disk.

The microphone is read as a stream of 80 ms frames of 16 kHz int16 audio.
`record_command` works on any iterator of such frames, so it is testable
without hardware.
"""

import sys
from contextlib import closing
from typing import Callable, Iterable, Iterator

import numpy as np

from ..config import CaptureConfig

TARGET_RATE = 16_000
FRAME_SAMPLES = 1280
FRAME_SECONDS = FRAME_SAMPLES / TARGET_RATE  # 0.08 s


def record_command(
    frames: Iterable[np.ndarray],
    config: CaptureConfig,
    max_wait_seconds: float,
) -> np.ndarray:
    """Record until speech is followed by silence.

    Returns float32 mono 16 kHz audio, or an empty array when no speech started
    within `max_wait_seconds` or the frames ran out (muted; audio is discarded).
    """
    silence_frames_needed = max(1, round(config.silence_seconds / FRAME_SECONDS))
    chunks: list[np.ndarray] = []
    silent_frames = 0
    speech_detected = False
    elapsed = 0.0

    for frame in frames:
        chunks.append(frame)
        elapsed += FRAME_SECONDS
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        if rms >= config.speech_rms_threshold:
            speech_detected = True
            silent_frames = 0
        elif speech_detected:
            silent_frames += 1
            if silent_frames >= silence_frames_needed:
                break
        if speech_detected and elapsed >= config.max_record_seconds:
            break
        if not speech_detected and elapsed >= max_wait_seconds:
            return np.empty(0, dtype=np.float32)
    else:
        return np.empty(0, dtype=np.float32)  # stream ended: muted mid-recording

    return np.concatenate(chunks).astype(np.float32) / 32768.0


def choose_input_device(configured: int | None) -> int:
    """Show microphones and select the configured, chosen, or default input."""
    import sounddevice as sd

    inputs = [
        (index, info)
        for index, info in enumerate(sd.query_devices())
        if info["max_input_channels"] > 0
    ]
    if not inputs:
        raise RuntimeError("No microphone input devices were found.")

    default_device = sd.default.device[0]
    print("Available microphones:")
    for index, info in inputs:
        marker = " (default)" if index == default_device else ""
        print(f"  {index}: {info['name']}{marker}")

    fallback = default_device if default_device is not None and default_device >= 0 else inputs[0][0]
    if configured is not None:
        selected = configured
    elif sys.stdin.isatty():
        choice = input(f"Microphone index [Enter for {fallback}]: ").strip()
        selected = fallback if not choice else int(choice)
    else:
        # Headless (e.g. systemd on the Pi): never block waiting for input.
        selected = fallback

    if selected not in {index for index, _ in inputs}:
        raise RuntimeError(f"Device {selected} is not an available microphone.")
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
