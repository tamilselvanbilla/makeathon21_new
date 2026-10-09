"""Microphone capture. Audio is held in memory only and never written to disk."""

import sys
import time
from typing import Callable

import numpy as np

from ..config import CaptureConfig

BLOCK_SECONDS = 0.1


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

    def _pick_sample_rate(self) -> int:
        """Prefer capturing at 16 kHz natively to skip resampling on the Pi."""
        try:
            self._sd.check_input_settings(
                device=self.device,
                channels=1,
                samplerate=self.config.sample_rate,
                dtype="int16",
            )
            return self.config.sample_rate
        except (self._sd.PortAudioError, ValueError):
            return int(self._sd.query_devices(self.device, "input")["default_samplerate"])

    def record_utterance(self, should_abort: Callable[[], bool] = lambda: False) -> np.ndarray:
        """Record until speech is followed by silence.

        Returns float32 mono audio at the configured sample rate, or an empty
        array when no speech was heard or `should_abort` became true (e.g. the
        mute switch was flipped mid-recording, in which case audio is discarded).
        """
        cfg = self.config
        block_size = max(1, int(self.sample_rate * BLOCK_SECONDS))
        silence_blocks_needed = max(1, int(cfg.silence_seconds / BLOCK_SECONDS))
        chunks: list[np.ndarray] = []
        silent_blocks = 0
        speech_detected = False
        started_at = time.monotonic()

        with self._sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            dtype="int16",
            blocksize=block_size,
        ) as stream:
            while True:
                if should_abort():
                    chunks.clear()
                    return np.empty(0, dtype=np.float32)

                chunk, overflowed = stream.read(block_size)
                if overflowed:
                    print("Warning: microphone input overflowed; audio may be incomplete.")

                chunk = chunk.reshape(-1)
                chunks.append(chunk.copy())
                rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

                if rms >= cfg.speech_rms_threshold:
                    speech_detected = True
                    silent_blocks = 0
                elif speech_detected:
                    silent_blocks += 1
                    if silent_blocks >= silence_blocks_needed:
                        break

                elapsed = time.monotonic() - started_at
                if speech_detected and elapsed >= cfg.max_record_seconds:
                    break
                if not speech_detected and elapsed >= cfg.max_wait_for_speech_seconds:
                    return np.empty(0, dtype=np.float32)

        audio = np.concatenate(chunks).astype(np.float32) / 32768.0
        return self._to_target_rate(audio)

    def _to_target_rate(self, audio: np.ndarray) -> np.ndarray:
        target = self.config.sample_rate
        if self.sample_rate == target:
            return audio
        from scipy.signal import resample_poly

        divisor = np.gcd(self.sample_rate, target)
        return resample_poly(audio, target // divisor, self.sample_rate // divisor).astype(np.float32)
