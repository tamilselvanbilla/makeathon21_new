"""Microphone capture. Audio is held in memory only and never written to disk.

The microphone is read as a stream of 80 ms frames of 16 kHz int16 audio.
`record_command` works on any iterator of such frames, so it is testable
without hardware.
"""

import queue
import re
import shutil
import subprocess
import sys
import threading
from contextlib import closing
from typing import Callable, Iterable, Iterator

import numpy as np

from ..config import CaptureConfig

TARGET_RATE = 16_000
FRAME_SAMPLES = 1280
FRAME_SECONDS = FRAME_SAMPLES / TARGET_RATE  # 0.08 s
# Audio waiting to be processed: up to 30 s, so a long stall (model loading, a busy
# Pi) delays processing instead of losing audio; beyond that, the oldest is dropped.
MAX_QUEUED_BLOCKS = int(30 / FRAME_SECONDS)


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


# PortAudio names ALSA devices like "USB PnP Sound Device: Audio (hw:2,0)".
ALSA_HW = re.compile(r"\(hw:(\d+),(\d+)\)")


def alsa_device_for(name: str) -> str | None:
    """The ALSA `plughw` device for a PortAudio device name, or None. `plughw`
    converts any microphone's native rate and format to 16 kHz mono in ALSA."""
    match = ALSA_HW.search(name)
    return f"plughw:{match.group(1)},{match.group(2)}" if match else None


class ArecordStream:
    """Records with ALSA's `arecord` in a separate process.

    On a Raspberry Pi, PortAudio's audio thread has to call back into Python for
    every block, and while Whisper or the LLM keep Python busy the callback can run
    too late for the sound card's buffer, so ALSA overruns and audio is lost.
    `arecord` is C code in its own process with a 1-second buffer; this class only
    reads finished 80 ms blocks from its output. Overruns it reports are counted.
    """

    def __init__(self, device: str, on_block: Callable[[np.ndarray], None], on_overrun: Callable[[], None],
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen):
        self.device = device
        self._on_block = on_block
        self._on_overrun = on_overrun
        self._popen = popen
        self._process = None

    def start(self) -> None:
        self._process = self._popen(
            # No -q: quiet mode may hide the "overrun!!!" messages counted below.
            ["arecord", "-D", self.device, "-f", "S16_LE", "-c", "1", "-r", str(TARGET_RATE), "-t", "raw",
             "--buffer-time=1000000", "--period-time=80000"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        threading.Thread(target=self._read_audio, daemon=True).start()
        threading.Thread(target=self._read_errors, daemon=True).start()

    def _read_audio(self) -> None:
        size = FRAME_SAMPLES * 2  # int16
        stdout = self._process.stdout
        buffer = b""
        while True:
            chunk = stdout.read(size - len(buffer))
            if not chunk:
                return  # arecord stopped
            buffer += chunk
            if len(buffer) == size:
                self._on_block(np.frombuffer(buffer, dtype=np.int16).copy())
                buffer = b""

    def _read_errors(self) -> None:
        for line in self._process.stderr:
            if b"overrun" in line:
                self._on_overrun()

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def close(self) -> None:
        self._process = None


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
        # On Linux, record with arecord when the device has an ALSA hardware name
        # (see ArecordStream); otherwise, or with MIC_BACKEND=portaudio, use PortAudio.
        self.alsa_device = alsa_device_for(self.device_name) if sys.platform.startswith("linux") else None
        use_arecord = config.backend == "arecord" or (
            config.backend == "auto" and self.alsa_device is not None and shutil.which("arecord") is not None
        )
        if config.backend == "arecord" and self.alsa_device is None:
            self.alsa_device = "default"
        self.backend = "arecord" if use_arecord else "portaudio"
        self.sample_rate = TARGET_RATE if use_arecord else self._pick_sample_rate()
        self.block_size = round(self.sample_rate * FRAME_SECONDS)
        self._stream = None
        self._blocks: queue.Queue = queue.Queue()
        self._overflows = 0

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
        """Yield 80 ms int16 frames at 16 kHz until `should_abort` returns true.

        The microphone stays open between utterances, so speech that starts while the
        previous utterance is being transcribed is still captured (on a Pi that takes
        seconds, and used to clip "Hey Sam"). PortAudio's audio thread queues each block
        as it arrives, so a slow moment here (resampling, garbage collection, a busy Pi)
        delays processing instead of overflowing the sound card's buffer. When
        `should_abort` fires (the mute switch), the microphone is closed.
        """
        self._open()
        while not should_abort():
            try:
                chunk = self._blocks.get(timeout=0.2)
            except queue.Empty:
                continue  # no audio yet; re-check mute
            yield self._to_16k(chunk.reshape(-1))
        self.close()  # muted: the microphone is off, and queued audio is discarded
        self._report_overflow()

    def _open(self) -> None:
        if self._stream is not None:
            return
        self._blocks = queue.Queue(maxsize=MAX_QUEUED_BLOCKS)
        self._overflows = 0
        if getattr(self, "backend", "portaudio") == "arecord":
            self._stream = ArecordStream(self.alsa_device, self._push, self._count_overflow)
            self._stream.start()
            return

        first = {"block": True}

        def on_audio(indata, frames, time_info, status) -> None:
            # ALSA often flags an overflow on the first block of a new stream; harmless.
            if status.input_overflow and not first["block"]:
                self._count_overflow()
            first["block"] = False
            self._push(indata.copy())

        self._stream = self._sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            dtype="int16",
            blocksize=self.block_size,
            latency=0.5,  # half a second of device buffer rides out scheduling hiccups
            callback=on_audio,
        )
        self._stream.start()

    def _push(self, block: np.ndarray) -> None:
        try:
            self._blocks.put_nowait(block)
        except queue.Full:  # 30 s behind: drop the oldest block, keep the newest
            self._count_overflow()
            self._blocks.get_nowait()
            self._blocks.put_nowait(block)

    def _count_overflow(self) -> None:
        self._overflows += 1

    def discard_pending(self) -> None:
        """Drop queued audio, e.g. the assistant's own voice captured while it spoke."""
        if self._stream is not None:
            while True:
                try:
                    self._blocks.get_nowait()
                except queue.Empty:
                    break

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.discard_pending()

    def _report_overflow(self) -> None:
        if self._overflows:
            print(f"Warning: microphone overflowed {self._overflows} time(s); some audio was lost.")
            self._overflows = 0

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
            audio = record_command(frames, self.config, wait)
        self._report_overflow()
        return audio
