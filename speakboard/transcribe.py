from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import gzip
import queue as _queue

import numpy as np

SAMPLE_RATE = 16000  # Required by Whisper models
MAX_AUDIO_SECONDS = 15


def is_hallucinated(text: str) -> bool:
    """Detect repetitive hallucination loops (e.g. '相信相信相信...')."""
    if len(text) < 10:
        return False
    compressed = len(gzip.compress(text.encode("utf-8")))
    ratio = len(text.encode("utf-8")) / compressed
    return ratio > 5.0


@dataclass
class SegmentResult:
    index: int
    duration_seconds: float
    text: str
    language: str
    hallucinated: bool


@dataclass
class TranscribeResult:
    text: str                              # full concatenated text
    language: str
    duration_seconds: float
    split: bool                            # whether silence splitting was used
    segments: list[SegmentResult] = field(default_factory=list)


def split_on_silence(
    audio: np.ndarray,
    min_silence_ms: int = 400,
    silence_threshold: float = 0.01,
) -> list[np.ndarray]:
    """Split audio into segments within MAX_AUDIO_SECONDS.

    Only splits when necessary. When a split is needed, picks the longest
    silence gap within the current window to minimize total segment count.
    Falls back to a hard cut if no silence gap is found.
    """
    max_segment_samples = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)

    if len(audio) <= max_segment_samples:
        return [audio]

    window_ms = 20
    window_size = int(SAMPLE_RATE * window_ms / 1000)
    min_silence_windows = max(1, min_silence_ms // window_ms)

    n_windows = len(audio) // window_size
    rms = np.sqrt(np.array([
        np.mean(audio[i * window_size:(i + 1) * window_size] ** 2)
        for i in range(n_windows)
    ]))
    is_silence = rms < silence_threshold

    # Collect all silence gaps as (midpoint_sample, length_in_windows)
    gaps: list[tuple[int, int]] = []
    i = 0
    while i < len(is_silence):
        if is_silence[i]:
            j = i
            while j < len(is_silence) and is_silence[j]:
                j += 1
            if j - i >= min_silence_windows:
                mid_sample = ((i + j) // 2) * window_size
                gaps.append((mid_sample, j - i))
            i = j
        else:
            i += 1

    # Greedy: advance as far as possible, only split when we must
    segments = []
    current = 0
    while len(audio) - current > max_segment_samples:
        window_end = current + max_segment_samples
        candidates = [(mid, length) for mid, length in gaps if current < mid < window_end]
        if candidates:
            split_at = max(candidates, key=lambda g: g[1])[0]  # longest gap
        else:
            split_at = window_end  # no silence found, hard cut
        segments.append(audio[current:split_at])
        current = split_at

    segments.append(audio[current:])
    return segments


def _transcribe_with_splitting(audio: np.ndarray, chunk_fn) -> "TranscribeResult":
    total_duration = len(audio) / SAMPLE_RATE

    def _process_chunk(i: int, chunk: np.ndarray) -> SegmentResult:
        result = chunk_fn(chunk)
        text = str(result["text"]).strip()
        language = str(result.get("language", "?"))
        hallucinated = is_hallucinated(text)
        if hallucinated or not text:
            print(f"[speakboard] Segment {i+1} discarded (empty or hallucination).")
        return SegmentResult(
            index=i,
            duration_seconds=round(len(chunk) / SAMPLE_RATE, 2),
            text=text if not hallucinated else "",
            language=language,
            hallucinated=hallucinated,
        )

    if total_duration <= MAX_AUDIO_SECONDS:
        seg = _process_chunk(0, audio)
        return TranscribeResult(
            text=seg.text,
            language=seg.language,
            duration_seconds=round(total_duration, 2),
            split=False,
            segments=[seg],
        )

    print(f"[speakboard] Audio {total_duration:.1f}s > {MAX_AUDIO_SECONDS}s, splitting on silence...")
    chunks = split_on_silence(audio)
    print(f"[speakboard] Split into {len(chunks)} segments.")
    segment_results = [_process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    language = next((s.language for s in reversed(segment_results) if s.language != "?"), "?")
    return TranscribeResult(
        text="".join(s.text for s in segment_results),
        language=language,
        duration_seconds=round(total_duration, 2),
        split=True,
        segments=segment_results,
    )


class Transcriber(ABC):
    @abstractmethod
    def load(self) -> None:
        """Warm up the model (download if needed)."""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> TranscribeResult:
        """Transcribe audio. Handles long audio via silence splitting."""
        ...


class MLXWhisperTranscriber(Transcriber):
    """Apple Silicon backend via mlx-whisper."""

    def __init__(
        self,
        model: str = "mlx-community/whisper-large-v3-mlx",
        initial_prompt: str = "好的，我明白了，我将使用简体中文。",
    ):
        self.model = model
        self.initial_prompt = initial_prompt

    def load(self) -> None:
        import mlx_whisper
        print(f"[speakboard] Loading model {self.model} (first run requires download)...")
        mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=self.model)
        print("[speakboard] Model loaded.")

    def _transcribe_chunk(self, audio: np.ndarray):
        import mlx_whisper
        return mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
            compression_ratio_threshold=1.8,
        )

    def transcribe(self, audio: np.ndarray) -> TranscribeResult:
        return _transcribe_with_splitting(audio, self._transcribe_chunk)


class CPUWhisperTranscriber(Transcriber):
    """Cross-platform CPU backend via openai-whisper (PyTorch)."""

    def __init__(
        self,
        model: str = "large-v3",
        initial_prompt: str = "好的，我明白了，我将使用简体中文。",
    ):
        self.model_name = model
        self.initial_prompt = initial_prompt
        self._model = None

    def load(self) -> None:
        import whisper
        print(f"[speakboard] Loading model {self.model_name} (first run requires download)...")
        self._model = whisper.load_model(self.model_name)
        print("[speakboard] Model loaded.")

    def _transcribe_chunk(self, audio: np.ndarray):
        import whisper
        if self._model is None:
            self.load()
        return self._model.transcribe(
            audio,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
        )

    def transcribe(self, audio: np.ndarray) -> TranscribeResult:
        return _transcribe_with_splitting(audio, self._transcribe_chunk)


# ---------------------------------------------------------------------------
# Subprocess worker (must be top-level for multiprocessing spawn pickling)
# ---------------------------------------------------------------------------

def _worker_process(factory, req_q, res_q) -> None:
    """Runs in a subprocess. Loads the model once, then serves transcription requests."""
    transcriber = factory()
    transcriber.load()
    res_q.put(("ready", None))
    while True:
        audio = req_q.get()
        if audio is None:
            break
        try:
            result = transcriber.transcribe(audio)
            res_q.put(("ok", result))
        except Exception as e:
            res_q.put(("error", str(e)))


class WatchdogTranscriber(Transcriber):
    """Wraps any Transcriber in a subprocess with a watchdog timer.

    Timeout per request = max(min_timeout_seconds, audio_duration * timeout_multiplier).
    If inference exceeds the timeout, the subprocess is killed and restarted.
    Thread-safe: internal lock ensures only one inference runs at a time.

    Suggested defaults:
      MLX backend:  min_timeout_seconds=5,  timeout_multiplier=1
      CPU backend:  min_timeout_seconds=10, timeout_multiplier=5
    """

    def __init__(
        self,
        factory,
        min_timeout_seconds: float = 5.0,
        timeout_multiplier: float = 1.0,
    ):
        self._factory = factory
        self._min_timeout = min_timeout_seconds
        self._multiplier = timeout_multiplier
        self._proc = None
        self._req_q = None
        self._res_q = None
        import threading
        self._lock = threading.Lock()

    def load(self) -> None:
        self._start_worker()

    def _start_worker(self) -> None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self._req_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_worker_process,
            args=(self._factory, self._req_q, self._res_q),
            daemon=True,
        )
        self._proc.start()
        kind, _ = self._res_q.get()  # block until model is loaded and verified
        assert kind == "ready"

    def _kill_and_restart(self) -> None:
        print("[speakboard] Watchdog: killing hung inference process, restarting...")
        if self._proc is not None:
            self._proc.kill()
            self._proc.join()
        self._start_worker()

    def transcribe(self, audio: np.ndarray) -> TranscribeResult:
        audio_duration = len(audio) / SAMPLE_RATE
        timeout = max(self._min_timeout, audio_duration * self._multiplier)
        with self._lock:
            self._req_q.put(audio)
            try:
                kind, value = self._res_q.get(timeout=timeout)
            except _queue.Empty:
                self._kill_and_restart()
                raise RuntimeError(
                    f"Inference timed out after {timeout:.1f}s (audio: {audio_duration:.1f}s)"
                )
        if kind == "error":
            raise RuntimeError(f"Inference subprocess error: {value}")
        return value
