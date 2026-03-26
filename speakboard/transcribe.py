from abc import ABC, abstractmethod

import mlx_whisper
import numpy as np

SAMPLE_RATE = 16000  # Required by Whisper models


class Transcriber(ABC):
    @abstractmethod
    def load(self) -> None:
        """Warm up the model (download if needed)."""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """Return (text, detected_language)."""
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
        print(f"[speakboard] Loading model {self.model} (first run requires download)...")
        mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=self.model)
        print("[speakboard] Model loaded.")

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
        )
        text = str(result["text"]).strip()
        language = str(result.get("language", "?"))
        return text, language
