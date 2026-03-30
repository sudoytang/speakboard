from abc import ABC, abstractmethod

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
        import mlx_whisper
        print(f"[speakboard] Loading model {self.model} (first run requires download)...")
        mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), path_or_hf_repo=self.model)
        print("[speakboard] Model loaded.")

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
        )
        text = str(result["text"]).strip()
        language = str(result.get("language", "?"))
        return text, language


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

    def transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        import whisper
        if self._model is None:
            self.load()
        result = self._model.transcribe(
            audio,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
        )
        text = str(result["text"]).strip()
        language = str(result.get("language", "?"))
        return text, language
