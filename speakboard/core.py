import threading
import time

import numpy as np
import pyperclip
import sounddevice as sd
from pynput import keyboard

from .transcribe import SAMPLE_RATE, Transcriber

HOTKEY = keyboard.Key.alt_r
SILENCE_RMS_THRESHOLD = 0.002


class Whisperer:
    def __init__(self, transcriber: Transcriber) -> None:
        self._transcriber = transcriber
        self._recording = False
        self._audio_frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def run(self) -> None:
        self._transcriber.load()
        print("[speakboard] Hold right Option key to record. Press Ctrl+C to exit.")
        with keyboard.Listener(on_press=self._on_press, on_release=self._on_release) as listener:
            listener.join()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == HOTKEY and not self._recording:
            self._recording = True
            self._audio_frames = []
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
            print("[speakboard] Recording...")

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key == HOTKEY and self._recording:
            self._recording = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            threading.Thread(target=self._transcribe_and_copy, daemon=True).start()

    def _audio_callback(self, indata: np.ndarray, frames: int, t, status) -> None:
        if self._recording:
            self._audio_frames.append(indata.copy())

    def _transcribe_and_copy(self) -> None:
        if not self._audio_frames:
            return

        audio = np.concatenate(self._audio_frames).flatten()
        duration = len(audio) / SAMPLE_RATE
        print(f"[speakboard] Recording duration {duration:.2f}s, starting transcription...")

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < SILENCE_RMS_THRESHOLD:
            print("[speakboard] No speech detected (silence).")
            return

        t0 = time.perf_counter()
        text, language = self._transcriber.transcribe(audio)
        print(f"[speakboard] Transcription took {time.perf_counter() - t0:.2f}s")

        if not text:
            print("[speakboard] No speech detected.")
            return

        print(f"[speakboard] [{language}] {text}")
        pyperclip.copy(text)
        print("[speakboard] Copied to clipboard.")
