import io
import re

import httpx
import numpy as np
import pytest
import soundfile as sf
from jiwer import wer

# WER threshold: large-v3 on LibriSpeech test-clean typically achieves ~2-3%.
# 15% gives plenty of headroom for quantized models and varied conditions.
WER_THRESHOLD = 0.15


def normalize(text: str) -> str:
    """Lowercase and strip punctuation for fair WER comparison."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def to_wav_bytes(array: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, array, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def test_health(server):
    r = httpx.get(f"{server}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_empty_body_returns_400(server):
    r = httpx.post(f"{server}/transcribe", content=b"")
    assert r.status_code == 400


def test_invalid_audio_returns_422(server):
    r = httpx.post(f"{server}/transcribe", content=b"this is not audio")
    assert r.status_code == 422


@pytest.mark.parametrize("idx", range(5))
def test_transcription_wer(server, librispeech_samples, idx):
    sample = librispeech_samples[idx]
    # datasets returns raw bytes when decode=False; we decode with soundfile
    audio, sample_rate = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
    reference = normalize(sample["text"])

    wav_bytes = to_wav_bytes(audio, sample_rate)
    r = httpx.post(f"{server}/transcribe", content=wav_bytes, timeout=120)

    assert r.status_code == 200, f"Unexpected status {r.status_code}: {r.text}"
    body = r.json()
    assert "text" in body
    assert "language" in body

    result = normalize(body["text"])
    error_rate = wer(reference, result)

    assert error_rate < WER_THRESHOLD, (
        f"Sample {idx} WER {error_rate:.1%} exceeds {WER_THRESHOLD:.0%}\n"
        f"  reference : {reference!r}\n"
        f"  got       : {result!r}"
    )
