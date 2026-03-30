import subprocess
import sys
import time

import httpx
import pytest

SERVER_URL = "http://127.0.0.1:8000"
# Model loading can take up to a few minutes on first run
STARTUP_TIMEOUT_S = 300


@pytest.fixture(scope="session")
def server():
    """Start the speakboard HTTP server as a subprocess for the test session."""
    backend = "mlx" if sys.platform == "darwin" else "cpu"
    proc = subprocess.Popen(
        [sys.executable, "-m", "speakboard", "serve", "--backend", backend],
    )

    # Poll /health until the server is up (model loading happens before uvicorn starts)
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                break
        except httpx.TransportError:
            pass
        time.sleep(2)
    else:
        proc.kill()
        raise RuntimeError(
            f"Server did not become ready within {STARTUP_TIMEOUT_S}s. "
            "Check that the model can be loaded and that port 8000 is free."
        )

    yield SERVER_URL

    proc.terminate()
    proc.wait()


@pytest.fixture(scope="session")
def librispeech_samples():
    """Stream the first 5 samples from the LibriSpeech ASR demo dataset."""
    import itertools
    from datasets import load_dataset, Audio

    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_demo",
        "clean",
        split="validation",
        streaming=True,
        trust_remote_code=False,
    ).cast_column("audio", Audio(decode=False))  # return raw bytes, decode ourselves
    return list(itertools.islice(ds, 5))
