import io
import time

import numpy as np
import soundfile as sf
import scipy.signal

from .transcribe import SAMPLE_RATE, Transcriber

_transcriber: Transcriber | None = None


def _decode_audio(data: bytes) -> np.ndarray:
    """Decode raw audio bytes to a float32 mono array at SAMPLE_RATE."""
    buf = io.BytesIO(data)
    audio, src_rate = sf.read(buf, dtype="float32", always_2d=True)
    # Mix down to mono
    audio = audio.mean(axis=1)
    # Resample to 16 kHz if needed
    if src_rate != SAMPLE_RATE:
        num_samples = int(len(audio) * SAMPLE_RATE / src_rate)
        audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
    return audio


def create_app(transcriber: Transcriber):
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse

    global _transcriber
    _transcriber = transcriber

    app = FastAPI(title="speakboard", description="Audio transcription service")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(request: Request):
        import asyncio
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="Empty audio body")

        try:
            audio = _decode_audio(data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")

        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        text, language = await loop.run_in_executor(None, _transcriber.transcribe, audio)
        elapsed = time.perf_counter() - t0
        print(f"[speakboard] [{language}] transcription took {elapsed:.2f}s: {text}")

        return JSONResponse({"text": text, "language": language})

    return app


def start_server(transcriber: Transcriber, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    transcriber.load()
    app = create_app(transcriber)
    print(f"[speakboard] Server listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
