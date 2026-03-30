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
    import asyncio
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse

    global _transcriber
    _transcriber = transcriber
    _inference_lock = asyncio.Lock()

    # "available" — idle, ready to accept requests
    # "busy"      — inference in progress, new requests will queue behind the lock
    # "broken"    — last inference timed out; subprocess is restarting
    #               recovers to "available" once restart completes (before 503 is sent)
    _state = {"status": "available", "queue_depth": 0}

    app = FastAPI(title="speakboard", description="Audio transcription service")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/status")
    async def status():
        return {
            "status": _state["status"],
            "queue_depth": _state["queue_depth"],
        }

    @app.post("/transcribe")
    async def transcribe(request: Request):
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="Empty audio body")

        assert _transcriber is not None

        try:
            audio = _decode_audio(data)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")

        _state["queue_depth"] += 1
        t0 = time.perf_counter()
        loop = asyncio.get_event_loop()
        try:
            async with _inference_lock:
                _state["queue_depth"] -= 1
                _state["status"] = "busy"
                try:
                    result = await loop.run_in_executor(None, _transcriber.transcribe, audio)
                except RuntimeError as e:
                    _state["status"] = "broken"
                    _state["status"] = "available"  # restart already completed inside transcribe()
                    raise HTTPException(status_code=503, detail=str(e))
                _state["status"] = "available"
        except HTTPException:
            raise
        except Exception as e:
            _state["status"] = "available"
            raise HTTPException(status_code=500, detail=str(e))

        elapsed = time.perf_counter() - t0
        print(f"[speakboard] [{result.language}] transcription took {elapsed:.2f}s: {result.text}")

        return JSONResponse({
            "text": result.text,
            "language": result.language,
            "duration_seconds": result.duration_seconds,
            "split": result.split,
            "segments": [
                {
                    "index": s.index,
                    "duration_seconds": s.duration_seconds,
                    "text": s.text,
                    "language": s.language,
                    "hallucinated": s.hallucinated,
                }
                for s in result.segments
            ],
        })

    return app


def start_server(transcriber: Transcriber, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    transcriber.load()
    app = create_app(transcriber)
    print(f"[speakboard] Server listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
