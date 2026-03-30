# Changelog — 2026-03-29: Service/HCI Separation Refactor

## Summary

Restructured the project to fully decouple the transcription service from
human-computer interaction, enabling the backend to be used as a local sidecar
or remote HTTP service for GUI frontends (Electron, Tauri, etc.).

---

## Changes

### `transcribe.py` — Added `CPUWhisperTranscriber`

Added a second concrete transcriber implementation alongside
`MLXWhisperTranscriber`. `CPUWhisperTranscriber` uses the `openai-whisper`
library (PyTorch-based) and runs on any platform (Linux, Windows, macOS).
Both implementations satisfy the existing `Transcriber` ABC with identical
interfaces, requiring no changes to callers.

Imports of `mlx_whisper` and `whisper` were moved inside methods so that
neither backend is imported at module load time — only the one actually used
gets imported at runtime.

### `cli.py` — Replaced `core.py`

`core.py` was renamed to `cli.py` to make its responsibility explicit: it is
the standalone HCI layer only. The `Whisperer` class logic (hotkey listener,
audio stream, silence detection, clipboard copy, console feedback) is unchanged.
`cli.py` depends on `pynput`, `pyperclip`, and `sounddevice`. It has no
knowledge of HTTP.

### `server.py` — New HTTP server

A new module exposing an HTTP service via FastAPI:

- `GET /health` — returns `{"status": "ok"}`, used by sidecar launchers to
  poll readiness
- `POST /transcribe` — accepts raw audio bytes (WAV or any format readable by
  `soundfile`), decodes to float32 mono at 16 kHz (resampling via `scipy` if
  needed), runs transcription in a thread pool executor, and returns
  `{"text": "...", "language": "..."}` as JSON

`server.py` has no knowledge of hotkeys, audio devices, or clipboard.
Transcription is injected via the `Transcriber` interface.

### `__main__.py` — Subcommand dispatch, platform-based backend selection, and `--backend` flag

Replaced the single `main()` with an `argparse`-based dispatcher:

```
python -m speakboard run              # standalone CLI (macOS only)
python -m speakboard serve            # HTTP server (all platforms)
python -m speakboard serve --port 9000 --host 0.0.0.0
```

Backend selection is centralised here and nowhere else:

- `darwin` → `MLXWhisperTranscriber` (Apple Silicon, mlx-whisper)
- other → `CPUWhisperTranscriber` (Linux/Windows, openai-whisper)

A `--backend {mlx,cpu}` flag was added to both subcommands to override the
platform default, useful for testing the CPU backend on macOS:

```
python -m speakboard run --backend cpu
python -m speakboard serve --backend cpu
```

`run` raises `NotImplementedError` on non-macOS since the right Option key
hotkey is macOS-specific. `serve` works on any platform.

### `pyproject.toml` — Optional dependency groups

Dependencies are now split into optional groups so each deployment installs
only what it needs:

| Group | Contents |
|-------|----------|
| `mlx` | mlx-whisper |
| `cpu` | openai-whisper |
| `server` | fastapi, uvicorn, soundfile, scipy |
| `cli` | pynput, pyperclip, sounddevice |
| `mlx-cli` | mlx + cli — macOS standalone |
| `mlx-server` | mlx + server — macOS HTTP sidecar |
| `cpu-cli` | cpu + cli — any platform standalone |
| `cpu-server` | cpu + server — Linux/Windows HTTP server |
| `all` | everything |

Install examples:
```bash
uv sync --extra mlx-cli       # macOS standalone
uv sync --extra mlx-server    # macOS HTTP sidecar
uv sync --extra cpu-server    # Linux server
uv sync --extra all           # everything
```

A `test` optional group was also added for running the test suite:

```bash
uv run --extra mlx-server --extra test pytest tests/ -v
```

Version bumped from `0.1.0` → `0.2.0`.

---

### `server.py` — Inference lock

Added an `asyncio.Lock` around the transcription call to prevent concurrent
requests from racing on the shared model instance. Whisper models are not
thread-safe due to internal KV cache and GPU command queue state. With the
lock, a second request waits for the first to complete before starting
inference. For single-user sidecar use this has no practical latency impact.

### `tests/` — Pytest test suite

Added an integration test suite covering the HTTP server:

- `test_health` — verifies `/health` returns 200
- `test_empty_body_returns_400` — verifies empty request is rejected
- `test_invalid_audio_returns_422` — verifies non-audio bytes are rejected
- `test_transcription_wer[0–4]` — streams 5 samples from the
  `hf-internal-testing/librispeech_asr_demo` dataset, POSTs each to
  `/transcribe`, and asserts WER < 15%

The `datasets` library audio column is loaded with `decode=False` to avoid the
`torchcodec` dependency introduced in `datasets` 3.x; audio is decoded manually
with `soundfile` instead.

---

## Dependency Map

```
transcribe.py          (MLXWhisperTranscriber + CPUWhisperTranscriber; no HCI deps)
    ↑
    ├── cli.py         (imports transcribe; also imports pynput, pyperclip, sounddevice)
    └── server.py      (imports transcribe; also imports fastapi, uvicorn, soundfile, scipy)
         ↑
         └── __main__.py  (selects transcriber by platform; dispatches run/serve subcommand)
```

---

## Frontend Integration (Electron / Tauri)

The frontend ignores `cli.py` entirely. It:

1. Spawns `python -m speakboard serve` as a sidecar process on app start
2. Polls `GET /health` until the service responds (model is loaded and ready)
3. Records audio using its own native APIs (Web Audio API, Tauri mic plugin, etc.)
4. Sends `POST /transcribe` with raw audio bytes on hotkey release
5. Receives `{"text": "...", "language": "..."}` and handles it (paste, display, etc.)
6. Kills the sidecar process on app quit

The same endpoint works unchanged if the service is moved to a remote host —
the frontend only changes the base URL.

---

## Audio Format Contract

The preferred format for `POST /transcribe` is **WAV, 16 kHz, mono**. The
server also accepts any format readable by `soundfile` (FLAC, OGG, etc.) and
resamples to 16 kHz automatically. Formats requiring an external codec (WebM,
MP3, Opus) are not supported without additional dependencies.
