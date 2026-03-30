# Speakboard

Hold a key, speak, release — text lands in your clipboard.

Speakboard runs entirely on-device using [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on Apple Silicon (or `openai-whisper` on other platforms). No internet connection, no API keys, no cloud latency.

It can be used in two ways:

- **Standalone CLI** — hotkey-driven, copies result to clipboard
- **HTTP server** — accepts audio bytes, returns transcription JSON; acts as a local sidecar for a GUI app or a remote cloud service

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- Apple Silicon recommended (MLX backend); Linux/Windows supported via CPU backend

## Install

```bash
git clone https://github.com/sudoytang/speakboard
cd speakboard
```

Install only what you need:

```bash
uv sync --extra mlx-cli       # macOS standalone CLI
uv sync --extra mlx-server    # macOS HTTP server (GUI sidecar)
uv sync --extra cpu-server    # Linux/Windows HTTP server
uv sync --extra cpu-cli       # any platform standalone CLI
```

The first run will automatically download the Whisper model (~1.6 GB).

## Usage

### Standalone CLI (macOS only)

```bash
uv run python -m speakboard run
```

- **Hold right Option** → recording starts
- **Release** → transcription runs, result is copied to clipboard
- **Cmd+V** to paste anywhere
- **Ctrl+C** to quit

### HTTP Server

```bash
uv run python -m speakboard serve
uv run python -m speakboard serve --host 0.0.0.0 --port 9000
```

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ok"}` when model is loaded and ready |
| `POST` | `/transcribe` | Accepts raw audio bytes, returns `{"text": "...", "language": "..."}` |

#### Audio format

Preferred input: **WAV, 16 kHz, mono**. Any format readable by `soundfile` (FLAC, OGG, etc.) is also accepted; the server resamples to 16 kHz automatically.

#### Example

```bash
curl -X POST http://127.0.0.1:8000/transcribe \
     --data-binary @recording.wav
# {"text": "Hello world", "language": "en"}
```

## Backend selection

The transcription backend is chosen automatically by platform, but can be overridden:

```bash
python -m speakboard run --backend cpu    # force CPU on macOS (for testing)
python -m speakboard serve --backend mlx  # force MLX explicitly
```

| Flag | Backend | Requires |
|------|---------|----------|
| `--backend mlx` | Apple Silicon (mlx-whisper) | macOS, `mlx` extra |
| `--backend cpu` | CPU/PyTorch (openai-whisper) | any platform, `cpu` extra |

## Frontend integration (Electron / Tauri)

A GUI app can use speakboard as a sidecar:

1. Spawn `python -m speakboard serve` on app start
2. Poll `GET /health` until ready (model loading takes 5–30 s)
3. Record audio with native APIs (Web Audio, Tauri mic plugin, etc.)
4. `POST /transcribe` with raw audio bytes on hotkey release
5. Receive `{"text": "...", "language": "..."}` and handle (paste, display, etc.)
6. Kill the process on app quit

The same endpoint works unchanged if the service is later moved to a remote host — just change the base URL.

## Architecture

```
speakboard/
  __main__.py   — entry point; subcommand dispatch (run / serve); platform → backend selection
  transcribe.py — Transcriber ABC + MLXWhisperTranscriber + CPUWhisperTranscriber
  cli.py        — standalone HCI layer: hotkey + audio recording + clipboard
  server.py     — HTTP server: POST /transcribe + GET /health
```

Dependency map:

```
transcribe.py          (no HCI dependencies)
    ↑
    ├── cli.py         (pynput, pyperclip, sounddevice)
    └── server.py      (fastapi, uvicorn, soundfile, scipy)
         ↑
         └── __main__.py
```

`cli.py` and `server.py` are fully independent of each other. A GUI frontend uses only `server.py` and ignores `cli.py` entirely.

## Running tests

```bash
uv run --extra mlx-server --extra test pytest tests/ -v
```

Tests start a real server subprocess, download 5 samples from the LibriSpeech demo dataset, and assert WER < 15% on each.
