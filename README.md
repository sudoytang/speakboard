# Speakboard

Hold a key, speak, release — text lands in your clipboard.

Speakboard runs entirely on-device using [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) on Apple Silicon. No internet connection, no API keys, no latency from the cloud.

## Requirements

- macOS with Apple Silicon (M1 or later)
- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

## Install

```bash
git clone https://github.com/yourname/speakboard
cd speakboard
uv sync
```

The first run will automatically download the Whisper model (~1.6 GB).

## Usage

```bash
uv run main.py
```

- **Hold right Option** → recording starts
- **Release** → transcription runs, result is copied to clipboard
- **Cmd+V** to paste anywhere
- **Ctrl+C** to quit

## Configuration

Edit `speakboard/transcribe.py` to change the model or transcription behavior:

```python
MLXWhisperTranscriber(
    model="mlx-community/whisper-large-v3-mlx",  # or whisper-medium-mlx, whisper-small-mlx
    initial_prompt="Okay, got it. 好的，我明白了。",
)
```

## Architecture

```
speakboard/
  __main__.py   — entry point
  core.py       — hotkey + audio recording + clipboard (no OS-specific code)
  transcribe.py — Transcriber ABC + MLX Whisper implementation
```

`core.py` has zero OS-specific dependencies. Porting to another platform means adding a new transcription backend in `transcribe.py` — the rest stays unchanged.
