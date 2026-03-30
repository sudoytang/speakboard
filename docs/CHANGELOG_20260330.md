# Changelog — 2026-03-30: Hallucination Detection, Silence Splitting, and Rich Transcription Response

## Summary

This change improves transcription reliability for long or repetitive audio, and
extends the `POST /transcribe` response with per-segment detail so frontends can
surface richer UI (segment boundaries, per-segment language, hallucination
warnings). The top-level `text` and `language` fields are unchanged — existing
frontend code that reads only those fields continues to work without modification.

---

## Breaking Changes

None. The response schema is additive only.

---

## `POST /transcribe` — Response Schema Change

### Before

```json
{
  "text": "转录结果",
  "language": "zh"
}
```

### After

```json
{
  "text": "第一段内容第二段内容",
  "language": "zh",
  "duration_seconds": 27.83,
  "split": true,
  "segments": [
    {
      "index": 0,
      "duration_seconds": 12.41,
      "text": "第一段内容",
      "language": "zh",
      "hallucinated": false
    },
    {
      "index": 1,
      "duration_seconds": 15.42,
      "text": "第二段内容",
      "language": "en",
      "hallucinated": false
    }
  ]
}
```

### Field reference

| Field | Type | Always present | Description |
|---|---|---|---|
| `text` | `string` | yes | Full transcribed text, concatenation of all non-hallucinated segment texts. Empty string if all segments were hallucinated or audio was silent. |
| `language` | `string` | yes | Language code detected across the whole recording (e.g. `"zh"`, `"en"`). Taken from the last segment with a known language. |
| `duration_seconds` | `float` | yes | Total audio duration in seconds. |
| `split` | `boolean` | yes | `true` if the audio was longer than 15 s and was split into multiple segments before inference. `false` if processed as a single chunk — in this case `segments` will have exactly one entry. |
| `segments` | `array` | yes | Always present, even when `split` is `false`. Contains exactly one entry for short audio. |
| `segments[].index` | `int` | yes | Zero-based segment index in chronological order. |
| `segments[].duration_seconds` | `float` | yes | Duration of this segment in seconds. |
| `segments[].text` | `string` | yes | Transcribed text for this segment. Empty string if `hallucinated` is `true`. |
| `segments[].language` | `string` | yes | Language detected for this specific segment. Useful when the speaker switches languages mid-recording (e.g. Chinese followed by English). |
| `segments[].hallucinated` | `boolean` | yes | `true` if this segment was detected as a hallucination loop (e.g. a word repeated hundreds of times). Its `text` will be `""`. The segment is still included in the array so the frontend can show a warning at the correct position. |

---

## Hallucination Detection

Whisper is known to produce repetitive hallucination loops (e.g. `"相信相信相信..."`)
on long, silent, or repetitive audio. The backend now detects this using a gzip
compression ratio heuristic: if the compressed size of the output text is more
than 5× smaller than the raw UTF-8 size, the output is flagged as a hallucination
loop.

**Frontend implications:**

- A segment with `hallucinated: true` should be displayed differently (warning
  icon, greyed-out placeholder, tooltip, etc.) rather than inserted into the
  transcript as empty text.
- The top-level `text` field never contains hallucinated content — safe to use
  directly for clipboard paste or final output.
- Example: a 3-segment recording where segment 1 was hallucinated:

```json
{
  "text": "段落二内容段落三内容",
  "split": true,
  "segments": [
    { "index": 0, "text": "", "hallucinated": true, ... },
    { "index": 1, "text": "段落二内容", "hallucinated": false, ... },
    { "index": 2, "text": "段落三内容", "hallucinated": false, ... }
  ]
}
```

---

## Silence-Based Audio Splitting

Audio longer than 15 s is automatically split into segments before inference.
The splitting algorithm:

1. Computes RMS energy in 20 ms windows.
2. Identifies silence gaps ≥ 400 ms where RMS < 0.01.
3. Greedily builds the longest possible segments (up to 15 s each), splitting
   only when necessary. When a split is needed, it picks the **longest** silence
   gap within the current 15 s window — not simply the first one — to minimise
   total segment count.
4. Falls back to a hard cut at the 15 s boundary if no silence gap is found.

**Frontend implications:**

- There is **no client-side audio length limit**. Send the full recording
  regardless of duration; the backend handles chunking.
- `split: false` — audio was ≤ 15 s, processed in one shot.
- `split: true` — audio was split; use the `segments` array to show boundaries
  or a waveform breakdown if desired.
- Segments are ordered chronologically by `index`. Concatenating `segment.text`
  for all non-hallucinated segments in index order reproduces the top-level
  `text` field exactly.

---

## Per-Segment Language Detection

Each segment now reports the language Whisper detected for that specific chunk.
This matters for multilingual recordings (e.g. a Chinese preamble followed by
an English quote).

**Frontend implications:**

- Use `segments[].language` to label individual segments in a multi-language UI.
- Use the top-level `language` for single-language display or when `split` is
  `false`.
- Language codes follow Whisper's output (ISO 639-1 two-letter codes, e.g.
  `"zh"`, `"en"`, `"ja"`). May be `"?"` if detection failed.

---

## No Changes to Audio Format Contract

`POST /transcribe` still accepts any audio format readable by `soundfile` (WAV,
FLAC, OGG). Preferred format remains **WAV, 16 kHz, mono**. No client-side
changes required for audio encoding.

---

## Dependency Map (unchanged)

```
transcribe.py          (MLXWhisperTranscriber, CPUWhisperTranscriber,
                        TranscribeResult, SegmentResult, split_on_silence)
    ↑
    ├── cli.py         (standalone macOS HCI — not relevant to frontend)
    └── server.py      (FastAPI HTTP server — frontend integration point)
         ↑
         └── __main__.py
```
