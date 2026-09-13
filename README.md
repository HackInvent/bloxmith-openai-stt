# OpenAI STT Block

<!-- block-metadata:start -->
[![Block version: 0.1.0](https://img.shields.io/badge/block-0.1.0-blue)](model.json)
[![BloxSmith compatibility: 1.0.9](https://img.shields.io/badge/BloxSmith-1.0.9-brightgreen)](compatibility.json)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Verified BloxSmith versions: **1.0.9** (bundled-block tests; see [test evidence](compatibility.json)).
<!-- block-metadata:end -->


## Role

`openai_stt` transcribes an audio file through the OpenAI audio transcription API and emits the transcript text plus the raw JSON response.

## Files

- `block.py`: audio path resolution, OpenAI constraint validation, upload chunking, multipart call, response normalization, and runtime outputs.
- `model.json`: audio input, transcript/raw JSON outputs, and API/model configuration.
- `block_modal.html`: autonomous tabbed modal for full transcription settings.
- `inspector_panel.html`, `parameters_fields.html`: inspector and shared parameter form for model selection and credentials.
- `assets/css/block_modal.css`: modal navigation and parameter layout.
- `assets/js/block_modal.js`: modal Apply workflow and tab interactions.
- `assets/js/inspector_panel.js`: inspector Apply workflow.
- `node_card.html`: block-owned canvas card body.

## Ports

- Inputs:
  - `audio` (`id: 1`): optional audio file path from `file/path`, `audio/*`, or plain message text.
- Outputs:
  - `transcript` (`id: 1`): emits the transcript as `text/plain`.
  - `raw_json` (`id: 2`): emits the API response as `application/json`.

## Configuration

- `model`: one of the file transcription models supported by this block.
- `api_key`: OpenAI API key stored on the node config as requested by the product owner; the inspector does not echo it back in HTML.
- `api_base_url`: OpenAI-compatible base URL, defaults to `https://api.openai.com`.
- `audio_path`: fallback audio file path when the input port is not connected.
- `language`: optional BCP-47/ISO language hint.
- `prompt`: optional transcription prompt.
- `response_format`: response format sent to the API.
- `timeout_sec`: HTTP timeout.
- `chunking_enabled`: when true, files larger than `chunk_size_mb` are split before upload.
- `chunk_size_mb`: local chunk target between 1 and 25 MB. Defaults to 24 MB.
- `chunk_duration_sec`: local transcription window duration. Defaults to 30 seconds.
- `chunk_overlap_sec`: pre-roll overlap added before every chunk after the first. Defaults to 2 seconds.
- `chunking_strategy`: diarization API chunking strategy. `auto` is sent for `gpt-4o-transcribe-diarize`; `none` omits it.
- `experimental_audio_filter_enabled`: when true, each generated chunk is processed through the configured FFmpeg audio filter before upload.
- `experimental_audio_filter`: FFmpeg `-af` filter chain, defaulting to `highpass=f=90,lowpass=f=7600,afftdn=nr=8,loudnorm=I=-16:TP=-2:LRA=8`.
- `sliding_context_enabled`: when true, the block sends a short text context from the previous chunk with the current chunk prompt to improve continuity.

## Runtime Behavior

`execute_runtime()` resolves an input/configured audio path, posts it as multipart form data to `/v1/audio/transcriptions`, normalizes text/JSON responses, and emits outputs for every configured output port.

The block is a generic runtime executable, so the same `execute_runtime()` path is used in centralized and `zeromq_active` runs.

The block enforces the OpenAI upload constraints locally:

- supported file extensions: `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`;
- each uploaded file or generated chunk must be at most 25 MB;
- `gpt-4o-transcribe` and `gpt-4o-mini-transcribe` use `json` or `text`;
- `gpt-4o-transcribe-diarize` uses `json`, `text`, or `diarized_json`, receives `chunking_strategy=auto` by default, and does not receive `prompt`;
- `whisper-1` may use `json`, `text`, `srt`, `verbose_json`, or `vtt`.

When local chunking is enabled, the block follows the Prudia strategy: 30-second nominal windows by default, 2 seconds of pre-roll overlap on every chunk after the first, and offset metadata on every chunk. Segment timestamps returned by OpenAI are rebased to the source audio timeline; overlap segments whose local start is inside the ignored pre-roll are skipped from the main transcript and grouped into 30-second timestamp buckets in `raw_json.timestamped_text`. The full per-chunk text remains available in `raw_json.raw_text`.

When `sliding_context_enabled` is enabled, the block derives a short context from the previous chunk transcript and appends it to the prompt of the next chunk. This context is not a runtime graph feedback link; it is internal request context managed by the block.

WAV files are chunked with Python stdlib unless the FFmpeg audio filter is enabled. Other supported formats use `ffmpeg`/`ffprobe` when local chunking is needed; without an audio filter the original audio stream is copied, and with a filter FFmpeg re-encodes while preserving source sample rate/channel hints when available.

## UI Behavior

The inspector and modal edit model, credentials, audio fallback, language, prompt, response format,
chunking, FFmpeg filter, timeout, and API base URL through `inspector_update_openai_stt` or
`modal_update_openai_stt`. Both surfaces render the same parameter form from
`parameters_fields.html`, with separate DOM IDs per surface to avoid collisions while preserving the
same controls and layout. Edits stay pending until the user clicks **Apply**. The audio fallback
uses the shared `CWPathBrowser` control in file mode. Sensitive values such as API keys are not
echoed back in HTML, and an empty key field does not overwrite an existing key unless the clear
checkbox is applied.

The modal declares `data-block-runtime-refresh="autonomous"`, so runtime polling keeps the open tab, focus, scroll, and draft values stable until **Apply** persists the changes.

## Editor Display

The canvas card is rendered by this block through `node_card.html`. It exposes the selected transcription model, response format, and chunking state while the shared editor shell keeps ports, dragging, status, and graph links generic.

## Modal

`block_modal.html` is owned by this block. It opens on a **Parameters** tab that mirrors the inspector
parameter controls, with secondary tabs for identity/technical attributes and ports/runtime state.
Sensitive values such as API keys are not exposed in HTML and empty secret fields do not overwrite
existing values.

## Maintenance Notes

Keep OpenAI-specific credentials, model choices, request formatting, and response parsing inside this block. The orchestrator must only see a generic block execution result.

## Compatibility policy

[compatibility.json](compatibility.json) records HackInvent's verified BloxSmith versions and test evidence. Only the versions listed above have been verified, using the block-owned suites in a **bundled-block test installation**. This is not a certification of managed-package installation, every browser/OS, or live provider availability. Other framework versions are unverified, not necessarily incompatible.

The block-version badge follows `model.json`, not a published Git tag. `unversioned` means that no block release version is declared; no number is inferred from the framework version. The framework still uses `model.json` for its runtime/install contract; the tester-owned JSON does not replace it. Official integration tests run in the private `bloxmith-blocs` workspace. Test helpers and the proprietary framework are not bundled in this public block repository.
