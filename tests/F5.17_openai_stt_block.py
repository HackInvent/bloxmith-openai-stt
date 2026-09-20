#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Role: Verifies openai STT block behavior for the OpenAI STT block.
# File Name: F5.17_openai_stt_block.py
# Author: Alexandre EL
# Email: alex@hackinvent.com
# Created Date: 2024-01-25
# -----------------------------------------------------------------------------

"""F5.17 - Bloc OpenAI STT."""

# Test cases:
# - FB1/FB4/FB5/FB7 - Run text -> openai_stt -> display in centralized and active runtime against a local OpenAI-compatible STT endpoint, verify transcript/raw JSON outputs and no api_key leak in logs.
# - FB2 - Refuse unsupported audio formats and oversized single uploads when chunking is disabled.
# - FB3 - Split a valid WAV file into multiple upload-safe chunks.
# - FB6 - Render the block-owned inspector and modal parameter tab from the shared form, verify the api_key is masked from HTML, and verify UI updates preserve or update the stored key intentionally.

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from threading import Thread
from typing import Any
import json
import shutil
import sys
import wave


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import blocs.openai_stt.block as openai_stt_module
from blocs.openai_stt.block import OpenAiSttBlock
from bloxsmith_app.block_runtime import BlockRuntimeContext
from bloxsmith_app.block_ui import handle_block_ui_action, render_block_inspector_panel, render_block_modal
from bloxsmith_app.graph_introspection import describe_block

from ui_smoke_common import (
    create_run_api,
    data_edge,
    display_node,
    expect,
    graph_payload,
    isolated_server,
    text_node,
    wait_for_run_terminal,
)
from urllib.parse import quote
from block_test_packages import install_test_package, release_key, surface_payload


SECRET = "sk-test-openai-stt-secret"
TRANSCRIPT = "hello from the fake stt"


def runtime_node_id_for_kind(run: dict[str, Any], kind: str) -> str:
    """Return the runtime node id for a block kind after graph id normalization."""

    for node_id, result in (run.get("results") or {}).items():
        bindings = result.get("bindings") if isinstance(result, dict) else {}
        if isinstance(bindings, dict) and bindings.get("kind") == kind:
            return str(node_id)
    return ""


class FakeOpenAiSttHttpServer(ThreadingHTTPServer):
    requests_log: list[dict[str, Any]]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), FakeOpenAiSttHandler)
        self.requests_log = []

    @property
    def base_url(self) -> str:
        host, port = self.server_address
        return f"http://{host}:{port}"


class FakeOpenAiSttHandler(BaseHTTPRequestHandler):
    server: FakeOpenAiSttHttpServer

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(length)
        self.server.requests_log.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization") or "",
                "content_type": self.headers.get("Content-Type") or "",
                "body": body,
            }
        )
        payload = self._payload_for_body(body)
        response = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _payload_for_body(self, body: bytes) -> dict[str, Any]:
        marker = b"-chunk-"
        marker_index = body.find(marker)
        if marker_index < 0:
            return {
                "text": TRANSCRIPT,
                "segments": [{"id": 0, "text": TRANSCRIPT, "start": 0.0, "end": 1.0}],
            }
        chunk_token = body[marker_index + len(marker) : marker_index + len(marker) + 4]
        try:
            chunk_index = int(chunk_token.decode("ascii"))
        except ValueError:
            chunk_index = 1
        if chunk_index <= 1:
            return {
                "text": "chunk 1",
                "segments": [{"id": 0, "text": "chunk 1", "start": 0.0, "end": 1.0}],
            }
        return {
            "text": f"overlap {chunk_index} chunk {chunk_index}",
            "segments": [
                {"id": 0, "text": f"overlap {chunk_index}", "start": 0.2, "end": 1.0},
                {"id": 1, "text": f"chunk {chunk_index}", "start": 2.2, "end": 3.0},
            ],
        }

    def log_message(self, format: str, *args: Any) -> None:
        return


class FakeSttServer:
    def __enter__(self) -> FakeOpenAiSttHttpServer:
        self.server = FakeOpenAiSttHttpServer()
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self.server

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def openai_stt_node(api_base_url: str, audio_path: str = "") -> dict[str, Any]:
    return {
        "id": "openai-stt-1",
        "kind": "openai_stt",
        "title": "OpenAI STT",
        "position": {"x": 360, "y": 120},
        "inputs": [
            {
                "id": 1,
                "name": "audio",
                "title": "Audio",
                "accepts": ["file/path", "audio/*", "message/*"],
                "multiplicity": "many",
            }
        ],
        "outputs": [
            {"id": 1, "name": "transcript", "title": "Transcript", "emits": ["text/plain", "message/*"], "multiplicity": "many"},
            {"id": 2, "name": "raw_json", "title": "Raw JSON", "emits": ["application/json", "message/*"], "multiplicity": "many"},
        ],
        "config": {
            "model": "gpt-4o-mini-transcribe",
            "api_key": SECRET,
            "api_base_url": api_base_url,
            "audio_path": audio_path,
            "language": "fr",
            "prompt": "Conversation courte.",
            "response_format": "json",
            "timeout_sec": 10,
            "chunking_enabled": True,
            "chunk_size_mb": 24,
            "chunk_duration_sec": 30,
            "chunk_overlap_sec": 2,
            "chunking_strategy": "auto",
            "experimental_audio_filter_enabled": False,
            "experimental_audio_filter": "highpass=f=90,lowpass=f=7600,afftdn=nr=8,loudnorm=I=-16:TP=-2:LRA=8",
            "sliding_context_enabled": False,
        },
    }


def write_silent_wav(path: Path, *, seconds: int, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = seconds * sample_rate
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frame_count)


def direct_context(
    *,
    root_dir: Path,
    run_dir: Path,
    config: dict[str, Any],
    input_message: str = "",
    services: dict[str, Any] | None = None,
) -> BlockRuntimeContext:
    return BlockRuntimeContext(
        run_id="direct-run",
        node_id="openai-stt-direct",
        kind="openai_stt",
        title="OpenAI STT",
        config=config,
        inputs={},
        input_content_types={},
        input_message=input_message,
        input_ports=(),
        output_ports=(
            SimpleNamespace(id=1, name="transcript"),
            SimpleNamespace(id=2, name="raw_json"),
        ),
        root_dir=root_dir,
        run_dir=run_dir,
        services=dict(services or {}),
    )


def run_openai_stt_case(runtime_mode: str, fake_server: FakeOpenAiSttHttpServer) -> None:
    with isolated_server() as server:
        # Surfaces are release assets: a bundled kind serves none of them.
        model = install_test_package(server, "openai_stt")
        key = quote(release_key(model), safe="")
        served = lambda payload, suffix: next(
            asset["path"] for asset in payload["assets"] if asset["path"].endswith(suffix))
        audio_path = server.root_dir / "tmp" / f"audio-{runtime_mode}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"RIFF....WAVEfmt fake audio")
        document = graph_payload(
            f"F5 OpenAI STT {runtime_mode}",
            [
                text_node("text-1", "Audio path", str(audio_path), 80, 120),
                openai_stt_node(fake_server.base_url),
                display_node("display-1", "Display", 700, 120),
            ],
            [
                data_edge("edge-text-stt", "text-1", 1, "openai-stt-1", 1),
                data_edge("edge-stt-display", "openai-stt-1", 1, "display-1", 1),
            ],
        )
        created = create_run_api(server, document, runtime_mode=runtime_mode)
        run = wait_for_run_terminal(server, str(created.get("run_id") or ""), timeout_sec=25)

        logs = "\n".join(run.get("logs", []))
        stt_node_id = runtime_node_id_for_kind(run, "openai_stt")
        node_logs = "\n".join(run.get("node_logs", {}).get(stt_node_id, []))
        expect(run.get("status") == "success", f"Le run OpenAI STT {runtime_mode} doit reussir.")
        expect(
            run.get("output_values", {}).get(f"{stt_node_id}:1", {}).get("value") == TRANSCRIPT,
            "The STT transcript must be published on the transcript port.",
        )
        raw_json = run.get("output_values", {}).get(f"{stt_node_id}:2", {}).get("value") or ""
        expect(TRANSCRIPT in raw_json and "segments" in raw_json, "The raw JSON response must be published on raw_json.")
        expect(SECRET not in logs and SECRET not in node_logs, "The API key must not appear in the logs.")
        expect("fallback centralized" not in logs, "The run must not fall back to centralized.")
        if runtime_mode == "centralized":
            run_log_text = f"{logs}\n{node_logs}"
            expect("file validation OK" in run_log_text, "The centralized run must receive the live STT logs.")
            expect("full-file OpenAI upload" in run_log_text, "The centralized run must trace the OpenAI STT call.")
        if runtime_mode == "zeromq_active":
            expect(
                run.get("results", {}).get(stt_node_id, {}).get("transport") == "zeromq_active",
                "openai_stt must run through zeromq_active.",
            )


def test_http_requests(fake_server: FakeOpenAiSttHttpServer) -> None:
    expect(len(fake_server.requests_log) >= 2, "The fake STT endpoint must receive one request per run.")
    for request in fake_server.requests_log:
        body = request["body"]
        expect(request["path"] == "/v1/audio/transcriptions", "The block must call /v1/audio/transcriptions.")
        expect(request["authorization"] == f"Bearer {SECRET}", "The block must send the API key as a bearer token.")
        expect(b'gpt-4o-mini-transcribe' in body, "The STT model must be present in the multipart body.")
        expect(b'Conversation courte.' in body, "The STT prompt must be present in the multipart body.")
        expect(b'audio-' in body, "The audio file must be present in the multipart body.")


def test_inspector_contract(fake_server: FakeOpenAiSttHttpServer) -> None:
    node = openai_stt_node(fake_server.base_url)
    rendered = render_block_inspector_panel("openai_stt", {"node": node})
    html = str(rendered.get("html") or "")
    expect("data-openai-stt-inspector-root" in html, "The OpenAI STT inspector panel must come from the block.")
    expect(SECRET not in html, "The API key must not be rendered in clear text in the HTML.")
    expect("data-path-browser" in html, "The OpenAI STT panel must use the shared path browser.")
    expect("data-openai-stt-audio-path" in html, "The OpenAI STT panel must expose the audio path.")
    expect("gpt-4o-transcribe" in html and "gpt-4o-mini-transcribe" in html, "The select must offer the STT models.")
    for marker in (
        "data-openai-stt-model",
        "data-openai-stt-api-key",
        "data-openai-stt-language",
        "data-openai-stt-prompt",
        "data-openai-stt-response-format",
        "data-openai-stt-chunk-size-mb",
        "data-openai-stt-timeout-sec",
        "data-openai-stt-api-base-url",
    ):
        expect(marker in html, f"The inspector panel must expose the control {marker}.")
    expect(
        "data-openai-stt-experimental-audio-filter-enabled" in html,
        "The inspector panel must expose the ffmpeg audio filter toggle.",
    )
    expect(
        "data-openai-stt-experimental-audio-filter" in html,
        "The inspector panel must expose the ffmpeg audio filter as editable.",
    )
    expect(
        "data-openai-stt-sliding-context-enabled" in html,
        "The inspector panel must expose the sliding context option.",
    )
    expect("data-block-apply" in html, "The OpenAI STT panel must expose the Apply button.")
    expect(
        rendered.get("context", {}).get("inspector_title") == "OpenAI STT",
        "The OpenAI STT inspector title must come from the block.",
    )

    keep_result = handle_block_ui_action(
        "openai_stt",
        {
            "node": node,
            "action": "inspector_update_openai_stt",
            "values": {
                "model": "gpt-4o-transcribe",
                "api_key": "",
                "audio_path": "./next.wav",
                "language": "fr",
                "prompt": "next",
                "response_format": "json",
                "timeout_sec": "30",
                "api_base_url": fake_server.base_url,
                "chunking_enabled": True,
                "chunk_size_mb": "1",
                "chunk_duration_sec": "30",
                "chunk_overlap_sec": "2",
                "chunking_strategy": "auto",
                "experimental_audio_filter_enabled": True,
                "experimental_audio_filter": "highpass=f=120,lowpass=f=5000",
                "sliding_context_enabled": True,
            },
        },
    )
    keep_config = keep_result.get("node_patch", {}).get("config", {})
    expect("api_key" not in keep_config, "An empty entry must keep the existing key.")
    expect(keep_config.get("timeout_sec") == 30, "The inspector timeout must be normalized.")
    expect(keep_config.get("chunk_size_mb") == 1, "The inspector chunk size must be normalized.")
    expect(keep_config.get("chunk_duration_sec") == 30, "The inspector chunk duration must be normalized.")
    expect(keep_config.get("chunk_overlap_sec") == 2, "The inspector overlap must be normalized.")
    expect(keep_config.get("experimental_audio_filter_enabled") is True, "The audio filter must be switchable on.")
    expect(
        keep_config.get("experimental_audio_filter") == "highpass=f=120,lowpass=f=5000",
        "The audio filter must be editable.",
    )
    expect(keep_config.get("sliding_context_enabled") is True, "The sliding context option must be preserved.")

    modal = render_block_modal("openai_stt", {"node": node, "runtime": {}})
    modal_html = str(modal.get("html") or "")
    modal_assets = modal.get("assets") or []
    expect("data-openai-stt-modal-root" in modal_html, "The OpenAI STT modal must expose its dedicated root.")
    expect('data-block-runtime-refresh="autonomous"' in modal_html, "The OpenAI STT modal must own its runtime refresh.")
    expect("data-openai-stt-modal-tab" in modal_html, "The OpenAI STT modal must expose the tabs.")
    expect(
        'data-openai-stt-tab-id="parameters"' in modal_html,
        "The OpenAI STT modal must open a Settings tab.",
    )
    expect("data-path-browser" in modal_html, "The OpenAI STT modal must use the shared path browser.")
    expect("data-openai-stt-audio-path" in modal_html, "The OpenAI STT modal must expose the audio path.")
    expect("data-openai-stt-apply" in modal_html, "The OpenAI STT modal must expose the Apply action.")
    expect(
        'id="openaiSttModalModel"' in modal_html and 'id="openaiSttModalApiKey"' in modal_html,
        "The OpenAI STT modal must reuse the same form with modal-specific ids.",
    )
    for marker in (
        "data-openai-stt-model",
        "data-openai-stt-api-key",
        "data-openai-stt-language",
        "data-openai-stt-prompt",
        "data-openai-stt-response-format",
        "data-openai-stt-chunk-size-mb",
        "data-openai-stt-timeout-sec",
        "data-openai-stt-api-base-url",
    ):
        expect(marker in modal_html, f"The modal must mirror the inspector control {marker}.")
    expect(SECRET not in modal_html, "The API key must not be rendered in clear text in the modal.")

    modal_result = handle_block_ui_action(
        "openai_stt",
        {
            "node": node,
            "action": "modal_update_openai_stt",
            "values": {
                "model": "gpt-4o-mini-transcribe",
                "api_key": "",
                "audio_path": "./modal.wav",
                "language": "fr",
                "prompt": "modal",
                "response_format": "json",
                "timeout_sec": "45",
                "api_base_url": fake_server.base_url,
                "chunking_enabled": True,
                "chunk_size_mb": "2",
                "chunk_duration_sec": "15",
                "chunk_overlap_sec": "1",
                "chunking_strategy": "auto",
                "experimental_audio_filter_enabled": False,
                "experimental_audio_filter": "",
                "sliding_context_enabled": False,
            },
        },
    )
    modal_config = modal_result.get("node_patch", {}).get("config", {})
    expect(modal_config.get("audio_path") == "./modal.wav", "The modal must persist the audio path.")
    expect(modal_config.get("timeout_sec") == 45, "The modal must normalize the timeout.")

    update_result = handle_block_ui_action(
        "openai_stt",
        {
            "node": node,
            "action": "inspector_update_openai_stt",
            "values": {"model": "gpt-4o-transcribe-diarize", "api_key": "sk-new", "response_format": ""},
        },
    )
    update_config = update_result.get("node_patch", {}).get("config", {})
    expect(update_config.get("model") == "gpt-4o-transcribe-diarize", "The model chosen in the inspector must be kept.")
    expect(update_config.get("api_key") == "sk-new", "A newly entered key must be persisted.")
    expect(update_config.get("response_format") == "diarized_json", "The diarize model must use diarized_json by default.")


def test_model_constraints(fake_server: FakeOpenAiSttHttpServer) -> None:
    with isolated_server() as server:
        audio_path = server.root_dir / "tmp" / "short.wav"
        write_silent_wav(audio_path, seconds=1)
        block = OpenAiSttBlock()
        base_config = openai_stt_node(fake_server.base_url, str(audio_path))["config"]

        start = len(fake_server.requests_log)
        result = block.execute_runtime(
            direct_context(
                root_dir=server.root_dir,
                run_dir=server.root_dir / "runs" / "direct",
                config={
                    **base_config,
                    "model": "gpt-4o-transcribe",
                    "response_format": "verbose_json",
                },
            )
        )
        expect(result.status == "success", "An invalid format for gpt-4o-transcribe must be normalized.")
        body = fake_server.requests_log[start]["body"]
        expect(b"verbose_json" not in body, "gpt-4o-transcribe must not send verbose_json.")
        expect(b'name="response_format"\r\n\r\njson' in body, "gpt-4o-transcribe must fall back to json.")

        start = len(fake_server.requests_log)
        result = block.execute_runtime(
            direct_context(
                root_dir=server.root_dir,
                run_dir=server.root_dir / "runs" / "direct-diarize",
                config={
                    **base_config,
                    "model": "gpt-4o-transcribe-diarize",
                    "prompt": "PROMPT_SHOULD_NOT_BE_SENT",
                    "response_format": "",
                    "chunking_strategy": "auto",
                },
            )
        )
        expect(result.status == "success", "The diarize model must be accepted.")
        body = fake_server.requests_log[start]["body"]
        expect(b"PROMPT_SHOULD_NOT_BE_SENT" not in body, "The prompt must not be sent to the diarize model.")
        expect(b"chunking_strategy" in body and b"auto" in body, "The diarize model must send chunking_strategy=auto.")
        expect(b"diarized_json" in body, "The diarize model must use diarized_json by default.")


def test_chunking_and_file_constraints(fake_server: FakeOpenAiSttHttpServer) -> None:
    with isolated_server() as server:
        block = OpenAiSttBlock()
        audio_path = server.root_dir / "tmp" / "long.wav"
        write_silent_wav(audio_path, seconds=70)
        base_config = openai_stt_node(fake_server.base_url, str(audio_path))["config"]

        start = len(fake_server.requests_log)
        result = block.execute_runtime(
            direct_context(
                root_dir=server.root_dir,
                run_dir=server.root_dir / "runs" / "chunked",
                config={**base_config, "chunking_enabled": True, "chunk_size_mb": 1},
            )
        )
        request_count = len(fake_server.requests_log) - start
        expect(result.status == "success", "WAV chunking must succeed without ffmpeg.")
        expect(request_count >= 2, "The WAV file must be sent as several chunks.")
        expect(
            result.metadata.get("openai_stt", {}).get("chunk_count") == request_count,
            "The metadata must expose the chunk count.",
        )
        chunk_logs = "\n".join(result.logs)
        expect("file validation OK" in chunk_logs, "The STT logs must trace the file validation.")
        expect("audio split into" in chunk_logs, "The STT logs must trace the chunk count.")
        expect("chunk 1/" in chunk_logs, "The STT logs must trace the OpenAI call per chunk.")
        expect("merge complete" in chunk_logs, "The STT logs must trace the final merge.")
        transcript = result.outputs[0].value
        expect("chunk 2" in transcript, "The transcript must keep the useful segments after the overlap.")
        expect("overlap 2" not in transcript, "The transcript must drop the pre-roll overlap segments.")
        expect("---00:30" in transcript, "The transcript must mark the move to the next chunk with mm:ss.")
        expect(transcript.index("---00:30") < transcript.index("chunk 2"), "The time marker must precede the chunk text.")
        chunked_raw = result.outputs[1].value
        expect('"chunks"' in chunked_raw, "The raw_json output must keep the per-chunk responses.")
        chunked_payload = json.loads(chunked_raw)
        expect(chunked_payload.get("chunk_duration_sec") == 30, "raw_json must expose the chunk window.")
        expect(chunked_payload.get("chunk_overlap_sec") == 2, "raw_json must expose the overlap.")
        expect("timestamped_text" in chunked_payload, "raw_json must expose the timestamped text.")
        expect(
            any(float(chunk.get("ignore_before_sec") or 0.0) == 2.0 for chunk in chunked_payload.get("chunks", [])[1:]),
            "Chunks after the first must expose ignore_before_sec=2.",
        )

        sliding_start = len(fake_server.requests_log)
        sliding_result = block.execute_runtime(
            direct_context(
                root_dir=server.root_dir,
                run_dir=server.root_dir / "runs" / "chunked-sliding-context",
                config={**base_config, "chunking_enabled": True, "chunk_size_mb": 1, "sliding_context_enabled": True},
            )
        )
        sliding_requests = fake_server.requests_log[sliding_start:]
        sliding_payload = json.loads(sliding_result.outputs[1].value)
        expect(sliding_result.status == "success", "Chunking with the sliding context must succeed.")
        expect(len(sliding_requests) >= 2, "The sliding context test must process several chunks.")
        expect(
            any(b"Context from the previous audio chunk" in request["body"] for request in sliding_requests[1:]),
            "The sliding context must add an instruction from the second chunk on.",
        )
        expect(
            any(b"chunk 1" in request["body"] for request in sliding_requests[1:]),
            "The sliding context must inject the previous chunk text.",
        )
        expect(
            any(chunk.get("sliding_context_used") for chunk in sliding_payload.get("chunks", [])[1:]),
            "raw_json must report which chunks used the sliding context.",
        )

        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            filtered_start = len(fake_server.requests_log)
            filtered_result = block.execute_runtime(
                direct_context(
                    root_dir=server.root_dir,
                    run_dir=server.root_dir / "runs" / "chunked-filtered",
                    config={
                        **base_config,
                        "chunking_enabled": True,
                        "chunk_size_mb": 1,
                        "experimental_audio_filter_enabled": True,
                        "experimental_audio_filter": "highpass=f=90,lowpass=f=7600,afftdn=nr=8,loudnorm=I=-16:TP=-2:LRA=8",
                    },
                )
            )
            filtered_requests = fake_server.requests_log[filtered_start:]
            filtered_logs = "\n".join(filtered_result.logs)
            expect(filtered_result.status == "success", "Chunking with the audio filter must succeed.")
            expect("audio filter" in filtered_logs, "The logs must trace the audio filter.")
            expect(
                any(b"Content-Type: audio/x-wav" in request["body"] for request in filtered_requests),
                "The audio filter must keep the source file's WAV container in this test.",
            )

        live_logs: list[str] = []
        start = len(fake_server.requests_log)
        live_result = block.execute_runtime(
            direct_context(
                root_dir=server.root_dir,
                run_dir=server.root_dir / "runs" / "chunked-live",
                config={**base_config, "chunking_enabled": True, "chunk_size_mb": 1},
                services={"append_log": live_logs.append},
            )
        )
        expect(live_result.status == "success", "Chunking with the live logger must succeed.")
        expect(len(fake_server.requests_log) - start >= 2, "The live logger test must process several chunks.")
        live_log_text = "\n".join(live_logs)
        expect("chunk 1/" in live_log_text, "The STT logs must be emitted live while the chunks run.")
        expect("merge complete" in live_log_text, "The live STT logs must trace the merge.")
        expect(not live_result.logs, "With live append_log, the STT logs must not be duplicated at the end of the node.")

        cancel_start = len(fake_server.requests_log)
        cancel_logs: list[str] = []
        cancel_result = block.execute_runtime(
            direct_context(
                root_dir=server.root_dir,
                run_dir=server.root_dir / "runs" / "chunked-cancel",
                config={**base_config, "chunking_enabled": True, "chunk_size_mb": 1},
                services={
                    "append_log": cancel_logs.append,
                    "cancel_requested": lambda: len(fake_server.requests_log) > cancel_start,
                },
            )
        )
        expect(cancel_result.status == "cancelled", "The STT block must honor cancel_requested between two chunks.")
        expect(len(fake_server.requests_log) - cancel_start == 1, "The STT block must stop before the next chunk.")
        expect("cancellation requested" in cancel_result.error, "The cancelled result must explain the cancellation.")
        expect("cancellation requested" in "\n".join(cancel_logs), "The live logs must trace the STT cancellation.")

        invalid_path = server.root_dir / "tmp" / "audio.txt"
        invalid_path.write_text("not audio", encoding="utf-8")
        invalid_result = block.execute_runtime(
            direct_context(
                root_dir=server.root_dir,
                run_dir=server.root_dir / "runs" / "invalid",
                config={**base_config, "audio_path": str(invalid_path)},
            )
        )
        expect(invalid_result.status == "failed", "An unsupported format must be refused.")
        expect("unsupported audio format" in invalid_result.error, "The error must explain the unsupported format.")

        old_max_size = openai_stt_module.OPENAI_STT_MAX_UPLOAD_BYTES
        try:
            openai_stt_module.OPENAI_STT_MAX_UPLOAD_BYTES = 8
            too_large_result = block.execute_runtime(
                direct_context(
                    root_dir=server.root_dir,
                    run_dir=server.root_dir / "runs" / "too-large",
                    config={**base_config, "chunking_enabled": False},
                )
            )
        finally:
            openai_stt_module.OPENAI_STT_MAX_UPLOAD_BYTES = old_max_size
        expect(too_large_result.status == "failed", "A file too large without chunking must be refused.")
        expect("chunking" in too_large_result.error, "The error must suggest enabling chunking.")


def test_introspection() -> None:
    description = describe_block("openai_stt")
    expect(description["default_config"]["model"] == "gpt-4o-transcribe", "The introspected default model must be gpt-4o-transcribe.")
    expect(description["default_config"]["chunk_size_mb"] == 24, "The default chunk size must be introspected.")
    expect(description["default_config"]["chunk_duration_sec"] == 30, "The default chunk duration must be introspected.")
    expect(description["default_config"]["chunk_overlap_sec"] == 2, "The default overlap must be introspected.")
    expect(
        description["default_config"]["experimental_audio_filter_enabled"] is False,
        "The audio filter must be disabled by default.",
    )
    expect(
        "loudnorm" in description["default_config"]["experimental_audio_filter"],
        "The default audio filter must be introspected.",
    )
    expect(description["default_config"]["sliding_context_enabled"] is False, "The sliding context must be disabled by default.")
    expect(description["capabilities"]["runtime_executable"], "openai_stt must be runtime_executable.")
    expect(description["capabilities"]["active_worker"], "openai_stt must be active_worker.")
    expect(description["capabilities"]["file_browser"], "openai_stt must expose the audio file browsing.")


def main() -> None:
    with FakeSttServer() as fake_server:
        run_openai_stt_case("centralized", fake_server)
        run_openai_stt_case("zeromq_active", fake_server)
        test_http_requests(fake_server)
        test_inspector_contract(fake_server)
        test_model_constraints(fake_server)
        test_chunking_and_file_constraints(fake_server)
        test_introspection()
    # Direct smoke on the class export as well.
    expect(OpenAiSttBlock().kind == "openai_stt", "The OpenAI STT block must expose its kind.")
    print("[ok] F5.17_openai_stt_block")


if __name__ == "__main__":
    main()
