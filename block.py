# -----------------------------------------------------------------------------
# Role: Implements the OpenAI STT block runtime and UI contract.
# File Name: block.py
# Author: Alexandre EL
# Email: alex@hackinvent.com
# Created Date: 2024-12-19
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import uuid
import wave

from bloxsmith_app.block_api import (
    APPLICATION_JSON,
    BlockDefinition,
    BlockRuntimeContext,
    BlockRuntimeOutput,
    BlockRuntimeResult,
    render_inspector_template,
    render_node_card_template,
    render_path_browser_control,
    TEXT_PLAIN,
)


OPENAI_STT_MODELS = (
    "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe-diarize",
    "whisper-1",
)
OPENAI_STT_RESPONSE_FORMATS = (
    "json",
    "text",
    "verbose_json",
    "diarized_json",
    "srt",
    "vtt",
)
DEFAULT_OPENAI_STT_MODEL = "gpt-4o-transcribe"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"
DEFAULT_OPENAI_STT_TIMEOUT_SEC = 120
DEFAULT_OPENAI_STT_CHUNK_SIZE_MB = 24
DEFAULT_OPENAI_STT_CHUNK_DURATION_SEC = 30
DEFAULT_OPENAI_STT_CHUNK_OVERLAP_SEC = 2
DEFAULT_OPENAI_STT_AUDIO_FILTER = "highpass=f=90,lowpass=f=7600,afftdn=nr=8,loudnorm=I=-16:TP=-2:LRA=8"
OPENAI_STT_MIN_CHUNK_SIZE_MB = 1
OPENAI_STT_MAX_CHUNK_SIZE_MB = 25
OPENAI_STT_MIN_CHUNK_DURATION_SEC = 1
OPENAI_STT_MAX_CHUNK_DURATION_SEC = 600
OPENAI_STT_MIN_CHUNK_OVERLAP_SEC = 0
OPENAI_STT_MAX_CHUNK_OVERLAP_SEC = 60
OPENAI_STT_MAX_UPLOAD_BYTES = 25 * 1024 * 1024
OPENAI_STT_SUPPORTED_EXTENSIONS = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm")
OPENAI_STT_DIARIZE_MODEL = "gpt-4o-transcribe-diarize"
OPENAI_STT_RESPONSE_FORMATS_BY_MODEL = {
    "gpt-4o-transcribe": ("json", "text"),
    "gpt-4o-mini-transcribe": ("json", "text"),
    OPENAI_STT_DIARIZE_MODEL: ("json", "text", "diarized_json"),
    "whisper-1": ("json", "text", "srt", "verbose_json", "vtt"),
}
FFMPEG_PROCESS_TIMEOUT_SEC = 600
TIMESTAMP_BUCKET_SECONDS = 30
OVERLAP_DROP_TOLERANCE_SECONDS = 1.0


class OpenAiSttBlockError(ValueError):
    """Raised when the OpenAI STT block cannot transcribe the input audio."""


class OpenAiSttBlockCancelled(RuntimeError):
    """Raised when the OpenAI STT run is cancelled by the runtime."""


@dataclass(frozen=True)
class AudioSegmentPlan:
    """Structured data used by this block implementation."""
    path: Path
    index: int
    offset_sec: float
    ignore_before_sec: float
    nominal_start_sec: float
    nominal_duration_sec: float
    extract_start_sec: float
    extract_duration_sec: float


# Functional behavior:
# FB1 - Resolve audio from the input port or from config.audio_path.
# FB2 - Validate OpenAI STT input constraints before calling the HTTP API.
# FB3 - Split audio into upload-safe, timestamped chunks with overlap when enabled.
# FB4 - Call the OpenAI-compatible /v1/audio/transcriptions endpoint with multipart form data.
# FB5 - Normalize text, JSON, and diarized JSON responses into transcript/raw JSON outputs without leaking api_key.
# FB6 - Render and update model/API configuration through the block-owned inspector.
# FB7 - Run through the generic runtime path used by both centralized and zeromq_active execution modes.
class OpenAiSttBlock(BlockDefinition):
    """Autonomous block implementation for `OpenAiSttBlock`."""
    kind = "openai_stt"

    def render_node_card(self, *, node: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the OpenAI STT canvas card body from the block-owned template."""

        config = self._ui_config(node)
        return render_node_card_template(
            block=self,
            node=node,
            node_classes=["openai-stt-node"],
            replacements={
                "title": node.get("title") or self.default_title(),
                "model": config["model"],
                "chunking": "chunking on" if config.get("chunking_enabled") else "chunking off",
                "format": config["response_format"],
            },
        )

    def render_modal(self, *, node: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the OpenAI STT modal with the same path browser and settings as the inspector."""

        config = self._ui_config(node)
        template = (self.directory / "block_modal.html").read_text(encoding="utf-8")
        template = template.replace("{{ config_fields_html }}", self._render_modal_config_fields(config))
        template = template.replace(
            "{{ technical_config_fields_html }}",
            self._render_modal_technical_config_fields(node),
        )
        html = self._render_generic_modal_template(
            template=template,
            node=node,
            payload=payload or {},
        )
        return {
            "html": html,
            "context": {
                "node_id": str(node.get("id") or ""),
                "node_kind": self.kind,
                "model": str(config["model"]),
                "api_key_configured": bool(config.get("api_key")),
            },
        }

    def render_inspector_panel(self, *, node: dict[str, Any], payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Render the block-owned inspector panel HTML for the selected node.

        Args:
            node: Serialized graph node handled by the block.
            payload: Optional UI or runtime payload provided by the framework.
        """
        config = self._ui_config(node)
        template = (self.directory / "inspector_panel.html").read_text(encoding="utf-8")
        html = render_inspector_template(
            template=self._apply_ui_replacements(template, config),
            node={**node, "type": self.kind, "kind": self.kind},
            payload=payload,
            show_duplicate=False,
        )
        return {
            "html": html,
            "context": {
                "node_id": str(node.get("id") or ""),
                "model": str(config["model"]),
                "api_key_configured": bool(config.get("api_key")),
                "full_panel": True,
            },
        }

    def handle_ui_action(
        self,
        *,
        node: dict[str, Any],
        action: str,
        values: dict[str, Any],
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle a block-owned UI action and return the updated node payload.

        Args:
            node: Serialized graph node handled by the block.
            action: Block-owned action name requested by the frontend.
            values: Values value used by this block helper.
            payload: Optional UI or runtime payload provided by the framework.
        """
        if action not in {"inspector_update_openai_stt", "modal_update_openai_stt"}:
            return super().handle_ui_action(node=node, action=action, values=values, payload=payload)

        config_patch = {
            "model": self._normalize_model(values.get("model")),
            "audio_path": str(values.get("audio_path") or "").strip(),
            "language": str(values.get("language") or "").strip(),
            "prompt": str(values.get("prompt") or ""),
            "response_format": self._normalize_response_format(
                values.get("response_format"),
                model=self._normalize_model(values.get("model")),
            ),
            "timeout_sec": self._normalize_timeout(values.get("timeout_sec")),
            "api_base_url": self._normalize_base_url(values.get("api_base_url")),
            "chunking_enabled": self._bool(values.get("chunking_enabled", True)),
            "chunk_size_mb": self._normalize_chunk_size_mb(values.get("chunk_size_mb")),
            "chunk_duration_sec": self._normalize_chunk_duration_sec(values.get("chunk_duration_sec")),
            "chunk_overlap_sec": self._normalize_chunk_overlap_sec(
                values.get("chunk_overlap_sec"),
                chunk_duration_sec=self._normalize_chunk_duration_sec(values.get("chunk_duration_sec")),
            ),
            "chunking_strategy": self._normalize_chunking_strategy(values.get("chunking_strategy")),
            "experimental_audio_filter_enabled": self._bool(
                values.get("experimental_audio_filter_enabled", False)
            ),
            "experimental_audio_filter": self._normalize_audio_filter(values.get("experimental_audio_filter")),
            "sliding_context_enabled": self._bool(values.get("sliding_context_enabled", False)),
        }
        if self._bool(values.get("clear_api_key")):
            config_patch["api_key"] = ""
        else:
            api_key = str(values.get("api_key") or "").strip()
            if api_key:
                config_patch["api_key"] = api_key
        return {
            "node_patch": {"config": config_patch},
            "rerender_inspector": False,
        }

    def _apply_ui_replacements(self, template: str, config: dict[str, Any]) -> str:
        """Fill OpenAI STT UI placeholders shared by modal and inspector templates."""

        return self._fill_ui_template(
            template,
            {
                **self._ui_replacements(config, id_prefix="openaiStt"),
                "parameters_fields_html": self._render_parameter_fields(
                    config,
                    id_prefix="openaiStt",
                    apply_button_attr="data-block-apply",
                ),
            },
        )

    def _fill_ui_template(self, template: str, replacements: dict[str, str]) -> str:
        """Replace OpenAI STT placeholders in a block-owned HTML fragment."""

        html = template
        for key, value in replacements.items():
            html = html.replace(f"{{{{ {key} }}}}", str(value))
        return html

    def _ui_replacements(self, config: dict[str, Any], *, id_prefix: str) -> dict[str, str]:
        """Return escaped OpenAI STT configuration values for block-owned HTML."""

        return {
            "id_prefix": escape(id_prefix, quote=True),
            "model_options": self._select_options(OPENAI_STT_MODELS, str(config["model"])),
            "response_format_options": self._select_options(
                self._response_formats_for_model(str(config["model"])),
                str(config["response_format"]),
            ),
            "api_key_placeholder": "Cle configuree" if config.get("api_key") else "sk-...",
            "audio_path_browser_html": self._render_audio_path_browser(
                config,
                input_id=f"{id_prefix}AudioPath",
            ),
            "language": escape(str(config.get("language") or "")),
            "prompt": escape(str(config.get("prompt") or "")),
            "timeout_sec": escape(str(config.get("timeout_sec") or DEFAULT_OPENAI_STT_TIMEOUT_SEC)),
            "api_base_url": escape(str(config.get("api_base_url") or DEFAULT_OPENAI_BASE_URL)),
            "chunking_enabled_checked": "checked" if config.get("chunking_enabled") else "",
            "chunk_size_mb": escape(str(config.get("chunk_size_mb") or DEFAULT_OPENAI_STT_CHUNK_SIZE_MB)),
            "chunk_duration_sec": escape(
                str(config.get("chunk_duration_sec") or DEFAULT_OPENAI_STT_CHUNK_DURATION_SEC)
            ),
            "chunk_overlap_sec": escape(
                str(config.get("chunk_overlap_sec") or DEFAULT_OPENAI_STT_CHUNK_OVERLAP_SEC)
            ),
            "chunking_strategy_options": self._select_options(
                ("auto", "none"),
                str(config.get("chunking_strategy") or "auto"),
            ),
            "experimental_audio_filter_enabled_checked": (
                "checked" if config.get("experimental_audio_filter_enabled") else ""
            ),
            "experimental_audio_filter": escape(
                str(config.get("experimental_audio_filter") or DEFAULT_OPENAI_STT_AUDIO_FILTER)
            ),
            "sliding_context_checked": "checked" if config.get("sliding_context_enabled") else "",
        }

    def _render_audio_path_browser(self, config: dict[str, Any], *, input_id: str) -> str:
        """Render the shared file browser used to select the default audio path."""

        return render_path_browser_control(
            input_id=input_id,
            label="Default audio path",
            value=str(config.get("audio_path") or ""),
            placeholder="./audio.wav",
            input_attrs="data-openai-stt-audio-path",
            select_mode="file",
        )

    def _render_parameter_fields(self, config: dict[str, Any], *, id_prefix: str, apply_button_attr: str) -> str:
        """Render the shared OpenAI STT parameter form for inspector and modal."""

        template = (self.directory / "parameters_fields.html").read_text(encoding="utf-8")
        return self._fill_ui_template(
            template,
            {
                **self._ui_replacements(config, id_prefix=id_prefix),
                "apply_button_attr": apply_button_attr,
            },
        )

    def _render_modal_config_fields(self, config: dict[str, Any]) -> str:
        """Render modal parameters from the same block-owned fragment as the inspector."""

        return self._render_parameter_fields(
            config,
            id_prefix="openaiSttModal",
            apply_button_attr="data-openai-stt-apply",
        )

    def _render_modal_technical_config_fields(self, node: dict[str, Any]) -> str:
        """Render OpenAI STT config keys not owned by the dedicated modal controls."""

        config = self.default_config()
        node_config = node.get("config")
        if isinstance(node_config, dict):
            config.update(node_config)
        hidden_keys = {
            "model",
            "api_key",
            "audio_path",
            "language",
            "prompt",
            "response_format",
            "timeout_sec",
            "api_base_url",
            "chunking_enabled",
            "chunk_size_mb",
            "chunk_duration_sec",
            "chunk_overlap_sec",
            "chunking_strategy",
            "experimental_audio_filter_enabled",
            "experimental_audio_filter",
            "sliding_context_enabled",
        }
        fields = [
            self._render_generic_modal_config_field(key, value)
            for key, value in config.items()
            if str(key) not in hidden_keys
        ]
        if not fields:
            return '<div class="ports-editor-empty">Aucun attribut technique supplementaire.</div>'
        return "\n".join(
            [
                '<div class="ports-editor-section">',
                '  <div class="ports-editor-header"><span class="group-label">Attributs techniques</span></div>',
                *fields,
                "</div>",
            ]
        )

    def preview_received(self, *, node: Any, **runtime_services: Any) -> str:
        """Return a compact preview value for runtime display surfaces.

        Args:
            node: Serialized graph node handled by the block.
            runtime_services: Runtime services value used by this block helper.
        """
        config = getattr(node, "config", {}) if isinstance(getattr(node, "config", {}), dict) else {}
        return str(config.get("audio_path") or "audio non defini")

    def execute_runtime(self, context: BlockRuntimeContext) -> BlockRuntimeResult:
        """Execute the block through the generic runtime context and return runtime outputs.

        Args:
            context: Generic runtime context injected by the execution engine.
        """
        logs: list[str] = []
        try:
            config = self._runtime_config(context.config)
            audio_path = self._resolve_audio_path(context=context, config=config)
            display_path = self._display_path(audio_path, context.root_dir)
            self._emit_log(
                context,
                logs,
                f"[openai-stt] {context.node_id}: file={display_path} "
                f"size={self._format_bytes(audio_path.stat().st_size)} model={config['model']} "
                f"format={config['response_format']} timeout={config['timeout_sec']}s.",
            )
            self._emit_log(
                context,
                logs,
                f"[openai-stt] {context.node_id}: chunking="
                f"{'on' if config['chunking_enabled'] else 'off'} "
                f"max={config['chunk_size_mb']} MB window={config['chunk_duration_sec']}s "
                f"overlap={config['chunk_overlap_sec']}s "
                f"audio_filter={'on' if config['experimental_audio_filter_enabled'] else 'off'} "
                f"sliding_context={'on' if config['sliding_context_enabled'] else 'off'}.",
            )
            if config["experimental_audio_filter_enabled"]:
                self._emit_log(
                    context,
                    logs,
                    f"[openai-stt] {context.node_id}: audio filter ffmpeg={config['experimental_audio_filter']}.",
                )
            response = self._transcribe_audio(
                audio_path=audio_path,
                config=config,
                context=context,
                logs=logs,
            )
            transcript = self._transcript_from_response(response)
            raw_json = self._raw_json_from_response(response, transcript=transcript)
            outputs = self._runtime_outputs(context, transcript=transcript, raw_json=raw_json)
            chunk_count = int(response.get("chunk_count") or 1)
            self._emit_log(
                context,
                logs,
                f"[done] OpenAI STT {context.node_id}: {len(transcript)} character(s), "
                f"{chunk_count} chunk(s).",
            )
            return BlockRuntimeResult(
                status="success",
                outputs=outputs,
                logs=logs,
                last_message=transcript,
                content_type=TEXT_PLAIN,
                worker_received=display_path,
                metadata={
                    "openai_stt": {
                        "model": config["model"],
                        "response_format": config["response_format"],
                        "language": config["language"],
                        "audio_path": display_path,
                        "chunk_count": chunk_count,
                        "chunk_size_mb": config["chunk_size_mb"],
                        "experimental_audio_filter_enabled": config["experimental_audio_filter_enabled"],
                        "experimental_audio_filter": config["experimental_audio_filter"]
                        if config["experimental_audio_filter_enabled"]
                        else "",
                        "sliding_context_enabled": config["sliding_context_enabled"],
                    }
                },
            )
        except OpenAiSttBlockCancelled as exc:
            message = str(exc) or "cancelled"
            self._emit_log(context, logs, f"[cancel] OpenAI STT {context.node_id}: {message}.")
            return BlockRuntimeResult(
                status="cancelled",
                outputs=[],
                logs=logs,
                error=message,
                exit_code=0,
                last_message=message,
                content_type=TEXT_PLAIN,
                worker_received="-",
            )
        except OpenAiSttBlockError as exc:
            message = str(exc)
            self._emit_log(context, logs, f"[openai-stt-error] {context.node_id}: {message}")
            return BlockRuntimeResult(
                status="failed",
                outputs=[],
                logs=logs,
                error=message,
                exit_code=1,
                last_message=message,
                content_type=TEXT_PLAIN,
                worker_received="-",
            )

    def _ui_config(self, node: dict[str, Any]) -> dict[str, Any]:
        """Provide internal OpenAiSttBlock behavior for `_ui_config`.

        Args:
            node: Serialized graph node handled by the block.
        """
        raw = node.get("config") if isinstance(node.get("config"), dict) else {}
        return self._runtime_config(raw)

    def _runtime_config(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        """Provide internal OpenAiSttBlock behavior for `_runtime_config`.

        Args:
            raw: Raw value used by this block helper.
        """
        config = raw if isinstance(raw, dict) else {}
        model = self._normalize_model(config.get("model"))
        chunk_duration_sec = self._normalize_chunk_duration_sec(config.get("chunk_duration_sec"))
        return {
            "model": model,
            "api_key": str(config.get("api_key") or os.getenv("OPENAI_API_KEY") or "").strip(),
            "api_base_url": self._normalize_base_url(config.get("api_base_url")),
            "audio_path": str(config.get("audio_path") or "").strip(),
            "language": str(config.get("language") or "").strip(),
            "prompt": str(config.get("prompt") or ""),
            "response_format": self._normalize_response_format(config.get("response_format"), model=model),
            "timeout_sec": self._normalize_timeout(config.get("timeout_sec")),
            "chunking_enabled": self._bool(config.get("chunking_enabled", True)),
            "chunk_size_mb": self._normalize_chunk_size_mb(config.get("chunk_size_mb")),
            "chunk_duration_sec": chunk_duration_sec,
            "chunk_overlap_sec": self._normalize_chunk_overlap_sec(
                config.get("chunk_overlap_sec"),
                chunk_duration_sec=chunk_duration_sec,
            ),
            "chunking_strategy": self._normalize_chunking_strategy(config.get("chunking_strategy")),
            "experimental_audio_filter_enabled": self._bool(
                config.get("experimental_audio_filter_enabled", False)
            ),
            "experimental_audio_filter": self._normalize_audio_filter(config.get("experimental_audio_filter")),
            "sliding_context_enabled": self._bool(config.get("sliding_context_enabled", False)),
        }

    def _normalize_model(self, value: Any) -> str:
        """Normalize a raw value into the format expected by the block.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        model = str(value or DEFAULT_OPENAI_STT_MODEL).strip()
        if model in OPENAI_STT_MODELS:
            return model
        return DEFAULT_OPENAI_STT_MODEL

    def _normalize_response_format(self, value: Any, *, model: str) -> str:
        """Normalize a raw value into the format expected by the block.

        Args:
            value: Value to normalize, render, serialize, or process.
            model: Model value used by this block helper.
        """
        raw = str(value or "").strip()
        allowed = self._response_formats_for_model(model)
        if not raw:
            return "diarized_json" if model == OPENAI_STT_DIARIZE_MODEL else "json"
        if raw in allowed:
            return raw
        return "diarized_json" if model == OPENAI_STT_DIARIZE_MODEL else "json"

    def _response_formats_for_model(self, model: str) -> tuple[str, ...]:
        """Provide internal OpenAiSttBlock behavior for `_response_formats_for_model`.

        Args:
            model: Model value used by this block helper.
        """
        return OPENAI_STT_RESPONSE_FORMATS_BY_MODEL.get(model, ("json", "text"))

    def _normalize_timeout(self, value: Any) -> int:
        """Normalize a raw value into the format expected by the block.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        try:
            timeout = int(value)
        except (TypeError, ValueError):
            timeout = DEFAULT_OPENAI_STT_TIMEOUT_SEC
        return max(1, min(timeout, 3600))

    def _normalize_base_url(self, value: Any) -> str:
        """Normalize a raw value into the format expected by the block.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        base_url = str(value or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
        return base_url or DEFAULT_OPENAI_BASE_URL

    def _normalize_audio_filter(self, value: Any) -> str:
        """Normalize a raw value into the format expected by the block.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        audio_filter = str(value or "").strip()
        return audio_filter or DEFAULT_OPENAI_STT_AUDIO_FILTER

    def _normalize_chunk_size_mb(self, value: Any) -> int:
        """Normalize a raw value into the format expected by the block.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        try:
            size = int(value)
        except (TypeError, ValueError):
            size = DEFAULT_OPENAI_STT_CHUNK_SIZE_MB
        return max(OPENAI_STT_MIN_CHUNK_SIZE_MB, min(size, OPENAI_STT_MAX_CHUNK_SIZE_MB))

    def _normalize_chunk_duration_sec(self, value: Any) -> int:
        """Normalize a raw value into the format expected by the block.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        try:
            duration = int(value)
        except (TypeError, ValueError):
            duration = DEFAULT_OPENAI_STT_CHUNK_DURATION_SEC
        return max(OPENAI_STT_MIN_CHUNK_DURATION_SEC, min(duration, OPENAI_STT_MAX_CHUNK_DURATION_SEC))

    def _normalize_chunk_overlap_sec(self, value: Any, *, chunk_duration_sec: int) -> int:
        """Normalize a raw value into the format expected by the block.

        Args:
            value: Value to normalize, render, serialize, or process.
            chunk_duration_sec: Duration in seconds.
        """
        try:
            overlap = int(value)
        except (TypeError, ValueError):
            overlap = DEFAULT_OPENAI_STT_CHUNK_OVERLAP_SEC
        overlap = max(OPENAI_STT_MIN_CHUNK_OVERLAP_SEC, min(overlap, OPENAI_STT_MAX_CHUNK_OVERLAP_SEC))
        return min(overlap, max(0, int(chunk_duration_sec) - 1))

    def _normalize_chunking_strategy(self, value: Any) -> str:
        """Normalize a raw value into the format expected by the block.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        raw = str(value or "auto").strip().lower()
        if raw in {"", "auto"}:
            return "auto"
        if raw in {"none", "off", "disabled", "false", "0"}:
            return "none"
        return "auto"

    def _select_options(self, values: tuple[str, ...], selected: str) -> str:
        """Provide internal OpenAiSttBlock behavior for `_select_options`.

        Args:
            values: Values value used by this block helper.
            selected: Selected value used by this block helper.
        """
        return "\n".join(
            f'<option value="{escape(value)}"{" selected" if value == selected else ""}>{escape(value)}</option>'
            for value in values
        )

    def _resolve_audio_path(self, *, context: BlockRuntimeContext, config: dict[str, Any]) -> Path:
        """Resolve a configured value against runtime or project context.

        Args:
            context: Generic runtime context injected by the execution engine.
            config: Raw or normalized block configuration.
        """
        raw_path = str(
            context.input_value("audio")
            or context.input_value("1")
            or context.input_message
            or config.get("audio_path")
            or ""
        ).strip()
        if raw_path.startswith("file://"):
            raw_path = raw_path[7:]
        if not raw_path:
            raise OpenAiSttBlockError("chemin audio manquant.")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (context.root_dir.resolve() / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise OpenAiSttBlockError(f"audio file not found: {self._display_path(path, context.root_dir)}")
        return path

    def _transcribe_audio(
        self,
        *,
        audio_path: Path,
        config: dict[str, Any],
        context: BlockRuntimeContext,
        logs: list[str],
    ) -> dict[str, Any]:
        """Provide internal OpenAiSttBlock behavior for `_transcribe_audio`.

        Args:
            audio_path: Filesystem path handled by the block.
            config: Raw or normalized block configuration.
            context: Generic runtime context injected by the execution engine.
            logs: Logs value used by this block helper.
        """
        self._validate_audio_file(audio_path=audio_path, config=config)
        self._emit_log(context, logs, f"[openai-stt] {context.node_id}: file validation OK.")
        self._raise_if_cancelled(context)
        segment_plans, temp_dir = self._audio_segment_plans(audio_path=audio_path, config=config, context=context, logs=logs)
        try:
            if len(segment_plans) == 1 and segment_plans[0].path == audio_path:
                self._raise_if_cancelled(context)
                self._emit_log(
                    context,
                    logs,
                    f"[openai-stt] {context.node_id}: full-file OpenAI upload "
                    f"({self._format_bytes(audio_path.stat().st_size)}).",
                )
                response = self._transcribe(audio_path=segment_plans[0].path, config=config)
                self._emit_log(
                    context,
                    logs,
                    f"[openai-stt] {context.node_id}: OpenAI HTTP response "
                    f"{response.get('status_code', '?')} "
                    f"({len(self._transcript_from_response(response))} character(s)).",
                )
                response.setdefault("chunk_count", 1)
                return response

            responses: list[dict[str, Any]] = []
            self._raise_if_cancelled(context)
            self._emit_log(
                context,
                logs,
                f"[openai-stt] {context.node_id}: audio split into {len(segment_plans)} chunk(s).",
            )
            previous_chunk_text = ""
            for segment_plan in segment_plans:
                self._raise_if_cancelled(context)
                if segment_plan.path.stat().st_size > OPENAI_STT_MAX_UPLOAD_BYTES:
                    raise OpenAiSttBlockError(
                        "audio segment too large after splitting; "
                        "reduce chunk_size_mb or compress the file."
                    )
                self._emit_log(
                    context,
                    logs,
                    f"[openai-stt] {context.node_id}: chunk {segment_plan.index}/{len(segment_plans)} "
                    f"upload={self._format_bytes(segment_plan.path.stat().st_size)} "
                    f"nominal={self._format_duration(segment_plan.nominal_start_sec)}"
                    f"+{self._format_duration(segment_plan.nominal_duration_sec)} "
                    f"extract={self._format_duration(segment_plan.extract_start_sec)}"
                    f"+{self._format_duration(segment_plan.extract_duration_sec)} "
                    f"overlap_ignore={self._format_duration(segment_plan.ignore_before_sec)}.",
                )
                prompt_override = self._prompt_for_chunk(config=config, previous_chunk_text=previous_chunk_text)
                if config.get("sliding_context_enabled") and previous_chunk_text:
                    self._emit_log(
                        context,
                        logs,
                        f"[openai-stt] {context.node_id}: chunk {segment_plan.index}/{len(segment_plans)} "
                        "sliding context injected.",
                    )
                segment_response = self._transcribe(
                    audio_path=segment_plan.path,
                    config=config,
                    prompt_override=prompt_override,
                )
                self._raise_if_cancelled(context)
                chunk_text = self._transcript_from_response(segment_response).strip()
                self._emit_log(
                    context,
                    logs,
                    f"[openai-stt] {context.node_id}: chunk {segment_plan.index}/{len(segment_plans)} "
                    f"finished HTTP {segment_response.get('status_code', '?')} "
                    f"({len(chunk_text)} character(s)).",
                )
                segment_payload = dict(segment_response)
                segment_payload["chunk_index"] = segment_plan.index
                segment_payload["chunk_file"] = segment_plan.path.name
                segment_payload["sliding_context_used"] = bool(
                    config.get("sliding_context_enabled") and previous_chunk_text
                )
                segment_payload["offset_sec"] = round(segment_plan.offset_sec, 3)
                segment_payload["ignore_before_sec"] = round(segment_plan.ignore_before_sec, 3)
                segment_payload["nominal_start_sec"] = round(segment_plan.nominal_start_sec, 3)
                segment_payload["nominal_duration_sec"] = round(segment_plan.nominal_duration_sec, 3)
                segment_payload["extract_start_sec"] = round(segment_plan.extract_start_sec, 3)
                segment_payload["extract_duration_sec"] = round(segment_plan.extract_duration_sec, 3)
                responses.append(segment_payload)
                previous_chunk_text = chunk_text

            self._raise_if_cancelled(context)
            self._emit_log(
                context,
                logs,
                f"[openai-stt] {context.node_id}: merge of {len(responses)} chunk(s) "
                f"with overlap removal={config['chunk_overlap_sec']}s.",
            )
            merged = self._merge_chunk_responses(responses=responses, config=config)
            self._emit_log(
                context,
                logs,
                f"[openai-stt] {context.node_id}: merge complete "
                f"({len(self._transcript_from_response(merged))} character(s)).",
            )
            return merged
        finally:
            if temp_dir is not None:
                self._emit_log(context, logs, f"[openai-stt] {context.node_id}: temporary chunk cleanup.")
                temp_dir.cleanup()

    def _validate_audio_file(self, *, audio_path: Path, config: dict[str, Any]) -> None:
        """Provide internal OpenAiSttBlock behavior for `_validate_audio_file`.

        Args:
            audio_path: Filesystem path handled by the block.
            config: Raw or normalized block configuration.
        """
        suffix = audio_path.suffix.lower()
        if suffix not in OPENAI_STT_SUPPORTED_EXTENSIONS:
            allowed = ", ".join(ext.lstrip(".") for ext in OPENAI_STT_SUPPORTED_EXTENSIONS)
            raise OpenAiSttBlockError(f"unsupported audio format ({suffix or 'no extension'}). Formats: {allowed}.")
        size = audio_path.stat().st_size
        if size <= 0:
            raise OpenAiSttBlockError("empty audio file.")
        if size > OPENAI_STT_MAX_UPLOAD_BYTES and not config.get("chunking_enabled"):
            raise OpenAiSttBlockError(
                "audio file larger than 25 MB; enable chunking or compress the file."
            )

    def _audio_segment_plans(
        self,
        *,
        audio_path: Path,
        config: dict[str, Any],
        context: BlockRuntimeContext,
        logs: list[str],
    ) -> tuple[list[AudioSegmentPlan], tempfile.TemporaryDirectory[str] | None]:
        """Provide internal OpenAiSttBlock behavior for `_audio_segment_plans`.

        Args:
            audio_path: Filesystem path handled by the block.
            config: Raw or normalized block configuration.
            context: Generic runtime context injected by the execution engine.
            logs: Logs value used by this block helper.
        """
        size = audio_path.stat().st_size
        max_bytes = min(int(config["chunk_size_mb"]) * 1024 * 1024, OPENAI_STT_MAX_UPLOAD_BYTES)
        single_plan = AudioSegmentPlan(
            path=audio_path,
            index=1,
            offset_sec=0.0,
            ignore_before_sec=0.0,
            nominal_start_sec=0.0,
            nominal_duration_sec=0.0,
            extract_start_sec=0.0,
            extract_duration_sec=0.0,
        )
        audio_filter = str(config.get("experimental_audio_filter") or "").strip()
        use_audio_filter = bool(config.get("experimental_audio_filter_enabled") and audio_filter)
        if not config.get("chunking_enabled"):
            if not use_audio_filter:
                self._emit_log(context, logs, f"[openai-stt] {context.node_id}: chunking disabled, no splitting.")
                return [single_plan], None
            self._emit_log(
                context,
                logs,
                f"[openai-stt] {context.node_id}: chunking disabled, audio filter through ffmpeg.",
            )
            return self._single_ffmpeg_segment_plan(
                audio_path=audio_path,
                config=config,
                context=context,
                logs=logs,
                max_bytes=max_bytes,
            )

        suffix = audio_path.suffix.lower()
        if suffix == ".wav":
            self._emit_log(context, logs, f"[openai-stt] {context.node_id}: native WAV analysis.")
            try:
                duration_seconds = self._probe_wav_duration_seconds(audio_path)
                self._emit_log(
                    context,
                    logs,
                    f"[openai-stt] {context.node_id}: audio duration={self._format_duration(duration_seconds)}.",
                )
            except (wave.Error, EOFError, OSError):
                if size <= max_bytes and not use_audio_filter:
                    self._emit_log(
                        context,
                        logs,
                        f"[openai-stt] {context.node_id}: non-standard WAV but size <= limit; direct upload.",
                    )
                    return [single_plan], None
                if use_audio_filter:
                    self._emit_log(
                        context,
                        logs,
                        f"[openai-stt] {context.node_id}: non-standard WAV, ffprobe analysis for the audio filter.",
                    )
                    duration_seconds = self._probe_duration_seconds(audio_path)
                else:
                    duration_seconds = -1.0
            if size <= max_bytes and 0 < duration_seconds <= float(config["chunk_duration_sec"]):
                if use_audio_filter:
                    self._emit_log(
                        context,
                        logs,
                        f"[openai-stt] {context.node_id}: no split needed, audio filter in a single segment.",
                    )
                    return self._single_ffmpeg_segment_plan(
                        audio_path=audio_path,
                        config=config,
                        context=context,
                        logs=logs,
                        max_bytes=max_bytes,
                    )
                self._emit_log(
                    context,
                    logs,
                    f"[openai-stt] {context.node_id}: no split needed "
                    f"({self._format_bytes(size)} <= {self._format_bytes(max_bytes)}, "
                    f"duration <= {config['chunk_duration_sec']}s).",
                )
                return [single_plan], None
        else:
            if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
                if size <= max_bytes and not use_audio_filter:
                    self._emit_log(
                        context,
                        logs,
                        f"[openai-stt] {context.node_id}: ffmpeg missing but size <= limit; direct upload.",
                    )
                    return [single_plan], None
                raise OpenAiSttBlockError(
                    "chunking is required for this file, but ffmpeg/ffprobe is unavailable."
                )
            self._emit_log(context, logs, f"[openai-stt] {context.node_id}: audio analysis through ffprobe.")
            duration_seconds = self._probe_duration_seconds(audio_path)
            self._emit_log(
                context,
                logs,
                f"[openai-stt] {context.node_id}: audio duration="
                f"{self._format_duration(duration_seconds) if duration_seconds > 0 else 'unknown'}.",
            )
            if size <= max_bytes and (
                duration_seconds <= 0 or duration_seconds <= float(config["chunk_duration_sec"])
            ):
                if use_audio_filter:
                    self._emit_log(
                        context,
                        logs,
                        f"[openai-stt] {context.node_id}: no split needed, audio filter in a single segment.",
                    )
                    return self._single_ffmpeg_segment_plan(
                        audio_path=audio_path,
                        config=config,
                        context=context,
                        logs=logs,
                        max_bytes=max_bytes,
                    )
                self._emit_log(
                    context,
                    logs,
                    f"[openai-stt] {context.node_id}: no split needed "
                    f"({self._format_bytes(size)} <= {self._format_bytes(max_bytes)}).",
                )
                return [single_plan], None

        parent_dir = context.run_dir if context.run_dir else None
        if parent_dir is not None:
            parent_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(
            prefix=f"{context.node_id}-openai-stt-",
            dir=str(parent_dir) if parent_dir else None,
        )
        work_dir = Path(temp_dir.name)
        try:
            if audio_path.suffix.lower() == ".wav":
                if use_audio_filter:
                    self._emit_log(
                        context,
                        logs,
                        f"[openai-stt] {context.node_id}: WAV split through ffmpeg with the audio filter "
                        f"window={config['chunk_duration_sec']}s overlap={config['chunk_overlap_sec']}s "
                        f"max={self._format_bytes(max_bytes)}.",
                    )
                    chunks = self._chunk_with_ffmpeg(
                        audio_path=audio_path,
                        work_dir=work_dir,
                        max_bytes=max_bytes,
                        chunk_duration_sec=int(config["chunk_duration_sec"]),
                        overlap_sec=int(config["chunk_overlap_sec"]),
                        audio_filter=audio_filter,
                    )
                else:
                    try:
                        self._emit_log(
                            context,
                            logs,
                            f"[openai-stt] {context.node_id}: WAV split "
                            f"window={config['chunk_duration_sec']}s overlap={config['chunk_overlap_sec']}s "
                            f"max={self._format_bytes(max_bytes)}.",
                        )
                        chunks = self._chunk_wav_by_time(
                            audio_path=audio_path,
                            work_dir=work_dir,
                            max_bytes=max_bytes,
                            chunk_duration_sec=int(config["chunk_duration_sec"]),
                            overlap_sec=int(config["chunk_overlap_sec"]),
                        )
                    except (wave.Error, EOFError, OSError):
                        self._emit_log(
                            context,
                            logs,
                            f"[openai-stt] {context.node_id}: native WAV split impossible, ffmpeg fallback.",
                        )
                        chunks = self._chunk_with_ffmpeg(
                            audio_path=audio_path,
                            work_dir=work_dir,
                            max_bytes=max_bytes,
                            chunk_duration_sec=int(config["chunk_duration_sec"]),
                            overlap_sec=int(config["chunk_overlap_sec"]),
                            audio_filter="",
                        )
            else:
                self._emit_log(
                    context,
                    logs,
                    f"[openai-stt] {context.node_id}: ffmpeg split "
                    f"window={config['chunk_duration_sec']}s overlap={config['chunk_overlap_sec']}s "
                    f"max={self._format_bytes(max_bytes)} "
                    f"audio_filter={'on' if use_audio_filter else 'off'}.",
                )
                chunks = self._chunk_with_ffmpeg(
                    audio_path=audio_path,
                    work_dir=work_dir,
                    max_bytes=max_bytes,
                    chunk_duration_sec=int(config["chunk_duration_sec"]),
                    overlap_sec=int(config["chunk_overlap_sec"]),
                    audio_filter=audio_filter if use_audio_filter else "",
                )
            if not chunks:
                raise OpenAiSttBlockError("audio split impossible: no segment produced.")
            self._emit_log(
                context,
                logs,
                f"[openai-stt] {context.node_id}: {len(chunks)} chunk(s) prepared "
                f"total_size={self._format_bytes(sum(chunk.path.stat().st_size for chunk in chunks))}.",
            )
            return chunks, temp_dir
        except Exception:
            temp_dir.cleanup()
            raise

    def _single_ffmpeg_segment_plan(
        self,
        *,
        audio_path: Path,
        config: dict[str, Any],
        context: BlockRuntimeContext,
        logs: list[str],
        max_bytes: int,
    ) -> tuple[list[AudioSegmentPlan], tempfile.TemporaryDirectory[str]]:
        """Provide internal OpenAiSttBlock behavior for `_single_ffmpeg_segment_plan`.

        Args:
            audio_path: Filesystem path handled by the block.
            config: Raw or normalized block configuration.
            context: Generic runtime context injected by the execution engine.
            logs: Logs value used by this block helper.
            max_bytes: Max bytes value used by this block helper.
        """
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise OpenAiSttBlockError("filtre audio requiert ffmpeg/ffprobe.")
        duration_seconds = self._probe_duration_seconds(audio_path)
        if duration_seconds <= 0 and audio_path.suffix.lower() == ".wav":
            try:
                duration_seconds = self._probe_wav_duration_seconds(audio_path)
            except (wave.Error, EOFError, OSError):
                duration_seconds = -1.0
        if duration_seconds <= 0:
            raise OpenAiSttBlockError("audio duration cannot be measured for the audio filter.")

        parent_dir = context.run_dir if context.run_dir else None
        if parent_dir is not None:
            parent_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.TemporaryDirectory(
            prefix=f"{context.node_id}-openai-stt-",
            dir=str(parent_dir) if parent_dir else None,
        )
        work_dir = Path(temp_dir.name)
        try:
            chunk_path = work_dir / f"{audio_path.stem}-filtered.mp3"
            extract_duration = self._write_ffmpeg_chunk(
                audio_path=audio_path,
                chunk_path=chunk_path,
                start=0.0,
                duration=duration_seconds,
                max_bytes=max_bytes,
                audio_filter=str(config.get("experimental_audio_filter") or DEFAULT_OPENAI_STT_AUDIO_FILTER),
            )
            if extract_duration + 0.5 < duration_seconds:
                raise OpenAiSttBlockError(
                    "the audio filter does not fit a single upload; enable chunking or reduce the window."
                )
            self._emit_log(
                context,
                logs,
                f"[openai-stt] {context.node_id}: filtered file "
                f"size={self._format_bytes(chunk_path.stat().st_size)} duration={self._format_duration(extract_duration)}.",
            )
            return [
                AudioSegmentPlan(
                    path=chunk_path,
                    index=1,
                    offset_sec=0.0,
                    ignore_before_sec=0.0,
                    nominal_start_sec=0.0,
                    nominal_duration_sec=extract_duration,
                    extract_start_sec=0.0,
                    extract_duration_sec=extract_duration,
                )
            ], temp_dir
        except Exception:
            temp_dir.cleanup()
            raise

    def _chunk_wav_by_time(
        self,
        *,
        audio_path: Path,
        work_dir: Path,
        max_bytes: int,
        chunk_duration_sec: int,
        overlap_sec: int,
    ) -> list[AudioSegmentPlan]:
        """Provide internal OpenAiSttBlock behavior for `_chunk_wav_by_time`.

        Args:
            audio_path: Filesystem path handled by the block.
            work_dir: Directory path used by the block runtime.
            max_bytes: Max bytes value used by this block helper.
            chunk_duration_sec: Duration in seconds.
            overlap_sec: Duration in seconds.
        """
        chunks: list[AudioSegmentPlan] = []
        with wave.open(str(audio_path), "rb") as source:
            params = source.getparams()
            frame_rate = max(1, params.framerate)
            total_frames = max(0, params.nframes)
            frame_width = max(1, params.nchannels * params.sampwidth)
            max_extract_frames = max(1, (max_bytes - 4096) // frame_width)
            overlap_frames = min(int(overlap_sec * frame_rate), max(0, max_extract_frames // 2))
            requested_nominal_frames = max(1, int(chunk_duration_sec * frame_rate))
            nominal_frames_per_chunk = max(1, min(requested_nominal_frames, max_extract_frames - overlap_frames))
            index = 1
            nominal_start_frame = 0
            while nominal_start_frame < total_frames:
                pre_roll_frames = 0 if index == 1 else min(overlap_frames, nominal_start_frame)
                nominal_frames = min(nominal_frames_per_chunk, total_frames - nominal_start_frame)
                extract_start_frame = max(0, nominal_start_frame - pre_roll_frames)
                extract_frames = pre_roll_frames + nominal_frames
                source.setpos(extract_start_frame)
                frames = source.readframes(extract_frames)
                chunk_path = work_dir / f"{audio_path.stem}-chunk-{index:04d}.wav"
                with wave.open(str(chunk_path), "wb") as target:
                    target.setparams(params)
                    target.writeframes(frames)
                chunks.append(
                    AudioSegmentPlan(
                        path=chunk_path,
                        index=index,
                        offset_sec=extract_start_frame / frame_rate,
                        ignore_before_sec=pre_roll_frames / frame_rate,
                        nominal_start_sec=nominal_start_frame / frame_rate,
                        nominal_duration_sec=nominal_frames / frame_rate,
                        extract_start_sec=extract_start_frame / frame_rate,
                        extract_duration_sec=extract_frames / frame_rate,
                    )
                )
                nominal_start_frame += nominal_frames
                index += 1
                if index > 10000:
                    raise OpenAiSttBlockError("audio split interrupted: too many segments.")
        return chunks

    def _chunk_with_ffmpeg(
        self,
        *,
        audio_path: Path,
        work_dir: Path,
        max_bytes: int,
        chunk_duration_sec: int,
        overlap_sec: int,
        audio_filter: str = "",
    ) -> list[AudioSegmentPlan]:
        """Provide internal OpenAiSttBlock behavior for `_chunk_with_ffmpeg`.

        Args:
            audio_path: Filesystem path handled by the block.
            work_dir: Directory path used by the block runtime.
            max_bytes: Max bytes value used by this block helper.
            chunk_duration_sec: Duration in seconds.
            overlap_sec: Duration in seconds.
            audio_filter: Audio filter value used by this block helper.
        """
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise OpenAiSttBlockError(
                "chunking is required for this file, but ffmpeg/ffprobe is unavailable."
            )
        duration_seconds = self._probe_duration_seconds(audio_path)
        if duration_seconds <= 0:
            raise OpenAiSttBlockError("audio duration cannot be measured with ffprobe.")

        input_size = max(1, audio_path.stat().st_size)
        size_bound_duration = max(1.0, math.floor((max_bytes * 0.90 / input_size) * duration_seconds))
        target_duration = min(float(chunk_duration_sec), size_bound_duration)
        chunks: list[AudioSegmentPlan] = []
        nominal_start = 0.0
        index = 1
        while nominal_start < duration_seconds - 0.001:
            remaining = duration_seconds - nominal_start
            nominal_duration = min(float(target_duration), remaining)
            pre_roll = 0.0 if index == 1 else min(float(overlap_sec), nominal_start)
            extract_start = max(0.0, nominal_start - pre_roll)
            requested_extract_duration = nominal_duration + pre_roll
            chunk_path = work_dir / f"{audio_path.stem}-chunk-{index:04d}{audio_path.suffix.lower() or '.mp3'}"
            extract_duration = self._write_ffmpeg_chunk(
                audio_path=audio_path,
                chunk_path=chunk_path,
                start=extract_start,
                duration=requested_extract_duration,
                max_bytes=max_bytes,
                audio_filter=audio_filter,
            )
            effective_nominal_duration = min(nominal_duration, max(0.001, extract_duration - pre_roll))
            chunks.append(
                AudioSegmentPlan(
                    path=chunk_path,
                    index=index,
                    offset_sec=extract_start,
                    ignore_before_sec=pre_roll,
                    nominal_start_sec=nominal_start,
                    nominal_duration_sec=effective_nominal_duration,
                    extract_start_sec=extract_start,
                    extract_duration_sec=extract_duration,
                )
            )
            nominal_start += max(0.001, effective_nominal_duration)
            index += 1
            if index > 10000:
                raise OpenAiSttBlockError("audio split interrupted: too many segments.")
        return chunks

    def _write_ffmpeg_chunk(
        self,
        *,
        audio_path: Path,
        chunk_path: Path,
        start: float,
        duration: float,
        max_bytes: int,
        audio_filter: str = "",
    ) -> float:
        """Provide internal OpenAiSttBlock behavior for `_write_ffmpeg_chunk`.

        Args:
            audio_path: Filesystem path handled by the block.
            chunk_path: Filesystem path handled by the block.
            start: Start value used by this block helper.
            duration: Duration value used by this block helper.
            max_bytes: Max bytes value used by this block helper.
            audio_filter: Audio filter value used by this block helper.
        """
        current_duration = max(1.0, duration)
        audio_filter = str(audio_filter or "").strip()
        stream_info = self._probe_audio_stream_info(audio_path) if audio_filter else {}
        output_options: list[str] = []
        if audio_filter:
            output_options.extend(["-af", audio_filter])
            sample_rate = stream_info.get("sample_rate")
            channels = stream_info.get("channels")
            bit_rate = stream_info.get("bit_rate")
            if sample_rate:
                output_options.extend(["-ar", str(sample_rate)])
            if channels:
                output_options.extend(["-ac", str(channels)])
            if audio_path.suffix.lower() in {".mp3", ".mpga", ".mpeg"} and bit_rate:
                output_options.extend(["-b:a", str(bit_rate)])
        else:
            output_options.extend(["-c", "copy"])
        while True:
            if chunk_path.exists():
                chunk_path.unlink()
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio_path),
                "-map",
                "0:a:0",
                "-ss",
                self._format_seconds(start),
                "-t",
                self._format_seconds(current_duration),
                "-vn",
                *output_options,
                str(chunk_path),
            ]
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=FFMPEG_PROCESS_TIMEOUT_SEC,
                )
            except subprocess.TimeoutExpired as exc:
                raise OpenAiSttBlockError("ffmpeg exceeded the timeout during the audio split.") from exc
            if result.returncode != 0 or not chunk_path.exists() or chunk_path.stat().st_size <= 0:
                details = (result.stderr or result.stdout or "").strip()
                raise OpenAiSttBlockError(f"ffmpeg failed during the audio split: {details[:300]}")
            if chunk_path.stat().st_size <= max_bytes:
                return current_duration
            if current_duration <= 1.0:
                raise OpenAiSttBlockError(
                    "one ffmpeg segment still exceeds the maximum allowed size."
                )
            current_duration = max(1.0, current_duration * 0.75)

    def _probe_wav_duration_seconds(self, audio_path: Path) -> float:
        """Provide internal OpenAiSttBlock behavior for `_probe_wav_duration_seconds`.

        Args:
            audio_path: Filesystem path handled by the block.
        """
        with wave.open(str(audio_path), "rb") as source:
            frame_rate = max(1, source.getframerate())
            return float(source.getnframes()) / frame_rate

    def _probe_duration_seconds(self, audio_path: Path) -> float:
        """Provide internal OpenAiSttBlock behavior for `_probe_duration_seconds`.

        Args:
            audio_path: Filesystem path handled by the block.
        """
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(audio_path),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=FFMPEG_PROCESS_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return -1.0
        if result.returncode != 0:
            return -1.0
        try:
            return float((result.stdout or "").strip())
        except ValueError:
            return -1.0

    def _probe_audio_stream_info(self, audio_path: Path) -> dict[str, int]:
        """Provide internal OpenAiSttBlock behavior for `_probe_audio_stream_info`.

        Args:
            audio_path: Filesystem path handled by the block.
        """
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,bit_rate",
            "-of",
            "json",
            str(audio_path),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=FFMPEG_PROCESS_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return {}
        if result.returncode != 0:
            return {}
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return {}
        streams = payload.get("streams") if isinstance(payload, dict) else None
        if not isinstance(streams, list) or not streams:
            return {}
        stream = streams[0] if isinstance(streams[0], dict) else {}
        info: dict[str, int] = {}
        for source_key, target_key in (
            ("sample_rate", "sample_rate"),
            ("channels", "channels"),
            ("bit_rate", "bit_rate"),
        ):
            try:
                value = int(stream.get(source_key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                info[target_key] = value
        return info

    def _format_seconds(self, seconds: float) -> str:
        """Format a value for logs, UI display, or runtime output.

        Args:
            seconds: Seconds value used by this block helper.
        """
        return f"{max(0.0, seconds):.3f}"

    def _prompt_for_chunk(self, *, config: dict[str, Any], previous_chunk_text: str) -> str:
        """Provide internal OpenAiSttBlock behavior for `_prompt_for_chunk`.

        Args:
            config: Raw or normalized block configuration.
            previous_chunk_text: Previous chunk text value used by this block helper.
        """
        base_prompt = str(config.get("prompt") or "").strip()
        if (
            not config.get("sliding_context_enabled")
            or not previous_chunk_text
            or str(config.get("model")) == OPENAI_STT_DIARIZE_MODEL
        ):
            return base_prompt
        context_text = self._sliding_context_text(previous_chunk_text)
        if not context_text:
            return base_prompt
        sliding_prompt = (
            "Context from the previous audio chunk, to be used only for continuity. "
            "Do not copy it if the content is not audible in the current chunk:\n"
            f"{context_text}"
        )
        return "\n\n".join(part for part in (base_prompt, sliding_prompt) if part)

    def _sliding_context_text(self, value: str) -> str:
        """Provide internal OpenAiSttBlock behavior for `_sliding_context_text`.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        text = " ".join(str(value or "").split()).strip()
        if not text:
            return ""
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        if len(sentences) >= 2:
            text = " ".join(sentences[-2:])
        return text[-1200:]

    def _transcribe(
        self,
        *,
        audio_path: Path,
        config: dict[str, Any],
        prompt_override: str | None = None,
    ) -> dict[str, Any]:
        """Provide internal OpenAiSttBlock behavior for `_transcribe`.

        Args:
            audio_path: Filesystem path handled by the block.
            config: Raw or normalized block configuration.
            prompt_override: Prompt override value used by this block helper.
        """
        api_key = str(config.get("api_key") or "").strip()
        if not api_key:
            raise OpenAiSttBlockError("api_key OpenAI manquante.")
        endpoint = self._transcription_endpoint(str(config["api_base_url"]))
        fields = self._request_fields(config, prompt_override=prompt_override)
        body, content_type = self._multipart_body(
            fields=fields,
            file_field="file",
            file_path=audio_path,
        )
        request = urlrequest.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": content_type,
                "Accept": "application/json, text/plain;q=0.9, */*;q=0.8",
                "User-Agent": "bloxsmith-openai-stt/1",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=int(config["timeout_sec"])) as response:
                status_code = int(getattr(response, "status", 200) or 200)
                payload = response.read()
                response_content_type = response.headers.get("Content-Type", "")
        except urlerror.HTTPError as exc:
            body_text = exc.read().decode("utf-8", "replace")
            raise OpenAiSttBlockError(
                f"OpenAI STT HTTP {exc.code}: {self._mask_secret(body_text, api_key)[:600]}"
            ) from exc
        except urlerror.URLError as exc:
            raise OpenAiSttBlockError(f"OpenAI STT inaccessible: {exc.reason}") from exc
        except OSError as exc:
            raise OpenAiSttBlockError(f"appel OpenAI STT impossible: {exc}") from exc

        text = payload.decode("utf-8", "replace")
        parsed = self._parse_response_payload(text, content_type=response_content_type)
        parsed.setdefault("status_code", status_code)
        return parsed

    def _request_fields(self, config: dict[str, Any], *, prompt_override: str | None = None) -> dict[str, str]:
        """Provide internal OpenAiSttBlock behavior for `_request_fields`.

        Args:
            config: Raw or normalized block configuration.
            prompt_override: Prompt override value used by this block helper.
        """
        model = str(config["model"])
        prompt = str(config.get("prompt") if prompt_override is None else prompt_override).strip()
        fields: dict[str, str] = {
            "model": model,
            "response_format": str(config["response_format"]),
        }
        if config.get("language"):
            fields["language"] = str(config["language"])
        if model == OPENAI_STT_DIARIZE_MODEL:
            if config.get("chunking_strategy") != "none":
                fields["chunking_strategy"] = str(config.get("chunking_strategy") or "auto")
        elif prompt:
            fields["prompt"] = prompt
        return fields

    def _transcription_endpoint(self, base_url: str) -> str:
        """Provide internal OpenAiSttBlock behavior for `_transcription_endpoint`.

        Args:
            base_url: Base url value used by this block helper.
        """
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            return f"{normalized}/audio/transcriptions"
        return f"{normalized}/v1/audio/transcriptions"

    def _multipart_body(
        self,
        *,
        fields: dict[str, str],
        file_field: str,
        file_path: Path,
    ) -> tuple[bytes, str]:
        """Provide internal OpenAiSttBlock behavior for `_multipart_body`.

        Args:
            fields: Fields value used by this block helper.
            file_field: File field value used by this block helper.
            file_path: Filesystem path handled by the block.
        """
        boundary = f"----bloxsmith-openai-stt-{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode("utf-8"),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                    str(value).encode("utf-8"),
                    b"\r\n",
                ]
            )
        filename = file_path.name or "audio"
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{file_field}"; '
                    f'filename="{filename}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                file_path.read_bytes(),
                b"\r\n",
                f"--{boundary}--\r\n".encode("utf-8"),
            ]
        )
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"

    def _parse_response_payload(self, text: str, *, content_type: str) -> dict[str, Any]:
        """Parse a raw value into the block internal representation.

        Args:
            text: Text value used by this block helper.
            content_type: Content type value used by this block helper.
        """
        if "json" not in str(content_type or "").lower():
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {"text": text}
            return parsed if isinstance(parsed, dict) else {"value": parsed, "text": text}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
        return parsed if isinstance(parsed, dict) else {"value": parsed, "text": text}

    def _transcript_from_response(self, response: dict[str, Any]) -> str:
        """Provide internal OpenAiSttBlock behavior for `_transcript_from_response`.

        Args:
            response: Response value used by this block helper.
        """
        for key in ("text", "transcript", "content"):
            value = response.get(key)
            if isinstance(value, str):
                return value
        segments = response.get("segments")
        if isinstance(segments, list):
            parts = [str(item.get("text") or "").strip() for item in segments if isinstance(item, dict)]
            joined = "\n".join(part for part in parts if part)
            if joined:
                return joined
        return json.dumps(response, ensure_ascii=False)

    def _merge_chunk_responses(self, *, responses: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
        """Provide internal OpenAiSttBlock behavior for `_merge_chunk_responses`.

        Args:
            responses: Responses value used by this block helper.
            config: Raw or normalized block configuration.
        """
        raw_parts: list[str] = []
        transcript_parts: list[str] = []
        timestamp_buckets: dict[int, list[str]] = {}

        for response in responses:
            chunk_text = self._transcript_from_response(response).strip()
            if chunk_text:
                raw_parts.append(chunk_text)

            used_segments = False
            chunk_transcript_parts: list[str] = []
            segments = response.get("segments")
            if isinstance(segments, list):
                for segment in segments:
                    if not isinstance(segment, dict):
                        continue
                    local_start = self._float_or_none(segment.get("start"))
                    if local_start is None:
                        continue
                    ignore_before = self._float_or_none(response.get("ignore_before_sec")) or 0.0
                    if ignore_before > 0 and local_start < ignore_before - OVERLAP_DROP_TOLERANCE_SECONDS:
                        continue
                    segment_text = self._normalize_transcript_text(segment.get("text"))
                    if not segment_text:
                        continue
                    chunk_transcript_parts.append(segment_text)
                    bucket = self._timestamp_bucket(local_start + float(response.get("offset_sec") or 0.0))
                    timestamp_buckets.setdefault(bucket, []).append(segment_text)
                    used_segments = True

            if not used_segments and chunk_text:
                chunk_transcript_parts.append(chunk_text)
                offset = float(response.get("offset_sec") or 0.0)
                ignore_before = float(response.get("ignore_before_sec") or 0.0)
                timestamp_buckets.setdefault(self._timestamp_bucket(offset + ignore_before), []).append(chunk_text)

            if chunk_transcript_parts:
                if transcript_parts:
                    transcript_parts.append(self._format_chunk_progress_marker(response))
                transcript_parts.extend(chunk_transcript_parts)

        transcript = "\n".join(transcript_parts).strip() or "\n".join(raw_parts).strip()
        raw_text = "\n".join(raw_parts).strip()
        return {
            "text": transcript,
            "raw_text": raw_text,
            "timestamped_text": self._format_timestamped_text(timestamp_buckets),
            "chunks": responses,
            "chunk_count": len(responses),
            "model": config["model"],
            "response_format": config["response_format"],
            "chunk_duration_sec": config["chunk_duration_sec"],
            "chunk_overlap_sec": config["chunk_overlap_sec"],
            "experimental_audio_filter_enabled": config["experimental_audio_filter_enabled"],
            "experimental_audio_filter": config["experimental_audio_filter"]
            if config["experimental_audio_filter_enabled"]
            else "",
            "sliding_context_enabled": config["sliding_context_enabled"],
        }

    def _float_or_none(self, value: Any) -> float | None:
        """Provide internal OpenAiSttBlock behavior for `_float_or_none`.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _normalize_transcript_text(self, value: Any) -> str:
        """Normalize a raw value into the format expected by the block.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        return " ".join(str(value or "").split()).strip()

    def _timestamp_bucket(self, seconds: float) -> int:
        """Provide internal OpenAiSttBlock behavior for `_timestamp_bucket`.

        Args:
            seconds: Seconds value used by this block helper.
        """
        return int(math.floor(max(0.0, seconds) / TIMESTAMP_BUCKET_SECONDS) * TIMESTAMP_BUCKET_SECONDS)

    def _format_chunk_progress_marker(self, response: dict[str, Any]) -> str:
        """Format a value for logs, UI display, or runtime output.

        Args:
            response: Response value used by this block helper.
        """
        marker_seconds = (
            self._float_or_none(response.get("nominal_start_sec"))
            or (
                (self._float_or_none(response.get("offset_sec")) or 0.0)
                + (self._float_or_none(response.get("ignore_before_sec")) or 0.0)
            )
        )
        total_seconds = int(round(max(0.0, marker_seconds)))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"---{minutes:02d}:{seconds:02d}"

    def _format_timestamped_text(self, buckets: dict[int, list[str]]) -> str:
        """Format a value for logs, UI display, or runtime output.

        Args:
            buckets: Buckets value used by this block helper.
        """
        lines: list[str] = []
        for bucket in sorted(buckets):
            text = " ".join(part for part in buckets[bucket] if part).strip()
            if not text:
                continue
            minutes = bucket // 60
            seconds = bucket % 60
            lines.append(f"[{minutes:02d}:{seconds:02d}] {text}")
        return "\n".join(lines)

    def _raw_json_from_response(self, response: dict[str, Any], *, transcript: str) -> str:
        """Provide internal OpenAiSttBlock behavior for `_raw_json_from_response`.

        Args:
            response: Response value used by this block helper.
            transcript: Transcript value used by this block helper.
        """
        payload = dict(response)
        payload.setdefault("text", transcript)
        return json.dumps(payload, ensure_ascii=False)

    def _runtime_outputs(
        self,
        context: BlockRuntimeContext,
        *,
        transcript: str,
        raw_json: str,
    ) -> list[BlockRuntimeOutput]:
        """Provide internal OpenAiSttBlock behavior for `_runtime_outputs`.

        Args:
            context: Generic runtime context injected by the execution engine.
            transcript: Transcript value used by this block helper.
            raw_json: Raw value received from configuration or runtime input.
        """
        outputs: list[BlockRuntimeOutput] = []
        for port in context.output_ports:
            port_id = int(getattr(port, "id", 0) or 0)
            port_name = str(getattr(port, "name", "") or "")
            normalized_name = port_name.strip().lower()
            if normalized_name in {"raw_json", "json", "segments", "diarized_json"} or port_id == 2:
                outputs.append(
                    BlockRuntimeOutput(
                        port_id=port_id,
                        port_name=port_name,
                        value=raw_json,
                        content_type=APPLICATION_JSON,
                    )
                )
            else:
                outputs.append(
                    BlockRuntimeOutput(
                        port_id=port_id,
                        port_name=port_name,
                        value=transcript,
                        content_type=TEXT_PLAIN,
                    )
                )
        return outputs

    def _emit_log(self, context: BlockRuntimeContext, logs: list[str], line: str) -> None:
        """Emit a runtime log or event through the injected context.

        Args:
            context: Generic runtime context injected by the execution engine.
            logs: Logs value used by this block helper.
            line: Line value used by this block helper.
        """
        live_logger = context.services.get("append_log")
        if callable(live_logger):
            live_logger(str(line))
            return
        logs.append(str(line))

    def _cancel_requested(self, context: BlockRuntimeContext) -> bool:
        """Provide internal OpenAiSttBlock behavior for `_cancel_requested`.

        Args:
            context: Generic runtime context injected by the execution engine.
        """
        callback = context.services.get("cancel_requested")
        return bool(callback()) if callable(callback) else False

    def _raise_if_cancelled(self, context: BlockRuntimeContext) -> None:
        """Check runtime state and raise or return when execution should stop.

        Args:
            context: Generic runtime context injected by the execution engine.
        """
        if self._cancel_requested(context):
            raise OpenAiSttBlockCancelled("cancellation requested")

    def _format_bytes(self, value: int | float) -> str:
        """Format a value for logs, UI display, or runtime output.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        size = max(0.0, float(value or 0))
        units = ("o", "Ko", "Mo", "Go")
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        return f"{size:.1f} {units[unit_index]}"

    def _format_duration(self, seconds: float | int) -> str:
        """Format a value for logs, UI display, or runtime output.

        Args:
            seconds: Seconds value used by this block helper.
        """
        total = max(0.0, float(seconds or 0))
        if total < 60:
            return f"{total:.1f}s"
        minutes = int(total // 60)
        remaining = total - (minutes * 60)
        return f"{minutes}m{remaining:04.1f}s"

    def _display_path(self, path: Path, root_dir: Path) -> str:
        """Provide internal OpenAiSttBlock behavior for `_display_path`.

        Args:
            path: Filesystem path handled by the block.
            root_dir: Directory path used by the block runtime.
        """
        try:
            return str(path.resolve().relative_to(root_dir.resolve()))
        except ValueError:
            return str(path)

    def _mask_secret(self, value: str, secret: str) -> str:
        """Provide internal OpenAiSttBlock behavior for `_mask_secret`.

        Args:
            value: Value to normalize, render, serialize, or process.
            secret: Secret value used by this block helper.
        """
        if not secret:
            return value
        return value.replace(secret, self._masked_secret(secret))

    def _masked_secret(self, secret: str) -> str:
        """Provide internal OpenAiSttBlock behavior for `_masked_secret`.

        Args:
            secret: Secret value used by this block helper.
        """
        if len(secret) <= 8:
            return "***"
        return f"{secret[:4]}...{secret[-4:]}"

    def _bool(self, value: Any) -> bool:
        """Normalize a raw boolean-like configuration value.

        Args:
            value: Value to normalize, render, serialize, or process.
        """
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
