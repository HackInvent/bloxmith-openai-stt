/**
 * Role: Mounts the OpenAI STT block modal frontend.
 * File Name: block_modal.js
 * Author: Alexandre EL
 * Email: alex@hackinvent.com
 * Created Date: 2026-06-10
 */
/**
 * Collect the editable OpenAI STT modal fields into the block action payload.
 *
 * @param {HTMLElement} root - Mounted OpenAI STT modal root.
 * @returns {object} Current modal field values.
 */
function values(root) {
  const read = (selector) => root.querySelector(selector);
  return {
    model: read("[data-openai-stt-model]")?.value || "",
    api_key: read("[data-openai-stt-api-key]")?.value || "",
    clear_api_key: Boolean(read("[data-openai-stt-clear-api-key]")?.checked),
    audio_path: read("[data-openai-stt-audio-path]")?.value || "",
    language: read("[data-openai-stt-language]")?.value || "",
    prompt: read("[data-openai-stt-prompt]")?.value || "",
    response_format: read("[data-openai-stt-response-format]")?.value || "",
    chunking_enabled: Boolean(read("[data-openai-stt-chunking-enabled]")?.checked),
    chunk_size_mb: read("[data-openai-stt-chunk-size-mb]")?.value || "",
    chunk_duration_sec: read("[data-openai-stt-chunk-duration-sec]")?.value || "",
    chunk_overlap_sec: read("[data-openai-stt-chunk-overlap-sec]")?.value || "",
    chunking_strategy: read("[data-openai-stt-chunking-strategy]")?.value || "",
    experimental_audio_filter_enabled: Boolean(read("[data-openai-stt-experimental-audio-filter-enabled]")?.checked),
    experimental_audio_filter: read("[data-openai-stt-experimental-audio-filter]")?.value || "",
    sliding_context_enabled: Boolean(read("[data-openai-stt-sliding-context-enabled]")?.checked),
    timeout_sec: read("[data-openai-stt-timeout-sec]")?.value || "",
    api_base_url: read("[data-openai-stt-api-base-url]")?.value || "",
  };
}

/**
 * Bind OpenAI STT modal fields to the explicit Apply workflow.
 *
 * @param {HTMLElement} root - Mounted OpenAI STT modal root.
 * @param {object} api - Generic block UI API exposing applyAction/log.
 */
function mountFields(root, api) {
  const applyButton = root.querySelector("[data-openai-stt-apply]") || root.querySelector("[data-block-apply]");
  let dirty = false;

  function markDirty() {
    dirty = true;
    if (applyButton) {
      applyButton.disabled = false;
    }
  }

  function apply() {
    if (!dirty) {
      return;
    }
    void api.applyAction("modal_update_openai_stt", values(root)).then(() => {
      dirty = false;
      if (applyButton) {
        applyButton.disabled = true;
      }
    }).catch((error) => {
      api.log?.(`[error] Mise a jour OpenAI STT impossible: ${error.message}`);
    });
  }

  root.querySelectorAll("[data-openai-stt-model], [data-openai-stt-clear-api-key], [data-openai-stt-response-format], [data-openai-stt-chunking-enabled], [data-openai-stt-chunking-strategy], [data-openai-stt-experimental-audio-filter-enabled], [data-openai-stt-sliding-context-enabled]").forEach((element) => {
    element.addEventListener("change", markDirty);
  });
  root.querySelectorAll("[data-openai-stt-api-key], [data-openai-stt-audio-path], [data-openai-stt-language], [data-openai-stt-prompt], [data-openai-stt-chunk-size-mb], [data-openai-stt-chunk-duration-sec], [data-openai-stt-chunk-overlap-sec], [data-openai-stt-experimental-audio-filter], [data-openai-stt-timeout-sec], [data-openai-stt-api-base-url]").forEach((element) => {
    element.addEventListener("input", markDirty);
    element.addEventListener("change", markDirty);
    element.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && element.tagName !== "TEXTAREA") {
        event.preventDefault();
        apply();
      }
    });
  });
  applyButton?.addEventListener("click", apply);
}

function tabs(root) {
  return Array.from(root.querySelectorAll("[data-openai-stt-modal-tab]"));
}

function panels(root) {
  return Array.from(root.querySelectorAll("[data-openai-stt-modal-panel]"));
}

/**
 * Activate one OpenAI STT modal tab and preserve draft values in other panels.
 *
 * @param {HTMLElement} root - Mounted modal root.
 * @param {HTMLElement} tab - Tab to activate.
 * @param {object} options - Focus behavior.
 */
function activateTab(root, tab, { focus = false } = {}) {
  if (!(tab instanceof HTMLElement)) {
    return;
  }
  const tabId = String(tab.dataset.openaiSttTabId || "");
  for (const candidate of tabs(root)) {
    const selected = candidate === tab;
    candidate.setAttribute("aria-selected", selected ? "true" : "false");
    candidate.tabIndex = selected ? 0 : -1;
  }
  for (const panel of panels(root)) {
    panel.hidden = String(panel.dataset.openaiSttTabId || "") !== tabId;
  }
  if (focus) {
    tab.focus();
  }
}

function moveTab(root, current, direction) {
  const tabList = tabs(root);
  const index = tabList.indexOf(current);
  if (index < 0 || !tabList.length) {
    return;
  }
  activateTab(root, tabList[(index + direction + tabList.length) % tabList.length], { focus: true });
}

/**
 * Bind OpenAI STT modal tab controls.
 *
 * @param {HTMLElement} root - Mounted modal root.
 */
function mountTabs(root) {
  activateTab(root, root.querySelector('[data-openai-stt-modal-tab][aria-selected="true"]') || root.querySelector("[data-openai-stt-modal-tab]"));
  root.addEventListener("click", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const tab = target?.closest("[data-openai-stt-modal-tab]");
    if (!tab || !root.contains(tab)) {
      return;
    }
    event.preventDefault();
    activateTab(root, tab, { focus: true });
  });
  root.addEventListener("keydown", (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const tab = target?.closest("[data-openai-stt-modal-tab]");
    if (!tab || !root.contains(tab)) {
      return;
    }
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault();
      moveTab(root, tab, 1);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault();
      moveTab(root, tab, -1);
    } else if (event.key === "Home") {
      event.preventDefault();
      activateTab(root, tabs(root)[0], { focus: true });
    } else if (event.key === "End") {
      event.preventDefault();
      const tabList = tabs(root);
      activateTab(root, tabList[tabList.length - 1], { focus: true });
    }
  });
}

/**
 * Mount OpenAI STT modal fields and tabs.
 *
 * @param {HTMLElement} root - Mounted OpenAI STT modal root.
 * @param {object} api - Generic block UI API.
 */
export function mount(root, api = {}) {
  mountFields(root, api);
  mountTabs(root);
}
