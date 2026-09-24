import { withProperties } from "./properties.js";

/**
 * Role: Mounts the OpenAI STT block inspector panel frontend.
 * File Name: inspector_panel.js
 * Author: Alexandre EL
 * Email: alex@hackinvent.com
 * Created Date: 2024-11-12
 */
/**
 * Bind OpenAI STT settings for the inspector panel.
 *
 * @param {HTMLElement} root - Mounted inspector root.
 * @param {object} api - Generic block UI API exposing block actions.
 * @param {object} options - Action name used by the current surface.
 */
function mountOpenAiSttEditor(root, api, { actionName = "inspector_update_openai_stt" } = {}) {
    /**
     * Query one OpenAI STT control inside the mounted surface.
     *
     * @param {string} selector - CSS selector for the desired control.
     * @returns {Element|null} Matching control or null.
     */
    const read = (selector) => root.querySelector(selector);
    const model = read("[data-openai-stt-model]");
    const apiKey = read("[data-openai-stt-api-key]");
    const clearApiKey = read("[data-openai-stt-clear-api-key]");
    const audioPath = read("[data-openai-stt-audio-path]");
    const language = read("[data-openai-stt-language]");
    const prompt = read("[data-openai-stt-prompt]");
    const responseFormat = read("[data-openai-stt-response-format]");
    const chunkingEnabled = read("[data-openai-stt-chunking-enabled]");
    const chunkSizeMb = read("[data-openai-stt-chunk-size-mb]");
    const chunkDurationSec = read("[data-openai-stt-chunk-duration-sec]");
    const chunkOverlapSec = read("[data-openai-stt-chunk-overlap-sec]");
    const chunkingStrategy = read("[data-openai-stt-chunking-strategy]");
    const experimentalAudioFilterEnabled = read("[data-openai-stt-experimental-audio-filter-enabled]");
    const experimentalAudioFilter = read("[data-openai-stt-experimental-audio-filter]");
    const slidingContextEnabled = read("[data-openai-stt-sliding-context-enabled]");
    const timeoutSec = read("[data-openai-stt-timeout-sec]");
    const apiBaseUrl = read("[data-openai-stt-api-base-url]");
    const applyButton = read("[data-openai-stt-apply]") || read("[data-block-apply]");
    let dirty = false;

    /**
     * Collect OpenAI STT form values into the block action payload.
     *
     * @returns {object} Current block configuration values.
     */
    const values = () => ({
      model: model?.value || "",
      api_key: apiKey?.value || "",
      clear_api_key: Boolean(clearApiKey?.checked),
      audio_path: audioPath?.value || "",
      language: language?.value || "",
      prompt: prompt?.value || "",
      response_format: responseFormat?.value || "",
      chunking_enabled: Boolean(chunkingEnabled?.checked),
      chunk_size_mb: chunkSizeMb?.value || "",
      chunk_duration_sec: chunkDurationSec?.value || "",
      chunk_overlap_sec: chunkOverlapSec?.value || "",
      chunking_strategy: chunkingStrategy?.value || "",
      experimental_audio_filter_enabled: Boolean(experimentalAudioFilterEnabled?.checked),
      experimental_audio_filter: experimentalAudioFilter?.value || "",
      sliding_context_enabled: Boolean(slidingContextEnabled?.checked),
      timeout_sec: timeoutSec?.value || "",
      api_base_url: apiBaseUrl?.value || "",
    });

    /**
     * Mark OpenAI STT settings as changed and enable Apply.
     */
    const markDirty = () => {
      dirty = true;
      if (applyButton) {
        applyButton.disabled = false;
      }
    };

    /**
     * Persist the current OpenAI STT settings through the block action contract.
     */
    const apply = () => {
      if (!dirty) {
        return;
      }
      void api.applyAction(actionName, values()).then(() => {
        dirty = false;
        if (applyButton) {
          applyButton.disabled = true;
        }
      }).catch((error) => {
        api.log?.(`[error] Mise a jour OpenAI STT impossible: ${error.message}`);
      });
    };

    [model, clearApiKey, responseFormat, chunkingEnabled, chunkingStrategy, experimentalAudioFilterEnabled, slidingContextEnabled].forEach(
      (element) => element?.addEventListener("change", markDirty),
    );
    [apiKey, audioPath, language, prompt, chunkSizeMb, chunkDurationSec, chunkOverlapSec, experimentalAudioFilter, timeoutSec, apiBaseUrl].forEach((element) => {
      element?.addEventListener("input", markDirty);
      element?.addEventListener("change", markDirty);
      element?.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && element.tagName !== "TEXTAREA") {
          event.preventDefault();
          apply();
        }
      });
    });
    applyButton?.addEventListener("click", apply);
}

/**
 * Mount the OpenAI STT inspector panel bindings.
 *
 * @param {HTMLElement} root - Mounted OpenAI STT inspector root.
 * @param {object} api - Generic block UI API exposing block actions.
 */
function mountOwned(root, api) {
  mountOpenAiSttEditor(root, api, { actionName: "inspector_update_openai_stt" });
}

/** Keep the block behavior and add properties-only accessibility. */
export function mount(root, ...args) {
  return withProperties(mountOwned).call(this, root, ...args);
}
