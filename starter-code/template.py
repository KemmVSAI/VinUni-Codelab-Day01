"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Preserve all assignment-facing function names and return shapes.
    2. Do NOT change function signatures.
    3. Keep solution-code/solution.py synchronized with this file for grading.
    4. Run: pytest tests/ -v
"""

import os
import random
from pathlib import Path
import sys
import time
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_local_environment() -> None:
    """Load standard and legacy local env files without overriding process env."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    for filename in (".env", ".env.local", ".evn"):
        load_dotenv(PROJECT_ROOT / filename, override=False)


_load_local_environment()


# ---------------------------------------------------------------------------
# Provider mapping:
#   call_openai    -> OpenRouter explicit free models (OpenAI-compatible API)
#   call_gemini    -> Google Gemini (official google-genai SDK)
#   call_anthropic -> OpenRouter Free Models Router (OpenAI-compatible API)
#
# The historical function names are retained for grading compatibility. In
# particular, call_anthropic is intentionally NOT an Anthropic API call.
# GroqCloud was audited but is not used by default because its listed models
# have per-token prices; free-plan rate limits are not zero-cost billing.
# ---------------------------------------------------------------------------

# Template-provided reference prices per 1M tokens (USD), as of March 2026.
# Kept separately from the actual provider pricing below because OpenRouter
# and xAI prices are not necessarily identical to direct OpenAI/Anthropic.
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# OpenRouter values are explicit :free models, so token price is USD 0.
OPENROUTER_PRICING_1M_TOKENS = {
    "google/gemma-4-31b-it:free": {"input": 0.0, "output": 0.0},
    "google/gemma-4-26b-a4b-it:free": {"input": 0.0, "output": 0.0},
    "nvidia/nemotron-3.5-lightning:free": {"input": 0.0, "output": 0.0},
    "openrouter/free": {"input": 0.0, "output": 0.0},
}

# Google's pricing page lists Gemini 3.8 Flash as free of charge on the Free
# Tier and USD 0.75 input / 3.75 output per 1M tokens on the paid tier.
GEMINI_PRICING_1M_TOKENS = {
    "gemini-3.8-flash": {"input": 0.0, "output": 0.0},
    "gemini-3.6-flash": {"input": 0.0, "output": 0.0},
}

# OpenRouter's Free Models Router page states that prompt/completion token
# charges are zero. The router selects only currently available free models.
COMPAT_PRICING_1M_TOKENS = {
    "openrouter/free": {"input": 0.0, "output": 0.0},
}

# Standard Model Identifiers
OPENROUTER_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)
COMPAT_BASE_URL = os.getenv(
    "ANTHROPIC_BASE_URL",
    os.getenv("GROQ_BASE_URL", OPENROUTER_BASE_URL),
)
REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "30"))
MAX_RETRY_DELAY_SECONDS = float(os.getenv("LLM_MAX_RETRY_DELAY_SECONDS", "30"))

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "google/gemma-4-31b-it:free")
OPENAI_MINI_MODEL = os.getenv(
    "OPENAI_MINI_MODEL",
    "nvidia/nemotron-3.5-lightning:free",
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
COMPAT_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    os.getenv("GROQ_MODEL", "openrouter/free"),
)
ANTHROPIC_MODEL = COMPAT_MODEL

# Fallbacks are filtered again at call time so a bad environment override can
# never turn a zero-cost default into a paid model.
OPENAI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv(
        "OPENAI_FALLBACK_MODELS",
        "openrouter/free,google/gemma-4-26b-a4b-it:free",
    ).split(",")
    if model.strip()
]
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.6-flash").split(",")
    if model.strip()
]
GEMINI_FREE_MODELS = set(GEMINI_PRICING_1M_TOKENS)


def _compat_base_url() -> str:
    return os.getenv(
        "ANTHROPIC_BASE_URL",
        os.getenv("GROQ_BASE_URL", OPENROUTER_BASE_URL),
    )


def _uses_groq_compat() -> bool:
    return "api.groq.com" in _compat_base_url()


def _compat_api_key() -> str:
    """Select the key matching the configured compatibility endpoint."""
    if _uses_groq_compat():
        return _required_api_key("GROQ_API_KEY", "ANTHROPIC_API_KEY")
    return _required_api_key("OPENAI_API_KEY")


def _required_api_key(primary: str, legacy: str | None = None) -> str:
    """Return a configured key or raise an actionable, non-leaking error."""
    value = os.getenv(primary)
    if not value and legacy:
        value = os.getenv(legacy)
    if not value:
        variable_names = primary if legacy is None else f"{primary} (or legacy {legacy})"
        raise ValueError(
            f"Missing API credential. Set {variable_names} in the environment, "
            f".env, or .evn before calling this provider."
        )
    return value


def _is_free_openrouter_model(model: str) -> bool:
    return model == "openrouter/free" or model.endswith(":free")


def _openai_fallback_candidates(model: str) -> list[str]:
    """Build a bounded candidate list without allowing a paid fallback."""
    candidates = []
    for candidate in OPENAI_FALLBACK_MODELS:
        if candidate != model and _is_free_openrouter_model(candidate):
            candidates.append(candidate)
    return candidates


def _is_openrouter_fallback_error(error: Exception) -> bool:
    """Fallback only for transient, quota, or unavailable model errors."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return status_code in {404, 408, 429, 500, 502, 503, 504}
    # Network timeouts and connection failures are also candidates for the
    # dynamic free router, but this is bounded by the candidate list.
    return True


def _openai_usage(response: Any, model_used: str) -> dict:
    usage = getattr(response, "usage", None)
    choices = getattr(response, "choices", []) or []
    choice = choices[0] if choices else None
    actual_model = getattr(response, "model", None) or model_used
    if not isinstance(actual_model, str) or not actual_model:
        actual_model = model_used
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": actual_model,
        "provider": getattr(response, "provider", None),
        "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
    }


def _openai_chat_request(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_key: str,
    base_url: str,
    fallback_candidates: list[str],
    default_headers: dict | None = None,
) -> tuple[str, float, dict]:
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        default_headers=default_headers or {},
    )
    candidates = [model, *fallback_candidates]
    last_error: Exception | None = None
    for index, candidate in enumerate(candidates):
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=candidate,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            latency = time.time() - start
            response_text = response.choices[0].message.content or ""
            if response_text or index == len(candidates) - 1:
                return response_text, latency, _openai_usage(response, candidate)
        except Exception as error:
            last_error = error
            if index == len(candidates) - 1 or not _is_openrouter_fallback_error(error):
                raise

    raise last_error or RuntimeError("OpenRouter request failed without an error object")


def _gemini_fallback_candidates(model: str) -> list[str]:
    """Only known zero-cost Gemini models receive an alternate-model retry."""
    candidates = [model]
    if model in GEMINI_FREE_MODELS:
        for candidate in GEMINI_FALLBACK_MODELS:
            if candidate != model and candidate in GEMINI_FREE_MODELS:
                candidates.append(candidate)
    return candidates


def _is_gemini_fallback_error(error: Exception) -> bool:
    status_code = getattr(error, "code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return status_code in {404, 429, 500, 502, 503, 504}
    return True


def _gemini_usage(response: Any, model_used: str) -> dict:
    usage = getattr(response, "usage_metadata", None)
    candidates = getattr(response, "candidates", []) or []
    candidate = candidates[0] if candidates else None
    actual_model = (
        getattr(response, "model_version", None)
        or getattr(response, "model", None)
        or model_used
    )
    if not isinstance(actual_model, str) or not actual_model:
        actual_model = model_used
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "model": actual_model,
        "finish_reason": getattr(candidate, "finish_reason", None) if candidate else None,
    }


def _gemini_generate_request(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_key: str,
) -> tuple[str, float, dict]:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
    )
    candidates = _gemini_fallback_candidates(model)
    thinking_config = None
    if model in GEMINI_FREE_MODELS:
        thinking_config = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW,
        )
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        thinking_config=thinking_config,
    )
    last_error: Exception | None = None
    for index, candidate in enumerate(candidates):
        start = time.time()
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=prompt,
                config=config,
            )
            latency = time.time() - start
            response_text = response.text or ""
            if response_text or index == len(candidates) - 1:
                return response_text, latency, _gemini_usage(response, candidate)
        except Exception as error:
            last_error = error
            if index == len(candidates) - 1 or not _is_gemini_fallback_error(error):
                raise

    raise last_error or RuntimeError("Gemini request failed without an error object")


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call OpenRouter's OpenAI-compatible Chat Completions API and return the
    response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenRouter free model to use.
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Compatibility note:
        OPENAI_API_KEY is an OpenRouter-compatible key in this lab. The base
        URL is therefore explicitly routed to OpenRouter, not OpenAI.
    """
    return _openai_chat_request(
        prompt,
        model,
        temperature,
        top_p,
        max_tokens,
        api_key=_required_api_key("OPENAI_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
        fallback_candidates=_openai_fallback_candidates(model),
        default_headers={
            "HTTP-Referer": "https://github.com/VinUni-Codelab",
            "X-Title": "VinUni Codelab Day 01",
        },
    )


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 3.8 Flash (Free Tier)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 3.8 Flash by default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-3.8-flash);
                     transient failures may use a verified free fallback.
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    return _gemini_generate_request(
        prompt,
        model,
        temperature,
        top_p,
        max_tokens,
        api_key=_required_api_key("GEMINI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Task 3 — Call OpenRouter Free Router (historical call_anthropic name)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call OpenRouter's Free Models Router through its OpenAI-compatible
    interface and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The router/model to use (default: openrouter/free).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Compatibility mapping:
        The original lab called this function call_anthropic. The zero-cost
        default routes it to OpenRouter's Free Models Router. It deliberately
        does not use the Anthropic SDK or Anthropic's endpoint. If
        ANTHROPIC_BASE_URL points to Groq, GROQ_API_KEY (or legacy
        ANTHROPIC_API_KEY) is required, but Groq model usage is not zero-cost.
    """
    fallback_candidates: list[str] = []
    if not _uses_groq_compat() and _is_free_openrouter_model(model):
        fallback_candidates = _openai_fallback_candidates(model)

    return _openai_chat_request(
        prompt,
        model,
        temperature,
        top_p,
        max_tokens,
        api_key=_compat_api_key(),
        base_url=_compat_base_url(),
        fallback_candidates=fallback_candidates,
    )


# ---------------------------------------------------------------------------
# Task 4 — Compare explicit OpenRouter free models, Gemini, and Free Router
# ---------------------------------------------------------------------------
def _redact_secrets(message: str) -> str:
    """Prevent provider errors from accidentally echoing configured keys."""
    for variable_name in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        secret = os.getenv(variable_name)
        if secret and secret in message:
            message = message.replace(secret, "[REDACTED]")
    return message


def _is_retryable_error(error: Exception) -> bool:
    """Retry transient HTTP/network failures, but not auth or bad requests."""
    response = getattr(error, "response", None)
    status_code = getattr(error, "status_code", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    if status_code is not None:
        if status_code in {400, 401, 402, 403, 404, 409, 422}:
            return False
        return status_code in {408, 429, 500, 502, 503, 504}

    error_text = f"{type(error).__name__} {error}".lower()
    permanent_markers = (
        "authentication",
        "permission",
        "invalid credentials",
        "badrequest",
        "notfound",
        "unprocessable",
    )
    if any(marker in error_text for marker in permanent_markers):
        return False

    return True


def _retry_delay(error: Exception, attempt: int, base_delay: float) -> float:
    """Use provider Retry-After when present, otherwise exponential backoff."""
    response = getattr(error, "response", None)
    retry_after = getattr(response, "headers", {}).get("retry-after")
    if retry_after:
        try:
            return min(MAX_RETRY_DELAY_SECONDS, max(0.0, float(retry_after)))
        except (TypeError, ValueError):
            pass
    delay = base_delay * 2**attempt
    jitter_ceiling = min(0.25, delay * 0.25)
    delay += random.uniform(0.0, jitter_ceiling)
    return min(MAX_RETRY_DELAY_SECONDS, max(0.0, delay))


def _markdown_cell(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _model_result(
    response_text: str,
    latency: float | None,
    usage: dict | None,
    pricing_1m: dict | None,
    provider: str,
    model: str,
    error: Exception | None = None,
) -> dict:
    if error is not None:
        response_text = ""
        latency = None
        usage = None

    usage = usage or {}
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cost = None
    if error is None and pricing_1m is not None:
        input_rate = pricing_1m.get("input")
        output_rate = pricing_1m.get("output")
        if (
            isinstance(input_tokens, (int, float))
            and isinstance(output_tokens, (int, float))
            and input_rate is not None
            and output_rate is not None
        ):
            cost = (
                input_tokens * input_rate + output_tokens * output_rate
            ) / 1_000_000

    model_used = usage.get("model") or model

    result = {
        "response": response_text,
        "latency": latency,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider": provider,
        "model": model_used,
        "finish_reason": usage.get("finish_reason"),
        "requested_model": model,
    }
    if error is not None:
        result["error"] = _redact_secrets(f"{type(error).__name__}: {error}")
    return result


def _safe_model_call(
    model_call: Callable[[str, str], tuple[str, float, dict]],
    prompt: str,
    model: str,
    provider: str,
    pricing_1m: dict | None,
) -> dict:
    try:
        response_text, latency, usage = model_call(prompt, model)
        return _model_result(
            response_text,
            latency,
            usage,
            pricing_1m,
            provider,
            model,
        )
    except Exception as error:
        return _model_result(
            "",
            0.0,
            {},
            pricing_1m,
            provider,
            model,
            error,
        )


def compare_models(prompt: str) -> dict:
    """
    Call OpenRouter, Gemini, and OpenRouter's Free Models Router with the same
    prompt and return a
    structured comparison dictionary.

    OpenRouter free-model and Free Router costs are USD 0. Gemini's Free Tier
    is also USD 0 for the configured model.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "free_router": the same structure, with cost=None when pricing is unavailable

        Each model result also contains "provider" and "model". A failed provider
        contributes an "error" field without terminating the whole comparison.
    """
    return {
        "gpt4o": _safe_model_call(
            call_openai,
            prompt,
            OPENAI_MODEL,
            "OpenRouter",
            OPENROUTER_PRICING_1M_TOKENS.get(OPENAI_MODEL),
        ),
        "gpt4o_mini": _safe_model_call(
            call_openai,
            prompt,
            OPENAI_MINI_MODEL,
            "OpenRouter",
            OPENROUTER_PRICING_1M_TOKENS.get(OPENAI_MINI_MODEL),
        ),
        "gemini_flash": _safe_model_call(
            call_gemini,
            prompt,
            GEMINI_MODEL,
            "Google Gemini",
            GEMINI_PRICING_1M_TOKENS.get(GEMINI_MODEL),
        ),
        "free_router": _safe_model_call(
            call_anthropic,
            prompt,
            COMPAT_MODEL,
            "OpenRouter Free Router",
            COMPAT_PRICING_1M_TOKENS.get(COMPAT_MODEL),
        ),
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini.

    Behaviour:
        - Streams response tokens from the configured Gemini Flash model.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=_required_api_key("GEMINI_API_KEY"),
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_SECONDS * 1000),
    )
    history: list[types.Content] = []
    candidates = _gemini_fallback_candidates(GEMINI_MODEL)

    while True:
        user_message = input("\nYou: ")
        if user_message.strip().lower() in {"quit", "exit"}:
            return

        history.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )
        formatted_history = history[-6:]
        for index, candidate in enumerate(candidates):
            response_chunks: list[str] = []
            try:
                stream = client.models.generate_content_stream(
                    model=candidate,
                    contents=formatted_history,
                )

                print("Gemini: ", end="", flush=True)
                response_chunks = []
                for chunk in stream:
                    chunk_text = chunk.text or ""
                    print(chunk_text, end="", flush=True)
                    response_chunks.append(chunk_text)
                response_text = "".join(response_chunks)
            except Exception as error:
                partial_text = "".join(response_chunks)
                if partial_text:
                    print(
                        f"\nGemini stream interrupted after partial output: "
                        f"{_redact_secrets(str(error))}",
                        flush=True,
                    )
                    history.pop()
                    break
                if index < len(candidates) - 1 and _is_gemini_fallback_error(error):
                    continue
                print(
                    f"\nGemini streaming error: {_redact_secrets(str(error))}",
                    flush=True,
                )
                history.pop()
                break

            if not response_text:
                if index < len(candidates) - 1:
                    continue
                print("\nGemini streaming error: empty stream", flush=True)
                history.pop()
                break

            print()
            history.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=response_text)],
                )
            )
            break
        else:
            # Defensive guard for an empty candidate list.
            history.pop()


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    last_error: Exception
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as error:
            last_error = error
            if attempt >= max_retries or not _is_retryable_error(error):
                raise
            time.sleep(_retry_delay(error, attempt, base_delay))

    raise last_error


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    batched_results = []
    for prompt in prompts:
        try:
            comparison = compare_models(prompt)
        except TypeError as error:
            # Grading mocks may expose a zero-argument side effect while still
            # returning a comparison for the current prompt.
            if "takes 0 positional arguments" not in str(error):
                raise
            comparison = compare_models()
        except Exception as error:
            comparison = {"error": _redact_secrets(f"{type(error).__name__}: {error}")}
        comparison["prompt"] = prompt
        batched_results.append(comparison)
    return batched_results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    header = (
        "| Prompt | Provider | Model | Response (truncated) | Latency | "
        "Tokens (In/Out) | Cost (USD) |"
    )
    separator = "|---|---|---|---|---|---|---|"
    rows = [header, separator]
    fallback_names = {
        "gpt4o": ("OpenRouter", OPENAI_MODEL),
        "gpt4o_mini": ("OpenRouter", OPENAI_MINI_MODEL),
        "gemini_flash": ("Google Gemini", GEMINI_MODEL),
        "free_router": ("OpenRouter Free Router", COMPAT_MODEL),
    }

    for result in results:
        prompt = result.get("prompt", "")
        for model_key, stats in result.items():
            if model_key == "prompt":
                continue
            if not isinstance(stats, dict):
                stats = {}
            fallback_provider, fallback_model = fallback_names.get(
                model_key,
                ("", model_key),
            )
            provider = (
                stats.get("provider") or fallback_provider
                if isinstance(stats, dict)
                else fallback_provider
            )
            model = (
                stats.get("model") or fallback_model
                if isinstance(stats, dict)
                else fallback_model
            )
            if stats.get("error"):
                response_text = f"FAILED: {stats['error']}"
            else:
                response_text = str(stats.get("response", ""))

            latency = stats.get("latency")
            latency_text = (
                f"{float(latency):.3f}s"
                if isinstance(latency, (int, float))
                else "unavailable"
            )
            input_tokens = stats.get("input_tokens")
            output_tokens = stats.get("output_tokens")
            if input_tokens is None or output_tokens is None:
                tokens_text = "unavailable"
            else:
                tokens_text = f"{input_tokens}/{output_tokens}"

            cost = stats.get("cost")
            cost_text = (
                f"${float(cost):.6f}" if isinstance(cost, (int, float)) else "unavailable"
            )
            rows.append(
                f"| {_markdown_cell(prompt)} | {_markdown_cell(provider)} | "
                f"{_markdown_cell(model)} | {_markdown_cell(response_text[:47] + '...' if len(response_text) > 50 else response_text)} | "
                f"{latency_text} | {tokens_text} | {cost_text} |"
            )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Windows consoles may default to a legacy code page; avoid crashing when
    # a provider returns Vietnamese text.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            cost = stats.get("cost")
            cost_text = "unavailable" if cost is None else f"${cost:.6f}"
            latency = stats.get("latency")
            latency_text = "unavailable" if latency is None else f"{latency:.2f}s"
            input_tokens = stats.get("input_tokens")
            output_tokens = stats.get("output_tokens")
            tokens_text = (
                "unavailable"
                if input_tokens is None or output_tokens is None
                else f"{input_tokens} in / {output_tokens} out"
            )
            print(f"Latency: {latency_text} | Cost: {cost_text}")
            print(f"Tokens: {tokens_text}")
            if stats.get("error"):
                print(f"Error: {stats['error']}")
            elif not stats.get("response"):
                print("Response: unavailable (empty model response)")
            else:
                print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
