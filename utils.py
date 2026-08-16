"""Shared utility functions for the RAG application."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"

GPT_54_MINI_PRICING_USD_PER_MILLION = {
    "input": 0.75,
    "cached_input": 0.075,
    "output": 4.50,
}


def calculate_rag_cost(
    response: Any,
    model: str | None = None,
) -> dict[str, Any]:
    """Calculate the cost of one OpenAI Responses API call.

    ``response`` can be the original OpenAI ``Response`` object, a parsed
    response dictionary, or a JSON string.  Pass the original SDK object when
    possible; its ``usage`` and ``model`` attributes are read directly.

    Responses API usage uses ``input_tokens`` and ``output_tokens``.  The
    Chat Completions aliases ``prompt_tokens`` and ``completion_tokens`` are
    accepted as well.  Cached input is charged at the discounted rate when
    ``input_tokens_details.cached_tokens`` is present.
    """

    payload = _parse_response(response)
    response_model = _read_field(payload, "model")
    model_name = model or response_model or DEFAULT_OPENAI_MODEL

    if not _is_supported_model(model_name):
        raise ValueError(f"Unsupported pricing model: {model_name}")

    usage = _read_field(payload, "usage")
    if usage is None:
        usage = payload
    if usage is None:
        raise ValueError("The response must contain a usage object.")

    input_tokens = _read_token_count(usage, "input_tokens", "prompt_tokens")
    output_tokens = _read_token_count(usage, "output_tokens", "completion_tokens")
    cached_input_tokens = _read_cached_input_tokens(usage)

    if cached_input_tokens > input_tokens:
        raise ValueError("cached input tokens cannot exceed total input tokens.")

    uncached_input_tokens = input_tokens - cached_input_tokens
    input_cost_usd = (
        uncached_input_tokens * GPT_54_MINI_PRICING_USD_PER_MILLION["input"]
        + cached_input_tokens * GPT_54_MINI_PRICING_USD_PER_MILLION["cached_input"]
    ) / 1_000_000
    output_cost_usd = (
        output_tokens * GPT_54_MINI_PRICING_USD_PER_MILLION["output"]
    ) / 1_000_000

    total_tokens = _read_field(usage, "total_tokens")
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    return {
        "model": model_name,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": round(input_cost_usd, 10),
        "output_cost_usd": round(output_cost_usd, 10),
        "total_cost_usd": round(input_cost_usd + output_cost_usd, 10),
    }


def _parse_response(response: Any) -> Any:
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "response must be an OpenAI Response object, a JSON object, or valid JSON."
            ) from exc

    if isinstance(response, Mapping):
        return response

    # OpenAI SDK response models expose model/usage as attributes.  Returning
    # the object preserves those attributes for _read_field below.
    if _read_field(response, "usage") is not None:
        return response

    for method_name in ("model_dump", "to_dict"):
        method = getattr(response, method_name, None)
        if callable(method):
            dumped = method()
            if isinstance(dumped, Mapping):
                return dumped

    raise TypeError("response must be an OpenAI Response object, JSON object, or JSON string.")


def _read_field(value: Any, field: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def _is_supported_model(model: str) -> bool:
    return model == DEFAULT_OPENAI_MODEL or model.startswith(f"{DEFAULT_OPENAI_MODEL}-")


def _read_token_count(usage: Any, *keys: str) -> int:
    for key in keys:
        value = _read_field(usage, key)
        if value is not None:
            try:
                token_count = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer.") from exc
            if token_count < 0:
                raise ValueError(f"{key} cannot be negative.")
            return token_count
    raise ValueError(f"Missing token usage field; expected one of: {', '.join(keys)}")


def _read_cached_input_tokens(usage: Any) -> int:
    direct_value = _read_field(usage, "cached_input_tokens")
    details = _read_field(usage, "input_tokens_details")
    if details is None:
        details = _read_field(usage, "prompt_tokens_details")
    if direct_value is None and details is not None:
        direct_value = _read_field(details, "cached_tokens")
    if direct_value is None:
        return 0

    try:
        cached_tokens = int(direct_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cached input tokens must be an integer.") from exc
    if cached_tokens < 0:
        raise ValueError("cached input tokens cannot be negative.")
    return cached_tokens
