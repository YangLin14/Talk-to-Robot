"""Gemini API client wrapper. Reads GEMINI_API_KEY from env (or .env)."""

import os
import time
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Treat these substrings as retryable (rate limit / transient server).
_RETRYABLE_HINTS = (
    "429",
    "resource_exhausted",
    "rate limit",
    "rate_limit",
    "quota",
    "503",
    "unavailable",
    "deadline",
    "timeout",
    "internal error",
    "500",
)


def _is_retryable(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(h in msg for h in _RETRYABLE_HINTS)


class GeminiClient:
    """One-shot Gemini JSON generation.

    Handles two things the harness used to do by hand:
      - request spacing so the free-tier rate limit (default 5 req/min ->
        12 s) is not exceeded
      - bounded retry with exponential backoff on transient errors
    """

    def __init__(
        self,
        # model: str = "gemini-2.5-flash",
        model: str = "gemini-3.1-flash-lite",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        min_interval_s: float = 12.0,
        max_retries: int = 2,
        backoff_base_s: float = 4.0,
    ):
        from google import genai
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in .env at the project root "
                "or export it in the shell."
            )
        self._genai = genai
        self.client = genai.Client(api_key=key)
        self.model = model
        self.temperature = temperature
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._last_call_t: Optional[float] = None

    def _wait_for_slot(self) -> None:
        if self._last_call_t is None or self.min_interval_s <= 0:
            return
        elapsed = time.monotonic() - self._last_call_t
        remaining = self.min_interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def generate(self, prompt: str, json_mode: bool = True) -> dict:
        from google.genai import types

        config_kwargs = {"temperature": self.temperature}
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        config = types.GenerateContentConfig(**config_kwargs)

        attempt = 0
        last_exc: Optional[BaseException] = None
        while attempt <= self.max_retries:
            self._wait_for_slot()
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                self._last_call_t = time.monotonic()
                break
            except Exception as exc:  # noqa: BLE001
                self._last_call_t = time.monotonic()
                last_exc = exc
                if attempt == self.max_retries or not _is_retryable(exc):
                    raise
                time.sleep(self.backoff_base_s * (2 ** attempt))
                attempt += 1
        else:  # pragma: no cover -- loop always breaks or raises
            assert last_exc is not None
            raise last_exc

        usage = None
        if getattr(response, "usage_metadata", None) is not None:
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", None),
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", None),
                "total_tokens": getattr(response.usage_metadata, "total_token_count", None),
            }

        finish_reason = None
        try:
            finish_reason = str(response.candidates[0].finish_reason)
        except (AttributeError, IndexError, TypeError):
            pass

        return {
            "text": response.text or "",
            "model": self.model,
            "finish_reason": finish_reason,
            "usage": usage,
        }
