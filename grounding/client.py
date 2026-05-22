"""Gemini API client wrapper. Reads GEMINI_API_KEY from env (or .env)."""

import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class GeminiClient:
    """One-shot Gemini JSON generation.

    Uses response_mime_type='application/json' rather than response_schema
    so the CoT variant can include a free-form `reasoning` field.
    """

    def __init__(
        self,
        # model: str = "gemini-2.5-flash",
        model: str = "gemini-3.1-flash-lite",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
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

    def generate(self, prompt: str, json_mode: bool = True) -> dict:
        from google.genai import types

        config_kwargs = {"temperature": self.temperature}
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        config = types.GenerateContentConfig(**config_kwargs)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

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
