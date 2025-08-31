import os
import io
import base64
import mimetypes
from typing import Optional

import anyio
from PIL import Image

try:
    from openai import AsyncOpenAI, RateLimitError  # type: ignore
except Exception:  # pragma: no cover - openai may not be installed
    AsyncOpenAI = None  # type: ignore

    class RateLimitError(Exception):
        """Fallback RateLimitError if openai is not installed."""
        pass

try:
    from google import genai  # type: ignore[reportMissingImports]
    from google.genai import types  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - google-genai may not be installed
    genai = None  # type: ignore
    types = None  # type: ignore


class ImageProvider:
    """Generate or edit images using different model providers."""

    def __init__(self, provider: str = "gemini") -> None:
        self.provider = provider.lower()
        self._openai_client: Optional[AsyncOpenAI] = None
        self._gemini_client: Optional["genai.Client"] = None

    async def edit_image(
        self,
        image_path: str,
        prompt: str,
        *,
        size: str = "1024x1536",
        quality: str = "high",
    ) -> Image.Image:
        """Return a PIL image generated from the given prompt and base image."""
        if self.provider == "gemini":
            print("Using Gemini")
            return await self._edit_with_gemini(image_path, prompt)
        return await self._edit_with_openai(image_path, prompt, size=size, quality=quality)

    async def _edit_with_openai(
        self,
        image_path: str,
        prompt: str,
        *,
        size: str,
        quality: str,
    ) -> Image.Image:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is not available")
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._openai_client = AsyncOpenAI(api_key=api_key)
        with open(image_path, "rb") as image_file:
            response = await self._openai_client.images.edit(
                model="gpt-image-1",
                image=image_file,
                prompt=prompt,
                quality=quality,
                size=size,
            )
        result_image_base64 = response.data[0].b64_json
        result_image_bytes = base64.b64decode(result_image_base64)
        return Image.open(io.BytesIO(result_image_bytes))

    async def _edit_with_gemini(self, image_path: str, prompt: str) -> Image.Image:
        if genai is None or types is None:
            raise RuntimeError("google-genai package is not available")
        if self._gemini_client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            self._gemini_client = genai.Client(api_key=api_key)

        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]

        config = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])

        def _run_generation() -> bytes:
            for chunk in self._gemini_client.models.generate_content_stream(
                model="gemini-2.5-flash-image-preview",
                contents=contents,
                config=config,
            ):
                if (
                    chunk.candidates
                    and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts
                ):
                    for part in chunk.candidates[0].content.parts:
                        inline = getattr(part, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            return inline.data
            raise RuntimeError("No image data returned from Gemini")

        image_data = await anyio.to_thread.run_sync(_run_generation)
        return Image.open(io.BytesIO(image_data))
