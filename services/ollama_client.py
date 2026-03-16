from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.request
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover - fallback for minimal local env
    httpx = None


logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://218.4.33.190:26316")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120"))


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


async def _post_generate(payload: dict) -> str:
    url = f"{OLLAMA_BASE_URL}/api/generate"
    if httpx is not None:
        timeout = httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    else:
        data = await asyncio.to_thread(_post_generate_via_urllib, url, payload)

    result = data.get("response", "")
    if not result:
        raise RuntimeError("Ollama 返回为空")
    return result.strip()


def _post_generate_via_urllib(url: str, payload: dict) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


async def generate_json(system_prompt: str, user_prompt: str, schema: dict, model: str | None = None) -> dict:
    actual_model = model or OLLAMA_DEFAULT_MODEL
    payload = {
        "model": actual_model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0.1},
    }
    logger.info("Calling Ollama JSON model=%s", actual_model)
    try:
        raw = await _post_generate(payload)
        return _extract_json(raw)
    except Exception as exc:
        logger.exception("Ollama JSON generation failed: %s", exc)
        raise RuntimeError(f"Ollama JSON generation failed: {exc}") from exc


async def generate_text(system_prompt: str, user_prompt: str, model: str | None = None) -> str:
    actual_model = model or OLLAMA_DEFAULT_MODEL
    payload = {
        "model": actual_model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    logger.info("Calling Ollama text model=%s", actual_model)
    try:
        return await _post_generate(payload)
    except Exception as exc:
        logger.exception("Ollama text generation failed: %s", exc)
        raise RuntimeError(f"Ollama text generation failed: {exc}") from exc
