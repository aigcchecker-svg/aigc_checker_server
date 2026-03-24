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
OLLAMA_JSON_NUM_CTX = int(os.getenv("OLLAMA_JSON_NUM_CTX", "1536"))
OLLAMA_JSON_NUM_PREDICT = int(os.getenv("OLLAMA_JSON_NUM_PREDICT", "96"))


def _extract_json(text: str) -> dict[str, Any]:
    """从 Ollama 响应字符串中提取 JSON，直接解析失败时用正则匹配第一个 {...} 块。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Ollama 偶尔在 JSON 前后附加思考过程文字，用正则兜底
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _extract_generate_text(data: dict[str, Any]) -> str:
    """从 Ollama 返回体中提取主文本，兼容 response/message/content 等不同字段。"""
    candidates = [
        data.get("response"),
        data.get("message", {}).get("content") if isinstance(data.get("message"), dict) else None,
        data.get("content"),
        data.get("output_text"),
        data.get("text"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _summarize_empty_response(data: dict[str, Any]) -> str:
    """生成空响应的诊断摘要，便于定位 Ollama 兼容性问题。"""
    return (
        f"keys={sorted(data.keys())} "
        f"done={data.get('done')} "
        f"done_reason={data.get('done_reason')} "
        f"response_len={len(str(data.get('response') or ''))} "
        f"thinking_len={len(str(data.get('thinking') or ''))}"
    )


async def _post_generate(payload: dict) -> str:
    """向 Ollama /api/generate 发送请求，返回响应文本。

    优先使用 httpx 异步客户端（推荐）；若未安装则通过线程池调用 urllib 同步版本。
    响应为空时抛出 RuntimeError，防止后续 JSON 解析静默失败。
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    if httpx is not None:
        # connect 超时独立设置为 10s，避免网络问题长时间阻塞
        timeout = httpx.Timeout(OLLAMA_TIMEOUT, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    else:
        # httpx 未安装时降级为 urllib，需通过线程池避免阻塞事件循环
        data = await asyncio.to_thread(_post_generate_via_urllib, url, payload)

    result = _extract_generate_text(data)
    if result:
        return result

    if data.get("error"):
        raise RuntimeError(f"Ollama error: {data.get('error')}")
    raise RuntimeError(f"Ollama 返回为空 ({_summarize_empty_response(data)})")


def _post_generate_via_urllib(url: str, payload: dict) -> dict[str, Any]:
    """urllib 同步版本的 Ollama 请求，在线程池中运行，作为 httpx 不可用时的兜底。"""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


async def generate_json(
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    model: str | None = None,
    options: dict[str, Any] | None = None,
) -> dict:
    """调用 Ollama 生成结构化 JSON 响应，通过 format 字段传入 JSON Schema 约束输出格式。

    temperature 设为 0.1 以确保输出尽可能确定（适合评分场景）。
    异常时重新抛出 RuntimeError，由调用方决定是否降级为启发式方法。
    """
    actual_model = model or OLLAMA_DEFAULT_MODEL
    merged_options = {
        "temperature": 0.0,
        "num_ctx": OLLAMA_JSON_NUM_CTX,
        "num_predict": OLLAMA_JSON_NUM_PREDICT,
    }
    if options:
        merged_options.update(options)

    payload = {
        "model": actual_model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "keep_alive": "10m",
        "format": schema,  # 通过 format 字段强制 Ollama 按 Schema 输出 JSON
        "options": merged_options,
    }
    logger.info(
        "Calling Ollama JSON: base_url=%s model=%s prompt_len=%d schema_keys=%s options=%s",
        OLLAMA_BASE_URL,
        actual_model,
        len(user_prompt),
        list(schema.get("properties", {}).keys()) if isinstance(schema, dict) else None,
        payload.get("options"),
    )
    try:
        raw = await _post_generate(payload)
        return _extract_json(raw)
    except Exception as exc:
        logger.warning("Ollama schema JSON failed, retrying with format=json: %s", exc)
        fallback_payload = {
            **payload,
            "format": "json",
        }
        try:
            raw = await _post_generate(fallback_payload)
            return _extract_json(raw)
        except Exception as retry_exc:
            logger.exception("Ollama JSON generation failed after retry: %s", retry_exc)
            raise RuntimeError(f"Ollama JSON generation failed: {retry_exc}") from retry_exc


async def generate_text(system_prompt: str, user_prompt: str, model: str | None = None) -> str:
    """调用 Ollama 生成纯文本响应（不使用 Schema 约束）。

    temperature 设为 0.3，比 JSON 模式略高，允许适度的表达多样性（适合改写场景）。
    """
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
