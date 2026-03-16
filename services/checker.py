from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.request
from typing import Optional

try:
    import httpx
except ImportError:  # pragma: no cover - fallback for minimal local env
    httpx = None

from services.aggregate import aggregate_document, score_chunk
from services.features import extract_chunk_features, extract_document_features
from services.judges import judge_chunk_with_qwen
from services.ollama_client import OLLAMA_DEFAULT_MODEL, generate_json
from services.preprocess import chunk_text, clean_text, detect_genre
from services.prompts import REDUCE_REWRITE_SYSTEM_PROMPT, REMOTE_REVIEW_SYSTEM_PROMPT
from services.schemas import ReduceRewriteResult, RemoteReviewResult


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("nohup.out", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

API_SOURCE = os.getenv("API_SOURCE", "ollama").lower()
AZURE_DEFAULT_MODEL = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "claude-3.5-haiku")
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "http://119.28.110.115:5000")
PROXY_TOKEN = os.getenv("PROXY_TOKEN", "10a8ed53-e497-4f59-9662-0c650dd889ff")
PRO_REVIEW_SOURCE = os.getenv("PRO_REVIEW_SOURCE", "openrouter").lower()
PRO_REVIEW_MODEL = os.getenv("PRO_REVIEW_MODEL", OPENROUTER_DEFAULT_MODEL)

_azure_client = None
_openrouter_client = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _default_model_for(source: str) -> str:
    if source == "azure":
        return AZURE_DEFAULT_MODEL
    if source == "openrouter":
        return OPENROUTER_DEFAULT_MODEL
    return OLLAMA_DEFAULT_MODEL


def _review_model_for(source: str) -> str:
    if source == "azure":
        return os.getenv("PRO_REVIEW_MODEL", AZURE_DEFAULT_MODEL)
    if source == "openrouter":
        return os.getenv("PRO_REVIEW_MODEL", OPENROUTER_DEFAULT_MODEL)
    return PRO_REVIEW_MODEL


def _get_azure_client():
    global _azure_client
    if _azure_client is not None:
        return _azure_client

    azure_key = os.getenv("AZURE_API_KEY")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    if not azure_key or not azure_endpoint:
        return None

    from openai import AsyncAzureOpenAI

    _azure_client = AsyncAzureOpenAI(
        api_key=azure_key,
        azure_endpoint=azure_endpoint,
        api_version=os.getenv("AZURE_API_VERSION", "2025-01-01-preview"),
    )
    return _azure_client


def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is not None:
        return _openrouter_client

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        return None

    from openai import AsyncOpenAI

    _openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
    return _openrouter_client


async def _send_by_proxy(messages: list, model: str) -> str:
    url = f"{PROXY_BASE_URL}/api/chats/openrouter/{model}"
    payload = {
        "messages": messages,
        "token": PROXY_TOKEN,
        "version": 0,
    }
    if httpx is not None:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    else:
        data = await asyncio.to_thread(_send_by_proxy_via_urllib, url, payload)
    if data.get("errno") != 0:
        raise RuntimeError(f"Proxy error: {data.get('message', 'unknown error')}")
    return data["re"]


def _send_by_proxy_via_urllib(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


async def _call_remote_json(system_prompt: str, user_prompt: str, model: str | None, api_source: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    actual_model = model or _default_model_for(api_source)

    if api_source == "azure":
        client = _get_azure_client()
        if not client:
            raise RuntimeError("Azure 未配置")
        logger.info("Calling Azure model=%s for reduce", actual_model)
        response = await client.chat.completions.create(
            model=actual_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return _extract_json(response.choices[0].message.content)

    if api_source == "openrouter":
        send_mode = os.getenv("OPENROUTER_SEND_MODE", "self").lower()
        if send_mode == "proxy":
            logger.info("Calling OpenRouter proxy model=%s for reduce", actual_model)
            return _extract_json(await _send_by_proxy(messages, actual_model))

        client = _get_openrouter_client()
        if not client:
            raise RuntimeError("OpenRouter 未配置")
        logger.info("Calling OpenRouter self model=%s for reduce", actual_model)
        response = await client.chat.completions.create(
            model=actual_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
            extra_headers={
                "HTTP-Referer": "https://aigcchecker.com",
                "X-Title": "AIGC Checker Detection Engine",
            },
        )
        return _extract_json(response.choices[0].message.content)

    raise RuntimeError(f"Unsupported remote api_source: {api_source}")


def _rewrite_schema() -> dict:
    return ReduceRewriteResult.model_json_schema()


def _remote_review_schema() -> dict:
    return RemoteReviewResult.model_json_schema()


def _build_reduce_prompt(content: str, detection_result: dict) -> str:
    high_risk_chunks = detection_result.get("analysis", {}).get("high_risk_chunks", [])
    summary_reasons = detection_result.get("analysis", {}).get("summary_reasons", [])
    chunk_payload = [
        {
            "chunk_id": chunk["chunk_id"],
            "ai_score": chunk["ai_score"],
            "label": chunk["label"],
            "reasons": chunk["reasons"][:3],
            "text": chunk["text"],
        }
        for chunk in detection_result.get("chunks", [])
        if chunk["chunk_id"] in high_risk_chunks[:5]
    ]
    return (
        f"原文 AI 概率: {detection_result.get('ai_probability')}\n"
        f"文体: {detection_result.get('analysis', {}).get('genre')}\n"
        f"摘要原因: {json.dumps(summary_reasons, ensure_ascii=False)}\n"
        f"优先改写分块: {json.dumps(chunk_payload, ensure_ascii=False)}\n\n"
        f"原始文本:\n{content}"
    )


def _build_remote_review_prompt(content: str, detection_result: dict) -> str:
    analysis = detection_result.get("analysis", {})
    doc_features = detection_result.get("document_features", {})
    candidate_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "ai_score": chunk["ai_score"],
            "label": chunk["label"],
            "confidence": chunk["confidence"],
            "reasons": chunk["reasons"][:3],
            "text": chunk["text"],
        }
        for chunk in detection_result.get("chunks", [])[:4]
    ]
    payload = {
        "genre": analysis.get("genre"),
        "document_ai_probability": detection_result.get("ai_probability"),
        "document_confidence": detection_result.get("confidence"),
        "summary_reasons": analysis.get("summary_reasons", []),
        "document_features": {
            "char_count": doc_features.get("char_count"),
            "sentence_count": doc_features.get("sentence_count"),
            "avg_sentence_length": doc_features.get("avg_sentence_length"),
            "sentence_length_std": doc_features.get("sentence_length_std"),
            "burstiness": doc_features.get("burstiness"),
            "lexical_diversity": doc_features.get("lexical_diversity"),
            "repeated_ngram_ratio": doc_features.get("repeated_ngram_ratio"),
            "connector_density": doc_features.get("connector_density"),
            "detail_signal_count": doc_features.get("detail_signal_count"),
        },
        "chunks": candidate_chunks,
        "content": content,
    }
    return json.dumps(payload, ensure_ascii=False)


def _should_trigger_remote_review(result: dict, plan: str, can_remote_review: bool) -> tuple[bool, str]:
    if plan != "pro" or not can_remote_review:
        return False, "plan_disabled"

    score = float(result.get("ai_probability", 0))
    confidence = result.get("confidence")
    genre = result.get("analysis", {}).get("genre")
    high_risk_count = len(result.get("analysis", {}).get("high_risk_chunks", []))

    if confidence == "low":
        return True, "low_confidence"
    if 40 <= score <= 65:
        return True, "borderline_score"
    if genre in {"academic", "business_doc", "list_or_table"} and high_risk_count <= 1:
        return True, "conservative_genre"
    return False, "not_needed"


def _map_review_label(label: str) -> str:
    if label == "ai":
        return "AI Generated"
    if label == "mixed":
        return "Mixed"
    return "Human Written"


def _map_score_label(score: float) -> str:
    if score >= 70:
        return "AI Generated"
    if score >= 40:
        return "Mixed"
    return "Human Written"


def _summary_percentages(score: float) -> dict:
    ai_strength = _clamp((score - 45) / 55, 0, 1) ** 1.2
    human_strength = _clamp((55 - score) / 55, 0, 1) ** 1.2
    mixed_strength = max(0.1, 1 - abs(score - 50) / 45)
    total = ai_strength + human_strength + mixed_strength
    percentages = {
        "ai": int(round(ai_strength / total * 100)),
        "mixed": int(round(mixed_strength / total * 100)),
        "human": int(round(human_strength / total * 100)),
    }
    diff = 100 - sum(percentages.values())
    if diff:
        percentages["mixed"] += diff
    return percentages


def _merge_review_result(result: dict, review_result: dict, review_meta: dict) -> dict:
    original_score = float(result["ai_probability"])
    review_score = float(review_result["ai_score"])
    blended = 0.6 * original_score + 0.4 * review_score if result.get("confidence") == "low" else 0.72 * original_score + 0.28 * review_score
    blended = _clamp(blended, 0, 100)
    result["ai_probability"] = f"{blended:.2f}"
    result["label"] = _map_score_label(blended)
    result["summary"]["confidence_label"] = result["label"]
    result["summary"]["percentages"] = _summary_percentages(blended)
    review_conf = review_result.get("confidence", 0.0)
    if review_conf >= 0.75:
        result["confidence"] = "high"
    elif review_conf >= 0.45 and result["confidence"] == "low":
        result["confidence"] = "medium"
    result["analysis"]["summary_reasons"] = list(
        dict.fromkeys(review_result.get("reasons", []) + result["analysis"].get("summary_reasons", []))
    )[:5]
    result["review"] = {
        **review_meta,
        "used": True,
        "result": review_result,
    }
    return result


async def _run_remote_review(content: str, result: dict) -> dict | None:
    if PRO_REVIEW_SOURCE not in {"azure", "openrouter"}:
        logger.info("Remote review skipped: unsupported source=%s", PRO_REVIEW_SOURCE)
        return None

    try:
        raw = await _call_remote_json(
            REMOTE_REVIEW_SYSTEM_PROMPT,
            _build_remote_review_prompt(content, result),
            model=_review_model_for(PRO_REVIEW_SOURCE),
            api_source=PRO_REVIEW_SOURCE,
        )
        return RemoteReviewResult.model_validate(raw).model_dump()
    except Exception as exc:
        logger.warning("Remote review skipped due to error: %s", exc)
        return None


def _trim_result_by_plan(result: dict, plan: str) -> dict:
    result["plan"] = plan
    if "review" not in result:
        result["review"] = {
            "enabled": plan == "pro",
            "used": False,
            "provider": PRO_REVIEW_SOURCE if plan == "pro" else None,
            "model": _review_model_for(PRO_REVIEW_SOURCE) if plan == "pro" else None,
        }

    if plan == "pro":
        return result

    trimmed = dict(result)
    trimmed.pop("document_features", None)
    trimmed.pop("chunks", None)
    trimmed.pop("engine", None)
    trimmed_analysis = dict(trimmed.get("analysis", {}))
    trimmed_analysis.pop("summary_reasons", None)
    trimmed["analysis"] = {
        "genre": trimmed_analysis.get("genre"),
        "high_risk_chunk_count": len(trimmed_analysis.get("high_risk_chunks", [])),
    }
    return trimmed


def _quality_score(original: str, rewritten: str, before: float, after: float) -> float:
    original_tokens = set(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]", original.lower()))
    rewritten_tokens = set(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]", rewritten.lower()))
    overlap = len(original_tokens & rewritten_tokens) / max(len(original_tokens | rewritten_tokens), 1)
    length_ratio = min(len(rewritten) / max(len(original), 1), max(len(original), 1) / max(len(rewritten), 1))
    preservation = 100 * (0.7 * overlap + 0.3 * length_ratio)
    improvement = _clamp(before - after, 0, 60) / 60 * 100
    return round(_clamp(0.6 * preservation + 0.4 * improvement, 0, 100), 2)


async def _rewrite_content(content: str, detection_result: dict, model: str | None, api_source: str | None) -> dict:
    source = (api_source or API_SOURCE).lower()
    prompt = _build_reduce_prompt(content, detection_result)
    schema = _rewrite_schema()

    try:
        if source in {"azure", "openrouter"}:
            raw = await _call_remote_json(REDUCE_REWRITE_SYSTEM_PROMPT, prompt, model=model, api_source=source)
        else:
            raw = await generate_json(REDUCE_REWRITE_SYSTEM_PROMPT, prompt, schema=schema, model=model or OLLAMA_DEFAULT_MODEL)
        return ReduceRewriteResult.model_validate(raw).model_dump()
    except Exception as exc:
        logger.warning("Rewrite failed, falling back to original text: %s", exc)
        return ReduceRewriteResult(
            reduced=content,
            quality_score=55.0,
            model="light",
            changes=[],
        ).model_dump()


async def run_check(
    content: str,
    model: Optional[str] = None,
    api_source: Optional[str] = None,
    plan: str = "free",
    can_remote_review: bool = False,
) -> dict:
    cleaned = clean_text(content)
    genre = detect_genre(cleaned)
    doc_features = extract_document_features(cleaned)
    chunks = chunk_text(cleaned)
    logger.info("Running check: genre=%s, model=%s, chunks=%d", genre, model or OLLAMA_DEFAULT_MODEL, len(chunks))

    enriched_chunks: list[dict] = []
    for chunk in chunks:
        features = extract_chunk_features(chunk["text"])
        qwen_result = await judge_chunk_with_qwen(chunk["text"], genre, features, model=model or OLLAMA_DEFAULT_MODEL)
        scored = score_chunk(features, qwen_result, genre)
        enriched_chunks.append(
            {
                **chunk,
                **scored,
                "features": features,
                "signals": qwen_result.get("signals", {}),
                "judge_label": qwen_result.get("label"),
            }
        )

    result = aggregate_document(enriched_chunks, genre, doc_features)
    result["engine"] = {
        "checker": "hybrid-local",
        "qwen_model": model or OLLAMA_DEFAULT_MODEL,
        "api_source": api_source or API_SOURCE,
    }
    should_review, review_reason = _should_trigger_remote_review(result, plan=plan, can_remote_review=can_remote_review)
    review_meta = {
        "enabled": plan == "pro" and can_remote_review,
        "used": False,
        "provider": PRO_REVIEW_SOURCE if plan == "pro" and can_remote_review else None,
        "model": _review_model_for(PRO_REVIEW_SOURCE) if plan == "pro" and can_remote_review else None,
        "reason": review_reason,
    }
    result["review"] = review_meta

    if should_review:
        review_result = await _run_remote_review(cleaned, result)
        if review_result:
            result = _merge_review_result(result, review_result, review_meta)

    return _trim_result_by_plan(result, plan=plan)


async def run_reduce(
    content: str,
    model: Optional[str] = None,
    api_source: Optional[str] = None,
    plan: str = "pro",
    can_remote_review: bool = False,
) -> dict:
    original_result = await run_check(
        content=content,
        model=model,
        api_source=api_source,
        plan=plan,
        can_remote_review=can_remote_review,
    )
    rewrite_result = await _rewrite_content(content, original_result, model=model, api_source=api_source)
    reduced_text = clean_text(rewrite_result.get("reduced") or content)
    reduced_result = await run_check(
        content=reduced_text,
        model=model,
        api_source=api_source,
        plan=plan,
        can_remote_review=can_remote_review,
    )

    before = float(original_result["ai_probability"])
    after = float(reduced_result["ai_probability"])
    quality = _quality_score(content, reduced_text, before, after)

    return {
        "reduced": reduced_text,
        "ai_probability": f"{before:.2f}",
        "ai_reduced_probability": f"{after:.2f}",
        "quality_score": quality,
        "model": rewrite_result.get("model", "moderate"),
        "changes": rewrite_result.get("changes", []),
    }
