from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.request
import uuid
from typing import Optional

try:
    import httpx
except ImportError:  # pragma: no cover - fallback for minimal local env
    httpx = None

from services.aggregate import aggregate_document, score_chunk
from services.features import extract_chunk_features, extract_document_features
from services.judges import judge_chunk_with_qwen
from services.ollama_client import OLLAMA_DEFAULT_MODEL
from services.preprocess import chunk_text, clean_text, detect_genre, detect_language
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
SUPPORTED_API_SOURCES = {"ollama", "azure", "openrouter"}
SCAN_API_SOURCE = "ollama"
AZURE_DEFAULT_MODEL = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")
AZURE_REWRITE_DEPLOYMENT = os.getenv("AZURE_REWRITE_DEPLOYMENT", AZURE_DEFAULT_MODEL)
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-plus")
OPENROUTER_REWRITE_MODEL = os.getenv("OPENROUTER_REWRITE_MODEL", OPENROUTER_DEFAULT_MODEL)
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "http://119.28.110.115:5000")
PROXY_TOKEN = os.getenv("PROXY_TOKEN", "10a8ed53-e497-4f59-9662-0c650dd889ff")
PRO_REVIEW_SOURCE = os.getenv("PRO_REVIEW_SOURCE", "openrouter").lower()
PRO_REVIEW_MODEL = os.getenv("PRO_REVIEW_MODEL", OPENROUTER_DEFAULT_MODEL)
OPENROUTER_EXCLUDE_REASONING = os.getenv("OPENROUTER_EXCLUDE_REASONING", "true").lower() not in {"false", "0", "no"}
# 远端二审全局开关，设为 false/0/no 可临时关闭，不影响本地检测流程
REMOTE_REVIEW_ENABLED = os.getenv("REMOTE_REVIEW_ENABLED", "true").lower() not in {"false", "0", "no"}

_azure_client = None
_openrouter_client = None


def _clamp(value: float, low: float, high: float) -> float:
    """将 value 限定在 [low, high] 区间内。"""
    return max(low, min(high, value))


def new_task_id(prefix: str) -> str:
    """生成简短任务 ID，便于串联 API 请求全链路日志。"""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _log_task(task_id: str | None, step: str, **fields) -> None:
    """统一输出结构化任务日志。"""
    payload = " ".join(f"{key}={value}" for key, value in fields.items())
    if payload:
        logger.info("task=%s step=%s %s", task_id or "-", step, payload)
    else:
        logger.info("task=%s step=%s", task_id or "-", step)


def _normalize_api_source(source: str | None) -> str:
    """规范化请求来源，未知值回退为默认来源。"""
    normalized = (source or API_SOURCE).lower()
    return normalized if normalized in SUPPORTED_API_SOURCES else API_SOURCE


def _resolve_scan_model(requested_api_source: str, requested_model: str | None) -> str:
    """scan 主链路固定走 Ollama，仅在明确请求 Ollama 时允许覆盖本地模型。"""
    if requested_api_source == SCAN_API_SOURCE and requested_model:
        return requested_model
    return OLLAMA_DEFAULT_MODEL


def _review_model_for(source: str) -> str:
    """根据 API 来源返回 Pro 二审所用模型，优先读取环境变量覆盖值。"""
    if source == "azure":
        return os.getenv("PRO_REVIEW_MODEL", AZURE_DEFAULT_MODEL)
    if source == "openrouter":
        return os.getenv("PRO_REVIEW_MODEL", OPENROUTER_DEFAULT_MODEL)
    return PRO_REVIEW_MODEL


def _rewrite_attempts() -> list[tuple[str, str]]:
    """返回固定的改写提供方顺序：先 Azure，再 OpenRouter。"""
    return [
        ("azure", AZURE_REWRITE_DEPLOYMENT),
        ("openrouter", OPENROUTER_REWRITE_MODEL),
    ]


def _get_azure_client():
    """获取 Azure OpenAI 异步客户端（单例模式），未配置环境变量时返回 None。"""
    global _azure_client
    # 已初始化则直接复用，避免重复创建连接
    if _azure_client is not None:
        return _azure_client

    azure_key = os.getenv("AZURE_API_KEY")
    azure_endpoint = os.getenv("AZURE_ENDPOINT")
    # 缺少必要配置时安全返回 None，由调用方抛出更友好的错误
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
    """获取 OpenRouter 异步客户端（单例模式），未配置 API Key 时返回 None。"""
    global _openrouter_client
    if _openrouter_client is not None:
        return _openrouter_client

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        return None

    from openai import AsyncOpenAI

    # OpenRouter 使用兼容 OpenAI 的接口，仅替换 base_url
    _openrouter_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
    return _openrouter_client


async def _send_by_proxy(messages: list, model: str) -> str:
    """通过内部代理转发请求到 OpenRouter，返回模型的原始文本响应。

    优先使用 httpx 异步客户端，若未安装则退回到标准库 urllib（阻塞线程）。
    """
    url = f"{PROXY_BASE_URL}/api/chats/openrouter/{model}"
    payload = {
        "messages": messages,
        "token": PROXY_TOKEN,
        "version": 0,
    }
    if httpx is not None:
        # httpx 异步方式，适合高并发场景
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    else:
        # 回退到 urllib，需借助线程池以避免阻塞事件循环
        data = await asyncio.to_thread(_send_by_proxy_via_urllib, url, payload)
    # errno != 0 表示代理服务本身返回了业务错误
    if data.get("errno") != 0:
        raise RuntimeError(f"Proxy error: {data.get('message', 'unknown error')}")
    return data["re"]


def _send_by_proxy_via_urllib(url: str, payload: dict) -> dict:
    """urllib 同步实现的代理请求，在线程池中调用，避免阻塞异步事件循环。"""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _extract_json(text: str) -> dict:
    """从字符串中提取 JSON 对象，先直接解析，失败后用正则提取第一个 {...} 块再解析。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 模型有时会在 JSON 前后附加解释性文字，用正则兜底提取花括号内容
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _strip_html(text: str) -> str:
    """去掉模型偶发返回的 HTML 标签，避免污染最终改写文本。"""
    cleaned = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _format_probability(value, default: str = "0.00") -> str:
    """将概率字段规范为保留两位小数的字符串。"""
    try:
        if value is None or value == "":
            return default
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return default


def _normalize_reduce_changes(changes) -> list[dict]:
    """宽松规范化 changes，兼容字符串列表和宽松字典结构。"""
    normalized: list[dict] = []
    if not isinstance(changes, list):
        return normalized

    for item in changes[:8]:
        if isinstance(item, dict):
            normalized.append(
                {
                    "original": str(item.get("original") or item.get("before") or ""),
                    "revised": str(item.get("revised") or item.get("after") or ""),
                    "reason": str(item.get("reason") or item.get("detail") or item.get("type") or ""),
                }
            )
            continue
        if isinstance(item, str):
            normalized.append(
                {
                    "original": "",
                    "revised": "",
                    "reason": item.strip(),
                }
            )
    return normalized


def _normalize_rewrite_result(raw: dict, original_content: str) -> dict:
    """将 Azure/OpenRouter 的宽松 JSON 结果规范成 ReduceRewriteResult 可接受的结构。"""
    if not isinstance(raw, dict):
        raise ValueError("rewrite result must be a dict")

    reduced_candidate = raw.get("reduced")
    rewrite_candidate = raw.get("rewrite")

    reduced_text = ""
    if isinstance(reduced_candidate, str) and reduced_candidate.strip():
        reduced_text = reduced_candidate
    elif isinstance(rewrite_candidate, str) and rewrite_candidate.strip():
        # 某些模型会把 rewrite 字段误写成改写正文
        reduced_text = rewrite_candidate
    elif isinstance(raw.get("content"), str) and raw.get("content", "").strip():
        reduced_text = raw["content"]

    reduced_text = _strip_html(reduced_text) if reduced_text else original_content
    rewrite_flag = raw.get("rewrite")
    if isinstance(rewrite_flag, bool):
        rewrite_success = rewrite_flag
    else:
        rewrite_success = bool(reduced_text and reduced_text != original_content)

    model_value = str(raw.get("model") or "moderate").lower()
    if model_value not in {"light", "moderate", "deep"}:
        model_value = "moderate"

    quality = raw.get("quality_score", 0)
    try:
        quality_score = round(_clamp(float(quality), 0, 100), 2)
    except (TypeError, ValueError):
        quality_score = 0.0

    return {
        "reduced": reduced_text,
        "rewrite": rewrite_success,
        "ai_probability": _format_probability(raw.get("ai_probability")),
        "ai_reduced_probability": _format_probability(raw.get("ai_reduced_probability")),
        "quality_score": quality_score,
        "model": model_value,
        "changes": _normalize_reduce_changes(raw.get("changes")),
    }


async def _call_azure_json(system_prompt: str, user_prompt: str, deployment_name: str, purpose: str, task_id: str | None = None) -> dict:
    """调用 Azure OpenAI Chat Completions JSON 模式。

    Azure `model` 参数应传 deployment name，而不是公共模型 ID。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    client = _get_azure_client()
    if not client:
        raise RuntimeError("Azure 未配置")
    _log_task(task_id, "provider.azure.call", purpose=purpose, deployment=deployment_name)
    response = await client.chat.completions.create(
        model=deployment_name,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return _extract_json(response.choices[0].message.content)


async def _call_openrouter_json(system_prompt: str, user_prompt: str, model_id: str, purpose: str, task_id: str | None = None) -> dict:
    """调用 OpenRouter Chat Completions JSON 模式。

    OpenRouter `model` 参数应传公开 model ID，例如 `qwen/qwen-plus`。
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    send_mode = os.getenv("OPENROUTER_SEND_MODE", "self").lower()
    if send_mode == "proxy":
        _log_task(task_id, "provider.openrouter.proxy_call", purpose=purpose, model=model_id)
        return _extract_json(await _send_by_proxy(messages, model_id))

    client = _get_openrouter_client()
    if not client:
        raise RuntimeError("OpenRouter 未配置")

    request_kwargs = {
        "model": model_id,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "extra_headers": {
            "HTTP-Referer": "https://aigcchecker.com",
            "X-Title": "AIGC Checker Detection Engine",
        },
    }
    if OPENROUTER_EXCLUDE_REASONING:
        request_kwargs["extra_body"] = {"reasoning": {"exclude": True}}

    _log_task(task_id, "provider.openrouter.call", purpose=purpose, model=model_id)
    response = await client.chat.completions.create(**request_kwargs)
    return _extract_json(response.choices[0].message.content)


def _build_reduce_prompt(content: str, detection_result: dict) -> str:
    """构建降 AI 改写的用户 Prompt，携带检测结果摘要、统计特征目标和分块信息。

    改写范围扩展至 score>=40 的中高风险块（最多 8 个，按风险降序），
    同时传入具体的统计特征偏差，让模型知道需要改变哪些可测量指标。
    """
    doc_features = detection_result.get("document_features", {})

    # 按 ai_score 降序，纳入所有 score>=40 的中高风险块，最多 8 个
    risk_chunks = sorted(
        [c for c in detection_result.get("chunks", []) if c.get("ai_score", 0) >= 40],
        key=lambda c: c.get("ai_score", 0),
        reverse=True,
    )[:8]
    chunk_payload = [
        {"chunk_id": c["chunk_id"], "ai_score": c["ai_score"], "label": c["label"], "text": c["text"]}
        for c in risk_chunks
    ]

    # 根据统计特征生成具体改写目标，让模型有明确的可测量目标
    feature_targets = []
    if doc_features.get("burstiness", 1.0) < 0.3:
        feature_targets.append("句长过于均匀（burstiness 低），需在段落中插入短句（≤10字）或超长句（≥40字）")
    if (doc_features.get("sentence_length_std") or 0.0) < 8:
        feature_targets.append("句长标准差过小，需主动制造长短句混搭，拉大标准差")
    if doc_features.get("repeated_ngram_ratio", 0.0) > 0.05:
        feature_targets.append("存在重复短语（n-gram 重复率高），需替换同义或近义表达")
    if doc_features.get("lexical_diversity", 1.0) < 0.55:
        feature_targets.append("词汇多样性不足，需丰富用词，减少同一词汇反复出现")
    if doc_features.get("connector_density", 0.0) > 0.07:
        feature_targets.append("连接词过密（因此/然而/总之/首先/其次），需删减 50% 以上")
    if not feature_targets:
        feature_targets.append("整体指标尚可，做轻度去模板化处理即可")

    targets_str = "\n".join(f"- {t}" for t in feature_targets)
    return (
        f"原文 AI 概率: {detection_result.get('ai_probability')}\n"
        f"文体: {detection_result.get('analysis', {}).get('genre')}\n"
        f"统计特征改写目标:\n{targets_str}\n"
        f"需改写分块（score≥40，按风险降序，共 {len(chunk_payload)} 个）: "
        f"{json.dumps(chunk_payload, ensure_ascii=False)}\n\n"
        f"原始文本:\n{content}"
    )


def _build_remote_review_prompt(content: str, detection_result: dict) -> str:
    """构建 Pro 远端二审的用户 Prompt，包含文体、文档特征、前 4 个分块及原文。

    结构化为 JSON 字符串，便于模型理解各字段含义并给出更准确的二审意见。
    """
    analysis = detection_result.get("analysis", {})
    doc_features = detection_result.get("document_features", {})
    # 只取前 4 个分块，避免 Prompt 过长
    candidate_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "ai_score": chunk["ai_score"],
            "label": chunk["label"],
            "confidence": chunk["confidence"],
            "text": chunk["text"],
        }
        for chunk in detection_result.get("chunks", [])[:4]
    ]
    payload = {
        "genre": analysis.get("genre"),
        "document_ai_probability": detection_result.get("ai_probability"),
        "document_confidence": detection_result.get("confidence"),
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
    """判断是否需要触发 Pro 远端二审，返回 (是否触发, 触发原因)。

    触发条件（满足任一即触发）：
    1. 本地置信度为 low（不确定）
    2. AI 概率处于模糊边界区间 [40, 65]
    3. 正式文体（学术/商业/表格）且高风险分块不多，可能存在误判
    """
    if not REMOTE_REVIEW_ENABLED:
        return False, "review_disabled"
    if plan != "pro" or not can_remote_review:
        return False, "plan_disabled"

    score = float(result.get("ai_probability", 0))
    confidence = result.get("confidence")
    genre = result.get("analysis", {}).get("genre")
    high_risk_count = len(result.get("analysis", {}).get("high_risk_chunks", []))

    if confidence == "low":
        # 本地模型对结果没有把握，交给远端做更强的二审
        return True, "low_confidence"
    if 40 <= score <= 65:
        # 边界分数段，本地判断不稳定，二审有助于提升准确率
        return True, "borderline_score"
    if genre in {"academic", "business_doc", "list_or_table"} and high_risk_count <= 1:
        # 正式文体天然偏结构化，容易误判；高风险块少说明整体尚可，需保守复核
        return True, "conservative_genre"
    return False, "not_needed"


def _map_score_label(score: float) -> str:
    """根据 AI 概率分数映射为用户可读标签：>=70 AI Generated，>=40 Mixed，其余 Human Written。"""
    if score >= 70:
        return "AI Generated"
    if score >= 40:
        return "Mixed"
    return "Human Written"


def _summary_percentages(score: float) -> dict:
    """将单一 AI 概率分数转换为 ai/mixed/human 三方百分比分布。

    使用幂次缩放让高低分区间的分布更明显，mixed 项吸收四舍五入带来的误差，确保三项合计恰好为 100。
    """
    ai_strength = _clamp((score - 45) / 55, 0, 1) ** 1.2
    human_strength = _clamp((55 - score) / 55, 0, 1) ** 1.2
    mixed_strength = max(0.1, 1 - abs(score - 50) / 45)
    total = ai_strength + human_strength + mixed_strength
    percentages = {
        "ai": int(round(ai_strength / total * 100)),
        "mixed": int(round(mixed_strength / total * 100)),
        "human": int(round(human_strength / total * 100)),
    }
    # 四舍五入后三项之和可能不等于 100，用 mixed 补齐差值
    diff = 100 - sum(percentages.values())
    if diff:
        percentages["mixed"] += diff
    return percentages


def _merge_review_result(result: dict, review_result: dict, review_meta: dict) -> dict:
    """将远端二审分数融合到本地检测结果中，更新 label、置信度和摘要原因。

    融合权重：本地置信度 low 时更信任二审（0.6 本地 + 0.4 二审），否则以本地为主（0.72 + 0.28）。
    二审置信度 >= 0.75 时将结果置信度提升为 high；>= 0.45 且原为 low 时提升为 medium。
    """
    original_score = float(result["ai_probability"])
    review_score = float(review_result["ai_score"])
    # 本地置信度低时，适当增大二审权重以修正本地误差
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
    result["review"] = {
        **review_meta,
        "used": True,
        "result": review_result,
    }
    return result


async def _run_remote_review(content: str, result: dict, task_id: str | None = None) -> dict | None:
    """调用远端 LLM 进行二审，返回验证后的 RemoteReviewResult dict，失败时返回 None 不影响主流程。"""
    if PRO_REVIEW_SOURCE not in {"azure", "openrouter"}:
        _log_task(task_id, "review.skip", reason="unsupported_source", source=PRO_REVIEW_SOURCE)
        return None

    try:
        review_prompt = _build_remote_review_prompt(content, result)
        if PRO_REVIEW_SOURCE == "azure":
            raw = await _call_azure_json(
                REMOTE_REVIEW_SYSTEM_PROMPT,
                review_prompt,
                deployment_name=_review_model_for(PRO_REVIEW_SOURCE),
                purpose="remote_review",
                task_id=task_id,
            )
        else:
            raw = await _call_openrouter_json(
                REMOTE_REVIEW_SYSTEM_PROMPT,
                review_prompt,
                model_id=_review_model_for(PRO_REVIEW_SOURCE),
                purpose="remote_review",
                task_id=task_id,
            )
        return RemoteReviewResult.model_validate(raw).model_dump()
    except Exception as exc:
        _log_task(task_id, "review.skip", reason="provider_error", source=PRO_REVIEW_SOURCE, error=exc)
        return None


def _trim_result_by_plan(result: dict, plan: str) -> dict:
    """统一补充套餐信息与 review 元数据，不再按套餐裁剪 /scan 返回字段。"""
    result["plan"] = plan
    if "review" not in result:
        result["review"] = {
            "enabled": plan == "pro",
            "used": False,
            "provider": PRO_REVIEW_SOURCE if plan == "pro" else None,
            "model": _review_model_for(PRO_REVIEW_SOURCE) if plan == "pro" else None,
        }
    return result


def _quality_score(original: str, rewritten: str, before: float, after: float) -> float:
    """计算改写质量综合得分（0-100），兼顾内容保留度和 AI 概率下降幅度。

    - preservation（内容保留）：词汇重叠（70%）+ 长度比（30%），满分 100
    - improvement（降 AI 效果）：before-after 最多 60 分映射到 100，超出部分截断
    - 最终：60% preservation + 40% improvement
    """
    original_tokens = set(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]", original.lower()))
    rewritten_tokens = set(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]", rewritten.lower()))
    # Jaccard 相似度衡量词汇层面的内容保留程度
    overlap = len(original_tokens & rewritten_tokens) / max(len(original_tokens | rewritten_tokens), 1)
    # 长度比取两个方向的最小值，防止改写后长度差异过大
    length_ratio = min(len(rewritten) / max(len(original), 1), max(len(original), 1) / max(len(rewritten), 1))
    preservation = 100 * (0.7 * overlap + 0.3 * length_ratio)
    # AI 概率下降幅度，上限 60 分以避免单一维度权重过大
    improvement = _clamp(before - after, 0, 60) / 60 * 100
    return round(_clamp(0.6 * preservation + 0.4 * improvement, 0, 100), 2)


async def _rewrite_content(content: str, detection_result: dict, model: str | None, api_source: str | None, task_id: str | None = None) -> dict:
    """调用 LLM 对原文进行降 AI 改写，返回 ReduceRewriteResult dict。

    当前固定策略：
    - scan 始终走 Ollama
    - reduce 的 rewrite 始终优先走 Azure GPT-4o（deployment）
    - 仅当 Azure 失败时，才回退到 OpenRouter 的 Qwen-Plus（model ID）
    改写失败时返回原文作为 fallback，并通过 rewrite=False 标记未成功改写。
    """
    requested_api_source = _normalize_api_source(api_source)
    prompt = _build_reduce_prompt(content, detection_result)
    for rewrite_provider, provider_model in _rewrite_attempts():
        try:
            _log_task(
                task_id,
                "rewrite.attempt",
                provider=rewrite_provider,
                target_model=provider_model,
                requested_api_source=requested_api_source,
                requested_model=model or "default",
            )
            if rewrite_provider == "azure":
                raw = await _call_azure_json(
                    REDUCE_REWRITE_SYSTEM_PROMPT,
                    prompt,
                    deployment_name=provider_model,
                    purpose="rewrite",
                    task_id=task_id,
                )
            else:
                raw = await _call_openrouter_json(
                    REDUCE_REWRITE_SYSTEM_PROMPT,
                    prompt,
                    model_id=provider_model,
                    purpose="rewrite_fallback",
                    task_id=task_id,
                )
            normalized = _normalize_rewrite_result(raw, content)
            _log_task(task_id, "rewrite.success", provider=rewrite_provider, target_model=provider_model, rewrite=normalized.get("rewrite"))
            return ReduceRewriteResult.model_validate(normalized).model_dump()
        except Exception as exc:
            _log_task(task_id, "rewrite.failure", provider=rewrite_provider, target_model=provider_model, error=exc)

    _log_task(task_id, "rewrite.fallback", reason="all_providers_failed")
    return ReduceRewriteResult(
        reduced=content,
        rewrite=False,
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
    task_id: str | None = None,
) -> dict:
    """主检测流程：清洗文本 → 检测文体 → 提取特征 → 分块打分 → 聚合结果 → 可选二审 → 按套餐裁剪。

    Args:
        content: 待检测的原始文本
        model: 请求侧模型参数；当前仅在请求来源为 ollama 时才会覆盖本地检测模型
        api_source: 请求侧来源（ollama/azure/openrouter），当前 scan 实际执行仍固定为 Ollama
        plan: 订阅套餐（free/pro），影响返回字段和二审权限
        can_remote_review: 是否允许触发 Pro 远端二审

    Returns:
        包含 label、ai_probability、confidence、chunks 等字段的检测结果 dict
    """
    # Step 1: 文本预处理
    requested_api_source = _normalize_api_source(api_source)
    effective_scan_model = _resolve_scan_model(requested_api_source, model)
    cleaned = clean_text(content)
    language = detect_language(cleaned)
    genre = detect_genre(cleaned)
    doc_features = extract_document_features(cleaned, language=language)
    chunks = chunk_text(cleaned)
    _log_task(
        task_id,
        "scan.start",
        requested_api_source=requested_api_source,
        requested_model=model or "default",
        scan_source=SCAN_API_SOURCE,
        scan_model=effective_scan_model,
        plan=plan,
        language=language,
        chars=len(cleaned),
        chunks=len(chunks),
    )
    _log_task(task_id, "scan.genre", genre=genre)

    # Step 2: 逐分块提取特征 + LLM 打分 + 规则综合评分
    enriched_chunks: list[dict] = []
    for chunk in chunks:
        chunk_language = detect_language(chunk["text"]) if language == "mixed" else language
        features = extract_chunk_features(chunk["text"], language=chunk_language)
        _log_task(task_id, "scan.chunk", chunk_id=chunk["chunk_id"], chunk_language=chunk_language, chars=len(chunk["text"]))
        # 调用 Qwen 本地模型对分块进行风格判断，失败时自动降级为启发式规则
        qwen_result = await judge_chunk_with_qwen(
            chunk["text"],
            genre,
            features,
            model=effective_scan_model,
            task_id=task_id,
            chunk_id=chunk["chunk_id"],
        )
        # 融合 LLM 分数（55%）、统计特征分（30%）、风格信号分（15%）
        scored = score_chunk(features, qwen_result, genre)
        enriched_chunks.append(
            {
                **chunk,
                **scored,
                "features": features,
                "signals": qwen_result.get("signals", {}),
                "judge_label": qwen_result.get("label"),
                "judge_mode": qwen_result.get("judge_mode", "qwen"),
                "judge_skip_reason": qwen_result.get("judge_skip_reason"),
                "perplexity_proxy": qwen_result.get("perplexity_proxy"),
                "binoculars_score": qwen_result.get("binoculars_score"),
            }
        )

    # Step 3: 将分块结果聚合为文档级结论
    result = aggregate_document(enriched_chunks, genre, doc_features)
    result["engine"] = {
        "checker": "hybrid-local",
        "requested_api_source": requested_api_source,
        "requested_model": model,
        "scan_api_source": SCAN_API_SOURCE,
        "qwen_model": effective_scan_model,
    }

    # Step 4: 判断是否需要触发 Pro 远端二审，并在结果中记录二审元数据
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
        _log_task(task_id, "review.trigger", reason=review_reason, provider=PRO_REVIEW_SOURCE)
        review_result = await _run_remote_review(cleaned, result, task_id=task_id)
        if review_result:
            # 将二审分数融合进最终结果
            result = _merge_review_result(result, review_result, review_meta)
            _log_task(task_id, "review.merge", provider=PRO_REVIEW_SOURCE, model=_review_model_for(PRO_REVIEW_SOURCE))
    else:
        _log_task(task_id, "review.skip", reason=review_reason)

    # Step 5: 按套餐裁剪返回字段
    _log_task(task_id, "scan.finish", ai_probability=result.get("ai_probability"), confidence=result.get("confidence"), label=result.get("label"))
    return _trim_result_by_plan(result, plan=plan)


async def run_reduce(
    content: str,
    model: Optional[str] = None,
    api_source: Optional[str] = None,
    plan: str = "pro",
    can_remote_review: bool = False,
    task_id: str | None = None,
) -> dict:
    """降 AI 改写主流程：先 Ollama 检测 → Azure 改写 → OpenRouter 兜底 → 再检测 → 计算质量分。

    流程：
    1. run_check 获取原文 AI 概率及高风险分块信息
    2. _rewrite_content 根据检测结果改写原文
    3. 对改写后的文本再次 run_check，获取降 AI 后的概率
    4. _quality_score 综合内容保留度和降 AI 幅度计算质量分

    Returns:
        包含 reduced（改写文本）、rewrite（是否成功改写）、ai_probability（原始概率）、
        ai_reduced_probability（改写后概率）、quality_score 的 dict
    """
    _log_task(task_id, "reduce.start", requested_api_source=_normalize_api_source(api_source), requested_model=model or "default", chars=len(content))
    # Step 1: 检测原文，获取 AI 概率和高风险分块
    original_result = await run_check(
        content=content,
        model=model,
        api_source=api_source,
        plan=plan,
        can_remote_review=can_remote_review,
        task_id=task_id,
    )
    # Step 2: 根据检测结果生成改写文本
    rewrite_result = await _rewrite_content(content, original_result, model=model, api_source=api_source, task_id=task_id)
    # Step 3: 清洗改写文本，回退到原文防止改写结果为空
    reduced_text = clean_text(rewrite_result.get("reduced") or content)
    _log_task(task_id, "reduce.post_rewrite", rewrite=rewrite_result.get("rewrite"), reduced_chars=len(reduced_text))
    reduced_result = await run_check(
        content=reduced_text,
        model=model,
        api_source=api_source,
        plan=plan,
        can_remote_review=can_remote_review,
        task_id=task_id,
    )

    # Step 4: 计算改写前后概率差异和质量综合得分
    before = float(original_result["ai_probability"])
    after = float(reduced_result["ai_probability"])
    rewrite_succeeded = bool(rewrite_result.get("rewrite", True))
    quality = rewrite_result.get("quality_score", 55.0) if not rewrite_succeeded else _quality_score(content, reduced_text, before, after)
    _log_task(task_id, "reduce.finish", rewrite=rewrite_succeeded, before=f"{before:.2f}", after=f"{after:.2f}", quality=quality)

    return {
        "reduced": reduced_text,
        "rewrite": rewrite_succeeded,
        "ai_probability": f"{before:.2f}",
        "ai_reduced_probability": f"{after:.2f}",
        "quality_score": quality,
        "model": rewrite_result.get("model", "moderate"),
        "changes": rewrite_result.get("changes", []),
    }
