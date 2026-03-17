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
# 远端二审全局开关，设为 false/0/no 可临时关闭，不影响本地检测流程
REMOTE_REVIEW_ENABLED = os.getenv("REMOTE_REVIEW_ENABLED", "true").lower() not in {"false", "0", "no"}

_azure_client = None
_openrouter_client = None


def _clamp(value: float, low: float, high: float) -> float:
    """将 value 限定在 [low, high] 区间内。"""
    return max(low, min(high, value))


def _default_model_for(source: str) -> str:
    """根据 API 来源返回对应的默认推理模型名称。"""
    if source == "azure":
        return AZURE_DEFAULT_MODEL
    if source == "openrouter":
        return OPENROUTER_DEFAULT_MODEL
    return OLLAMA_DEFAULT_MODEL


def _review_model_for(source: str) -> str:
    """根据 API 来源返回 Pro 二审所用模型，优先读取环境变量覆盖值。"""
    if source == "azure":
        return os.getenv("PRO_REVIEW_MODEL", AZURE_DEFAULT_MODEL)
    if source == "openrouter":
        return os.getenv("PRO_REVIEW_MODEL", OPENROUTER_DEFAULT_MODEL)
    return PRO_REVIEW_MODEL


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


async def _call_remote_json(system_prompt: str, user_prompt: str, model: str | None, api_source: str) -> dict:
    """统一的远端 LLM JSON 调用入口，支持 Azure 和 OpenRouter 两种后端。

    temperature 设为 0.2 以保证输出稳定性，返回值为解析后的 dict。
    OpenRouter 支持 self（直连）和 proxy（代理转发）两种发送模式，通过环境变量控制。
    """
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
            response_format={"type": "json_object"},  # 强制 JSON 输出格式
            temperature=0.2,
        )
        return _extract_json(response.choices[0].message.content)

    if api_source == "openrouter":
        send_mode = os.getenv("OPENROUTER_SEND_MODE", "self").lower()
        if send_mode == "proxy":
            # 走内部代理，适用于无法直连 OpenRouter 的环境
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
    """返回 ReduceRewriteResult 的 JSON Schema，供 Ollama 结构化输出使用。"""
    return ReduceRewriteResult.model_json_schema()


def _remote_review_schema() -> dict:
    """返回 RemoteReviewResult 的 JSON Schema，供 Ollama 结构化输出使用。"""
    return RemoteReviewResult.model_json_schema()


def _build_reduce_prompt(content: str, detection_result: dict) -> str:
    """构建降 AI 改写的用户 Prompt，携带检测结果摘要和高风险分块信息。

    只传入前 5 个高风险分块，避免 Prompt 过长导致超出模型上下文。
    每个分块的 reasons 仅取前 3 条，保持 Prompt 简洁。
    """
    high_risk_chunks = detection_result.get("analysis", {}).get("high_risk_chunks", [])
    summary_reasons = detection_result.get("analysis", {}).get("summary_reasons", [])
    # 只筛选高风险分块，并限制数量为 5，降低 Prompt token 用量
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
    """构建 Pro 远端二审的用户 Prompt，包含文体、文档特征、前 4 个分块及原文。

    结构化为 JSON 字符串，便于模型理解各字段含义并给出更准确的二审意见。
    """
    analysis = detection_result.get("analysis", {})
    doc_features = detection_result.get("document_features", {})
    # 只取前 4 个分块，避免 Prompt 过长；每块 reasons 限 3 条
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


def _map_review_label(label: str) -> str:
    """将二审返回的内部标签（ai/mixed/human）转换为用户可读的英文标签。"""
    if label == "ai":
        return "AI Generated"
    if label == "mixed":
        return "Mixed"
    return "Human Written"


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
    # 二审原因优先排在前面，去重后最多保留 5 条
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
    """调用远端 LLM 进行二审，返回验证后的 RemoteReviewResult dict，失败时返回 None 不影响主流程。"""
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


async def _rewrite_content(content: str, detection_result: dict, model: str | None, api_source: str | None) -> dict:
    """调用 LLM 对原文进行降 AI 改写，返回 ReduceRewriteResult dict。

    Azure/OpenRouter 使用通用 JSON 调用；Ollama 使用结构化 Schema 约束输出格式。
    改写失败时返回原文作为 fallback，quality_score 给低分提示异常。
    """
    source = (api_source or API_SOURCE).lower()
    prompt = _build_reduce_prompt(content, detection_result)
    schema = _rewrite_schema()

    try:
        if source in {"azure", "openrouter"}:
            # 远端模型支持 json_object 格式，不需要显式传 schema
            raw = await _call_remote_json(REDUCE_REWRITE_SYSTEM_PROMPT, prompt, model=model, api_source=source)
        else:
            # Ollama 通过 format 字段传入 JSON Schema 约束输出结构
            raw = await generate_json(REDUCE_REWRITE_SYSTEM_PROMPT, prompt, schema=schema, model=model or OLLAMA_DEFAULT_MODEL)
        return ReduceRewriteResult.model_validate(raw).model_dump()
    except Exception as exc:
        logger.warning("Rewrite failed, falling back to original text: %s", exc)
        # 降级返回原文，quality_score 给 55 表示"未改写"状态
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
    """主检测流程：清洗文本 → 检测文体 → 提取特征 → 分块打分 → 聚合结果 → 可选二审 → 按套餐裁剪。

    Args:
        content: 待检测的原始文本
        model: 指定推理模型，None 时使用各来源默认模型
        api_source: 推理后端来源（ollama/azure/openrouter）
        plan: 订阅套餐（free/pro），影响返回字段和二审权限
        can_remote_review: 是否允许触发 Pro 远端二审

    Returns:
        包含 label、ai_probability、confidence、chunks 等字段的检测结果 dict
    """
    # Step 1: 文本预处理
    cleaned = clean_text(content)
    genre = detect_genre(cleaned)
    doc_features = extract_document_features(cleaned)
    chunks = chunk_text(cleaned)
    logger.info("Running check: genre=%s, model=%s, chunks=%d", genre, model or OLLAMA_DEFAULT_MODEL, len(chunks))

    # Step 2: 逐分块提取特征 + LLM 打分 + 规则综合评分
    enriched_chunks: list[dict] = []
    for chunk in chunks:
        features = extract_chunk_features(chunk["text"])
        # 调用 Qwen 本地模型对分块进行风格判断，失败时自动降级为启发式规则
        qwen_result = await judge_chunk_with_qwen(chunk["text"], genre, features, model=model or OLLAMA_DEFAULT_MODEL)
        # 融合 LLM 分数（55%）、统计特征分（30%）、风格信号分（15%）
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

    # Step 3: 将分块结果聚合为文档级结论
    result = aggregate_document(enriched_chunks, genre, doc_features)
    result["engine"] = {
        "checker": "hybrid-local",
        "qwen_model": model or OLLAMA_DEFAULT_MODEL,
        "api_source": api_source or API_SOURCE,
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
        review_result = await _run_remote_review(cleaned, result)
        if review_result:
            # 将二审分数融合进最终结果
            result = _merge_review_result(result, review_result, review_meta)

    # Step 5: 按套餐裁剪返回字段
    return _trim_result_by_plan(result, plan=plan)


async def run_reduce(
    content: str,
    model: Optional[str] = None,
    api_source: Optional[str] = None,
    plan: str = "pro",
    can_remote_review: bool = False,
) -> dict:
    """降 AI 改写主流程：先检测 → 改写 → 再检测 → 计算质量分。

    流程：
    1. run_check 获取原文 AI 概率及高风险分块信息
    2. _rewrite_content 根据检测结果改写原文
    3. 对改写后的文本再次 run_check，获取降 AI 后的概率
    4. _quality_score 综合内容保留度和降 AI 幅度计算质量分

    Returns:
        包含 reduced（改写文本）、ai_probability（原始概率）、
        ai_reduced_probability（改写后概率）、quality_score 的 dict
    """
    # Step 1: 检测原文，获取 AI 概率和高风险分块
    original_result = await run_check(
        content=content,
        model=model,
        api_source=api_source,
        plan=plan,
        can_remote_review=can_remote_review,
    )
    # Step 2: 根据检测结果生成改写文本
    rewrite_result = await _rewrite_content(content, original_result, model=model, api_source=api_source)
    # Step 3: 清洗改写文本，回退到原文防止改写结果为空
    reduced_text = clean_text(rewrite_result.get("reduced") or content)
    reduced_result = await run_check(
        content=reduced_text,
        model=model,
        api_source=api_source,
        plan=plan,
        can_remote_review=can_remote_review,
    )

    # Step 4: 计算改写前后概率差异和质量综合得分
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
