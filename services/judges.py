from __future__ import annotations

import os
import logging
import json

from services.ollama_client import generate_json
from services.prompts import QWEN_CHUNK_SYSTEM_PROMPT
from services.schemas import QwenJudgeResult


logger = logging.getLogger(__name__)

LOCAL_JUDGE_SKIP_SHORT_CHARS = int(os.getenv("LOCAL_JUDGE_SKIP_SHORT_CHARS", "170"))
LOCAL_JUDGE_SKIP_LIST_RATIO = float(os.getenv("LOCAL_JUDGE_SKIP_LIST_RATIO", "0.55"))
LOCAL_JUDGE_SKIP_DETAIL_COUNT = int(os.getenv("LOCAL_JUDGE_SKIP_DETAIL_COUNT", "5"))

_QWEN_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "ai_score": {"type": "number", "minimum": 0, "maximum": 100},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "signals": {
            "type": "object",
            "properties": {
                "over_smooth": {"type": "number", "minimum": 0, "maximum": 1},
                "template_pattern": {"type": "number", "minimum": 0, "maximum": 1},
                "sentence_uniformity": {"type": "number", "minimum": 0, "maximum": 1},
                "human_detail": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["over_smooth", "template_pattern", "sentence_uniformity", "human_detail"],
            "additionalProperties": False,
        },
        "label": {"type": "string", "enum": ["human", "mixed", "ai"]},
    },
    "required": ["ai_score", "confidence", "signals", "label"],
    "additionalProperties": False,
}


def _clamp(value: float, low: float, high: float) -> float:
    """将 value 限定在 [low, high] 区间内。"""
    return max(low, min(high, value))


def _heuristic_fallback(features: dict, genre: str) -> dict:
    """当 LLM 调用失败时的纯统计启发式兜底方案，仅使用文本特征计算 AI 概率。

    使用与 aggregate._feature_score 类似的信号，但权重略有调整，
    confidence 固定为 0.38（低置信度），以提示调用方结果可靠性较低。
    """
    burstiness = features.get("burstiness", 0.0)
    sentence_std = features.get("sentence_length_std", 0.0)
    repeated = features.get("repeated_ngram_ratio", 0.0)
    lexical = features.get("lexical_diversity", 0.0)
    detail = features.get("detail_signal_count", 0)

    over_smooth = _clamp(1 - burstiness / 0.55, 0, 1)
    template_pattern = _clamp(repeated / 0.08, 0, 1)
    sentence_uniformity = _clamp(1 - sentence_std / 18, 0, 1)
    human_detail = _clamp(detail / 6, 0, 1)
    lexical_signal = _clamp((0.62 - lexical) / 0.25, 0, 1)

    score = 100 * (
        0.28 * over_smooth
        + 0.28 * template_pattern
        + 0.18 * sentence_uniformity
        + 0.16 * lexical_signal
        + 0.10 * (1 - human_detail)
    )

    # 正式文体保守降分，与 aggregate 层策略保持一致
    if genre in {"business_doc", "academic", "list_or_table"}:
        score -= 8
    # 具体细节越多，降分越多（最多 -12）
    if detail >= 3:
        score -= min(12, detail * 2)

    score = _clamp(score, 0, 100)
    label = "ai" if score >= 70 else "mixed" if score >= 40 else "human"
    return {
        "ai_score": round(score, 2),
        "confidence": 0.38,  # 启发式兜底固定低置信度，提示调用方结果仅供参考
        "signals": {
            "over_smooth": round(over_smooth, 4),
            "template_pattern": round(template_pattern, 4),
            "sentence_uniformity": round(sentence_uniformity, 4),
            "human_detail": round(human_detail, 4),
        },
        "label": label,
        "judge_mode": "heuristic_fallback",
    }


def _should_skip_qwen(features: dict, genre: str) -> tuple[bool, str]:
    """对低信息量或结构化强的分块直接走启发式，减少本地模型调用。"""
    char_count = int(features.get("char_count", 0) or 0)
    sentence_count = int(features.get("sentence_count", 0) or 0)
    list_ratio = float(features.get("list_line_ratio", 0.0) or 0.0)
    detail_count = int(features.get("detail_signal_count", 0) or 0)
    repeated = float(features.get("repeated_ngram_ratio", 0.0) or 0.0)

    if char_count <= LOCAL_JUDGE_SKIP_SHORT_CHARS or sentence_count <= 1:
        return True, "short_or_single_sentence"
    if genre == "list_or_table" or list_ratio >= LOCAL_JUDGE_SKIP_LIST_RATIO:
        return True, "structured_chunk"
    if genre in {"business_doc", "academic"} and detail_count >= LOCAL_JUDGE_SKIP_DETAIL_COUNT and repeated < 0.035:
        return True, "detail_heavy_formal"
    return False, ""


async def judge_chunk_with_qwen(
    chunk_text: str,
    genre: str,
    features: dict,
    model: str | None = None,
) -> dict:
    """调用 Qwen（本地 Ollama）对单个分块进行风格判断，返回 QwenJudgeResult dict。

    将预提取的统计特征作为上下文随 Prompt 一起发送，帮助模型更准确地判断风格倾向。
    若 LLM 调用失败（超时/格式异常等），自动降级为 _heuristic_fallback，不影响主流程。
    """
    compact_features = {
        "c": features.get("char_count"),
        "s": features.get("sentence_count"),
        "asl": features.get("avg_sentence_length"),
        "std": features.get("sentence_length_std"),
        "b": features.get("burstiness"),
        "lex": features.get("lexical_diversity"),
        "rep": features.get("repeated_ngram_ratio"),
        "conn": features.get("connector_density"),
        "detail": features.get("detail_signal_count"),
        "list": features.get("list_line_ratio"),
    }
    user_prompt = (
        f"文体={genre}\n"
        f"特征={json.dumps(compact_features, ensure_ascii=False, separators=(',', ':'))}\n"
        f"文本={chunk_text}\n"
        "只输出 JSON，不要解释，不要 reasons，不要 markdown。"
    )

    should_skip, skip_reason = _should_skip_qwen(features, genre)
    if should_skip:
        heuristic = _heuristic_fallback(features, genre)
        heuristic["judge_mode"] = "heuristic_skip"
        heuristic["judge_skip_reason"] = skip_reason
        return heuristic

    try:
        data = await generate_json(
            system_prompt=QWEN_CHUNK_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=_QWEN_JUDGE_SCHEMA,
            model=model,
        )
        # 用 Pydantic 校验模型返回格式，字段缺失或类型不符时会抛出 ValidationError
        validated = QwenJudgeResult.model_validate(data)
        result = validated.model_dump()
        result["judge_mode"] = "qwen"
        return result
    except Exception as exc:
        # LLM 调用失败（超时、格式错误等），降级为启发式规则，记录警告日志
        logger.warning("Qwen judge fallback triggered: %s", exc)
        return _heuristic_fallback(features, genre)
