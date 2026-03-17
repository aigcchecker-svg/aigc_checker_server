from __future__ import annotations

import logging

from services.ollama_client import generate_json
from services.prompts import QWEN_CHUNK_SYSTEM_PROMPT
from services.schemas import QwenJudgeResult


logger = logging.getLogger(__name__)


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
    reasons = []
    if over_smooth > 0.7:
        reasons.append("句式节奏偏平滑，语言行为较稳定。")
    if template_pattern > 0.45:
        reasons.append("重复 n-gram 偏多，存在模板化痕迹。")
    if human_detail > 0.45:
        reasons.append("文本包含较多真实细节，降低纯 AI 判定。")
    if genre in {"business_doc", "academic", "list_or_table"}:
        reasons.append("当前文体本身较规整，按保守策略避免误判。")
    if not reasons:
        reasons.append("未观察到足够强的单侧证据，结果偏保守。")

    return {
        "ai_score": round(score, 2),
        "confidence": 0.38,  # 启发式兜底固定低置信度，提示调用方结果仅供参考
        "signals": {
            "over_smooth": round(over_smooth, 4),
            "template_pattern": round(template_pattern, 4),
            "sentence_uniformity": round(sentence_uniformity, 4),
            "human_detail": round(human_detail, 4),
        },
        "reasons": reasons[:4],
        "label": label,
    }


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
    schema = QwenJudgeResult.model_json_schema()
    user_prompt = f"""
请分析下面这个文本分块，只基于风格和语言行为判断 AI 倾向。

文体: {genre}
程序特征摘要:
- char_count: {features.get("char_count")}
- sentence_count: {features.get("sentence_count")}
- avg_sentence_length: {features.get("avg_sentence_length")}
- sentence_length_std: {features.get("sentence_length_std")}
- burstiness: {features.get("burstiness")}
- lexical_diversity: {features.get("lexical_diversity")}
- repeated_ngram_ratio: {features.get("repeated_ngram_ratio")}
- punctuation_density: {features.get("punctuation_density")}
- connector_density: {features.get("connector_density")}
- paragraph_length_variance: {features.get("paragraph_length_variance")}
- detail_signal_count: {features.get("detail_signal_count")}
- list_line_ratio: {features.get("list_line_ratio")}

文本分块:
{chunk_text}
""".strip()

    try:
        data = await generate_json(
            system_prompt=QWEN_CHUNK_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=schema,
            model=model,
        )
        # 用 Pydantic 校验模型返回格式，字段缺失或类型不符时会抛出 ValidationError
        validated = QwenJudgeResult.model_validate(data)
        return validated.model_dump()
    except Exception as exc:
        # LLM 调用失败（超时、格式错误等），降级为启发式规则，记录警告日志
        logger.warning("Qwen judge fallback triggered: %s", exc)
        return _heuristic_fallback(features, genre)

