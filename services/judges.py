from __future__ import annotations

import logging

from services.ollama_client import generate_json
from services.prompts import QWEN_CHUNK_SYSTEM_PROMPT
from services.schemas import QwenJudgeResult


logger = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _heuristic_fallback(features: dict, genre: str) -> dict:
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

    if genre in {"business_doc", "academic", "list_or_table"}:
        score -= 8
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
        "confidence": 0.38,
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
        validated = QwenJudgeResult.model_validate(data)
        return validated.model_dump()
    except Exception as exc:
        logger.warning("Qwen judge fallback triggered: %s", exc)
        return _heuristic_fallback(features, genre)

