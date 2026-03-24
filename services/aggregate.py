from __future__ import annotations

from services.calibration import calibrate_chunk_score, calibrate_document_score


def _clamp(value: float, low: float, high: float) -> float:
    """将 value 限定在 [low, high] 区间内。"""
    return max(low, min(high, value))


def _map_label(score: float) -> str:
    """AI 概率分数 → 可读标签：>= 70 AI Generated，>= 40 Mixed，其余 Human Written。"""
    if score >= 70:
        return "AI Generated"
    if score >= 40:
        return "Mixed"
    return "Human Written"


def _confidence_label(confidence: float) -> str:
    """将 0-1 置信度数值转换为 high/medium/low 三级标签。"""
    if confidence >= 0.72:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _feature_score(features: dict) -> float:
    """基于统计特征计算 AI 概率分数（0-100）。

    各信号含义：
    - uniformity：句长均匀程度（高 → 偏 AI）
    - repeat_signal：n-gram 重复率（高 → 偏 AI）
    - lexical_signal：词汇多样性低（高 → 偏 AI）
    - connector_signal：连接词密度过高（高 → 偏 AI）
    - punctuation_signal：标点密度异常低（高 → 偏 AI）
    - paragraph_signal：段落长度方差小（高 → 偏 AI）
    - human_detail：具体细节多（高 → 偏人类）
    """
    burstiness = features.get("burstiness", 0.0)
    sentence_std = features.get("sentence_length_std", 0.0)
    lexical = features.get("lexical_diversity", 0.0)
    repeated = features.get("repeated_ngram_ratio", 0.0)
    connectors = features.get("connector_density", 0.0)
    punctuation = features.get("punctuation_density", 0.0)
    paragraph_var = features.get("paragraph_length_variance", 0.0)
    detail = features.get("detail_signal_count", 0)

    # burstiness 和 sentence_std 越小，说明句长越均匀，偏 AI 风格
    uniformity = _clamp((1 - burstiness / 0.55) * 0.6 + (1 - sentence_std / 18) * 0.4, 0, 1)
    # 词汇多样性低于 0.62 时信号增强（AI 倾向于重复用词）
    lexical_signal = _clamp((0.62 - lexical) / 0.24, 0, 1)
    repeat_signal = _clamp(repeated / 0.08, 0, 1)
    connector_signal = _clamp((connectors - 0.03) / 0.08, 0, 1)
    punctuation_signal = _clamp((0.05 - punctuation) / 0.04, 0, 1)
    paragraph_signal = _clamp(1 - paragraph_var / 4500, 0, 1)
    # 具体细节是人类写作的强信号，用 (1 - human_detail) 反向参与 AI 分计算
    human_detail = _clamp(detail / 6, 0, 1)

    score = 100 * (
        0.28 * uniformity
        + 0.18 * repeat_signal
        + 0.16 * lexical_signal
        + 0.12 * connector_signal
        + 0.08 * punctuation_signal
        + 0.08 * paragraph_signal
        + 0.10 * (1 - human_detail)
    )

    return score


def _style_score(qwen_result: dict) -> float:
    """从 Qwen 模型的 signals 中提取风格维度的 AI 概率分数（0-100）。

    over_smooth 和 template_pattern 权重最高，是 AI 生成文本的主要风格特征。
    """
    signals = qwen_result.get("signals", {})
    return 100 * (
        0.35 * signals.get("over_smooth", 0.0)
        + 0.30 * signals.get("template_pattern", 0.0)
        + 0.20 * signals.get("sentence_uniformity", 0.0)
        + 0.15 * (1 - signals.get("human_detail", 0.0))
    )


def _triad_percentages(score: float) -> dict:
    """将 AI 概率分数转换为 ai/mixed/human 三方百分比，确保合计为 100。

    使用幂次缩放让高低端分数区间的分布更有区分度，mixed 项吸收四舍五入误差。
    """
    ai_strength = _clamp((score - 45) / 55, 0, 1) ** 1.2
    human_strength = _clamp((55 - score) / 55, 0, 1) ** 1.2
    mixed_strength = max(0.1, 1 - abs(score - 50) / 45)
    total = ai_strength + human_strength + mixed_strength
    raw = {
        "ai": ai_strength / total * 100,
        "mixed": mixed_strength / total * 100,
        "human": human_strength / total * 100,
    }
    rounded = {key: int(round(value)) for key, value in raw.items()}
    diff = 100 - sum(rounded.values())
    if diff:
        rounded["mixed"] += diff
    return rounded


def score_chunk(features: dict, qwen_result: dict, genre: str) -> dict:
    """融合 LLM 打分、统计特征分和风格分，计算单个分块的最终 AI 概率。

    权重组合：Qwen LLM 55% + 统计特征 30% + 风格信号 15%。
    之后按文体、细节数量、列表比例等维度做规则校正，短文本额外降低置信度。
    """
    feature_score = _feature_score(features)
    style_score = _style_score(qwen_result)
    qwen_score = float(qwen_result.get("ai_score", 0.0))

    # 三路分数加权融合
    final_score = 0.55 * qwen_score + 0.30 * feature_score + 0.15 * style_score
    confidence_scale = 1.0

    # 文体惩罚/奖励：正式文体倾向降分（保守），营销文体略微加分
    if genre in {"business_doc", "academic"}:
        final_score -= 8
    elif genre == "list_or_table":
        final_score -= 15
    elif genre == "translation_like":
        final_score -= 6
    elif genre == "marketing":
        final_score += 3

    # 具体细节（日期/数字/人名）是人类写作强信号，每个扣 2.5 分，最多扣 14 分
    detail_count = features.get("detail_signal_count", 0)
    if detail_count >= 3:
        final_score -= min(14, detail_count * 2.5)

    # 列表行占比高时，结构化排版不代表 AI 生成
    if features.get("list_line_ratio", 0.0) > 0.35:
        final_score -= 8

    # 短文本（< 120 字符）特征不稳定，将分数向 50 中心收缩并降低置信度
    if features.get("char_count", 0) < 120:
        final_score = 50 + (final_score - 50) * 0.7
        confidence_scale *= 0.72

    # 单句文本信息量不足，置信度额外下调
    if features.get("sentence_count", 0) <= 1:
        confidence_scale *= 0.85

    final_score = _clamp(calibrate_chunk_score(final_score, genre, features), 0, 100)
    # agreement 衡量 LLM 分与统计特征分的一致性，越一致置信度越高
    agreement = 1 - abs(qwen_score - feature_score) / 100
    confidence = _clamp(
        (
            0.45 * qwen_result.get("confidence", 0.0)
            + 0.25 * min(features.get("char_count", 0) / 260, 1)
            + 0.30 * agreement
        )
        * confidence_scale,
        0,
        1,
    )

    return {
        "ai_score": round(final_score, 2),
        "label": _map_label(final_score),
        "confidence": round(confidence, 4),
        "score_breakdown": {
            "feature_score": round(feature_score, 2),
            "qwen_score": round(qwen_score, 2),
            "style_score": round(style_score, 2),
        },
    }


def aggregate_document(chunks: list[dict], genre: str, doc_features: dict) -> dict:
    """将所有分块的评分聚合为文档级检测结果。

    加权策略：按分块字符数加权平均，长分块对最终分数影响更大。
    文档级额外调整：短文档收缩至中间值，正式文体且分数未达到高风险线时额外降分。
    """
    total_chars = max(doc_features.get("char_count", 0), 1)
    # 按分块长度加权计算文档整体 AI 概率，避免短块权重过大
    weighted_score = sum(chunk["ai_score"] * max(len(chunk.get("text", "")), 1) for chunk in chunks) / sum(
        max(len(chunk.get("text", "")), 1) for chunk in chunks
    )
    avg_confidence = sum(chunk.get("confidence", 0.0) for chunk in chunks) / max(len(chunks), 1)

    document_score = weighted_score
    # 整篇文档字数过少时，判断可靠性下降，向中间值收缩
    if total_chars < 120:
        document_score = 50 + (document_score - 50) * 0.65
        avg_confidence *= 0.7
    # 正式文体在分数未达到 70 时额外降 4 分，执行保守策略
    if genre in {"business_doc", "academic", "list_or_table"} and document_score < 70:
        document_score -= 4

    document_score = _clamp(calibrate_document_score(document_score, genre, doc_features), 0, 100)
    label = _map_label(document_score)
    confidence = _confidence_label(avg_confidence)
    # 高风险分块：ai_score >= 65 的分块 ID 列表，用于指导改写优先级
    high_risk_chunks = [chunk["chunk_id"] for chunk in chunks if chunk.get("ai_score", 0) >= 65]

    # 从各分块聚合 perplexity_proxy 和 binoculars_score（按字符数加权平均）
    ppl_values = [(c.get("perplexity_proxy"), max(len(c.get("text", "")), 1)) for c in chunks if c.get("perplexity_proxy") is not None]
    bino_values = [(c.get("binoculars_score"), max(len(c.get("text", "")), 1)) for c in chunks if c.get("binoculars_score") is not None]
    total_ppl_w = sum(w for _, w in ppl_values) or 1
    total_bino_w = sum(w for _, w in bino_values) or 1
    agg_perplexity = round(sum(v * w for v, w in ppl_values) / total_ppl_w, 2) if ppl_values else doc_features.get("pseudo_perplexity")
    agg_binoculars = round(sum(v * w for v, w in bino_values) / total_bino_w, 4) if bino_values else None

    metrics = {
        "perplexity": agg_perplexity,
        "burstiness": doc_features.get("burstiness"),
        "binoculars": agg_binoculars,
    }

    # 将每个分块映射为句子级结果，level 为 1-10 的可视化热力级别
    sentences = [
        {
            "text": chunk["text"],
            "ai_probability": int(round(chunk["ai_score"])),
            "level": max(1, min(10, int(round(chunk["ai_score"] / 10)))),
        }
        for chunk in chunks
    ]

    ai_probability = f"{document_score:.2f}"

    return {
        "label": label,
        "ai_probability": ai_probability,
        "confidence": confidence,
        "document_features": doc_features,
        "chunks": chunks,
        "analysis": {
            "high_risk_chunks": high_risk_chunks,
            "genre": genre,
            "language": doc_features.get("language"),
        },
        "summary": {
            "confidence_label": label,
            "percentages": _triad_percentages(document_score),
        },
        "metrics": metrics,
        "sentences": sentences,
        "vocab": {
            "count": len(doc_features.get("suspicious_terms", [])[:10]),
            "words": doc_features.get("suspicious_terms", [])[:10],
        },
    }
