from __future__ import annotations

from collections import Counter

from services.calibration import calibrate_chunk_score, calibrate_document_score


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _map_label(score: float) -> str:
    if score >= 70:
        return "AI Generated"
    if score >= 40:
        return "Mixed"
    return "Human Written"


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.72:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _feature_score(features: dict) -> tuple[float, list[str]]:
    burstiness = features.get("burstiness", 0.0)
    sentence_std = features.get("sentence_length_std", 0.0)
    lexical = features.get("lexical_diversity", 0.0)
    repeated = features.get("repeated_ngram_ratio", 0.0)
    connectors = features.get("connector_density", 0.0)
    punctuation = features.get("punctuation_density", 0.0)
    paragraph_var = features.get("paragraph_length_variance", 0.0)
    detail = features.get("detail_signal_count", 0)

    uniformity = _clamp((1 - burstiness / 0.55) * 0.6 + (1 - sentence_std / 18) * 0.4, 0, 1)
    lexical_signal = _clamp((0.62 - lexical) / 0.24, 0, 1)
    repeat_signal = _clamp(repeated / 0.08, 0, 1)
    connector_signal = _clamp((connectors - 0.03) / 0.08, 0, 1)
    punctuation_signal = _clamp((0.05 - punctuation) / 0.04, 0, 1)
    paragraph_signal = _clamp(1 - paragraph_var / 4500, 0, 1)
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

    reasons: list[str] = []
    if uniformity > 0.72:
        reasons.append("句长波动较小，整体节奏偏均匀。")
    if repeat_signal > 0.45:
        reasons.append("重复片段比例偏高，存在模板化复用迹象。")
    if lexical_signal > 0.55:
        reasons.append("词汇变化度偏低，表达较集中。")
    if human_detail > 0.45:
        reasons.append("检测到较多具体时间、数字或执行约束，偏向人工写作。")

    return score, reasons


def _style_score(qwen_result: dict) -> float:
    signals = qwen_result.get("signals", {})
    return 100 * (
        0.35 * signals.get("over_smooth", 0.0)
        + 0.30 * signals.get("template_pattern", 0.0)
        + 0.20 * signals.get("sentence_uniformity", 0.0)
        + 0.15 * (1 - signals.get("human_detail", 0.0))
    )


def _triad_percentages(score: float) -> dict:
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
    feature_score, feature_reasons = _feature_score(features)
    style_score = _style_score(qwen_result)
    qwen_score = float(qwen_result.get("ai_score", 0.0))

    final_score = 0.55 * qwen_score + 0.30 * feature_score + 0.15 * style_score
    confidence_scale = 1.0

    if genre in {"business_doc", "academic"}:
        final_score -= 8
    elif genre == "list_or_table":
        final_score -= 15
    elif genre == "translation_like":
        final_score -= 6
    elif genre == "marketing":
        final_score += 3

    detail_count = features.get("detail_signal_count", 0)
    if detail_count >= 3:
        final_score -= min(14, detail_count * 2.5)

    if features.get("list_line_ratio", 0.0) > 0.35:
        final_score -= 8

    if features.get("char_count", 0) < 120:
        final_score = 50 + (final_score - 50) * 0.7
        confidence_scale *= 0.72

    if features.get("sentence_count", 0) <= 1:
        confidence_scale *= 0.85

    final_score = _clamp(calibrate_chunk_score(final_score, genre, features), 0, 100)
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

    reasons = list(dict.fromkeys(qwen_result.get("reasons", []) + feature_reasons))
    if genre in {"business_doc", "academic", "list_or_table"}:
        reasons.append("当前文体天然偏正式，规则层已额外降低误判。")
    if features.get("char_count", 0) < 120:
        reasons.append("分块较短，本段判断置信度已自动下调。")

    return {
        "ai_score": round(final_score, 2),
        "label": _map_label(final_score),
        "confidence": round(confidence, 4),
        "reasons": reasons[:6],
        "score_breakdown": {
            "feature_score": round(feature_score, 2),
            "qwen_score": round(qwen_score, 2),
            "style_score": round(style_score, 2),
        },
    }


def aggregate_document(chunks: list[dict], genre: str, doc_features: dict) -> dict:
    total_chars = max(doc_features.get("char_count", 0), 1)
    weighted_score = sum(chunk["ai_score"] * max(len(chunk.get("text", "")), 1) for chunk in chunks) / sum(
        max(len(chunk.get("text", "")), 1) for chunk in chunks
    )
    avg_confidence = sum(chunk.get("confidence", 0.0) for chunk in chunks) / max(len(chunks), 1)

    document_score = weighted_score
    if total_chars < 120:
        document_score = 50 + (document_score - 50) * 0.65
        avg_confidence *= 0.7
    if genre in {"business_doc", "academic", "list_or_table"} and document_score < 70:
        document_score -= 4

    document_score = _clamp(calibrate_document_score(document_score, genre, doc_features), 0, 100)
    label = _map_label(document_score)
    confidence = _confidence_label(avg_confidence)
    high_risk_chunks = [chunk["chunk_id"] for chunk in chunks if chunk.get("ai_score", 0) >= 65]

    reason_counter = Counter()
    for chunk in chunks:
        for reason in chunk.get("reasons", []):
            reason_counter[reason] += 1
    summary_reasons = [reason for reason, _ in reason_counter.most_common(5)]

    metrics = {
        "perplexity": doc_features.get("pseudo_perplexity"),
        "burstiness": doc_features.get("burstiness"),
        "binoculars": None,
    }

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
            "summary_reasons": summary_reasons,
            "genre": genre,
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

