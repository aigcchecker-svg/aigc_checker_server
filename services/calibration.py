from __future__ import annotations


def calibrate_chunk_score(score: float, genre: str, features: dict) -> float:
    """第一版先保留 identity mapping，后续可接人工标注校准。"""
    return score


def calibrate_document_score(score: float, genre: str, doc_features: dict) -> float:
    """第一版先保留 identity mapping，后续可按文体和样本分布做温度缩放。"""
    return score

