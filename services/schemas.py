from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QwenSignals(BaseModel):
    over_smooth: float = Field(..., ge=0, le=1)
    template_pattern: float = Field(..., ge=0, le=1)
    sentence_uniformity: float = Field(..., ge=0, le=1)
    human_detail: float = Field(..., ge=0, le=1)


class QwenJudgeResult(BaseModel):
    ai_score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    signals: QwenSignals
    label: Literal["human", "mixed", "ai"]
    perplexity_proxy: float = Field(default=100.0, ge=1, le=2000)
    binoculars_score: float = Field(default=0.5, ge=0, le=1)


class ReduceChange(BaseModel):
    original: str
    revised: str
    reason: str


class ReduceRewriteResult(BaseModel):
    reduced: str
    ai_probability: str = "0.00"
    ai_reduced_probability: str = "0.00"
    quality_score: float = Field(default=0, ge=0, le=100)
    model: Literal["light", "moderate", "deep"] = "moderate"
    changes: list[ReduceChange] = Field(default_factory=list)


class ChunkSpan(BaseModel):
    chunk_id: int
    start: int
    end: int
    text: str


class RemoteReviewResult(BaseModel):
    ai_score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    label: Literal["human", "mixed", "ai"]
