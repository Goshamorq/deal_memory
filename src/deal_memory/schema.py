"""Pydantic models for DealMemory."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Scenario = Literal["cold_call", "demo", "spec_alignment", "budget_discussion"]


class FieldEvidence(BaseModel):
    """Single extracted fact with evidence and confidence.

    `value=None` means "not mentioned in the transcript" — required state
    to keep hallucination rate honest.
    """

    value: str | None = None
    quote: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class DealFacts(BaseModel):
    """Six deal fields extracted (or labeled) per dialog."""

    budget: FieldEvidence = Field(default_factory=FieldEvidence)
    decision_maker: FieldEvidence = Field(default_factory=FieldEvidence)
    technical_requirements: FieldEvidence = Field(default_factory=FieldEvidence)
    objections: list[FieldEvidence] = Field(default_factory=list)
    promises: list[FieldEvidence] = Field(default_factory=list)
    next_step: FieldEvidence = Field(default_factory=FieldEvidence)


class SyntheticSample(BaseModel):
    """One labeled synthetic dialog."""

    id: str
    scenario: Scenario
    transcript: str
    ground_truth: DealFacts
    meta: dict = Field(default_factory=dict)


class Prediction(BaseModel):
    """Extractor output for one transcript."""

    id: str
    prediction: DealFacts
    raw_response: str | None = None
    parse_repaired: bool = False
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
