"""Pydantic models for DealMemory."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Scenario = Literal["cold_call", "demo", "spec_alignment", "budget_discussion", "handover"]
Verdict = Literal["correct", "partial", "wrong"]
FIELD_KEYS: tuple[str, ...] = (
    "budget",
    "decision_maker",
    "technical_requirements",
    "objections",
    "promises",
    "next_step",
)


class FieldEvidence(BaseModel):
    """Single extracted fact with evidence and confidence.

    `value=None` means "not mentioned in the transcript" — required state
    to keep hallucination honest.
    """

    value: str | None = None
    quote: str | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class DealFacts(BaseModel):
    """Six deal fields extracted (or annotated) per dialog."""

    budget: FieldEvidence = Field(default_factory=FieldEvidence)
    decision_maker: FieldEvidence = Field(default_factory=FieldEvidence)
    technical_requirements: FieldEvidence = Field(default_factory=FieldEvidence)
    objections: list[FieldEvidence] = Field(default_factory=list)
    promises: list[FieldEvidence] = Field(default_factory=list)
    next_step: FieldEvidence = Field(default_factory=FieldEvidence)


class Dialog(BaseModel):
    """A transcript-only dialog in a pool. No ground truth — only annotations."""

    id: str
    scenario: Scenario | None = None
    transcript: str
    meta: dict = Field(default_factory=dict)


class Prediction(BaseModel):
    """Extractor output for one transcript."""

    id: str
    prediction: DealFacts
    raw_response: str | None = None
    parse_repaired: bool = False
    extracted_at: datetime = Field(default_factory=datetime.utcnow)


class Annotation(BaseModel):
    """User-provided verdict on a single (dialog, field) prediction.

    `field` is one of FIELD_KEYS; verdict is the manual judgement on
    whether the extractor's prediction for that field is correct/partial/wrong.
    """

    dialog_id: str
    field: str
    verdict: Verdict
    annotated_at: datetime = Field(default_factory=datetime.utcnow)
