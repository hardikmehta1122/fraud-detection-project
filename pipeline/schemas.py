"""Pydantic schemas for the extraction pipeline.

Each extracted field carries its parsed value and an extraction-confidence
score, so downstream code can weigh how much to trust it.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

T = TypeVar("T")

VALID_CURRENCIES = {"USD", "CAD", "EUR", "GBP", "AUD"}

VALID_MERCHANT_CATEGORIES = {
    "electronics", "grocery", "travel", "software", "gambling",
    "jewelry", "utilities", "restaurant", "apparel", "crypto_exchange",
}

VALID_PAYMENT_METHODS = {"credit_card", "debit_card", "ach", "wire", "digital_wallet"}


class ExtractedField(BaseModel, Generic[T]):
    """A single extracted value with an extraction-confidence score."""

    value: T
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


class DocumentStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class BoundingBox(BaseModel):
    """Normalized (0-1) bounding box for visual grounding on the source doc."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)


class FinancialDocument(BaseModel):
    """A single extracted transaction record.

    The validation boundary for the pipeline: raw extractor output is
    coerced into this schema at ingestion, before any split or feature work.
    """

    document_id: str
    template: str

    vendor_name: ExtractedField[str]
    account_number: ExtractedField[str]
    transaction_date: ExtractedField[date]
    amount: ExtractedField[float]
    currency: ExtractedField[str]
    merchant_category: ExtractedField[str]
    payment_method: ExtractedField[str]
    billing_country: ExtractedField[str]
    shipping_country: ExtractedField[str]

    account_age_days: int = Field(ge=0)
    documents_last_24h: int = Field(ge=0)
    hour_of_day: int = Field(ge=0, le=23)

    bounding_boxes: dict[str, BoundingBox]

    # Present in the training feed, absent from the live inference feed.
    is_fraud: int | None = None

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: ExtractedField[float]) -> ExtractedField[float]:
        if v.value <= 0:
            raise ValueError("amount must be positive")
        if v.value > 1_000_000:
            raise ValueError("amount exceeds plausible transaction ceiling")
        return v

    @field_validator("currency")
    @classmethod
    def currency_known(cls, v: ExtractedField[str]) -> ExtractedField[str]:
        if v.value not in VALID_CURRENCIES:
            raise ValueError(f"unknown currency code: {v.value}")
        return v

    @field_validator("merchant_category")
    @classmethod
    def category_known(cls, v: ExtractedField[str]) -> ExtractedField[str]:
        if v.value not in VALID_MERCHANT_CATEGORIES:
            raise ValueError(f"unknown merchant category: {v.value}")
        return v

    @field_validator("payment_method")
    @classmethod
    def payment_method_known(cls, v: ExtractedField[str]) -> ExtractedField[str]:
        if v.value not in VALID_PAYMENT_METHODS:
            raise ValueError(f"unknown payment method: {v.value}")
        return v

    @model_validator(mode="after")
    def account_number_shape(self) -> FinancialDocument:
        acct = self.account_number.value
        if not (6 <= len(acct) <= 20):
            raise ValueError("account_number has an implausible length")
        return self


class ReviewDecision(BaseModel):
    decision: DocumentStatus
    analyst_note: str | None = None

    @field_validator("decision")
    @classmethod
    def decision_is_terminal(cls, v: DocumentStatus) -> DocumentStatus:
        if v == DocumentStatus.PENDING:
            raise ValueError("decision must be 'approved' or 'rejected'")
        return v
