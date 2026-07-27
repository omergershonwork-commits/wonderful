"""Validated domain and tool-boundary models for airport intelligence.

These models describe normalized aviation data and structured analytical results.
They intentionally contain no scoring or metric-calculation logic.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DomainModel(BaseModel):
    """Base model with strict boundary behavior shared by all domain records."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class DataMode(StrEnum):
    """Visible provenance mode used for every analytical response."""

    LIVE_PUBLIC_DATA = "LIVE PUBLIC DATA"
    CACHED_PUBLIC_DATA = "CACHED PUBLIC DATA"
    ILLUSTRATIVE_DEMO_DATA = "ILLUSTRATIVE DEMO DATA"


class ConfidenceLevel(StrEnum):
    """Human-readable confidence classification for deterministic results."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNAVAILABLE = "UNAVAILABLE"


class MetricUnit(StrEnum):
    """Supported display units for calculated metrics."""

    RATIO = "ratio"
    PERCENT = "percent"
    SCORE = "score_0_100"
    COUNT = "count"
    MINUTES = "minutes"
    MILES = "miles"
    DEPARTURES_PER_RUNWAY = "departures_per_runway"


def _normalize_airport_code(value: str) -> str:
    code = value.strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError("airport code must contain exactly three letters")
    return code


def _normalize_state_code(value: str) -> str:
    code = value.strip().upper()
    if len(code) != 2 or not code.isalpha():
        raise ValueError("state code must contain exactly two letters")
    return code


def _normalize_airport_codes(values: list[str]) -> list[str]:
    normalized = [_normalize_airport_code(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError("airport codes must be unique")
    return normalized


class DataPeriod(DomainModel):
    """Inclusive period represented by normalized public or fixture data."""

    start_date: date
    end_date: date
    label: str | None = None

    @model_validator(mode="after")
    def validate_date_order(self) -> "DataPeriod":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        return self


class SourceMetadata(DomainModel):
    """Provenance attached to normalized records and analytical responses."""

    source_name: str = Field(min_length=1)
    data_mode: DataMode
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    period: DataPeriod | None = None
    source_url: str | None = None
    notes: list[str] = Field(default_factory=list)


class ConfidenceInfo(DomainModel):
    """Deterministic confidence result and its explicit limitations."""

    level: ConfidenceLevel
    score: float = Field(ge=0, le=1)
    missing_fields: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unavailable_confidence(self) -> "ConfidenceInfo":
        if self.level is ConfidenceLevel.UNAVAILABLE and self.score != 0:
            raise ValueError("UNAVAILABLE confidence must have a score of 0")
        return self


class AirportRecord(DomainModel):
    """Normalized airport metadata independent of any public-source schema."""

    airport_code: str
    name: str = Field(min_length=1)
    city: str = Field(min_length=1)
    state_code: str
    region: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    usable_runway_count: int | None = Field(default=None, ge=1)
    source: SourceMetadata

    @field_validator("airport_code", mode="before")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalize_airport_code(value)

    @field_validator("state_code", mode="before")
    @classmethod
    def validate_state_code(cls, value: str) -> str:
        return _normalize_state_code(value)


class TrafficRecord(DomainModel):
    """Airport-level passenger and seat totals for one analysis period."""

    airport_code: str
    period: DataPeriod
    passengers: int = Field(ge=0)
    previous_period_passengers: int | None = Field(default=None, ge=0)
    available_seats: int | None = Field(default=None, ge=0)
    source: SourceMetadata

    @field_validator("airport_code", mode="before")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalize_airport_code(value)


class RouteRecord(DomainModel):
    """Normalized directional route activity used for long-haul analysis."""

    origin_airport_code: str
    destination_airport_code: str
    destination_name: str | None = None
    distance_miles: int = Field(ge=0)
    departures: int = Field(ge=0)
    passengers: int = Field(ge=0)
    available_seats: int | None = Field(default=None, ge=0)
    period: DataPeriod
    source: SourceMetadata

    @field_validator(
        "origin_airport_code",
        "destination_airport_code",
        mode="before",
    )
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalize_airport_code(value)

    @model_validator(mode="after")
    def validate_distinct_airports(self) -> "RouteRecord":
        if self.origin_airport_code == self.destination_airport_code:
            raise ValueError("route origin and destination must differ")
        return self


class OperationalData(DomainModel):
    """Normalized operational indicators for congestion calculations."""

    airport_code: str
    period: DataPeriod
    scheduled_departures: int = Field(ge=0)
    performed_departures: int = Field(ge=0)
    reported_cancellations: int | None = Field(default=None, ge=0)
    average_departure_delay_minutes: float | None = Field(default=None, ge=0)
    average_taxi_out_minutes: float | None = Field(default=None, ge=0)
    usable_runway_count: int | None = Field(default=None, ge=1)
    source: SourceMetadata

    @field_validator("airport_code", mode="before")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalize_airport_code(value)

    @model_validator(mode="after")
    def validate_departure_counts(self) -> "OperationalData":
        if self.performed_departures > self.scheduled_departures:
            raise ValueError(
                "performed_departures cannot exceed scheduled_departures"
            )
        if (
            self.reported_cancellations is not None
            and self.reported_cancellations > self.scheduled_departures
        ):
            raise ValueError(
                "reported_cancellations cannot exceed scheduled_departures"
            )
        return self


class MetricValue(DomainModel):
    """One calculated value with unit and availability information."""

    name: str = Field(min_length=1)
    value: float | int | None
    unit: MetricUnit
    available: bool = True
    missing_reason: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> "MetricValue":
        if self.available and self.value is None:
            raise ValueError("available metrics must contain a value")
        if not self.available and self.value is not None:
            raise ValueError("unavailable metrics cannot contain a value")
        if not self.available and not self.missing_reason:
            raise ValueError("unavailable metrics require a missing_reason")
        return self


class CalculatedMetrics(DomainModel):
    """Deterministically calculated airport metrics; values may be unavailable."""

    passenger_growth: float | None = None
    load_factor: float | None = Field(default=None, ge=0)
    completion_rate: float | None = Field(default=None, ge=0, le=1)
    cancellation_rate: float | None = Field(default=None, ge=0, le=1)
    departures_per_runway: float | None = Field(default=None, ge=0)
    congestion_score: float | None = Field(default=None, ge=0, le=100)
    investment_opportunity_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    estimated_unmet_capacity_proxy: float | None = Field(default=None, ge=0)
    market_scale: float | None = Field(default=None, ge=0)
    missing_components: list[str] = Field(default_factory=list)


class RankAirportsInput(DomainModel):
    """Validated input for ``rank_airports``."""

    region: str | None = None
    state_codes: list[str] | None = None
    limit: int = Field(default=5, ge=1, le=10)
    excluded_airports: list[str] = Field(default_factory=list)

    @field_validator("state_codes", mode="before")
    @classmethod
    def validate_state_codes(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [_normalize_state_code(item) for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("state codes must be unique")
        return normalized

    @field_validator("excluded_airports", mode="before")
    @classmethod
    def validate_excluded_airports(cls, value: list[str]) -> list[str]:
        return _normalize_airport_codes(value)


class CompareAirportsInput(DomainModel):
    """Validated input for ``compare_airports``."""

    airport_codes: list[str] = Field(min_length=2, max_length=10)
    metrics: list[str] | None = None

    @field_validator("airport_codes", mode="before")
    @classmethod
    def validate_airport_codes(cls, value: list[str]) -> list[str]:
        return _normalize_airport_codes(value)


class CalculateLongHaulShareInput(DomainModel):
    """Validated input for ``calculate_long_haul_share``."""

    airport_code: str
    threshold_miles: int = Field(default=3000, gt=0)

    @field_validator("airport_code", mode="before")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalize_airport_code(value)


class EstimateUnmetCapacityInput(DomainModel):
    """Validated input for ``estimate_unmet_capacity``."""

    airport_code: str
    target_load_factor: float = Field(default=0.82, gt=0, le=1)

    @field_validator("airport_code", mode="before")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalize_airport_code(value)


class GetAirportProfileInput(DomainModel):
    """Validated input for ``get_airport_profile``."""

    airport_code: str

    @field_validator("airport_code", mode="before")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalize_airport_code(value)


class AirportAnalysis(DomainModel):
    """Reusable structured analysis for one airport."""

    airport: AirportRecord
    metrics: CalculatedMetrics
    confidence: ConfidenceInfo
    sources: list[SourceMetadata]
    assumptions: list[str] = Field(default_factory=list)


class RankedAirport(DomainModel):
    """One deterministic ranking row."""

    rank: int = Field(ge=1)
    analysis: AirportAnalysis
    recommendation: str


class RankAirportsOutput(DomainModel):
    """Structured output from ``rank_airports``."""

    input: RankAirportsInput
    results: list[RankedAirport]
    data_mode: DataMode
    period: DataPeriod
    assumptions: list[str] = Field(default_factory=list)


class CompareAirportsOutput(DomainModel):
    """Structured output from ``compare_airports``."""

    input: CompareAirportsInput
    airports: list[AirportAnalysis]
    data_mode: DataMode
    period: DataPeriod
    assumptions: list[str] = Field(default_factory=list)


class QualifyingRoute(DomainModel):
    """Route contribution included in a long-haul result."""

    destination_airport_code: str
    destination_name: str | None = None
    distance_miles: int = Field(ge=0)
    departures: int = Field(ge=0)
    passengers: int = Field(ge=0)

    @field_validator("destination_airport_code", mode="before")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalize_airport_code(value)


class LongHaulShareOutput(DomainModel):
    """Structured output from ``calculate_long_haul_share``."""

    input: CalculateLongHaulShareInput
    long_haul_departures: int = Field(ge=0)
    all_departures: int = Field(ge=0)
    departure_share: float = Field(ge=0, le=1)
    long_haul_passengers: int = Field(ge=0)
    all_route_passengers: int = Field(ge=0)
    passenger_share: float = Field(ge=0, le=1)
    qualifying_routes: list[QualifyingRoute]
    confidence: ConfidenceInfo
    data_mode: DataMode
    period: DataPeriod
    assumptions: list[str] = Field(default_factory=list)


class UnmetCapacityBreakdown(DomainModel):
    """Deterministic components of the estimated unmet-capacity proxy."""

    current_passengers: int = Field(ge=0)
    current_available_seats: int = Field(ge=0)
    raw_passenger_growth: float
    clamped_passenger_growth: float = Field(ge=-0.10, le=0.20)
    projected_passengers: float = Field(ge=0)
    target_load_factor: float = Field(gt=0, le=1)
    required_seats: float = Field(ge=0)
    estimated_unmet_capacity_proxy: float = Field(ge=0)


class UnmetCapacityOutput(DomainModel):
    """Structured output from ``estimate_unmet_capacity``."""

    input: EstimateUnmetCapacityInput
    breakdown: UnmetCapacityBreakdown
    airport: AirportRecord
    confidence: ConfidenceInfo
    data_mode: DataMode
    period: DataPeriod
    assumptions: list[str] = Field(default_factory=list)


class AirportProfileOutput(DomainModel):
    """Structured output from ``get_airport_profile``."""

    input: GetAirportProfileInput
    analysis: AirportAnalysis
    data_mode: DataMode
    period: DataPeriod


class UnavailableDataResponse(DomainModel):
    """Typed response returned when no validated source mode can satisfy a request."""

    tool_name: str = Field(min_length=1)
    message: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    airport_code: str | None = None
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("airport_code", mode="before")
    @classmethod
    def validate_airport_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_airport_code(value)
