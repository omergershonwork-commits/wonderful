"""Validated domain and tool-boundary models for airport intelligence.

These models describe normalized aviation data and structured analytical results.
They intentionally contain no scoring or metric-calculation logic.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


US_STATE_CODES = frozenset(
    {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "DC",
    }
)
NEW_ENGLAND_STATE_CODES = ("CT", "ME", "MA", "NH", "RI", "VT")


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


class RegionName(StrEnum):
    """Regions explicitly supported by the current normalized dataset."""

    NEW_ENGLAND = "New England"
    WEST = "West"
    ALASKA = "Alaska"


class ComparisonMetric(StrEnum):
    """Approved metrics accepted by the airport-comparison tool."""

    PASSENGER_GROWTH = "passenger_growth"
    LOAD_FACTOR = "load_factor"
    COMPLETION_RATE = "completion_rate"
    CANCELLATION_RATE = "cancellation_rate"
    DEPARTURES_PER_RUNWAY = "departures_per_runway"
    DEPARTURE_DELAY = "average_departure_delay_minutes"
    TAXI_OUT = "average_taxi_out_minutes"
    CONGESTION_SCORE = "congestion_score"
    INVESTMENT_OPPORTUNITY_SCORE = "investment_opportunity_score"
    ESTIMATED_UNMET_CAPACITY_PROXY = "estimated_unmet_capacity_proxy"
    MARKET_SCALE = "market_scale"


class RecommendationBand(StrEnum):
    """Deterministic interpretation bands for the opportunity score."""

    STRONG = "Strong candidate for deeper diligence"
    POTENTIAL = "Potential candidate"
    MIXED = "Mixed evidence"
    WEAK = "Weak current expansion signal"


def _normalize_airport_code(value: str) -> str:
    code = value.strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError("airport code must contain exactly three letters")
    return code


def _normalize_state_code(value: str) -> str:
    code = value.strip().upper()
    if code not in US_STATE_CODES:
        raise ValueError("state code must be a valid US state or District of Columbia code")
    return code


def _normalize_region(value: str | RegionName) -> RegionName:
    if isinstance(value, RegionName):
        return value
    normalized = " ".join(value.strip().split()).casefold()
    aliases = {region.value.casefold(): region for region in RegionName}
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported region: {value}") from exc


def _normalize_airport_codes(values: list[str]) -> list[str]:
    normalized = [_normalize_airport_code(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError("airport codes must be unique")
    return normalized


def _periods_are_comparable(current: "DataPeriod", previous: "DataPeriod") -> bool:
    """Return whether periods cover the same calendar window in different years."""

    return (
        (current.start_date.month, current.start_date.day)
        == (previous.start_date.month, previous.start_date.day)
        and (current.end_date.month, current.end_date.day)
        == (previous.end_date.month, previous.end_date.day)
    )


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
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
    region: RegionName | None = None
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

    @field_validator("region", mode="before")
    @classmethod
    def validate_region(cls, value: str | RegionName | None) -> RegionName | None:
        return None if value is None else _normalize_region(value)

    @model_validator(mode="after")
    def validate_region_membership(self) -> "AirportRecord":
        is_new_england_state = self.state_code in NEW_ENGLAND_STATE_CODES
        if self.region is RegionName.NEW_ENGLAND and not is_new_england_state:
            raise ValueError("New England airports must be in CT, ME, MA, NH, RI, or VT")
        if is_new_england_state and self.region not in {None, RegionName.NEW_ENGLAND}:
            raise ValueError("New England state airports cannot use another region")
        return self


class TrafficRecord(DomainModel):
    """Airport-level passenger and seat totals for comparable analysis periods."""

    airport_code: str
    period: DataPeriod
    passengers: int = Field(ge=0)
    previous_period_passengers: int | None = Field(default=None, ge=0)
    previous_period: DataPeriod | None = None
    previous_source: SourceMetadata | None = None
    available_seats: int | None = Field(default=None, ge=0)
    source: SourceMetadata

    @field_validator("airport_code", mode="before")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalize_airport_code(value)

    @model_validator(mode="after")
    def validate_comparison_metadata(self) -> "TrafficRecord":
        comparison_values = (
            self.previous_period_passengers,
            self.previous_period,
            self.previous_source,
        )
        supplied = [value is not None for value in comparison_values]
        if any(supplied) and not all(supplied):
            raise ValueError(
                "previous passenger count, previous period, and previous source "
                "must be supplied together"
            )
        if self.source.period is not None and self.source.period != self.period:
            raise ValueError("traffic source period must match the current traffic period")
        if self.previous_period is not None and self.previous_source is not None:
            if self.previous_period.end_date >= self.period.start_date:
                raise ValueError("previous period must end before the current period")
            if not _periods_are_comparable(self.period, self.previous_period):
                raise ValueError("current and previous traffic periods must be comparable")
            if self.previous_source.period != self.previous_period:
                raise ValueError("previous source period must match the previous traffic period")
        return self


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

    @field_validator("origin_airport_code", "destination_airport_code", mode="before")
    @classmethod
    def validate_airport_code(cls, value: str) -> str:
        return _normalize_airport_code(value)

    @model_validator(mode="after")
    def validate_route(self) -> "RouteRecord":
        if self.origin_airport_code == self.destination_airport_code:
            raise ValueError("route origin and destination must differ")
        if self.source.period is not None and self.source.period != self.period:
            raise ValueError("route source period must match the route period")
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
    def validate_operations(self) -> "OperationalData":
        if self.performed_departures > self.scheduled_departures:
            raise ValueError("performed_departures cannot exceed scheduled_departures")
        if (
            self.reported_cancellations is not None
            and self.reported_cancellations > self.scheduled_departures
        ):
            raise ValueError("reported_cancellations cannot exceed scheduled_departures")
        if self.source.period is not None and self.source.period != self.period:
            raise ValueError("operational source period must match the operations period")
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
    load_factor: float | None = Field(default=None, ge=0, le=1)
    completion_rate: float | None = Field(default=None, ge=0, le=1)
    cancellation_rate: float | None = Field(default=None, ge=0, le=1)
    departures_per_runway: float | None = Field(default=None, ge=0)
    congestion_score: float | None = Field(default=None, ge=0, le=100)
    investment_opportunity_score: float | None = Field(default=None, ge=0, le=100)
    estimated_unmet_capacity_proxy: float | None = Field(default=None, ge=0)
    market_scale: float | None = Field(default=None, ge=0)
    missing_components: list[str] = Field(default_factory=list)


class RankAirportsInput(DomainModel):
    """Validated input for ``rank_airports``."""

    region: RegionName | None = None
    state_codes: list[str] | None = None
    limit: int = Field(default=5, ge=1, le=10)
    excluded_airports: list[str] = Field(default_factory=list)

    @field_validator("region", mode="before")
    @classmethod
    def validate_region(cls, value: str | RegionName | None) -> RegionName | None:
        return None if value is None else _normalize_region(value)

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

    @model_validator(mode="after")
    def resolve_region_states(self) -> "RankAirportsInput":
        if self.region is RegionName.NEW_ENGLAND:
            expected = list(NEW_ENGLAND_STATE_CODES)
            if self.state_codes is None:
                self.state_codes = expected
            elif set(self.state_codes) != set(expected):
                raise ValueError(
                    "New England must resolve to exactly CT, ME, MA, NH, RI, and VT"
                )
        return self


class CompareAirportsInput(DomainModel):
    """Validated input for ``compare_airports``."""

    airport_codes: list[str] = Field(min_length=2, max_length=10)
    metrics: list[ComparisonMetric] | None = Field(default=None, min_length=1, max_length=11)

    @field_validator("airport_codes", mode="before")
    @classmethod
    def validate_airport_codes(cls, value: list[str]) -> list[str]:
        return _normalize_airport_codes(value)

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(
        cls, value: list[str | ComparisonMetric] | None
    ) -> list[str | ComparisonMetric] | None:
        if value is None:
            return None
        normalized = [
            item if isinstance(item, ComparisonMetric) else item.strip().lower().replace("-", "_")
            for item in value
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("comparison metrics must be unique")
        return normalized


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
    sources: list[SourceMetadata] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)


class RankedAirport(DomainModel):
    """One deterministic ranking row."""

    rank: int = Field(ge=1)
    analysis: AirportAnalysis
    recommendation: RecommendationBand


def _collect_nested_sources(value: Any) -> list[SourceMetadata]:
    if isinstance(value, SourceMetadata):
        return [value]
    if isinstance(value, BaseModel):
        collected: list[SourceMetadata] = []
        for field_name in value.__class__.model_fields:
            collected.extend(_collect_nested_sources(getattr(value, field_name)))
        return collected
    if isinstance(value, (list, tuple, set)):
        collected = []
        for item in value:
            collected.extend(_collect_nested_sources(item))
        return collected
    if isinstance(value, dict):
        collected = []
        for item in value.values():
            collected.extend(_collect_nested_sources(item))
        return collected
    return []


class AnalyticalOutput(DomainModel):
    """Base for outputs that must expose complete, internally consistent provenance."""

    data_mode: DataMode
    period: DataPeriod
    sources: list[SourceMetadata] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_provenance(self) -> "AnalyticalOutput":
        for source in self.sources:
            if source.data_mode is not self.data_mode:
                raise ValueError("output data mode must match every source data mode")
            if source.period is None or source.period != self.period:
                raise ValueError("every output source period must match the output period")

        nested_sources: list[SourceMetadata] = []
        for field_name in self.__class__.model_fields:
            if field_name in {"data_mode", "period", "sources"}:
                continue
            nested_sources.extend(_collect_nested_sources(getattr(self, field_name)))
        for source in nested_sources:
            if source.data_mode is not self.data_mode:
                raise ValueError("nested source data mode contradicts the output data mode")
            if source.period is not None and source.period != self.period:
                raise ValueError("nested source period contradicts the output period")
        return self


class RankAirportsOutput(AnalyticalOutput):
    """Structured output from ``rank_airports``."""

    input: RankAirportsInput
    results: list[RankedAirport]
    assumptions: list[str] = Field(default_factory=list)


class CompareAirportsOutput(AnalyticalOutput):
    """Structured output from ``compare_airports``."""

    input: CompareAirportsInput
    airports: list[AirportAnalysis]
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


class LongHaulShareOutput(AnalyticalOutput):
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


class UnmetCapacityOutput(AnalyticalOutput):
    """Structured output from ``estimate_unmet_capacity``."""

    input: EstimateUnmetCapacityInput
    breakdown: UnmetCapacityBreakdown
    airport: AirportRecord
    confidence: ConfidenceInfo
    assumptions: list[str] = Field(default_factory=list)


class AirportProfileOutput(AnalyticalOutput):
    """Structured output from ``get_airport_profile``."""

    input: GetAirportProfileInput
    analysis: AirportAnalysis


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
