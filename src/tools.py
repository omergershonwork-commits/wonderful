"""Approved deterministic analytical tools for the airport MVP."""

from __future__ import annotations

from collections.abc import Iterable

from src.data.repository import FixtureAirportRepository
from src.exceptions import DataNotFoundError
from src.metrics import long_haul_share, metrics_from_records, passenger_growth, unmet_capacity_proxy
from src.models import (
    AirportAnalysis,
    AirportProfileOutput,
    CalculateLongHaulShareInput,
    CalculatedMetrics,
    CompareAirportsInput,
    CompareAirportsOutput,
    ConfidenceInfo,
    ConfidenceLevel,
    EstimateUnmetCapacityInput,
    GetAirportProfileInput,
    LongHaulShareOutput,
    QualifyingRoute,
    RankAirportsInput,
    RankAirportsOutput,
    RankedAirport,
    SourceMetadata,
    UnmetCapacityBreakdown,
    UnmetCapacityOutput,
)
from src.scoring import AirportScore, AirportScoringInput, deterministic_ranking_key, score_airports

DEFAULT_TARGET_LOAD_FACTOR = 0.82
DEFAULT_MIN_ANNUAL_PASSENGERS = 100_000


def _dedupe_sources(sources: Iterable[SourceMetadata]) -> list[SourceMetadata]:
    seen: set[tuple[object, ...]] = set()
    results: list[SourceMetadata] = []
    for source in sources:
        key = (
            source.source_name,
            source.data_mode,
            source.retrieved_at,
            source.period.start_date if source.period else None,
            source.period.end_date if source.period else None,
            source.source_url,
            tuple(source.notes),
        )
        if key not in seen:
            seen.add(key)
            results.append(source.model_copy(deep=True))
    return results


class AirportAnalyticsService:
    """Execute approved tools using repository data and deterministic Python."""

    def __init__(
        self,
        repository: FixtureAirportRepository | None = None,
        *,
        min_annual_passengers: int = DEFAULT_MIN_ANNUAL_PASSENGERS,
    ) -> None:
        self.repository = repository or FixtureAirportRepository.from_default_fixtures()
        self.min_annual_passengers = min_annual_passengers

    @property
    def period(self):
        return self.repository.bundle.period

    @property
    def data_mode(self):
        return self.repository.bundle.data_mode

    def _require_supported_codes(self, codes: Iterable[str]) -> None:
        supported = set(self.repository.supported_airport_codes)
        unknown = [code for code in codes if code not in supported]
        if unknown:
            raise DataNotFoundError(
                "Unsupported airport code(s): " + ", ".join(sorted(unknown))
            )

    def _sources(
        self,
        airport_code: str,
        *,
        include_routes: bool = False,
        include_previous_traffic: bool = True,
    ) -> list[SourceMetadata]:
        airport = self.repository.get_airport(airport_code)
        traffic = self.repository.get_traffic(airport_code)
        operations = self.repository.get_operations(airport_code)
        sources = [airport.source, traffic.source, operations.source]
        if include_previous_traffic and traffic.previous_source is not None:
            sources.append(traffic.previous_source)
        if include_routes:
            sources.extend(route.source for route in self.repository.get_routes(airport_code))
        return _dedupe_sources(sources)

    def _all_sources(self, *, include_previous_traffic: bool = True) -> list[SourceMetadata]:
        return _dedupe_sources(
            source
            for code in self.repository.supported_airport_codes
            for source in self._sources(
                code,
                include_previous_traffic=include_previous_traffic,
            )
        )

    def _raw_scoring_input(
        self,
        airport_code: str,
        *,
        target_load_factor: float = DEFAULT_TARGET_LOAD_FACTOR,
    ) -> AirportScoringInput:
        traffic = self.repository.get_traffic(airport_code)
        operations = self.repository.get_operations(airport_code)
        metrics = metrics_from_records(
            traffic, operations, target_load_factor=target_load_factor
        )
        return AirportScoringInput(
            airport_code=airport_code,
            passengers=traffic.passengers,
            passenger_growth=metrics["passenger_growth"],
            load_factor=metrics["load_factor"],
            average_departure_delay_minutes=metrics["average_departure_delay_minutes"],
            average_taxi_out_minutes=metrics["average_taxi_out_minutes"],
            cancellation_rate=metrics["cancellation_rate"],
            departures_per_runway=metrics["departures_per_runway"],
            estimated_unmet_capacity_proxy=metrics["estimated_unmet_capacity_proxy"],
        )

    def _scored_universe(
        self,
        *,
        target_load_factor: float = DEFAULT_TARGET_LOAD_FACTOR,
    ) -> tuple[dict[str, AirportScoringInput], dict[str, AirportScore]]:
        raw = {
            code: self._raw_scoring_input(code, target_load_factor=target_load_factor)
            for code in self.repository.supported_airport_codes
        }
        return raw, score_airports(raw.values())

    def _analysis(
        self,
        airport_code: str,
        raw: AirportScoringInput,
        score: AirportScore,
        *,
        selected_metrics: set[str] | None = None,
    ) -> AirportAnalysis:
        traffic = self.repository.get_traffic(airport_code)
        operations = self.repository.get_operations(airport_code)
        raw_metrics = metrics_from_records(traffic, operations)
        values = {
            "passenger_growth": raw.passenger_growth,
            "load_factor": raw.load_factor,
            "completion_rate": raw_metrics["completion_rate"],
            "cancellation_rate": raw.cancellation_rate,
            "departures_per_runway": raw.departures_per_runway,
            "average_departure_delay_minutes": raw.average_departure_delay_minutes,
            "average_taxi_out_minutes": raw.average_taxi_out_minutes,
            "congestion_score": score.congestion_score,
            "investment_opportunity_score": score.investment_opportunity_score,
            "estimated_unmet_capacity_proxy": raw.estimated_unmet_capacity_proxy,
            "market_scale": float(raw.passengers),
        }
        if selected_metrics is not None:
            values = {
                name: value if name in selected_metrics else None
                for name, value in values.items()
            }
        calculated = CalculatedMetrics(
            **values,
            missing_components=list(score.missing_components),
        )
        return AirportAnalysis(
            airport=self.repository.get_airport(airport_code),
            metrics=calculated,
            confidence=score.confidence,
            sources=self._sources(airport_code),
            assumptions=[
                "Scores are deterministic screening proxies, not expected financial return.",
                "Percentiles use the complete supported fixture-airport reference set.",
            ],
        )

    def rank_airports(self, request: RankAirportsInput) -> RankAirportsOutput:
        airports = self.repository.list_airports(
            region=request.region.value if request.region else None,
            state_codes=set(request.state_codes or []),
            excluded_airports=set(request.excluded_airports),
        )
        eligible = [
            airport
            for airport in airports
            if self.repository.get_traffic(airport.airport_code).passengers
            >= self.min_annual_passengers
        ]
        if not eligible:
            return RankAirportsOutput(
                input=request,
                results=[],
                data_mode=self.data_mode,
                period=self.period,
                sources=self._all_sources(include_previous_traffic=False),
                assumptions=[
                    "No supported airports matched the validated filters and passenger floor."
                ],
            )
        raw, scores = self._scored_universe()
        ordered = sorted(
            [(raw[a.airport_code], scores[a.airport_code]) for a in eligible],
            key=deterministic_ranking_key,
        )[: request.limit]
        results = [
            RankedAirport(
                rank=index,
                analysis=self._analysis(row.airport_code, row, score),
                recommendation=score.recommendation,
            )
            for index, (row, score) in enumerate(ordered, start=1)
        ]
        return RankAirportsOutput(
            input=request,
            results=results,
            data_mode=self.data_mode,
            period=self.period,
            sources=self._all_sources(),
            assumptions=[
                "New England means CT, ME, MA, NH, RI, and VT.",
                "Airports below the configured annual-passenger floor are excluded.",
            ],
        )

    def compare_airports(self, request: CompareAirportsInput) -> CompareAirportsOutput:
        self._require_supported_codes(request.airport_codes)
        raw, scores = self._scored_universe()
        selected = {metric.value for metric in request.metrics} if request.metrics else None
        analyses = [
            self._analysis(code, raw[code], scores[code], selected_metrics=selected)
            for code in request.airport_codes
        ]
        return CompareAirportsOutput(
            input=request,
            airports=analyses,
            data_mode=self.data_mode,
            period=self.period,
            sources=self._all_sources(),
            assumptions=[
                "Congestion compares normalized pressure and does not represent terminal-specific crowding."
            ],
        )

    def calculate_long_haul_share(
        self, request: CalculateLongHaulShareInput
    ) -> LongHaulShareOutput:
        self._require_supported_codes([request.airport_code])
        routes = self.repository.get_routes(request.airport_code)
        result = long_haul_share(routes, request.threshold_miles)
        if result.departure_share is None or result.passenger_share is None:
            raise ValueError("long-haul shares require non-zero complete route denominators")
        operations = self.repository.get_operations(request.airport_code)
        if result.all_departures != operations.performed_departures:
            raise ValueError("route departures must equal performed departures")
        qualifying = [
            QualifyingRoute(
                destination_airport_code=route.destination_airport_code,
                destination_name=route.destination_name,
                distance_miles=route.distance_miles,
                departures=route.departures,
                passengers=route.passengers,
            )
            for route in sorted(
                result.qualifying_routes,
                key=lambda item: (-item.departures, item.destination_airport_code),
            )
        ]
        return LongHaulShareOutput(
            input=request,
            long_haul_departures=result.long_haul_departures,
            all_departures=result.all_departures,
            departure_share=result.departure_share,
            long_haul_passengers=result.long_haul_passengers,
            all_route_passengers=result.all_route_passengers,
            passenger_share=result.passenger_share,
            qualifying_routes=qualifying,
            confidence=ConfidenceInfo(
                level=ConfidenceLevel.HIGH,
                score=1.0,
                reasons=["The validated route universe equals performed departures."],
            ),
            data_mode=self.data_mode,
            period=self.period,
            sources=self._sources(
                request.airport_code,
                include_routes=True,
                include_previous_traffic=False,
            ),
            assumptions=[
                "Long haul means distance greater than or equal to "
                f"{request.threshold_miles:,} statute miles."
            ],
        )

    @staticmethod
    def _context_value(label: str, value: float | None, suffix: str = "") -> str:
        if value is None:
            return f"{label} is unavailable"
        return f"{label} is {value:.2f}{suffix}"

    def estimate_unmet_capacity(
        self, request: EstimateUnmetCapacityInput
    ) -> UnmetCapacityOutput:
        self._require_supported_codes([request.airport_code])
        traffic = self.repository.get_traffic(request.airport_code)
        growth = passenger_growth(traffic.passengers, traffic.previous_period_passengers)
        result = unmet_capacity_proxy(
            traffic.passengers,
            traffic.available_seats,
            growth,
            request.target_load_factor,
        )
        if result is None:
            raise ValueError(
                "unmet-capacity proxy requires comparable traffic and available-seat data"
            )
        raw, scores = self._scored_universe(target_load_factor=request.target_load_factor)
        airport_raw = raw[request.airport_code]
        airport_score = scores[request.airport_code]
        analysis = self._analysis(request.airport_code, airport_raw, airport_score)
        context = "; ".join(
            [
                self._context_value(
                    "average departure delay",
                    airport_raw.average_departure_delay_minutes,
                    " minutes",
                ),
                self._context_value(
                    "average taxi-out",
                    airport_raw.average_taxi_out_minutes,
                    " minutes",
                ),
                self._context_value("congestion score", airport_score.congestion_score),
            ]
        )
        return UnmetCapacityOutput(
            input=request,
            breakdown=UnmetCapacityBreakdown(
                current_passengers=result.current_passengers,
                current_available_seats=result.current_available_seats,
                raw_passenger_growth=result.raw_growth,
                clamped_passenger_growth=result.clamped_growth,
                projected_passengers=result.projected_passengers,
                target_load_factor=result.target_load_factor,
                required_seats=result.required_seats,
                estimated_unmet_capacity_proxy=result.estimated_unmet_capacity_proxy,
            ),
            airport=analysis.airport,
            confidence=analysis.confidence,
            data_mode=self.data_mode,
            period=self.period,
            sources=self._all_sources(),
            assumptions=[
                "This is an estimated unmet-capacity proxy, not observed lost demand.",
                "Passenger growth is clamped to the range -10% to 20%.",
                "Supporting context only: " + context + ". These are not inputs to the capacity formula.",
            ],
        )

    def get_airport_profile(self, request: GetAirportProfileInput) -> AirportProfileOutput:
        self._require_supported_codes([request.airport_code])
        raw, scores = self._scored_universe()
        analysis = self._analysis(
            request.airport_code, raw[request.airport_code], scores[request.airport_code]
        )
        return AirportProfileOutput(
            input=request,
            analysis=analysis,
            data_mode=self.data_mode,
            period=self.period,
            sources=self._all_sources(),
        )


def rank_airports(
    region: str | None = None,
    state_codes: list[str] | None = None,
    limit: int = 5,
    excluded_airports: list[str] | None = None,
    *,
    repository: FixtureAirportRepository | None = None,
) -> RankAirportsOutput:
    return AirportAnalyticsService(repository).rank_airports(
        RankAirportsInput(
            region=region,
            state_codes=state_codes,
            limit=limit,
            excluded_airports=excluded_airports or [],
        )
    )


def compare_airports(
    airport_codes: list[str],
    metrics: list[str] | None = None,
    *,
    repository: FixtureAirportRepository | None = None,
) -> CompareAirportsOutput:
    return AirportAnalyticsService(repository).compare_airports(
        CompareAirportsInput(airport_codes=airport_codes, metrics=metrics)
    )


def calculate_long_haul_share(
    airport_code: str,
    threshold_miles: int = 3000,
    *,
    repository: FixtureAirportRepository | None = None,
) -> LongHaulShareOutput:
    return AirportAnalyticsService(repository).calculate_long_haul_share(
        CalculateLongHaulShareInput(
            airport_code=airport_code, threshold_miles=threshold_miles
        )
    )


def estimate_unmet_capacity(
    airport_code: str,
    target_load_factor: float = DEFAULT_TARGET_LOAD_FACTOR,
    *,
    repository: FixtureAirportRepository | None = None,
) -> UnmetCapacityOutput:
    return AirportAnalyticsService(repository).estimate_unmet_capacity(
        EstimateUnmetCapacityInput(
            airport_code=airport_code, target_load_factor=target_load_factor
        )
    )


def get_airport_profile(
    airport_code: str,
    *,
    repository: FixtureAirportRepository | None = None,
) -> AirportProfileOutput:
    return AirportAnalyticsService(repository).get_airport_profile(
        GetAirportProfileInput(airport_code=airport_code)
    )
