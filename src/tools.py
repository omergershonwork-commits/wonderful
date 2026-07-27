"""Approved deterministic analytical tools for the airport MVP."""

from __future__ import annotations

from collections.abc import Iterable

from src.data.repository import FixtureAirportRepository
from src.metrics import (
    long_haul_share,
    metrics_from_records,
    passenger_growth,
    unmet_capacity_proxy,
)
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
from src.scoring import (
    AirportScore,
    AirportScoringInput,
    deterministic_ranking_key,
    score_airports,
)

DEFAULT_TARGET_LOAD_FACTOR = 0.82
DEFAULT_MIN_ANNUAL_PASSENGERS = 100_000


def _dedupe_sources(
    sources: Iterable[SourceMetadata],
) -> list[SourceMetadata]:
    """Return deterministic defensive copies of unique source records."""

    seen: set[tuple[object, ...]] = set()
    results: list[SourceMetadata] = []
    for source in sources:
        key = (
            source.source_name,
            source.data_mode,
            source.retrieved_at,
            source.period.start_date if source.period else None,
            source.period.end_date if source.period else None,
        )
        if key in seen:
            continue
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
        self.repository = (
            repository or FixtureAirportRepository.from_default_fixtures()
        )
        self.min_annual_passengers = min_annual_passengers

    @property
    def period(self):
        """Return the repository analysis period."""

        bundle = self.repository.bundle
        return bundle.period

    @property
    def data_mode(self):
        """Return the repository's explicit source mode."""

        bundle = self.repository.bundle
        return bundle.data_mode

    def _raw_scoring_input(
        self,
        airport_code: str,
        *,
        target_load_factor: float = DEFAULT_TARGET_LOAD_FACTOR,
    ) -> AirportScoringInput:
        traffic = self.repository.get_traffic(airport_code)
        operations = self.repository.get_operations(airport_code)
        metrics = metrics_from_records(
            traffic,
            operations,
            target_load_factor=target_load_factor,
        )
        return AirportScoringInput(
            airport_code=airport_code,
            passengers=traffic.passengers,
            passenger_growth=metrics["passenger_growth"],
            load_factor=metrics["load_factor"],
            average_departure_delay_minutes=metrics[
                "average_departure_delay_minutes"
            ],
            average_taxi_out_minutes=metrics[
                "average_taxi_out_minutes"
            ],
            cancellation_rate=metrics["cancellation_rate"],
            departures_per_runway=metrics["departures_per_runway"],
            estimated_unmet_capacity_proxy=metrics[
                "estimated_unmet_capacity_proxy"
            ],
        )

    def _scored_universe(
        self,
        *,
        target_load_factor: float = DEFAULT_TARGET_LOAD_FACTOR,
    ) -> tuple[dict[str, AirportScoringInput], dict[str, AirportScore]]:
        raw = {
            code: self._raw_scoring_input(
                code,
                target_load_factor=target_load_factor,
            )
            for code in self.repository.supported_airport_codes
        }
        return raw, score_airports(raw.values())

    def _sources(
        self,
        airport_code: str,
        *,
        include_routes: bool = False,
    ) -> list[SourceMetadata]:
        airport = self.repository.get_airport(airport_code)
        traffic = self.repository.get_traffic(airport_code)
        operations = self.repository.get_operations(airport_code)
        sources = [airport.source, traffic.source, operations.source]
        if include_routes:
            sources.extend(
                route.source
                for route in self.repository.get_routes(airport_code)
            )
        return _dedupe_sources(sources)

    def _analysis(
        self,
        airport_code: str,
        raw: AirportScoringInput,
        score: AirportScore,
    ) -> AirportAnalysis:
        traffic = self.repository.get_traffic(airport_code)
        operations = self.repository.get_operations(airport_code)
        raw_metrics = metrics_from_records(traffic, operations)
        calculated = CalculatedMetrics(
            passenger_growth=raw.passenger_growth,
            load_factor=raw.load_factor,
            completion_rate=raw_metrics["completion_rate"],
            cancellation_rate=raw.cancellation_rate,
            departures_per_runway=raw.departures_per_runway,
            congestion_score=score.congestion_score,
            investment_opportunity_score=(
                score.investment_opportunity_score
            ),
            estimated_unmet_capacity_proxy=(
                raw.estimated_unmet_capacity_proxy
            ),
            market_scale=float(raw.passengers),
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

    def rank_airports(
        self,
        request: RankAirportsInput,
    ) -> RankAirportsOutput:
        """Rank filtered airports using deterministic opportunity scores."""

        raw, scores = self._scored_universe()
        airports = self.repository.list_airports(
            region=request.region.value if request.region else None,
            state_codes=set(request.state_codes or []),
            excluded_airports=set(request.excluded_airports),
        )
        candidates = [
            (raw[airport.airport_code], scores[airport.airport_code])
            for airport in airports
            if raw[airport.airport_code].passengers
            >= self.min_annual_passengers
        ]
        ordered = sorted(
            candidates,
            key=deterministic_ranking_key,
        )[: request.limit]
        results = [
            RankedAirport(
                rank=index,
                analysis=self._analysis(
                    raw_row.airport_code,
                    raw_row,
                    score,
                ),
                recommendation=score.recommendation,
            )
            for index, (raw_row, score) in enumerate(ordered, start=1)
        ]
        sources = _dedupe_sources(
            source
            for result in results
            for source in result.analysis.sources
        )
        return RankAirportsOutput(
            input=request,
            results=results,
            data_mode=self.data_mode,
            period=self.period,
            sources=sources,
            assumptions=[
                "New England means CT, ME, MA, NH, RI, and VT.",
                "Airports below the configured annual-passenger floor are excluded.",
            ],
        )

    def compare_airports(
        self,
        request: CompareAirportsInput,
    ) -> CompareAirportsOutput:
        """Return comparable analyses in the caller's requested order."""

        raw, scores = self._scored_universe()
        analyses = [
            self._analysis(code, raw[code], scores[code])
            for code in request.airport_codes
        ]
        return CompareAirportsOutput(
            input=request,
            airports=analyses,
            data_mode=self.data_mode,
            period=self.period,
            sources=_dedupe_sources(
                source
                for analysis in analyses
                for source in analysis.sources
            ),
            assumptions=[
                "Congestion compares normalized pressure and does not represent terminal-specific crowding."
            ],
        )

    def calculate_long_haul_share(
        self,
        request: CalculateLongHaulShareInput,
    ) -> LongHaulShareOutput:
        """Return flight- and passenger-weighted long-haul shares."""

        routes = self.repository.get_routes(request.airport_code)
        result = long_haul_share(routes, request.threshold_miles)
        if result.departure_share is None or result.passenger_share is None:
            raise ValueError(
                "long-haul shares require non-zero complete route denominators"
            )

        operations = self.repository.get_operations(request.airport_code)
        if result.all_departures != operations.performed_departures:
            raise ValueError(
                "route departures must equal performed departures"
            )

        qualifying_routes = [
            QualifyingRoute(
                destination_airport_code=route.destination_airport_code,
                destination_name=route.destination_name,
                distance_miles=route.distance_miles,
                departures=route.departures,
                passengers=route.passengers,
            )
            for route in sorted(
                result.qualifying_routes,
                key=lambda item: (
                    -item.departures,
                    item.destination_airport_code,
                ),
            )
        ]
        confidence = ConfidenceInfo(
            level=ConfidenceLevel.HIGH,
            score=1.0,
            reasons=[
                "The validated route universe equals performed departures."
            ],
        )
        return LongHaulShareOutput(
            input=request,
            long_haul_departures=result.long_haul_departures,
            all_departures=result.all_departures,
            departure_share=result.departure_share,
            long_haul_passengers=result.long_haul_passengers,
            all_route_passengers=result.all_route_passengers,
            passenger_share=result.passenger_share,
            qualifying_routes=qualifying_routes,
            confidence=confidence,
            data_mode=self.data_mode,
            period=self.period,
            sources=self._sources(
                request.airport_code,
                include_routes=True,
            ),
            assumptions=[
                "Long haul means distance greater than or equal to "
                f"{request.threshold_miles:,} statute miles."
            ],
        )

    def estimate_unmet_capacity(
        self,
        request: EstimateUnmetCapacityInput,
    ) -> UnmetCapacityOutput:
        """Return the approved estimated unmet-seat-capacity proxy."""

        traffic = self.repository.get_traffic(request.airport_code)
        growth = passenger_growth(
            traffic.passengers,
            traffic.previous_period_passengers,
        )
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

        raw, scores = self._scored_universe(
            target_load_factor=request.target_load_factor
        )
        airport_raw = raw[request.airport_code]
        airport_score = scores[request.airport_code]
        analysis = self._analysis(
            request.airport_code,
            airport_raw,
            airport_score,
        )
        breakdown = UnmetCapacityBreakdown(
            current_passengers=result.current_passengers,
            current_available_seats=result.current_available_seats,
            raw_passenger_growth=result.raw_growth,
            clamped_passenger_growth=result.clamped_growth,
            projected_passengers=result.projected_passengers,
            target_load_factor=result.target_load_factor,
            required_seats=result.required_seats,
            estimated_unmet_capacity_proxy=(
                result.estimated_unmet_capacity_proxy
            ),
        )
        context = (
            "Supporting context only: average departure delay is "
            f"{airport_raw.average_departure_delay_minutes} minutes, "
            "average taxi-out is "
            f"{airport_raw.average_taxi_out_minutes} minutes, and "
            f"congestion score is {airport_score.congestion_score:.2f}. "
            "These are not inputs to the capacity formula."
        )
        return UnmetCapacityOutput(
            input=request,
            breakdown=breakdown,
            airport=analysis.airport,
            confidence=analysis.confidence,
            data_mode=self.data_mode,
            period=self.period,
            sources=analysis.sources,
            assumptions=[
                "This is an estimated unmet-capacity proxy, not observed lost demand.",
                "Passenger growth is clamped to the range -10% to 20%.",
                context,
            ],
        )

    def get_airport_profile(
        self,
        request: GetAirportProfileInput,
    ) -> AirportProfileOutput:
        """Return one airport's deterministic metrics, scores, and provenance."""

        raw, scores = self._scored_universe()
        analysis = self._analysis(
            request.airport_code,
            raw[request.airport_code],
            scores[request.airport_code],
        )
        return AirportProfileOutput(
            input=request,
            analysis=analysis,
            data_mode=self.data_mode,
            period=self.period,
            sources=analysis.sources,
        )


def rank_airports(
    region: str | None = None,
    state_codes: list[str] | None = None,
    limit: int = 5,
    excluded_airports: list[str] | None = None,
    *,
    repository: FixtureAirportRepository | None = None,
) -> RankAirportsOutput:
    """Validated public boundary for airport ranking."""

    request = RankAirportsInput(
        region=region,
        state_codes=state_codes,
        limit=limit,
        excluded_airports=excluded_airports or [],
    )
    return AirportAnalyticsService(repository).rank_airports(request)


def compare_airports(
    airport_codes: list[str],
    metrics: list[str] | None = None,
    *,
    repository: FixtureAirportRepository | None = None,
) -> CompareAirportsOutput:
    """Validated public boundary for airport comparison."""

    request = CompareAirportsInput(
        airport_codes=airport_codes,
        metrics=metrics,
    )
    return AirportAnalyticsService(repository).compare_airports(request)


def calculate_long_haul_share(
    airport_code: str,
    threshold_miles: int = 3000,
    *,
    repository: FixtureAirportRepository | None = None,
) -> LongHaulShareOutput:
    """Validated public boundary for long-haul analysis."""

    request = CalculateLongHaulShareInput(
        airport_code=airport_code,
        threshold_miles=threshold_miles,
    )
    return AirportAnalyticsService(repository).calculate_long_haul_share(
        request
    )


def estimate_unmet_capacity(
    airport_code: str,
    target_load_factor: float = DEFAULT_TARGET_LOAD_FACTOR,
    *,
    repository: FixtureAirportRepository | None = None,
) -> UnmetCapacityOutput:
    """Validated public boundary for the estimated capacity proxy."""

    request = EstimateUnmetCapacityInput(
        airport_code=airport_code,
        target_load_factor=target_load_factor,
    )
    return AirportAnalyticsService(repository).estimate_unmet_capacity(request)


def get_airport_profile(
    airport_code: str,
    *,
    repository: FixtureAirportRepository | None = None,
) -> AirportProfileOutput:
    """Validated public boundary for one airport profile."""

    request = GetAirportProfileInput(airport_code=airport_code)
    return AirportAnalyticsService(repository).get_airport_profile(request)
