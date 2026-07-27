"""Read-only normalized repository backed by validated local fixtures."""

from __future__ import annotations

from collections import defaultdict

from src.data.fixture_loader import FixtureDataBundle, load_fixture_bundle
from src.exceptions import DataNotFoundError
from src.models import AirportRecord, OperationalData, RouteRecord, TrafficRecord


def _copy_bundle(bundle: FixtureDataBundle) -> FixtureDataBundle:
    """Return a deep copy so mutable Pydantic records never cross the boundary."""

    return FixtureDataBundle(
        airports=tuple(item.model_copy(deep=True) for item in bundle.airports),
        traffic=tuple(item.model_copy(deep=True) for item in bundle.traffic),
        routes=tuple(item.model_copy(deep=True) for item in bundle.routes),
        operations=tuple(item.model_copy(deep=True) for item in bundle.operations),
        period=bundle.period.model_copy(deep=True),
        data_mode=bundle.data_mode,
        disclaimer=bundle.disclaimer,
    )


class FixtureAirportRepository:
    """Provide deterministic airport data access without exposing mutable state."""

    def __init__(self, bundle: FixtureDataBundle) -> None:
        internal_bundle = _copy_bundle(bundle)
        self._bundle = internal_bundle
        self._airports = {
            item.airport_code: item for item in internal_bundle.airports
        }
        self._traffic = {
            item.airport_code: item for item in internal_bundle.traffic
        }
        self._operations = {
            item.airport_code: item for item in internal_bundle.operations
        }
        routes: dict[str, list[RouteRecord]] = defaultdict(list)
        for route in internal_bundle.routes:
            routes[route.origin_airport_code].append(route)
        self._routes = {
            code: tuple(sorted(items, key=lambda item: item.destination_airport_code))
            for code, items in routes.items()
        }

    @classmethod
    def from_default_fixtures(cls) -> "FixtureAirportRepository":
        """Create a repository from the checked-in demonstration datasets."""

        return cls(load_fixture_bundle())

    @property
    def bundle(self) -> FixtureDataBundle:
        """Return a defensive deep copy of the validated bundle."""

        return _copy_bundle(self._bundle)

    @property
    def supported_airport_codes(self) -> tuple[str, ...]:
        """Return supported codes in deterministic alphabetical order."""

        return tuple(sorted(self._airports))

    def _normalize_supported_code(self, airport_code: str) -> str:
        code = airport_code.strip().upper()
        if code not in self._airports:
            raise DataNotFoundError(f"Unsupported airport code: {code or airport_code!r}")
        return code

    def get_airport(self, airport_code: str) -> AirportRecord:
        return self._airports[
            self._normalize_supported_code(airport_code)
        ].model_copy(deep=True)

    def get_traffic(self, airport_code: str) -> TrafficRecord:
        return self._traffic[
            self._normalize_supported_code(airport_code)
        ].model_copy(deep=True)

    def get_operations(self, airport_code: str) -> OperationalData:
        return self._operations[
            self._normalize_supported_code(airport_code)
        ].model_copy(deep=True)

    def get_routes(self, airport_code: str) -> tuple[RouteRecord, ...]:
        routes = self._routes[self._normalize_supported_code(airport_code)]
        return tuple(item.model_copy(deep=True) for item in routes)

    def list_airports(
        self,
        *,
        region: str | None = None,
        state_codes: set[str] | None = None,
        excluded_airports: set[str] | None = None,
    ) -> tuple[AirportRecord, ...]:
        """Filter airports deterministically by region, state, and exclusion code."""

        normalized_region = region.strip().casefold() if region else None
        normalized_states = {state.strip().upper() for state in state_codes or set()}
        normalized_exclusions = {
            code.strip().upper() for code in excluded_airports or set()
        }

        results = []
        for airport in self._airports.values():
            if normalized_region and (airport.region or "").casefold() != normalized_region:
                continue
            if normalized_states and airport.state_code not in normalized_states:
                continue
            if airport.airport_code in normalized_exclusions:
                continue
            results.append(airport.model_copy(deep=True))
        return tuple(sorted(results, key=lambda item: item.airport_code))
