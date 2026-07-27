"""Load and validate the bundled illustrative airport fixtures."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.exceptions import FixtureValidationError
from src.models import (
    AirportRecord,
    DataMode,
    DataPeriod,
    OperationalData,
    RouteRecord,
    SourceMetadata,
    TrafficRecord,
)

REQUIRED_AIRPORT_CODES = frozenset(
    {"BOS", "BDL", "PVD", "MHT", "PWM", "BTV", "LAX", "SNA", "ANC", "SFO"}
)
NEW_ENGLAND_STATE_CODES = frozenset({"CT", "ME", "MA", "NH", "RI", "VT"})
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class _ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(min_length=1)
    data_mode: DataMode
    retrieved_at: datetime
    period: DataPeriod
    disclaimer: str = Field(min_length=1)
    files: dict[str, _ManifestFile]


@dataclass(frozen=True, slots=True)
class FixtureDataBundle:
    """Fully validated fixture records loaded from disk."""

    airports: tuple[AirportRecord, ...]
    traffic: tuple[TrafficRecord, ...]
    routes: tuple[RouteRecord, ...]
    operations: tuple[OperationalData, ...]
    period: DataPeriod
    data_mode: DataMode
    disclaimer: str


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureValidationError(f"Could not read JSON fixture {path.name}: {exc}") from exc


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    except OSError as exc:
        raise FixtureValidationError(f"Could not read CSV fixture {path.name}: {exc}") from exc


def _source_for(
    manifest: _Manifest,
    filename: str,
    *,
    period: DataPeriod | None = None,
) -> SourceMetadata:
    try:
        file_info = manifest.files[filename]
    except KeyError as exc:
        raise FixtureValidationError(f"Manifest does not describe {filename}") from exc

    return SourceMetadata(
        source_name=file_info.source_name,
        data_mode=manifest.data_mode,
        retrieved_at=manifest.retrieved_at,
        period=period or manifest.period,
        notes=[*file_info.notes, manifest.disclaimer],
    )


def _period_from_row(row: dict[str, str], prefix: str = "period") -> DataPeriod:
    return DataPeriod(
        start_date=row[f"{prefix}_start"],
        end_date=row[f"{prefix}_end"],
        label=row.get(f"{prefix}_label") or None,
    )


def _optional_int(value: str | None) -> int | None:
    return None if value is None or value == "" else int(value)


def _optional_float(value: str | None) -> float | None:
    return None if value is None or value == "" else float(value)


def _validate_coverage(
    airports: tuple[AirportRecord, ...],
    traffic: tuple[TrafficRecord, ...],
    routes: tuple[RouteRecord, ...],
    operations: tuple[OperationalData, ...],
) -> None:
    airport_codes = {item.airport_code for item in airports}
    traffic_codes = {item.airport_code for item in traffic}
    operation_codes = {item.airport_code for item in operations}
    route_origins = {item.origin_airport_code for item in routes}

    coverage = {
        "demo_airports.json": airport_codes,
        "demo_traffic.csv": traffic_codes,
        "demo_operations.csv": operation_codes,
        "demo_routes.csv origins": route_origins,
    }
    for dataset_name, codes in coverage.items():
        missing = REQUIRED_AIRPORT_CODES - codes
        unexpected = codes - REQUIRED_AIRPORT_CODES
        if missing or unexpected:
            raise FixtureValidationError(
                f"{dataset_name} coverage mismatch; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

    if len(airports) != len(airport_codes):
        raise FixtureValidationError("demo_airports.json contains duplicate airport codes")
    if len(traffic) != len(traffic_codes):
        raise FixtureValidationError("demo_traffic.csv must contain one row per airport")
    if len(operations) != len(operation_codes):
        raise FixtureValidationError("demo_operations.csv must contain one row per airport")

    route_keys = {
        (
            item.origin_airport_code,
            item.destination_airport_code,
            item.period.start_date,
            item.period.end_date,
        )
        for item in routes
    }
    if len(routes) != len(route_keys):
        raise FixtureValidationError(
            "demo_routes.csv contains duplicate origin/destination/period rows"
        )


def _validate_periods(bundle_period: DataPeriod, records: list[Any]) -> None:
    mismatches = [record for record in records if record.period != bundle_period]
    if mismatches:
        raise FixtureValidationError("All fixture records must use the manifest data period")


def _validate_cross_file_consistency(
    airports: tuple[AirportRecord, ...],
    traffic: tuple[TrafficRecord, ...],
    routes: tuple[RouteRecord, ...],
    operations: tuple[OperationalData, ...],
) -> None:
    airports_by_code = {item.airport_code: item for item in airports}
    traffic_by_code = {item.airport_code: item for item in traffic}
    operations_by_code = {item.airport_code: item for item in operations}

    for code in REQUIRED_AIRPORT_CODES:
        airport = airports_by_code[code]
        traffic_record = traffic_by_code[code]
        operation = operations_by_code[code]

        if airport.usable_runway_count != operation.usable_runway_count:
            raise FixtureValidationError(
                f"Runway count mismatch for {code}: airport={airport.usable_runway_count}, "
                f"operations={operation.usable_runway_count}"
            )
        if (
            traffic_record.available_seats is not None
            and traffic_record.passengers > traffic_record.available_seats
        ):
            raise FixtureValidationError(
                f"Traffic passengers cannot exceed available seats for {code}"
            )
        expected_cancellations = (
            operation.scheduled_departures - operation.performed_departures
        )
        if operation.reported_cancellations != expected_cancellations:
            raise FixtureValidationError(
                f"Cancellation totals are contradictory for {code}: "
                f"reported={operation.reported_cancellations}, "
                f"scheduled-minus-performed={expected_cancellations}"
            )

    route_departures_by_origin = {code: 0 for code in REQUIRED_AIRPORT_CODES}
    for route in routes:
        if route.available_seats is not None and route.passengers > route.available_seats:
            raise FixtureValidationError(
                "Route passengers cannot exceed available seats for "
                f"{route.origin_airport_code}-{route.destination_airport_code}"
            )
        route_departures_by_origin[route.origin_airport_code] += route.departures

    for code, operation in operations_by_code.items():
        route_departures = route_departures_by_origin[code]
        if route_departures != operation.performed_departures:
            raise FixtureValidationError(
                f"Route departure denominator is incomplete for {code}: "
                f"routes={route_departures}, performed={operation.performed_departures}"
            )


def load_fixture_bundle(data_dir: Path | None = None) -> FixtureDataBundle:
    """Load all normalized fixture datasets and fail on inconsistent coverage."""

    root = data_dir or DEFAULT_DATA_DIR
    try:
        manifest = _Manifest.model_validate(_read_json(root / "source_manifest.json"))
        if manifest.data_mode is not DataMode.ILLUSTRATIVE_DEMO_DATA:
            raise FixtureValidationError(
                "Bundled demonstration fixtures must declare ILLUSTRATIVE DEMO DATA"
            )

        airport_source = _source_for(manifest, "demo_airports.json")
        traffic_source = _source_for(manifest, "demo_traffic.csv")
        route_source = _source_for(manifest, "demo_routes.csv")
        operations_source = _source_for(manifest, "demo_operations.csv")

        airports = tuple(
            AirportRecord.model_validate({**row, "source": airport_source})
            for row in _read_json(root / "demo_airports.json")
        )
        traffic_records: list[TrafficRecord] = []
        for row in _read_csv(root / "demo_traffic.csv"):
            previous_passengers = _optional_int(row.get("previous_period_passengers"))
            previous_period = (
                _period_from_row(row, "previous_period")
                if previous_passengers is not None
                else None
            )
            traffic_records.append(
                TrafficRecord(
                    airport_code=row["airport_code"],
                    period=_period_from_row(row),
                    passengers=int(row["passengers"]),
                    previous_period_passengers=previous_passengers,
                    previous_period=previous_period,
                    previous_source=(
                        _source_for(
                            manifest,
                            "demo_traffic.csv",
                            period=previous_period,
                        )
                        if previous_period is not None
                        else None
                    ),
                    available_seats=_optional_int(row.get("available_seats")),
                    source=traffic_source,
                )
            )
        traffic = tuple(traffic_records)
        routes = tuple(
            RouteRecord(
                origin_airport_code=row["origin_airport_code"],
                destination_airport_code=row["destination_airport_code"],
                destination_name=row.get("destination_name") or None,
                distance_miles=int(row["distance_miles"]),
                departures=int(row["departures"]),
                passengers=int(row["passengers"]),
                available_seats=_optional_int(row.get("available_seats")),
                period=_period_from_row(row),
                source=route_source,
            )
            for row in _read_csv(root / "demo_routes.csv")
        )
        operations = tuple(
            OperationalData(
                airport_code=row["airport_code"],
                period=_period_from_row(row),
                scheduled_departures=int(row["scheduled_departures"]),
                performed_departures=int(row["performed_departures"]),
                reported_cancellations=_optional_int(row.get("reported_cancellations")),
                average_departure_delay_minutes=_optional_float(
                    row.get("average_departure_delay_minutes")
                ),
                average_taxi_out_minutes=_optional_float(row.get("average_taxi_out_minutes")),
                usable_runway_count=_optional_int(row.get("usable_runway_count")),
                source=operations_source,
            )
            for row in _read_csv(root / "demo_operations.csv")
        )
    except FixtureValidationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise FixtureValidationError(f"Fixture validation failed: {exc}") from exc

    _validate_coverage(airports, traffic, routes, operations)
    _validate_periods(manifest.period, [*traffic, *routes, *operations])
    _validate_cross_file_consistency(airports, traffic, routes, operations)

    return FixtureDataBundle(
        airports=airports,
        traffic=traffic,
        routes=routes,
        operations=operations,
        period=manifest.period,
        data_mode=manifest.data_mode,
        disclaimer=manifest.disclaimer,
    )
