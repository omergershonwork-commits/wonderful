"""Offline validation for bundled illustrative fixture data."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from src.data.fixture_loader import (
    DEFAULT_DATA_DIR,
    NEW_ENGLAND_STATE_CODES,
    REQUIRED_AIRPORT_CODES,
    load_fixture_bundle,
)
from src.data.repository import FixtureAirportRepository
from src.exceptions import DataNotFoundError, FixtureValidationError
from src.models import DataMode


def test_fixture_bundle_loads_all_required_airports() -> None:
    bundle = load_fixture_bundle()

    assert {airport.airport_code for airport in bundle.airports} == REQUIRED_AIRPORT_CODES
    assert {traffic.airport_code for traffic in bundle.traffic} == REQUIRED_AIRPORT_CODES
    assert {operations.airport_code for operations in bundle.operations} == REQUIRED_AIRPORT_CODES
    assert {route.origin_airport_code for route in bundle.routes} == REQUIRED_AIRPORT_CODES


def test_every_fixture_record_is_visibly_illustrative() -> None:
    bundle = load_fixture_bundle()
    records = [*bundle.airports, *bundle.traffic, *bundle.routes, *bundle.operations]

    assert bundle.data_mode is DataMode.ILLUSTRATIVE_DEMO_DATA
    assert all(record.source.data_mode is DataMode.ILLUSTRATIVE_DEMO_DATA for record in records)
    assert "synthetic" in bundle.disclaimer.lower()
    assert "not official" in bundle.disclaimer.lower()


def test_all_time_series_records_share_manifest_period() -> None:
    bundle = load_fixture_bundle()

    assert all(record.period == bundle.period for record in bundle.traffic)
    assert all(record.period == bundle.period for record in bundle.routes)
    assert all(record.period == bundle.period for record in bundle.operations)


def test_repository_returns_complete_sfo_profile_inputs() -> None:
    repository = FixtureAirportRepository.from_default_fixtures()

    assert repository.get_airport("sfo").city == "San Francisco"
    assert repository.get_traffic("SFO").passengers > 0
    assert repository.get_operations("SFO").scheduled_departures > 0
    assert repository.get_routes("SFO")


def test_repository_lists_new_england_airports_deterministically() -> None:
    repository = FixtureAirportRepository.from_default_fixtures()

    airports = repository.list_airports(state_codes=set(NEW_ENGLAND_STATE_CODES))

    assert [airport.airport_code for airport in airports] == [
        "BDL",
        "BOS",
        "BTV",
        "MHT",
        "PVD",
        "PWM",
    ]


def test_repository_region_and_exclusion_filter() -> None:
    repository = FixtureAirportRepository.from_default_fixtures()

    airports = repository.list_airports(
        region="new england",
        excluded_airports={"bos"},
    )

    assert [airport.airport_code for airport in airports] == [
        "BDL",
        "BTV",
        "MHT",
        "PVD",
        "PWM",
    ]


def test_unknown_airport_is_rejected() -> None:
    repository = FixtureAirportRepository.from_default_fixtures()

    with pytest.raises(DataNotFoundError, match="Unsupported airport code"):
        repository.get_airport("XYZ")


def test_anc_fixture_contains_long_haul_and_short_haul_routes() -> None:
    repository = FixtureAirportRepository.from_default_fixtures()
    distances = [route.distance_miles for route in repository.get_routes("ANC")]

    assert any(distance >= 3000 for distance in distances)
    assert any(distance < 3000 for distance in distances)


def test_missing_airport_row_fails_coverage_validation(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "data"
    shutil.copytree(DEFAULT_DATA_DIR, fixture_dir)

    traffic_path = fixture_dir / "demo_traffic.csv"
    with traffic_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
        fieldnames = list(rows[0])
    with traffic_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for row in rows if row["airport_code"] != "SFO")

    with pytest.raises(FixtureValidationError, match="coverage mismatch"):
        load_fixture_bundle(fixture_dir)


def test_manifest_rejects_non_illustrative_mode(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "data"
    shutil.copytree(DEFAULT_DATA_DIR, fixture_dir)
    manifest_path = fixture_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data_mode"] = "CACHED PUBLIC DATA"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FixtureValidationError, match="must declare"):
        load_fixture_bundle(fixture_dir)


def _copy_fixture_dir(tmp_path: Path) -> Path:
    fixture_dir = tmp_path / "data"
    shutil.copytree(DEFAULT_DATA_DIR, fixture_dir)
    return fixture_dir


def _read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows, list(rows[0])


def _write_csv_rows(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_route_departures_form_complete_denominator() -> None:
    bundle = load_fixture_bundle()
    operations = {item.airport_code: item for item in bundle.operations}
    route_totals = {code: 0 for code in REQUIRED_AIRPORT_CODES}
    for route in bundle.routes:
        route_totals[route.origin_airport_code] += route.departures

    assert all(
        route_totals[code] == operations[code].performed_departures
        for code in REQUIRED_AIRPORT_CODES
    )
    assert route_totals["ANC"] == 13_400
    assert operations["ANC"].performed_departures == 13_400


def test_traffic_has_comparable_previous_period_metadata() -> None:
    bundle = load_fixture_bundle()

    assert all(record.previous_period_passengers is not None for record in bundle.traffic)
    assert all(record.previous_period is not None for record in bundle.traffic)
    assert all(record.previous_source is not None for record in bundle.traffic)
    assert all(record.previous_period.start_date.year == 2024 for record in bundle.traffic)


def test_repository_returns_defensive_copies() -> None:
    bundle = load_fixture_bundle()
    repository = FixtureAirportRepository(bundle)

    returned_traffic = repository.get_traffic("SFO")
    original_passengers = returned_traffic.passengers
    returned_traffic.passengers = 1
    returned_traffic.source.notes.append("caller mutation")

    returned_airport = repository.get_airport("SFO")
    returned_airport.name = "Caller changed name"

    returned_route = repository.get_routes("ANC")[0]
    returned_route.departures = 0

    bundle.traffic[0].passengers = 2

    assert repository.get_traffic("SFO").passengers == original_passengers
    assert "caller mutation" not in repository.get_traffic("SFO").source.notes
    assert repository.get_airport("SFO").name == "San Francisco International Airport"
    assert repository.get_routes("ANC")[0].departures > 0
    assert repository.bundle.traffic[0].passengers != 2


def test_duplicate_route_rows_are_rejected(tmp_path: Path) -> None:
    fixture_dir = _copy_fixture_dir(tmp_path)
    path = fixture_dir / "demo_routes.csv"
    rows, fieldnames = _read_csv_rows(path)
    rows.append(dict(rows[0]))
    _write_csv_rows(path, rows, fieldnames)

    with pytest.raises(FixtureValidationError, match="duplicate origin/destination/period"):
        load_fixture_bundle(fixture_dir)


def test_runway_count_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture_dir = _copy_fixture_dir(tmp_path)
    path = fixture_dir / "demo_airports.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows[0]["usable_runway_count"] = 5
    path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(FixtureValidationError, match="Runway count mismatch"):
        load_fixture_bundle(fixture_dir)


def test_traffic_passengers_above_seats_are_rejected(tmp_path: Path) -> None:
    fixture_dir = _copy_fixture_dir(tmp_path)
    path = fixture_dir / "demo_traffic.csv"
    rows, fieldnames = _read_csv_rows(path)
    rows[0]["passengers"] = str(int(rows[0]["available_seats"]) + 1)
    _write_csv_rows(path, rows, fieldnames)

    with pytest.raises(FixtureValidationError, match="passengers cannot exceed"):
        load_fixture_bundle(fixture_dir)


def test_route_passengers_above_seats_are_rejected(tmp_path: Path) -> None:
    fixture_dir = _copy_fixture_dir(tmp_path)
    path = fixture_dir / "demo_routes.csv"
    rows, fieldnames = _read_csv_rows(path)
    rows[0]["passengers"] = str(int(rows[0]["available_seats"]) + 1)
    _write_csv_rows(path, rows, fieldnames)

    with pytest.raises(FixtureValidationError, match="Route passengers cannot exceed"):
        load_fixture_bundle(fixture_dir)


def test_contradictory_cancellation_totals_are_rejected(tmp_path: Path) -> None:
    fixture_dir = _copy_fixture_dir(tmp_path)
    path = fixture_dir / "demo_operations.csv"
    rows, fieldnames = _read_csv_rows(path)
    rows[0]["reported_cancellations"] = str(int(rows[0]["reported_cancellations"]) + 1)
    _write_csv_rows(path, rows, fieldnames)

    with pytest.raises(FixtureValidationError, match="Cancellation totals are contradictory"):
        load_fixture_bundle(fixture_dir)


def test_incomplete_route_denominator_is_rejected(tmp_path: Path) -> None:
    fixture_dir = _copy_fixture_dir(tmp_path)
    path = fixture_dir / "demo_routes.csv"
    rows, fieldnames = _read_csv_rows(path)
    anc_row = next(row for row in rows if row["origin_airport_code"] == "ANC")
    anc_row["departures"] = str(int(anc_row["departures"]) - 1)
    _write_csv_rows(path, rows, fieldnames)

    with pytest.raises(FixtureValidationError, match="denominator is incomplete for ANC"):
        load_fixture_bundle(fixture_dir)
