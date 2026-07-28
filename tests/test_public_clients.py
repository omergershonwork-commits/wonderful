from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from src.data.public_clients import (
    BtsOnTimeClient,
    BtsT100Client,
    CachedHttpClient,
    DiskResponseCache,
    FaaAirportMetadataClient,
    PublicDataHTTPError,
    PublicDataInvalidResponseError,
    PublicDataNotFoundError,
    PublicDataTimeoutError,
)
from src.models import DataMode


def _transport(handler):
    return httpx.MockTransport(handler)


def _http(
    tmp_path: Path,
    handler,
    *,
    ttl: int = 3600,
    max_bytes: int = 10_000_000,
) -> CachedHttpClient:
    return CachedHttpClient(
        cache=DiskResponseCache(tmp_path / "cache", ttl_seconds=ttl),
        timeout_seconds=2,
        max_download_bytes=max_bytes,
        transport=_transport(handler),
    )


def test_t100_live_response_is_normalized_then_reused_from_cache(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.host == "data.bts.gov"
        assert request.url.path == "/resource/r495-tyji.json"
        params = dict(request.url.params)
        assert "origin_airport_code='SFO'" in params["$where"]
        assert params["$limit"] == "1"
        assert params["$group"] == "origin_airport_code,year"
        assert "sum(total_passengers)" in params["$select"]
        return httpx.Response(
            200,
            json=[
                {
                    "origin_airport_code": "SFO",
                    "year": "2025",
                    "reporting_month": "2025-12-01T00:00:00.000",
                    "total_departures": "182000",
                    "total_passengers": "54000000",
                    "total_seats": "62000000",
                }
            ],
            request=request,
        )

    client = BtsT100Client(_http(tmp_path, handler))
    live = client.fetch_airport_year("sfo", 2025)
    cached = client.fetch_airport_year("SFO", 2025)

    assert calls == 1
    assert live.departures == 182_000
    assert live.load_factor == pytest.approx(54_000_000 / 62_000_000)
    assert live.source.data_mode is DataMode.LIVE_PUBLIC_DATA
    assert cached.source.data_mode is DataMode.CACHED_PUBLIC_DATA
    assert cached.source.retrieved_at == live.source.retrieved_at
    assert cached.source.period.start_date.isoformat() == "2025-01-01"
    assert cached.source.period.end_date.isoformat() == "2025-12-31"
    assert cached.source.period.label == "Calendar year 2025"


def test_t100_partial_year_uses_latest_reporting_month_as_period_end(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "origin_airport_code": "SFO",
                    "year": "2026",
                    "reporting_month": "2026-08-01T00:00:00.000",
                    "total_departures": "120000",
                    "total_passengers": "35000000",
                    "total_seats": "42000000",
                }
            ],
            request=request,
        )

    result = BtsT100Client(_http(tmp_path, handler)).fetch_airport_year(
        "SFO", 2026
    )
    assert result.source.period.start_date.isoformat() == "2026-01-01"
    assert result.source.period.end_date.isoformat() == "2026-08-31"
    assert result.source.period.label == "Year-to-date through 2026-08"
    assert any("must not be treated as a complete calendar year" in note for note in result.source.notes)


def test_t100_rejects_mismatched_or_impossible_rows(tmp_path: Path) -> None:
    responses = [
        [{"origin_airport_code": "LAX", "year": "2025"}],
        [
            {
                "origin_airport_code": "SFO",
                "year": "2025",
                "reporting_month": "2025-12-01T00:00:00",
                "total_departures": "1",
                "total_passengers": "2",
                "total_seats": "1",
            }
        ],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0), request=request)

    client = BtsT100Client(_http(tmp_path, handler, ttl=0))
    with pytest.raises(PublicDataInvalidResponseError, match="does not match"):
        client.fetch_airport_year("SFO", 2025, force_refresh=True)
    with pytest.raises(PublicDataInvalidResponseError, match="normalized summary"):
        client.fetch_airport_year("SFO", 2025, force_refresh=True)


def test_t100_rejects_reporting_month_outside_requested_year(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "origin_airport_code": "SFO",
                    "year": "2025",
                    "reporting_month": "2024-12-01T00:00:00",
                    "total_departures": "1",
                    "total_passengers": "1",
                    "total_seats": "1",
                }
            ],
            request=request,
        )

    client = BtsT100Client(_http(tmp_path, handler))
    with pytest.raises(PublicDataInvalidResponseError, match="outside"):
        client.fetch_airport_year("SFO", 2025)


def test_t100_empty_result_is_typed_not_found(tmp_path: Path) -> None:
    client = BtsT100Client(
        _http(tmp_path, lambda request: httpx.Response(200, json=[], request=request))
    )
    with pytest.raises(PublicDataNotFoundError):
        client.fetch_airport_year("SFO", 2025)


def test_faa_html_identity_is_normalized_and_cached(tmp_path: Path) -> None:
    calls = 0
    body = """
    <table><tbody><tr>
      <td>CA</td><td>SAN FRANCISCO</td><td>SAN FRANCISCO INTL</td>
      <td>SFO (KSFO)</td><td>SW-2</td><td></td><td>APD</td>
    </tr></tbody></table>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.params["ident"] == "SFO"
        return httpx.Response(200, text=body, request=request)

    client = FaaAirportMetadataClient(_http(tmp_path, handler))
    live = client.fetch_airport("sfo")
    cached = client.fetch_airport("SFO")

    assert calls == 1
    assert live.name == "SAN FRANCISCO INTL"
    assert live.city == "SAN FRANCISCO"
    assert live.state_code == "CA"
    assert live.icao_code == "KSFO"
    assert live.source.data_mode is DataMode.LIVE_PUBLIC_DATA
    assert cached.source.data_mode is DataMode.CACHED_PUBLIC_DATA


def test_faa_response_without_requested_airport_fails_closed(tmp_path: Path) -> None:
    body = (
        "<table><tr><td>CA</td><td>LOS ANGELES</td>"
        "<td>LOS ANGELES INTL</td><td>LAX (KLAX)</td></tr></table>"
    )
    client = FaaAirportMetadataClient(
        _http(
            tmp_path,
            lambda request: httpx.Response(200, text=body, request=request),
        )
    )
    with pytest.raises(PublicDataInvalidResponseError, match="did not match"):
        client.fetch_airport("SFO")


def _zip_rows(rows: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("On_Time.csv", rows)
    return buffer.getvalue()


def _on_time_zip() -> bytes:
    return _zip_rows(
        "Origin,Cancelled,DepDelay,TaxiOut\n"
        "SFO,0,12,18\n"
        "SFO,1,,\n"
        "SFO,0,3,22\n"
        "LAX,0,30,15\n"
    )


def test_on_time_zip_is_aggregated_for_requested_origin(tmp_path: Path) -> None:
    payload = _on_time_zip()
    client = BtsOnTimeClient(
        _http(
            tmp_path,
            lambda request: httpx.Response(200, content=payload, request=request),
        )
    )
    result = client.fetch_origin_month("SFO", 2026, 5)

    assert result.scheduled_departures == 3
    assert result.cancelled_departures == 1
    assert result.performed_departures == 2
    assert result.average_departure_delay_minutes == pytest.approx(7.5)
    assert result.average_taxi_out_minutes == pytest.approx(20.0)
    assert result.source.period.end_date.isoformat() == "2026-05-31"
    assert result.source.data_mode is DataMode.LIVE_PUBLIC_DATA


@pytest.mark.parametrize("flag", ["-1", "0.5", "2"])
def test_on_time_rejects_cancellation_flags_outside_zero_or_one(
    tmp_path: Path,
    flag: str,
) -> None:
    payload = _zip_rows(
        "Origin,Cancelled,DepDelay,TaxiOut\n"
        f"SFO,{flag},12,18\n"
    )
    client = BtsOnTimeClient(
        _http(
            tmp_path,
            lambda request: httpx.Response(200, content=payload, request=request),
        )
    )
    with pytest.raises(PublicDataInvalidResponseError, match="exactly 0 or 1"):
        client.fetch_origin_month("SFO", 2026, 5)


@pytest.mark.parametrize(
    "row",
    [
        "SFO,0,-3,18\n",
        "SFO,0,12,-1\n",
    ],
)
def test_on_time_rejects_negative_delay_or_taxi_values(
    tmp_path: Path,
    row: str,
) -> None:
    payload = _zip_rows("Origin,Cancelled,DepDelay,TaxiOut\n" + row)
    client = BtsOnTimeClient(
        _http(
            tmp_path,
            lambda request: httpx.Response(200, content=payload, request=request),
        )
    )
    with pytest.raises(PublicDataInvalidResponseError, match="negative"):
        client.fetch_origin_month("SFO", 2026, 5)


def test_on_time_zip_requires_expected_csv_contract(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("bad.csv", "Origin,Cancelled\nSFO,0\n")
    client = BtsOnTimeClient(
        _http(
            tmp_path,
            lambda request: httpx.Response(
                200,
                content=buffer.getvalue(),
                request=request,
            ),
        )
    )
    with pytest.raises(PublicDataInvalidResponseError, match="missing"):
        client.fetch_origin_month("SFO", 2026, 5)


def test_on_time_no_matching_origin_is_typed_not_found(tmp_path: Path) -> None:
    client = BtsOnTimeClient(
        _http(
            tmp_path,
            lambda request: httpx.Response(
                200,
                content=_on_time_zip(),
                request=request,
            ),
        )
    )
    with pytest.raises(PublicDataNotFoundError):
        client.fetch_origin_month("ANC", 2026, 5)


def test_http_timeout_and_status_errors_are_typed(tmp_path: Path) -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(PublicDataTimeoutError):
        _http(tmp_path / "timeout", timeout).get_json(
            "https://data.bts.gov/test"
        )

    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"}, request=request)

    with pytest.raises(PublicDataHTTPError, match="503"):
        _http(tmp_path / "http", unavailable).get_json(
            "https://data.bts.gov/test"
        )


def test_invalid_json_and_size_limit_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(PublicDataInvalidResponseError, match="invalid JSON"):
        _http(
            tmp_path / "json",
            lambda request: httpx.Response(
                200,
                content=b"not-json",
                request=request,
            ),
        ).get_json("https://data.bts.gov/test")

    with pytest.raises(PublicDataInvalidResponseError, match="size limit"):
        _http(
            tmp_path / "size",
            lambda request: httpx.Response(
                200,
                content=b"12345",
                request=request,
            ),
            max_bytes=4,
        ).get_bytes("https://data.bts.gov/test")


def test_corrupt_or_expired_cache_is_not_returned(tmp_path: Path) -> None:
    cache = DiskResponseCache(tmp_path / "cache", ttl_seconds=10)
    key = cache.key_for("GET https://example.test?")
    fetched = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cache.put(
        key,
        b"payload",
        source_url="https://example.test",
        fetched_at=fetched,
    )

    assert cache.get(key, now=fetched + timedelta(seconds=9)) is not None
    assert cache.get(key, now=fetched + timedelta(seconds=10)) is None

    meta_path, payload_path = cache._paths(key)
    meta_path.write_text(json.dumps({"bad": True}), encoding="utf-8")
    assert cache.get(key, now=fetched) is None
    assert not meta_path.exists()
    assert not payload_path.exists()


@pytest.mark.parametrize(
    "row",
    [
        "SFO,0,nan,18\n",
        "SFO,0,inf,18\n",
        "SFO,0,-inf,18\n",
        "SFO,0,12,nan\n",
        "SFO,0,12,inf\n",
        "SFO,0,12,-inf\n",
    ],
)
def test_on_time_rejects_nan_and_infinite_delay_or_taxi_values(
    tmp_path: Path,
    row: str,
) -> None:
    payload = _zip_rows("Origin,Cancelled,DepDelay,TaxiOut\n" + row)
    client = BtsOnTimeClient(
        _http(
            tmp_path,
            lambda request: httpx.Response(200, content=payload, request=request),
        )
    )
    with pytest.raises(PublicDataInvalidResponseError, match="non-finite"):
        client.fetch_origin_month("SFO", 2026, 5)
