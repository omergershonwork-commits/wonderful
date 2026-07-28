"""Thin, cached clients for FAA airport metadata and BTS aviation datasets."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generic, Mapping, TypeVar
from urllib.parse import urlencode

import httpx
from pydantic import Field, ValidationError, field_validator, model_validator

from src.config import Settings
from src.models import DataMode, DataPeriod, DomainModel, SourceMetadata
from src.numeric_tokens import NumericTokenError, parse_canonical_decimal

T = TypeVar("T")


class PublicDataError(RuntimeError):
    """Base error for public-data boundaries."""


class PublicDataTimeoutError(PublicDataError):
    """Raised when a public endpoint exceeds the configured timeout."""


class PublicDataConnectionError(PublicDataError):
    """Raised when a public endpoint cannot be reached."""


class PublicDataHTTPError(PublicDataError):
    """Raised when a public endpoint returns a non-success status."""


class PublicDataInvalidResponseError(PublicDataError):
    """Raised when a response cannot be parsed or normalized safely."""


class PublicDataNotFoundError(PublicDataError):
    """Raised when a valid query returns no matching public data."""


def _airport_code(value: str) -> str:
    code = value.strip().upper()
    if len(code) != 3 or not code.isalpha():
        raise ValueError("airport code must contain exactly three letters")
    return code


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_number(
    value: Any,
    *,
    field_name: str,
    integer: bool = False,
) -> int | float:
    if value is None or str(value).strip() == "":
        raise PublicDataInvalidResponseError(f"missing numeric field: {field_name}")
    token = str(value).strip()
    try:
        number = parse_canonical_decimal(token, field_name=field_name)
    except NumericTokenError as exc:
        raise PublicDataInvalidResponseError(
            f"invalid numeric field: {field_name}"
        ) from exc
    if not math.isfinite(number):
        raise PublicDataInvalidResponseError(
            f"non-finite numeric field: {field_name}"
        )
    if integer:
        if not number.is_integer():
            raise PublicDataInvalidResponseError(
                f"expected integer field: {field_name}"
            )
        return int(number)
    return number


def _parse_iso_date(value: Any, *, field_name: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise PublicDataInvalidResponseError(f"missing date field: {field_name}")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError:
        try:
            return date.fromisoformat(normalized[:10])
        except ValueError as exc:
            raise PublicDataInvalidResponseError(
                f"invalid date field: {field_name}"
            ) from exc


def _month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


class FaaAirportMetadata(DomainModel):
    """Normalized identity fields from the FAA airport search boundary."""

    airport_code: str
    icao_code: str | None = None
    name: str = Field(min_length=1)
    city: str = Field(min_length=1)
    state_code: str = Field(min_length=2, max_length=2)
    source: SourceMetadata

    @field_validator("airport_code", mode="before")
    @classmethod
    def normalize_airport_code(cls, value: str) -> str:
        return _airport_code(value)

    @field_validator("icao_code", mode="before")
    @classmethod
    def normalize_icao(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        if len(normalized) != 4 or not normalized.isalpha():
            raise ValueError("ICAO code must contain exactly four letters")
        return normalized

    @field_validator("state_code", mode="before")
    @classmethod
    def normalize_state(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 2 or not normalized.isalpha():
            raise ValueError("state code must contain exactly two letters")
        return normalized


class BtsT100AirportSummary(DomainModel):
    """Normalized T-100 totals through the latest reported month."""

    airport_code: str
    year: int = Field(ge=1900, le=2200)
    reporting_date: date
    departures: int = Field(ge=0)
    passengers: int = Field(ge=0)
    seats: int = Field(ge=0)
    load_factor: float = Field(ge=0, le=1)
    source: SourceMetadata

    @field_validator("airport_code", mode="before")
    @classmethod
    def normalize_airport_code(cls, value: str) -> str:
        return _airport_code(value)

    @model_validator(mode="after")
    def values_are_consistent(self) -> "BtsT100AirportSummary":
        if self.reporting_date.year != self.year:
            raise ValueError("reporting_date must be inside the requested year")
        if self.passengers > self.seats:
            raise ValueError("passengers cannot exceed seats")
        return self


class BtsOnTimeAirportSummary(DomainModel):
    """Normalized monthly on-time operations for one origin airport."""

    airport_code: str
    year: int = Field(ge=1987, le=2200)
    month: int = Field(ge=1, le=12)
    scheduled_departures: int = Field(ge=0)
    performed_departures: int = Field(ge=0)
    cancelled_departures: int = Field(ge=0)
    average_departure_delay_minutes: float | None = Field(default=None, ge=0)
    average_taxi_out_minutes: float | None = Field(default=None, ge=0)
    source: SourceMetadata

    @field_validator("airport_code", mode="before")
    @classmethod
    def normalize_airport_code(cls, value: str) -> str:
        return _airport_code(value)

    @model_validator(mode="after")
    def totals_are_consistent(self) -> "BtsOnTimeAirportSummary":
        if self.performed_departures + self.cancelled_departures != self.scheduled_departures:
            raise ValueError(
                "performed plus cancelled departures must equal scheduled departures"
            )
        return self


@dataclass(frozen=True, slots=True)
class CachePayload:
    data: bytes
    fetched_at: datetime
    source_url: str


class DiskResponseCache:
    """Small atomic file cache with payload integrity and TTL checks."""

    def __init__(self, directory: Path | str, *, ttl_seconds: int = 86_400) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds cannot be negative")
        self.directory = Path(directory)
        self.ttl_seconds = ttl_seconds
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(signature: str) -> str:
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()

    def _paths(self, key: str) -> tuple[Path, Path]:
        return (
            self.directory / f"{key}.meta.json",
            self.directory / f"{key}.payload",
        )

    def get(self, key: str, *, now: datetime | None = None) -> CachePayload | None:
        meta_path, payload_path = self._paths(key)
        if not meta_path.is_file() or not payload_path.is_file():
            return None
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(metadata["fetched_at"])
            expires_at = datetime.fromisoformat(metadata["expires_at"])
            payload = payload_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
                raise ValueError("cache payload checksum mismatch")
            if (now or _utc_now()) >= expires_at:
                return None
            return CachePayload(payload, fetched_at, str(metadata["source_url"]))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self.delete(key)
            return None

    def put(
        self,
        key: str,
        payload: bytes,
        *,
        source_url: str,
        fetched_at: datetime | None = None,
    ) -> CachePayload:
        fetched = fetched_at or _utc_now()
        expires = fetched + timedelta(seconds=self.ttl_seconds)
        metadata = {
            "fetched_at": fetched.isoformat(),
            "expires_at": expires.isoformat(),
            "source_url": source_url,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        meta_path, payload_path = self._paths(key)
        self._atomic_write(payload_path, payload)
        self._atomic_write(
            meta_path,
            json.dumps(metadata, sort_keys=True).encode("utf-8"),
        )
        return CachePayload(payload, fetched, source_url)

    def delete(self, key: str) -> None:
        for path in self._paths(key):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            dir=path.parent,
        )
        try:
            with os.fdopen(file_descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


@dataclass(frozen=True, slots=True)
class RetrievedPayload(Generic[T]):
    payload: T
    data_mode: DataMode
    fetched_at: datetime
    source_url: str


class CachedHttpClient:
    """Typed HTTP boundary shared by the three thin adapters."""

    def __init__(
        self,
        *,
        cache: DiskResponseCache,
        timeout_seconds: float = 20,
        max_download_bytes: int = 100 * 1024 * 1024,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_download_bytes <= 0:
            raise ValueError("max_download_bytes must be greater than zero")
        self.cache = cache
        self.max_download_bytes = max_download_bytes
        self.client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "User-Agent": "airport-investment-intelligence/1.0",
                **dict(headers or {}),
            },
            transport=transport,
            follow_redirects=True,
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> "CachedHttpClient":
        headers: dict[str, str] = {}
        if settings.bts_app_token.strip():
            headers["X-App-Token"] = settings.bts_app_token.strip()
        return cls(
            cache=DiskResponseCache(
                settings.public_data_cache_dir,
                ttl_seconds=settings.public_data_cache_ttl_seconds,
            ),
            timeout_seconds=settings.http_timeout_seconds,
            max_download_bytes=(
                settings.public_data_max_download_mb * 1024 * 1024
            ),
            headers=headers,
            transport=transport,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "CachedHttpClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _signature(
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
    ) -> str:
        normalized = urlencode(
            sorted((str(key), str(value)) for key, value in (params or {}).items())
        )
        return f"{method.upper()} {url}?{normalized}"

    def invalidate(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        key = self.cache.key_for(self._signature("GET", url, params))
        self.cache.delete(key)

    def get_bytes(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> RetrievedPayload[bytes]:
        signature = self._signature("GET", url, params)
        key = self.cache.key_for(signature)
        if not force_refresh:
            cached = self.cache.get(key)
            if cached is not None:
                return RetrievedPayload(
                    cached.data,
                    DataMode.CACHED_PUBLIC_DATA,
                    cached.fetched_at,
                    cached.source_url,
                )
        try:
            chunks: list[bytes] = []
            size = 0
            with self.client.stream("GET", url, params=params) as response:
                response.raise_for_status()
                source_url = str(response.request.url)
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self.max_download_bytes:
                        raise PublicDataInvalidResponseError(
                            "response exceeded configured size limit of "
                            f"{self.max_download_bytes} bytes"
                        )
                    chunks.append(chunk)
            payload = b"".join(chunks)
        except PublicDataInvalidResponseError:
            raise
        except httpx.TimeoutException as exc:
            raise PublicDataTimeoutError(
                f"public endpoint timed out: {url}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise PublicDataHTTPError(
                "public endpoint returned HTTP "
                f"{exc.response.status_code}: {url}"
            ) from exc
        except httpx.RequestError as exc:
            raise PublicDataConnectionError(
                f"could not connect to public endpoint: {url}"
            ) from exc
        stored = self.cache.put(key, payload, source_url=source_url)
        return RetrievedPayload(
            stored.data,
            DataMode.LIVE_PUBLIC_DATA,
            stored.fetched_at,
            stored.source_url,
        )

    def get_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> RetrievedPayload[Any]:
        retrieved = self.get_bytes(
            url,
            params=params,
            force_refresh=force_refresh,
        )
        try:
            payload = json.loads(retrieved.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.invalidate(url, params=params)
            raise PublicDataInvalidResponseError(
                f"endpoint returned invalid JSON: {url}"
            ) from exc
        return RetrievedPayload(
            payload,
            retrieved.data_mode,
            retrieved.fetched_at,
            retrieved.source_url,
        )

    def get_text(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> RetrievedPayload[str]:
        retrieved = self.get_bytes(
            url,
            params=params,
            force_refresh=force_refresh,
        )
        try:
            payload = retrieved.payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            self.invalidate(url, params=params)
            raise PublicDataInvalidResponseError(
                f"endpoint returned invalid UTF-8: {url}"
            ) from exc
        return RetrievedPayload(
            payload,
            retrieved.data_mode,
            retrieved.fetched_at,
            retrieved.source_url,
        )


class FaaAirportMetadataClient:
    """Read basic airport identity metadata from the FAA procedure search page."""

    DEFAULT_URL = (
        "https://www.faa.gov/air_traffic/flight_info/aeronav/"
        "digital_products/dtpp/search/results/"
    )
    _ROW_PATTERN = re.compile(
        r"<tr[^>]*>\s*<td[^>]*>(?P<state>[A-Z]{2})</td>\s*"
        r"<td[^>]*>(?P<city>.*?)</td>\s*<td[^>]*>(?P<name>.*?)</td>\s*"
        r"<td[^>]*>\s*(?P<code>[A-Z]{3})\s*\((?P<icao>[A-Z]{4})\)",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, http: CachedHttpClient, *, base_url: str = DEFAULT_URL) -> None:
        self.http = http
        self.base_url = base_url

    def fetch_airport(
        self,
        airport_code: str,
        *,
        force_refresh: bool = False,
    ) -> FaaAirportMetadata:
        code = _airport_code(airport_code)
        params = {"ident": code}
        response = self.http.get_text(
            self.base_url,
            params=params,
            force_refresh=force_refresh,
        )
        try:
            match = self._ROW_PATTERN.search(response.payload)
            if match is None:
                raise PublicDataNotFoundError(
                    f"FAA returned no airport metadata for {code}"
                )
            row_code = match.group("code").upper()
            if row_code != code:
                raise PublicDataInvalidResponseError(
                    f"FAA response airport {row_code} did not match requested airport {code}"
                )
            return FaaAirportMetadata(
                airport_code=code,
                icao_code=match.group("icao"),
                name=_clean_html(match.group("name")),
                city=_clean_html(match.group("city")),
                state_code=match.group("state"),
                source=SourceMetadata(
                    source_name="FAA Terminal Procedures airport search",
                    data_mode=response.data_mode,
                    retrieved_at=response.fetched_at,
                    source_url=response.source_url,
                    notes=[
                        "Thin metadata adapter: identity fields only; not a complete FAA airport master record."
                    ],
                ),
            )
        except PublicDataNotFoundError:
            raise
        except (PublicDataInvalidResponseError, ValidationError, ValueError) as exc:
            self.http.invalidate(self.base_url, params=params)
            if isinstance(exc, PublicDataInvalidResponseError):
                raise
            raise PublicDataInvalidResponseError(
                "FAA response violated the normalized airport metadata contract"
            ) from exc


def _clean_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(without_tags).split())


class BtsT100Client:
    """Read normalized summaries from BTS T-100 origin-airport data."""

    DATASET_ID = "r495-tyji"
    DEFAULT_URL = f"https://data.bts.gov/resource/{DATASET_ID}.json"

    def __init__(self, http: CachedHttpClient, *, base_url: str = DEFAULT_URL) -> None:
        self.http = http
        self.base_url = base_url

    def fetch_airport_year(
        self,
        airport_code: str,
        year: int,
        *,
        force_refresh: bool = False,
    ) -> BtsT100AirportSummary:
        code = _airport_code(airport_code)
        if not 1900 <= year <= 2200:
            raise ValueError("year must be between 1900 and 2200")
        params = {
            "$select": (
                "origin_airport_code,year,max(reporting_month) AS reporting_month,"
                "sum(total_departures) AS total_departures,"
                "sum(total_passengers) AS total_passengers,"
                "sum(total_seats) AS total_seats"
            ),
            "$where": f"origin_airport_code='{code}' AND year='{year}'",
            "$group": "origin_airport_code,year",
            "$limit": 1,
        }
        response = self.http.get_json(
            self.base_url,
            params=params,
            force_refresh=force_refresh,
        )
        try:
            if not isinstance(response.payload, list):
                raise PublicDataInvalidResponseError(
                    "BTS T-100 response must be a JSON list"
                )
            if not response.payload:
                raise PublicDataNotFoundError(
                    f"BTS T-100 returned no data for {code} in {year}"
                )
            row = response.payload[0]
            if not isinstance(row, dict):
                raise PublicDataInvalidResponseError(
                    "BTS T-100 row must be an object"
                )
            row_code = _airport_code(str(row.get("origin_airport_code", "")))
            row_year = int(
                _parse_number(row.get("year"), field_name="year", integer=True)
            )
            if row_code != code or row_year != year:
                raise PublicDataInvalidResponseError(
                    "BTS T-100 row does not match the requested airport and year"
                )
            reporting_date = _parse_iso_date(
                row.get("reporting_month"),
                field_name="reporting_month",
            )
            if reporting_date.year != year:
                raise PublicDataInvalidResponseError(
                    "BTS T-100 reporting month is outside the requested year"
                )
            departures = int(
                _parse_number(
                    row.get("total_departures"),
                    field_name="total_departures",
                    integer=True,
                )
            )
            passengers = int(
                _parse_number(
                    row.get("total_passengers"),
                    field_name="total_passengers",
                    integer=True,
                )
            )
            seats = int(
                _parse_number(
                    row.get("total_seats"),
                    field_name="total_seats",
                    integer=True,
                )
            )
            period_end = _month_end(year, reporting_date.month)
            complete_year = reporting_date.month == 12
            period = DataPeriod(
                start_date=date(year, 1, 1),
                end_date=period_end,
                label=(
                    f"Calendar year {year}"
                    if complete_year
                    else f"Year-to-date through {year}-{reporting_date.month:02d}"
                ),
            )
            return BtsT100AirportSummary(
                airport_code=code,
                year=year,
                reporting_date=reporting_date,
                departures=departures,
                passengers=passengers,
                seats=seats,
                load_factor=0.0 if seats == 0 else passengers / seats,
                source=SourceMetadata(
                    source_name="BTS AFF T-100 Segment Summary By Origin Airport",
                    data_mode=response.data_mode,
                    retrieved_at=response.fetched_at,
                    period=period,
                    source_url=response.source_url,
                    notes=[
                        f"BTS Socrata dataset ID: {self.DATASET_ID}",
                        (
                            "Values cover the complete calendar year."
                            if complete_year
                            else "Values are year-to-date through the latest reporting month and must not be treated as a complete calendar year."
                        ),
                    ],
                ),
            )
        except PublicDataNotFoundError:
            raise
        except (
            PublicDataInvalidResponseError,
            ValidationError,
            ValueError,
            TypeError,
        ) as exc:
            self.http.invalidate(self.base_url, params=params)
            if isinstance(exc, PublicDataInvalidResponseError):
                raise
            raise PublicDataInvalidResponseError(
                "BTS T-100 response violated the normalized summary contract"
            ) from exc


class BtsOnTimeClient:
    """Aggregate one airport from BTS's official monthly on-time ZIP archive."""

    BASE_URL = "https://transtats.bts.gov/PREZIP"
    FILE_TEMPLATE = (
        "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_"
        "{year}_{month}.zip"
    )

    def __init__(self, http: CachedHttpClient, *, base_url: str = BASE_URL) -> None:
        self.http = http
        self.base_url = base_url.rstrip("/")

    def fetch_origin_month(
        self,
        airport_code: str,
        year: int,
        month: int,
        *,
        force_refresh: bool = False,
    ) -> BtsOnTimeAirportSummary:
        code = _airport_code(airport_code)
        if not 1987 <= year <= 2200:
            raise ValueError("year must be between 1987 and 2200")
        if not 1 <= month <= 12:
            raise ValueError("month must be between 1 and 12")
        filename = self.FILE_TEMPLATE.format(year=year, month=month)
        url = f"{self.base_url}/{filename}"
        response = self.http.get_bytes(url, force_refresh=force_refresh)
        try:
            summary = _aggregate_on_time_zip(
                response.payload,
                code,
                max_uncompressed_bytes=self.http.max_download_bytes * 10,
            )
            if summary["scheduled"] == 0:
                raise PublicDataNotFoundError(
                    "BTS on-time archive returned no origin rows for "
                    f"{code} in {year}-{month:02d}"
                )
            period = DataPeriod(
                start_date=date(year, month, 1),
                end_date=_month_end(year, month),
                label=f"{year}-{month:02d}",
            )
            return BtsOnTimeAirportSummary(
                airport_code=code,
                year=year,
                month=month,
                scheduled_departures=summary["scheduled"],
                performed_departures=summary["performed"],
                cancelled_departures=summary["cancelled"],
                average_departure_delay_minutes=_mean_or_none(summary["delays"]),
                average_taxi_out_minutes=_mean_or_none(summary["taxi_out"]),
                source=SourceMetadata(
                    source_name="BTS Airline On-Time Performance monthly archive",
                    data_mode=response.data_mode,
                    retrieved_at=response.fetched_at,
                    period=period,
                    source_url=response.source_url,
                    notes=[
                        "Aggregated locally from rows whose Origin matches the requested airport.",
                        "Cancelled flights are excluded from performed departures and delay/taxi averages.",
                    ],
                ),
            )
        except PublicDataNotFoundError:
            raise
        except (PublicDataInvalidResponseError, ValidationError, ValueError) as exc:
            self.http.invalidate(url)
            if isinstance(exc, PublicDataInvalidResponseError):
                raise
            raise PublicDataInvalidResponseError(
                "BTS on-time response violated the normalized summary contract"
            ) from exc


def _aggregate_on_time_zip(
    payload: bytes,
    airport_code: str,
    *,
    max_uncompressed_bytes: int = 1_000_000_000,
) -> dict[str, Any]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (zipfile.BadZipFile, OSError) as exc:
        raise PublicDataInvalidResponseError(
            "BTS on-time response is not a valid ZIP archive"
        ) from exc
    with archive:
        csv_infos = [
            info
            for info in archive.infolist()
            if info.filename.casefold().endswith(".csv")
        ]
        if len(csv_infos) != 1:
            raise PublicDataInvalidResponseError(
                "BTS on-time ZIP must contain exactly one CSV file"
            )
        csv_info = csv_infos[0]
        if csv_info.flag_bits & 0x1:
            raise PublicDataInvalidResponseError(
                "BTS on-time ZIP CSV must not be encrypted"
            )
        if csv_info.file_size > max_uncompressed_bytes:
            raise PublicDataInvalidResponseError(
                "BTS on-time ZIP exceeds the uncompressed size limit"
            )
        scheduled = cancelled = performed = 0
        delays: list[float] = []
        taxi_out: list[float] = []
        with archive.open(csv_info) as raw_stream:
            text_stream = io.TextIOWrapper(
                raw_stream,
                encoding="utf-8-sig",
                newline="",
            )
            reader = csv.DictReader(text_stream)
            required = {"Origin", "Cancelled", "TaxiOut"}
            fields = set(reader.fieldnames or ())
            if not required.issubset(fields) or not (
                {"DepDelay", "DepDelayMinutes"} & fields
            ):
                raise PublicDataInvalidResponseError(
                    "BTS on-time CSV is missing Origin, Cancelled, TaxiOut, "
                    "or departure-delay fields"
                )
            for row in reader:
                if str(row.get("Origin", "")).strip().upper() != airport_code:
                    continue
                scheduled += 1
                if _parse_flag(row.get("Cancelled")):
                    cancelled += 1
                    continue
                performed += 1
                delay = _optional_nonnegative(
                    (
                        row.get("DepDelayMinutes")
                        if str(row.get("DepDelayMinutes", "")).strip()
                        else row.get("DepDelay")
                    ),
                    field_name="departure delay",
                )
                taxi = _optional_nonnegative(
                    row.get("TaxiOut"),
                    field_name="taxi-out",
                )
                if delay is not None:
                    delays.append(delay)
                if taxi is not None:
                    taxi_out.append(taxi)
        return {
            "scheduled": scheduled,
            "cancelled": cancelled,
            "performed": performed,
            "delays": delays,
            "taxi_out": taxi_out,
        }


def _parse_flag(value: Any) -> bool:
    if value is None or str(value).strip() == "":
        raise PublicDataInvalidResponseError(
            "missing Cancelled flag in BTS on-time CSV"
        )
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise PublicDataInvalidResponseError(
            "invalid Cancelled flag in BTS on-time CSV"
        ) from exc
    if not math.isfinite(number) or number not in {0.0, 1.0}:
        raise PublicDataInvalidResponseError(
            "Cancelled flag must be exactly 0 or 1"
        )
    return number == 1.0


def _optional_nonnegative(value: Any, *, field_name: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(str(value).strip())
    except ValueError as exc:
        raise PublicDataInvalidResponseError(
            f"invalid {field_name} value in BTS on-time CSV"
        ) from exc
    if not math.isfinite(number):
        raise PublicDataInvalidResponseError(
            f"non-finite {field_name} value in BTS on-time CSV"
        )
    if number < 0:
        raise PublicDataInvalidResponseError(
            f"negative {field_name} value in BTS on-time CSV"
        )
    return number


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)
