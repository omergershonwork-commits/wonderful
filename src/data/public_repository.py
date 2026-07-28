"""Public FAA/BTS repository compatible with the deterministic analytics service."""
from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Mapping

import httpx

from src.config import Settings
from src.data.public_clients import (
    BtsOnTimeClient,
    BtsT100Client,
    CachedHttpClient,
    FaaAirportMetadataClient,
    PublicDataConnectionError,
    PublicDataHTTPError,
    PublicDataInvalidResponseError,
    PublicDataNotFoundError,
    PublicDataTimeoutError,
    RetrievedPayload,
    _parse_number,
)
from src.models import (
    NEW_ENGLAND_STATE_CODES,
    AirportRecord,
    DataMode,
    DataPeriod,
    OperationalData,
    RegionName,
    RouteRecord,
    SourceMetadata,
    TrafficRecord,
)

SUPPORTED_AIRPORT_CODES = (
    "BOS",
    "BDL",
    "PVD",
    "MHT",
    "PWM",
    "BTV",
    "LAX",
    "SNA",
    "ANC",
    "SFO",
)


@dataclass(frozen=True, slots=True)
class PublicRepositoryBundle:
    """Minimal bundle contract consumed by :class:`AirportAnalyticsService`."""

    period: DataPeriod
    data_mode: DataMode
    disclaimer: str


class BtsT100SegmentClient:
    """Download and normalize official T-100 Segment (All Carriers) rows.

    BTS exposes the table through its TranStats download form. The annual ZIP is
    cached once and then filtered locally by origin airport, which prevents one
    large download per requested airport.
    """

    PAGE_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx"
    QUERY = {
        "gnoyr_VQ": "FMG",
        "QO_fu146_anzr": "Nv4+Pn44vr45",
    }

    def __init__(self, http: CachedHttpClient) -> None:
        self.http = http
        self._parsed_years: dict[int, dict[str, list[RouteRecord]]] = {}

    @staticmethod
    def _hidden(page: str, name: str) -> str:
        match = re.search(
            rf'id=["\']{re.escape(name)}["\'][^>]*value=["\']([^"\']+)',
            page,
            re.IGNORECASE,
        )
        if match is None:
            raise PublicDataInvalidResponseError(
                f"TranStats download page omitted hidden field {name}"
            )
        return match.group(1)

    def _download_year(
        self,
        year: int,
        *,
        force_refresh: bool = False,
    ) -> RetrievedPayload[bytes]:
        signature = f"POST T100 SEGMENT ALL CARRIERS {year}"
        key = self.http.cache.key_for(signature)
        if not force_refresh:
            cached = self.http.cache.get(key)
            if cached is not None:
                return RetrievedPayload(
                    cached.data,
                    DataMode.CACHED_PUBLIC_DATA,
                    cached.fetched_at,
                    cached.source_url,
                )

        try:
            page_response = self.http.client.get(self.PAGE_URL, params=self.QUERY)
            page_response.raise_for_status()
            page = page_response.text
            form = {
                "__EVENTARGUMENT": "",
                "__LASTFOCUS": "",
                "__VIEWSTATE": self._hidden(page, "__VIEWSTATE"),
                "__VIEWSTATEGENERATOR": self._hidden(
                    page,
                    "__VIEWSTATEGENERATOR",
                ),
                "__EVENTVALIDATION": self._hidden(page, "__EVENTVALIDATION"),
                "txtSearch": "",
                "btnDownload": "Download",
                "cboGeography": "All",
                "cboYear": str(year),
                "cboPeriod": "All",
                "chkAllVars": "on",
                "ORIGIN": "on",
                "DEST": "on",
                "DEST_CITY_NAME": "on",
                "YEAR": "on",
                "MONTH": "on",
                "DISTANCE": "on",
                "DEPARTURES_PERFORMED": "on",
                "PASSENGERS": "on",
                "SEATS": "on",
            }
            response = self.http.client.post(
                self.PAGE_URL,
                params=self.QUERY,
                data=form,
                headers={"Referer": str(page_response.url)},
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise PublicDataTimeoutError(
                "BTS T-100 segment download exceeded the configured timeout"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise PublicDataHTTPError(
                f"BTS T-100 segment download returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise PublicDataConnectionError(
                "BTS T-100 segment download could not be reached"
            ) from exc

        payload = response.content
        if len(payload) > self.http.max_download_bytes:
            raise PublicDataInvalidResponseError(
                "BTS T-100 segment download exceeded the configured size limit"
            )
        if not zipfile.is_zipfile(io.BytesIO(payload)):
            raise PublicDataInvalidResponseError(
                "BTS T-100 segment endpoint did not return a ZIP archive"
            )
        stored = self.http.cache.put(
            key,
            payload,
            source_url=str(response.url),
        )
        return RetrievedPayload(
            stored.data,
            DataMode.LIVE_PUBLIC_DATA,
            stored.fetched_at,
            stored.source_url,
        )

    @staticmethod
    def _row_value(row: Mapping[str, str], *names: str) -> str | None:
        normalized = {str(key).strip().upper(): value for key, value in row.items()}
        for name in names:
            if name in normalized:
                return normalized[name]
        return None

    def _parse_year(
        self,
        year: int,
        *,
        force_refresh: bool = False,
    ) -> dict[str, list[RouteRecord]]:
        if year in self._parsed_years and not force_refresh:
            return self._parsed_years[year]
        retrieved = self._download_year(year, force_refresh=force_refresh)
        period = DataPeriod(
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31),
            label=f"Calendar year {year}",
        )
        source = SourceMetadata(
            source_name="BTS T-100 Segment (All Carriers)",
            data_mode=retrieved.data_mode,
            retrieved_at=retrieved.fetched_at,
            period=period,
            source_url=retrieved.source_url,
            notes=[
                "Official non-stop segment data downloaded from BTS TranStats.",
                "Rows are aggregated by origin, destination, and distance.",
            ],
        )
        aggregates: dict[tuple[str, str, int], dict[str, Any]] = {}
        try:
            with zipfile.ZipFile(io.BytesIO(retrieved.payload)) as archive:
                names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if not names:
                    raise PublicDataInvalidResponseError(
                        "BTS T-100 segment ZIP contained no CSV file"
                    )
                with archive.open(names[0]) as raw:
                    reader = csv.DictReader(
                        io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
                    )
                    for row in reader:
                        row_year = self._row_value(row, "YEAR")
                        if row_year and int(_parse_number(row_year, field_name="year", integer=True)) != year:
                            continue
                        origin = str(self._row_value(row, "ORIGIN") or "").strip().upper()
                        destination = str(self._row_value(row, "DEST") or "").strip().upper()
                        if (
                            len(origin) != 3
                            or not origin.isalpha()
                            or len(destination) != 3
                            or not destination.isalpha()
                            or origin == destination
                        ):
                            continue
                        distance = int(
                            _parse_number(
                                self._row_value(row, "DISTANCE"),
                                field_name="distance",
                                integer=True,
                            )
                        )
                        departures = int(
                            _parse_number(
                                self._row_value(
                                    row,
                                    "DEPARTURES_PERFORMED",
                                    "DEPPERFORMED",
                                ),
                                field_name="departures performed",
                                integer=True,
                            )
                        )
                        passengers = int(
                            _parse_number(
                                self._row_value(row, "PASSENGERS"),
                                field_name="passengers",
                                integer=True,
                            )
                        )
                        seats_value = self._row_value(row, "SEATS")
                        seats = (
                            int(
                                _parse_number(
                                    seats_value,
                                    field_name="seats",
                                    integer=True,
                                )
                            )
                            if seats_value not in {None, ""}
                            else 0
                        )
                        if min(distance, departures, passengers, seats) < 0:
                            raise PublicDataInvalidResponseError(
                                "BTS T-100 segment rows cannot contain negative values"
                            )
                        if passengers > seats and seats > 0:
                            raise PublicDataInvalidResponseError(
                                "BTS T-100 segment passengers cannot exceed seats"
                            )
                        key = (origin, destination, distance)
                        item = aggregates.setdefault(
                            key,
                            {
                                "departures": 0,
                                "passengers": 0,
                                "seats": 0,
                                "destination_name": self._row_value(
                                    row,
                                    "DEST_CITY_NAME",
                                ),
                            },
                        )
                        item["departures"] += departures
                        item["passengers"] += passengers
                        item["seats"] += seats
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError, csv.Error) as exc:
            raise PublicDataInvalidResponseError(
                "BTS T-100 segment archive could not be parsed"
            ) from exc

        by_origin: dict[str, list[RouteRecord]] = {}
        for (origin, destination, distance), values in aggregates.items():
            if values["departures"] == 0:
                continue
            by_origin.setdefault(origin, []).append(
                RouteRecord(
                    origin_airport_code=origin,
                    destination_airport_code=destination,
                    destination_name=(
                        str(values["destination_name"]).strip()
                        if values["destination_name"]
                        else None
                    ),
                    distance_miles=distance,
                    departures=values["departures"],
                    passengers=values["passengers"],
                    available_seats=values["seats"] or None,
                    period=period,
                    source=source,
                )
            )
        self._parsed_years[year] = by_origin
        return by_origin

    def fetch_routes(
        self,
        airport_code: str,
        year: int,
        *,
        force_refresh: bool = False,
    ) -> list[RouteRecord]:
        code = airport_code.strip().upper()
        rows = self._parse_year(year, force_refresh=force_refresh).get(code, [])
        if not rows:
            raise PublicDataNotFoundError(
                f"BTS T-100 segment data contained no routes for {code} in {year}"
            )
        return [row.model_copy(deep=True) for row in rows]


class PublicAirportRepository:
    """Lazy repository backed only by official FAA and BTS public sources."""

    def __init__(
        self,
        *,
        year: int,
        http: CachedHttpClient,
        faa: FaaAirportMetadataClient,
        t100: BtsT100Client,
        on_time: BtsOnTimeClient,
        segments: BtsT100SegmentClient,
    ) -> None:
        self.year = year
        self.http = http
        self.faa = faa
        self.t100 = t100
        self.on_time = on_time
        self.segments = segments
        self.bundle = PublicRepositoryBundle(
            period=DataPeriod(
                start_date=date(year, 1, 1),
                end_date=date(year, 12, 31),
                label=f"Calendar year {year}",
            ),
            data_mode=DataMode.CACHED_PUBLIC_DATA,
            disclaimer=(
                "Answers use official FAA and BTS public data through a local disk cache. "
                "No synthetic fixture values are used in live-data mode."
            ),
        )
        self._airports: dict[str, AirportRecord] = {}
        self._traffic: dict[str, TrafficRecord] = {}
        self._routes: dict[str, list[RouteRecord]] = {}
        self._operations: dict[str, OperationalData] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "PublicAirportRepository":
        http = CachedHttpClient.from_settings(settings)
        return cls(
            year=settings.public_data_year,
            http=http,
            faa=FaaAirportMetadataClient(http),
            t100=BtsT100Client(http),
            on_time=BtsOnTimeClient(http),
            segments=BtsT100SegmentClient(http),
        )

    @property
    def supported_airport_codes(self) -> tuple[str, ...]:
        return SUPPORTED_AIRPORT_CODES

    def close(self) -> None:
        self.http.close()

    def _require_code(self, airport_code: str) -> str:
        code = airport_code.strip().upper()
        if code not in SUPPORTED_AIRPORT_CODES:
            raise KeyError(f"unsupported airport code: {code}")
        return code

    @staticmethod
    def _source(source: SourceMetadata, *, note: str | None = None) -> SourceMetadata:
        notes = [
            *source.notes,
            f"Adapter retrieval mode: {source.data_mode.value}.",
            "Normalized by the cache-backed public repository.",
        ]
        if note:
            notes.append(note)
        return source.model_copy(
            update={
                "data_mode": DataMode.CACHED_PUBLIC_DATA,
                "notes": notes,
            },
            deep=True,
        )

    def get_airport(self, airport_code: str) -> AirportRecord:
        code = self._require_code(airport_code)
        if code not in self._airports:
            row = self.faa.fetch_airport(code)
            self._airports[code] = AirportRecord(
                airport_code=code,
                name=row.name,
                city=row.city,
                state_code=row.state_code,
                region=(
                    RegionName.NEW_ENGLAND
                    if row.state_code in NEW_ENGLAND_STATE_CODES
                    else None
                ),
                source=self._source(row.source),
            )
        return self._airports[code].model_copy(deep=True)

    def get_traffic(self, airport_code: str) -> TrafficRecord:
        code = self._require_code(airport_code)
        if code not in self._traffic:
            current = self.t100.fetch_airport_year(code, self.year)
            previous = self.t100.fetch_airport_year(code, self.year - 1)
            current_source = self._source(current.source)
            previous_source = self._source(previous.source)
            self._traffic[code] = TrafficRecord(
                airport_code=code,
                period=current_source.period,
                passengers=current.passengers,
                previous_period_passengers=previous.passengers,
                previous_period=previous_source.period,
                previous_source=previous_source,
                available_seats=current.seats,
                source=current_source,
            )
        return self._traffic[code].model_copy(deep=True)

    def get_routes(self, airport_code: str) -> list[RouteRecord]:
        code = self._require_code(airport_code)
        if code not in self._routes:
            rows = self.segments.fetch_routes(code, self.year)
            self._routes[code] = [
                row.model_copy(update={"source": self._source(row.source)}, deep=True)
                for row in rows
            ]
        return [row.model_copy(deep=True) for row in self._routes[code]]

    def get_operations(self, airport_code: str) -> OperationalData:
        code = self._require_code(airport_code)
        if code not in self._operations:
            routes = self.get_routes(code)
            performed_departures = sum(route.departures for route in routes)
            monthly = [
                self.on_time.fetch_origin_month(code, self.year, month)
                for month in range(1, 13)
            ]
            on_time_scheduled = sum(row.scheduled_departures for row in monthly)
            on_time_cancelled = sum(row.cancelled_departures for row in monthly)
            if on_time_scheduled <= 0 or on_time_cancelled >= on_time_scheduled:
                raise PublicDataInvalidResponseError(
                    f"BTS on-time data cannot derive a cancellation rate for {code}"
                )
            cancellation_rate = on_time_cancelled / on_time_scheduled
            estimated_cancellations = round(
                performed_departures * cancellation_rate / (1 - cancellation_rate)
            )
            scheduled_departures = performed_departures + estimated_cancellations
            delay_weight = sum(
                row.performed_departures
                for row in monthly
                if row.average_departure_delay_minutes is not None
            )
            taxi_weight = sum(
                row.performed_departures
                for row in monthly
                if row.average_taxi_out_minutes is not None
            )
            average_delay = (
                sum(
                    row.average_departure_delay_minutes * row.performed_departures
                    for row in monthly
                    if row.average_departure_delay_minutes is not None
                )
                / delay_weight
                if delay_weight
                else None
            )
            average_taxi = (
                sum(
                    row.average_taxi_out_minutes * row.performed_departures
                    for row in monthly
                    if row.average_taxi_out_minutes is not None
                )
                / taxi_weight
                if taxi_weight
                else None
            )
            retrieved_at = max(row.source.retrieved_at for row in monthly)
            source = SourceMetadata(
                source_name="BTS T-100 Segment and On-Time Performance",
                data_mode=DataMode.CACHED_PUBLIC_DATA,
                retrieved_at=retrieved_at,
                period=self.bundle.period,
                source_url=monthly[-1].source.source_url,
                notes=[
                    "Performed departures come from T-100 Segment so route denominators align.",
                    "Cancellation rate and delay/taxi averages come from twelve BTS on-time monthly files.",
                    "Reported cancellations are scaled to the T-100 performed-departure universe.",
                ],
            )
            self._operations[code] = OperationalData(
                airport_code=code,
                period=self.bundle.period,
                scheduled_departures=scheduled_departures,
                performed_departures=performed_departures,
                reported_cancellations=estimated_cancellations,
                average_departure_delay_minutes=average_delay,
                average_taxi_out_minutes=average_taxi,
                usable_runway_count=None,
                source=source,
            )
        return self._operations[code].model_copy(deep=True)

    def list_airports(
        self,
        *,
        region: str | None = None,
        state_codes: set[str] | None = None,
        excluded_airports: set[str] | None = None,
    ) -> list[AirportRecord]:
        normalized_states = {value.strip().upper() for value in (state_codes or set())}
        excluded = {value.strip().upper() for value in (excluded_airports or set())}
        rows = [self.get_airport(code) for code in SUPPORTED_AIRPORT_CODES]
        if region is not None:
            rows = [row for row in rows if row.region and row.region.value == region]
        if normalized_states:
            rows = [row for row in rows if row.state_code in normalized_states]
        rows = [row for row in rows if row.airport_code not in excluded]
        return sorted(rows, key=lambda row: row.airport_code)
