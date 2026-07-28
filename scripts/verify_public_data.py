"""Manual Step-14 live smoke test: retrieve BTS T-100 once, then prove cache reuse."""
from __future__ import annotations

from src.config import Settings
from src.data.public_clients import BtsT100Client, CachedHttpClient
from src.models import DataMode


def main() -> None:
    settings = Settings()
    with CachedHttpClient.from_settings(settings) as http:
        client = BtsT100Client(http)
        first = client.fetch_airport_year("SFO", 2025, force_refresh=True)
        second = client.fetch_airport_year("SFO", 2025)
    if first.source.data_mode is not DataMode.LIVE_PUBLIC_DATA:
        raise RuntimeError("forced refresh did not return live public data")
    if second.source.data_mode is not DataMode.CACHED_PUBLIC_DATA:
        raise RuntimeError("second request did not use the disk cache")
    print(
        f"verified SFO {first.year}: passengers={first.passengers}, "
        f"departures={first.departures}, cache={second.source.data_mode.value}"
    )


if __name__ == "__main__":
    main()
