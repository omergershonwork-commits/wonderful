"""Analytics service adjustments for the official public-data repository."""
from __future__ import annotations

from typing import Any

from src.models import AirportAnalysis
from src.tools import AirportAnalyticsService


class PublicAirportAnalyticsService(AirportAnalyticsService):
    """Reuse deterministic tools while reporting the correct reference universe."""

    def _analysis(self, *args: Any, **kwargs: Any) -> AirportAnalysis:
        analysis = super()._analysis(*args, **kwargs)
        assumptions = [
            item.replace(
                "complete supported fixture-airport reference set",
                "complete supported public-airport reference set",
            )
            for item in analysis.assumptions
        ]
        return analysis.model_copy(update={"assumptions": assumptions})
