"""Typed application errors for data loading and repository access."""


class AirportIntelligenceError(Exception):
    """Base class for expected application-level errors."""


class FixtureValidationError(AirportIntelligenceError):
    """Raised when bundled fixture data is malformed or internally inconsistent."""


class DataNotFoundError(AirportIntelligenceError):
    """Raised when a requested airport or dataset is not available."""
