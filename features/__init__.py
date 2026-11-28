"""Feature extraction library for the X5 uplift pipeline (Prathmesh workstream).

The public surface is the pure feature functions in :mod:`features.aggregates`.
These take one client's cleaned transactions plus a profile record and return the
fixed set of gold ``client_features`` columns. They hold no file paths or global
state so the batch builder and the streaming consumer can call identical code.
"""

from features.aggregates import (
    FEATURE_NAMES,
    client_features_from_tx,
    empty_client_features,
)

__all__ = [
    "FEATURE_NAMES",
    "client_features_from_tx",
    "empty_client_features",
]
