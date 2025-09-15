# Type system module for pyamlvus
# Provides type mappings, constants, and type-related utilities

from .compat import (
    OPTIONAL_INDEX_REQUIREMENTS,
    OPTIONAL_INDEX_SUPPORT,
    OPTIONAL_TYPE_REQUIREMENTS,
    OPTIONAL_TYPE_SUPPORT,
    PYMILVUS_VERSION,
    PYMILVUS_VERSION_INFO,
    parse_version,
    version_at_least,
    version_at_most,
    version_matches,
)
from .constants import (
    BINARY_METRICS,
    FLOAT_METRICS,
    GPU_INDEX_TYPES,
    RECOMMENDED_INDEX_TYPES,
    REQUIRED_PARAMS,
    VALID_INDEX_TYPES,
)
from .mappings import TYPE_MAPPING

__all__ = [
    "TYPE_MAPPING",
    "FLOAT_METRICS",
    "BINARY_METRICS",
    "REQUIRED_PARAMS",
    "VALID_INDEX_TYPES",
    "RECOMMENDED_INDEX_TYPES",
    "GPU_INDEX_TYPES",
    "OPTIONAL_TYPE_SUPPORT",
    "OPTIONAL_TYPE_REQUIREMENTS",
    "OPTIONAL_INDEX_SUPPORT",
    "OPTIONAL_INDEX_REQUIREMENTS",
    "PYMILVUS_VERSION",
    "PYMILVUS_VERSION_INFO",
    "parse_version",
    "version_at_least",
    "version_at_most",
    "version_matches",
]
