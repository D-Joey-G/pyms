"""Compatibility helpers for optional pymilvus features."""

import re

from pymilvus import __version__ as pymilvus_version


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse a version string into a numeric tuple for comparison."""

    parts: list[int] = []
    for component in re.split(r"[.+-]", version):
        if component.isdigit():
            parts.append(int(component))
        else:
            break
    return tuple(parts)


PYMILVUS_VERSION = pymilvus_version
PYMILVUS_VERSION_INFO = _version_tuple(pymilvus_version)


def _supports(min_version: tuple[int, ...]) -> bool:
    """Return True if the installed pymilvus version meets the minimum."""

    return PYMILVUS_VERSION_INFO >= min_version


def _version_str(version_tuple: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version_tuple)


OPTIONAL_TYPE_MIN_VERSIONS: dict[str, tuple[int, ...]] = {
    "float16_vector": (2, 6, 0),
    "bfloat16_vector": (2, 6, 0),
    "int8_vector": (2, 6, 0),
}

OPTIONAL_INDEX_MIN_VERSIONS: dict[str, tuple[int, ...]] = {
    "GPU_CAGRA": (2, 6, 0),
    "GPU_BRUTE_FORCE": (2, 6, 0),
}


OPTIONAL_TYPE_SUPPORT: dict[str, bool] = {
    name: _supports(version) for name, version in OPTIONAL_TYPE_MIN_VERSIONS.items()
}

OPTIONAL_INDEX_SUPPORT: dict[str, bool] = {
    name: _supports(version) for name, version in OPTIONAL_INDEX_MIN_VERSIONS.items()
}


OPTIONAL_TYPE_REQUIREMENTS: dict[str, str] = {
    name: f"Requires pymilvus>={_version_str(version)}"
    for name, version in OPTIONAL_TYPE_MIN_VERSIONS.items()
}

OPTIONAL_INDEX_REQUIREMENTS: dict[str, str] = {
    name: f"Requires pymilvus>={_version_str(version)}"
    for name, version in OPTIONAL_INDEX_MIN_VERSIONS.items()
}


def parse_version(version: str) -> tuple[int, ...]:
    """Public helper to parse version strings consistently."""

    if not isinstance(version, str) or not version.strip():
        raise ValueError("Version string must be a non-empty string")
    return _version_tuple(version)


def version_matches(version: str) -> bool:
    """Check if the installed pymilvus matches the provided exact version."""

    return PYMILVUS_VERSION_INFO == parse_version(version)


def version_at_least(min_version: str) -> bool:
    """Check if the installed pymilvus is at least the provided version."""

    return PYMILVUS_VERSION_INFO >= parse_version(min_version)


def version_at_most(max_version: str) -> bool:
    """Check if the installed pymilvus is at most the provided version."""

    return PYMILVUS_VERSION_INFO <= parse_version(max_version)
