"""Shared validation utilities for collection and alias names."""

import re

# Collection and alias names must start with a letter and contain only
# letters, digits, and underscores.
NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


def is_valid_collection_name(name: str) -> bool:
    if not isinstance(name, str) or not name:
        return False
    if name.startswith("_"):
        return False
    return bool(NAME_RE.match(name))


def is_valid_alias(alias: str) -> bool:
    if not isinstance(alias, str) or not alias:
        return False
    if alias.startswith("_"):
        return False
    return bool(NAME_RE.match(alias))
