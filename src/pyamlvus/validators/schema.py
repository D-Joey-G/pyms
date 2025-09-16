"""Schema-level validation orchestration and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..exceptions import MilvusYamlError, SchemaConversionError
from ..types import PYMILVUS_VERSION, PYMILVUS_VERSION_INFO, parse_version
from .field import FieldValidator
from .function import FunctionValidator
from .index import IndexValidator
from .result import ValidationResult, ValidationSeverity


@dataclass(slots=True)
class SchemaValidationContext:
    """Lightweight context describing validated schema components."""

    field_types: dict[str, str]
    fields: list[dict[str, Any]]


def resolve_autoindex_flag(schema: dict[str, Any]) -> bool:
    """Return whether autoindex is enabled, raising on conflicting settings."""

    autoindex_candidates = [
        ("autoindex", schema.get("autoindex")),
        ("enable_autoindex", schema.get("enable_autoindex")),
        ("use_autoindex", schema.get("use_autoindex")),
    ]

    settings = schema.get("settings")
    if isinstance(settings, dict):
        autoindex_candidates.extend(  # type: ignore[list-item]
            [
                ("settings.autoindex", settings.get("autoindex")),
                ("settings.enable_autoindex", settings.get("enable_autoindex")),
                ("settings.use_autoindex", settings.get("use_autoindex")),
            ]
        )

    found = [(key, value) for key, value in autoindex_candidates if value is not None]

    if len(found) > 1:
        keys = ", ".join(key for key, _ in found)
        raise SchemaConversionError(
            f"Multiple autoindex settings found: {keys}. Please specify only one "
            "autoindex setting."
        )

    if not found:
        return False

    key, value = found[0]
    if isinstance(value, bool):
        return value

    raise SchemaConversionError(
        f"Invalid autoindex value '{value}' (type: {type(value).__name__}) for key "
        f"'{key}'. Expected boolean value (true or false)"
    )


def ensure_runtime_requirements(schema: dict[str, Any]) -> None:
    """Validate runtime pymilvus requirements for the schema."""

    requirements = schema.get("pymilvus")
    if requirements is None:
        return

    if not isinstance(requirements, dict):
        raise SchemaConversionError(
            "Schema 'pymilvus' section must be a mapping with version bounds."
        )

    allowed_keys = {"min_version", "max_version", "version", "require", "exact_version"}
    unknown = set(requirements.keys()) - allowed_keys
    if unknown:
        raise SchemaConversionError(
            "Schema 'pymilvus' section contains unsupported keys: "
            + ", ".join(sorted(unknown))
        )

    def _parse(key: str) -> tuple[int, ...] | None:
        value = requirements.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise SchemaConversionError(
                f"Schema 'pymilvus.{key}' must be a version string"
            )
        try:
            return parse_version(value)
        except ValueError as exc:
            raise SchemaConversionError(
                f"Invalid version string for 'pymilvus.{key}': {value}"
            ) from exc

    min_version = _parse("min_version")
    max_version = _parse("max_version")
    version_exact = _parse("version") or _parse("require") or _parse("exact_version")

    if version_exact is not None and (min_version or max_version):
        raise SchemaConversionError(
            "Schema 'pymilvus' section cannot combine 'version' with min/max bounds."
        )

    if min_version and max_version and min_version > max_version:
        raise SchemaConversionError(
            "Schema 'pymilvus' min_version must be less than or equal to max_version"
        )

    current_tuple = PYMILVUS_VERSION_INFO

    if version_exact is not None and current_tuple != version_exact:
        requested = ".".join(str(part) for part in version_exact)
        raise SchemaConversionError(
            f"Schema requires pymilvus=={requested}, but current version is "
            "f{PYMILVUS_VERSION}."
        )

    if min_version is not None and current_tuple < min_version:
        requested = ".".join(str(part) for part in min_version)
        raise SchemaConversionError(
            f"Schema requires pymilvus>={requested}, but current version is "
            f"{PYMILVUS_VERSION}."
        )

    if max_version is not None and current_tuple > max_version:
        requested = ".".join(str(part) for part in max_version)
        raise SchemaConversionError(
            f"Schema requires pymilvus<={requested}, but current version is "
            f"{PYMILVUS_VERSION}."
        )


class SchemaValidator:
    """Co-ordinates all schema-level validators and aggregates results."""

    def __init__(self, schema_dict: dict[str, Any]):
        self.schema_dict = schema_dict

    def validate(self) -> tuple[ValidationResult, SchemaValidationContext]:
        result = ValidationResult()

        try:
            ensure_runtime_requirements(self.schema_dict)
        except SchemaConversionError as exc:
            result.add_error(str(exc))

        try:
            resolve_autoindex_flag(self.schema_dict)
        except SchemaConversionError as exc:
            result.add_error(str(exc))

        field_validator = FieldValidator()
        raw_fields = self.schema_dict.get("fields", []) or []
        validated_fields: list[dict[str, Any]] = []
        field_types: dict[str, str] = {}

        for field_def in raw_fields:
            try:
                field_validator.validate(field_def)
            except MilvusYamlError as exc:
                result.add_error(str(exc))
                continue

            name = field_def.get("name")
            field_type = field_def.get("type")
            if isinstance(name, str) and isinstance(field_type, str):
                field_types[name] = field_type
            validated_fields.append(field_def)

        index_validator = IndexValidator(field_types, self.schema_dict)
        indexes = self.schema_dict.get("indexes", []) or []

        for index_def in indexes:
            try:
                index_validator.validate(index_def)
            except MilvusYamlError as exc:
                result.add_error(str(exc))

        for warning in index_validator.get_index_warnings(
            set(field_types.keys()), indexes
        ):
            if warning.severity is ValidationSeverity.ERROR:
                result.add_error(warning.text)
            elif warning.severity is ValidationSeverity.WARNING:
                result.add_warning(warning.text)
            else:
                result.add_info(warning.text)

        function_validator = FunctionValidator(set(field_types.keys()), raw_fields)
        functions = self.schema_dict.get("functions", []) or []

        for func_def in functions:
            try:
                function_validator.validate(func_def)
            except MilvusYamlError as exc:
                result.add_error(str(exc))

        for message in function_validator.validate_function_index_relationships(
            functions, indexes
        ):
            if message.severity is ValidationSeverity.ERROR:
                result.add_error(message.text)
            elif message.severity is ValidationSeverity.WARNING:
                result.add_warning(message.text)
            else:
                result.add_info(message.text)

        context = SchemaValidationContext(
            field_types=field_types, fields=validated_fields
        )
        return result, context
