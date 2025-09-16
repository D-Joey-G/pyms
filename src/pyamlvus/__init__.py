# pyamlvus package
# A Python library for managing Milvus schemas through YAML configuration files

from pathlib import Path

# High-level API
from .api import (
    build_collection_from_dict,
    build_collection_from_yaml,
    create_collection_from_dict,
    create_collection_from_yaml,
    load_schema,
    load_schema_dict,
    validate_schema_file,
)

# Core components
from .builders.schema import SchemaBuilder
from .parser import SchemaLoader
from .validators import ValidationResult
from .validators.schema import SchemaValidator

__version__ = "0.1.0"

__all__ = [
    # Core components
    "SchemaLoader",
    "SchemaBuilder",
    "validate_schema",
    "validate_schema_result",
    # High-level API
    "load_schema",
    "load_schema_dict",
    "validate_schema_file",
    "build_collection_from_yaml",
    "build_collection_from_dict",
    "create_collection_from_yaml",
    "create_collection_from_dict",
]


def validate_schema_result(file_path: str | Path) -> ValidationResult:
    """Validate a YAML schema file and return structured messages."""
    result = ValidationResult()

    try:
        loader = SchemaLoader(file_path)
        # Trigger parser validations explicitly by accessing key properties
        _ = loader.name
        _ = loader.fields
        _ = loader.settings
        _ = loader.indexes
        _ = loader.functions

        schema_dict = loader.to_dict()
        schema_validator = SchemaValidator(schema_dict)
        validation_result, context = schema_validator.validate()
        result.extend(validation_result)

        if result.has_errors():
            return result

        builder = SchemaBuilder(schema_dict, context=context)
        try:
            builder.build()
        except Exception as exc:  # pragma: no cover - defensive
            result.add_error(str(exc))

    except Exception as exc:
        result.add_error(str(exc))

    return result


def validate_schema(file_path: str | Path) -> list[str]:
    """Legacy helper returning validation messages as prefixed strings."""

    return validate_schema_result(file_path).as_strings()
