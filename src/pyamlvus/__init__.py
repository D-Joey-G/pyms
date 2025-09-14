# pyamlvus package
# A Python library for managing Milvus schemas through YAML configuration files

from pathlib import Path

# High-level API
from .api import (
    create_collection_from_dict,
    create_collection_from_yaml,
    load_schema,
    load_schema_dict,
    validate_schema_file,
)

# Core components
from .builders.schema import SchemaBuilder
from .parser import SchemaLoader

__version__ = "0.1.0"

__all__ = [
    # Core components
    "SchemaLoader",
    "SchemaBuilder",
    "validate_schema",
    # High-level API
    "load_schema",
    "load_schema_dict",
    "validate_schema_file",
    "create_collection_from_yaml",
    "create_collection_from_dict",
]


def validate_schema(file_path: str | Path) -> list[str]:
    """Validate a YAML schema file and return any errors and warnings.

    Args:
        file_path: Path to the YAML schema file

    Returns:
        list of validation messages (errors and warnings). Empty if valid with no
        warnings.
        Errors come first, followed by warnings.
    """
    errors = []
    warnings = []

    try:
        # First validate parser-level constraints to get better error messages
        loader = SchemaLoader(file_path)
        # Trigger parser validations explicitly by accessing properties
        _ = loader.name  # Validates name exists
        _ = loader.fields  # Validates fields exist and not empty
        _ = loader.settings  # Validates settings format
        _ = loader.indexes  # Validates indexes format
        _ = loader.functions  # Validates functions format

        # Then validate builder-level constraints
        schema_dict = loader.to_dict()
        builder = SchemaBuilder(schema_dict)

        # Validate the schema build
        builder.build()

        # Validate function definitions explicitly (even if not applied)
        for func_def in builder.functions or []:
            builder.validate_function(func_def)

        # Get index warnings
        warnings = builder.get_index_warnings()

        # Get function-index relationship warnings/errors
        func_index_messages = builder.get_function_index_warnings()
        warnings.extend(func_index_messages)

    except Exception as e:
        errors.append(str(e))

    # Return errors first, then warnings
    return errors + warnings
