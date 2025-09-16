# High-level API functions for pyamlvus
# Provides convenient one-liner functions for common operations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymilvus import CollectionSchema, MilvusClient


from .builders.schema import SchemaBuilder
from .exceptions import SchemaConversionError
from .parser import SchemaLoader


def load_schema(file_path: str | Path) -> "CollectionSchema":
    """Load and build a PyMilvus CollectionSchema from a YAML file.

    This is a convenience function that combines parsing, validation, and building
    into a single operation.

    Args:
        file_path: Path to the YAML schema file

    Returns:
        PyMilvus CollectionSchema object

    Raises:
        SchemaParseError: If the YAML file cannot be parsed
        SchemaValidationError: If the schema is invalid
        SchemaConversionError: If the schema cannot be converted to CollectionSchema

    Example:
        >>> schema = load_schema("my_schema.yaml")
        >>> # Use schema with Milvus client
        >>> client.create_collection(schema)
    """

    loader = SchemaLoader(file_path)
    schema_dict = loader.to_dict()
    builder = SchemaBuilder(schema_dict)
    return builder.build()


def load_schema_dict(file_path: str | Path) -> dict[str, Any]:
    """Load a YAML schema file and return the parsed dictionary.

    Args:
        file_path: Path to the YAML schema file

    Returns:
        Dictionary representation of the schema

    Raises:
        SchemaParseError: If the YAML file cannot be parsed
    """
    loader = SchemaLoader(file_path)
    return loader.to_dict()


def validate_schema_file(file_path: str | Path) -> list[str]:
    """Validate a YAML schema file and return any errors and warnings.

    Args:
        file_path: Path to the YAML schema file

    Returns:
        List of validation messages (errors and warnings).
        Empty if valid with no warnings.
        Errors come first, followed by warnings.

    Example:
        >>> errors = validate_schema_file("my_schema.yaml")
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Validation issue: {error}")
    """
    from . import validate_schema

    return validate_schema(file_path)


def build_collection_from_yaml(
    file_path: str | Path,
    **kwargs: Any,
) -> "CollectionSchema":
    """Build a CollectionSchema from a YAML schema file.

    Args:
        file_path: Path to the YAML schema file
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        PyMilvus CollectionSchema object

    Raises:
        SchemaParseError: If the YAML file cannot be parsed
        SchemaValidationError: If the schema is invalid
        SchemaConversionError: If the schema cannot be converted
    """
    return load_schema(file_path)


def build_collection_from_dict(
    schema_dict: dict[str, Any],
    **kwargs: Any,
) -> "CollectionSchema":
    """Build a CollectionSchema from a schema dictionary.

    Args:
        schema_dict: Schema dictionary (from load_schema_dict or manual creation)
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        PyMilvus CollectionSchema object

    Raises:
        SchemaValidationError: If the schema is invalid
        SchemaConversionError: If the schema cannot be converted
    """
    builder = SchemaBuilder(schema_dict)
    return builder.build()


def create_collection_from_yaml(
    file_path: str | Path,
    client: "MilvusClient",
    **kwargs: Any,
) -> Any:
    """Build a schema from YAML and create the collection via MilvusClient.

    Args:
        file_path: Path to the YAML schema file
        client: MilvusClient instance used to create the collection
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Result of ``MilvusClient.create_collection``
    """
    loader = SchemaLoader(file_path)
    schema_dict = loader.to_dict()
    collection_name = loader.name

    schema = build_collection_from_dict(schema_dict)
    return client.create_collection(collection_name=collection_name, schema=schema)


def create_collection_from_dict(
    schema_dict: dict[str, Any],
    client: "MilvusClient",
    **kwargs: Any,
) -> Any:
    """Build a schema from a dictionary and create the collection via MilvusClient.

    Args:
        schema_dict: Schema dictionary (from load_schema_dict or manual creation)
        client: MilvusClient instance used to create the collection
        **kwargs: Additional arguments (ignored for compatibility)

    Returns:
        Result of ``MilvusClient.create_collection``
    """
    collection_name = schema_dict.get("name")
    if not isinstance(collection_name, str):
        raise SchemaConversionError(
            "Schema dictionary must include a string 'name' field"
        )

    schema = build_collection_from_dict(schema_dict)
    return client.create_collection(collection_name=collection_name, schema=schema)
