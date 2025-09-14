# Field validation logic for pyamlvus
# Handles validation of individual field definitions

from typing import Any

from ..exceptions import SchemaConversionError, UnsupportedTypeError
from ..types import TYPE_MAPPING
from .base import BaseValidator

# Validation ranges and helper functions
VECTOR_DIM_RANGES = {
    "float_vector": (1, 32768),  # Milvus limits
    "binary_vector": (1, 32768 * 8),  # bits
    "sparse_float_vector": (1, 2**31 - 1),
}


def validate_vector_dim(field_name: str, dim: int, vector_type: str):
    """Validate vector dimension against Milvus limits.

    Args:
        field_name: Name of the field for error messages
        dim: Dimension value to validate
        vector_type: Type of vector field

    Raises:
        SchemaConversionError: If dimension is out of range
    """
    min_dim, max_dim = VECTOR_DIM_RANGES[vector_type]
    if not (min_dim <= dim <= max_dim):
        raise SchemaConversionError(
            f"Field '{field_name}': {vector_type} dimension {dim} must be between "
            f"{min_dim}-{max_dim}"
        )


VARCHAR_LENGTH_RANGE = (1, 65535)  # Milvus limit


def validate_varchar_length(field_name: str, max_length: int):
    """Validate VARCHAR max_length against Milvus limits.

    Args:
        field_name: Name of the field for error messages
        max_length: Max length value to validate

    Raises:
        SchemaConversionError: If max_length is out of range
    """
    min_len, max_len = VARCHAR_LENGTH_RANGE
    if not (min_len <= max_length <= max_len):
        raise SchemaConversionError(
            f"VARCHAR field '{field_name}' max_length {max_length} must be between "
            f"{min_len}-{max_len}"
        )


ARRAY_CAPACITY_RANGE = (1, 4096)  # Milvus limit


def validate_array_params(field_name: str, max_capacity: int):
    """Validate array max_capacity against Milvus limits.

    Args:
        field_name: Name of the field for error messages
        max_capacity: Max capacity value to validate

    Raises:
        SchemaConversionError: If max_capacity is out of range
    """
    min_cap, max_cap = ARRAY_CAPACITY_RANGE
    if not (min_cap <= max_capacity <= max_cap):
        raise SchemaConversionError(
            f"Array field '{field_name}' max_capacity {max_capacity} must be between "
            f"{min_cap}-{max_cap}"
        )


class FieldValidator(BaseValidator):
    """Validator for individual field definitions."""

    def validate(self, item: dict[str, Any]) -> None:
        """Validate a field definition.

        Args:
            field_def: Field definition dictionary

        Raises:
            SchemaConversionError: If field definition is invalid
        """
        # Validate required fields
        name = item.get("name")
        if name is None:
            raise SchemaConversionError("Field missing required 'name' attribute")

        field_type = item.get("type")
        if field_type is None:
            raise SchemaConversionError(
                f"Field '{name}' missing required 'type' attribute"
            )

        # Validate field name
        self.validate_string_not_empty(name, "name")
        self._validate_field_name_format(name)

        # Validate field type
        self._validate_field_type(field_type, name)

        # Validate type-specific parameters
        self._validate_type_specific_params(item, field_type, name)

        # Validate nullable flag if present
        if "nullable" in item and not isinstance(item.get("nullable"), bool):
            raise SchemaConversionError(
                f"Field '{name}': 'nullable' must be a boolean value"
            )

        # Validate enable_match flag if present
        if "enable_match" in item and not isinstance(item.get("enable_match"), bool):
            raise SchemaConversionError(
                f"Field '{name}': 'enable_match' must be a boolean value"
            )

    def _validate_field_name_format(self, name: str) -> None:
        """Validate field name format.

        Args:
            name: Field name to validate

        Raises:
            SchemaConversionError: If name format is invalid
        """
        if not name.replace("_", "").replace("-", "").isalnum():
            raise SchemaConversionError(
                f"Field name '{name}' contains invalid characters. "
                "Only alphanumeric characters, underscores, and hyphens are allowed."
            )

        if name.startswith("_"):
            raise SchemaConversionError(
                f"Field name '{name}' cannot start with underscore (reserved for "
                f"system fields)"
            )

    def _validate_field_type(self, field_type: str, field_name: str) -> None:
        """Validate that field type is supported.

        Args:
            field_type: Type string to validate
            field_name: Name of the field for error messages

        Raises:
            UnsupportedTypeError: If field type is not supported
        """
        if field_type not in TYPE_MAPPING:
            supported_types = sorted(TYPE_MAPPING.keys())
            raise UnsupportedTypeError(
                f"Unsupported field type '{field_type}' for field '{field_name}'. "
                f"Supported types: {supported_types}"
            )

    def _validate_type_specific_params(
        self, field_def: dict[str, Any], field_type: str, field_name: str
    ) -> None:
        """Validate parameters specific to the field type.

        Args:
            field_def: Complete field definition
            field_type: Type of the field
            field_name: Name of the field

        Raises:
            SchemaConversionError: If required parameters are missing or invalid
        """
        if field_type == "varchar":
            self._validate_varchar_params(field_def, field_name)
        elif field_type in {"float_vector", "binary_vector"}:
            self._validate_vector_params(field_def, field_type, field_name)
        elif field_type == "sparse_float_vector":
            # sparse_float_vector doesn't require dim parameter
            pass
        elif field_type == "array":
            self._validate_array_params(field_def, field_name)

    def _validate_varchar_params(
        self, field_def: dict[str, Any], field_name: str
    ) -> None:
        """Validate VARCHAR field parameters.

        Args:
            field_def: Field definition
            field_name: Name of the field

        Raises:
            SchemaConversionError: If max_length is missing or invalid
        """
        if "max_length" not in field_def:
            raise SchemaConversionError(
                f"VARCHAR field '{field_name}' missing required 'max_length' parameter"
            )

        max_length = field_def["max_length"]
        self.validate_positive_integer(max_length, f"{field_name}.max_length")

        # Use the centralized validation function
        validate_varchar_length(field_name, max_length)

    def _validate_vector_params(
        self, field_def: dict[str, Any], field_type: str, field_name: str
    ) -> None:
        """Validate vector field parameters.

        Args:
            field_def: Field definition
            field_type: Type of the vector field
            field_name: Name of the field

        Raises:
            SchemaConversionError: If dim is missing or invalid
        """
        if "dim" not in field_def:
            raise SchemaConversionError(
                f"Vector field '{field_name}' missing required 'dim' parameter"
            )

        dim = field_def["dim"]
        self.validate_positive_integer(dim, f"{field_name}.dim")

        # Use the centralized validation function
        validate_vector_dim(field_name, dim, field_type)

    def _validate_array_params(
        self, field_def: dict[str, Any], field_name: str
    ) -> None:
        """Validate array field parameters.

        Args:
            field_def: Field definition
            field_name: Name of the field

        Raises:
            SchemaConversionError: If required parameters are missing or invalid
        """
        if "element_type" not in field_def:
            raise SchemaConversionError(
                f"Array field '{field_name}' missing required 'element_type' parameter"
            )

        if "max_capacity" not in field_def:
            raise SchemaConversionError(
                f"Array field '{field_name}' missing required 'max_capacity' parameter"
            )

        # Validate element_type is supported
        element_type = field_def["element_type"]
        if element_type not in TYPE_MAPPING:
            raise UnsupportedTypeError(
                f"Unsupported array element type '{element_type}' for field "
                f"'{field_name}'"
            )

        # Validate max_capacity
        max_capacity = field_def["max_capacity"]
        self.validate_positive_integer(max_capacity, f"{field_name}.max_capacity")

        # Use the centralized validation function
        validate_array_params(field_name, max_capacity)
