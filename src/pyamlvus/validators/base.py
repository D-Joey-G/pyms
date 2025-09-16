# Base validator classes for pyamlvus
# Provides common validation infrastructure and patterns

from abc import ABC, abstractmethod
from typing import Any

from ..exceptions import SchemaConversionError


class BaseValidator(ABC):
    """Base class for all schema validators.

    Provides common validation patterns and error handling.
    """

    def __init__(self, schema_dict: dict[str, Any] | None = None):
        """Initialize validator with optional schema context.

        Args:
            schema_dict: Full schema dictionary for context-aware validation
        """
        self.schema_dict = schema_dict or {}

    @abstractmethod
    def validate(self, item: Any) -> None:
        """Validate a single item.

        Args:
            item: Item to validate

        Raises:
            SchemaConversionError: If validation fails
        """
        pass

    def validate_required_field(
        self, data: dict[str, Any], field_name: str, field_type: str = "field"
    ) -> Any:
        """Validate that a required field is present and return its value.

        Args:
            data: dictionary to check
            field_name: Name of the required field
            field_type: Type of field for error messages

        Returns:
            Value of the field if present

        Raises:
            SchemaConversionError: If field is missing
        """
        if field_name not in data:
            raise SchemaConversionError(
                f"{field_type.title()} definition missing required '{field_name}'"
            )
        return data[field_name]

    def validate_string_not_empty(self, value: str, field_name: str) -> None:
        """Validate that a string is not empty.

        Args:
            value: String to check
            field_name: Name of the field for error messages

        Raises:
            SchemaConversionError: If string is empty
        """
        if not value or not value.strip():
            raise SchemaConversionError(f"Field '{field_name}' cannot be empty")

    def validate_positive_integer(self, value: int, field_name: str) -> None:
        """Validate that a value is a positive integer.

        Args:
            value: Value to check
            field_name: Name of the field for error messages

        Raises:
            SchemaConversionError: If value is not a positive integer
        """
        if not isinstance(value, int) or value <= 0:
            raise SchemaConversionError(
                f"Field '{field_name}' must be a positive integer, got {value}"
            )
