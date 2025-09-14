# Custom exceptions for pyamlvus


class MilvusYamlError(Exception):
    """Base exception for all pyamlvus errors."""

    pass


class SchemaParseError(MilvusYamlError):
    """Raised when YAML parsing fails."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ):
        self.file_path = file_path
        self.line = line
        self.column = column

        location_info = ""
        if file_path:
            location_info = f" in {file_path}"
        if line is not None:
            location_info += f" at line {line}"
        if column is not None:
            location_info += f", column {column}"

        super().__init__(f"{message}{location_info}")


class SchemaValidationError(MilvusYamlError):
    """Raised when schema validation fails."""

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        errors: list[str] | None = None,
    ):
        self.field_name = field_name
        self.errors = errors or []

        field_info = f" in field '{field_name}'" if field_name else ""
        error_details = ""
        if self.errors:
            error_details = f". Details: {'; '.join(self.errors)}"

        super().__init__(f"{message}{field_info}{error_details}")


class SchemaConversionError(MilvusYamlError):
    """Raised when converting dict to CollectionSchema fails."""

    pass


class ValidationError(SchemaValidationError):
    """Alias for validation-specific errors.

    This class exists to provide a distinct, semantically clear exception
    for validation failures while remaining compatible with code expecting
    SchemaValidationError semantics.
    """

    pass


class UnsupportedTypeError(MilvusYamlError):
    """Raised when encountering unknown or unsupported field types."""

    def __init__(self, field_type: str, field_name: str | None = None):
        self.field_type = field_type
        self.field_name = field_name

        field_info = f" for field '{field_name}'" if field_name else ""
        super().__init__(f"Unsupported field type '{field_type}'{field_info}")


class InvalidParameterError(MilvusYamlError):
    """Raised when field parameters are invalid."""

    def __init__(
        self,
        parameter: str,
        value: str,
        field_name: str | None = None,
        expected: str | None = None,
    ):
        self.parameter = parameter
        self.value = value
        self.field_name = field_name
        self.expected = expected

        field_info = f" for field '{field_name}'" if field_name else ""
        expected_info = f" (expected: {expected})" if expected else ""
        super().__init__(f"Invalid {parameter} '{value}'{field_info}{expected_info}")


class MissingRequiredParameterError(MilvusYamlError):
    """Raised when required parameters are missing."""

    def __init__(
        self,
        parameter: str,
        field_name: str | None = None,
        field_type: str | None = None,
    ):
        self.parameter = parameter
        self.field_name = field_name
        self.field_type = field_type

        field_info = f" for field '{field_name}'" if field_name else ""
        type_info = f" of type '{field_type}'" if field_type else ""
        super().__init__(
            f"Missing required parameter '{parameter}'{field_info}{type_info}"
        )
