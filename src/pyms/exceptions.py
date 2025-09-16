class MilvusYamlError(Exception):
    """Base exception for all pyms errors."""

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


class SchemaConversionError(MilvusYamlError):
    """Raised when converting dict to CollectionSchema fails."""

    pass


class UnsupportedTypeError(MilvusYamlError):
    """Raised when encountering unknown or unsupported field types."""

    def __init__(self, field_type: str, field_name: str | None = None):
        self.field_type = field_type
        self.field_name = field_name

        field_info = f" for field '{field_name}'" if field_name else ""
        super().__init__(f"Unsupported field type '{field_type}'{field_info}")
