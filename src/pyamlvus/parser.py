# YAML parsing logic for Milvus schemas
import os
import re

from pathlib import Path
from typing import Any

import yaml

from .exceptions import SchemaParseError

# Collection and alias names must start with a letter and contain only letters,
# digits, and underscores.
_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


class SchemaLoader:
    """Loads and parses YAML schema files into Python dictionaries."""

    def __init__(self, file_path: str | Path):
        """Initialize with path to YAML schema file.

        Args:
            file_path: Path to the YAML schema file

        Raises:
            SchemaParseError: If file doesn't exist or isn't readable
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise SchemaParseError(f"Schema file not found: {self.file_path}")

        if not self.file_path.is_file():
            raise SchemaParseError(f"Path is not a file: {self.file_path}")

        if not os.access(self.file_path, os.R_OK):
            raise SchemaParseError(f"Schema file not readable: {self.file_path}")

        self._schema_dict: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        """Load and parse the YAML schema file.

        Returns:
            dictionary representation of the schema

        Raises:
            SchemaParseError: If YAML parsing fails
        """
        if self._schema_dict is not None:
            return self._schema_dict

        try:
            with open(self.file_path, encoding="utf-8") as f:
                self._schema_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            line = getattr(e, "problem_mark", None)
            line_num = line.line + 1 if line else None
            col_num = line.column + 1 if line else None

            raise SchemaParseError(
                f"YAML parsing failed: {e}",
                file_path=str(self.file_path),
                line=line_num,
                column=col_num,
            ) from e
        except Exception as e:
            raise SchemaParseError(
                f"Failed to read schema file: {e}", file_path=str(self.file_path)
            ) from e

        if self._schema_dict is None:
            raise SchemaParseError("Empty schema file", file_path=str(self.file_path))

        if not isinstance(self._schema_dict, dict):
            raise SchemaParseError(
                f"Schema must be a dictionary, got {type(self._schema_dict).__name__}",
                file_path=str(self.file_path),
            )

        return self._schema_dict

    def to_dict(self) -> dict[str, Any]:
        """Get the schema as a dictionary (alias for load()).

        Returns:
            dictionary representation of the schema
        """
        return self.load()

    @property
    def name(self) -> str:
        """Get the collection name from the schema.

        Returns:
            Collection name

        Raises:
            SchemaParseError: If name is missing from schema or invalid
        """
        schema = self.load()
        if "name" not in schema:
            raise SchemaParseError(
                "Schema missing required 'name' field", file_path=str(self.file_path)
            )

        collection_name = schema["name"]
        self._validate_collection_name(collection_name)
        return collection_name

    def _validate_collection_name(self, name: str) -> None:
        """Validate collection name according to Milvus rules.

        Args:
            name: Collection name to validate

        Raises:
            SchemaParseError: If name is invalid
        """

        if not isinstance(name, str):
            raise SchemaParseError(
                f"Collection name must be a string, got {type(name).__name__}",
                file_path=str(self.file_path),
            )

        if not name:
            raise SchemaParseError(
                "Collection name cannot be empty", file_path=str(self.file_path)
            )

        # Collection name must start with a letter and contain only letters, digits,
        # and underscores

        if not _NAME_RE.match(name):
            raise SchemaParseError(
                f"Collection name '{name}' is invalid. Collection name must start with "
                f"a letter and contain only letters, digits, and underscores",
                file_path=str(self.file_path),
            )

        # Cannot start with underscore (additional validation)
        if name.startswith("_"):
            raise SchemaParseError(
                f"Collection name '{name}' cannot start with an underscore",
                file_path=str(self.file_path),
            )

    def _validate_collection_alias(self, alias: str) -> None:
        """Validate collection alias according to Milvus rules.

        Args:
            alias: Collection alias to validate

        Raises:
            SchemaParseError: If alias is invalid
        """
        # Collection alias must start with a letter and contain only letters, digits,
        # and underscores

        if not _NAME_RE.match(alias):
            raise SchemaParseError(
                f"Collection alias '{alias}' is invalid. Collection alias must start "
                f"with a letter and contain only letters, digits, and underscores",
                file_path=str(self.file_path),
            )

    @property
    def description(self) -> str:
        """Get the collection description from the schema.

        Returns:
            Collection description (empty string if not specified)
        """
        schema = self.load()
        return schema.get("description", "")

    @property
    def alias(self) -> str:
        """Get the collection alias from the schema.

        Returns:
            Collection alias (empty string if not specified)

        Raises:
            SchemaParseError: If alias is invalid
        """
        schema = self.load()
        alias = schema.get("alias")
        if alias is not None:
            if not isinstance(alias, str):
                raise SchemaParseError(
                    f"Collection alias must be a string, got {type(alias).__name__}",
                    file_path=str(self.file_path),
                )
            if alias == "":
                raise SchemaParseError(
                    "Collection alias cannot be empty", file_path=str(self.file_path)
                )
            self._validate_collection_alias(alias)
            return alias
        return ""

    @property
    def fields(self) -> list[dict[str, Any]]:
        """Get the field definitions from the schema.

        Returns:
            list of field definition dictionaries

        Raises:
            SchemaParseError: If fields are missing or invalid
        """
        schema = self.load()
        if "fields" not in schema:
            raise SchemaParseError(
                "Schema missing required 'fields' field", file_path=str(self.file_path)
            )

        fields = schema["fields"]
        if not isinstance(fields, list):
            raise SchemaParseError(
                f"Fields must be a list, got {type(fields).__name__}",
                file_path=str(self.file_path),
            )

        if not fields:
            raise SchemaParseError(
                "Schema must have at least one field", file_path=str(self.file_path)
            )

        return fields

    @property
    def indexes(self) -> list[dict[str, Any]]:
        """Get the index definitions from the schema.

        Returns:
            list of index definition dictionaries (empty if not specified)
        """
        schema = self.load()
        indexes = schema.get("indexes", [])
        if not isinstance(indexes, list):
            raise SchemaParseError(
                f"Indexes must be a list, got {type(indexes).__name__}",
                file_path=str(self.file_path),
            )
        return indexes

    @property
    def functions(self) -> list[dict[str, Any]]:
        """Get the function definitions from the schema.

        Returns:
            list of function definition dictionaries (empty if not specified)
        """
        schema = self.load()
        functions = schema.get("functions", [])
        if not isinstance(functions, list):
            raise SchemaParseError(
                f"Functions must be a list, got {type(functions).__name__}",
                file_path=str(self.file_path),
            )
        return functions

    @property
    def settings(self) -> dict[str, Any]:
        """Get the collection-level settings from the schema.

        Returns:
            dictionary of collection settings (empty if not specified)
        """
        schema = self.load()
        settings = schema.get("settings", {})
        if not isinstance(settings, dict):
            raise SchemaParseError(
                f"Settings must be a dictionary, got {type(settings).__name__}",
                file_path=str(self.file_path),
            )
        return settings
