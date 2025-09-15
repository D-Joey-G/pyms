# Main schema builder for pyamlvus
# Orchestrates the building of complete CollectionSchema objects

from typing import TYPE_CHECKING, Any

from pymilvus import CollectionSchema

from ..exceptions import SchemaConversionError, UnsupportedTypeError
from ..types import PYMILVUS_VERSION, PYMILVUS_VERSION_INFO, parse_version
from .field import FieldBuilder
from .function import FunctionBuilder
from .index import IndexBuilder

if TYPE_CHECKING:
    from pymilvus import MilvusClient


class SchemaBuilder:
    """Main schema builder that orchestrates field, index, and function building."""

    def __init__(self, schema_dict: dict[str, Any]):
        """Initialize with a schema dictionary.

        Args:
            schema_dict: dictionary representation of the schema
        """
        self.schema_dict = schema_dict
        # Map field name -> yaml type string for validation and index checks
        self._field_types: dict[str, str] = {}
        for f in self.schema_dict.get("fields", []) or []:
            name = f.get("name")
            t = f.get("type")
            if isinstance(name, str) and isinstance(t, str):
                self._field_types[name] = t
        self._validate_runtime_requirements()
        self._autoindex: bool = self.autoindex_enabled

        # Initialize specialized builders
        self.field_builder = FieldBuilder()
        self.index_builder = IndexBuilder(self)
        self.function_builder = FunctionBuilder(self)

    def build(self) -> CollectionSchema:
        """Build a complete PyMilvus CollectionSchema.

        Returns:
            PyMilvus CollectionSchema object

        Raises:
            SchemaConversionError: If schema building fails
        """

        try:
            fields = self._build_fields()
            description = self.schema_dict.get("description", "")

            # Check for collection-level settings
            settings = self.schema_dict.get("settings", {})
            enable_dynamic_field = settings.get("enable_dynamic_field", False)

            schema = CollectionSchema(
                fields=fields,
                description=description,
                enable_dynamic_field=enable_dynamic_field,
            )

            # Verify schema integrity using PyMilvus built-in validation
            try:
                schema.verify()
            except Exception as e:
                raise SchemaConversionError(f"Schema validation failed: {e}") from e

            # Validate indexes if any are defined
            if self.indexes:
                for index_def in self.indexes:
                    self.validate_index_params(index_def)

            return schema
        except (UnsupportedTypeError, SchemaConversionError):
            # Re-raise our custom exceptions without wrapping
            raise
        except Exception as e:
            raise SchemaConversionError(f"Failed to build CollectionSchema: {e}") from e

    def _build_fields(self) -> list[Any]:
        """Build list of FieldSchema objects from field definitions.

        Returns:
            list of FieldSchema objects
        """

        fields = []
        primary_field_count = 0

        for field_def in self.schema_dict["fields"]:
            # Validate field parameters before building
            self.validate_field_params(field_def)
            field = self.field_builder.build_field(field_def)
            fields.append(field)

            if field_def.get("is_primary", False):
                primary_field_count += 1

        if primary_field_count == 0:
            raise SchemaConversionError("Schema must have exactly one primary field")
        if primary_field_count > 1:
            raise SchemaConversionError(
                f"Schema has {primary_field_count} primary fields, must have exactly "
                f"one"
            )

        return fields

    def validate_field_params(self, field_def: dict[str, Any]) -> None:
        """Validate per-field parameters based on type.

        Raises SchemaConversionError on invalid or out-of-range parameters.
        """
        from ..validators import FieldValidator

        validator = FieldValidator()
        validator.validate(field_def)

    # Delegate index-related methods to IndexBuilder
    def get_index_params(self, index_def: dict[str, Any]) -> dict[str, Any]:
        """Convert index definition to PyMilvus index_params format."""
        return self.index_builder.get_index_params(index_def)

    def validate_index_params(self, index_def: dict[str, Any]) -> None:
        """Validate index definition."""
        return self.index_builder.validate_index_params(index_def)

    def get_index_warnings(self) -> list[str]:
        """Get warnings for missing or suboptimal indexes."""
        return self.index_builder.get_index_warnings()

    def get_create_index_calls(self) -> list[tuple[str, dict[str, Any]]]:
        """Get all create_index calls needed for this schema."""
        return self.index_builder.get_create_index_calls()

    def get_milvus_index_params(self, client: "MilvusClient"):
        """Get MilvusClient index parameters for this schema."""
        return self.index_builder.get_milvus_index_params(client)

    # Delegate function-related methods to FunctionBuilder
    def validate_function(self, func_def: dict[str, Any]) -> None:
        """Validate function definition."""
        return self.function_builder.validate_function(func_def)

    def get_milvus_function_objects(self) -> list[Any]:
        """Convert function definitions to PyMilvus Function objects."""
        return self.function_builder.get_milvus_function_objects()

    def get_function_index_warnings(self) -> list[str]:
        """Get warnings for function-index relationship issues."""
        return self.function_builder.get_function_index_warnings()

    def _validate_runtime_requirements(self) -> None:
        """Validate schema-level runtime requirements (e.g., pymilvus version)."""

        requirements = self.schema_dict.get("pymilvus")
        if requirements is None:
            return

        if not isinstance(requirements, dict):
            raise SchemaConversionError(
                "Schema 'pymilvus' section must be a mapping with version bounds."
            )

        allowed_keys = {
            "min_version",
            "max_version",
            "version",
            "require",
            "exact_version",
        }
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
            except ValueError as exc:  # pragma: no cover - defensive
                raise SchemaConversionError(
                    f"Invalid version string for 'pymilvus.{key}': {value}"
                ) from exc

        min_version = _parse("min_version")
        max_version = _parse("max_version")
        version_exact = (
            _parse("version") or _parse("require") or _parse("exact_version")
        )

        if version_exact is not None and (min_version or max_version):
            raise SchemaConversionError(
                "Schema 'pymilvus' section cannot combine 'version' with min/max "
                "bounds."
            )

        if min_version and max_version and min_version > max_version:
            raise SchemaConversionError(
                "Schema 'pymilvus' min_version must be less than or equal to "
                "max_version"
            )

        current_tuple = PYMILVUS_VERSION_INFO

        if version_exact is not None and current_tuple != version_exact:
            requested = ".".join(str(part) for part in version_exact)
            raise SchemaConversionError(
                f"Schema requires pymilvus=={requested}, but current version is "
                f"{PYMILVUS_VERSION}."
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

    # Delegate other methods to original builder
    @property
    def indexes(self) -> list[dict[str, Any]]:
        """Get list of index definitions."""
        return self.schema_dict.get("indexes", [])

    @property
    def functions(self) -> list[dict[str, Any]]:
        """Get list of function definitions."""
        return self.schema_dict.get("functions", [])

    def is_bm25_function_output_field(self, field_name: str) -> bool:
        """Check if field is output of BM25 function."""
        for func_def in self.functions:
            func_type = (
                func_def.get("type") or func_def.get("function_type", "").upper()
            )
            if func_type == "BM25":
                # Check various possible output field keys
                output_fields = []
                for key in ["output_field", "output_field_names", "output_fields"]:
                    value = func_def.get(key)
                    if value:
                        if isinstance(value, str):
                            output_fields = [value]
                        elif isinstance(value, list):
                            output_fields = value
                        break

                if field_name in output_fields:
                    return True
        return False

    @property
    def alias(self) -> str:
        """Get the collection alias from the schema dictionary."""
        return self.schema_dict.get("alias", "")

    @property
    def autoindex_enabled(self) -> bool:
        """Get whether AUTOINDEX is enabled for this schema."""

        schema = self.schema_dict

        # Check for autoindex setting in multiple possible locations and key names
        autoindex_candidates = [
            ("autoindex", schema.get("autoindex")),
            ("enable_autoindex", schema.get("enable_autoindex")),
            ("use_autoindex", schema.get("use_autoindex")),
        ]

        # Check settings section
        if isinstance(schema.get("settings"), dict):
            settings = schema["settings"]
            autoindex_candidates.extend(
                [
                    ("settings.autoindex", settings.get("autoindex")),
                    ("settings.enable_autoindex", settings.get("enable_autoindex")),
                    ("settings.use_autoindex", settings.get("use_autoindex")),
                ]
            )

        # Find all non-None values
        found_candidates = [
            (key, value) for key, value in autoindex_candidates if value is not None
        ]

        # If multiple autoindex settings found, raise error
        if len(found_candidates) > 1:
            keys_found = [key for key, _ in found_candidates]
            raise SchemaConversionError(
                f"Multiple autoindex settings found: {', '.join(keys_found)}. "
                f"Please specify only one autoindex setting."
            )

        # If no autoindex setting found, default to False
        if not found_candidates:
            return False

        # Get the single autoindex value
        key, autoindex = found_candidates[0]

        # Only accept boolean values - reject everything else
        if isinstance(autoindex, bool):
            return autoindex

        # Reject any other type
        raise SchemaConversionError(
            f"Invalid autoindex value '{autoindex}' (type: "
            f"{type(autoindex).__name__}) for key '{key}'. Expected boolean value "
            f"(true or false)"
        )
