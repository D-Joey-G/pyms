from typing import TYPE_CHECKING, Any

from pymilvus import CollectionSchema

from ..exceptions import SchemaConversionError, UnsupportedTypeError
from ..validators.schema import (
    SchemaValidationContext,
    ensure_runtime_requirements,
    resolve_autoindex_flag,
)
from .field import FieldBuilder
from .function import FunctionBuilder
from .index import IndexBuilder

if TYPE_CHECKING:
    from pymilvus import MilvusClient


class SchemaBuilder:
    """Main schema builder that orchestrates field, index, and function building."""

    def __init__(
        self,
        schema_dict: dict[str, Any],
        context: SchemaValidationContext | None = None,
    ):
        """Initialize with a schema dictionary.

        Args:
            schema_dict: dictionary representation of the schema
        """
        self.schema_dict = schema_dict

        if context is not None:
            self._field_types = dict(context.field_types)
        else:
            self._field_types: dict[str, str] = {}
            for f in self.schema_dict.get("fields", []) or []:
                name = f.get("name")
                t = f.get("type")
                if isinstance(name, str) and isinstance(t, str):
                    self._field_types[name] = t

        ensure_runtime_requirements(self.schema_dict)
        self._autoindex = resolve_autoindex_flag(self.schema_dict)

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

            settings = self.schema_dict.get("settings", {})
            enable_dynamic_field = settings.get("enable_dynamic_field", False)

            schema = CollectionSchema(
                fields=fields,
                description=description,
                enable_dynamic_field=enable_dynamic_field,
            )

            try:
                schema.verify()
            except Exception as e:
                raise SchemaConversionError(f"Schema validation failed: {e}") from e

            if self.indexes:
                for index_def in self.indexes:
                    self.validate_index_params(index_def)

            return schema
        except (UnsupportedTypeError, SchemaConversionError):
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

    def validate_function(self, func_def: dict[str, Any]) -> None:
        """Validate function definition."""
        return self.function_builder.validate_function(func_def)

    def get_milvus_function_objects(self) -> list[Any]:
        """Convert function definitions to PyMilvus Function objects."""
        return self.function_builder.get_milvus_function_objects()

    def get_function_index_warnings(self) -> list[str]:
        """Get warnings for function-index relationship issues."""
        return self.function_builder.get_function_index_warnings()

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
