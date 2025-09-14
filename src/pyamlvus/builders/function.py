# Function building logic for pyamlvus
# Handles conversion of YAML function definitions to PyMilvus Function objects

from typing import TYPE_CHECKING, Any

from pymilvus import Function
from pymilvus.client.types import FunctionType

if TYPE_CHECKING:
    from .index import SchemaBuilderProtocol


class FunctionBuilder:
    """Builder for function definitions."""

    def __init__(self, schema_builder: "SchemaBuilderProtocol"):
        """Initialize with reference to the schema builder.

        Args:
            schema_builder: The schema builder instance (must implement
            SchemaBuilderProtocol)
        """
        self.schema_builder = schema_builder

    def get_milvus_function_objects(self) -> list[Function]:
        """Convert YAML function definitions to PyMilvus Function objects or dicts.

        Returns:
            list of Function objects

        Notes:
            - Attempts to construct pymilvus.Function via from_dict() or **kwargs.
            - Falls back to returning the original dict on any import/constructor error.
        """
        functions: list[Function] = []
        func_defs = self.schema_builder.functions or []

        # Validate all definitions first for better error messages
        for func_def in func_defs:
            self.validate_function(func_def)

        if not func_defs:
            return functions

        def _normalize(defn: dict[str, Any]) -> dict[str, Any]:
            d = dict(defn)
            # Normalize type key and map to PyMilvus FunctionType enum
            if "function_type" not in d and "type" in d:
                d["function_type"] = d.pop("type")

            # Accept common aliases and case/underscore variations
            if "function_type" in d:
                ft_raw = d["function_type"]
                # If already an enum or int, keep as-is; else map from string
                if isinstance(ft_raw, FunctionType):
                    pass
                elif isinstance(ft_raw, int):
                    # Let pymilvus handle value validation
                    pass
                elif isinstance(ft_raw, str):
                    token = "".join(ch for ch in ft_raw if ch.isalnum()).upper()
                    alias_map = {
                        "BM25": FunctionType.BM25,
                        "TEXTEMBEDDING": FunctionType.TEXTEMBEDDING,
                        "TEXTEMBED": FunctionType.TEXTEMBEDDING,
                        "TEXTEMBEDDINGS": FunctionType.TEXTEMBEDDING,
                        "RERANK": FunctionType.RERANK,
                        "RANKER": FunctionType.RERANK,
                    }
                    if token in alias_map:
                        d["function_type"] = alias_map[token]
                    else:
                        # Leave it as-is; downstream will raise a clearer error
                        d["function_type"] = ft_raw
                else:
                    # Unexpected type; let pymilvus raise
                    pass

            # Map input fields - PyMilvus expects string for single field, list for
            # multiple
            if "input_field_names" not in d:
                if "input_fields" in d:
                    d["input_field_names"] = d.pop("input_fields")
                elif "fields" in d:
                    d["input_field_names"] = d.pop("fields")
                elif "input_field" in d:
                    val = d.pop("input_field")
                    d["input_field_names"] = val
                elif "field" in d:
                    val = d.pop("field")
                    d["input_field_names"] = val

            if "output_field_names" not in d and "output_field" in d:
                val = d.pop("output_field")
                d["output_field_names"] = val

            return d

        for func_def in func_defs:
            normalized_def = _normalize(func_def)

            # Prefer calling the constructor directly to control arg names
            func_obj = Function(**normalized_def)  # type: ignore[call-arg]

            functions.append(func_obj)

        return functions

    def validate_function(self, func_def: dict[str, Any]) -> None:
        """Validate function definition using FunctionValidator rules."""
        from ..validators import FunctionValidator

        field_definitions = self.schema_builder.schema_dict.get("fields", [])
        validator = FunctionValidator(
            set(self.schema_builder._field_types.keys()), field_definitions
        )
        validator.validate(func_def)

    def get_function_index_warnings(self) -> list[str]:
        """Get warnings for function-index relationship issues."""
        from ..validators import FunctionValidator

        field_definitions = self.schema_builder.schema_dict.get("fields", [])
        validator = FunctionValidator(
            set(self.schema_builder._field_types.keys()), field_definitions
        )
        # If autoindex is enabled, normalize missing index types before validation.
        # Also normalize BM25 function output fields to SPARSE_INVERTED_INDEX when
        # type is missing.
        functions = self.schema_builder.functions
        raw_indexes = self.schema_builder.indexes

        adjusted_indexes: list[dict[str, Any]] = []
        for idx in raw_indexes:
            idx_copy = dict(idx)
            field_name = idx_copy.get("field")
            idx_type = (idx_copy.get("type") or "").upper()

            if field_name and not idx_type:
                if self.schema_builder.is_bm25_function_output_field(field_name):
                    idx_copy["type"] = "SPARSE_INVERTED_INDEX"
                    idx_copy.setdefault("metric", "BM25")
                elif self.schema_builder._autoindex:
                    idx_copy["type"] = "AUTOINDEX"

            adjusted_indexes.append(idx_copy)

        return validator.validate_function_index_relationships(
            functions, adjusted_indexes
        )
