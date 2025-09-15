# Index validation logic for pyamlvus
# Handles validation of index definitions

from typing import Any

from ..exceptions import SchemaConversionError
from ..types import (
    BINARY_METRICS,
    FLOAT_METRICS,
    GPU_INDEX_TYPES,
    OPTIONAL_INDEX_REQUIREMENTS,
    OPTIONAL_INDEX_SUPPORT,
    OPTIONAL_TYPE_REQUIREMENTS,
    OPTIONAL_TYPE_SUPPORT,
    PYMILVUS_VERSION,
    RECOMMENDED_INDEX_TYPES,
    REQUIRED_PARAMS,
    VALID_INDEX_TYPES,
)
from .base import BaseValidator


class IndexValidator(BaseValidator):
    """Validator for index definitions."""

    def __init__(
        self, field_types: dict[str, str], schema_dict: dict[str, Any] | None = None
    ):
        """Initialize with field type mapping.

        Args:
            field_types: Mapping of field name to field type
            schema_dict: Full schema dictionary for context
        """
        super().__init__(schema_dict)
        self.field_types = field_types

    def validate(self, item: dict[str, Any]) -> None:
        """Validate an index definition.

        Args:
            index_def: Index definition dictionary

        Raises:
            SchemaConversionError: If index definition is invalid
        """
        # Validate required fields
        field_name = self.validate_required_field(item, "field", "index")

        # Get index type (may be auto-determined)
        index_type = item.get("type")

        # Validate field exists
        if field_name not in self.field_types:
            raise SchemaConversionError(f"Index refers to unknown field '{field_name}'")

        field_type = self.field_types[field_name]

        if index_type:
            index_type = index_type.upper()  # Normalize to uppercase
            self._validate_index_type(index_type, field_type, field_name)
        else:
            # Will be auto-determined later
            pass

        # Validate metric if provided
        if "metric" in item:
            self._validate_metric(item["metric"], field_type, field_name, index_type)

        # Validate parameters (always check required params even if no params provided)
        params = item.get("params", {})
        self._validate_index_params(params, index_type or "", field_name)

    def _validate_index_type(
        self, index_type: str, field_type: str, field_name: str
    ) -> None:
        """Validate that index type is supported for the field type.

        Args:
            index_type: Type of index
            field_type: Type of field being indexed
            field_name: Name of the field

        Raises:
            SchemaConversionError: If index type is not valid for field type
        """
        # If the field type itself is optional and unsupported, surface the field error.
        if (
            field_type in OPTIONAL_TYPE_SUPPORT
            and not OPTIONAL_TYPE_SUPPORT[field_type]
        ):
            req = OPTIONAL_TYPE_REQUIREMENTS[field_type]
            raise SchemaConversionError(
                f"Field type '{field_type}' requires additional support ({req})."
            )

        valid_types = VALID_INDEX_TYPES.get(field_type, set())

        # Normalize to uppercase for case-insensitive comparison
        index_type_upper = index_type.upper()

        req = OPTIONAL_INDEX_REQUIREMENTS.get(index_type_upper)
        if req and not OPTIONAL_INDEX_SUPPORT.get(index_type_upper, False):
            raise SchemaConversionError(
                f"Index type '{index_type}' requires additional support "
                f"({req}). Current pymilvus version: {PYMILVUS_VERSION}."
            )

        # Check if type is valid (case-insensitive)
        if index_type_upper not in valid_types:
            recommended = RECOMMENDED_INDEX_TYPES.get(field_type)
            suggestion = f" Recommended: {recommended}" if recommended else ""
            raise SchemaConversionError(
                f"Index type '{index_type}' is not valid for {field_type} field "
                f"'{field_name}'. "
                f"Valid types: {sorted(valid_types)}{suggestion}. "
                f"Fix: Use one of the valid index types listed above."
            )

    def _validate_metric(
        self,
        metric: str,
        field_type: str,
        field_name: str,
        index_type: str | None,
    ) -> None:
        """Validate metric compatibility with field type.

        Args:
            metric: Metric name
            field_type: Type of field
            field_name: Name of the field
            index_type: Selected index type if provided

        Raises:
            SchemaConversionError: If metric is not compatible
        """
        metric_upper = metric.upper()

        if index_type and index_type in GPU_INDEX_TYPES and metric_upper == "COSINE":
            raise SchemaConversionError(
                f"Metric 'COSINE' is not supported for GPU index '{index_type}' on "
                f"field '{field_name}'. Use L2 or IP after normalizing vectors."
            )

        if field_type in {
            "float_vector",
            "float16_vector",
            "bfloat16_vector",
            "int8_vector",
        }:
            if metric_upper not in FLOAT_METRICS:
                raise SchemaConversionError(
                    f"Invalid metric '{metric}' for {field_type} field '{field_name}'. "
                    f"Allowed: {sorted(FLOAT_METRICS)}"
                )
        elif field_type == "binary_vector":
            if metric_upper not in BINARY_METRICS:
                raise SchemaConversionError(
                    f"Invalid metric '{metric}' for binary_vector field "
                    f"'{field_name}'. "
                    f"Allowed: {sorted(BINARY_METRICS)}"
                )
        elif field_type == "sparse_float_vector":
            # Sparse vectors typically use BM25 or IP
            if metric_upper not in {"BM25", "IP"}:
                raise SchemaConversionError(
                    f"Invalid metric '{metric}' for sparse_float_vector field "
                    f"'{field_name}'. "
                    f"Allowed: BM25, IP"
                )

    def _validate_index_params(
        self, params: dict[str, Any], index_type: str, field_name: str
    ) -> None:
        """Validate index parameters.

        Args:
            params: Parameter dictionary
            index_type: Type of index
            field_name: Name of the field

        Raises:
            SchemaConversionError: If parameters are invalid
        """
        if not isinstance(params, dict):
            raise SchemaConversionError(
                f"Index params for field '{field_name}' must be a dictionary, got "
                f"{type(params)}"
            )

        # Check required parameters
        required = REQUIRED_PARAMS.get(index_type, set())
        missing = required - set(params.keys())
        if missing:
            raise SchemaConversionError(
                f"Index '{index_type}' for field '{field_name}' missing required "
                f"parameters: {sorted(missing)}"
            )

        # Validate parameter types and values
        for param_name, param_value in params.items():
            self._validate_index_param(param_name, param_value, index_type, field_name)

    def _validate_index_param(
        self, param_name: str, param_value: Any, index_type: str, field_name: str
    ) -> None:
        """Validate a single index parameter.

        Args:
            param_name: Name of the parameter
            param_value: Value of the parameter
            index_type: Type of index
            field_name: Name of the field

        Raises:
            SchemaConversionError: If parameter is invalid
        """
        # HNSW parameters
        if index_type == "HNSW":
            if param_name in {"M", "efConstruction"}:
                if not isinstance(param_value, int) or param_value <= 0:
                    raise SchemaConversionError(
                        f"HNSW parameter '{param_name}' for field '{field_name}' must "
                        f"be a positive integer, got {param_value}"
                    )
                if param_name == "M" and param_value > 100:
                    raise SchemaConversionError(
                        f"HNSW parameter 'M' for field '{field_name}' is too large "
                        f"({param_value}). "
                        "Recommended: 4-100"
                    )

        # IVF parameters
        elif index_type in {"IVF_FLAT", "IVF_SQ8", "IVF_PQ"}:
            if param_name == "nlist":
                if not isinstance(param_value, int) or param_value <= 0:
                    raise SchemaConversionError(
                        f"IVF parameter 'nlist' for field '{field_name}' must be a "
                        f"positive integer, got {param_value}"
                    )
                if param_value > 65536:  # Practical limit
                    raise SchemaConversionError(
                        f"IVF parameter 'nlist' for field '{field_name}' is too large "
                        f"({param_value}). "
                        "Recommended: 100-10000"
                    )

        # PQ-specific parameters
        if index_type == "IVF_PQ":
            if param_name == "m":
                if not isinstance(param_value, int) or param_value <= 0:
                    raise SchemaConversionError(
                        f"PQ parameter 'm' for field '{field_name}' "
                        f"must be a positive integer, got {param_value}"
                    )

    def get_index_warnings(
        self, all_field_names: set[str], indexes: list[dict[str, Any]]
    ) -> list[str]:
        """Get warnings for missing or suboptimal indexes.

        Args:
            all_field_names: set of all field names in the schema
            indexes: list of index definitions

        Returns:
            list of warning messages
        """
        warnings = []

        # Get indexed fields
        indexed_fields = {idx.get("field") for idx in indexes if idx.get("field")}

        # Check for unindexed vector fields
        for field_name in all_field_names:
            field_type = self.field_types.get(field_name, "")
            if field_type in {
                "float_vector",
                "float16_vector",
                "bfloat16_vector",
                "int8_vector",
                "binary_vector",
                "sparse_float_vector",
            }:
                if field_name not in indexed_fields:
                    warnings.append(
                        f"WARNING: {field_type.upper()} field '{field_name}' has no "
                        f"index defined. "
                        "This will result in slow queries. Consider adding an index."
                    )

        # Check for suboptimal index choices
        for index_def in indexes:
            field_name = index_def.get("field")
            if not field_name:
                continue

            field_type = self.field_types.get(field_name, "")
            index_type = index_def.get("type", "").upper()

            if field_type and index_type:
                recommended = RECOMMENDED_INDEX_TYPES.get(field_type)
                if recommended and index_type != recommended.upper():
                    warnings.append(
                        f"INFO: Field '{field_name}' uses '{index_type}' "
                        f"index. "
                        f"Consider '{recommended}' for better performance on "
                        f"{field_type} fields."
                    )

        return warnings
