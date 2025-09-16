# Index building logic for pyamlvus
# Handles conversion of YAML index definitions to PyMilvus index parameters

from typing import TYPE_CHECKING, Any, Protocol

from ..exceptions import SchemaConversionError

if TYPE_CHECKING:
    from pymilvus import MilvusClient  # type: ignore[import]


class SchemaBuilderProtocol(Protocol):
    """Protocol defining the interface expected by IndexBuilder and FunctionBuilder."""

    @property
    def indexes(self) -> list[dict[str, Any]]: ...

    @property
    def functions(self) -> list[dict[str, Any]]: ...

    @property
    def schema_dict(self) -> dict[str, Any]: ...

    def is_bm25_function_output_field(self, field_name: str) -> bool: ...

    @property
    def _field_types(self) -> dict[str, str]: ...

    @property
    def _autoindex(self) -> bool: ...


class IndexBuilder:
    """Builder for index definitions and parameters."""

    def __init__(self, schema_builder: SchemaBuilderProtocol):
        """Initialize with reference to the schema builder.

        Args:
            schema_builder: The schema builder instance (must implement
            SchemaBuilderProtocol)
        """
        self.schema_builder = schema_builder

    def get_index_params(self, index_def: dict[str, Any]) -> dict[str, Any]:
        """Convert a single YAML index definition to PyMilvus index_params format.

        Args:
            index_def: Index definition dictionary from YAML

        Returns:
            dictionary compatible with create_index index_params

        Raises:
            SchemaConversionError: If index definition is invalid
        """
        # Validate index params strictly before constructing dictionary
        self.validate_index_params(index_def)

        field_name = index_def.get("field")
        if not field_name:
            raise SchemaConversionError("Index definition missing required 'field'")

        index_type = index_def.get("type")
        if not index_type:
            # Special handling for BM25 function output fields
            if self.schema_builder.is_bm25_function_output_field(field_name):
                index_type = "SPARSE_INVERTED_INDEX"
                index_def["type"] = index_type
                # Ensure BM25 metric is set for BM25 function output fields
                if "metric" not in index_def:
                    index_def["metric"] = "BM25"
            # Use AUTOINDEX for other fields if autoindex is enabled
            elif self.schema_builder._autoindex:
                index_type = "AUTOINDEX"
                index_def["type"] = index_type
            else:
                raise SchemaConversionError(
                    f"Index for field '{field_name}' missing required 'type'. "
                    f"Either specify an index type or enable autoindex in your schema."
                )

        index_params = {"index_type": index_type}

        # Add metric_type if specified, or set default for BM25 function output fields
        metric = index_def.get("metric")
        if metric:
            index_params["metric_type"] = metric
        elif (
            self.schema_builder.is_bm25_function_output_field(field_name)
            and index_type == "SPARSE_INVERTED_INDEX"
        ):
            # BM25 function output fields with SPARSE_INVERTED_INDEX must use BM25
            # metric
            index_params["metric_type"] = "BM25"

        # Add params if specified, or set defaults for SPARSE_INVERTED_INDEX
        params: dict[str, Any] = index_def.get("params") or {}

        # Set defaults for SPARSE_INVERTED_INDEX if not already present
        if index_type == "SPARSE_INVERTED_INDEX":
            # Always set the inverted_index_algo default if not specified
            if "inverted_index_algo" not in params:
                params["inverted_index_algo"] = "DAAT_MAXSCORE"

            # Add BM25-specific params for BM25 function output fields if not specified
            if self.schema_builder.is_bm25_function_output_field(field_name):
                if "bm25_k1" not in params:
                    params["bm25_k1"] = 1.2
                if "bm25_b" not in params:
                    params["bm25_b"] = 0.75

        if params:
            index_params["params"] = params

        return index_params

    def validate_index_params(self, index_def: dict[str, Any]) -> None:
        """Validate index definition using IndexValidator rules."""
        from ..validators import IndexValidator

        validator = IndexValidator(self.schema_builder._field_types)
        validator.validate(index_def)

    def get_index_warnings(self) -> list[str]:
        """Get warnings for missing or suboptimal indexes."""
        from ..validators import IndexValidator

        validator = IndexValidator(self.schema_builder._field_types)
        all_field_names = set(self.schema_builder._field_types.keys())
        messages = validator.get_index_warnings(
            all_field_names, self.schema_builder.indexes
        )
        return [message.as_prefixed() for message in messages]

    def get_create_index_calls(self) -> list[tuple[str, dict[str, Any]]]:
        """Get all create_index calls needed for this schema.

        Returns:
            list of (field_name, index_params) tuples for each index

        Raises:
            SchemaConversionError: If any index definition is invalid
        """
        calls = []
        for index_def in self.schema_builder.indexes:
            field_name = index_def.get("field")
            if not field_name:
                raise SchemaConversionError("Index definition missing required 'field'")

            index_params = self.get_index_params(index_def)
            calls.append((field_name, index_params))

        return calls

    def get_milvus_index_params(self, client: "MilvusClient"):
        """Get MilvusClient index parameters for this schema.

        Args:
            client: MilvusClient instance to prepare index parameters with

        Returns:
            IndexParams object ready for client.create_index()
        """
        index_params = client.prepare_index_params()

        for index_def in self.schema_builder.indexes:
            # Get the basic index params
            index_dict = self.get_index_params(index_def)

            # Extract parameters for MilvusClient.add_index()
            kwargs = {
                "field_name": index_def["field"],
                "index_type": index_dict["index_type"],
            }

            # Add metric_type if present
            if "metric_type" in index_dict:
                kwargs["metric_type"] = index_dict["metric_type"]

            # Add params if present
            if "params" in index_dict:
                kwargs["params"] = index_dict["params"]

            index_params.add_index(**kwargs)  # type: ignore[call-arg]

        return index_params
