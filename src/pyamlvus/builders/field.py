# Field building logic for pyamlvus
# Handles conversion of YAML field definitions to PyMilvus FieldSchema objects

from typing import Any

from pymilvus import FieldSchema

from ..exceptions import SchemaConversionError, UnsupportedTypeError
from ..types import TYPE_MAPPING


class FieldBuilder:
    """Builder for individual field definitions."""

    @staticmethod
    def build_field(field_def: dict[str, Any]) -> FieldSchema:
        """Build a PyMilvus FieldSchema from a YAML field definition.

        Args:
            field_def: Field definition dictionary

        Returns:
            PyMilvus FieldSchema object

        Raises:
            SchemaConversionError: If field definition is invalid
        """
        name = field_def.get("name")
        type_str = field_def.get("type")

        if not name:
            raise SchemaConversionError("Field missing required 'name' attribute")

        if not type_str:
            raise SchemaConversionError(
                f"Field '{name}' missing required 'type' attribute"
            )

        if type_str not in TYPE_MAPPING:
            raise UnsupportedTypeError(
                f"Unsupported field type '{type_str}' for field '{name}'"
            )

        # Build kwargs for FieldSchema
        kwargs = FieldBuilder._build_field_kwargs(field_def, name, type_str)

        try:
            return FieldSchema(**kwargs)  # type: ignore[call-arg]
        except Exception as e:
            raise SchemaConversionError(
                f"Failed to create FieldSchema for field '{name}': {e}"
            ) from e

    @staticmethod
    def _build_field_kwargs(
        field_def: dict[str, Any], name: str, type_str: str
    ) -> dict[str, Any]:
        """Build kwargs dictionary for FieldSchema constructor.

        Args:
            field_def: Field definition dictionary
            name: Field name
            type_str: Field type string

        Returns:
            dictionary of kwargs for FieldSchema
        """
        kwargs = {
            "name": name,
            "dtype": TYPE_MAPPING[type_str],
            "description": field_def.get("description", ""),
        }

        # Handle primary key
        if field_def.get("is_primary", False):
            kwargs["is_primary"] = True
            # auto_id defaults to False if not specified for primary keys
            kwargs["auto_id"] = field_def.get("auto_id", False)

        # Handle type-specific parameters
        params = FieldBuilder._build_type_params(field_def, type_str, name)

        # For VARCHAR fields, max_length is a direct parameter, not in params
        if type_str == "varchar":
            max_length = field_def.get("max_length")
            if max_length is not None:
                kwargs["max_length"] = max_length
                # Remove max_length from params if it exists
                params.pop("max_length", None)

        # For vector fields, dim is a direct parameter, not in params
        elif type_str in {
            "float_vector",
            "float16_vector",
            "bfloat16_vector",
            "int8_vector",
            "binary_vector",
            "sparse_float_vector",
        }:
            dim = field_def.get("dim")
            if dim is not None:
                kwargs["dim"] = dim
                # Remove dim from params if it exists
                params.pop("dim", None)

        # For array fields, element_type, max_length, and max_capacity are direct
        # parameters
        elif type_str == "array":
            element_type_str = field_def.get("element_type")
            if element_type_str and element_type_str in TYPE_MAPPING:
                kwargs["element_type"] = TYPE_MAPPING[element_type_str]
                params.pop("element_type", None)

            max_capacity = field_def.get("max_capacity")
            if max_capacity is not None:
                kwargs["max_capacity"] = max_capacity
                params.pop("max_capacity", None)

            # For varchar element arrays, max_length is also a direct parameter
            if element_type_str == "varchar":
                max_length = field_def.get("max_length")
                if max_length is not None:
                    kwargs["max_length"] = max_length
                    params.pop("max_length", None)

        # Text analysis flags (needed for BM25 input VARCHAR fields)
        if "enable_analyzer" in field_def:
            kwargs["enable_analyzer"] = bool(field_def.get("enable_analyzer"))
        # Text match flag (needed for TEXT_MATCH expressions)
        if "enable_match" in field_def:
            kwargs["enable_match"] = bool(field_def.get("enable_match"))
        # Default analyzer params when analyzer is enabled and params are missing
        if kwargs.get("enable_analyzer") is True:
            provided_params = field_def.get("analyzer_params")
            if not provided_params:
                kwargs["analyzer_params"] = {"type": "english"}
            else:
                kwargs["analyzer_params"] = provided_params
        if "multi_analyzer_params" in field_def:
            kwargs["multi_analyzer_params"] = field_def.get("multi_analyzer_params")

        # Nullable support
        if "nullable" in field_def:
            kwargs["nullable"] = bool(field_def.get("nullable"))

        if params:
            kwargs["params"] = params

        return kwargs

    @staticmethod
    def _build_type_params(
        field_def: dict[str, Any], type_str: str, field_name: str
    ) -> dict[str, Any]:
        """Build type-specific parameters for the field.

        Args:
            field_def: Field definition dictionary
            type_str: Field type string
            field_name: Field name for error messages

        Returns:
            dictionary of type-specific parameters
        """
        params = {}

        if type_str == "varchar":
            max_length = field_def.get("max_length")
            if max_length is not None:
                params["max_length"] = max_length

        elif type_str in {
            "float_vector",
            "float16_vector",
            "bfloat16_vector",
            "int8_vector",
            "binary_vector",
            "sparse_float_vector",
        }:
            dim = field_def.get("dim")
            if dim is not None:
                params["dim"] = dim

        elif type_str == "array":
            # Handle array-specific parameters
            element_type_str = field_def.get("element_type")
            if element_type_str:
                if element_type_str not in TYPE_MAPPING:
                    raise UnsupportedTypeError(
                        f"Unsupported array element type '{element_type_str}' for "
                        f"field '{field_name}'"
                    )
                params["element_type"] = TYPE_MAPPING[element_type_str]

            max_capacity = field_def.get("max_capacity")
            if max_capacity is not None:
                params["max_capacity"] = max_capacity

            # Element-specific parameters (like max_length) are handled as direct
            # parameters
            # in the main _build_field_kwargs method, not in params

        return params
