from typing import Any

from ..exceptions import SchemaConversionError
from .base import BaseValidator
from .result import ValidationMessage, ValidationSeverity


class FunctionValidator(BaseValidator):
    """Validator for function definitions."""

    def __init__(self, field_names: set[str], field_definitions: list[dict[str, Any]]):
        """Initialize with field information.

        Args:
            field_names: Set of all field names in the schema
            field_definitions: list of all field definitions
        """
        super().__init__()
        self.field_names = field_names
        self.field_definitions = field_definitions
        self.field_types = {f["name"]: f["type"] for f in field_definitions}

    def validate(self, item: dict[str, Any]) -> None:
        """Validate a function definition.

        Args:
            func_def: Function definition dictionary

        Raises:
            SchemaConversionError: If function definition is invalid
        """
        func_type = self.validate_required_field(item, "type", "function")

        self._validate_function_type(func_type)

        self._validate_input_fields(item)

        self._validate_output_field(item)

        self._validate_function_params(item, func_type)

    def _validate_function_type(self, func_type: str) -> None:
        """Validate function type.

        Args:
            func_type: Type of function

        Raises:
            SchemaConversionError: If function type is not supported
        """
        token = "".join(ch for ch in str(func_type) if ch.isalnum()).upper()
        supported_aliases = {
            "BM25": "BM25",
            "TEXTEMBEDDING": "TEXT_EMBEDDING",
            "TEXTEMBED": "TEXT_EMBEDDING",
            "TEXTEMBEDDINGS": "TEXT_EMBEDDING",
            "RERANK": "RERANK",
            "RANKER": "RERANK",
        }

        if token not in supported_aliases:
            raise SchemaConversionError(
                f"Unsupported function type '{func_type}'. "
                f"Supported types: {sorted(set(supported_aliases.values()))}"
            )

    def _validate_input_fields(self, func_def: dict[str, Any]) -> None:
        """Validate input field references.

        Args:
            func_def: Function definition

        Raises:
            SchemaConversionError: If input fields are invalid
        """
        input_fields = None
        if "input_field_names" in func_def:
            input_fields = func_def["input_field_names"]
        elif "input_fields" in func_def:
            input_fields = func_def["input_fields"]
        elif "fields" in func_def:
            input_fields = func_def["fields"]
        elif "input_field" in func_def:
            input_fields = func_def["input_field"]
        elif "field" in func_def:
            input_fields = func_def["field"]

        if input_fields is None:
            raise SchemaConversionError(
                "Function definition missing input field specification. Use "
                "'input_field_names', 'input_fields', 'fields', 'input_field', or "
                "'field'."
            )

        if isinstance(input_fields, str):
            input_fields = [input_fields]
        elif not isinstance(input_fields, list):
            raise SchemaConversionError(
                f"Input fields must be a string or list, got {type(input_fields)}"
            )

        for field_name in input_fields:
            if field_name not in self.field_names:
                raise SchemaConversionError(
                    f"Function input field '{field_name}' does not exist in schema"
                )

            field_type = self.field_types.get(field_name)
            if field_type not in {"varchar", "json"}:
                raise SchemaConversionError(
                    f"Function input field '{field_name}' has incompatible type "
                    f"'{field_type}'. Functions can only operate on 'varchar' or "
                    f"'json' fields."
                )

    def _validate_output_field(self, func_def: dict[str, Any]) -> None:
        """Validate output field specification.

        Args:
            func_def: Function definition

        Raises:
            SchemaConversionError: If output field is invalid
        """
        output_field = None
        if "output_field_names" in func_def:
            output_field = func_def["output_field_names"]
        elif "output_field" in func_def:
            output_field = func_def["output_field"]

        if output_field is None:
            raise SchemaConversionError(
                "Function definition missing output field specification. "
                "Use 'output_field_names' or 'output_field'."
            )

        if isinstance(output_field, str):
            output_fields = [output_field]
        elif isinstance(output_field, list):
            if not output_field:
                raise SchemaConversionError(
                    "Function output field list cannot be empty"
                )
            if not all(isinstance(f, str) for f in output_field):
                raise SchemaConversionError(
                    "All function output fields must be strings"
                )
            output_fields = output_field
        else:
            raise SchemaConversionError(
                f"Function output field must be a string or list, got "
                f"{type(output_field)}"
            )

        for field_name in output_fields:
            if field_name not in self.field_names:
                raise SchemaConversionError(
                    f"Function output field '{field_name}' must be an existing field"
                )

    def _validate_function_params(
        self, func_def: dict[str, Any], func_type: str
    ) -> None:
        """Validate function-specific parameters.

        Args:
            func_def: Function definition
            func_type: Type of function

        Raises:
            SchemaConversionError: If parameters are invalid
        """
        norm = "".join(ch for ch in str(func_type) if ch.isalnum()).upper()
        if norm == "TEXTEMBEDDING" or norm == "TEXTEMBED" or norm == "TEXTEMBEDDINGS":
            self._validate_text_embedding_params(func_def)
        elif norm == "BM25":
            self._validate_bm25_params(func_def)

    def _validate_text_embedding_params(self, func_def: dict[str, Any]) -> None:
        """Validate text embedding function parameters.

        Args:
            func_def: Function definition

        Raises:
            SchemaConversionError: If parameters are invalid
        """
        if "params" not in func_def:
            raise SchemaConversionError(
                "TEXT_EMBEDDING function missing required 'params' section"
            )

        params = func_def["params"]
        if not isinstance(params, dict):
            raise SchemaConversionError(
                "TEXT_EMBEDDING function 'params' must be a dictionary"
            )

        if "model" not in params:
            raise SchemaConversionError(
                "TEXT_EMBEDDING function missing required 'model' parameter"
            )

        model = params["model"]
        if not isinstance(model, str) or not model.strip():
            raise SchemaConversionError(
                "TEXT_EMBEDDING function 'model' must be a non-empty string"
            )

    def _validate_bm25_params(self, func_def: dict[str, Any]) -> None:
        """Validate BM25 function parameters.

        Args:
            func_def: Function definition

        Raises:
            SchemaConversionError: If parameters are invalid
        """
        input_fields = None
        for key in (
            "input_field_names",
            "input_fields",
            "fields",
            "input_field",
            "field",
        ):
            if key in func_def:
                input_fields = func_def[key]
                break
        if isinstance(input_fields, str):
            input_fields = [input_fields]
        if not input_fields:
            raise SchemaConversionError(
                "BM25 function missing input field specification; set "
                "'input_field_names' or 'input_field'."
            )

        defs_by_name = {f.get("name"): f for f in self.field_definitions}
        for fname in input_fields:
            fdef = defs_by_name.get(fname) or {}
            if not fdef.get("enable_analyzer", False):
                raise SchemaConversionError(
                    f"BM25 function input field '{fname}' requires "
                    f"enable_analyzer: true"
                )

        if "params" in func_def:
            params = func_def["params"]
            if not isinstance(params, dict):
                raise SchemaConversionError(
                    "BM25 function 'params' must be a dictionary"
                )

            valid_params = {"k1", "b"}
            for param_name in params:
                if param_name not in valid_params:
                    raise SchemaConversionError(
                        f"BM25 function has invalid parameter '{param_name}'. "
                        f"Valid parameters: {sorted(valid_params)}"
                    )

                param_value = params[param_name]
                if not isinstance(param_value, int | float):
                    raise SchemaConversionError(
                        f"BM25 parameter '{param_name}' must be a number, got "
                        f"{type(param_value)}"
                    )

                if param_value <= 0:
                    raise SchemaConversionError(
                        f"BM25 parameter '{param_name}' must be positive, got "
                        f"{param_value}"
                    )

    def validate_function_index_relationships(
        self, functions: list[dict[str, Any]], indexes: list[dict[str, Any]]
    ) -> list[ValidationMessage]:
        """Validate relationships between functions and indexes.

        Args:
            functions: list of function definitions
            indexes: list of index definitions

        Returns:
            list of warning/error messages
        """
        messages: list[ValidationMessage] = []

        function_outputs = set()
        for func in functions:
            output_field = func.get("output_field_names") or func.get("output_field")
            if isinstance(output_field, str):
                function_outputs.add(output_field)
            elif isinstance(output_field, list):
                function_outputs.update(output_field)

        for func in functions:
            func_type = func.get("type", "").upper()
            output_field = func.get("output_field_names") or func.get("output_field")

            if isinstance(output_field, str):
                output_fields = [output_field]
            elif isinstance(output_field, list):
                output_fields = output_field
            else:
                continue

            for field_name in output_fields:
                has_index = any(idx.get("field") == field_name for idx in indexes)

                if not has_index:
                    if func_type == "BM25":
                        messages.append(
                            ValidationMessage(
                                ValidationSeverity.WARNING,
                                (
                                    f"BM25 function output field '{field_name}' has no "
                                    "index. BM25 functions require "
                                    "SPARSE_INVERTED_INDEX for optimal performance."
                                ),
                            )
                        )
                    else:
                        messages.append(
                            ValidationMessage(
                                ValidationSeverity.WARNING,
                                (
                                    f"Function output field '{field_name}' has no "
                                    "index. Consider adding an appropriate index for "
                                    "query performance."
                                ),
                            )
                        )

                for idx in indexes:
                    if idx.get("field") == field_name:
                        index_type = idx.get("type", "").upper()
                        if (
                            func_type == "BM25"
                            and index_type != "SPARSE_INVERTED_INDEX"
                        ):
                            messages.append(
                                ValidationMessage(
                                    ValidationSeverity.ERROR,
                                    (
                                        f"BM25 function output field '{field_name}' "
                                        "uses '{index_type}' index. "
                                        "BM25 functions require SPARSE_INVERTED_INDEX."
                                    ),
                                )
                            )

        return messages
