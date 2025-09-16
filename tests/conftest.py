from pathlib import Path
from typing import Any

import pytest

from pymilvus import DataType


@pytest.fixture
def valid_schema_dict() -> dict[str, Any]:
    """Basic valid schema dictionary for testing."""
    return {
        "name": "test_collection",
        "description": "Test collection schema",
        "fields": [
            {"name": "id", "type": "int64", "is_primary": True, "auto_id": False},
            {"name": "text", "type": "varchar", "max_length": 256},
            {"name": "vector", "type": "float_vector", "dim": 128},
        ],
    }


@pytest.fixture
def minimal_schema_dict() -> dict[str, Any]:
    """Minimal valid schema with just required fields."""
    return {
        "name": "minimal_test",
        "fields": [
            {"name": "id", "type": "int64", "is_primary": True, "auto_id": True}
        ],
    }


@pytest.fixture
def complex_schema_dict() -> dict[str, Any]:
    """Complex schema with all field types and settings."""
    return {
        "name": "complex_collection",
        "description": "Complex test schema with all features",
        "fields": [
            {"name": "id", "type": "int64", "is_primary": True, "auto_id": False},
            {"name": "bool_field", "type": "bool"},
            {"name": "int8_field", "type": "int8"},
            {"name": "int16_field", "type": "int16"},
            {"name": "int32_field", "type": "int32"},
            {"name": "float_field", "type": "float"},
            {"name": "double_field", "type": "double"},
            {"name": "varchar_field", "type": "varchar", "max_length": 100},
            {"name": "json_field", "type": "json"},
            {"name": "float_vector_field", "type": "float_vector", "dim": 512},
            {"name": "binary_vector_field", "type": "binary_vector", "dim": 512},
            {"name": "sparse_vector_field", "type": "sparse_float_vector"},
            {
                "name": "array_field",
                "type": "array",
                "element_type": "int32",
                "max_capacity": 100,
            },
        ],
        "indexes": [
            {
                "field": "float_vector_field",
                "type": "IVF_FLAT",
                "metric": "L2",
                "params": {"nlist": 1024},
            }
        ],
        "settings": {"enable_dynamic_field": True, "consistency_level": "Strong"},
    }


@pytest.fixture
def field_type_mapping() -> dict[str, DataType]:
    """Mapping of YAML field types to PyMilvus DataType enum values."""
    return {
        "int8": DataType.INT8,
        "int16": DataType.INT16,
        "int32": DataType.INT32,
        "int64": DataType.INT64,
        "bool": DataType.BOOL,
        "float": DataType.FLOAT,
        "double": DataType.DOUBLE,
        "varchar": DataType.VARCHAR,
        "json": DataType.JSON,
        "array": DataType.ARRAY,
        "float_vector": DataType.FLOAT_VECTOR,
        "binary_vector": DataType.BINARY_VECTOR,
        "sparse_float_vector": DataType.SPARSE_FLOAT_VECTOR,
    }


@pytest.fixture
def valid_yaml_content() -> str:
    """Valid YAML schema content as string."""
    return """
name: "fixture_test_collection"
description: "Test collection from fixture"

fields:
  - name: "id"
    type: "int64"
    is_primary: true
    auto_id: false

  - name: "username"
    type: "varchar"
    max_length: 100

  - name: "embedding"
    type: "float_vector"
    dim: 768

settings:
  consistency_level: "Strong"
"""


@pytest.fixture
def invalid_yaml_content() -> str:
    """Invalid YAML content for testing error cases."""
    return """
name: "invalid_test"
fields:
  - name: "bad_field"
    type: "unknown_type"
  - name: "vector_no_dim"
    type: "float_vector"
"""


@pytest.fixture
def yaml_parse_error_content() -> str:
    """YAML content with syntax errors."""
    return """
name: "broken_yaml"
fields:
  - name: "field1"
    type: "int64"
    is_primary: true
    bad_syntax: [unclosed_list
"""


@pytest.fixture
def create_temp_yaml(tmp_path):
    """Factory fixture to create temporary YAML files with given content."""

    def _create_yaml(content: str, filename: str = "test_schema.yaml") -> Path:
        yaml_file = tmp_path / filename
        yaml_file.write_text(content, encoding="utf-8")
        return yaml_file

    return _create_yaml


@pytest.fixture
def sample_field_definitions() -> dict[str, dict[str, Any]]:
    """Sample field definitions for different types."""
    return {
        "primary_int64": {
            "name": "id",
            "type": "int64",
            "is_primary": True,
            "auto_id": False,
        },
        "varchar": {"name": "text_field", "type": "varchar", "max_length": 256},
        "float_vector": {"name": "embedding", "type": "float_vector", "dim": 768},
        "binary_vector": {"name": "binary_emb", "type": "binary_vector", "dim": 512},
        "json": {"name": "metadata", "type": "json"},
        "array": {
            "name": "tags",
            "type": "array",
            "element_type": "varchar",
            "max_capacity": 50,
        },
    }


@pytest.fixture
def invalid_field_definitions() -> dict[str, dict[str, Any]]:
    """Invalid field definitions for error testing."""
    return {
        "missing_name": {"type": "int64"},
        "missing_type": {"name": "bad_field"},
        "unknown_type": {"name": "bad_field", "type": "unknown_type"},
        "varchar_no_length": {"name": "text", "type": "varchar"},
        "vector_no_dim": {"name": "vector", "type": "float_vector"},
        "array_no_element_type": {
            "name": "array",
            "type": "array",
            "max_capacity": 100,
        },
        "array_no_capacity": {
            "name": "array",
            "type": "array",
            "element_type": "int32",
        },
    }


def assert_field_properties(
    field, expected_name: str, expected_dtype: DataType, **kwargs
):
    """Helper function to assert field properties."""
    assert field.name == expected_name
    assert field.dtype == expected_dtype

    for key, value in kwargs.items():
        assert hasattr(field, key), f"Field missing attribute: {key}"
        assert getattr(field, key) == value, (
            f"Field {key} mismatch: got {getattr(field, key)}, expected {value}"
        )


def create_schema_dict(
    name: str, fields: list[dict[str, Any]], **kwargs
) -> dict[str, Any]:
    """Helper function to create schema dictionaries."""
    schema = {"name": name, "fields": fields}
    schema.update(kwargs)
    return schema
