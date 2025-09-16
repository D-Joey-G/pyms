from unittest.mock import Mock

import pytest

from pymilvus import CollectionSchema, DataType

from pyms.builders.schema import SchemaBuilder
from pyms.exceptions import SchemaConversionError, UnsupportedTypeError
from pyms.types import TYPE_MAPPING
from tests.conftest import assert_field_properties, create_schema_dict


@pytest.mark.unit
class TestSchemaBuilder:
    def test_basic_schema_building(self, valid_schema_dict):
        """Test building a basic CollectionSchema."""
        builder = SchemaBuilder(valid_schema_dict)
        schema = builder.build()

        assert isinstance(schema, CollectionSchema)
        assert schema.description == "Test collection schema"
        assert len(schema.fields) == 3

        # Check primary field using helper function
        id_field = schema.fields[0]
        assert_field_properties(
            id_field, "id", DataType.INT64, is_primary=True, auto_id=False
        )

        # Check varchar field
        text_field = schema.fields[1]
        assert_field_properties(text_field, "text", DataType.VARCHAR, max_length=256)

        # Check vector field
        vector_field = schema.fields[2]
        assert_field_properties(vector_field, "vector", DataType.FLOAT_VECTOR, dim=128)

    def test_autoindex_missing_index_type(self, valid_schema_dict):
        """Test behavior when index type is missing and autoindex is
        enabled/disabled."""
        # Add an index without type
        schema_dict = valid_schema_dict.copy()
        schema_dict["indexes"] = [{"field": "vector"}]  # Missing 'type'
        schema_dict["autoindex"] = True

        # With autoindex enabled, should work and use AUTOINDEX
        builder = SchemaBuilder(schema_dict)
        index_params = builder.get_index_params({"field": "vector"})
        assert index_params["index_type"] == "AUTOINDEX"

        schema_dict["autoindex"] = False
        # With autoindex disabled, should raise error
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="missing required 'type'"):
            builder.get_index_params({"field": "vector"})

    def test_autoindex_multiple_settings_error(self, valid_schema_dict):
        """Test that multiple autoindex settings raise an error."""
        # Test multiple top-level settings
        schema_dict = valid_schema_dict.copy()
        schema_dict["autoindex"] = True
        schema_dict["enable_autoindex"] = False

        with pytest.raises(
            SchemaConversionError, match="Multiple autoindex settings found"
        ):
            SchemaBuilder(schema_dict)

        # Test top-level and settings
        schema_dict2 = valid_schema_dict.copy()
        schema_dict2["autoindex"] = True
        schema_dict2["settings"] = {"autoindex": False}

        with pytest.raises(
            SchemaConversionError, match="Multiple autoindex settings found"
        ):
            SchemaBuilder(schema_dict2)

        # Test multiple in settings
        schema_dict3 = valid_schema_dict.copy()
        schema_dict3["settings"] = {"autoindex": True, "enable_autoindex": False}

        with pytest.raises(
            SchemaConversionError, match="Multiple autoindex settings found"
        ):
            SchemaBuilder(schema_dict3)

    def test_schema_min_version_requirement(self, valid_schema_dict, monkeypatch):
        schema_dict = valid_schema_dict.copy()
        schema_dict["pymilvus"] = {"min_version": "99.0.0"}

        monkeypatch.setattr(
            "pyms.validators.schema.PYMILVUS_VERSION_INFO",
            (2, 5, 0),
            raising=False,
        )
        monkeypatch.setattr(
            "pyms.validators.schema.PYMILVUS_VERSION", "2.5.0", raising=False
        )

        with pytest.raises(
            SchemaConversionError,
            match="pymilvus>=99.0.0",
        ):
            SchemaBuilder(schema_dict)

    def test_schema_exact_version_requirement_conflict(
        self, valid_schema_dict, monkeypatch
    ):
        schema_dict = valid_schema_dict.copy()
        schema_dict["pymilvus"] = {"version": "1.0.0"}

        monkeypatch.setattr(
            "pyms.validators.schema.PYMILVUS_VERSION_INFO",
            (2, 6, 0),
            raising=False,
        )
        monkeypatch.setattr(
            "pyms.validators.schema.PYMILVUS_VERSION", "2.6.0", raising=False
        )

        with pytest.raises(
            SchemaConversionError,
            match="pymilvus==1.0.0",
        ):
            SchemaBuilder(schema_dict)

    def test_schema_pymilvus_invalid_shape(self, valid_schema_dict):
        schema_dict = valid_schema_dict.copy()
        schema_dict["pymilvus"] = "2.6.0"

        with pytest.raises(
            SchemaConversionError, match="pymilvus' section must be a mapping"
        ):
            SchemaBuilder(schema_dict)

    def test_bm25_function_output_field_auto_index(self, valid_schema_dict):
        """Test that BM25 function output fields get correct index automatically."""
        schema_dict = valid_schema_dict.copy()
        schema_dict["autoindex"] = True

        # Add a sparse vector field and BM25 function
        schema_dict["fields"].extend(
            [
                {
                    "name": "text_field",
                    "type": "varchar",
                    "max_length": 1000,
                    "enable_analyzer": True,
                },
                {"name": "sparse_output", "type": "sparse_float_vector"},
            ]
        )

        # Add BM25 function
        schema_dict["functions"] = [
            {
                "name": "text_bm25",
                "type": "BM25",
                "input_field": "text_field",
                "output_field": "sparse_output",
            }
        ]

        # Add index without type (should auto-detect BM25 function output)
        schema_dict["indexes"] = [
            {"field": "sparse_output"}  # No type specified
        ]

        builder = SchemaBuilder(schema_dict)

        # Check that BM25 function output field is detected
        assert builder.is_bm25_function_output_field("sparse_output") is True
        assert builder.is_bm25_function_output_field("text_field") is False

        # Check that correct index params are generated
        index_def = {"field": "sparse_output"}
        params = builder.get_index_params(index_def.copy())
        assert params["index_type"] == "SPARSE_INVERTED_INDEX"
        assert params["metric_type"] == "BM25"

    def test_all_field_types(self, complex_schema_dict):
        """Test all supported field types."""
        builder = SchemaBuilder(complex_schema_dict)
        schema = builder.build()

        field_types = {field.name: field.dtype for field in schema.fields}

        # Verify all field types are correctly mapped
        expected_mappings = {
            "id": DataType.INT64,
            "bool_field": DataType.BOOL,
            "int8_field": DataType.INT8,
            "int16_field": DataType.INT16,
            "int32_field": DataType.INT32,
            "float_field": DataType.FLOAT,
            "double_field": DataType.DOUBLE,
            "varchar_field": DataType.VARCHAR,
            "json_field": DataType.JSON,
            "float_vector_field": DataType.FLOAT_VECTOR,
            "binary_vector_field": DataType.BINARY_VECTOR,
            "sparse_vector_field": DataType.SPARSE_FLOAT_VECTOR,
            "array_field": DataType.ARRAY,
        }

        for field_name, expected_dtype in expected_mappings.items():
            assert field_types[field_name] == expected_dtype

    def test_settings_handling(self):
        """Test collection-level settings handling."""
        schema_dict = create_schema_dict(
            "test_settings",
            [{"name": "id", "type": "int64", "is_primary": True}],
            description="Test with settings",
            settings={
                "enable_dynamic_field": True,
                "consistency_level": "Strong",  # This won't affect
                # CollectionSchema directly
            },
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        assert schema.enable_dynamic_field is True
        assert schema.description == "Test with settings"

    @pytest.mark.error_cases
    def test_no_primary_field_error(self):
        """Test error when no primary field is defined."""
        schema_dict = create_schema_dict(
            "test_no_primary", [{"name": "text", "type": "varchar", "max_length": 100}]
        )

        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            SchemaConversionError, match="Schema must have exactly one primary field"
        ):
            builder.build()

    @pytest.mark.error_cases
    def test_multiple_primary_fields_error(self):
        """Test error when multiple primary fields are defined."""
        schema_dict = create_schema_dict(
            "test_multi_primary",
            [
                {"name": "id1", "type": "int64", "is_primary": True},
                {"name": "id2", "type": "int64", "is_primary": True},
            ],
        )

        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="Schema has 2 primary fields"):
            builder.build()

    @pytest.mark.error_cases
    def test_missing_field_name(self):
        """Test error when field is missing name."""
        schema_dict = create_schema_dict(
            "test_missing_name", [{"type": "int64", "is_primary": True}]
        )

        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            SchemaConversionError, match="Field missing required 'name' attribute"
        ):
            builder.build()

    @pytest.mark.error_cases
    def test_missing_field_type(self):
        """Test error when field is missing type."""
        schema_dict = create_schema_dict(
            "test_missing_type", [{"name": "id", "is_primary": True}]
        )

        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            SchemaConversionError, match="Field 'id' missing required 'type' attribute"
        ):
            builder.build()

    @pytest.mark.error_cases
    def test_unsupported_type(self):
        """Test error for unsupported field types."""
        schema_dict = create_schema_dict(
            "test_unsupported",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "bad_field", "type": "unknown_type"},
            ],
        )

        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            UnsupportedTypeError,
            match="Unsupported field type 'unknown_type' for field 'bad_field'",
        ):
            builder.build()

    @pytest.mark.error_cases
    def test_varchar_missing_max_length(self):
        """Test error when varchar field is missing max_length."""
        schema_dict = create_schema_dict(
            "test_varchar_no_length",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar"},
            ],
        )

        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            SchemaConversionError,
            match="VARCHAR field 'text' missing required 'max_length' parameter",
        ):
            builder.build()

    @pytest.mark.error_cases
    def test_vector_missing_dim(self):
        """Test error when vector field is missing dim parameter."""
        schema_dict = create_schema_dict(
            "test_vector_no_dim",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vector", "type": "float_vector"},
            ],
        )

        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            SchemaConversionError,
            match="Vector field 'vector' missing required 'dim' parameter",
        ):
            builder.build()

    @pytest.mark.error_cases
    def test_array_missing_params(self):
        """Test error when array field is missing required parameters."""
        # Test missing element_type
        schema_dict1 = create_schema_dict(
            "test_array_no_element_type",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "array_field", "type": "array", "max_capacity": 100},
            ],
        )

        builder1 = SchemaBuilder(schema_dict1)
        with pytest.raises(
            SchemaConversionError,
            match="Array field 'array_field' missing required 'element_type' parameter",
        ):
            builder1.build()

        # Test missing max_capacity
        schema_dict2 = create_schema_dict(
            "test_array_no_capacity",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "array_field", "type": "array", "element_type": "int32"},
            ],
        )

        builder2 = SchemaBuilder(schema_dict2)
        with pytest.raises(
            SchemaConversionError,
            match="Array field 'array_field' missing required 'max_capacity' parameter",
        ):
            builder2.build()

    def test_field_descriptions(self):
        """Test that field descriptions are preserved."""
        schema_dict = create_schema_dict(
            "test_descriptions",
            [
                {
                    "name": "id",
                    "type": "int64",
                    "is_primary": True,
                    "description": "Primary key field",
                },
                {
                    "name": "text",
                    "type": "varchar",
                    "max_length": 100,
                    "description": "Text content",
                },
            ],
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        assert schema.fields[0].description == "Primary key field"
        assert schema.fields[1].description == "Text content"

    def test_nullable_field_handling(self):
        """Test that nullable flag is passed to FieldSchema."""
        schema_dict = create_schema_dict(
            "test_nullable",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "age", "type": "int64", "nullable": True},
            ],
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        age_field = next(f for f in schema.fields if f.name == "age")
        assert getattr(age_field, "nullable", False) is True

    def test_analyzer_default_params_builds(self):
        """When enable_analyzer is true and no params provided, builder
        supplies defaults."""
        schema_dict = create_schema_dict(
            "test_analyzer_default",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text",
                    "type": "varchar",
                    "max_length": 128,
                    "enable_analyzer": True,  # no analyzer_params supplied
                },
            ],
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()
        assert isinstance(schema, CollectionSchema)

    def test_analyzer_custom_params_builds(self):
        """Custom analyzer_params should pass through without error."""
        schema_dict = create_schema_dict(
            "test_analyzer_custom",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text",
                    "type": "varchar",
                    "max_length": 128,
                    "enable_analyzer": True,
                    "analyzer_params": {
                        "tokenizer": "standard",
                        "filters": ["lowercase"],
                    },
                },
            ],
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()
        assert isinstance(schema, CollectionSchema)

    def test_enable_match_builds(self):
        """Test that enable_match parameter builds correctly."""
        schema_dict = create_schema_dict(
            "test_enable_match",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text",
                    "type": "varchar",
                    "max_length": 128,
                    "enable_match": True,
                },
            ],
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()
        assert isinstance(schema, CollectionSchema)

        # Verify the enable_match parameter is set on the field
        text_field = next(f for f in schema.fields if f.name == "text")
        assert getattr(text_field, "enable_match", False) is True

    def test_enable_match_false_builds(self):
        """Test that enable_match=false builds correctly."""
        schema_dict = create_schema_dict(
            "test_enable_match_false",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text",
                    "type": "varchar",
                    "max_length": 128,
                    "enable_match": False,
                },
            ],
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()
        assert isinstance(schema, CollectionSchema)

        # Verify the enable_match parameter is set to False
        text_field = next(f for f in schema.fields if f.name == "text")
        assert getattr(text_field, "enable_match", True) is False

    def test_enable_analyzer_and_match_combined(self):
        """Test that enable_analyzer and enable_match work together."""
        schema_dict = create_schema_dict(
            "test_combined_flags",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text",
                    "type": "varchar",
                    "max_length": 128,
                    "enable_analyzer": True,
                    "enable_match": True,
                    "analyzer_params": {"type": "english"},
                },
            ],
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()
        assert isinstance(schema, CollectionSchema)

        # Verify both parameters are set correctly
        text_field = next(f for f in schema.fields if f.name == "text")
        assert getattr(text_field, "enable_analyzer", False) is True
        assert getattr(text_field, "enable_match", False) is True

    def test_auto_id_handling(self, minimal_schema_dict):
        """Test auto_id parameter handling."""
        builder = SchemaBuilder(minimal_schema_dict)
        schema = builder.build()

        assert schema.fields[0].auto_id is True

    def test_alias_property(self):
        """Test that alias property works correctly."""
        schema_dict = create_schema_dict(
            "test_collection",
            [{"name": "id", "type": "int64", "is_primary": True}],
            alias="my_test_alias",
        )

        builder = SchemaBuilder(schema_dict)
        assert builder.alias == "my_test_alias"

    def test_alias_empty_when_not_specified(self):
        """Test that alias returns empty string when not specified."""
        schema_dict = create_schema_dict(
            "test_collection", [{"name": "id", "type": "int64", "is_primary": True}]
        )

        builder = SchemaBuilder(schema_dict)
        assert builder.alias == ""


@pytest.mark.unit
class TestSchemaBuilderParametrized:
    """Parametrized tests for SchemaBuilder with various field type scenarios."""

    @pytest.mark.parametrize(
        "field_name,field_type,expected_dtype,extra_params",
        [
            ("bool_test", "bool", DataType.BOOL, {}),
            ("int8_test", "int8", DataType.INT8, {}),
            ("int16_test", "int16", DataType.INT16, {}),
            ("int32_test", "int32", DataType.INT32, {}),
            ("int64_test", "int64", DataType.INT64, {}),
            ("float_test", "float", DataType.FLOAT, {}),
            ("double_test", "double", DataType.DOUBLE, {}),
            ("json_test", "json", DataType.JSON, {}),
            ("varchar_test", "varchar", DataType.VARCHAR, {"max_length": 128}),
            ("float_vec_test", "float_vector", DataType.FLOAT_VECTOR, {"dim": 256}),
            ("binary_vec_test", "binary_vector", DataType.BINARY_VECTOR, {"dim": 128}),
            (
                "sparse_vec_test",
                "sparse_float_vector",
                DataType.SPARSE_FLOAT_VECTOR,
                {},
            ),
            (
                "array_test",
                "array",
                DataType.ARRAY,
                {"element_type": "varchar", "max_capacity": 50},
            ),
        ],
    )
    def test_individual_field_types(
        self, field_name, field_type, expected_dtype, extra_params
    ):
        """Test building schemas with individual field types."""
        field_def = {"name": field_name, "type": field_type}
        field_def.update(extra_params)

        schema_dict = create_schema_dict(
            "field_type_test",
            [{"name": "id", "type": "int64", "is_primary": True}, field_def],
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        # Find the test field
        test_field = next(f for f in schema.fields if f.name == field_name)
        assert test_field.dtype == expected_dtype

        # Verify extra parameters
        for param_name, param_value in extra_params.items():
            assert hasattr(test_field, param_name)
            actual_value = getattr(test_field, param_name)

            # Special handling for array element_type - it gets converted to
            # DataType enum
            if param_name == "element_type" and isinstance(param_value, str):
                expected_value = TYPE_MAPPING.get(param_value)
                assert actual_value == expected_value
            else:
                assert actual_value == param_value

    @pytest.mark.parametrize(
        "primary_field_def,auto_id_value",
        [
            (
                {"name": "id1", "type": "int64", "is_primary": True, "auto_id": True},
                True,
            ),
            (
                {"name": "id2", "type": "int64", "is_primary": True, "auto_id": False},
                False,
            ),
            (
                {"name": "id3", "type": "int64", "is_primary": True},
                False,
            ),  # Default auto_id
            (
                {
                    "name": "str_id",
                    "type": "varchar",
                    "max_length": 50,
                    "is_primary": True,
                },
                False,
            ),
        ],
    )
    def test_primary_key_configurations(self, primary_field_def, auto_id_value):
        """Test various primary key field configurations."""
        schema_dict = create_schema_dict("primary_test", [primary_field_def])

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        primary_field = schema.fields[0]
        assert primary_field.is_primary is True
        assert primary_field.auto_id == auto_id_value

    @pytest.mark.parametrize(
        "settings_dict,expected_dynamic",
        [
            ({}, False),  # Default
            ({"enable_dynamic_field": False}, False),
            ({"enable_dynamic_field": True}, True),
            ({"enable_dynamic_field": True, "other_setting": "value"}, True),
        ],
    )
    def test_dynamic_field_settings(self, settings_dict, expected_dynamic):
        """Test enable_dynamic_field setting variations."""
        schema_dict = create_schema_dict(
            "settings_test",
            [{"name": "id", "type": "int64", "is_primary": True}],
            settings=settings_dict,
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        assert schema.enable_dynamic_field == expected_dynamic

    @pytest.mark.error_cases
    @pytest.mark.parametrize(
        "field_definition,expected_error",
        [
            (
                {"name": "bad_varchar", "type": "varchar"},
                "VARCHAR field 'bad_varchar' missing required 'max_length' parameter",
            ),
            (
                {"name": "bad_vector", "type": "float_vector"},
                "Vector field 'bad_vector' missing required 'dim' parameter",
            ),
            (
                {"name": "bad_binary", "type": "binary_vector"},
                "Vector field 'bad_binary' missing required 'dim' parameter",
            ),
            (
                {"name": "bad_array", "type": "array", "max_capacity": 100},
                "Array field 'bad_array' missing required 'element_type' parameter",
            ),
            (
                {"name": "bad_array2", "type": "array", "element_type": "int32"},
                "Array field 'bad_array2' missing required 'max_capacity' parameter",
            ),
            ({"type": "int64"}, "Field missing required 'name' attribute"),
            ({"name": "no_type"}, "Field 'no_type' missing required 'type' attribute"),
            (
                {"name": "unknown", "type": "fake_type"},
                "Unsupported field type 'fake_type' for field 'unknown'",
            ),
        ],
    )
    def test_invalid_field_definitions(
        self, field_definition, expected_error, invalid_field_definitions
    ):
        """Test various invalid field definition scenarios."""
        schema_dict = create_schema_dict(
            "error_test",
            [{"name": "id", "type": "int64", "is_primary": True}, field_definition],
        )

        builder = SchemaBuilder(schema_dict)

        if "Unsupported field type" in expected_error:
            with pytest.raises(UnsupportedTypeError, match=expected_error):
                builder.build()
        else:
            with pytest.raises(SchemaConversionError, match=expected_error):
                builder.build()

    @pytest.mark.parametrize("field_count", [1, 2, 5, 10, 20])
    def test_varying_field_counts(self, field_count):
        """Test schemas with varying numbers of fields."""
        fields = [{"name": "id", "type": "int64", "is_primary": True}]

        # Add additional fields
        for i in range(1, field_count):
            fields.append({"name": f"field_{i}", "type": "varchar", "max_length": 100})

        schema_dict = create_schema_dict("field_count_test", fields)

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        assert len(schema.fields) == field_count

        # Verify primary key
        primary_fields = [f for f in schema.fields if f.is_primary]
        assert len(primary_fields) == 1
        assert primary_fields[0].name == "id"

    @pytest.mark.parametrize(
        "description",
        [
            "",
            "Short desc",
            "A very long description with lots of details about this test "
            "collection schema",
        ],
    )
    def test_description_variations(self, description):
        """Test schemas with various description lengths."""
        schema_dict = create_schema_dict(
            "desc_test",
            [{"name": "id", "type": "int64", "is_primary": True}],
            description=description,
        )

        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        assert schema.description == description


@pytest.mark.unit
class TestSchemaBuilderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_valid_schema(self, minimal_schema_dict):
        """Test the absolute minimum valid schema."""
        builder = SchemaBuilder(minimal_schema_dict)
        schema = builder.build()

        assert len(schema.fields) == 1
        assert schema.fields[0].is_primary is True
        assert schema.description == ""

    def test_complex_schema_full_build(self, complex_schema_dict):
        """Test building the most complex schema configuration."""
        builder = SchemaBuilder(complex_schema_dict)
        schema = builder.build()

        # Verify basic properties
        assert isinstance(schema, CollectionSchema)
        assert schema.enable_dynamic_field is True
        assert len(schema.fields) == 13  # All field types including bool

        # Verify primary key exists and is unique
        primary_fields = [f for f in schema.fields if f.is_primary]
        assert len(primary_fields) == 1

        # Verify all field types are present
        field_types = [f.dtype for f in schema.fields]
        expected_types = [
            DataType.INT64,
            DataType.BOOL,
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.FLOAT,
            DataType.DOUBLE,
            DataType.VARCHAR,
            DataType.JSON,
            DataType.FLOAT_VECTOR,
            DataType.BINARY_VECTOR,
            DataType.SPARSE_FLOAT_VECTOR,
            DataType.ARRAY,
        ]

        for expected_type in expected_types:
            assert expected_type in field_types

    @pytest.mark.error_cases
    def test_empty_schema_dict(self):
        """Test completely empty schema dictionary."""
        builder = SchemaBuilder({})
        with pytest.raises(
            SchemaConversionError, match="Failed to build CollectionSchema"
        ):
            builder.build()


@pytest.mark.unit
class TestSchemaBuilderIndexes:
    """Test index-related functionality in SchemaBuilder."""

    def test_indexes_property_empty(self, valid_schema_dict):
        """Test indexes property with schema that has no indexes."""
        builder = SchemaBuilder(valid_schema_dict)
        assert builder.indexes == []

    def test_indexes_property_with_indexes(self):
        """Test indexes property with schema that has index definitions."""
        schema_dict = create_schema_dict(
            "test_indexes",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vector", "type": "float_vector", "dim": 128},
                {"name": "text", "type": "varchar", "max_length": 100},
            ],
            indexes=[
                {
                    "field": "vector",
                    "type": "IVF_FLAT",
                    "metric": "L2",
                    "params": {"nlist": 1024},
                },
                {"field": "text", "type": "TRIE"},
            ],
        )

        builder = SchemaBuilder(schema_dict)
        indexes = builder.indexes

        assert len(indexes) == 2
        assert indexes[0]["field"] == "vector"
        assert indexes[0]["type"] == "IVF_FLAT"
        assert indexes[1]["field"] == "text"
        assert indexes[1]["type"] == "TRIE"

    def test_get_index_params_complete(self):
        """Test get_index_params with complete index definition."""
        schema_dict = {
            "fields": [{"name": "vector_field", "type": "float_vector", "dim": 128}]
        }
        builder = SchemaBuilder(schema_dict)
        index_def = {
            "field": "vector_field",
            "type": "IVF_FLAT",
            "metric": "L2",
            "params": {"nlist": 1024},
        }

        index_params = builder.get_index_params(index_def)

        assert index_params == {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }

    def test_get_index_params_minimal(self):
        """Test get_index_params with minimal index definition."""
        schema_dict = {
            "fields": [{"name": "text_field", "type": "varchar", "max_length": 100}]
        }
        builder = SchemaBuilder(schema_dict)
        index_def = {"field": "text_field", "type": "TRIE"}

        index_params = builder.get_index_params(index_def)

        assert index_params == {"index_type": "TRIE"}

    def test_get_index_params_with_metric_no_params(self):
        """Test get_index_params with metric but no params."""
        schema_dict = {
            "fields": [{"name": "vector_field", "type": "float_vector", "dim": 128}]
        }
        builder = SchemaBuilder(schema_dict)
        index_def = {"field": "vector_field", "type": "FLAT", "metric": "IP"}

        index_params = builder.get_index_params(index_def)

        assert index_params == {"index_type": "FLAT", "metric_type": "IP"}

    @pytest.mark.error_cases
    def test_get_index_params_missing_field(self):
        """Test error when index definition missing field."""
        builder = SchemaBuilder({})
        index_def = {"type": "IVF_FLAT"}

        with pytest.raises(
            SchemaConversionError, match="Index definition missing required 'field'"
        ):
            builder.get_index_params(index_def)

    @pytest.mark.error_cases
    def test_get_index_params_missing_type(self):
        """Test error when index definition missing type."""
        schema_dict = {
            "name": "test_collection",
            "fields": [{"name": "vector_field", "type": "float_vector", "dim": 128}],
        }
        builder = SchemaBuilder(schema_dict)
        index_def = {"field": "vector_field"}

        with pytest.raises(
            SchemaConversionError,
            match="Index for field 'vector_field' missing required 'type'",
        ):
            builder.get_index_params(index_def)

    def test_get_create_index_calls_empty(self, valid_schema_dict):
        """Test get_create_index_calls with no indexes."""
        builder = SchemaBuilder(valid_schema_dict)
        calls = builder.get_create_index_calls()
        assert calls == []

    def test_get_create_index_calls_multiple(self):
        """Test get_create_index_calls with multiple indexes."""
        schema_dict = create_schema_dict(
            "test_calls",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vector", "type": "float_vector", "dim": 768},
                {"name": "text", "type": "varchar", "max_length": 100},
            ],
            indexes=[
                {
                    "field": "vector",
                    "type": "IVF_FLAT",
                    "metric": "L2",
                    "params": {"nlist": 1024},
                },
                {"field": "text", "type": "TRIE"},
            ],
        )

        builder = SchemaBuilder(schema_dict)
        calls = builder.get_create_index_calls()

        assert len(calls) == 2

        field_name, index_params = calls[0]
        assert field_name == "vector"
        assert index_params == {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }

        field_name, index_params = calls[1]
        assert field_name == "text"
        assert index_params == {"index_type": "TRIE"}

        # Verify parity with get_milvus_index_params
        mock_client = Mock()
        mock_index_params = Mock()
        mock_client.prepare_index_params.return_value = mock_index_params

        _milvus_params = builder.get_milvus_index_params(mock_client)

        # Should have called add_index twice with same parameters
        assert mock_index_params.add_index.call_count == 2

        # Verify the calls match
        call_args_list = mock_index_params.add_index.call_args_list
        expected_calls = [
            {
                "field_name": "vector",
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 1024},
            },
            {
                "field_name": "text",
                "index_type": "TRIE",
            },
        ]

        for i, call in enumerate(call_args_list):
            args, kwargs = call
            assert kwargs == expected_calls[i]

    @pytest.mark.error_cases
    def test_get_create_index_calls_invalid_definition(self):
        """Test error in get_create_index_calls with invalid index definition."""
        schema_dict = create_schema_dict(
            "test_error",
            [{"name": "id", "type": "int64", "is_primary": True}],
            indexes=[{"type": "IVF_FLAT"}],  # Missing field
        )

        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            SchemaConversionError, match="Index definition missing required 'field'"
        ):
            builder.get_create_index_calls()


@pytest.mark.unit
class TestMilvusClientIntegration:
    """Test MilvusClient-specific functionality."""

    def test_get_milvus_index_params_empty(self, valid_schema_dict):
        """Test get_milvus_index_params with no indexes."""
        builder = SchemaBuilder(valid_schema_dict)
        mock_client = Mock()
        mock_index_params = Mock()
        mock_client.prepare_index_params.return_value = mock_index_params

        result = builder.get_milvus_index_params(mock_client)

        mock_client.prepare_index_params.assert_called_once()
        mock_index_params.add_index.assert_not_called()
        assert result == mock_index_params

    def test_get_milvus_index_params_single_index(self):
        """Test get_milvus_index_params with single index."""
        schema_dict = create_schema_dict(
            "test_milvus_index",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vector", "type": "float_vector", "dim": 768},
            ],
            indexes=[
                {
                    "field": "vector",
                    "type": "IVF_FLAT",
                    "metric": "L2",
                    "params": {"nlist": 1024},
                }
            ],
        )

        builder = SchemaBuilder(schema_dict)
        mock_client = Mock()
        mock_index_params = Mock()
        mock_client.prepare_index_params.return_value = mock_index_params

        result = builder.get_milvus_index_params(mock_client)

        mock_client.prepare_index_params.assert_called_once()
        mock_index_params.add_index.assert_called_once_with(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 1024},
        )
        assert result == mock_index_params

    def test_get_milvus_index_params_multiple_indexes(self):
        """Test get_milvus_index_params with multiple indexes."""
        schema_dict = create_schema_dict(
            "test_multiple",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vector", "type": "float_vector", "dim": 768},
                {"name": "text", "type": "varchar", "max_length": 100},
            ],
            indexes=[
                {
                    "field": "vector",
                    "type": "IVF_FLAT",
                    "metric": "L2",
                    "params": {"nlist": 1024},
                },
                {"field": "text", "type": "TRIE"},
            ],
        )

        builder = SchemaBuilder(schema_dict)
        mock_client = Mock()
        mock_index_params = Mock()
        mock_client.prepare_index_params.return_value = mock_index_params

        builder.get_milvus_index_params(mock_client)

        mock_client.prepare_index_params.assert_called_once()
        assert mock_index_params.add_index.call_count == 2

        # Check first call
        first_call = mock_index_params.add_index.call_args_list[0]
        assert first_call.kwargs == {
            "field_name": "vector",
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }

        # Check second call
        second_call = mock_index_params.add_index.call_args_list[1]
        assert second_call.kwargs == {"field_name": "text", "index_type": "TRIE"}

    def test_get_milvus_index_params_minimal_index(self):
        """Test get_milvus_index_params with minimal index definition."""
        schema_dict = create_schema_dict(
            "test_minimal",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 100},
            ],
            indexes=[{"field": "text", "type": "TRIE"}],
        )

        builder = SchemaBuilder(schema_dict)
        mock_client = Mock()
        mock_index_params = Mock()
        mock_client.prepare_index_params.return_value = mock_index_params

        builder.get_milvus_index_params(mock_client)

        mock_index_params.add_index.assert_called_once_with(
            field_name="text", index_type="TRIE"
        )

    @pytest.mark.error_cases
    def test_get_milvus_index_params_missing_field(self):
        """Test error when index definition missing field."""
        schema_dict = create_schema_dict(
            "test_error",
            [{"name": "id", "type": "int64", "is_primary": True}],
            indexes=[{"type": "IVF_FLAT"}],
        )

        builder = SchemaBuilder(schema_dict)
        mock_client = Mock()
        mock_client.prepare_index_params.return_value = Mock()

        with pytest.raises(
            SchemaConversionError, match="Index definition missing required 'field'"
        ):
            builder.get_milvus_index_params(mock_client)

    @pytest.mark.error_cases
    def test_get_milvus_index_params_missing_type(self):
        """Test error when index definition missing type."""
        schema_dict = create_schema_dict(
            "test_error",
            [{"name": "id", "type": "int64", "is_primary": True}],
            indexes=[{"field": "id"}],
        )

        builder = SchemaBuilder(schema_dict)
        mock_client = Mock()
        mock_client.prepare_index_params.return_value = Mock()

        with pytest.raises(
            SchemaConversionError, match="Index for field 'id' missing required 'type'"
        ):
            builder.get_milvus_index_params(mock_client)


@pytest.mark.unit
class TestIndexAssemblyParity:
    """Test that get_index_params and get_milvus_index_params produce
    identical results."""

    def test_parity_basic_index(self):
        """Test parity for basic index definition."""
        schema_dict = create_schema_dict(
            "test_parity",
            [{"name": "id", "type": "int64", "is_primary": True}],
            indexes=[{"field": "id", "type": "INVERTED"}],
        )

        builder = SchemaBuilder(schema_dict)
        mock_client = Mock()
        mock_index_params = Mock()
        mock_client.prepare_index_params.return_value = mock_index_params

        # Get results from both methods
        index_params = builder.get_index_params({"field": "id", "type": "INVERTED"})
        _milvus_params = builder.get_milvus_index_params(mock_client)

        # Verify the index was added to milvus_params
        mock_index_params.add_index.assert_called_once()

        # Get the call arguments
        call_args, call_kwargs = mock_index_params.add_index.call_args

        # Compare the parameters
        expected_kwargs = {"field_name": "id", "index_type": index_params["index_type"]}

        if "metric_type" in index_params:
            expected_kwargs["metric_type"] = index_params["metric_type"]
        if "params" in index_params:
            expected_kwargs["params"] = index_params["params"]

        assert call_kwargs == expected_kwargs

    def test_parity_bm25_function_output_field(self):
        """Test parity for BM25 function output field auto-detection."""
        schema_dict = create_schema_dict(
            "test_bm25_parity",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text_field",
                    "type": "varchar",
                    "max_length": 1000,
                    "enable_analyzer": True,
                },
                {"name": "sparse_output", "type": "sparse_float_vector"},
            ],
            indexes=[{"field": "sparse_output"}],  # No type specified
            functions=[
                {
                    "name": "text_bm25",
                    "type": "BM25",
                    "input_field": "text_field",
                    "output_field": "sparse_output",
                }
            ],
        )

        builder = SchemaBuilder(schema_dict)
        mock_client = Mock()
        mock_index_params = Mock()
        mock_client.prepare_index_params.return_value = mock_index_params

        # Get results from both methods
        index_params = builder.get_index_params({"field": "sparse_output"})
        _milvus_params = builder.get_milvus_index_params(mock_client)

        # Verify BM25 function output field was detected
        assert index_params["index_type"] == "SPARSE_INVERTED_INDEX"
        assert index_params["metric_type"] == "BM25"
        assert "params" in index_params
        assert index_params["params"]["bm25_k1"] == 1.2
        assert index_params["params"]["bm25_b"] == 0.75

        # Verify the index was added to milvus_params with same parameters
        mock_index_params.add_index.assert_called_once()
        call_args, call_kwargs = mock_index_params.add_index.call_args

        expected_kwargs = {
            "field_name": "sparse_output",
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "BM25",
            "params": index_params["params"],
        }

        assert call_kwargs == expected_kwargs

    def test_parity_multiple_indexes(self):
        """Test parity for multiple index definitions."""
        schema_dict = create_schema_dict(
            "test_multi_parity",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vector", "type": "float_vector", "dim": 768},
                {"name": "text", "type": "varchar", "max_length": 100},
            ],
            indexes=[
                {"field": "id", "type": "INVERTED"},
                {
                    "field": "vector",
                    "type": "IVF_FLAT",
                    "metric": "COSINE",
                    "params": {"nlist": 1024},
                },
                {"field": "text", "type": "TRIE"},
            ],
        )

        builder = SchemaBuilder(schema_dict)
        mock_client = Mock()
        mock_index_params = Mock()
        mock_client.prepare_index_params.return_value = mock_index_params

        _milvus_params = builder.get_milvus_index_params(mock_client)

        # Verify all indexes were added
        assert mock_index_params.add_index.call_count == 3

        # Check each call matches get_index_params output
        calls = mock_index_params.add_index.call_args_list

        for i, index_def in enumerate(schema_dict["indexes"]):
            index_params = builder.get_index_params(index_def)
            call_args, call_kwargs = calls[i]

            expected_kwargs = {
                "field_name": index_def["field"],
                "index_type": index_params["index_type"],
            }

            if "metric_type" in index_params:
                expected_kwargs["metric_type"] = index_params["metric_type"]
            if "params" in index_params:
                expected_kwargs["params"] = index_params["params"]

            assert call_kwargs == expected_kwargs


@pytest.mark.unit
class TestDeprecationWarnings:
    """Test that deprecation warnings are properly emitted for legacy functions."""


@pytest.mark.unit
class TestBM25Defaults:
    """Test BM25 defaults when omitted in YAML."""

    def test_bm25_function_output_omits_metric_and_params(self):
        """Test that BM25 function output fields get correct defaults when
        metric/params omitted."""
        schema_dict = create_schema_dict(
            "test_bm25_defaults",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text_field",
                    "type": "varchar",
                    "max_length": 1000,
                    "enable_analyzer": True,
                },
                {"name": "sparse_output", "type": "sparse_float_vector"},
            ],
            indexes=[
                {
                    "field": "sparse_output",
                    "type": "SPARSE_INVERTED_INDEX",
                    # Note: No metric or params specified - should get BM25 defaults
                }
            ],
            functions=[
                {
                    "name": "text_bm25",
                    "type": "BM25",
                    "input_field": "text_field",
                    "output_field": "sparse_output",
                }
            ],
        )

        builder = SchemaBuilder(schema_dict)

        # Test get_index_params directly
        index_params = builder.get_index_params(
            {"field": "sparse_output", "type": "SPARSE_INVERTED_INDEX"}
        )

        # Should automatically add BM25 metric and params
        assert index_params["index_type"] == "SPARSE_INVERTED_INDEX"
        assert index_params["metric_type"] == "BM25"
        assert "params" in index_params
        assert index_params["params"]["bm25_k1"] == 1.2
        assert index_params["params"]["bm25_b"] == 0.75
        assert index_params["params"]["inverted_index_algo"] == "DAAT_MAXSCORE"

    def test_bm25_function_output_no_type_specified(self):
        """Test that BM25 function output fields auto-detect type when omitted."""
        schema_dict = create_schema_dict(
            "test_bm25_auto_type",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text_field",
                    "type": "varchar",
                    "max_length": 1000,
                    "enable_analyzer": True,
                },
                {"name": "sparse_output", "type": "sparse_float_vector"},
            ],
            indexes=[
                {
                    "field": "sparse_output",
                    # Note: No type specified - should auto-detect SPARSE_INVERTED_INDEX
                }
            ],
            functions=[
                {
                    "name": "text_bm25",
                    "type": "BM25",
                    "input_field": "text_field",
                    "output_field": "sparse_output",
                }
            ],
        )

        builder = SchemaBuilder(schema_dict)

        # Test get_index_params directly
        index_params = builder.get_index_params({"field": "sparse_output"})

        # Should automatically detect BM25 function output and set correct defaults
        assert index_params["index_type"] == "SPARSE_INVERTED_INDEX"
        assert index_params["metric_type"] == "BM25"
        assert "params" in index_params
        assert index_params["params"]["bm25_k1"] == 1.2
        assert index_params["params"]["bm25_b"] == 0.75
        assert index_params["params"]["inverted_index_algo"] == "DAAT_MAXSCORE"

    def test_bm25_function_output_with_partial_params(self):
        """Test that BM25 function output fields merge user params with defaults."""
        schema_dict = create_schema_dict(
            "test_bm25_partial",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text_field",
                    "type": "varchar",
                    "max_length": 1000,
                    "enable_analyzer": True,
                },
                {"name": "sparse_output", "type": "sparse_float_vector"},
            ],
            indexes=[
                {
                    "field": "sparse_output",
                    "type": "SPARSE_INVERTED_INDEX",
                    "params": {"bm25_k1": 2.0},  # Partial override
                }
            ],
            functions=[
                {
                    "name": "text_bm25",
                    "type": "BM25",
                    "input_field": "text_field",
                    "output_field": "sparse_output",
                }
            ],
        )

        builder = SchemaBuilder(schema_dict)

        # Test get_index_params directly
        index_params = builder.get_index_params(
            {
                "field": "sparse_output",
                "type": "SPARSE_INVERTED_INDEX",
                "params": {"bm25_k1": 2.0},
            }
        )

        # Should merge user params with defaults
        assert index_params["index_type"] == "SPARSE_INVERTED_INDEX"
        assert index_params["metric_type"] == "BM25"
        assert "params" in index_params
        assert index_params["params"]["bm25_k1"] == 2.0  # User override
        assert index_params["params"]["bm25_b"] == 0.75  # Default
        assert (
            index_params["params"]["inverted_index_algo"] == "DAAT_MAXSCORE"
        )  # Default


@pytest.mark.unit
class TestAutoindexBehavior:
    """Test that unspecified index types resolve to AUTOINDEX consistently."""

    def test_autoindex_enabled_unspecified_type(self):
        """Test that unspecified types resolve to AUTOINDEX when autoindex is
        enabled."""
        schema_dict = create_schema_dict(
            "test_autoindex",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vector", "type": "float_vector", "dim": 768},
                {"name": "text", "type": "varchar", "max_length": 100},
            ],
            indexes=[
                {"field": "vector"},  # No type specified
                {"field": "text"},  # No type specified
            ],
        )
        schema_dict["autoindex"] = True

        builder = SchemaBuilder(schema_dict)

        # Test get_index_params for vector field
        vector_params = builder.get_index_params({"field": "vector"})
        assert vector_params["index_type"] == "AUTOINDEX"

        # Test get_index_params for text field
        text_params = builder.get_index_params({"field": "text"})
        assert text_params["index_type"] == "AUTOINDEX"

    def test_autoindex_disabled_error_on_unspecified_type(self):
        """Test that unspecified types raise error when autoindex is disabled."""
        schema_dict = create_schema_dict(
            "test_no_autoindex",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vector", "type": "float_vector", "dim": 768},
            ],
            indexes=[
                {"field": "vector"},  # No type specified
            ],
        )
        # autoindex is False by default

        builder = SchemaBuilder(schema_dict)

        with pytest.raises(
            SchemaConversionError,
            match="Index for field 'vector' missing required 'type'",
        ):
            builder.get_index_params({"field": "vector"})

    def test_autoindex_consistency_between_methods(self):
        """Test that get_index_params and get_milvus_index_params handle
        autoindex consistently."""
        schema_dict = create_schema_dict(
            "test_consistency",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vector", "type": "float_vector", "dim": 768},
            ],
            indexes=[
                {"field": "vector"},  # No type specified
            ],
        )
        schema_dict["autoindex"] = True

        builder = SchemaBuilder(schema_dict)
        mock_client = Mock()
        mock_index_params = Mock()
        mock_client.prepare_index_params.return_value = mock_index_params

        # Both methods should handle autoindex the same way
        index_params = builder.get_index_params({"field": "vector"})
        _milvus_params = builder.get_milvus_index_params(mock_client)

        # Verify get_index_params result
        assert index_params["index_type"] == "AUTOINDEX"

        # Verify get_milvus_index_params called add_index with same result
        mock_index_params.add_index.assert_called_once()
        call_args, call_kwargs = mock_index_params.add_index.call_args
        assert call_kwargs["field_name"] == "vector"
        assert call_kwargs["index_type"] == "AUTOINDEX"
