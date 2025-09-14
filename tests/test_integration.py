# Integration tests for end-to-end YAML to CollectionSchema conversion
import pytest

from pymilvus import CollectionSchema, DataType

from pyamlvus import SchemaBuilder, SchemaLoader, validate_schema
from pyamlvus.exceptions import SchemaConversionError, UnsupportedTypeError


@pytest.mark.integration
class TestEndToEndIntegration:
    """Test complete end-to-end workflows from YAML to CollectionSchema."""

    def test_end_to_end_valid_schema(self, valid_yaml_content, create_temp_yaml):
        """Test complete workflow with valid schema fixture."""
        yaml_file = create_temp_yaml(valid_yaml_content)

        # Test validation
        errors = validate_schema(yaml_file)
        # Should have warning for float_vector without index
        assert len(errors) == 1
        assert "WARNING" in errors[0]
        assert "float_vector" in errors[0].lower()

        # Test direct schema loading and building
        loader = SchemaLoader(yaml_file)
        schema_dict = loader.to_dict()
        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        assert isinstance(schema, CollectionSchema)
        assert schema.description == "Test collection from fixture"
        assert len(schema.fields) == 3

        # Check field properties
        field_names = [f.name for f in schema.fields]
        field_types = [f.dtype for f in schema.fields]

        assert "id" in field_names
        assert "username" in field_names
        assert "embedding" in field_names

        assert DataType.INT64 in field_types
        assert DataType.VARCHAR in field_types
        assert DataType.FLOAT_VECTOR in field_types

        # Check primary field
        primary_fields = [f for f in schema.fields if f.is_primary]
        assert len(primary_fields) == 1
        assert primary_fields[0].name == "id"

    def test_step_by_step_workflow(self, valid_yaml_content, create_temp_yaml):
        """Test the step-by-step workflow as documented."""
        yaml_file = create_temp_yaml(valid_yaml_content)

        # Step 1: Load YAML
        loader = SchemaLoader(yaml_file)
        schema_dict = loader.to_dict()

        # Verify loaded dict
        assert schema_dict["name"] == "fixture_test_collection"
        assert len(schema_dict["fields"]) == 3

        # Step 2: Build CollectionSchema
        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        # Verify final schema
        assert isinstance(schema, CollectionSchema)
        assert schema.description == "Test collection from fixture"

    def test_complex_schema_full_pipeline(self, complex_schema_dict, create_temp_yaml):
        """Test end-to-end with complex schema containing all features."""
        import yaml

        yaml_content = yaml.dump(complex_schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        # Validate
        errors = validate_schema(yaml_file)
        # Should have warnings for vector fields without indexes
        assert len(errors) >= 1
        warning_texts = " ".join(errors).lower()
        assert "warning" in warning_texts

        # Load via direct loading and building
        loader = SchemaLoader(yaml_file)
        schema_dict = loader.to_dict()
        builder = SchemaBuilder(schema_dict)
        schema = builder.build()

        assert isinstance(schema, CollectionSchema)
        assert schema.enable_dynamic_field is True
        assert len(schema.fields) == 13

        # Verify all field types are present
        field_types = {f.name: f.dtype for f in schema.fields}
        expected_types = [
            ("id", DataType.INT64),
            ("varchar_field", DataType.VARCHAR),
            ("float_vector_field", DataType.FLOAT_VECTOR),
            ("array_field", DataType.ARRAY),
        ]

        for field_name, expected_dtype in expected_types:
            assert field_types[field_name] == expected_dtype

    @pytest.mark.error_cases
    def test_invalid_schema_error_propagation(
        self, invalid_yaml_content, create_temp_yaml
    ):
        """Test that validation catches invalid schemas and errors propagate
        correctly."""
        yaml_file = create_temp_yaml(invalid_yaml_content)

        errors = validate_schema(yaml_file)

        # Should have at least one error
        assert len(errors) > 0
        assert all(isinstance(error, str) for error in errors)

        # Test that schema loading and building also fails with same error
        with pytest.raises((SchemaConversionError, UnsupportedTypeError)):
            loader = SchemaLoader(yaml_file)
            schema_dict = loader.to_dict()
            builder = SchemaBuilder(schema_dict)
            builder.build()

    def test_schema_properties_consistency(self, valid_yaml_content, create_temp_yaml):
        """Test that schema properties are consistent across loader and builder."""
        yaml_file = create_temp_yaml(valid_yaml_content)
        loader = SchemaLoader(yaml_file)

        # Test loader properties
        assert loader.name == "fixture_test_collection"
        assert loader.description == "Test collection from fixture"
        assert len(loader.fields) == 3

        # Build schema and compare
        builder = SchemaBuilder(loader.to_dict())
        schema = builder.build()

        # Properties should match
        assert schema.description == loader.description
        assert len(schema.fields) == len(loader.fields)

        # Field names should match
        loader_field_names = [f["name"] for f in loader.fields]
        schema_field_names = [f.name for f in schema.fields]
        assert loader_field_names == schema_field_names


@pytest.mark.integration
class TestIntegrationParametrized:
    """Parametrized integration tests for various scenarios."""

    @pytest.mark.parametrize(
        "schema_fixture_name",
        ["valid_schema_dict", "minimal_schema_dict", "complex_schema_dict"],
    )
    def test_all_fixture_schemas(self, schema_fixture_name, request, create_temp_yaml):
        """Test end-to-end conversion with all schema fixtures."""
        import yaml

        # Get the fixture by name
        schema_dict = request.getfixturevalue(schema_fixture_name)
        yaml_content = yaml.dump(schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        # Test complete pipeline
        errors = validate_schema(yaml_file)
        # Filter out warnings - only errors should prevent schema loading
        actual_errors = [e for e in errors if "ERROR" in e]
        assert actual_errors == []

        loader = SchemaLoader(yaml_file)
        schema_dict = loader.to_dict()
        builder = SchemaBuilder(schema_dict)
        schema = builder.build()
        assert isinstance(schema, CollectionSchema)
        assert len(schema.fields) > 0

        # Verify exactly one primary field
        primary_fields = [f for f in schema.fields if f.is_primary]
        assert len(primary_fields) == 1

    @pytest.mark.parametrize(
        "field_combination",
        [
            # Basic combinations
            [{"name": "id", "type": "int64", "is_primary": True}],
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 100},
            ],
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 100},
                {"name": "vector", "type": "float_vector", "dim": 256},
            ],
            # All basic types
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "int8_f", "type": "int8"},
                {"name": "float_f", "type": "float"},
                {"name": "json_f", "type": "json"},
                {"name": "text_f", "type": "varchar", "max_length": 50},
            ],
        ],
    )
    def test_field_combinations(self, field_combination, create_temp_yaml):
        """Test various field combinations work end-to-end."""
        import yaml

        schema_dict = {
            "name": "combo_test",
            "description": "Field combination test",
            "fields": field_combination,
        }

        yaml_content = yaml.dump(schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        # Should validate successfully
        errors = validate_schema(yaml_file)
        # Filter out warnings - only errors should prevent schema loading
        actual_errors = [e for e in errors if "ERROR" in e]
        assert actual_errors == []

        # Should convert successfully
        loader = SchemaLoader(yaml_file)
        schema_dict = loader.to_dict()
        builder = SchemaBuilder(schema_dict)
        schema = builder.build()
        assert isinstance(schema, CollectionSchema)
        assert len(schema.fields) == len(field_combination)

    @pytest.mark.error_cases
    @pytest.mark.parametrize(
        "error_scenario,expected_error_type",
        [
            (
                # Missing primary key
                {
                    "name": "no_primary",
                    "fields": [{"name": "text", "type": "varchar", "max_length": 100}],
                },
                SchemaConversionError,
            ),
            (
                # Unknown field type
                {
                    "name": "bad_type",
                    "fields": [
                        {"name": "id", "type": "int64", "is_primary": True},
                        {"name": "bad", "type": "unknown"},
                    ],
                },
                UnsupportedTypeError,
            ),
            (
                # Missing required parameter
                {
                    "name": "bad_params",
                    "fields": [
                        {"name": "id", "type": "int64", "is_primary": True},
                        {"name": "vec", "type": "float_vector"},
                    ],
                },
                SchemaConversionError,
            ),
        ],
    )
    def test_error_scenarios_end_to_end(
        self, error_scenario, expected_error_type, create_temp_yaml
    ):
        """Test that various error scenarios are caught end-to-end."""
        import yaml

        yaml_content = yaml.dump(error_scenario)
        yaml_file = create_temp_yaml(yaml_content)

        # Validation should catch the error
        errors = validate_schema(yaml_file)
        assert len(errors) > 0

        # Direct loading should also fail with specific exception
        with pytest.raises(expected_error_type):
            loader = SchemaLoader(yaml_file)
            schema_dict = loader.to_dict()
            builder = SchemaBuilder(schema_dict)
            builder.build()


@pytest.mark.integration
@pytest.mark.performance
class TestPerformanceIntegration:
    """Performance tests for integration scenarios."""

    def test_large_schema_performance(self, create_temp_yaml):
        """Test performance with large schemas."""
        import time

        import yaml

        # Create a large schema with many fields
        fields = [{"name": "id", "type": "int64", "is_primary": True}]
        for i in range(100):  # 100 additional fields
            fields.append({"name": f"field_{i}", "type": "varchar", "max_length": 100})

        large_schema = {
            "name": "large_test",
            "description": "Large schema performance test",
            "fields": fields,
        }

        yaml_content = yaml.dump(large_schema)
        yaml_file = create_temp_yaml(yaml_content)

        # Time the conversion
        start_time = time.time()
        loader = SchemaLoader(yaml_file)
        schema_dict = loader.to_dict()
        builder = SchemaBuilder(schema_dict)
        schema = builder.build()
        end_time = time.time()

        # Should complete reasonably quickly (less than 1 second)
        duration = end_time - start_time
        assert duration < 1.0

        # Verify correctness
        assert isinstance(schema, CollectionSchema)
        assert len(schema.fields) == 101

    @pytest.mark.parametrize("iterations", [10, 50])
    def test_repeated_loading_performance(
        self, valid_yaml_content, create_temp_yaml, iterations
    ):
        """Test performance of repeated loading operations."""
        import time

        yaml_file = create_temp_yaml(valid_yaml_content)

        start_time = time.time()

        for _ in range(iterations):
            loader = SchemaLoader(yaml_file)
            schema_dict = loader.to_dict()
            builder = SchemaBuilder(schema_dict)
            schema = builder.build()
            assert isinstance(schema, CollectionSchema)

        end_time = time.time()
        duration = end_time - start_time

        # Should maintain reasonable performance
        avg_time_per_load = duration / iterations
        assert avg_time_per_load < 0.1  # Less than 100ms per load on average
