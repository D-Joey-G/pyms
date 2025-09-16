import pytest
import yaml

from pyamlvus import validate_schema


@pytest.mark.unit
class TestSchemaValidation:
    """Test the validate_schema convenience function."""

    def test_valid_schema_validation(self, valid_yaml_content, create_temp_yaml):
        """Test validation of valid schema returns no errors."""
        yaml_file = create_temp_yaml(valid_yaml_content)

        errors = validate_schema(yaml_file)

        # The schema has a float_vector field without an index, so it should
        # produce a warning
        assert len(errors) == 1
        assert "WARNING" in errors[0]
        assert "float_vector" in errors[0].lower()
        assert isinstance(errors, list)

    def test_minimal_schema_validation(self, minimal_schema_dict, create_temp_yaml):
        """Test validation of minimal valid schema."""

        yaml_content = yaml.dump(minimal_schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        errors = validate_schema(yaml_file)
        assert errors == []

    def test_complex_schema_validation(self, complex_schema_dict, create_temp_yaml):
        """Test validation of complex schema with all features."""

        yaml_content = yaml.dump(complex_schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        errors = validate_schema(yaml_file)
        # The complex schema has vector fields without indexes, so warnings are expected
        assert len(errors) >= 1
        # Should have warnings about missing indexes on vector fields
        warning_texts = " ".join(errors).lower()
        assert "warning" in warning_texts
        assert (
            "sparse" in warning_texts
            or "binary" in warning_texts
            or "dense" in warning_texts
        )

    @pytest.mark.error_cases
    def test_invalid_schema_validation(self, invalid_yaml_content, create_temp_yaml):
        """Test validation catches invalid schemas."""
        yaml_file = create_temp_yaml(invalid_yaml_content)

        errors = validate_schema(yaml_file)

        assert len(errors) > 0
        assert isinstance(errors, list)
        assert all(isinstance(error, str) for error in errors)

        # Should contain meaningful error information
        error_text = " ".join(errors)
        assert any(
            keyword in error_text.lower()
            for keyword in ["type", "field", "parameter", "missing"]
        )

    @pytest.mark.error_cases
    def test_yaml_parse_error_validation(
        self, yaml_parse_error_content, create_temp_yaml
    ):
        """Test validation catches YAML syntax errors."""
        yaml_file = create_temp_yaml(yaml_parse_error_content)

        errors = validate_schema(yaml_file)

        assert len(errors) > 0
        error_text = errors[0]
        assert "yaml" in error_text.lower() or "parsing" in error_text.lower()

    def test_nonexistent_file_validation(self):
        """Test validation of nonexistent file."""
        errors = validate_schema("nonexistent_file.yaml")

        assert len(errors) > 0
        assert "not found" in errors[0].lower() or "file" in errors[0].lower()

    @pytest.mark.parametrize(
        "error_schema,expected_keywords",
        [
            ({"name": "test", "fields": []}, ["field", "at least", "one"]),
            (
                {"name": "test", "fields": [{"name": "bad", "type": "unknown_type"}]},
                ["unsupported", "type"],
            ),
            (
                {
                    "name": "test",
                    "fields": [
                        {"name": "id", "type": "int64", "is_primary": True},
                        {"name": "text", "type": "varchar"},
                    ],
                },
                ["varchar", "max_length", "missing"],
            ),
            (
                {
                    "name": "test",
                    "fields": [{"name": "text", "type": "varchar", "max_length": 100}],
                },
                ["primary", "exactly one"],
            ),
        ],
    )
    def test_specific_validation_errors(
        self, error_schema, expected_keywords, create_temp_yaml
    ):
        """Test specific validation error scenarios and messages."""

        yaml_content = yaml.dump(error_schema)
        yaml_file = create_temp_yaml(yaml_content)

        errors = validate_schema(yaml_file)

        assert len(errors) > 0
        error_text = " ".join(errors).lower()

        # Check that expected keywords appear in error message
        for keyword in expected_keywords:
            assert keyword.lower() in error_text, (
                f"Expected keyword '{keyword}' not found in error: {error_text}"
            )

    def test_nullable_must_be_boolean(self, create_temp_yaml):
        """Test validation enforces boolean type for nullable flag."""
        import yaml

        schema = {
            "name": "nullable_flag_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "flag", "type": "bool", "nullable": "yes"},
            ],
        }

        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        errors = validate_schema(yaml_file)
        assert len(errors) > 0
        assert "nullable" in errors[0].lower()


@pytest.mark.unit
class TestValidationEdgeCases:
    """Test edge cases and boundary conditions for validation."""

    def test_empty_file_validation(self, create_temp_yaml):
        """Test validation of empty file."""
        yaml_file = create_temp_yaml("")

        errors = validate_schema(yaml_file)

        assert len(errors) > 0
        assert "empty" in errors[0].lower()

    def test_non_dict_yaml_validation(self, create_temp_yaml):
        """Test validation of YAML that's not a dictionary."""
        yaml_file = create_temp_yaml("- item1\n- item2")

        errors = validate_schema(yaml_file)

        assert len(errors) > 0
        assert "dictionary" in errors[0].lower()

    @pytest.mark.parametrize("field_count", [1, 10, 50])
    def test_validation_scales_with_field_count(self, field_count, create_temp_yaml):
        """Test that validation works with varying field counts."""

        # Create schema with specified number of fields
        fields = [{"name": "id", "type": "int64", "is_primary": True}]
        for i in range(1, field_count):
            fields.append({"name": f"field_{i}", "type": "varchar", "max_length": 100})

        schema_dict = {"name": "scale_test", "fields": fields}

        yaml_content = yaml.dump(schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        errors = validate_schema(yaml_file)
        assert errors == []

    def test_validation_with_all_field_types(
        self, complex_schema_dict, create_temp_yaml
    ):
        """Test validation works with all supported field types."""

        yaml_content = yaml.dump(complex_schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        errors = validate_schema(yaml_file)
        # Complex schema has vector fields without indexes, so warnings are expected
        assert len(errors) >= 1
        # Should have warnings about missing indexes
        warning_texts = " ".join(errors).lower()
        assert "warning" in warning_texts

    @pytest.mark.parametrize(
        "settings_config",
        [
            {},
            {"enable_dynamic_field": True},
            {"enable_dynamic_field": False, "other_setting": "value"},
        ],
    )
    def test_validation_with_various_settings(self, settings_config, create_temp_yaml):
        """Test validation with different settings configurations."""

        schema_dict = {
            "name": "settings_test",
            "fields": [{"name": "id", "type": "int64", "is_primary": True}],
            "settings": settings_config,
        }

        yaml_content = yaml.dump(schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        errors = validate_schema(yaml_file)
        assert errors == []


@pytest.mark.unit
class TestValidationPerformance:
    """Test validation performance characteristics."""

    @pytest.mark.performance
    def test_validation_performance(self, create_temp_yaml):
        """Test validation performance with reasonably sized schema."""
        import time

        # Create moderately complex schema
        fields = [{"name": "id", "type": "int64", "is_primary": True}]
        for i in range(20):
            fields.extend(
                [
                    {"name": f"text_{i}", "type": "varchar", "max_length": 100},
                    {"name": f"int_{i}", "type": "int32"},
                    {"name": f"vector_{i}", "type": "float_vector", "dim": 128},
                ]
            )

        schema_dict = {
            "name": "performance_test",
            "description": "Performance test schema",
            "fields": fields,
            "settings": {"enable_dynamic_field": True},
        }

        yaml_content = yaml.dump(schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        # Time the validation
        start_time = time.time()
        errors = validate_schema(yaml_file)
        end_time = time.time()

        # Should be fast (less than 100ms) and have warnings for missing indexes
        duration = end_time - start_time
        assert duration < 0.1
        # Should have warnings for the 20 vector fields without indexes
        assert len(errors) == 20
        assert all("WARNING" in error for error in errors)
        assert all("float_vector" in error.lower() for error in errors)

    @pytest.mark.parametrize("iterations", [5, 20])
    def test_repeated_validation_performance(
        self, valid_yaml_content, create_temp_yaml, iterations
    ):
        """Test performance of repeated validation calls."""
        import time

        yaml_file = create_temp_yaml(valid_yaml_content)

        start_time = time.time()

        for _ in range(iterations):
            errors = validate_schema(yaml_file)
            # Should have warning for float_vector field without index
            assert len(errors) == 1
            assert "WARNING" in errors[0]
            assert "float_vector" in errors[0].lower()

        end_time = time.time()
        duration = end_time - start_time

        # Should maintain reasonable performance
        avg_time_per_validation = duration / iterations
        assert avg_time_per_validation < 0.05  # Less than 50ms per validation


@pytest.mark.unit
class TestValidationExceptionHandling:
    """Test how validation handles various exception scenarios."""

    def test_validation_catches_conversion_errors(self, create_temp_yaml):
        """Test that validation properly catches and formats conversion errors."""

        # Schema that will cause SchemaConversionError
        error_schema = {
            "name": "conversion_error_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "id2",
                    "type": "int64",
                    "is_primary": True,
                },  # Multiple primary keys
            ],
        }

        yaml_content = yaml.dump(error_schema)
        yaml_file = create_temp_yaml(yaml_content)

        errors = validate_schema(yaml_file)

        assert len(errors) > 0
        assert "primary" in errors[0].lower()

    def test_validation_catches_unsupported_type_errors(self, create_temp_yaml):
        """Test that validation properly catches and formats unsupported type errors."""

        # Schema with unsupported type
        error_schema = {
            "name": "unsupported_type_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "bad_field", "type": "completely_unknown_type"},
            ],
        }

        yaml_content = yaml.dump(error_schema)
        yaml_file = create_temp_yaml(yaml_content)

        errors = validate_schema(yaml_file)

        assert len(errors) > 0
        error_text = errors[0].lower()
        assert "unsupported" in error_text or "unknown" in error_text
        assert "type" in error_text

    def test_validation_handles_unexpected_errors_gracefully(self, create_temp_yaml):
        """Test that validation handles unexpected errors gracefully."""
        # This is harder to test without mocking, but we can at least verify
        # that validation returns errors as strings, not exceptions

        yaml_file = create_temp_yaml("malformed: yaml: content: [")

        errors = validate_schema(yaml_file)

        # Should return errors as strings, not raise exceptions
        assert isinstance(errors, list)
        if errors:  # If there are errors, they should be strings
            assert all(isinstance(error, str) for error in errors)

    def test_validation_error_messages_are_user_friendly(self, create_temp_yaml):
        """Test that validation error messages are informative for users."""

        # Create a schema with multiple types of errors
        problematic_schema = {
            "name": "multi_error_test",
            "fields": [
                {"name": "bad_vector", "type": "float_vector"},  # Missing dim
                {"name": "bad_varchar", "type": "varchar"},  # Missing max_length
                {"name": "unknown_field", "type": "fake_type"},  # Unknown type
            ],
        }

        yaml_content = yaml.dump(problematic_schema)
        yaml_file = create_temp_yaml(yaml_content)

        errors = validate_schema(yaml_file)

        # Should catch multiple errors, but validation stops at first error
        assert len(errors) >= 1

        # Error message should be descriptive
        error_text = errors[0]
        assert len(error_text) > 10  # Should be a meaningful message
        assert any(
            word in error_text.lower()
            for word in ["field", "type", "parameter", "missing", "required"]
        )


@pytest.mark.unit
class TestEnhancedValidation:
    """Test the enhanced validation features."""

    def test_collection_name_validation(self, create_temp_yaml):
        """Test collection name validation rules."""

        # Valid collection names
        valid_names = [
            "my_collection",
            "MyCollection",
            "collection123",
            "col_lection_123",
        ]
        for name in valid_names:
            schema = {
                "name": name,
                "fields": [{"name": "id", "type": "int64", "is_primary": True}],
            }
            yaml_content = yaml.dump(schema)
            yaml_file = create_temp_yaml(yaml_content)
            errors = validate_schema(yaml_file)
            assert errors == [], (
                f"Valid name '{name}' should not produce errors: {errors}"
            )

        # Invalid collection names
        invalid_cases = [
            ("", "Collection name cannot be empty"),
            ("_invalid", "Collection name must start with a letter"),
            ("123invalid", "Collection name must start with a letter"),
            ("invalid-name", "Collection name must start with a letter"),
            ("invalid name", "Collection name must start with a letter"),
            ("invalid@name", "Collection name must start with a letter"),
        ]

        for invalid_name, expected_error in invalid_cases:
            schema = {
                "name": invalid_name,
                "fields": [{"name": "id", "type": "int64", "is_primary": True}],
            }
            yaml_content = yaml.dump(schema)
            yaml_file = create_temp_yaml(yaml_content)
            errors = validate_schema(yaml_file)
            assert len(errors) > 0, (
                f"Invalid name '{invalid_name}' should produce errors"
            )
            assert expected_error in errors[0], (
                f"Error for '{invalid_name}' should contain: {expected_error}"
            )

    def test_index_case_sensitivity_validation(self, create_temp_yaml):
        """Test index type case sensitivity handling."""

        # Test case sensitivity for TRIE index
        schema = {
            "name": "test_collection",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 100},
            ],
            "indexes": [
                {"field": "text", "type": "trie"},  # lowercase
            ],
        }
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        errors = validate_schema(yaml_file)
        assert len(errors) > 0
        assert (
            "should be 'TRIE'" in errors[0]
            or "Consider 'INVERTED' for better performance" in errors[0]
        )

        # Test valid uppercase (should give info about better alternatives)
        schema["indexes"][0]["type"] = "TRIE"
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        errors = validate_schema(yaml_file)
        assert len(errors) == 1
        assert "Consider 'INVERTED' for better performance" in errors[0]

    def test_sparse_vector_index_requirement(self, create_temp_yaml):
        """Test that sparse vector fields require appropriate indexes."""

        # Sparse vector without index should produce warning
        schema = {
            "name": "test_collection",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "sparse_vec", "type": "sparse_float_vector"},
            ],
        }
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        messages = validate_schema(yaml_file)
        assert len(messages) > 0
        assert "WARNING" in messages[0]
        assert "sparse_float_vector" in messages[0].lower()
        assert "no index defined" in messages[0].lower()

        # Sparse vector with wrong index type should produce error
        schema["indexes"] = [{"field": "sparse_vec", "type": "HNSW"}]
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        messages = validate_schema(yaml_file)
        assert len(messages) > 0
        assert "not valid" in messages[0] or "HNSW" in messages[0]

        # Sparse vector with correct index should be fine
        schema["indexes"] = [{"field": "sparse_vec", "type": "SPARSE_INVERTED_INDEX"}]
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        messages = validate_schema(yaml_file)
        # Should have no errors, only possibly other warnings
        errors = [m for m in messages if "ERROR" in m]
        assert len(errors) == 0

    def test_dense_vector_index_recommendation(self, create_temp_yaml):
        """Test that float_vector fields get index recommendations."""

        # Dense vector without index should produce warning
        schema = {
            "name": "test_collection",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "dense_vec", "type": "float_vector", "dim": 128},
            ],
        }
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        messages = validate_schema(yaml_file)
        assert len(messages) > 0
        assert "WARNING" in messages[0]
        assert "float_vector" in messages[0].lower()
        assert "no index defined" in messages[0].lower()

        # Dense vector with index should not produce warning
        schema["indexes"] = [{"field": "dense_vec", "type": "AUTOINDEX"}]
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        messages = validate_schema(yaml_file)
        # Should have no float_vector warnings
        dense_warnings = [m for m in messages if "float_vector" in m.lower()]
        assert len(dense_warnings) == 0

    def test_empty_index_type_handling(self, create_temp_yaml):
        """Test that empty index types default to AUTOINDEX."""

        # Empty index type should be handled gracefully
        schema = {
            "name": "test_collection",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "dense_vec", "type": "float_vector", "dim": 128},
            ],
            "indexes": [
                {"field": "dense_vec", "type": ""},  # Empty type
            ],
        }
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        messages = validate_schema(yaml_file)
        # Should not produce errors about missing type
        type_errors = [m for m in messages if "missing required 'type'" in m]
        assert len(type_errors) == 0

    def test_function_index_relationship_validation(self, create_temp_yaml):
        """Test validation of function-index relationships."""

        # BM25 function without index on output field
        schema = {
            "name": "test_collection",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text",
                    "type": "varchar",
                    "max_length": 1000,
                    "enable_analyzer": True,
                },
                {"name": "sparse_vec", "type": "sparse_float_vector"},
            ],
            "functions": [
                {
                    "name": "text_bm25",
                    "type": "BM25",
                    "input_field_names": ["text"],
                    "output_field_names": ["sparse_vec"],
                }
            ],
        }
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        messages = validate_schema(yaml_file)
        # Should have warning about missing index on BM25 output field
        bm25_warnings = [m for m in messages if "BM25" in m and "WARNING" in m]
        assert len(bm25_warnings) > 0

        # BM25 function with correct index should be fine
        schema["indexes"] = [{"field": "sparse_vec", "type": "SPARSE_INVERTED_INDEX"}]
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        messages = validate_schema(yaml_file)
        # Should have no BM25 errors
        bm25_errors = [m for m in messages if "BM25" in m and "ERROR" in m]
        assert len(bm25_errors) == 0

    def test_invalid_index_types_by_field_type(self, create_temp_yaml):
        """Test that invalid index types for field types are caught."""

        # TRIE index on non-varchar field
        schema = {
            "name": "test_collection",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "number", "type": "int32"},
            ],
            "indexes": [
                {"field": "number", "type": "TRIE"},
            ],
        }
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        errors = validate_schema(yaml_file)
        assert len(errors) > 0
        assert "TRIE" in errors[0] and "not valid" in errors[0]

        # Vector index on scalar field
        schema["indexes"] = [{"field": "number", "type": "HNSW"}]
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        errors = validate_schema(yaml_file)
        assert len(errors) > 0
        assert "HNSW" in errors[0] and "not valid" in errors[0]

    def test_enable_match_validation(self, create_temp_yaml):
        """Test validation of enable_match parameter."""
        # Valid enable_match=true
        schema = {
            "name": "test_collection",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text",
                    "type": "varchar",
                    "max_length": 100,
                    "enable_match": True,
                },
            ],
        }
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        errors = validate_schema(yaml_file)
        # Should not have validation errors for enable_match
        enable_match_errors = [e for e in errors if "enable_match" in e]
        assert len(enable_match_errors) == 0

        # Valid enable_match=false
        schema["fields"][1]["enable_match"] = False
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        errors = validate_schema(yaml_file)
        enable_match_errors = [e for e in errors if "enable_match" in e]
        assert len(enable_match_errors) == 0

        # Invalid enable_match (non-boolean)
        schema["fields"][1]["enable_match"] = "true"
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        errors = validate_schema(yaml_file)
        enable_match_errors = [
            e for e in errors if "enable_match" in e and "boolean" in e
        ]
        assert len(enable_match_errors) > 0

    def test_enable_match_with_analyzer(self, create_temp_yaml):
        """Test enable_match works with enable_analyzer."""
        schema = {
            "name": "test_collection",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text",
                    "type": "varchar",
                    "max_length": 100,
                    "enable_analyzer": True,
                    "enable_match": True,
                    "analyzer_params": {"type": "english"},
                },
            ],
        }
        yaml_content = yaml.dump(schema)
        yaml_file = create_temp_yaml(yaml_content)
        errors = validate_schema(yaml_file)
        # Should not have validation errors
        assert len(errors) == 0
