import pytest
import yaml

from pyamlvus.builders.schema import SchemaBuilder
from pyamlvus.exceptions import SchemaConversionError
from pyamlvus.parser import SchemaLoader


@pytest.mark.unit
class TestFunctionsParsing:
    def test_loader_functions_property(self, create_temp_yaml):
        schema_dict = {
            "name": "func_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "embedding", "type": "float_vector", "dim": 128},
            ],
            "functions": [
                {
                    "name": "embed_bm25",
                    "type": "BM25",
                    "field": "embedding",
                    "output_field": "embedding_norm",
                }
            ],
        }

        yaml_file = create_temp_yaml(yaml.dump(schema_dict))
        loader = SchemaLoader(yaml_file)

        funcs = loader.functions
        assert isinstance(funcs, list)
        assert len(funcs) == 1
        assert funcs[0]["type"] == "BM25"

    @pytest.mark.error_cases
    def test_loader_functions_type_error(self, create_temp_yaml):
        yaml_content = """
name: t
fields:
  - name: id
    type: int64
    is_primary: true
functions: "not a list"
"""
        yaml_file = create_temp_yaml(yaml_content)
        loader = SchemaLoader(yaml_file)
        with pytest.raises(Exception, match="Functions must be a list"):
            _ = loader.functions


@pytest.mark.unit
class TestFunctionsBuilder:
    def test_builder_functions_property(self):
        schema_dict = {
            "name": "f1",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "embedding", "type": "float_vector", "dim": 64},
            ],
            "functions": [
                {
                    "name": "embed_bm25",
                    "type": "BM25",
                    "field": "embedding",
                    "output_field": "embedding_norm",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        assert len(builder.functions) == 1

    def test_function_validation_unknown_field(self):
        schema_dict = {
            "name": "bad",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
            ],
            "functions": [
                {
                    "name": "test_bm25",
                    "type": "BM25",
                    "field": "missing_field",
                    "output_field": "sparse_out",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            Exception,
            match="Function input field 'missing_field' does not exist in schema",
        ):
            builder.get_milvus_function_objects()


@pytest.mark.unit
class TestFunctionTypes:
    """Test different function types and their validation."""

    def test_bm25_function_creation(self):
        """Test BM25 function with proper configuration."""
        schema_dict = {
            "name": "bm25_test",
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
                    "output_field_names": "sparse_vec",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        functions = builder.get_milvus_function_objects()
        assert len(functions) == 1
        assert functions[0].name == "text_bm25"

    def test_bm25_function_with_list_output(self):
        """Test BM25 function with list format output field."""
        schema_dict = {
            "name": "bm25_test",
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
        builder = SchemaBuilder(schema_dict)
        functions = builder.get_milvus_function_objects()
        assert len(functions) == 1

    def test_text_embedding_function_creation(self):
        """Test TEXT_EMBEDDING function with proper configuration."""
        schema_dict = {
            "name": "embedding_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 1000},
                {"name": "embedding", "type": "float_vector", "dim": 768},
            ],
            "functions": [
                {
                    "name": "text_embedding",
                    "type": "TEXT_EMBEDDING",
                    "input_field_names": ["text"],
                    "output_field_names": "embedding",
                    "params": {"model": "all-MiniLM-L6-v2"},
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        functions = builder.get_milvus_function_objects()
        assert len(functions) == 1
        assert functions[0].name == "text_embedding"

    @pytest.mark.parametrize("function_type", ["BM25", "bm25", "Bm25"])
    def test_function_type_case_insensitive(self, function_type):
        """Test that function types are case insensitive."""
        schema_dict = {
            "name": "case_test",
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
                    "name": "test_func",
                    "type": function_type,
                    "input_field_names": ["text"],
                    "output_field_names": "sparse_vec",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        functions = builder.get_milvus_function_objects()
        assert len(functions) == 1

    def test_unsupported_function_type(self):
        """Test error handling for unsupported function types."""
        schema_dict = {
            "name": "bad_func_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 1000},
            ],
            "functions": [
                {
                    "name": "bad_func",
                    "type": "UNKNOWN_TYPE",
                    "input_field_names": ["text"],
                    "output_field_names": "text",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="Unsupported function type"):
            builder.get_milvus_function_objects()


@pytest.mark.unit
class TestFunctionFieldValidation:
    """Test function field name validation and normalization."""

    @pytest.mark.parametrize(
        "input_field_param",
        ["input_field_names", "input_fields", "fields", "input_field", "field"],
    )
    def test_input_field_parameter_aliases(self, input_field_param):
        """Test that various input field parameter names work."""
        schema_dict = {
            "name": "field_test",
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
                    "name": "test_func",
                    "type": "BM25",
                    input_field_param: "text",
                    "output_field_names": "sparse_vec",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        functions = builder.get_milvus_function_objects()
        assert len(functions) == 1

    @pytest.mark.parametrize(
        "output_field_param", ["output_field_names", "output_field"]
    )
    def test_output_field_parameter_aliases(self, output_field_param):
        """Test that various output field parameter names work."""
        schema_dict = {
            "name": "field_test",
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
                    "name": "test_func",
                    "type": "BM25",
                    "input_field_names": "text",
                    output_field_param: "sparse_vec",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        functions = builder.get_milvus_function_objects()
        assert len(functions) == 1

    def test_input_field_as_list(self):
        """Test input field specified as list."""
        schema_dict = {
            "name": "list_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text1",
                    "type": "varchar",
                    "max_length": 1000,
                    "enable_analyzer": True,
                },
                {
                    "name": "text2",
                    "type": "varchar",
                    "max_length": 1000,
                    "enable_analyzer": True,
                },
                {"name": "sparse_vec", "type": "sparse_float_vector"},
            ],
            "functions": [
                {
                    "name": "multi_input",
                    "type": "BM25",
                    "input_field_names": ["text1", "text2"],
                    "output_field_names": "sparse_vec",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        functions = builder.get_milvus_function_objects()
        assert len(functions) == 1

    def test_missing_input_field(self):
        """Test error when input field is missing."""
        schema_dict = {
            "name": "missing_input_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "sparse_vec", "type": "sparse_float_vector"},
            ],
            "functions": [
                {
                    "name": "bad_func",
                    "type": "BM25",
                    "output_field_names": "sparse_vec",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="missing input field"):
            builder.get_milvus_function_objects()

    def test_missing_output_field(self):
        """Test error when output field is missing."""
        schema_dict = {
            "name": "missing_output_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 1000},
            ],
            "functions": [
                {
                    "name": "bad_func",
                    "type": "BM25",
                    "input_field_names": "text",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="missing output field"):
            builder.get_milvus_function_objects()

    def test_nonexistent_input_field(self):
        """Test error when input field doesn't exist."""
        schema_dict = {
            "name": "nonexistent_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "sparse_vec", "type": "sparse_float_vector"},
            ],
            "functions": [
                {
                    "name": "bad_func",
                    "type": "BM25",
                    "input_field_names": "nonexistent_field",
                    "output_field_names": "sparse_vec",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="does not exist in schema"):
            builder.get_milvus_function_objects()

    def test_nonexistent_output_field(self):
        """Test error when output field doesn't exist."""
        schema_dict = {
            "name": "nonexistent_output_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 1000},
            ],
            "functions": [
                {
                    "name": "bad_func",
                    "type": "BM25",
                    "input_field_names": "text",
                    "output_field_names": "nonexistent_output",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="must be an existing field"):
            builder.get_milvus_function_objects()

    def test_empty_output_field_list(self):
        """Test error when output field list is empty."""
        schema_dict = {
            "name": "empty_output_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 1000},
            ],
            "functions": [
                {
                    "name": "bad_func",
                    "type": "BM25",
                    "input_field_names": "text",
                    "output_field_names": [],
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="cannot be empty"):
            builder.get_milvus_function_objects()

    def test_invalid_input_field_type(self):
        """Test error when input field has incompatible type."""
        schema_dict = {
            "name": "bad_field_type_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "number", "type": "int32"},
                {"name": "sparse_vec", "type": "sparse_float_vector"},
            ],
            "functions": [
                {
                    "name": "bad_func",
                    "type": "BM25",
                    "input_field_names": "number",
                    "output_field_names": "sparse_vec",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="incompatible type"):
            builder.get_milvus_function_objects()


@pytest.mark.unit
class TestFunctionParameters:
    """Test function-specific parameter validation."""

    def test_bm25_with_valid_params(self):
        """Test BM25 function with valid parameters."""
        schema_dict = {
            "name": "bm25_params_test",
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
                    "name": "bm25_with_params",
                    "type": "BM25",
                    "input_field_names": "text",
                    "output_field_names": "sparse_vec",
                    "params": {"k1": 1.2, "b": 0.75},
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        functions = builder.get_milvus_function_objects()
        assert len(functions) == 1

    def test_bm25_without_analyzer_enabled(self):
        """Test BM25 function error when analyzer not enabled."""
        schema_dict = {
            "name": "bm25_no_analyzer_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "text",
                    "type": "varchar",
                    "max_length": 1000,
                },  # No enable_analyzer
                {"name": "sparse_vec", "type": "sparse_float_vector"},
            ],
            "functions": [
                {
                    "name": "bad_bm25",
                    "type": "BM25",
                    "input_field_names": "text",
                    "output_field_names": "sparse_vec",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            SchemaConversionError, match="requires enable_analyzer: true"
        ):
            builder.get_milvus_function_objects()

    def test_bm25_invalid_param_values(self):
        """Test BM25 function with invalid parameter values."""
        schema_dict = {
            "name": "bm25_bad_params_test",
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
                    "name": "bad_bm25",
                    "type": "BM25",
                    "input_field_names": "text",
                    "output_field_names": "sparse_vec",
                    "params": {
                        "k1": -1.0  # Invalid negative value
                    },
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="must be positive"):
            builder.get_milvus_function_objects()

    def test_bm25_invalid_param_names(self):
        """Test BM25 function with invalid parameter names."""
        schema_dict = {
            "name": "bm25_bad_param_names_test",
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
                    "name": "bad_bm25",
                    "type": "BM25",
                    "input_field_names": "text",
                    "output_field_names": "sparse_vec",
                    "params": {"invalid_param": 1.0},
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="invalid parameter"):
            builder.get_milvus_function_objects()

    def test_text_embedding_missing_model(self):
        """Test TEXT_EMBEDDING function without required model parameter."""
        schema_dict = {
            "name": "embedding_no_model_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 1000},
                {"name": "embedding", "type": "float_vector", "dim": 768},
            ],
            "functions": [
                {
                    "name": "bad_embedding",
                    "type": "TEXT_EMBEDDING",
                    "input_field_names": "text",
                    "output_field_names": "embedding",
                    "params": {},  # Missing model
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="missing required 'model'"):
            builder.get_milvus_function_objects()

    def test_text_embedding_missing_params(self):
        """Test TEXT_EMBEDDING function without params section."""
        schema_dict = {
            "name": "embedding_no_params_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 1000},
                {"name": "embedding", "type": "float_vector", "dim": 768},
            ],
            "functions": [
                {
                    "name": "bad_embedding",
                    "type": "TEXT_EMBEDDING",
                    "input_field_names": "text",
                    "output_field_names": "embedding",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="missing required 'params'"):
            builder.get_milvus_function_objects()


@pytest.mark.unit
class TestFunctionIndexRelationships:
    """Test function-index relationship validation."""

    def test_bm25_missing_index_warning(self):
        """Test that BM25 functions generate warnings when output field lacks index."""
        schema_dict = {
            "name": "bm25_no_index_test",
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
                    "name": "bm25_func",
                    "type": "BM25",
                    "input_field_names": "text",
                    "output_field_names": "sparse_vec",
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        warnings = builder.get_function_index_warnings()
        assert len(warnings) > 0
        assert any("BM25" in warning for warning in warnings)
        assert any("no index" in warning for warning in warnings)

    def test_bm25_with_correct_index(self):
        """Test that BM25 functions don't generate warnings with correct index."""
        schema_dict = {
            "name": "bm25_with_index_test",
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
                    "name": "bm25_func",
                    "type": "BM25",
                    "input_field_names": "text",
                    "output_field_names": "sparse_vec",
                }
            ],
            "indexes": [{"field": "sparse_vec", "type": "SPARSE_INVERTED_INDEX"}],
        }
        builder = SchemaBuilder(schema_dict)
        warnings = builder.get_function_index_warnings()
        # Should have no BM25 errors
        bm25_errors = [w for w in warnings if "BM25" in w and "ERROR" in w]
        assert len(bm25_errors) == 0

    def test_bm25_with_wrong_index_type(self):
        """Test that BM25 functions generate errors with wrong index type."""
        schema_dict = {
            "name": "bm25_wrong_index_test",
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
                    "name": "bm25_func",
                    "type": "BM25",
                    "input_field_names": "text",
                    "output_field_names": "sparse_vec",
                }
            ],
            "indexes": [
                {"field": "sparse_vec", "type": "HNSW"}  # Wrong index type for sparse
            ],
        }
        builder = SchemaBuilder(schema_dict)
        warnings = builder.get_function_index_warnings()
        # Should have BM25 errors about wrong index type
        bm25_errors = [w for w in warnings if "BM25" in w and "ERROR" in w]
        assert len(bm25_errors) > 0

    def test_function_output_with_multiple_fields(self):
        """Test function index validation with multiple output fields."""
        schema_dict = {
            "name": "multi_output_test",
            "fields": [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 1000},
                {"name": "vec1", "type": "float_vector", "dim": 128},
                {"name": "vec2", "type": "float_vector", "dim": 128},
            ],
            "functions": [
                {
                    "name": "multi_func",
                    "type": "TEXT_EMBEDDING",
                    "input_field_names": "text",
                    "output_field_names": ["vec1", "vec2"],
                    "params": {"model": "test-model"},
                }
            ],
        }
        builder = SchemaBuilder(schema_dict)
        warnings = builder.get_function_index_warnings()
        # Should warn about both output fields lacking indexes
        assert len(warnings) >= 2
        assert any("vec1" in w for w in warnings)
        assert any("vec2" in w for w in warnings)
