import pytest

from pyamlvus.builders.schema import SchemaBuilder
from pyamlvus.exceptions import SchemaConversionError
from tests.conftest import create_schema_dict


@pytest.mark.unit
class TestFieldParameterValidation:
    def test_float_vector_dim_out_of_range(self):
        schema_dict = create_schema_dict(
            "vec_dim_test",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vec", "type": "float_vector", "dim": 40000},
            ],
        )
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="dimension 40000"):
            builder.build()

    def test_varchar_length_out_of_range(self):
        schema_dict = create_schema_dict(
            "varchar_len_test",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "text", "type": "varchar", "max_length": 70000},
            ],
        )
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="max_length 70000"):
            builder.build()

    def test_array_capacity_out_of_range(self):
        schema_dict = create_schema_dict(
            "array_cap_test",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {
                    "name": "tags",
                    "type": "array",
                    "element_type": "int32",
                    "max_capacity": 10000,
                },
            ],
        )
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="max_capacity 10000"):
            builder.build()


@pytest.mark.unit
class TestIndexValidation:
    def test_ivf_missing_nlist(self):
        schema_dict = create_schema_dict(
            "ivf_missing_param",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vec", "type": "float_vector", "dim": 128},
            ],
            indexes=[{"field": "vec", "type": "IVF_FLAT", "metric": "L2"}],
        )
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            SchemaConversionError, match="missing required parameters: \\['nlist'\\]"
        ):
            builder.get_create_index_calls()

    def test_trie_on_non_varchar(self):
        schema_dict = create_schema_dict(
            "trie_wrong_field",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vec", "type": "float_vector", "dim": 128},
            ],
            indexes=[{"field": "vec", "type": "TRIE"}],
        )
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="TRIE"):
            builder.get_create_index_calls()

    def test_invalid_metric_for_float_vector(self):
        schema_dict = create_schema_dict(
            "bad_metric",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vec", "type": "float_vector", "dim": 128},
            ],
            indexes=[{"field": "vec", "type": "FLAT", "metric": "HAMMING"}],
        )
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="Invalid metric 'HAMMING'"):
            builder.get_create_index_calls()
