import pytest

from pyms.builders.schema import SchemaBuilder
from pyms.exceptions import SchemaConversionError
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

    def test_float16_vector_requires_dim(self):
        schema_dict = create_schema_dict(
            "float16_no_dim",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vec", "type": "float16_vector"},
            ],
        )
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="missing required 'dim'"):
            builder.build()

    def test_float16_vector_requires_newer_pymilvus(self, monkeypatch):
        from pyms import types

        monkeypatch.setitem(types.OPTIONAL_TYPE_SUPPORT, "float16_vector", False)

        schema_dict = create_schema_dict(
            "float16_requires_upgrade",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vec", "type": "float16_vector", "dim": 128},
            ],
        )

        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="Requires pymilvus>=2.6.0"):
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

    def test_gpu_index_disallows_cosine(self):
        schema_dict = create_schema_dict(
            "gpu_cosine",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vec", "type": "float_vector", "dim": 128},
            ],
            indexes=[{"field": "vec", "type": "GPU_IVF_FLAT", "metric": "COSINE"}],
        )
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(
            SchemaConversionError, match="COSINE' is not supported for GPU index"
        ):
            builder.get_create_index_calls()

    def test_gpu_cagra_requires_newer_pymilvus(self, monkeypatch):
        from pyms import types

        monkeypatch.setitem(types.OPTIONAL_INDEX_SUPPORT, "GPU_CAGRA", False)

        schema_dict = create_schema_dict(
            "gpu_cagra_requires_upgrade",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vec", "type": "float_vector", "dim": 128},
            ],
            indexes=[{"field": "vec", "type": "GPU_CAGRA", "metric": "L2"}],
        )
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match="Requires pymilvus>=2.6.0"):
            builder.get_create_index_calls()

    def test_int8_vector_allows_only_hnsw(self):
        schema_dict = create_schema_dict(
            "int8_vector_bad_index",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vec", "type": "int8_vector", "dim": 64},
            ],
            indexes=[{"field": "vec", "type": "IVF_FLAT", "metric": "L2"}],
        )
        builder = SchemaBuilder(schema_dict)
        with pytest.raises(SchemaConversionError, match=r"Valid types: \['HNSW'\]"):
            builder.get_create_index_calls()

        schema_dict_ok = create_schema_dict(
            "int8_vector_hnsw",
            [
                {"name": "id", "type": "int64", "is_primary": True},
                {"name": "vec", "type": "int8_vector", "dim": 64},
            ],
            indexes=[
                {
                    "field": "vec",
                    "type": "HNSW",
                    "metric": "L2",
                    "params": {"M": 16, "efConstruction": 200},
                }
            ],
        )
        builder_ok = SchemaBuilder(schema_dict_ok)
        # Should not raise
        builder_ok.get_create_index_calls()
