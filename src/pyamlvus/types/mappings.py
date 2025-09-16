from pymilvus import DataType

from .compat import OPTIONAL_TYPE_SUPPORT

TYPE_MAPPING = {
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

if OPTIONAL_TYPE_SUPPORT.get("float16_vector") and hasattr(DataType, "FLOAT16_VECTOR"):
    TYPE_MAPPING["float16_vector"] = DataType.FLOAT16_VECTOR

if OPTIONAL_TYPE_SUPPORT.get("bfloat16_vector") and hasattr(
    DataType, "BFLOAT16_VECTOR"
):
    TYPE_MAPPING["bfloat16_vector"] = DataType.BFLOAT16_VECTOR

if OPTIONAL_TYPE_SUPPORT.get("int8_vector") and hasattr(DataType, "INT8_VECTOR"):
    TYPE_MAPPING["int8_vector"] = DataType.INT8_VECTOR
