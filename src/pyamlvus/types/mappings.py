# Type mappings for pyamlvus
# Maps YAML field types to PyMilvus DataType objects

from pymilvus import DataType

# YAML type to PyMilvus DataType mapping
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
    "float16_vector": DataType.FLOAT16_VECTOR,
    "bfloat16_vector": DataType.BFLOAT16_VECTOR,
    "binary_vector": DataType.BINARY_VECTOR,
    "int8_vector": DataType.INT8_VECTOR,
    "sparse_float_vector": DataType.SPARSE_FLOAT_VECTOR,
}
