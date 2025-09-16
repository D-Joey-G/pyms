from .compat import (
    OPTIONAL_INDEX_SUPPORT,
    OPTIONAL_TYPE_SUPPORT,
)

FLOAT_METRICS = {"L2", "IP", "COSINE"}
BINARY_METRICS = {"HAMMING", "JACCARD", "TANIMOTO"}

REQUIRED_PARAMS = {
    "IVF_FLAT": {"nlist"},
    "IVF_SQ8": {"nlist"},
    "IVF_PQ": {"nlist"},
    "HNSW": {"M", "efConstruction"},
    # FLAT, TRIE: no required params
}

_BASE_VECTOR_INDEXES = {
    "FLAT",
    "IVF_FLAT",
    "IVF_SQ8",
    "IVF_PQ",
    "IVF_RABITQ",
    "HNSW",
    "DISKANN",
    "AUTOINDEX",
    "GPU_IVF_FLAT",
    "GPU_IVF_PQ",
}

_BASE_GPU_INDEXES = {"GPU_IVF_FLAT", "GPU_IVF_PQ"}
_OPTIONAL_GPU_INDEXES = {
    index for index, supported in OPTIONAL_INDEX_SUPPORT.items() if supported
}

_ALL_GPU_INDEXES = _BASE_GPU_INDEXES | _OPTIONAL_GPU_INDEXES

_FLOAT_VECTOR_INDEXES = _BASE_VECTOR_INDEXES | _OPTIONAL_GPU_INDEXES

VALID_INDEX_TYPES = {
    "float_vector": set(_FLOAT_VECTOR_INDEXES),
    "binary_vector": {"BIN_FLAT", "BIN_IVF_FLAT", "MINHASH_LSH"},
    "sparse_float_vector": {"SPARSE_INVERTED_INDEX"},
    "varchar": {"INVERTED", "BITMAP", "TRIE"},
    "int8": {"INVERTED", "STL_SORT"},
    "int16": {"INVERTED", "STL_SORT"},
    "int32": {"INVERTED", "STL_SORT"},
    "int64": {"INVERTED", "STL_SORT"},
    "float": {"INVERTED"},
    "double": {"INVERTED"},
    "bool": {"BITMAP", "INVERTED"},
    "array": {"BITMAP", "INVERTED"},  # Depends on element type
    "json": {"INVERTED"},
}

if OPTIONAL_TYPE_SUPPORT.get("float16_vector"):
    VALID_INDEX_TYPES["float16_vector"] = set(_FLOAT_VECTOR_INDEXES)

if OPTIONAL_TYPE_SUPPORT.get("bfloat16_vector"):
    VALID_INDEX_TYPES["bfloat16_vector"] = set(_FLOAT_VECTOR_INDEXES)

if OPTIONAL_TYPE_SUPPORT.get("int8_vector"):
    VALID_INDEX_TYPES["int8_vector"] = {"HNSW"}

# Recommended index types for each field type
RECOMMENDED_INDEX_TYPES = {
    "varchar": "INVERTED",
    "bool": "BITMAP",
    "int8": "INVERTED",
    "int16": "INVERTED",
    "int32": "INVERTED",
    "int64": "INVERTED",
    "float": "INVERTED",
    "double": "INVERTED",
    "array": "BITMAP",
    "json": "INVERTED",
}

if OPTIONAL_TYPE_SUPPORT.get("int8_vector"):
    RECOMMENDED_INDEX_TYPES["int8_vector"] = "HNSW"

GPU_INDEX_TYPES = set(_ALL_GPU_INDEXES)
