# Constants for pyamlvus
# Contains validation constants, index types, and metric definitions

# Minimal metric compatibility set for vector types
FLOAT_METRICS = {"L2", "IP", "COSINE"}
BINARY_METRICS = {"HAMMING", "JACCARD", "TANIMOTO"}

# Minimal required params by index type (subset used in tests/spec)
REQUIRED_PARAMS = {
    "IVF_FLAT": {"nlist"},
    "IVF_SQ8": {"nlist"},
    "IVF_PQ": {"nlist"},
    "HNSW": {"M", "efConstruction"},
    # FLAT, TRIE: no required params
}

# Valid index types by field type (case-insensitive input, normalized to uppercase)
VALID_INDEX_TYPES = {
    "float_vector": {
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
    },
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
