# pyamlvus

A Python library for managing Milvus schemas through YAML configuration files.

## Quick Start

**Load and validate schemas from YAML:**

```python
from pyamlvus import load_schema, validate_schema_file

# Load schema from YAML file
schema = load_schema("my_schema.yaml")

# Validate schema file
errors = validate_schema_file("my_schema.yaml")
if errors:
    for error in errors:
        print(f"Validation error: {error}")
else:
    print("Schema is valid!")
```

**Use with PyMilvus:**

```python
from pymilvus import MilvusClient

# Connect to Milvus
client = MilvusClient(...)

# Create collection using the loaded schema
client.create_collection(
    collection_name="my_collection",
    schema=schema
)
```

## YAML Schema Format

```yaml
name: "user_collection"
description: "User profile collection with embeddings"

fields:
  - name: "id"
    type: "int64"
    is_primary: true
    auto_id: false

   - name: "username"
     type: "varchar"
     max_length: 100

   - name: "description"
     type: "varchar"
     max_length: 1000
     enable_analyzer: true
     enable_match: true
     analyzer_params:
       type: "english"

  - name: "embedding"
    type: "float_vector"
    dim: 768

  - name: "metadata"
    type: "json"

indexes:
  - field: "embedding"
    type: "IVF_FLAT"
    metric: "L2"
    params:
      nlist: 1024

  - field: "username"
    type: "TRIE"

settings:
  consistency_level: "Strong"
  ttl_seconds: 3600
  enable_dynamic_field: true
```

## Architecture

YAML Schema → Parser → Validation → CollectionSchema → Milvus

**Clean separation of concerns:**

- **Parser**: YAML file loading with detailed error reporting
- **Validator**: Schema validation with field-specific rules
- **Builder**: CollectionSchema generation for PyMilvus
- **Types**: Comprehensive mappings to PyMilvus data types

## Installation

```bash
# Install with uv (when published)
uv add pyamlvus

# Development installation
git clone https://github.com/D-Joey-G/pyamlvus
cd pyamlvus
uv sync --all-groups
```

## API Reference

### Core API

**Load schema from YAML file:**

```python
from pyamlvus import load_schema

schema = load_schema("schema.yaml")
```

**Load schema dictionary:**

```python
from pyamlvus import load_schema_dict

schema_dict = load_schema_dict("schema.yaml")
```

**Validate schema file:**

```python
from pyamlvus import validate_schema_file

errors = validate_schema_file("schema.yaml")
if errors:
    for error in errors:
        print(f"Error: {error}")
```

### Schema Loading & Parsing

```python
from pyamlvus import SchemaLoader

loader = SchemaLoader("schema.yaml")
print(f"Collection: {loader.name}")
print(f"Fields: {len(loader.fields)}")
print(f"Indexes: {len(loader.indexes)}")

# Access parsed schema
schema_dict = loader.to_dict()
```

### Schema Building

```python
from pyamlvus import SchemaBuilder

builder = SchemaBuilder(schema_dict)
collection_schema = builder.build()

# Advanced: Get index parameters for MilvusClient
index_params = builder.get_milvus_index_params(client)
```

## Supported Field Types

| YAML Type               | PyMilvus Type            | Required Parameters        |
|------------------------|---------------------------|---------------------------|
| `int8`, `int16`, `int32`, `int64` | `DataType.INT*`    | None                      |
| `float`, `double`       | `DataType.FLOAT/DOUBLE`  | None                      |
| `varchar`              | `DataType.VARCHAR`       | `max_length`              |
| `json`                 | `DataType.JSON`          | None                      |
| `array`                | `DataType.ARRAY`         | `element_type`, `max_capacity` |
| `float_vector`         | `DataType.FLOAT_VECTOR`  | `dim`                     |
| `binary_vector`        | `DataType.BINARY_VECTOR` | `dim`                     |
| `float16_vector`       | `DataType.FLOAT16_VECTOR`| `dim` *(requires pymilvus ≥ 2.6)* |
| `bfloat16_vector`      | `DataType.BFLOAT16_VECTOR`| `dim` *(requires pymilvus ≥ 2.6)* |
| `sparse_float_vector`  | `DataType.SPARSE_FLOAT_VECTOR` | None                |
| `int8_vector`          | `DataType.INT8_VECTOR`   | `dim` *(requires pymilvus ≥ 2.6)* |
| `bool`                 | `DataType.BOOL`          | None                      |

## Supported Index Types

### Vector Indexes

- `FLAT`, `IVF_FLAT`, `IVF_SQ8`, `IVF_PQ`
- `HNSW`, `DISKANN`, `AUTOINDEX`
- `BIN_FLAT`, `BIN_IVF_FLAT` (for binary vectors)
- `SPARSE_INVERTED_INDEX` (for sparse vectors)
- GPU indexes: `GPU_IVF_FLAT`, `GPU_IVF_PQ`, `GPU_CAGRA`, `GPU_BRUTE_FORCE`
  - `GPU_*` indexes do not support the `COSINE` metric. Normalize vectors and use `IP` if cosine similarity is required.
  - `GPU_CAGRA` and `GPU_BRUTE_FORCE` require pymilvus ≥ 2.6.
  - `int8_vector` fields currently support only the `HNSW` index type.

### Scalar Indexes

- `TRIE` (for VARCHAR fields)
- `INVERTED` (for text search)
- `STL_SORT` (for numeric fields)

## Supported Metrics

- `L2` - Euclidean distance
- `IP` - Inner product
- `COSINE` - Cosine similarity
- `HAMMING` - Hamming distance (binary vectors)
- `JACCARD` - Jaccard distance (binary vectors)
- `TANIMOTO` - Tanimoto distance (binary vectors)
- `BM25` - BM25 scoring (sparse vectors)

## Runtime Requirements

Specify runtime compatibility directly in the schema when you rely on newer PyMilvus features:

```yaml
pymilvus:
  min_version: "2.6.0"
```

Supported keys: `min_version`, `max_version`, and `version` (aliases `require` / `exact_version`).
If the current client falls outside the declared range, pyamlvus raises a clear error before building the schema.

## Text Match Support

Enable keyword matching on VARCHAR fields for precise text retrieval:

```yaml
fields:
  - name: "description"
    type: "varchar"
    max_length: 1000
    enable_analyzer: true      # Required for text analysis
    enable_match: true         # Enable TEXT_MATCH expressions
    analyzer_params:
      type: "english"          # Optional: specify analyzer
```

Use in queries with `TEXT_MATCH` expressions:

```python
# Search for documents containing specific keywords
filter = "TEXT_MATCH(description, 'machine learning')"
results = client.search(
    collection_name="my_collection",
    filter=filter,
    # ... other search parameters
)
```

## Collection Settings

```yaml
settings:
  consistency_level: "Strong"     # Strong, Session, Bounded, Eventually
  ttl_seconds: 3600              # Time-to-live in seconds
  enable_dynamic_field: true     # Allow dynamic fields
```

## Examples

Check out the `examples/` directory:

- [`basic_schema.yaml`](examples/basic_schema.yaml) - Simple collection with ID, text, and vector
- [`complex_schema.yaml`](examples/complex_schema.yaml) - Advanced features with arrays, indexes, and settings
- [`text_match_example.yaml`](examples/text_match_example.yaml) - Text match functionality with keyword search
- [`usage_examples.py`](examples/usage_examples.py) - Comprehensive Python code examples

### Project Structure

```text
src/pyamlvus/
├── __init__.py          # Public API exports
├── api.py               # High-level convenience functions
├── parser.py            # YAML parsing and loading
├── exceptions.py        # Custom exception hierarchy
├── builders/
│   ├── schema.py        # CollectionSchema building
│   ├── field.py         # Field building
│   ├── index.py         # Index building
│   └── function.py      # Function building
├── types/
│   ├── constants.py     # Type and constant definitions
│   └── mappings.py      # Type mappings to PyMilvus
└── validators/          # Schema validation modules
tests/
├── test_parser.py       # Parser tests
├── test_builder.py      # Builder tests
├── test_integration.py  # End-to-end tests
└── fixtures/            # Test YAML files
```

## Features

- **YAML Schema Definition**: Define Milvus collection schemas using clean, declarative YAML syntax
- **Comprehensive Validation**: Validate schemas with detailed error messages and warnings
- **Type Safety**: Full type mappings to PyMilvus data types with parameter validation
- **Schema Building**: Convert YAML schemas to PyMilvus CollectionSchema objects
- **Error Handling**: Structured exceptions with helpful error messages
- **Text Match Support**: Enable keyword matching on VARCHAR fields for precise text retrieval

### Development Standards

- **Code Quality**: All code must pass `ruff` formatting and linting
- **Type Safety**: Use modern Python type hints throughout
- **Testing**: Write tests for new features
- **Documentation**: Update docstrings and examples

## License

This project is licensed under the MIT License - see LICENSE.md details.

## Acknowledgments

- Built on top of [PyMilvus](https://github.com/milvus-io/pymilvus) - the official Milvus Python SDK
- Inspired by the need for declarative schema management in vector databases
- Type mappings and constants sourced directly from PyMilvus codebase for accuracy

## Support

- **Documentation**: Check our examples and API reference above
- **Bug Reports**: Open an issue on GitHub
- **Feature Requests**: We'd love to hear your ideas!
- **Questions**: Start a discussion in our GitHub discussions

---

## IDE Support & Validation

### JSON Schema for YAML Validation

pyamlvus includes a JSON Schema file for IDE support and YAML validation. For VS Code, add the YAML extension from Red Hat, then add this to the top of your YAML schema files:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/D-Joey-G/pyamlvus/main/schema/pyamlvus_schema.json

name: "my_collection"
fields:
  # Your IDE will now provide autocompletion and validation!
```

Or configure your IDE to automatically validate `.yaml` files in your project using the schema at `schema/pyamlvus_schema.json`.

**Benefits:**

- **Autocompletion**: Field names, types, and parameters
- **Validation**: Real-time error checking as you type
- **Documentation**: Hover tooltips with field descriptions
- **Type Safety**: Catch schema errors before runtime

### VS Code Setup

Add to your `.vscode/settings.json`:

```json
{
  "yaml.schemas": {
    "./schema/pyamlvus_schema.json": ["*_schema.yaml", "**/schemas/*.yaml"]
  }
}
```

## CLI Tool

pyamlvus includes a command-line tool for schema validation and management:

```bash
# Validate a schema file
pyamlvus validate my_schema.yaml

# Convert YAML to CollectionSchema (future feature)
pyamlvus convert my_schema.yaml
```

**Status**: **Core Library** - Focused on YAML schema definition, validation, and CollectionSchema building for PyMilvus integration.

## Roadmap

### **Phase 1: Declarative Schema Definition** (Completed)

- [x] **YAML Parsing**: Load and parse schema files with error handling
- [x] **Type System**: Complete mappings to PyMilvus data types
- [x] **Exception Handling**: Structured error reporting
- [x] **Schema Validation**: Field and schema-level validation rules
- [x] **Schema Building**: Convert to PyMilvus CollectionSchema objects

### **Phase 2: Version Management** (Future)

- [ ] **Schema Diff**: Compare schema versions
- [ ] **Migration Recommendations**: Suggest migration paths
- [ ] **Backward Compatibility**: Check compatibility between versions
- [ ] **Version History**: Track schema evolution
