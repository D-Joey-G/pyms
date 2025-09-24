# pyms

pyms stands for **P**ython **Y**AML **M**ilvus **S**chemas — a library for declaratively managing Milvus collection schemas using YAML.

## Quick Start

**Load and validate schemas from YAML:**

```python
from pyms import load_schema, validate_schema_file, create_collection_from_yaml

# Load schema from YAML file
schema = load_schema("my_schema.yaml")

# Validate schema file
result = validate_schema_file("my_schema.yaml")
if result.has_errors():
    for message in result.errors:
        print(f"Validation error: {message.text}")
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

# Or create collection directly by passing client and path
create_collection_from_yaml(
    file_path="my_schema.yaml",
    client=client
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

## Runtime Requirements

Specify runtime compatibility directly in the schema when you rely on newer PyMilvus features:

```yaml
pymilvus:
  min_version: "2.6.0"
```

Supported keys: `min_version`, `max_version`, and `version` (aliases `require` / `exact_version`).
If the current client falls outside the declared range, pyms raises a clear error before building the schema.

## License

This project is licensed under the MIT License - see LICENSE.md details.

## Acknowledgments

- Built on top of [PyMilvus](https://github.com/milvus-io/pymilvus) - the official Milvus Python SDK
- Type mappings and constants sourced directly from PyMilvus codebase for accuracy

---

## IDE Support & Validation

### JSON Schema for YAML Validation

pyms includes a JSON Schema file for IDE support and YAML validation. For VS Code, add the YAML extension from Red Hat, then add this to the top of your YAML schema files:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/D-Joey-G/pyms/main/schema/pyms_schema.json

name: "my_collection"
fields:
  # Your IDE will now provide autocompletion and validation!
```

Or configure your IDE to automatically validate `.yaml` files in your project using the schema at `schema/pyms_schema.json`.

### VS Code Setup

Add to your `.vscode/settings.json`:

```json
{
  "yaml.schemas": {
    "./schema/pyms_schema.json": ["*_schema.yaml", "**/schemas/*.yaml"]
  }
}
```
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
