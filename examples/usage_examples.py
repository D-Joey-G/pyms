"""
Python usage examples for pyamlvus.

This file demonstrates how to use pyamlvus to work with Milvus schemas
defined in YAML format.
"""

from pathlib import Path

from pymilvus import MilvusClient

from pyamlvus import (
    SchemaBuilder,
    SchemaLoader,
    load_schema,
    validate_schema_file,
)


def example_basic_usage():
    """Basic usage: Load and validate a schema."""
    print("=== Basic Usage Example ===")

    schema_file = Path("examples/basic_schema.yaml")

    # Validate the schema file first
    print("Validating schema...")
    errors = validate_schema_file(schema_file)
    if errors:
        print("Validation issues:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Schema is valid!")

    # Load the schema as a CollectionSchema
    print("\nLoading schema...")
    schema = load_schema(schema_file)
    print(f"Loaded schema: {schema.description}")
    print(f"Fields: {len(schema.fields)}")
    for field in schema.fields:
        print(f"  - {field.name} ({field.dtype.name})")

    print()


def example_step_by_step():
    """Step-by-step example using SchemaLoader and SchemaBuilder."""
    print("=== Step-by-Step Example ===")

    schema_file = Path("examples/complex_schema.yaml")

    # Step 1: Load YAML into dictionary
    print("Step 1: Loading YAML...")
    loader = SchemaLoader(schema_file)
    print(f"Collection name: {loader.name}")
    print(f"Description: {loader.description}")
    print(f"Fields: {len(loader.fields)}")
    print(f"Indexes: {len(loader.indexes)}")
    print(f"Functions: {len(loader.functions)}")

    # Step 2: Get schema dictionary
    print("\nStep 2: Converting to dictionary...")
    schema_dict = loader.to_dict()
    print(f"Schema dict keys: {list(schema_dict.keys())}")

    # Step 3: Build CollectionSchema
    print("\nStep 3: Building CollectionSchema...")
    builder = SchemaBuilder(schema_dict)
    schema = builder.build()
    print(f"Built schema with {len(schema.fields)} fields")

    # Optional: Get warnings about the schema
    warnings = builder.get_index_warnings()
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print()


def example_with_milvus():
    """Example showing integration with Milvus client."""
    print("=== Milvus Integration Example ===")

    schema_file = Path("examples/basic_schema.yaml")

    # Load schema
    schema = load_schema(schema_file)

    # Alternative: Use the convenience builder function
    # schema = build_collection_from_yaml(schema_file)

    print(f"Schema loaded: {schema.description}")

    # Connect to Milvus (adjust URI as needed)
    # Note: This will fail if Milvus is not running
    try:
        client = MilvusClient(uri="http://localhost:19530")

        collection_name = "pyamlvus_example"

        # Drop collection if it exists
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"Dropped existing collection: {collection_name}")

        # Create collection using the schema
        client.create_collection(collection_name=collection_name, schema=schema)
        print(f"Created collection: {collection_name}")

        # Verify collection was created
        if client.has_collection(collection_name):
            print("✓ Collection created successfully!")

            # Get collection info
            desc = client.describe_collection(collection_name)
            print(f"Collection schema has {len(desc.schema.fields)} fields")

        # Clean up
        client.drop_collection(collection_name)
        print("✓ Collection cleaned up")

    except Exception as e:
        print(f"Milvus connection failed (this is expected if not running): {e}")

    print()


def example_error_handling():
    """Example showing error handling."""
    print("=== Error Handling Example ===")

    # Try to load a non-existent file
    try:
        _schema = load_schema("non_existent.yaml")
    except Exception as e:
        print(f"Expected error for non-existent file: {type(e).__name__}: {e}")

    # Try to validate an invalid schema
    invalid_schema = {
        "name": "invalid_test",
        "fields": [
            {"name": "bad_field", "type": "unknown_type"}  # Invalid type
        ],
    }

    try:
        builder = SchemaBuilder(invalid_schema)
        _schema = builder.build()
    except Exception as e:
        print(f"Expected error for invalid schema: {type(e).__name__}: {e}")

    print()


def example_advanced_features():
    """Example showing advanced features like text matching and functions."""
    print("=== Advanced Features Example ===")

    # Load schema with text match capabilities
    text_match_file = Path("examples/text_match_example.yaml")

    if text_match_file.exists():
        print("Loading text match example...")
        schema = load_schema(text_match_file)
        print(f"Schema: {schema.description}")

        # Show fields with text match enabled
        for field in schema.fields:
            if hasattr(field, "enable_match") and field.enable_match:
                print(f"  - {field.name}: Text match enabled")

    # Load schema with functions (BM25 example)
    complex_file = Path("examples/complex_schema.yaml")

    print("\nLoading complex schema with functions...")
    loader = SchemaLoader(complex_file)

    if loader.functions:
        print(f"Functions defined: {len(loader.functions)}")
        for func in loader.functions:
            print(f"  - {func['name']}: {func['type']} function")
            print(f"    Input: {func['input_field']} → Output: {func['output_field']}")

    print()


def example_custom_schema():
    """Example creating a schema programmatically."""
    print("=== Custom Schema Example ===")

    # Define schema dictionary programmatically
    custom_schema = {
        "name": "custom_collection",
        "description": "Programmatically created schema",
        "fields": [
            {"name": "id", "type": "int64", "is_primary": True, "auto_id": True},
            {"name": "title", "type": "varchar", "max_length": 256},
            {"name": "embedding", "type": "float_vector", "dim": 512},
            {"name": "metadata", "type": "json"},
        ],
        "indexes": [
            {
                "field": "embedding",
                "type": "IVF_FLAT",
                "metric": "L2",
                "params": {"nlist": 128},
            }
        ],
        "settings": {"consistency_level": "Strong", "enable_dynamic_field": True},
    }

    # Build the schema
    builder = SchemaBuilder(custom_schema)
    schema = builder.build()

    print(f"Created custom schema: {schema.description}")
    print(f"Fields: {len(schema.fields)}")
    print(f"Enable dynamic field: {schema.enable_dynamic_field}")

    print()


def main():
    """Run all examples."""
    print("PyAMLVus Usage Examples")
    print("=" * 50)

    example_basic_usage()
    example_step_by_step()
    example_with_milvus()
    example_error_handling()
    example_advanced_features()
    example_custom_schema()

    print("All examples completed!")


if __name__ == "__main__":
    main()
