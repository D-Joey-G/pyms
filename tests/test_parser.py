# Tests for YAML parser module

import pytest

from pyamlvus.exceptions import SchemaParseError
from pyamlvus.parser import SchemaLoader


@pytest.mark.unit
class TestSchemaLoader:
    def test_valid_schema_loading(self, valid_yaml_content, create_temp_yaml):
        """Test loading a valid schema file."""
        yaml_file = create_temp_yaml(valid_yaml_content)
        loader = SchemaLoader(yaml_file)
        schema = loader.load()

        assert schema["name"] == "fixture_test_collection"
        assert schema["description"] == "Test collection from fixture"
        assert len(schema["fields"]) == 3

        # Check first field
        id_field = schema["fields"][0]
        assert id_field["name"] == "id"
        assert id_field["type"] == "int64"
        assert id_field["is_primary"] is True
        assert id_field["auto_id"] is False

    def test_schema_properties(self, valid_yaml_content, create_temp_yaml):
        """Test schema property accessors."""
        yaml_file = create_temp_yaml(valid_yaml_content)
        loader = SchemaLoader(yaml_file)

        assert loader.name == "fixture_test_collection"
        assert loader.description == "Test collection from fixture"
        assert loader.alias == ""  # No alias in basic fixture
        assert len(loader.fields) == 3
        assert len(loader.indexes) == 0  # No indexes in basic schema
        assert len(loader.settings) == 1  # Has consistency_level setting

    def test_to_dict_alias(self, valid_yaml_content, create_temp_yaml):
        """Test that to_dict() is an alias for load()."""
        yaml_file = create_temp_yaml(valid_yaml_content)
        loader = SchemaLoader(yaml_file)

        schema1 = loader.load()
        schema2 = loader.to_dict()

        assert schema1 == schema2
        assert schema1 is schema2  # Should be cached

    def test_caching(self, valid_yaml_content, create_temp_yaml):
        """Test that schema is cached after first load."""
        yaml_file = create_temp_yaml(valid_yaml_content)
        loader = SchemaLoader(yaml_file)

        schema1 = loader.load()
        schema2 = loader.load()

        assert schema1 is schema2  # Should be the same object

    @pytest.mark.error_cases
    def test_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        with pytest.raises(SchemaParseError, match="Schema file not found"):
            SchemaLoader("nonexistent.yaml")

    @pytest.mark.error_cases
    def test_directory_instead_of_file(self, tmp_path):
        """Test error handling when path is a directory."""
        with pytest.raises(SchemaParseError, match="Path is not a file"):
            SchemaLoader(tmp_path)

    @pytest.mark.error_cases
    def test_unreadable_file(self, tmp_path):
        """Test error handling for unreadable files."""

        # Create a temporary file and make it unreadable
        yaml_file = tmp_path / "unreadable.yaml"
        yaml_file.write_text("name: test\nfields: []")
        yaml_file.chmod(0o000)

        try:
            with pytest.raises(SchemaParseError, match="Schema file not readable"):
                SchemaLoader(yaml_file)
        finally:
            # Clean up for proper deletion
            yaml_file.chmod(0o644)

    @pytest.mark.error_cases
    def test_invalid_yaml(self, yaml_parse_error_content, create_temp_yaml):
        """Test error handling for invalid YAML."""
        yaml_file = create_temp_yaml(yaml_parse_error_content)
        loader = SchemaLoader(yaml_file)
        with pytest.raises(SchemaParseError, match="YAML parsing failed"):
            loader.load()

    @pytest.mark.error_cases
    def test_empty_file(self, create_temp_yaml):
        """Test error handling for empty files."""
        yaml_file = create_temp_yaml("")
        loader = SchemaLoader(yaml_file)
        with pytest.raises(SchemaParseError, match="Empty schema file"):
            loader.load()

    @pytest.mark.error_cases
    def test_non_dict_yaml(self, create_temp_yaml):
        """Test error handling when YAML doesn't contain a dictionary."""
        yaml_file = create_temp_yaml("- item1\n- item2\n")
        loader = SchemaLoader(yaml_file)
        with pytest.raises(SchemaParseError, match="Schema must be a dictionary"):
            loader.load()

    @pytest.mark.error_cases
    def test_missing_name(self, create_temp_yaml):
        """Test error handling for missing name field."""
        yaml_file = create_temp_yaml("description: test\nfields: []")
        loader = SchemaLoader(yaml_file)
        with pytest.raises(
            SchemaParseError, match="Schema missing required 'name' field"
        ):
            _ = loader.name

    @pytest.mark.error_cases
    def test_missing_fields(self, create_temp_yaml):
        """Test error handling for missing fields."""
        yaml_file = create_temp_yaml("name: test\ndescription: test")
        loader = SchemaLoader(yaml_file)
        with pytest.raises(
            SchemaParseError, match="Schema missing required 'fields' field"
        ):
            _ = loader.fields

    @pytest.mark.error_cases
    def test_invalid_description_type(self, create_temp_yaml):
        """Test error handling for non-string description values."""
        yaml_file = create_temp_yaml("name: test\ndescription: {}\nfields: []")
        loader = SchemaLoader(yaml_file)

        with pytest.raises(
            SchemaParseError, match="Collection description must be a string"
        ):
            _ = loader.description

    @pytest.mark.error_cases
    def test_invalid_fields_type(self, create_temp_yaml):
        """Test error handling for invalid fields type."""
        yaml_file = create_temp_yaml("name: test\nfields: 'not a list'")
        loader = SchemaLoader(yaml_file)
        with pytest.raises(SchemaParseError, match="Fields must be a list"):
            _ = loader.fields

    @pytest.mark.error_cases
    def test_empty_fields(self, create_temp_yaml):
        """Test error handling for empty fields list."""
        yaml_file = create_temp_yaml("name: test\nfields: []")
        loader = SchemaLoader(yaml_file)
        with pytest.raises(
            SchemaParseError, match="Schema must have at least one field"
        ):
            _ = loader.fields

    def test_pathlib_path(self, valid_yaml_content, create_temp_yaml):
        """Test that Path objects work as input."""
        yaml_file = create_temp_yaml(valid_yaml_content)
        loader = SchemaLoader(yaml_file)

        schema = loader.load()
        assert schema["name"] == "fixture_test_collection"

    def test_alias_property(self, create_temp_yaml):
        """Test alias property access."""
        yaml_content = """
name: test_collection
alias: my_alias
description: Test collection
fields:
  - name: id
    type: int64
    is_primary: true
"""
        yaml_file = create_temp_yaml(yaml_content)
        loader = SchemaLoader(yaml_file)

        assert loader.alias == "my_alias"

    def test_alias_empty_when_not_specified(self, create_temp_yaml):
        """Test that alias returns empty string when not specified."""
        yaml_content = """
name: test_collection
description: Test collection
fields:
  - name: id
    type: int64
    is_primary: true
"""
        yaml_file = create_temp_yaml(yaml_content)
        loader = SchemaLoader(yaml_file)

        assert loader.alias == ""

    @pytest.mark.error_cases
    def test_invalid_alias_type(self, create_temp_yaml):
        """Test error handling for invalid alias type."""
        yaml_content = """
name: test_collection
alias: 123
fields:
  - name: id
    type: int64
    is_primary: true
"""
        yaml_file = create_temp_yaml(yaml_content)
        loader = SchemaLoader(yaml_file)

        with pytest.raises(SchemaParseError, match="Collection alias must be a string"):
            _ = loader.alias

    @pytest.mark.error_cases
    def test_empty_alias(self, create_temp_yaml):
        """Test error handling for empty alias."""
        yaml_content = """
name: test_collection
alias: ""
fields:
  - name: id
    type: int64
    is_primary: true
"""
        yaml_file = create_temp_yaml(yaml_content)
        loader = SchemaLoader(yaml_file)

        with pytest.raises(SchemaParseError, match="Collection alias cannot be empty"):
            _ = loader.alias

    @pytest.mark.error_cases
    def test_invalid_alias_format(self, create_temp_yaml):
        """Test error handling for invalid alias format."""
        yaml_content = """
name: test_collection
alias: "123invalid"
fields:
  - name: id
    type: int64
    is_primary: true
"""
        yaml_file = create_temp_yaml(yaml_content)
        loader = SchemaLoader(yaml_file)

        with pytest.raises(
            SchemaParseError, match="Collection alias.*must start with a letter"
        ):
            _ = loader.alias

    @pytest.mark.error_cases
    def test_alias_starting_with_underscore(self, create_temp_yaml):
        """Test error handling for alias starting with underscore."""
        yaml_content = """
name: test_collection
alias: "_invalid_alias"
fields:
  - name: id
    type: int64
    is_primary: true
"""
        yaml_file = create_temp_yaml(yaml_content)
        loader = SchemaLoader(yaml_file)

        with pytest.raises(
            SchemaParseError, match="Collection alias.*must start with a letter"
        ):
            _ = loader.alias


@pytest.mark.unit
class TestSchemaLoaderParametrized:
    """Parametrized tests for SchemaLoader with various scenarios."""

    @pytest.mark.parametrize(
        "schema_dict,expected_name,expected_field_count",
        [
            (
                {
                    "name": "test1",
                    "fields": [{"name": "id", "type": "int64", "is_primary": True}],
                },
                "test1",
                1,
            ),
            (
                {
                    "name": "test2",
                    "description": "desc",
                    "fields": [
                        {"name": "id", "type": "int64", "is_primary": True},
                        {"name": "text", "type": "varchar", "max_length": 100},
                    ],
                },
                "test2",
                2,
            ),
            (
                {
                    "name": "test3",
                    "fields": [{"name": "id", "type": "int64", "is_primary": True}],
                    "settings": {"enable_dynamic_field": True},
                },
                "test3",
                1,
            ),
        ],
    )
    def test_various_valid_schemas(
        self, schema_dict, expected_name, expected_field_count, create_temp_yaml
    ):
        """Test loading various valid schema configurations."""
        import yaml

        yaml_content = yaml.dump(schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        loader = SchemaLoader(yaml_file)
        schema = loader.load()

        assert schema["name"] == expected_name
        assert len(schema["fields"]) == expected_field_count

    @pytest.mark.parametrize(
        "field_type,extra_params",
        [
            ("int8", {}),
            ("int16", {}),
            ("int32", {}),
            ("int64", {}),
            ("float", {}),
            ("double", {}),
            ("json", {}),
            ("varchar", {"max_length": 256}),
            ("float_vector", {"dim": 128}),
            ("binary_vector", {"dim": 64}),
            ("sparse_float_vector", {}),
            ("array", {"element_type": "int32", "max_capacity": 100}),
        ],
    )
    def test_field_types_parsing(self, field_type, extra_params, create_temp_yaml):
        """Test that all field types are correctly parsed."""
        import yaml

        field_def = {"name": "test_field", "type": field_type}
        field_def.update(extra_params)

        schema_dict = {
            "name": "type_test",
            "fields": [{"name": "id", "type": "int64", "is_primary": True}, field_def],
        }

        yaml_content = yaml.dump(schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        loader = SchemaLoader(yaml_file)
        schema = loader.load()

        # Find the test field
        test_field = next(f for f in schema["fields"] if f["name"] == "test_field")
        assert test_field["type"] == field_type

        # Check extra parameters
        for key, value in extra_params.items():
            assert test_field[key] == value

    @pytest.mark.error_cases
    @pytest.mark.parametrize(
        "invalid_content,expected_error",
        [
            ("", "Empty schema file"),
            ("- not a dict", "Schema must be a dictionary"),
            (
                "description: no name\nfields: []",
                "Schema missing required 'name' field",
            ),
            (
                "name: test\ndescription: no fields",
                "Schema missing required 'fields' field",
            ),
            ("name: test\nfields: 'not a list'", "Fields must be a list"),
            ("name: test\nfields: []", "Schema must have at least one field"),
        ],
    )
    def test_invalid_schema_content(
        self, invalid_content, expected_error, create_temp_yaml
    ):
        """Test various invalid schema content scenarios."""
        yaml_file = create_temp_yaml(invalid_content)
        loader = SchemaLoader(yaml_file)

        with pytest.raises(SchemaParseError, match=expected_error):
            if "missing required 'name'" in expected_error:
                _ = loader.name
            elif "missing required 'fields'" in expected_error:
                _ = loader.fields
            elif "must be a list" in expected_error:
                _ = loader.fields
            elif "at least one" in expected_error:
                _ = loader.fields
            else:
                loader.load()

    @pytest.mark.parametrize(
        "indexes_content,expected_count",
        [
            ([], 0),
            ([{"field": "vector", "type": "IVF_FLAT", "metric": "L2"}], 1),
            (
                [
                    {"field": "vector1", "type": "HNSW"},
                    {"field": "vector2", "type": "IVF_FLAT"},
                ],
                2,
            ),
        ],
    )
    def test_indexes_parsing(self, indexes_content, expected_count, create_temp_yaml):
        """Test parsing of index definitions."""
        import yaml

        schema_dict = {
            "name": "index_test",
            "fields": [{"name": "id", "type": "int64", "is_primary": True}],
            "indexes": indexes_content,
        }

        yaml_content = yaml.dump(schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        loader = SchemaLoader(yaml_file)
        assert len(loader.indexes) == expected_count

    @pytest.mark.parametrize(
        "settings_content,expected_keys",
        [
            ({}, []),
            ({"enable_dynamic_field": True}, ["enable_dynamic_field"]),
            (
                {"consistency_level": "Strong", "ttl_seconds": 3600},
                ["consistency_level", "ttl_seconds"],
            ),
        ],
    )
    def test_settings_parsing(self, settings_content, expected_keys, create_temp_yaml):
        """Test parsing of collection settings."""
        import yaml

        schema_dict = {
            "name": "settings_test",
            "fields": [{"name": "id", "type": "int64", "is_primary": True}],
            "settings": settings_content,
        }

        yaml_content = yaml.dump(schema_dict)
        yaml_file = create_temp_yaml(yaml_content)

        loader = SchemaLoader(yaml_file)
        settings = loader.settings

        assert set(settings.keys()) == set(expected_keys)
        for key in expected_keys:
            assert settings[key] == settings_content[key]
