from __future__ import annotations

from unittest.mock import Mock

import pytest
import yaml

from pymilvus import CollectionSchema

from pyamlvus.api import (
    build_collection_from_dict,
    build_collection_from_yaml,
    create_collection_from_dict,
    create_collection_from_yaml,
)
from pyamlvus.exceptions import SchemaConversionError


@pytest.mark.unit
class TestBuildHelpers:
    def test_build_collection_from_yaml_alias(
        self, create_temp_yaml, valid_schema_dict
    ):
        yaml_file = create_temp_yaml(yaml.dump(valid_schema_dict))

        schema = build_collection_from_yaml(yaml_file)

        assert isinstance(schema, CollectionSchema)

    def test_build_collection_from_dict_alias(self, valid_schema_dict):
        schema = build_collection_from_dict(valid_schema_dict)

        assert isinstance(schema, CollectionSchema)


@pytest.mark.unit
class TestCreateCollectionApis:
    def test_create_collection_from_dict_invokes_client(self, valid_schema_dict):
        mock_client = Mock()
        mock_client.create_collection.return_value = "created"

        result = create_collection_from_dict(valid_schema_dict, client=mock_client)

        assert result == "created"
        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == valid_schema_dict["name"]
        assert isinstance(call_kwargs["schema"], CollectionSchema)

    def test_create_collection_from_dict_missing_name(self, minimal_schema_dict):
        schema_dict = dict(minimal_schema_dict)
        schema_dict.pop("name")
        mock_client = Mock()

        with pytest.raises(SchemaConversionError, match="must include a string 'name'"):
            create_collection_from_dict(schema_dict, client=mock_client)

        mock_client.create_collection.assert_not_called()

    def test_create_collection_from_yaml_invokes_client(
        self, create_temp_yaml, valid_schema_dict
    ):
        yaml_file = create_temp_yaml(yaml.dump(valid_schema_dict))
        mock_client = Mock()
        mock_client.create_collection.return_value = {"status": "ok"}

        result = create_collection_from_yaml(yaml_file, client=mock_client)

        assert result == {"status": "ok"}
        mock_client.create_collection.assert_called_once()
        call_kwargs = mock_client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == valid_schema_dict["name"]
        assert isinstance(call_kwargs["schema"], CollectionSchema)
