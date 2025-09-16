import textwrap

import pytest

from typer.testing import CliRunner

from pyamlvus.cli import _split_messages, app


@pytest.mark.unit
def test_split_messages_groups_by_prefix():
    messages = [
        "WARNING: first warning",
        "INFO: heads up",
        "ERROR: something failed",
        "Trailing spaces   ",
    ]

    errors, warnings, infos = _split_messages(messages)

    assert errors == ["ERROR: something failed", "Trailing spaces"]
    assert warnings == ["WARNING: first warning"]
    assert infos == ["INFO: heads up"]


@pytest.mark.unit
def test_validate_command_handles_info_messages(tmp_path):
    runner = CliRunner()

    schema_yaml = textwrap.dedent(
        """
        name: cli_info_test
        description: CLI validation info test
        fields:
          - name: id
            type: int64
            is_primary: true
          - name: title
            type: varchar
            max_length: 128
        indexes:
          - field: title
            type: TRIE
        """
    )

    schema_file = tmp_path / "schema.yaml"
    schema_file.write_text(schema_yaml)

    result = runner.invoke(app, ["validate", str(schema_file)])

    assert result.exit_code == 0
    assert "INFO:" in result.stdout
    assert "✗ Schema has" not in result.stdout
