"""
Command-line interface for pyamlvus.

Provides tools for validating and working with Milvus YAML schema files.
"""

from pathlib import Path

import typer

from rich.console import Console
from rich.table import Table

from pyamlvus import SchemaLoader, validate_schema_file
from pyamlvus.exceptions import SchemaParseError


def _split_messages(messages: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Group validation messages by severity based on a simple prefix convention."""

    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []

    for message in messages:
        normalized = message.strip()
        prefix = normalized.upper()

        if prefix.startswith("WARNING"):
            warnings.append(normalized)
        elif prefix.startswith("INFO"):
            infos.append(normalized)
        elif prefix.startswith("ERROR"):
            errors.append(normalized)
        else:
            errors.append(normalized)

    return errors, warnings, infos


app = typer.Typer(
    name="pyamlvus",
    help="CLI tools for Milvus YAML schema management",
    add_completion=False,
)

console = Console()


@app.command()
def validate(
    schema_file: Path = typer.Argument(..., help="Path to YAML schema file"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output"),
) -> None:
    """Validate a Milvus YAML schema file."""
    if not schema_file.exists():
        console.print(f"[red]Error: File '{schema_file}' does not exist[/red]")
        raise typer.Exit(1)

    try:
        console.print(f"Validating [blue]{schema_file}[/blue]...")

        messages = validate_schema_file(schema_file)

        if not messages:
            console.print("[green]✓ Schema is valid![/green]")

            if verbose:
                # Show schema info
                loader = SchemaLoader(schema_file)
                console.print("\n[bold]Schema Info:[/bold]")
                console.print(f"  Name: {loader.name}")
                console.print(f"  Description: {loader.description}")
                console.print(f"  Fields: {len(loader.fields)}")
                console.print(f"  Indexes: {len(loader.indexes)}")
                console.print(f"  Functions: {len(loader.functions)}")

                if loader.settings:
                    console.print(f"  Settings: {list(loader.settings.keys())}")

        else:
            actual_errors, warnings, infos = _split_messages(messages)

            if actual_errors:
                console.print(f"[red]✗ Schema has {len(actual_errors)} error(s):[/red]")
                for error in actual_errors:
                    console.print(f"  [red]• {error}[/red]")

            if warnings:
                console.print(
                    f"[yellow]⚠ Schema has {len(warnings)} warning(s):[/yellow]"
                )
                for warning in warnings:
                    console.print(f"  [yellow]• {warning}[/yellow]")

            if infos:
                console.print(
                    f"[cyan]i Schema has {len(infos)} info message(s):[/cyan]"
                )
                for info in infos:
                    console.print(f"  [cyan]• {info}[/cyan]")

            if actual_errors:
                raise typer.Exit(1)

    except SchemaParseError as e:
        console.print(f"[red]Parse Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    schema_file: Path = typer.Argument(..., help="Path to YAML schema file"),
) -> None:
    """Show detailed information about a schema file."""
    if not schema_file.exists():
        console.print(f"[red]Error: File '{schema_file}' does not exist[/red]")
        raise typer.Exit(1)

    try:
        loader = SchemaLoader(schema_file)

        console.print("\n[bold blue]Schema Information[/bold blue]")
        console.print(f"[bold]File:[/bold] {schema_file}")
        console.print(f"[bold]Name:[/bold] {loader.name}")
        console.print(f"[bold]Description:[/bold] {loader.description or 'None'}")

        if loader.alias:
            console.print(f"[bold]Alias:[/bold] {loader.alias}")

        # Fields table
        if loader.fields:
            console.print(f"\n[bold]Fields ({len(loader.fields)}):[/bold]")

            table = Table(show_header=True)
            table.add_column("Name", style="cyan")
            table.add_column("Type", style="magenta")
            table.add_column("Primary", style="green")
            table.add_column("Auto ID", style="blue")
            table.add_column("Parameters")

            for field in loader.fields:
                params = []
                if field.get("max_length"):
                    params.append(f"max_length={field['max_length']}")
                if field.get("dim"):
                    params.append(f"dim={field['dim']}")
                if field.get("element_type"):
                    params.append(f"element_type={field['element_type']}")
                if field.get("max_capacity"):
                    params.append(f"max_capacity={field['max_capacity']}")

                table.add_row(
                    field["name"],
                    field["type"],
                    "✓" if field.get("is_primary") else "",
                    "✓" if field.get("auto_id") else "",
                    ", ".join(params),
                )

            console.print(table)

        # Indexes
        if loader.indexes:
            console.print(f"\n[bold]Indexes ({len(loader.indexes)}):[/bold]")

            index_table = Table(show_header=True)
            index_table.add_column("Field", style="cyan")
            index_table.add_column("Type", style="magenta")
            index_table.add_column("Metric", style="green")
            index_table.add_column("Parameters")

            for index in loader.indexes:
                params = []
                if index.get("params"):
                    for key, value in index["params"].items():
                        params.append(f"{key}={value}")

                index_table.add_row(
                    index["field"],
                    index["type"],
                    index.get("metric", ""),
                    ", ".join(params),
                )

            console.print(index_table)

        # Functions
        if loader.functions:
            console.print(f"\n[bold]Functions ({len(loader.functions)}):[/bold]")

            func_table = Table(show_header=True)
            func_table.add_column("Name", style="cyan")
            func_table.add_column("Type", style="magenta")
            func_table.add_column("Input Field", style="blue")
            func_table.add_column("Output Field", style="green")

            for func in loader.functions:
                func_table.add_row(
                    func["name"],
                    func["type"],
                    func["input_field"],
                    func["output_field"],
                )

            console.print(func_table)

        # Settings
        if loader.settings:
            console.print("\n[bold]Settings:[/bold]")
            for key, value in loader.settings.items():
                console.print(f"  {key}: {value}")

    except SchemaParseError as e:
        console.print(f"[red]Parse Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def convert(
    schema_file: Path = typer.Argument(..., help="Path to YAML schema file"),
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Output file (defaults to stdout)"
    ),
) -> None:
    """Convert YAML schema to CollectionSchema representation (future feature)."""
    console.print("[yellow]Convert command is not yet implemented[/yellow]")
    console.print("This feature will allow conversion to various formats:")
    console.print("  - Python CollectionSchema code")
    console.print("  - JSON representation")
    console.print("  - Documentation format")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
