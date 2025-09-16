"""Validation result structures for schema checks."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import overload


class ValidationSeverity(StrEnum):
    """Severity levels for validation messages."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(slots=True)
class ValidationMessage:
    """Single validation message with severity metadata."""

    severity: ValidationSeverity
    text: str

    def as_prefixed(self) -> str:
        """Return the message text prefixed with its severity label."""

        return (
            f"{self.severity.value}: {self.text}" if self.text else self.severity.value
        )


@dataclass
class ValidationResult(Sequence[str]):
    """Collection of validation messages grouped by severity."""

    messages: list[ValidationMessage] = field(default_factory=list)

    def add(self, severity: ValidationSeverity, text: str) -> None:
        self.messages.append(ValidationMessage(severity, text))

    def add_error(self, text: str) -> None:
        self.add(ValidationSeverity.ERROR, text)

    def add_warning(self, text: str) -> None:
        self.add(ValidationSeverity.WARNING, text)

    def add_info(self, text: str) -> None:
        self.add(ValidationSeverity.INFO, text)

    def extend(self, other: ValidationResult) -> None:
        self.messages.extend(other.messages)

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[str]:
        for message in self.messages:
            yield message.as_prefixed()

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [msg.as_prefixed() for msg in self.messages[index]]
        return self.messages[index].as_prefixed()

    def __bool__(self) -> bool:  # pragma: no cover - delegation
        return bool(self.messages)

    def __eq__(self, other: object) -> bool:  # pragma: no cover - delegation
        if isinstance(other, ValidationResult):
            return self.as_strings() == other.as_strings()
        if isinstance(other, Sequence):
            return self.as_strings() == list(other)
        return NotImplemented

    @property
    def errors(self) -> list[ValidationMessage]:
        return [
            msg for msg in self.messages if msg.severity is ValidationSeverity.ERROR
        ]

    @property
    def warnings(self) -> list[ValidationMessage]:
        return [
            msg for msg in self.messages if msg.severity is ValidationSeverity.WARNING
        ]

    @property
    def infos(self) -> list[ValidationMessage]:
        return [msg for msg in self.messages if msg.severity is ValidationSeverity.INFO]

    def has_errors(self) -> bool:
        return bool(self.errors)

    def as_strings(self) -> list[str]:
        """Return messages as prefixed strings for backward compatibility."""

        return [msg.as_prefixed() for msg in self.messages]
