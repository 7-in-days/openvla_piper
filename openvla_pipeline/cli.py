"""Small dependency-free command-line parser for workspace entry points.

The workspace keeps durable settings in YAML. This module only handles
one-shot overrides so every entry point uses the same strict option syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable, Iterable, NoReturn, Sequence


Converter = Callable[[str], Any]


@dataclass(frozen=True)
class Option:
    name: str
    converter: Converter = str
    default: Any = None
    required: bool = False
    switch: bool = False
    boolean: bool = False
    choices: tuple[Any, ...] = ()
    help: str = ""
    metavar: str = "VALUE"

    @property
    def flag(self) -> str:
        return "--" + self.name.replace("_", "-")


def usage_error(message: str) -> NoReturn:
    print(f"error=invalid_arguments detail={message}", file=sys.stderr)
    raise SystemExit(2)


def _help_text(description: str, options: Sequence[Option], program: str) -> str:
    lines = [description, "", f"usage: {program} [options]", "", "options:"]
    for option in options:
        if option.boolean:
            spelling = f"{option.flag} | --no-{option.flag[2:]}"
        elif option.switch:
            spelling = option.flag
        else:
            spelling = f"{option.flag} {option.metavar}"
        required = " (required)" if option.required else ""
        lines.append(f"  {spelling:<38} {option.help}{required}".rstrip())
    lines.append("  -h, --help                            show this help")
    return "\n".join(lines)


def parse_options(
    argv: Sequence[str] | None,
    options: Sequence[Option],
    *,
    description: str,
    program: str | None = None,
    allow_unknown: bool = False,
) -> tuple[SimpleNamespace, list[str]]:
    """Parse long options and return values plus optionally forwarded tokens."""

    tokens = list(sys.argv[1:] if argv is None else argv)
    by_flag = {option.flag: option for option in options}
    by_negative = {
        f"--no-{option.flag[2:]}": option for option in options if option.boolean
    }
    values = {option.name: option.default for option in options}
    forwarded: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"-h", "--help"}:
            print(_help_text(description, options, program or Path(sys.argv[0]).name))
            raise SystemExit(0)
        if token == "--":
            remainder = tokens[index + 1 :]
            if allow_unknown:
                forwarded.extend(remainder)
                break
            usage_error(f"unexpected positional arguments: {remainder!r}")

        flag, separator, inline_value = token.partition("=")
        option = by_flag.get(flag)
        negative = by_negative.get(flag)
        if option is None and negative is None:
            if allow_unknown:
                forwarded.append(token)
                index += 1
                continue
            usage_error(f"unknown option {flag!r}")
        selected = option or negative
        assert selected is not None
        if selected.switch or selected.boolean:
            if separator:
                usage_error(f"{flag} does not accept a value")
            values[selected.name] = negative is None
            index += 1
            continue
        if not separator:
            index += 1
            if index >= len(tokens):
                usage_error(f"{flag} requires a value")
            inline_value = tokens[index]
        try:
            converted = selected.converter(inline_value)
        except (TypeError, ValueError) as exc:
            usage_error(f"invalid {flag} value {inline_value!r}: {exc}")
        if selected.choices and converted not in selected.choices:
            usage_error(
                f"{flag} must be one of {', '.join(map(str, selected.choices))}; "
                f"got {converted!r}"
            )
        values[selected.name] = converted
        index += 1

    for option in options:
        if option.required and values[option.name] is None:
            usage_error(f"{option.flag} is required")
    return SimpleNamespace(**values), forwarded


def selected_option(
    argv: Sequence[str] | None,
    name: str,
    converter: Converter = str,
) -> Any | None:
    """Read one option without consuming or validating the remaining command."""

    tokens = list(sys.argv[1:] if argv is None else argv)
    flag = "--" + name.replace("_", "-")
    selected: Any | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == flag:
            if index + 1 >= len(tokens):
                usage_error(f"{flag} requires a value")
            raw = tokens[index + 1]
            index += 2
        elif token.startswith(flag + "="):
            raw = token[len(flag) + 1 :]
            index += 1
        else:
            index += 1
            continue
        try:
            selected = converter(raw)
        except (TypeError, ValueError) as exc:
            usage_error(f"invalid {flag} value {raw!r}: {exc}")
    return selected


def without_switch(tokens: Iterable[str], name: str) -> tuple[bool, list[str]]:
    """Extract a switch while preserving every other token in order."""

    flag = "--" + name.replace("_", "-")
    found = False
    remaining: list[str] = []
    for token in tokens:
        if token == flag:
            found = True
        else:
            remaining.append(token)
    return found, remaining
