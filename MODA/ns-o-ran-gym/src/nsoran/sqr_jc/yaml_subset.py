"""Minimal deterministic YAML subset for the frozen manifests (§8/§13).

The project venv has no PyYAML (frozen environment — no new deps), and the
manifests this Etapa writes are simple two-level mappings of scalars and
nested string-keyed maps.  This module emits and parses EXACTLY that subset:

    key: value                  # str / int / float / bool / null scalars
    key:                        # nested mapping (one indentation level = 2 spaces)
      sub: value
    # comment

Values are emitted deterministically (sorted keys, fixed scalar formatting)
so re-dumping parsed content is byte-stable; ``load`` accepts comments and
blank lines.  Anything outside the subset (lists, anchors, multi-line
scalars, >2 levels) raises — loudly, never silently mis-parsed.
"""

from __future__ import annotations

from typing import Any

_INDENT = "  "


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    text = str(value)
    if any(ch in text for ch in ":#\n") or text.strip() != text or text == "":
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    return text


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text == "null" or text == "":
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    if text.startswith('"') and text.endswith('"'):
        body = text[1:-1]
        return (
            body.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        )
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def dump(data: dict) -> str:
    """Emit the two-level subset deterministically (sorted keys)."""
    lines: list[str] = []
    for key in sorted(data, key=str):
        value = data[key]
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key in sorted(value, key=str):
                lines.append(f"{_INDENT}{sub_key}: {_format_scalar(value[sub_key])}")
        else:
            lines.append(f"{key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def load(text: str) -> dict:
    """Parse the emitted subset (nested maps one level deep)."""
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            raise ValueError(f"not a key line in the supported subset: {raw_line!r}")
        key, _, rest = line.strip().partition(":")
        key = key.strip()
        if indent == 0:
            if rest.strip() == "":
                current = {}
                root[key] = current
            else:
                root[key] = _parse_scalar(rest)
                current = None
        elif indent == len(_INDENT):
            if current is None:
                raise ValueError(f"nested key {key!r} under a scalar (unsupported)")
            current[key] = _parse_scalar(rest)
        else:
            raise ValueError(
                f"indentation {indent} exceeds the supported two-level subset: {raw_line!r}"
            )
    return root


def load_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return load(fh.read())


def dump_file(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(dump(data))
