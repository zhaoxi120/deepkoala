"""Load optional, version-specific KO descriptions for CSV output."""

import csv
from pathlib import Path
import re
from typing import Iterable
import warnings


def load_label_descriptions(path: str | Path, labels: Iterable[str]) -> dict[str, str]:
    """Validate a description table and resolve model labels, including fusions.

    Missing KO definitions are reported once per load. The returned mapping can
    be applied to the final output labels, after confidence filtering.
    """
    path = Path(path)
    definitions: dict[str, str] = {}
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"KO description file not found: {path}. --description requires "
            "ko_description.csv in the selected model version directory; "
            "install the matching file or omit --description."
        ) from exc
    with handle:
        reader = csv.DictReader(handle, strict=True)
        try:
            if reader.fieldnames != ["ko", "description"]:
                raise ValueError(f"Invalid KO description file {path}: expected columns ko,description")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Invalid KO description file {path}, line {reader.line_num}: expected two fields")
                ko = row["ko"]
                definition = row["description"]
                if re.fullmatch(r"K\d{5}", ko) is None:
                    raise ValueError(f"Invalid KO identifier {ko!r} in {path}, line {reader.line_num}")
                if ko in definitions:
                    raise ValueError(f"Duplicate KO identifier {ko} in {path}, line {reader.line_num}")
                if not definition.strip():
                    raise ValueError(f"Empty description for {ko} in {path}, line {reader.line_num}")
                definitions[ko] = definition
        except (csv.Error, UnicodeError) as exc:
            raise ValueError(f"Invalid KO description file {path}, line {reader.line_num}: {exc}") from exc

    resolved: dict[str, str] = {}
    missing: set[str] = set()
    for label in labels:
        parts = label.split("+")
        missing.update(ko for ko in parts if ko not in definitions)
        if len(parts) == 1:
            resolved[label] = definitions.get(label, "")
        else:
            resolved[label] = " | ".join(
                f"{ko}: {definitions.get(ko, '[description unavailable]')}" for ko in parts
            )
    if missing:
        examples = ", ".join(sorted(missing)[:10])
        if len(missing) > 10:
            examples += f", ... ({len(missing) - 10} more)"
        warnings.warn(
            f"Missing descriptions for {len(missing)} model KO(s) in {path}: {examples}. "
            "Single-KO descriptions will be blank; missing compound components "
            "will be marked [description unavailable].",
            UserWarning,
            stacklevel=2,
        )
    return resolved
