"""Utility package for configuration, path resolution, and reproducibility."""

import os
from pathlib import Path
from typing import Any, Dict


def get_project_root() -> Path:
    """Return the root path of the person-segmentation project."""
    current = Path(__file__).resolve().parent
    # current is src/utils -> parent is src -> parent.parent is project root
    if current.name == "utils" and current.parent.name == "src":
        return current.parent.parent
    # Fallback search if loaded differently
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "config.yaml").exists():
            return parent
        if (parent / "person-segmentation" / "configs" / "config.yaml").exists():
            return parent / "person-segmentation"
    return Path.cwd()


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Fallback simple YAML parser for key-value and nested dicts/lists without external dependencies."""
    result: Dict[str, Any] = {}
    current_section = None

    for raw_line in text.splitlines():
        line = raw_line.split("#")[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        if indent == 0 and stripped.endswith(":"):
            section_name = stripped[:-1].strip()
            current_section = section_name
            result[current_section] = {}
        elif indent > 0 and current_section is not None:
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip()

                # Parse scalar value
                if v == "":
                    val = None
                elif v.lower() == "true":
                    val = True
                elif v.lower() == "false":
                    val = False
                elif v.startswith("[") and v.endswith("]"):
                    # Parse list
                    items = [item.strip() for item in v[1:-1].split(",") if item.strip()]
                    parsed_items = []
                    for item in items:
                        try:
                            if "." in item:
                                parsed_items.append(float(item))
                            else:
                                parsed_items.append(int(item))
                        except ValueError:
                            parsed_items.append(item.strip("\"'"))
                    val = parsed_items
                else:
                    try:
                        if "." in v:
                            val = float(v)
                        else:
                            val = int(v)
                    except ValueError:
                        val = v.strip("\"'")
                result[current_section][k] = val

    return result


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    root = get_project_root()
    if config_path is None:
        target_path = root / "configs" / "config.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = root / target_path

    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        return _parse_simple_yaml(content)


def resolve_path(rel_path: str) -> Path:
    """Resolve a relative path against the project root."""
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return get_project_root() / path
