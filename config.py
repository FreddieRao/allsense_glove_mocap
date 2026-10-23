#!/usr/bin/env python3
"""Load conf/calibration.yaml with dotted key=value overrides.

Copied from TableMocap-toolkit-padab/kinematics-replay-bimanual/config.py (the small OmegaConf stand-in),
minus that project's landmark cross-check and second config file. Same ergonomics:

    python -m calibration.fit_hand fit.iters_pose=500 mosca.pose.rmse_max_mm=3

    from config import load_config
    cfg = load_config()                     # conf/calibration.yaml
    cfg.mosca.pose.rmse_max_mm              # attribute access; an unknown key raises
    cfg.get("fit.weights.tip_cap")          # or dotted lookup
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

CONF_DIR = Path(__file__).resolve().parent / "conf"
DEFAULT_CONFIG = CONF_DIR / "calibration.yaml"   # data paths, MoSca, MANO fit


class Node:
    """Read-mostly attribute/dotted view over nested dicts. Unknown keys raise, never return None --
    a typo in a config path must fail loudly rather than silently disable a gate."""

    def __init__(self, data: dict, prefix: str = ""):
        self._data = data
        self._prefix = prefix

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            v = self._data[name]
        except KeyError:
            where = self._prefix or "<root>"
            raise AttributeError(
                f"no config key {name!r} under {where}; available: {sorted(self._data)}") from None
        return Node(v, f"{self._prefix}.{name}".lstrip(".")) if isinstance(v, dict) else v

    def get(self, dotted: str, default: Any = "__raise__") -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if isinstance(cur, Node):
                try:
                    cur = getattr(cur, part)
                except AttributeError:
                    if default == "__raise__":
                        raise
                    return default
            else:
                if default == "__raise__":
                    raise AttributeError(f"{dotted!r}: {part!r} is not a group")
                return default
        return cur

    def to_dict(self) -> dict:
        return self._data

    def frozen(self) -> SimpleNamespace:
        """A plain-attribute copy for hot loops. Every `Node` lookup builds a child and wraps the KeyError
        (~10 us); a stage reading a tolerance per candidate pays that ~300 000 times a take. Unknown keys
        still raise AttributeError."""
        return SimpleNamespace(**{k: Node(v).frozen() if isinstance(v, dict) else v for k, v in self._data.items()})

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def __repr__(self) -> str:
        return f"Node({self._prefix or 'root'}: {sorted(self._data)})"


def with_overrides(cfg: Node, overrides: dict) -> Node:
    """A copy of `cfg` with dotted `{key: value}` overrides applied (existing keys only, as on the CLI)."""
    import copy
    data = copy.deepcopy(cfg.to_dict())
    for k, v in overrides.items():
        _apply_override(data, k, v)
    return Node(data)


def _coerce(text: str) -> Any:
    """Parse an override value with YAML rules, so `true`, `3`, `1e-3`, `[1,2]`, `null` all work."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _apply_override(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = data
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            raise KeyError(f"override {dotted!r}: {p!r} is not an existing group")
        cur = cur[p]
    if parts[-1] not in cur:
        # Refuse to invent keys: a typo would otherwise be accepted and do nothing.
        raise KeyError(f"override {dotted!r}: {parts[-1]!r} is not an existing key "
                       f"(have {sorted(cur)})")
    cur[parts[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None,
                argv: list[str] | None = None) -> Node:
    """Read the YAML and apply `key.sub=value` overrides.

    `argv` (default sys.argv[1:]) is scanned for bare `k=v` tokens so any CLI gets overrides for
    free; argparse flags (leading '-') are left alone.
    """
    data = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    items = list(overrides or [])
    for tok in (sys.argv[1:] if argv is None else argv):
        if not tok.startswith("-") and "=" in tok:
            items.append(tok)
    for it in items:
        k, _, v = it.partition("=")
        _apply_override(data, k.strip(), _coerce(v.strip()))
    return Node(data)


def strip_overrides(argv: list[str] | None = None) -> list[str]:
    """argv with the bare `k=v` override tokens removed, so argparse does not choke on them."""
    return [t for t in (sys.argv[1:] if argv is None else argv)
            if t.startswith("-") or "=" not in t]


if __name__ == "__main__":
    cfg = load_config()
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False, width=100))
