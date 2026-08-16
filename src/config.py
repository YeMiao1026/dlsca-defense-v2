"""YAML config loading, merge, validation, snapshot.

Copied (not imported — see CLAUDE.md §2) from dlsca-attack-v2's `src/config.py`
and adapted to this repo's layout:

    configs/base.yaml
    configs/attacker/{name}.yaml
    configs/exp/{exp_id}.yaml

An exp config's top-level `attacker` key is either a string naming a file
under configs/attacker/ (resolved and merged in) or an inline dict — same
resolution rule dlsca-attack-v2 uses for its `data`/`model` keys.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = (
    "exp_id",
    "seed",
    "attacker.run",
    "data.n_train",
    "generator.epsilon",
    "train.epochs",
    "train.batch_size",
    "train.lr",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path) as f:
        loaded = yaml.safe_load(f)
    return loaded or {}


def _resolve_named(dir_path: Path, name: str) -> Path:
    return dir_path / name if name.endswith((".yaml", ".yml")) else dir_path / f"{name}.yaml"


def _resolve_layer(configs_root: Path, subdir: str, ref: Any) -> dict[str, Any]:
    """`ref` is either a string (load configs/{subdir}/{ref}.yaml) or an inline dict."""
    if isinstance(ref, str):
        return _load_yaml(_resolve_named(configs_root / subdir, ref))
    if isinstance(ref, dict):
        return {subdir: ref}
    raise TypeError(f"'{subdir}' must be a string reference or an inline dict, got {type(ref).__name__}")


def load_config(exp_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Merge base.yaml -> attacker/*.yaml -> exp/*.yaml -> CLI overrides. Later sources win."""
    exp_path = Path(exp_path)
    configs_root = exp_path.parent.parent  # configs/exp/{exp_id}.yaml -> configs/

    base_cfg = _load_yaml(configs_root / "base.yaml")
    exp_cfg = _load_yaml(exp_path)

    attacker_ref = exp_cfg.pop("attacker", None)

    layers = [base_cfg]
    if attacker_ref is not None:
        layers.append(_resolve_layer(configs_root, "attacker", attacker_ref))
    layers.append(exp_cfg)

    cfg = merge(*layers)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    validate(cfg)
    return cfg


def merge(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge configs left to right; later dicts override earlier ones."""
    result: dict[str, Any] = {}
    for cfg in configs:
        _deep_merge_into(result, cfg)
    return result


def _deep_merge_into(base: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_into(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply "dotted.key=value" CLI overrides on top of a merged config."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key_path, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"override must be in dotted.key=value form, got: {item!r}")
        value = yaml.safe_load(raw_value)
        keys = key_path.split(".")
        node = cfg
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return cfg


def _get_path(cfg: dict[str, Any], dotted_key: str) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def validate(cfg: dict[str, Any]) -> None:
    """Raise on missing required keys before training starts."""
    errors = [f"missing required key: {key}" for key in REQUIRED_KEYS if _get_path(cfg, key) is None]
    if errors:
        raise ValueError("invalid config:\n  " + "\n  ".join(errors))


def snapshot(cfg: dict[str, Any], run_dir: str | Path) -> None:
    """Write the fully-expanded config to `{run_dir}/config_snapshot.yaml`."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
