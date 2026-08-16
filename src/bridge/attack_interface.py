"""Bridge to a trained dlsca-attack-v2 attacker.

Loads a frozen attacker model plus the exact per-point preprocessing
statistics (Standardizer / MinMaxScaler) it was evaluated with, by reading
the same `config_snapshot.yaml` + `split_indices.npz` dlsca-attack-v2's own
`scripts/02_run_attack.py` reads, and refitting those statistics on the same
A-set indices. This mirrors that script's "refit is deterministic given A, so
nothing needs to be persisted separately" approach rather than re-implementing
attacker internals — see CLAUDE.md §2/§6.

`AttackerBridge.preprocess_tf` re-expresses that transform as pure TF
elementwise ops over constants (the fitted mean/std/min/max), so it is
differentiable with respect to an upstream perturbation and can sit inside an
adversarial training loop's `tf.GradientTape`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import keras
import numpy as np
import tensorflow as tf
import yaml


@dataclass
class AttackerBridge:
    model: keras.Model
    mean: np.ndarray | None  # (trace_len,) float64, None if preprocess.method == "none"
    std: np.ndarray | None
    minmax_min: np.ndarray | None  # None if preprocess.minmax is not enabled
    minmax_max: np.ndarray | None
    feature_range: tuple[float, float]
    trace_len: int
    n_classes: int

    def preprocess_tf(self, x: tf.Tensor) -> tf.Tensor:
        """x: (batch, trace_len, 1) raw-scale float32. Returns the attacker's
        native input scale. Differentiable w.r.t. x."""
        if self.mean is None:
            return x
        mean = tf.constant(self.mean.reshape(1, -1, 1), dtype=tf.float32)
        std_safe = np.where(self.std == 0, 1.0, self.std)
        std = tf.constant(std_safe.reshape(1, -1, 1), dtype=tf.float32)
        x = (x - mean) / std
        if self.minmax_min is not None:
            span = np.where(self.minmax_max == self.minmax_min, 1.0, self.minmax_max - self.minmax_min)
            lo, hi = self.feature_range
            mn = tf.constant(self.minmax_min.reshape(1, -1, 1), dtype=tf.float32)
            sp = tf.constant(span.reshape(1, -1, 1), dtype=tf.float32)
            unit = (x - mn) / sp
            x = unit * (hi - lo) + lo
        return x

    def predict_probs(self, x_raw: tf.Tensor, training: bool = False) -> tf.Tensor:
        """x_raw: (batch, trace_len, 1) raw-scale (not yet standardized/minmax'd)."""
        x = self.preprocess_tf(x_raw)
        return self.model(x, training=training)


def load(attacker_run_dir: str | Path) -> AttackerBridge:
    run_dir = Path(attacker_run_dir)
    with open(run_dir / "config_snapshot.yaml") as f:
        cfg = yaml.safe_load(f)

    if cfg.get("preprocess", {}).get("resync", {}).get("enabled"):
        raise NotImplementedError(
            f"attacker at {run_dir} was trained with preprocess.resync.enabled=true; "
            "the defense bridge does not replicate resync alignment yet"
        )

    # data.path in config_snapshot.yaml is relative to the dlsca-attack-v2
    # repo root (<attack_repo>/runs/{run}/config_snapshot.yaml), not this
    # repo's cwd — resolve it the same way dlsca-attack-v2's own scripts do.
    data_path = Path(cfg["data"]["path"])
    if not data_path.is_absolute():
        data_path = run_dir.parent.parent / data_path

    with h5py.File(data_path, "r") as f:
        profiling_traces = np.array(f["Profiling_traces/traces"], dtype=np.int8)

    with np.load(run_dir / "split_indices.npz") as split:
        a_idx = split["a"]
    traces_a = profiling_traces[a_idx].astype(np.float64)

    method = cfg.get("preprocess", {}).get("method", "standardize_per_point")
    mean = std = minmax_min = minmax_max = None
    if method == "standardize_per_point":
        mean = traces_a.mean(axis=0)
        std = traces_a.std(axis=0)
        if cfg.get("preprocess", {}).get("minmax"):
            std_safe = np.where(std == 0, 1.0, std)
            x_a = (traces_a - mean) / std_safe
            minmax_min = x_a.min(axis=0)
            minmax_max = x_a.max(axis=0)
    elif method != "none":
        raise ValueError(f"unknown preprocess.method: {method!r}")

    model = keras.models.load_model(run_dir / "model.keras")
    model.trainable = False

    return AttackerBridge(
        model=model,
        mean=mean,
        std=std,
        minmax_min=minmax_min,
        minmax_max=minmax_max,
        feature_range=(0.0, 1.0),
        trace_len=int(cfg["data"]["trace_len"]),
        n_classes=int(cfg["leakage"]["n_classes"]),
    )
