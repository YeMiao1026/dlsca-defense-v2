"""Verifies AttackerBridge.load() reproduces dlsca-attack-v2's exact
Standardizer/MinMaxScaler refit-on-A logic (src/data/preprocess.py there),
and that preprocess_tf's TF-constant version numerically matches the numpy
reference. This is the part of the port most prone to a silent mismatch —
CLAUDE.md's attack-repo history (B.17/B.19/B.33) shows this project has hit
that exact class of bug before.
"""

from __future__ import annotations

import h5py
import keras
import numpy as np
import tensorflow as tf
import yaml
from keras import layers

from src.bridge.attack_interface import load as load_bridge

TRACE_LEN = 20
N_CLASSES = 4


def _build_dummy_attacker() -> keras.Model:
    inp = layers.Input(shape=(TRACE_LEN, 1))
    x = layers.Flatten()(inp)
    x = layers.Dense(8, activation="relu")(x)
    out = layers.Dense(N_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out)


def _make_run_dir(tmp_path, *, minmax: bool, method: str = "standardize_per_point"):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    rng = np.random.default_rng(0)
    n_profiling = 50
    traces = rng.integers(-128, 127, size=(n_profiling, TRACE_LEN), dtype=np.int8)

    data_path = tmp_path / "ASCAD.h5"
    with h5py.File(data_path, "w") as f:
        f.create_dataset("Profiling_traces/traces", data=traces)
        f.create_dataset("Attack_traces/traces", data=traces[:5])

    a_idx = np.arange(30)
    np.savez(run_dir / "split_indices.npz", a=a_idx, v=np.arange(30, 40),
              d=np.arange(40, 50), e=np.arange(5))

    cfg = {
        "data": {"path": str(data_path), "trace_len": TRACE_LEN},
        "preprocess": {"method": method, "minmax": minmax},
        "leakage": {"n_classes": N_CLASSES},
    }
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    model = _build_dummy_attacker()
    model.save(run_dir / "model.keras")

    return run_dir, traces, a_idx


def test_fitted_mean_std_match_manual_standardizer(tmp_path):
    run_dir, traces, a_idx = _make_run_dir(tmp_path, minmax=False)
    bridge = load_bridge(run_dir)

    expected_mean = traces[a_idx].astype(np.float64).mean(axis=0)
    expected_std = traces[a_idx].astype(np.float64).std(axis=0)

    np.testing.assert_allclose(bridge.mean, expected_mean)
    np.testing.assert_allclose(bridge.std, expected_std)
    assert bridge.minmax_min is None


def test_preprocess_tf_matches_numpy_standardizer_reference(tmp_path):
    run_dir, traces, a_idx = _make_run_dir(tmp_path, minmax=False)
    bridge = load_bridge(run_dir)

    x_raw = traces[:5].astype(np.float32)
    std_safe = np.where(bridge.std == 0, 1.0, bridge.std)
    expected = ((x_raw - bridge.mean) / std_safe).astype(np.float32)

    got = bridge.preprocess_tf(tf.constant(x_raw[..., None])).numpy()[..., 0]
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_preprocess_tf_matches_numpy_with_minmax_enabled(tmp_path):
    run_dir, traces, a_idx = _make_run_dir(tmp_path, minmax=True)
    bridge = load_bridge(run_dir)
    assert bridge.minmax_min is not None

    x_raw = traces[:5].astype(np.float32)
    std_safe = np.where(bridge.std == 0, 1.0, bridge.std)
    standardized = (x_raw - bridge.mean) / std_safe
    span = np.where(bridge.minmax_max == bridge.minmax_min, 1.0, bridge.minmax_max - bridge.minmax_min)
    expected = ((standardized - bridge.minmax_min) / span).astype(np.float32)

    got = bridge.preprocess_tf(tf.constant(x_raw[..., None])).numpy()[..., 0]
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_preprocess_method_none_is_identity(tmp_path):
    run_dir, traces, a_idx = _make_run_dir(tmp_path, minmax=False, method="none")
    bridge = load_bridge(run_dir)
    assert bridge.mean is None

    x_raw = tf.constant(traces[:5].astype(np.float32)[..., None])
    got = bridge.preprocess_tf(x_raw).numpy()
    np.testing.assert_allclose(got, x_raw.numpy())


def test_predict_probs_rows_sum_to_one(tmp_path):
    run_dir, traces, a_idx = _make_run_dir(tmp_path, minmax=True)
    bridge = load_bridge(run_dir)

    x_raw = tf.constant(traces[:5].astype(np.float32)[..., None])
    probs = bridge.predict_probs(x_raw).numpy()
    assert probs.shape == (5, N_CLASSES)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), atol=1e-4)


def test_resync_enabled_raises_not_implemented(tmp_path):
    run_dir, traces, a_idx = _make_run_dir(tmp_path, minmax=False)
    with open(run_dir / "config_snapshot.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["preprocess"]["resync"] = {"enabled": True}
    with open(run_dir / "config_snapshot.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    import pytest
    with pytest.raises(NotImplementedError):
        load_bridge(run_dir)
