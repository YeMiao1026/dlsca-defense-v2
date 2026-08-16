"""src/metrics/leakage_loss.py -- see CLAUDE.md A.8/A.9, D04."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.metrics.leakage_loss import hamming_weight, pointwise_squared_correlation, soft_peak_leakage


def test_hamming_weight_matches_popcount():
    labels = tf.constant([0, 1, 3, 255, 128])
    hw = hamming_weight(labels).numpy()
    np.testing.assert_allclose(hw, [0, 1, 2, 8, 1])


def test_pointwise_correlation_is_high_at_a_leaking_column():
    rng = np.random.default_rng(0)
    n, trace_len = 500, 10
    label = rng.integers(0, 256, size=n)
    hw = np.array([bin(v).count("1") for v in label], dtype=np.float32)

    traces = rng.normal(0, 1, size=(n, trace_len)).astype(np.float32)
    traces[:, 3] += hw  # column 3 leaks the Hamming weight directly

    leak = pointwise_squared_correlation(tf.constant(traces), tf.constant(hw)).numpy()
    assert leak[3] > 0.5
    assert np.all(np.delete(leak, 3) < 0.2)


def test_pointwise_correlation_near_zero_for_unrelated_trace():
    rng = np.random.default_rng(1)
    n, trace_len = 500, 20
    label = rng.integers(0, 256, size=n)
    hw = np.array([bin(v).count("1") for v in label], dtype=np.float32)
    traces = rng.normal(0, 1, size=(n, trace_len)).astype(np.float32)  # independent of hw

    leak = pointwise_squared_correlation(tf.constant(traces), tf.constant(hw)).numpy()
    assert np.all(leak < 0.05)


def test_soft_peak_leakage_tracks_the_true_max():
    rng = np.random.default_rng(2)
    n, trace_len = 500, 10
    label = rng.integers(0, 256, size=n)
    hw = np.array([bin(v).count("1") for v in label], dtype=np.float32)
    traces = rng.normal(0, 1, size=(n, trace_len)).astype(np.float32)
    traces[:, 5] += 3.0 * hw  # strong leak at column 5

    leak = pointwise_squared_correlation(tf.constant(traces), tf.constant(hw)).numpy()
    true_max = leak.max()

    soft = float(soft_peak_leakage(tf.constant(traces), tf.constant(hw), temperature=50.0))
    assert abs(soft - true_max) < 0.05
    assert soft >= true_max - 1e-6  # logsumexp/T is always >= max


def test_higher_temperature_tracks_max_more_tightly():
    rng = np.random.default_rng(3)
    n, trace_len = 500, 10
    label = rng.integers(0, 256, size=n)
    hw = np.array([bin(v).count("1") for v in label], dtype=np.float32)
    traces = rng.normal(0, 1, size=(n, trace_len)).astype(np.float32)
    traces[:, 2] += 2.0 * hw
    traces[:, 7] += 1.0 * hw

    x, y = tf.constant(traces), tf.constant(hw)
    soft_low_temp = float(soft_peak_leakage(x, y, temperature=1.0))
    soft_high_temp = float(soft_peak_leakage(x, y, temperature=200.0))
    true_max = float(pointwise_squared_correlation(x, y).numpy().max())

    assert abs(soft_high_temp - true_max) < abs(soft_low_temp - true_max)


def test_leakage_loss_decreases_when_independent_noise_is_added():
    """Confirms the loss has a useful gradient toward the known fix (inject
    independent variance at the leaking point) -- not testing whether a plain
    rescale/shift of the leaking column helps, since squared correlation is
    invariant to any nonzero affine transform of one side by construction
    (that invariance is exactly why D02/D03's deterministic perturbations,
    which are just some function of the input, struggled: rescaling or
    shifting a leaking column doesn't touch its correlation with the label at
    all). Adding independent noise, the way Gaussian defenses actually work,
    is the one intervention that should move this loss -- this checks that it
    does, i.e. the loss rewards the right kind of fix.
    """
    rng = np.random.default_rng(4)
    n, trace_len = 300, 15
    label = rng.integers(0, 256, size=n)
    hw = tf.constant(np.array([bin(v).count("1") for v in label], dtype=np.float32))
    leaking_traces = rng.normal(0, 1, size=(n, trace_len)).astype(np.float32)
    leaking_traces[:, 4] += 3.0 * hw.numpy()  # inject a strong leak at column 4
    leaking_traces = tf.constant(leaking_traces)

    raw_unit_noise = tf.constant(rng.normal(0, 1, size=(n,)).astype(np.float32))
    log_noise_std = tf.Variable(-3.0)  # softplus(-3) ~ 0.05, start near-silent
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)

    initial_leak = float(soft_peak_leakage(leaking_traces, hw, temperature=50.0))
    for _ in range(60):
        with tf.GradientTape() as tape:
            noise = tf.nn.softplus(log_noise_std) * raw_unit_noise
            defended_col4 = leaking_traces[:, 4] + noise
            defended = tf.concat(
                [leaking_traces[:, :4], defended_col4[:, None], leaking_traces[:, 5:]], axis=1
            )
            loss = soft_peak_leakage(defended, hw, temperature=50.0)
        grads = tape.gradient(loss, [log_noise_std])
        optimizer.apply_gradients(zip(grads, [log_noise_std]))

    learned_std = float(tf.nn.softplus(log_noise_std))
    final_noise = learned_std * raw_unit_noise.numpy()
    final_defended = leaking_traces.numpy().copy()
    final_defended[:, 4] += final_noise
    final_leak = float(soft_peak_leakage(tf.constant(final_defended), hw, temperature=50.0))

    assert learned_std > 0.05  # the optimizer actually turned the noise up
    assert final_leak < initial_leak * 0.5
