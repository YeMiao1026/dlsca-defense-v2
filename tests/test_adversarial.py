"""src/train/adversarial.py against a tiny synthetic frozen attacker (not the
real ASCAD model) — fast, and isolates the training-loop logic from the
bridge's preprocessing correctness (covered separately in test_bridge.py).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from keras import layers

from src.generator.conv_perturber import build_generator
from src.train.adversarial import compute_loss, fit

TRACE_LEN = 40
N_CLASSES = 4


class _DummyBridge:
    """Duck-typed stand-in for AttackerBridge: a tiny frozen classifier with
    no preprocessing, so fit() only needs `.predict_probs`."""

    def __init__(self):
        import keras

        inp = layers.Input(shape=(TRACE_LEN, 1))
        x = layers.Flatten()(inp)
        x = layers.Dense(8, activation="relu")(x)
        out = layers.Dense(N_CLASSES, activation="softmax")(x)
        self.model = keras.Model(inp, out)
        self.model.trainable = False

    def predict_probs(self, x_raw, training=False):
        return self.model(x_raw, training=training)


def test_compute_loss_confuse_term_is_minimal_at_log_n_for_uniform_predictions():
    # categorical_crossentropy(uniform, uniform) = -sum((1/n)*log(1/n)) = log(n),
    # not 0 — cross-entropy against a uniform target is minimized (not zeroed)
    # when the prediction matches it, since it's still the uniform distribution's
    # own entropy. This is the loss's floor, not a bug.
    pred = tf.ones((3, N_CLASSES)) / N_CLASSES
    perturb = tf.zeros((3, 10, 1))
    total, parts = compute_loss(pred, perturb, lambda_l2=0.0, lambda_smooth=0.0, lambda_entropy=0.0)
    np.testing.assert_allclose(float(parts["confuse"]), np.log(N_CLASSES), atol=1e-5)
    np.testing.assert_allclose(float(total), np.log(N_CLASSES), atol=1e-5)


def test_compute_loss_confuse_term_is_positive_for_confident_wrong_predictions():
    pred = np.zeros((2, N_CLASSES), dtype=np.float32)
    pred[:, 0] = 1.0
    total, parts = compute_loss(tf.constant(pred), tf.zeros((2, 10, 1)))
    assert float(parts["confuse"]) > 1.0


def test_l2_and_smooth_terms_penalize_large_jagged_perturbations():
    pred = tf.ones((1, N_CLASSES)) / N_CLASSES
    small_smooth = tf.zeros((1, 10, 1))
    large_jagged = tf.constant(
        np.array([[6.0, -6.0, 6.0, -6.0, 6.0, -6.0, 6.0, -6.0, 6.0, -6.0]], dtype=np.float32).reshape(1, 10, 1)
    )
    _, parts_small = compute_loss(pred, small_smooth)
    _, parts_large = compute_loss(pred, large_jagged)
    assert float(parts_large["l2"]) > float(parts_small["l2"])
    assert float(parts_large["smooth"]) > float(parts_small["smooth"])


def test_fit_updates_generator_weights():
    generator = build_generator(trace_len=TRACE_LEN, epsilon=6.0)
    bridge = _DummyBridge()
    before = [w.numpy().copy() for w in generator.trainable_variables]

    x_train = np.random.default_rng(0).normal(0, 20, size=(16, TRACE_LEN)).astype(np.float32)
    fit(generator, bridge, x_train, epochs=1, batch_size=4, lr=1e-3, seed=0, patience=5)

    after = [w.numpy() for w in generator.trainable_variables]
    assert any(not np.allclose(b, a) for b, a in zip(before, after))


def test_fit_returns_one_history_row_per_epoch_when_no_early_stop():
    generator = build_generator(trace_len=TRACE_LEN, epsilon=6.0)
    bridge = _DummyBridge()
    x_train = np.random.default_rng(1).normal(0, 20, size=(8, TRACE_LEN)).astype(np.float32)

    history = fit(generator, bridge, x_train, epochs=3, batch_size=4, lr=1e-3, seed=0, patience=100)

    assert len(history) == 3
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert all("total" in row and "confuse" in row for row in history)
