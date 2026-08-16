"""Adversarial training loop: update the perturbation generator against a
frozen dlsca-attack-v2 attacker (via `AttackerBridge`). Ported from
`train_improved_defender.py`'s training loop, see CLAUDE.md §4/§6.

Loss is unchanged from the original in every coefficient, including the
`* 0.5` baked into the entropy term before `lambda_entropy` is applied
(`total += lambda_entropy * (|entropy - target| * 0.5)`) — kept exactly as
written in the original so results stay numerically comparable to it, rather
than being folded into a single equivalent constant.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def compute_loss(
    pred: tf.Tensor,
    perturbation: tf.Tensor,
    lambda_confuse: float = 1.0,
    lambda_l2: float = 0.005,
    lambda_smooth: float = 0.002,
    lambda_entropy: float = 0.1,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    n_classes = tf.shape(pred)[-1]
    uniform = tf.ones_like(pred) / tf.cast(n_classes, tf.float32)

    loss_confuse = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(uniform, pred))

    log_pred = tf.math.log(tf.clip_by_value(pred, 1e-7, 1.0))
    entropy = -tf.reduce_mean(tf.reduce_sum(pred * log_pred, axis=-1))
    target_entropy = tf.math.log(tf.cast(n_classes, tf.float32))
    loss_entropy = tf.abs(entropy - target_entropy) * 0.5

    loss_l2 = tf.reduce_mean(tf.square(perturbation))
    loss_smooth = tf.reduce_mean(tf.square(perturbation[:, 1:, :] - perturbation[:, :-1, :]))

    total = (
        lambda_confuse * loss_confuse
        + lambda_l2 * loss_l2
        + lambda_smooth * loss_smooth
        + lambda_entropy * loss_entropy
    )
    return total, {
        "confuse": loss_confuse,
        "l2": loss_l2,
        "smooth": loss_smooth,
        "entropy": loss_entropy,
    }


def fit(
    generator,
    bridge,
    x_train: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    lambda_confuse: float = 1.0,
    lambda_l2: float = 0.005,
    lambda_smooth: float = 0.002,
    lambda_entropy: float = 0.1,
    patience: int = 5,
) -> list[dict[str, float]]:
    """x_train: (N, trace_len) or (N, trace_len, 1) raw-scale array (int8/float).
    `bridge` needs only a `.predict_probs(x_raw, training=False) -> probs` method
    (an `AttackerBridge`, or any duck-typed stand-in with the same interface).
    Returns per-epoch loss history; training stops early if `patience` epochs
    pass without the mean total loss improving.
    """
    x_train = np.asarray(x_train, dtype=np.float32)
    if x_train.ndim == 2:
        x_train = x_train[..., None]

    tf.random.set_seed(seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(buffer_size=min(len(x_train), 10000), seed=seed)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_losses = {"total": [], "confuse": [], "l2": [], "smooth": [], "entropy": []}
        for x_batch in dataset:
            with tf.GradientTape() as tape:
                perturb = generator(x_batch, training=True)
                x_defended = x_batch + perturb
                pred = bridge.predict_probs(x_defended, training=False)
                total, parts = compute_loss(
                    pred, perturb, lambda_confuse, lambda_l2, lambda_smooth, lambda_entropy
                )
            grads = tape.gradient(total, generator.trainable_variables)
            optimizer.apply_gradients(zip(grads, generator.trainable_variables))

            epoch_losses["total"].append(float(total))
            for k, v in parts.items():
                epoch_losses[k].append(float(v))

        row = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        row["epoch"] = epoch
        history.append(row)
        print(
            f"epoch {epoch:03d}/{epochs} | loss={row['total']:.6f} | "
            f"confuse={row['confuse']:.6f} | l2={row['l2']:.6f} | "
            f"smooth={row['smooth']:.6f} | entropy={row['entropy']:.6f}"
        )

        if row["total"] < best_loss - 1e-9:
            best_loss = row["total"]
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    return history
