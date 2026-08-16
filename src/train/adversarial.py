"""Adversarial training loop: update the perturbation generator against a
frozen dlsca-attack-v2 attacker (via `AttackerBridge`). Ported from
`train_improved_defender.py`'s training loop, see CLAUDE.md §4/§6.

Loss is unchanged from the original in every coefficient, including the
`* 0.5` baked into the entropy term before `lambda_entropy` is applied
(`total += lambda_entropy * (|entropy - target| * 0.5)`) — kept exactly as
written in the original so results stay numerically comparable to it, rather
than being folded into a single equivalent constant.

`leakage_labels` (see CLAUDE.md appendix A.8/A.9, D04) adds an optional,
opt-in term (`lambda_leakage`, default 0.0 = no-op, exact D01/D02/D03
behavior) that directly penalizes leakage at known leak channels via
`src.metrics.leakage_loss`, rather than relying solely on fooling the frozen
attacker's single-query output.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.metrics.leakage_loss import hamming_weight, soft_peak_leakage


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
    leakage_labels: dict[str, np.ndarray] | None = None,
    lambda_leakage: float = 0.0,
    leakage_temperature: float = 50.0,
) -> list[dict[str, float]]:
    """x_train: (N, trace_len) or (N, trace_len, 1) raw-scale array (int8/float).
    `bridge` needs only a `.predict_probs(x_raw, training=False) -> probs` method
    (an `AttackerBridge`, or any duck-typed stand-in with the same interface).
    Returns per-epoch loss history; training stops early if `patience` epochs
    pass without the mean total loss improving.

    `leakage_labels`: optional {channel_name: (N,) int array in [0,255]},
    aligned index-for-index with `x_train` -- e.g. {"masked_value": ..., "mask_value":
    ...} for the two leak channels CLAUDE.md appendix A.8 identified. Each channel
    adds `lambda_leakage * soft_peak_leakage(x_defended, hamming_weight(label))` to
    the loss (see src/metrics/leakage_loss.py). `lambda_leakage=0.0` (default) is a
    byte-identical no-op regardless of whether `leakage_labels` is passed, so
    existing D01/D02/D03 configs are unaffected.
    """
    x_train = np.asarray(x_train, dtype=np.float32)
    if x_train.ndim == 2:
        x_train = x_train[..., None]

    use_leakage = leakage_labels is not None and lambda_leakage > 0.0
    channel_names = list(leakage_labels.keys()) if use_leakage else []

    tf.random.set_seed(seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if use_leakage:
        labels_f32 = {name: np.asarray(arr, dtype=np.int32) for name, arr in leakage_labels.items()}
        dataset = tf.data.Dataset.from_tensor_slices((x_train, labels_f32))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(buffer_size=min(len(x_train), 10000), seed=seed)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    loss_keys = ["total", "confuse", "l2", "smooth", "entropy"] + [f"leak_{n}" for n in channel_names]
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_losses = {k: [] for k in loss_keys}
        for batch in dataset:
            x_batch, label_batch = batch if use_leakage else (batch, {})
            with tf.GradientTape() as tape:
                perturb = generator(x_batch, training=True)
                x_defended = x_batch + perturb
                pred = bridge.predict_probs(x_defended, training=False)
                total, parts = compute_loss(
                    pred, perturb, lambda_confuse, lambda_l2, lambda_smooth, lambda_entropy
                )
                for name in channel_names:
                    hw = hamming_weight(label_batch[name])
                    leak = soft_peak_leakage(x_defended, hw, leakage_temperature)
                    parts[f"leak_{name}"] = leak
                    total = total + lambda_leakage * leak
            grads = tape.gradient(total, generator.trainable_variables)
            optimizer.apply_gradients(zip(grads, generator.trainable_variables))

            epoch_losses["total"].append(float(total))
            for k, v in parts.items():
                epoch_losses[k].append(float(v))

        row = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        row["epoch"] = epoch
        history.append(row)
        leak_str = " | ".join(f"{k}={row[k]:.6f}" for k in loss_keys if k.startswith("leak_"))
        print(
            f"epoch {epoch:03d}/{epochs} | loss={row['total']:.6f} | "
            f"confuse={row['confuse']:.6f} | l2={row['l2']:.6f} | "
            f"smooth={row['smooth']:.6f} | entropy={row['entropy']:.6f}"
            + (f" | {leak_str}" if leak_str else "")
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
