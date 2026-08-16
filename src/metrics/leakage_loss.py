"""Differentiable leakage-suppression loss term. See CLAUDE.md appendix A.8/A.9
and D04: the SNR diagnostic found that D01's confuse-only loss suppresses the
attacker's most gradient-sensitive leak point about as well as Gaussian noise,
but leaves a second leak point the attack also needs almost untouched --
because nothing in the loss required covering more than one point. This
module adds a term that directly penalizes leakage at *all* points of a
given channel, not just wherever the frozen attacker's gradient happens to
point.

Design choice: batch-level class-conditional SNR (src/metrics/leakage.py's
approach, copied from dlsca-attack-v2) needs many samples per class to be
stable -- with 256 classes and this project's batch_size=64, most classes
would have 0-1 samples per batch, an unusable estimator. Squared Pearson
correlation between the trace and the label's Hamming weight is the standard
low-sample-count alternative used in CPA-style SCA attacks (it assumes a
linear-in-Hamming-weight leakage model, which is the standard assumption for
CMOS power leakage and is what makes CPA work with far fewer traces than a
full class-conditional estimator needs) -- known limitation: if the true
leakage isn't well approximated by linear-in-HW, this term could miss it,
same blind spot CPA itself has.
"""

from __future__ import annotations

import tensorflow as tf

HW_TABLE = tf.constant([bin(i).count("1") for i in range(256)], dtype=tf.float32)


def hamming_weight(labels: tf.Tensor) -> tf.Tensor:
    """labels: (N,) int in [0, 255] -> (N,) float32 Hamming weight in [0, 8]."""
    return tf.gather(HW_TABLE, tf.cast(labels, tf.int32))


def pointwise_squared_correlation(x_defended: tf.Tensor, label_hw: tf.Tensor) -> tf.Tensor:
    """x_defended: (batch, trace_len, 1) or (batch, trace_len). label_hw: (batch,) float32.
    Returns (trace_len,) squared Pearson correlation per time point, in [0, 1] --
    the standard CPA leakage estimator, computed batch-wise (like a per-batch NICV
    under a linear-in-HW leakage model).
    """
    if x_defended.shape.rank == 3:
        x_defended = tf.squeeze(x_defended, axis=-1)
    x_centered = x_defended - tf.reduce_mean(x_defended, axis=0, keepdims=True)
    y_centered = label_hw - tf.reduce_mean(label_hw)
    numerator = tf.reduce_sum(x_centered * y_centered[:, None], axis=0)
    x_std = tf.sqrt(tf.reduce_sum(tf.square(x_centered), axis=0) + 1e-8)
    y_std = tf.sqrt(tf.reduce_sum(tf.square(y_centered)) + 1e-8)
    corr = numerator / (x_std * y_std)
    return tf.square(corr)


def soft_peak_leakage(x_defended: tf.Tensor, label_hw: tf.Tensor, temperature: float = 50.0) -> tf.Tensor:
    """Smooth approximation of max_t(squared correlation at t) via logsumexp --
    unlike a hard max (gradient only at the single worst point each step, prone
    to whack-a-mole instability) or a mean (dilutes the one point that actually
    matters among 700 mostly-irrelevant ones), this pushes down the whole upper
    tail while staying differentiable everywhere. Higher temperature tracks the
    true max more closely; lower temperature spreads gradient over more points.
    """
    leak = pointwise_squared_correlation(x_defended, label_hw)
    return tf.reduce_logsumexp(temperature * leak) / temperature
