"""Universal perturbation "generator" -- a single trainable per-point delta,
broadcast identically to every trace, as opposed to conv_perturber.py's
per-trace CNN output. This is the "Gu et al. 2020"-style defense class
Route A needs as a comparison point (B11209017/方向.md), and simultaneously
the only architecture that can support 方向.md claim②'s "cheaper than
masking" argument: deployed, this needs no per-operation inference at all --
just add a stored constant waveform, versus running a Conv1D forward pass
(conv_perturber.py) on every trace.

Trained through the exact same adversarial loop (src/train/adversarial.py)
and loss as conv_perturber.py -- only the architecture differs, everything
else (frozen attacker, loss weights, evaluation via Stage B) stays identical,
so any PSR/GE difference between the two is attributable to the
universal-vs-per-trace design choice alone.
"""

from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from src.generator.conv_perturber import BoundedPerturbation


@keras.saving.register_keras_serializable(package="dlsca_defense")
class UniversalTemplate(layers.Layer):
    """A single trainable (trace_len, 1) delta, tanh-bounded to (-1, 1) and
    broadcast to every example in the batch regardless of its content --
    the input tensor is used only to read the batch size, never its values,
    which is the defining property of a universal (non-adaptive) perturbation.
    """

    def __init__(self, trace_len: int, **kwargs):
        super().__init__(**kwargs)
        self.trace_len = int(trace_len)

    def build(self, input_shape):
        self.template = self.add_weight(
            name="universal_template",
            shape=(self.trace_len, 1),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        bounded = tf.tanh(self.template)
        return tf.tile(bounded[None, :, :], [batch_size, 1, 1])

    def get_config(self):
        config = super().get_config()
        config.update({"trace_len": self.trace_len})
        return config


def build_universal_perturbation(trace_len: int, epsilon: float = 6.0) -> keras.Model:
    """Same (trace_len, 1) in / (trace_len, 1) in [-epsilon, epsilon] out
    interface as conv_perturber.build_generator, so it's a drop-in swap
    everywhere a generator is used (01_train_defender.py, 02_generate_defended.py).
    """
    inp = layers.Input(shape=(trace_len, 1), name="trace_input")
    template = UniversalTemplate(trace_len, name="universal_template")(inp)
    perturb = BoundedPerturbation(epsilon, name="bounded_perturbation")(template)
    return keras.Model(inp, perturb, name="universal_perturber")
