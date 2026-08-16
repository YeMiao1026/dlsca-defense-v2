"""Perturbation generator, ported from the prior project's
`ASCAD/GAN/train_improved_defender.py::build_improved_defender` (see
CLAUDE.md §4/§6). Architecture is left unchanged from the original — only the
frozen-attacker target it trains against and the preprocessing discipline
around it have been fixed, not the network itself.

One deliberate deviation: the original used `layers.Lambda(lambda z: epsilon
* z)` for the bounded-output step, which requires
`keras.config.enable_unsafe_deserialization()` to reload from a saved
`.keras` file. `BoundedPerturbation` below is the same operation as a
registered layer, so `generator.save()` / `keras.models.load_model()` round-trip
without that workaround. This changes nothing about what the architecture
computes.
"""

from __future__ import annotations

import keras
from keras import layers


@keras.saving.register_keras_serializable(package="dlsca_defense")
class BoundedPerturbation(layers.Layer):
    """Scales a tanh-bounded [-1, 1] delta to [-epsilon, epsilon]."""

    def __init__(self, epsilon: float, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, inputs):
        return inputs * self.epsilon

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


def build_generator(trace_len: int, epsilon: float = 6.0) -> keras.Model:
    """Conv1D perturbation generator: raw trace (trace_len, 1) in, bounded
    additive perturbation (trace_len, 1) in [-epsilon, epsilon] out. Caller
    adds the output to the raw trace to get the defended waveform.
    """
    inp = layers.Input(shape=(trace_len, 1), name="trace_input")

    x = layers.Conv1D(32, 7, padding="same", activation="relu", name="def_conv1")(inp)
    x = layers.BatchNormalization(name="def_bn1")(x)

    x_skip = x
    x = layers.Conv1D(32, 7, padding="same", activation="relu", name="def_conv2")(x)
    x = layers.BatchNormalization(name="def_bn2")(x)
    x = layers.Add(name="def_residual1")([x, x_skip])

    x = layers.Conv1D(16, 5, padding="same", activation="relu", name="def_conv3")(x)
    x = layers.BatchNormalization(name="def_bn3")(x)

    x = layers.Conv1D(8, 5, padding="same", activation="relu", name="def_conv4")(x)
    x = layers.BatchNormalization(name="def_bn4")(x)

    x = layers.Conv1D(1, 3, padding="same", activation="tanh", name="def_tanh_delta")(x)

    perturb = BoundedPerturbation(epsilon, name="bounded_perturbation")(x)

    return keras.Model(inp, perturb, name="conv_perturber")
