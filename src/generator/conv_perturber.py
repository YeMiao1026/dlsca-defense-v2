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

A second addition (`AlwaysOnStochasticNoise`, `noise_std` param) is not part
of the original architecture — see CLAUDE.md appendix A.5/D03. D01/D02 showed
every loss-weight variant produces a *deterministic* function of the input
trace (same trace -> same perturbation, every call), unlike i.i.d. Gaussian
noise. This layer injects independent per-call Gaussian noise into the
bounded delta so the generator's output stops being a pure function of the
input, to test whether that's what was limiting D01/D02's efficiency relative
to the Gaussian baseline. It is opt-in (`noise_std=0.0` by default, which
skips inserting the layer entirely) so existing D01/D02 configs and their
saved graphs are unaffected.
"""

from __future__ import annotations

import keras
from keras import layers
import tensorflow as tf


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


@keras.saving.register_keras_serializable(package="dlsca_defense")
class AlwaysOnStochasticNoise(layers.Layer):
    """Adds N(0, noise_std^2) to its input and clips back to [-1, 1] — unlike
    Keras's built-in `GaussianNoise`, this samples fresh noise on every call
    regardless of the `training` flag, so the same input trace produces a
    different perturbation each time the generator runs (training step or
    inference/generation alike).
    """

    def __init__(self, noise_std: float, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = float(noise_std)

    def call(self, inputs):
        noise = tf.random.normal(tf.shape(inputs), stddev=self.noise_std)
        return tf.clip_by_value(inputs + noise, -1.0, 1.0)

    def get_config(self):
        config = super().get_config()
        config.update({"noise_std": self.noise_std})
        return config


def build_generator(trace_len: int, epsilon: float = 6.0, noise_std: float = 0.0) -> keras.Model:
    """Conv1D perturbation generator: raw trace (trace_len, 1) in, bounded
    additive perturbation (trace_len, 1) in [-epsilon, epsilon] out. Caller
    adds the output to the raw trace to get the defended waveform.

    `noise_std` (default 0.0, no-op) injects independent per-call Gaussian
    noise into the bounded delta before it's scaled by epsilon — see
    `AlwaysOnStochasticNoise` above and CLAUDE.md appendix A.5/D03.
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

    if noise_std > 0.0:
        x = AlwaysOnStochasticNoise(noise_std, name="stochastic_noise")(x)

    perturb = BoundedPerturbation(epsilon, name="bounded_perturbation")(x)

    return keras.Model(inp, perturb, name="conv_perturber")
