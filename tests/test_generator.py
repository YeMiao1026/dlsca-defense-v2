import numpy as np

from src.generator.conv_perturber import AlwaysOnStochasticNoise, BoundedPerturbation, build_generator


def test_output_shape_matches_input_length():
    gen = build_generator(trace_len=700, epsilon=6.0)
    x = np.zeros((4, 700, 1), dtype=np.float32)
    out = gen.predict(x, verbose=0)
    assert out.shape == (4, 700, 1)


def test_output_is_bounded_by_epsilon():
    epsilon = 6.0
    gen = build_generator(trace_len=700, epsilon=epsilon)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 20, size=(8, 700, 1)).astype(np.float32)
    out = gen.predict(x, verbose=0)
    assert np.all(np.abs(out) <= epsilon + 1e-4)


def test_save_and_load_round_trip(tmp_path):
    gen = build_generator(trace_len=100, epsilon=6.0)
    x = np.random.default_rng(1).normal(0, 1, size=(2, 100, 1)).astype(np.float32)
    before = gen.predict(x, verbose=0)

    path = tmp_path / "generator.keras"
    gen.save(path)

    import keras
    reloaded = keras.models.load_model(path)
    after = reloaded.predict(x, verbose=0)

    np.testing.assert_allclose(before, after, atol=1e-5)


def test_bounded_perturbation_layer_scales_by_epsilon():
    import tensorflow as tf

    layer = BoundedPerturbation(epsilon=3.0)
    x = tf.constant([[-1.0, 0.0, 1.0]])
    out = layer(x).numpy()
    np.testing.assert_allclose(out, [[-3.0, 0.0, 3.0]])


def test_stochastic_noise_is_identity_when_std_is_zero():
    import tensorflow as tf

    layer = AlwaysOnStochasticNoise(noise_std=0.0)
    x = tf.constant([[-0.9, 0.0, 0.9]])
    out = layer(x).numpy()
    np.testing.assert_allclose(out, x.numpy())


def test_stochastic_noise_differs_across_calls_on_the_same_input():
    import tensorflow as tf

    layer = AlwaysOnStochasticNoise(noise_std=0.2)
    x = tf.zeros((1, 700))
    out1 = layer(x).numpy()
    out2 = layer(x).numpy()
    assert not np.allclose(out1, out2)


def test_stochastic_noise_output_stays_within_bounds():
    import tensorflow as tf

    layer = AlwaysOnStochasticNoise(noise_std=5.0)  # deliberately huge relative to [-1,1]
    x = tf.zeros((100, 700))
    out = layer(x).numpy()
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_build_generator_with_noise_std_zero_is_deterministic():
    gen = build_generator(trace_len=100, epsilon=6.0, noise_std=0.0)
    x = np.random.default_rng(2).normal(0, 1, size=(2, 100, 1)).astype(np.float32)
    out1 = gen.predict(x, verbose=0)
    out2 = gen.predict(x, verbose=0)
    np.testing.assert_allclose(out1, out2)


def test_build_generator_with_noise_std_positive_is_stochastic():
    gen = build_generator(trace_len=100, epsilon=6.0, noise_std=0.2)
    x = np.random.default_rng(3).normal(0, 1, size=(2, 100, 1)).astype(np.float32)
    out1 = gen.predict(x, verbose=0)
    out2 = gen.predict(x, verbose=0)
    assert not np.allclose(out1, out2)
    assert np.all(np.abs(out1) <= 6.0 + 1e-4)


def test_stochastic_generator_save_and_load_round_trip_stays_stochastic(tmp_path):
    import keras

    gen = build_generator(trace_len=100, epsilon=6.0, noise_std=0.2)
    x = np.random.default_rng(4).normal(0, 1, size=(2, 100, 1)).astype(np.float32)

    path = tmp_path / "generator.keras"
    gen.save(path)
    reloaded = keras.models.load_model(path)

    out1 = reloaded.predict(x, verbose=0)
    out2 = reloaded.predict(x, verbose=0)
    assert not np.allclose(out1, out2)
