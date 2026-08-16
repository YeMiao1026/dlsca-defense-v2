import numpy as np

from src.generator.conv_perturber import BoundedPerturbation, build_generator


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
