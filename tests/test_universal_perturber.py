import keras
import numpy as np

from src.generator.universal_perturber import UniversalTemplate, build_universal_perturbation


def test_output_shape_matches_input_length():
    gen = build_universal_perturbation(trace_len=700, epsilon=6.0)
    x = np.zeros((4, 700, 1), dtype=np.float32)
    out = gen.predict(x, verbose=0)
    assert out.shape == (4, 700, 1)


def test_output_is_bounded_by_epsilon():
    epsilon = 6.0
    gen = build_universal_perturbation(trace_len=100, epsilon=epsilon)
    x = np.random.default_rng(0).normal(0, 20, size=(8, 100, 1)).astype(np.float32)
    out = gen.predict(x, verbose=0)
    assert np.all(np.abs(out) <= epsilon + 1e-4)


def test_output_ignores_input_content_and_is_identical_across_traces():
    # the defining property of a "universal" perturbation: two batches with
    # completely different content must produce the exact same output.
    gen = build_universal_perturbation(trace_len=50, epsilon=6.0)
    rng = np.random.default_rng(1)
    x1 = rng.normal(0, 20, size=(6, 50, 1)).astype(np.float32)
    x2 = rng.normal(100, 5, size=(6, 50, 1)).astype(np.float32)  # very different distribution

    out1 = gen.predict(x1, verbose=0)
    out2 = gen.predict(x2, verbose=0)

    np.testing.assert_allclose(out1, out2)
    # and every row within a batch is identical too (same template broadcast)
    for i in range(1, out1.shape[0]):
        np.testing.assert_allclose(out1[0], out1[i])


def test_untrained_template_starts_at_zero():
    # zeros initializer -> tanh(0) = 0 -> perturbation is exactly zero before training
    gen = build_universal_perturbation(trace_len=30, epsilon=6.0)
    x = np.random.default_rng(2).normal(0, 1, size=(3, 30, 1)).astype(np.float32)
    out = gen.predict(x, verbose=0)
    np.testing.assert_allclose(out, np.zeros_like(out))


def test_save_and_load_round_trip(tmp_path):
    gen = build_universal_perturbation(trace_len=40, epsilon=6.0)
    # nudge the template away from its zero initialization so the round-trip
    # actually exercises a non-trivial saved value
    gen.get_layer("universal_template").template.assign(
        np.random.default_rng(3).normal(0, 1, size=(40, 1)).astype(np.float32)
    )
    x = np.random.default_rng(4).normal(0, 1, size=(2, 40, 1)).astype(np.float32)
    before = gen.predict(x, verbose=0)

    path = tmp_path / "universal.keras"
    gen.save(path)
    reloaded = keras.models.load_model(path)
    after = reloaded.predict(x, verbose=0)

    np.testing.assert_allclose(before, after, atol=1e-5)


def test_gradient_flows_to_the_template_variable():
    import tensorflow as tf

    gen = build_universal_perturbation(trace_len=20, epsilon=6.0)
    x = tf.zeros((4, 20, 1))
    template_var = gen.get_layer("universal_template").template

    with tf.GradientTape() as tape:
        out = gen(x, training=True)
        loss = tf.reduce_sum(tf.square(out - 1.0))  # arbitrary target to force a nonzero gradient
    grads = tape.gradient(loss, [template_var])

    assert grads[0] is not None
    assert not np.allclose(grads[0].numpy(), 0.0)
