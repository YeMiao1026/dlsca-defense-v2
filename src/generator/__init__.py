# Importing this package registers BoundedPerturbation with Keras's custom
# object registry, which keras.models.load_model() needs to deserialize a
# saved generator.keras even in scripts that only load a model and never
# call build_generator() themselves (e.g. 02_generate_defended.py).
from src.generator.conv_perturber import AlwaysOnStochasticNoise, BoundedPerturbation, build_generator  # noqa: F401
from src.generator.universal_perturber import UniversalTemplate, build_universal_perturbation  # noqa: F401
