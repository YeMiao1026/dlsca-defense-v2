"""Perturbation-size metrics, for comparing defended vs. clean traces.
Interface reserved for dlsca-defense-v2 (see CLAUDE.md §1.3 non-goals);
inputs here are plain trace arrays so this module has no dependency on the
defense project's internals. All three return one value per trace (N,), not
a single scalar — a defense's cost distribution matters as much as its mean
(a defense that's cheap on average but has a heavy tail is a different claim
than one that's uniformly cheap).
"""

from __future__ import annotations

import numpy as np


def psr(clean: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """Perturbation-to-Signal Ratio per trace: ||perturbed-clean||_2 / ||clean||_2.

    Dimensionless, so it's comparable across defenses applied at different
    raw scales — the "cost" figure to report alongside GE/N_TGE degradation
    when claiming a defense is cheaper than another for the same effect.
    """
    clean = np.asarray(clean, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    signal_norm = np.linalg.norm(clean, axis=1)
    perturb_norm = np.linalg.norm(perturbed - clean, axis=1)
    return np.divide(perturb_norm, signal_norm, out=np.zeros_like(perturb_norm), where=signal_norm > 0)


def l2(clean: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """L2 norm of the per-trace perturbation (absolute, not normalized by signal)."""
    clean = np.asarray(clean, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    return np.linalg.norm(perturbed - clean, axis=1)


def linf(clean: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    """L-infinity norm (max absolute deviation at any single point) per trace —
    catches defenses that are cheap on average but spike hard at a few points.
    """
    clean = np.asarray(clean, dtype=np.float64)
    perturbed = np.asarray(perturbed, dtype=np.float64)
    return np.max(np.abs(perturbed - clean), axis=1)
