#!/usr/bin/env python3
"""TVLA-style leakage check (方向.md claim①: "TVLA / Welch t-test：對 defended
traces 做固定 vs 隨機明文的 t 檢定"). Complementary to 03_check_snr_reduction.py:
SNR/NICV give a continuous measure of how much leakage remains; TVLA gives the
standard security-evaluation yes/no answer of whether ANY detectable,
model-agnostic difference survives at all.

ASCAD's Attack_traces don't include a repeated identical-plaintext ("fixed")
acquisition, so a literal fixed-vs-random-plaintext TVLA campaign isn't
possible from this dataset. This runs the standard fallback used when only a
single random-input campaign is available: a bit-split ("specific") TVLA --
split traces into two groups by a single bit of a known intermediate value
and Welch's t-test them (dlsca-attack-v2's src/metrics/leakage.py::t_test,
imported not copied, same boundary 03_check_snr_reduction.py already uses).
Swept over all 8 bits of each of the two known leak channels (CLAUDE.md
appendix A.8: masked_value, mask_value), reporting the worst (max |t|) bit
per channel -- |t| >= 4.5 anywhere is the conventional TVLA threshold.

Usage (run from repo root):
    python scripts/04_check_tvla.py \
        --attack-repo /home/yemiao1026/Final_Project \
        --attacker-run /home/yemiao1026/Final_Project/runs/E01_baseline_clean_20260816_1302 \
        --gan-defended runs/D01_replicate_baseline_20260816_232552_948723/defended_traces.npy \
        --gaussian-defended /home/yemiao1026/Final_Project/defenses/gaussian_sigma0.25_20260816_225339_137443/defended_traces.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TVLA_THRESHOLD = 4.5
N_BITS = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bit-split TVLA check: GAN-defended vs Gaussian-defended vs clean")
    p.add_argument("--attack-repo", required=True, help="path to dlsca-attack-v2 (for its src/ and data/)")
    p.add_argument("--attacker-run", required=True, help="dlsca-attack-v2 run dir (for split_indices.npz + data.path)")
    p.add_argument("--gan-defended", required=True, help=".npy of GAN-defended E-set traces")
    p.add_argument("--gaussian-defended", required=True, help=".npy of Gaussian-defended E-set traces (matched PSR)")
    p.add_argument("--mask-index", type=int, default=0,
                    help="masks[] column (0 is the established value for ASCAD.h5/desync0, "
                         "see dlsca-attack-v2 CLAUDE.md B.6)")
    return p.parse_args()


def worst_bit_tvla(t_test_fn, traces, label, n_bits: int = N_BITS):
    """Sweep all bits of `label`, return (worst |t|, bit, time point) over the trace."""
    import numpy as np

    worst_peak, worst_bit, worst_poi = 0.0, None, None
    for bit in range(n_bits):
        bit_val = (label >> bit) & 1
        group_a = traces[bit_val == 0]
        group_b = traces[bit_val == 1]
        if len(group_a) < 2 or len(group_b) < 2:
            continue
        t = t_test_fn(group_a, group_b)
        peak = float(np.max(np.abs(t)))
        if peak > worst_peak:
            worst_peak, worst_bit, worst_poi = peak, bit, int(np.argmax(np.abs(t)))
    return worst_peak, worst_bit, worst_poi


def _load_attack_leakage_module(attack_repo: str):
    """dlsca-attack-v2 and dlsca-defense-v2 both have a top-level `src` package,
    so `sys.path.insert(0, attack_repo)` + `from src.X import Y` is unsafe here:
    whichever repo's `src` gets imported first wins the `sys.modules['src']`
    cache, and the other repo's `src.<anything>` imports then silently resolve
    against the wrong package (or fail, as this did during development --
    `src.leakage_probe` came up "not found" because `src` had already been
    bound to dlsca-attack-v2's package by an earlier `src.metrics.leakage`
    import). Loading attack repo's leakage.py by explicit file path sidesteps
    the collision entirely; it has no relative imports of its own (just numpy),
    so this is safe.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "dlsca_attack_leakage", Path(attack_repo) / "src" / "metrics" / "leakage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()

    import h5py
    import numpy as np
    import yaml

    attack_t_test = _load_attack_leakage_module(args.attack_repo).t_test
    from src.leakage_probe import compute_leakage_labels

    attacker_run = Path(args.attacker_run)
    with open(attacker_run / "config_snapshot.yaml") as f:
        cfg = yaml.safe_load(f)
    target_byte = cfg["data"]["target_byte"]

    data_path = Path(cfg["data"]["path"])
    if not data_path.is_absolute():
        data_path = Path(args.attack_repo) / data_path

    print(f"=== loading {data_path} ===")
    with h5py.File(data_path, "r") as f:
        attack_traces = np.array(f["Attack_traces/traces"], dtype=np.int8)
        attack_meta = np.array(f["Attack_traces/metadata"])
    with np.load(attacker_run / "split_indices.npz") as split:
        e_idx = split["e"]

    clean = attack_traces[e_idx].astype(np.float64)
    meta_e = attack_meta[e_idx]
    labels = compute_leakage_labels(meta_e, target_byte, args.mask_index)

    gan_defended = np.load(args.gan_defended).astype(np.float64)
    gaussian_defended = np.load(args.gaussian_defended).astype(np.float64)

    datasets = {
        "clean (positive control)": clean,
        "GAN-defended": gan_defended,
        "Gaussian-defended": gaussian_defended,
    }
    for name, arr in datasets.items():
        if arr.shape != clean.shape:
            raise ValueError(f"{name} shape {arr.shape} != clean E shape {clean.shape}")

    print()
    print(f"=== bit-split TVLA (Welch t-test, threshold |t| >= {TVLA_THRESHOLD}) ===")
    print("(clean is a positive control: it MUST show leakage, otherwise the test itself is broken)")

    results: dict[str, dict[str, tuple]] = {}
    for channel_name, label in labels.items():
        print(f"\n--- channel: {channel_name} ---")
        results[channel_name] = {}
        for ds_name, traces in datasets.items():
            peak, bit, poi = worst_bit_tvla(attack_t_test, traces, label)
            results[channel_name][ds_name] = (peak, bit, poi)
            verdict = "LEAKAGE DETECTED" if peak >= TVLA_THRESHOLD else "no leakage detected"
            print(f"  {ds_name:<28s} worst |t|={peak:8.2f}  (bit {bit}, point {poi:<4})  -> {verdict}")

    print()
    print("=== verdict ===")
    for channel_name in labels:
        clean_peak = results[channel_name]["clean (positive control)"][0]
        if clean_peak < TVLA_THRESHOLD:
            print(f"  {channel_name}: SKIPPED -- clean positive control didn't leak (|t|={clean_peak:.2f} < "
                  f"{TVLA_THRESHOLD}), so this channel/test setup can't validate anything here.")
            continue
        gan_peak = results[channel_name]["GAN-defended"][0]
        gaussian_peak = results[channel_name]["Gaussian-defended"][0]
        gan_leaks = gan_peak >= TVLA_THRESHOLD
        gaussian_leaks = gaussian_peak >= TVLA_THRESHOLD
        print(f"  {channel_name}: GAN {'still leaks' if gan_leaks else 'passes (no detectable leakage)'} "
              f"(|t|={gan_peak:.2f}), Gaussian {'still leaks' if gaussian_leaks else 'passes'} (|t|={gaussian_peak:.2f})")


if __name__ == "__main__":
    main()
