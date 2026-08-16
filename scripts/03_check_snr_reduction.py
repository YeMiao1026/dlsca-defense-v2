#!/usr/bin/env python3
"""Diagnostic for CLAUDE.md appendix A.7's hypothesis: does D01's perturbation
actually reduce the ID-leakage SNR the way Gaussian noise does at matched
cost, or does it just flip E01's single-query classification without
destroying the underlying signal (a classic adversarial-example shortcut)?

Compares per-point SNR (attack repo's src/metrics/leakage.py::snr, ID
leakage model) on: clean E, a GAN-defended E (e.g. D01's defended_traces.npy),
and a Gaussian-defended E at roughly matched PSR (from dlsca-attack-v2's
scripts/05_apply_defense.py sweep, CLAUDE.md appendix C.3). No retraining or
Stage B re-evaluation needed — reads existing artifacts only.

Usage (run from repo root):
    python scripts/03_check_snr_reduction.py \
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare SNR reduction: GAN-defended vs Gaussian-defended vs clean")
    p.add_argument("--attack-repo", required=True, help="path to dlsca-attack-v2 (for its src/ and data/)")
    p.add_argument("--attacker-run", required=True, help="dlsca-attack-v2 run dir (for split_indices.npz + data.path)")
    p.add_argument("--gan-defended", required=True, help=".npy of GAN-defended E-set traces")
    p.add_argument("--gaussian-defended", required=True, help=".npy of Gaussian-defended E-set traces (matched PSR)")
    p.add_argument("--leakage-model", choices=["ID", "ID_MASKED"], default="ID",
                    help="ID = what E01 was actually trained to predict (unmasked Z); on ASCAD's masked "
                         "traces this is expected to sit near the SNR noise floor even on clean data, "
                         "since first-order leakage of Z is masked away by design -- E01 only succeeds by "
                         "exploiting higher-order (multivariate) structure a univariate per-point SNR can't "
                         "see. ID_MASKED = the physical leak point the masking scheme doesn't hide (needs "
                         "--mask-index), useful as a second, more sensitive reference channel.")
    p.add_argument("--mask-index", type=int, default=0,
                    help="masks[] column for ID_MASKED (0 is the established value for ASCAD.h5/desync0, "
                         "see dlsca-attack-v2 CLAUDE.md B.6)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.attack_repo)

    import h5py
    import numpy as np
    import yaml

    from src.data.ascad import AES_SBOX
    from src.metrics.leakage import snr
    from src.metrics.perturbation import psr

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
    plaintext_byte = meta_e["plaintext"][:, target_byte].astype(np.uint8)
    key_byte = meta_e["key"][:, target_byte].astype(np.uint8)
    unmasked = AES_SBOX[plaintext_byte ^ key_byte]

    if args.leakage_model == "ID":
        labels = unmasked
        print(f"=== leakage model: ID (unmasked, what E01 was trained on) ===")
    else:
        mask = meta_e["masks"][:, args.mask_index].astype(np.uint8)
        labels = unmasked ^ mask
        print(f"=== leakage model: ID_MASKED (mask_index={args.mask_index}) ===")

    gan_defended = np.load(args.gan_defended).astype(np.float64)
    gaussian_defended = np.load(args.gaussian_defended).astype(np.float64)

    for name, arr in [("GAN-defended", gan_defended), ("Gaussian-defended", gaussian_defended)]:
        if arr.shape != clean.shape:
            raise ValueError(f"{name} shape {arr.shape} != clean E shape {clean.shape}")

    snr_clean = snr(clean, labels)
    snr_gan = snr(gan_defended, labels)
    snr_gaussian = snr(gaussian_defended, labels)

    peak_clean, poi_clean = float(snr_clean.max()), int(snr_clean.argmax())
    peak_gan, poi_gan = float(snr_gan.max()), int(snr_gan.argmax())
    peak_gaussian, poi_gaussian = float(snr_gaussian.max()), int(snr_gaussian.argmax())

    psr_gan = float(psr(clean, gan_defended).mean())
    psr_gaussian = float(psr(clean, gaussian_defended).mean())

    if args.leakage_model == "ID" and peak_clean < 0.5:
        print()
        print(f"  NOTE: clean-E peak SNR ({peak_clean:.4f}) is already near the noise floor for the ID "
              f"label -- expected on ASCAD's masked traces (first-order Z leakage is masked away by "
              f"design). Any % change here is likely noise, not a meaningful signal-destruction measurement. "
              f"Re-run with --leakage-model ID_MASKED for a channel that actually carries strong signal.")

    print()
    print("=== SNR peak comparison ===")
    print(f"  clean E             : peak={peak_clean:.4f} @ point {poi_clean}")
    print(f"  GAN-defended E      : peak={peak_gan:.4f} @ point {poi_gan}  "
          f"(PSR mean={psr_gan:.4f})  -> {100 * (1 - peak_gan / peak_clean):+.1f}% vs clean")
    print(f"  Gaussian-defended E : peak={peak_gaussian:.4f} @ point {poi_gaussian}  "
          f"(PSR mean={psr_gaussian:.4f})  -> {100 * (1 - peak_gaussian / peak_clean):+.1f}% vs clean")

    print()
    print("=== SNR at the clean-signal POI specifically (point", poi_clean, ") ===")
    print(f"  clean    : {snr_clean[poi_clean]:.4f}")
    print(f"  GAN      : {snr_gan[poi_clean]:.4f}  -> {100 * (1 - snr_gan[poi_clean] / snr_clean[poi_clean]):+.1f}% vs clean")
    print(f"  Gaussian : {snr_gaussian[poi_clean]:.4f}  -> {100 * (1 - snr_gaussian[poi_clean] / snr_clean[poi_clean]):+.1f}% vs clean")

    print()
    print("=== verdict (masked-value POI) ===")
    if args.leakage_model == "ID" and peak_clean < 0.5:
        print("  SKIPPED: clean-E peak SNR is near the noise floor for this label (see NOTE above) -- "
              "a verdict from this comparison would not be trustworthy. Re-run with --leakage-model ID_MASKED.")
        return
    gan_reduction = 1 - peak_gan / peak_clean
    gaussian_reduction = 1 - peak_gaussian / peak_clean
    if gan_reduction < 0.5 * gaussian_reduction:
        print("  GAN barely touches SNR relative to Gaussian at matched cost -> supports appendix A.7:")
        print("  D01's perturbation looks like an adversarial-example shortcut, not real leakage destruction.")
    elif gan_reduction >= gaussian_reduction:
        print("  GAN reduces SNR at least as much as Gaussian at matched cost -> A.7 does NOT explain the gap;")
        print("  the GE deficit must come from something else (e.g. how the reduction interacts with the")
        print("  attacker's specific decision function, not raw SNR).")
    else:
        print("  GAN reduces SNR less than Gaussian but not negligibly -> partial support for A.7,")
        print("  worth a closer look rather than a clean verdict either way.")

    if args.leakage_model == "ID_MASKED":
        # A masking countermeasure needs its raw mask value r_out to stay
        # secret too -- if the trace leaks r_out on its own at some other
        # point (independent of plaintext/key), a 2nd-order attacker can
        # combine that with the masked-value POI above to undo the masking
        # even if the masked-value POI's SNR looks well-suppressed in
        # isolation. Check whether that second ingredient survives too.
        mask_only = meta_e["masks"][:, args.mask_index].astype(np.uint8)
        snr_clean_mask = snr(clean, mask_only)
        snr_gan_mask = snr(gan_defended, mask_only)
        snr_gaussian_mask = snr(gaussian_defended, mask_only)
        peak_clean_m = float(snr_clean_mask.max())
        poi_clean_m = int(snr_clean_mask.argmax())

        print()
        print(f"=== second ingredient: raw mask value r_out leak point (independent of plaintext/key) ===")
        print(f"  clean E             : peak={peak_clean_m:.4f} @ point {poi_clean_m}")
        print(f"  GAN-defended E      : peak={float(snr_gan_mask.max()):.4f} @ point {int(snr_gan_mask.argmax())}"
              + (f"  -> {100 * (1 - snr_gan_mask.max() / peak_clean_m):+.1f}% vs clean" if peak_clean_m > 0 else ""))
        print(f"  Gaussian-defended E : peak={float(snr_gaussian_mask.max()):.4f} @ point {int(snr_gaussian_mask.argmax())}"
              + (f"  -> {100 * (1 - snr_gaussian_mask.max() / peak_clean_m):+.1f}% vs clean" if peak_clean_m > 0 else ""))
        if peak_clean_m < 0.5:
            print(f"  NOTE: clean-E peak SNR for the raw mask value is near the noise floor "
                  f"({peak_clean_m:.4f}) -- no single point directly leaks r_out on its own here; "
                  f"if E01 is exploiting a 2nd-order combination, it isn't through a univariate leak "
                  f"of r_out at a single point either, so this specific check is inconclusive too.")


if __name__ == "__main__":
    main()
