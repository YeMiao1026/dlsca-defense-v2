#!/usr/bin/env python3
"""Adversarial-train the perturbation generator against a frozen
dlsca-attack-v2 attacker. See CLAUDE.md §5/§6.

Usage (run from repo root):
    python scripts/01_train_defender.py --config configs/exp/D01_replicate_baseline.yaml
    python scripts/01_train_defender.py --config configs/exp/D01_replicate_baseline.yaml \
        --override train.epochs=2 data.n_train=200   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import tensorflow as tf
import yaml

from src.bridge.attack_interface import load as load_bridge
from src.config import load_config, snapshot
from src.generator.conv_perturber import build_generator
from src.generator.universal_perturber import build_universal_perturbation
from src.train.adversarial import fit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the adversarial perturbation generator (CLAUDE.md §6)")
    p.add_argument("--config", required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--runs-dir", default="runs",
                    help="output root; run_dir = {runs-dir}/{exp_id}_{timestamp}_{pid} "
                         "(PID-suffixed to avoid the collision dlsca-attack-v2 hit twice "
                         "with parallel launches, see its CLAUDE.md B.35/B.37)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    np.random.seed(cfg["seed"])
    tf.random.set_seed(cfg["seed"])

    attacker_run = Path(cfg["attacker"]["run"])
    print(f"=== loading frozen attacker from {attacker_run} ===")
    bridge = load_bridge(attacker_run)

    with open(attacker_run / "config_snapshot.yaml") as f:
        attacker_cfg = yaml.safe_load(f)
    data_path = Path(attacker_cfg["data"]["path"])
    if not data_path.is_absolute():
        data_path = attacker_run.parent.parent / data_path

    probe_cfg = cfg.get("leakage_probe", {})
    probe_enabled = probe_cfg.get("enabled", False)

    print(f"=== loading A-set training traces from {data_path} (n_train={cfg['data']['n_train']}) ===")
    with h5py.File(data_path, "r") as f:
        profiling_traces = np.array(f["Profiling_traces/traces"], dtype=np.int8)
        profiling_meta = np.array(f["Profiling_traces/metadata"]) if probe_enabled else None
    with np.load(attacker_run / "split_indices.npz") as split:
        a_idx = split["a"]
    n_train = min(cfg["data"]["n_train"], len(a_idx))
    train_idx = a_idx[:n_train]
    x_train = profiling_traces[train_idx].astype(np.float32)
    print(f"  x_train shape={x_train.shape}")

    leakage_labels = None
    if probe_enabled:
        from src.leakage_probe import compute_leakage_labels

        mask_index = probe_cfg.get("mask_index", 0)
        target_byte = attacker_cfg["data"]["target_byte"]
        meta_train = profiling_meta[train_idx]
        leakage_labels = compute_leakage_labels(meta_train, target_byte, mask_index)
        print(f"=== leakage_probe enabled: mask_index={mask_index}, "
              f"channels={list(leakage_labels.keys())} (CLAUDE.md A.8/A.9, D04) ===")

    architecture = cfg["generator"].get("architecture", "cnn")
    epsilon = cfg["generator"]["epsilon"]
    if architecture == "cnn":
        noise_std = cfg["generator"].get("noise_std", 0.0)
        print(f"=== building generator (architecture=cnn, epsilon={epsilon}, noise_std={noise_std}) ===")
        generator = build_generator(trace_len=x_train.shape[1], epsilon=epsilon, noise_std=noise_std)
    elif architecture == "universal":
        print(f"=== building generator (architecture=universal, epsilon={epsilon}) ===")
        generator = build_universal_perturbation(trace_len=x_train.shape[1], epsilon=epsilon)
    else:
        raise ValueError(f"unknown generator.architecture: {architecture!r}")

    loss_cfg = cfg["train"].get("loss", {})
    print("\n=== start adversarial training ===")
    history = fit(
        generator,
        bridge,
        x_train,
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        lr=cfg["train"]["lr"],
        seed=cfg["seed"],
        patience=cfg["train"].get("patience", 5),
        lambda_confuse=loss_cfg.get("lambda_confuse", 1.0),
        lambda_l2=loss_cfg.get("lambda_l2", 0.005),
        lambda_smooth=loss_cfg.get("lambda_smooth", 0.002),
        lambda_entropy=loss_cfg.get("lambda_entropy", 0.1),
        leakage_labels=leakage_labels,
        lambda_leakage=loss_cfg.get("lambda_leakage", 0.0),
        leakage_temperature=loss_cfg.get("leakage_temperature", 50.0),
    )

    run_dir = Path(args.runs_dir) / f"{cfg['exp_id']}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot(cfg, run_dir)
    generator.save(run_dir / "generator.keras")
    with open(run_dir / "train_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print(f"\n=== saved {run_dir / 'generator.keras'} ===")
    print(f"=== saved {run_dir / 'train_history.csv'} ===")
    print(f"\nnext: python scripts/02_generate_defended.py --run {run_dir}")


if __name__ == "__main__":
    main()
