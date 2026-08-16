#!/usr/bin/env python3
"""Run a trained generator on the attacker's E set, save defended_traces.npy +
cost_metrics.json (PSR/L2/Linf). Output feeds directly into dlsca-attack-v2's
Stage B interface (CLAUDE.md §2/§5) — this script never computes GE itself.

Usage (run from repo root):
    python scripts/02_generate_defended.py --run runs/D01_replicate_baseline_...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import keras
import numpy as np
import yaml

import src.generator  # noqa: F401 -- registers BoundedPerturbation before load_model()
from src.metrics.perturbation import l2, linf, psr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply a trained generator to the attacker's E set")
    p.add_argument("--run", required=True, help="run directory produced by 01_train_defender.py")
    p.add_argument("--out-dir", default=None, help="default: {run}/")
    return p.parse_args()


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir

    with open(run_dir / "config_snapshot.yaml") as f:
        cfg = yaml.safe_load(f)
    attacker_run = Path(cfg["attacker"]["run"])
    with open(attacker_run / "config_snapshot.yaml") as f:
        attacker_cfg = yaml.safe_load(f)

    data_path = Path(attacker_cfg["data"]["path"])
    if not data_path.is_absolute():
        data_path = attacker_run.parent.parent / data_path

    print(f"=== loading {data_path} ===")
    with h5py.File(data_path, "r") as f:
        attack_traces = np.array(f["Attack_traces/traces"], dtype=np.int8)
    with np.load(attacker_run / "split_indices.npz") as split:
        e_idx = split["e"]
    traces_e = attack_traces[e_idx].astype(np.float32)

    print(f"=== loading generator {run_dir / 'generator.keras'} ===")
    generator = keras.models.load_model(run_dir / "generator.keras")

    print(f"=== applying generator to {len(traces_e)} E-set traces ===")
    perturbation = generator.predict(traces_e[..., None], verbose=0)[..., 0]
    defended = traces_e + perturbation

    cost = {"psr": _summary(psr(traces_e, defended)), "l2": _summary(l2(traces_e, defended)),
            "linf": _summary(linf(traces_e, defended))}

    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "defended_traces.npy"
    np.save(traces_path, defended.astype(np.float32))
    with open(out_dir / "cost_metrics.json", "w") as f:
        json.dump(cost, f, indent=2)

    print(f"=== saved {traces_path} ===")
    print(f"=== saved {out_dir / 'cost_metrics.json'} ===")
    print(f"PSR mean={cost['psr']['mean']:.4f}  L2 mean={cost['l2']['mean']:.2f}  Linf mean={cost['linf']['mean']:.2f}")
    print()
    print("=== next: evaluate via dlsca-attack-v2's Stage B interface ===")
    print(f"python3 <dlsca-attack-v2>/scripts/02_run_attack.py --run {attacker_run} "
          f"--traces {traces_path} --out {out_dir / 'probs.npy'}")
    print(f"python3 <dlsca-attack-v2>/scripts/03_evaluate.py --run {attacker_run} "
          f"--probs {out_dir / 'probs.npy'} --out {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
