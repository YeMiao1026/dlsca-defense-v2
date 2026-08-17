#!/usr/bin/env python3
"""D06 (21-day roadmap item 4, CLAUDE.md §3's A1 scenario): run a trained
generator over the FULL Profiling_traces array (not just the E-set 02_generate_
defended.py targets), producing a drop-in replacement for dlsca-attack-v2's
`data.profiling_traces` -- the input dlsca-attack-v2's 01_train_attacker.py's
new `--profiling-traces` flag expects, to retrain an attacker that has seen
defended traces during training (the adaptive A1 attacker).

Only the trace waveform is substituted; plaintext/key/masks metadata still
comes from the original .h5 (defending the physical signal doesn't change
what values were being processed), so dlsca-attack-v2's own split/label/train
pipeline runs completely unmodified downstream of this substitution.

Usage (run from repo root):
    python scripts/06_generate_defended_trainset.py --run runs/D01_replicate_baseline_20260816_232552_948723
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

import src.generator  # noqa: F401 -- registers custom layers before load_model()
from src.metrics.perturbation import l2, linf, psr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply a trained generator to the full Profiling_traces set (A1 retraining)")
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

    print(f"=== loading {data_path} (full Profiling_traces, not just A/V/D split) ===")
    with h5py.File(data_path, "r") as f:
        clean = np.array(f["Profiling_traces/traces"], dtype=np.float32)
    print(f"  clean profiling traces shape={clean.shape}")

    print(f"=== loading generator {run_dir / 'generator.keras'} ===")
    generator = keras.models.load_model(run_dir / "generator.keras")

    print(f"=== applying generator to {len(clean)} profiling traces ===")
    perturbation = generator.predict(clean[..., None], verbose=0)[..., 0]
    defended = clean + perturbation

    cost = {"psr": _summary(psr(clean, defended)), "l2": _summary(l2(clean, defended)),
            "linf": _summary(linf(clean, defended))}

    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "defended_profiling_traces.npy"
    np.save(traces_path, defended.astype(np.float32))
    with open(out_dir / "trainset_cost_metrics.json", "w") as f:
        json.dump(cost, f, indent=2)

    print(f"=== saved {traces_path} ===")
    print(f"=== saved {out_dir / 'trainset_cost_metrics.json'} ===")
    print(f"PSR mean={cost['psr']['mean']:.4f}  L2 mean={cost['l2']['mean']:.2f}  Linf mean={cost['linf']['mean']:.2f}")
    print()
    print("=== next: retrain an adaptive attacker (run from <dlsca-attack-v2> root) ===")
    print(f"python3 scripts/01_train_attacker.py --config configs/exp/E01_adaptive_A1_retrain.yaml "
          f"--profiling-traces {traces_path.resolve()}")


if __name__ == "__main__":
    main()
