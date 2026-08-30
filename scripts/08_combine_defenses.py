#!/usr/bin/env python3
"""Stack D04's leakage-aware amplitude perturbation with D08's time-domain
jamming, to test the future-work hypothesis (CLAUDE.md 附錄N/report_draft.md
6.3 item 3): does a real information-destroying defense (D04) survive resync
better when combined with jamming, than jamming alone does (附錄N: jamming
alone at max_shift=20 is fully reversed by resync)?

Applies jamming_augment on top of an already-GAN-defended array (D04's
defended_traces.npy for the E-set, or defended_profiling_traces.npy for the
full 50000-trace Profiling_traces), not on raw clean traces -- the two
mechanisms stack, they don't replace each other. For --target e, cost is
measured against the true clean E set (not against the GAN-defended traces),
consistent with how every other defense in this project reports PSR; for
--target profiling, no cost accounting is done since that array feeds an
adaptive attacker's retraining, not a direct evaluation.

Usage (run from repo root):
    # static/A0 evaluation on the E-set:
    python scripts/08_combine_defenses.py --target e \
        --attack-repo ~/dlsca-attack-v2 \
        --attacker-run ~/dlsca-attack-v2/runs/E01_baseline_clean_20260816_1302 \
        --gan-defended-traces runs/D04_snr_aware_.../defended_traces.npy \
        --max-shift 20 --out-dir defenses_d08_combo

    # combined-defended profiling set, to retrain an adaptive (A1) attacker on:
    python scripts/08_combine_defenses.py --target profiling \
        --attack-repo ~/dlsca-attack-v2 \
        --attacker-run ~/dlsca-attack-v2/runs/E01_baseline_clean_20260816_1302 \
        --gan-defended-traces runs/D04_snr_aware_.../defended_profiling_traces.npy \
        --max-shift 20 --out-dir defenses_d08_combo
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import yaml


def _load_attack_module(attack_repo: str, relpath: str, name: str):
    """See CLAUDE.md 附錄 (04_check_tvla.py / 07_check_resync_bypass.py) for why
    this loads by explicit file path instead of sys.path.insert + `from
    src.X import Y`: both repos have a top-level `src` package, and the
    second one imported would silently resolve against the first repo's
    package via the shared `sys.modules['src']` cache.
    """
    spec = importlib.util.spec_from_file_location(name, Path(attack_repo) / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combine a GAN-defended E-set with jamming; measure cost vs true clean"
    )
    p.add_argument("--attack-repo", required=True, help="path to dlsca-attack-v2")
    p.add_argument("--attacker-run", required=True,
                    help="frozen attacker run dir (for data.path + split_indices.npz)")
    p.add_argument("--gan-defended-traces", required=True,
                    help=".npy of an already GAN-defended array (E-set for --target e, "
                         "or the full Profiling_traces for --target profiling, e.g. D04's "
                         "defended_traces.npy or defended_profiling_traces.npy)")
    p.add_argument("--target", choices=["e", "profiling"], default="e",
                    help="e: combine the attacker's E-set (for a static A0 evaluation, cost "
                         "measured against clean E). profiling: combine the full 50000-trace "
                         "Profiling_traces (to retrain an adaptive A1 attacker on -- no cost "
                         "accounting, this array isn't evaluated directly).")
    p.add_argument("--max-shift", type=int, default=20, help="jamming shift range")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="defenses_d08_combo")
    return p.parse_args()


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()

    preprocess = _load_attack_module(args.attack_repo, "src/data/preprocess.py", "dlsca_attack_preprocess")
    perturbation = _load_attack_module(args.attack_repo, "src/metrics/perturbation.py", "dlsca_attack_perturbation")

    attacker_run = Path(args.attacker_run)
    with open(attacker_run / "config_snapshot.yaml") as f:
        cfg = yaml.safe_load(f)

    data_path = Path(cfg["data"]["path"])
    if not data_path.is_absolute():
        data_path = Path(args.attack_repo) / data_path

    print(f"=== loading {data_path} ===")
    gan_defended = np.load(args.gan_defended_traces).astype(np.float32)

    if args.target == "e":
        with h5py.File(data_path, "r") as f:
            attack_traces = np.array(f["Attack_traces/traces"], dtype=np.int8)
        with np.load(attacker_run / "split_indices.npz") as split:
            e_idx = split["e"]
        clean_ref = attack_traces[e_idx].astype(np.float32)
        label = f"D04_jam{args.max_shift}"
    else:
        with h5py.File(data_path, "r") as f:
            clean_ref = np.array(f["Profiling_traces/traces"], dtype=np.int8).astype(np.float32)
        label = f"D04_jam{args.max_shift}_profiling"

    assert gan_defended.shape == clean_ref.shape, (
        f"GAN-defended array shape {gan_defended.shape} != clean reference shape {clean_ref.shape} "
        f"(target={args.target})"
    )

    print(f"=== applying jamming (max_shift={args.max_shift}) on top of GAN-defended {args.target} set ===")
    combined = preprocess.jamming_augment(gan_defended, max_shift=args.max_shift, seed=args.seed)

    out_dir = Path(args.out_dir) / f"{label}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.target == "e":
        cost = {
            "defense": "D04+jamming",
            "params": {"max_shift": args.max_shift, "seed": args.seed,
                        "gan_defended_source": str(args.gan_defended_traces)},
            "psr": summarize(perturbation.psr(clean_ref, combined)),
            "l2": summarize(perturbation.l2(clean_ref, combined)),
            "linf": summarize(perturbation.linf(clean_ref, combined)),
        }
        print(f"  PSR mean={cost['psr']['mean']:.4f}  L2 mean={cost['l2']['mean']:.2f}")
        traces_path = out_dir / "combined_traces.npy"
        np.save(traces_path, combined.astype(np.float32))
        (out_dir / "cost_metrics.json").write_text(json.dumps(cost, indent=2))
        print(f"=== saved {traces_path} + cost_metrics.json ===")
        print()
        print("=== next steps ===")
        print(f"  python3 <dlsca-attack-v2>/scripts/02_run_attack.py --run {attacker_run} "
              f"--traces {traces_path} --out {out_dir / 'probs.npy'}")
        print(f"  python3 <dlsca-attack-v2>/scripts/03_evaluate.py --run {attacker_run} "
              f"--probs {out_dir / 'probs.npy'} --out {out_dir / 'metrics.json'} --override attack.max_traces=9000")
        print(f"  python3 scripts/07_check_resync_bypass.py --attack-repo <dlsca-attack-v2> "
              f"--attacker-run {attacker_run} --jammed-traces {traces_path} --max-shift {args.max_shift}")
    else:
        traces_path = out_dir / "combined_profiling_traces.npy"
        np.save(traces_path, combined.astype(np.float32))
        print(f"=== saved {traces_path} (50000 traces, no cost accounting -- not evaluated directly) ===")
        print()
        print("=== next: retrain an adaptive attacker on this (run from <dlsca-attack-v2> root) ===")
        print("  python3 scripts/01_train_attacker.py --config configs/exp/E01_adaptive_A1_retrain.yaml "
              f"--profiling-traces {traces_path} --runs-dir runs_adaptive_sweep")


if __name__ == "__main__":
    main()
