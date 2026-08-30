#!/usr/bin/env python3
"""Stack D04's leakage-aware amplitude perturbation with D08's time-domain
jamming, to test the future-work hypothesis (CLAUDE.md 附錄N/report_draft.md
6.3 item 3): does a real information-destroying defense (D04) survive resync
better when combined with jamming, than jamming alone does (附錄N: jamming
alone at max_shift=20 is fully reversed by resync)?

Applies jamming_augment on top of an already-GAN-defended E-set (D04's
defended_traces.npy), not on raw clean traces -- the two mechanisms stack,
they don't replace each other. Cost is measured against the true clean E set
(not against the GAN-defended traces), consistent with how every other
defense in this project reports PSR.

Usage (run from repo root):
    python scripts/08_combine_defenses.py \
        --attack-repo ~/dlsca-attack-v2 \
        --attacker-run ~/dlsca-attack-v2/runs/E01_baseline_clean_20260816_1302 \
        --gan-defended-traces runs/D04_snr_aware_.../defended_traces.npy \
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
                    help=".npy of an already GAN-defended E-set (e.g. D04's defended_traces.npy)")
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
    with h5py.File(data_path, "r") as f:
        attack_traces = np.array(f["Attack_traces/traces"], dtype=np.int8)
    with np.load(attacker_run / "split_indices.npz") as split:
        e_idx = split["e"]
    clean_e = attack_traces[e_idx].astype(np.float32)

    gan_defended = np.load(args.gan_defended_traces).astype(np.float32)
    assert gan_defended.shape == clean_e.shape, (
        f"GAN-defended traces shape {gan_defended.shape} != clean E shape {clean_e.shape}"
    )

    print(f"=== applying jamming (max_shift={args.max_shift}) on top of GAN-defended traces ===")
    combined = preprocess.jamming_augment(gan_defended, max_shift=args.max_shift, seed=args.seed)

    cost = {
        "defense": "D04+jamming",
        "params": {"max_shift": args.max_shift, "seed": args.seed,
                    "gan_defended_source": str(args.gan_defended_traces)},
        "psr": summarize(perturbation.psr(clean_e, combined)),
        "l2": summarize(perturbation.l2(clean_e, combined)),
        "linf": summarize(perturbation.linf(clean_e, combined)),
    }
    print(f"  PSR mean={cost['psr']['mean']:.4f}  L2 mean={cost['l2']['mean']:.2f}")

    out_dir = Path(args.out_dir) / f"D04_jam{args.max_shift}_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
