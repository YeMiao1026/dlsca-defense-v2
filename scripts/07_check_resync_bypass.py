#!/usr/bin/env python3
"""D08's "informed attacker counters the known mechanism" check: can an
attacker who knows the deployed defense is time-domain jamming undo it with
the same blind cross-correlation realignment (`resync`) that dlsca-attack-v2
already validated for ASCAD_desync50/100 (CLAUDE.md 附錄 B.30-B.33, where it
recovered the SNR peak from ~0.07 to ~6.8)? This is a preprocessing bypass
applied before inference, not a retrain -- the frozen attacker model is
unchanged, only the input traces are realigned first.

Reference for alignment: the mean of the attacker's own A-set (Profiling
subset), which desync0 databases keep naturally aligned -- a real attacker
in this scenario already has this reference, since it's the same clean
profiling data used to train the frozen model in the first place.

Usage (run from repo root):
    python scripts/07_check_resync_bypass.py \
        --attack-repo ~/dlsca-attack-v2 \
        --attacker-run ~/dlsca-attack-v2/runs/E01_baseline_clean_20260816_1302 \
        --jammed-traces ~/dlsca-attack-v2/defenses_d08/jamming_shift20_.../defended_traces.npy \
        --max-shift 20 --out-dir defenses_d08_resync
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


def _load_attack_module(attack_repo: str, relpath: str, name: str):
    """See CLAUDE.md 附錄 (dlsca_attack_leakage loader in 04_check_tvla.py) for
    why this loads by explicit file path instead of sys.path.insert + `from
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
        description="Realign jamming-defended E-set traces and measure how much attack strength returns"
    )
    p.add_argument("--attack-repo", required=True, help="path to dlsca-attack-v2")
    p.add_argument("--attacker-run", required=True,
                    help="frozen attacker run dir (for data.path + split_indices.npz)")
    p.add_argument("--jammed-traces", required=True, help=".npy of jamming-defended E-set traces")
    p.add_argument("--max-shift", type=int, required=True,
                    help="resync search range; attacker's assumed bound on the jamming defense's shift range")
    p.add_argument("--out-dir", default="defenses_d08_resync")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    resync = _load_attack_module(args.attack_repo, "src/data/resync.py", "dlsca_attack_resync").resync

    attacker_run = Path(args.attacker_run)
    with open(attacker_run / "config_snapshot.yaml") as f:
        cfg = yaml.safe_load(f)

    data_path = Path(cfg["data"]["path"])
    if not data_path.is_absolute():
        data_path = Path(args.attack_repo) / data_path

    print(f"=== loading {data_path} ===")
    with h5py.File(data_path, "r") as f:
        profiling_traces = np.array(f["Profiling_traces/traces"], dtype=np.int8)
    with np.load(attacker_run / "split_indices.npz") as split:
        a_idx = split["a"]

    reference = profiling_traces[a_idx].astype(np.float64).mean(axis=0)
    print(f"=== built reference template from {len(a_idx)} A-set traces (naturally aligned, desync0) ===")

    jammed = np.load(args.jammed_traces).astype(np.float64)
    print(f"=== realigning {len(jammed)} jammed E-set traces (max_shift={args.max_shift}) ===")
    aligned, shifts = resync(jammed, reference, args.max_shift)
    print(f"  shift distribution: mean={shifts.mean():.2f} std={shifts.std():.2f} "
          f"range=[{shifts.min()},{shifts.max()}]")

    out_dir = Path(args.out_dir) / Path(args.jammed_traces).parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "resynced_traces.npy"
    np.save(out_path, aligned.astype(np.float32))
    print(f"=== saved {out_path} ===")
    print()
    print("=== next: re-evaluate through Stage B + re-check TVLA on the realigned traces ===")
    print(f"  python3 <dlsca-attack-v2>/scripts/02_run_attack.py --run {attacker_run} "
          f"--traces {out_path} --out {out_dir / 'probs.npy'}")
    print(f"  python3 <dlsca-attack-v2>/scripts/03_evaluate.py --run {attacker_run} "
          f"--probs {out_dir / 'probs.npy'} --out {out_dir / 'metrics.json'} "
          f"--override attack.max_traces=9000")


if __name__ == "__main__":
    main()
