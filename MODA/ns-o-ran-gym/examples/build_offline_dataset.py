#!/usr/bin/env python3
"""Consolidate metaheuristic + heuristic results into a Parquet dataset.

Thin CLI over nsoran.offline_dataset. Produces one flat contextual-bandit
table (context, action=slice weights, reward=score) combining:
  - all metaheuristic search evaluations (candidate.json sidecars), and
  - all heuristic/baseline runs (scored with the same nsoran.scoring objective;
    pure_* modes are always excluded — see nsoran/offline_dataset.py).

The default output is offline_dataset_v2.parquet. The original
offline_dataset.parquet (v1, metaheuristics only) is intentionally left
untouched so previously trained checkpoints remain reproducible.

Usage:
    cd ns-o-ran-gym
    python3 examples/build_offline_dataset.py
    python3 examples/build_offline_dataset.py --no-heuristics --output /tmp/ds.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nsoran.offline_dataset import SCENARIOS, SEEDS, build_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    default_data_root = REPO_ROOT / "resultados_cenarios_finalizados_20260715"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=default_data_root,
                        help="Finalized results folder (default: %(default)s)")
    parser.add_argument("--meta-root", type=Path, default=None,
                        help="Metaheuristics tree (default: <data-root>/metaheuristics)")
    parser.add_argument("--heuristics-root", type=Path, default=None,
                        help="Heuristic runs tree (default: <data-root>/heuristics_ns3/results_rslaq_network_only)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output parquet (default: <data-root>/offline_dataset_v2.parquet)")
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--no-heuristics", action="store_true",
                        help="Only ingest metaheuristic evals (v1-equivalent content)")
    parser.add_argument("--exclude-dynamic-modes", action="store_true",
                        help="Drop dynamic-weight heuristic modes (keep only static-weight ones)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_root = args.meta_root or args.data_root / "metaheuristics"
    heuristics_root = args.heuristics_root or (
        args.data_root / "heuristics_ns3" / "results_rslaq_network_only")
    output = args.output or args.data_root / "offline_dataset_v2.parquet"

    df, stats = build_dataset(
        meta_root=meta_root,
        heuristics_root=heuristics_root,
        scenarios=args.scenarios,
        seeds=args.seeds,
        include_heuristics=not args.no_heuristics,
        include_dynamic_modes=not args.exclude_dynamic_modes,
    )

    if df.empty:
        print("No rows produced — check the input paths.", file=sys.stderr)
        sys.exit(1)

    print(f"Total rows: {len(df)} | skips: {stats or 'none'}")
    for source, count in df["source"].value_counts().items():
        print(f"  source={source}: {count}")
    print(f"Scenarios: {sorted(df['scenario'].unique())}")
    print(f"Score range: {df['score'].min():.2f} to {df['score'].max():.2f} "
          f"(mean {df['score'].mean():.2f} +/- {df['score'].std():.2f})")

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"\nParquet saved: {output} ({output.stat().st_size / 1024:.0f} KB)")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
