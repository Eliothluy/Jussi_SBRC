#!/usr/bin/env python3
"""Cross-evaluate each seed's best weight vector on every other seed (10x10).

WHY THIS EXISTS
---------------
The v2 campaign found that the best weight vector varies enormously across
seeds: in `stressed` the optimal URLLC share spans 0.050 to 0.546 -- nearly half
the simplex -- and the ordering is not monotone (seed 2 is feasible at URLLC
0.082 while seed 3 is infeasible at 0.199). Two very different explanations fit
that observation equally well, and the campaign data cannot separate them
because each best candidate was only ever evaluated on the seed that produced it:

  (H1) Per-realization optimum. Each channel realization genuinely has its own
       optimum, so no fixed weight vector can serve all of them. This is the
       claim the paper wants to make -- it is precisely why fixed-weight
       baselines (P-STA equal, Weighted RR/PF/BCQI) cap out at <=6/10 seeds.

  (H2) Flat / multimodal objective. The landscape has many near-equivalent
       optima, and different RNG trajectories simply land in different ones. The
       dispersion would then be an artifact of the search, not a property of the
       channel, and the paper's central claim would not follow.

This script runs the discriminating experiment. Take the 10 best vectors (one
per seed) and evaluate all of them on all 10 seeds, scoring every cell through
the same objective (`score_summary_rows_v2`). The two hypotheses predict
different matrices:

  H1 -> the diagonal dominates: vector_i scores best on seed_i. Off-diagonal
        cells degrade, and the mean diagonal-minus-off-diagonal gap is positive
        and large relative to the within-column spread.
  H2 -> the matrix is homogeneous: a vector that is good on its own seed is
        about as good everywhere, so the diagonal carries no advantage.

`--analyze-only` reports that gap plus, for each evaluation seed, whether the
"native" vector actually won -- the direct count a reviewer will ask for.

Simulation parameters mirror the campaign exactly (slice_custom, PF intra-slice,
5 s, 10 parallel jobs) so scores are directly comparable to the campaign's; any
divergence here would silently invalidate the comparison.

Resume-friendly: a cell whose summary.csv already exists is skipped, so this can
be interrupted and relaunched (the campaign taught us to assume reboots).

Usage:
    cd ns-o-ran-gym
    python3 examples/cross_evaluate_best_weights.py --scenarios stressed
    python3 examples/cross_evaluate_best_weights.py --scenarios stressed --analyze-only
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from nsoran.scoring import read_summary, score_summary_rows_v2  # noqa: E402
from run_rslaq_metaheuristics import build_sim_command, summary_path  # noqa: E402

DEFAULT_META_ROOT = Path(
    "results_controlled/heuristics_metaheuristics_v2/v2_all_scenarios/metaheuristics")
PILOT_META_ROOT = Path(
    "results_controlled/heuristics_metaheuristics_v2/v2_pilot_congestion/metaheuristics")


def best_weights_per_seed(meta_roots: list[Path], scenario: str) -> dict[int, list[float]]:
    """The highest-scoring non-failed candidate per seed, re-scored under v2."""
    best: dict[int, tuple[float, list[float]]] = {}
    for root in meta_roots:
        pattern = str(root / f"scenario={scenario}" / "**" / "candidate.json")
        for sidecar in glob.glob(pattern, recursive=True):
            payload = json.loads(Path(sidecar).read_text(encoding="utf-8"))
            if payload.get("failed"):
                continue
            summaries = glob.glob(
                os.path.join(os.path.dirname(sidecar), "**", "summary.csv"), recursive=True)
            if not summaries:
                continue
            score = score_summary_rows_v2(read_summary(Path(summaries[0])))
            seed = int(payload["seed"])
            if seed not in best or score > best[seed][0]:
                best[seed] = (score, [float(w) for w in payload["weights"]])
    return {seed: weights for seed, (_, weights) in sorted(best.items())}


def run_cell(args, scenario: str, weight_seed: int, weights: list[float],
             eval_seed: int) -> dict:
    """Evaluate one (weight vector, seed) cell; reuse an existing result if present."""
    cell_root = (args.output_root / f"scenario={scenario}"
                 / f"wseed={weight_seed}" / f"eseed={eval_seed}")
    out_summary = summary_path(cell_root, scenario, eval_seed, args.run)
    row = {"scenario": scenario, "weight_seed": weight_seed, "eval_seed": eval_seed,
           "w_eMBB": weights[0], "w_URLLC": weights[1], "w_MTC": weights[2],
           "native": weight_seed == eval_seed}

    if not out_summary.exists():
        cell_root.mkdir(parents=True, exist_ok=True)
        command = build_sim_command(
            scenario=scenario, output_root=cell_root, weights=weights,
            intra_algo=args.intra_algo, sim_time=args.sim_time, app_start=args.app_start,
            drain_time=args.drain_time, period_ms=args.period_ms, seed=eval_seed,
            run=args.run, tx_power=args.tx_power, tdd_pattern=args.tdd_pattern,
            rlc_mode=args.rlc_mode)
        with (cell_root / "ns3.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(["./ns3", "run", command], cwd=args.ns3_dir,
                                       stdout=log, stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0 or not out_summary.exists():
            # A crashed cell is recorded as missing rather than penalized: unlike
            # the search, nothing here consumes the score, so inventing one would
            # only contaminate the diagonal-vs-off-diagonal statistics.
            row.update({"score": "", "feasible": "", "failed": True})
            return row

    score = score_summary_rows_v2(read_summary(out_summary))
    row.update({"score": score, "feasible": score >= 0.0, "failed": False})
    return row


def scan_completed_cells(output_root: Path, scenario: str,
                         weights_by_seed: dict[int, list[float]]) -> list[dict]:
    """Rebuild result rows from the summaries already on disk.

    The CSV is only written when a full run finishes, so an interrupted or
    still-running sweep would otherwise be unanalyzable. Scanning the tree makes
    partial results usable immediately -- and recovers a run whose CSV was lost.
    """
    rows = []
    pattern = str(output_root / f"scenario={scenario}" / "wseed=*" / "eseed=*"
                  / "**" / "summary.csv")
    for summ in glob.glob(pattern, recursive=True):
        parts = Path(summ).parts
        try:
            weight_seed = int(next(p for p in parts if p.startswith("wseed=")).split("=")[1])
            eval_seed = int(next(p for p in parts if p.startswith("eseed=")).split("=")[1])
        except StopIteration:
            continue
        score = score_summary_rows_v2(read_summary(Path(summ)))
        weights = weights_by_seed.get(weight_seed, [float("nan")] * 3)
        rows.append({"scenario": scenario, "weight_seed": weight_seed,
                     "eval_seed": eval_seed, "w_eMBB": weights[0],
                     "w_URLLC": weights[1], "w_MTC": weights[2],
                     "native": weight_seed == eval_seed, "score": score,
                     "feasible": score >= 0.0, "failed": False})
    return rows


def analyze(rows: list[dict], scenario: str) -> None:
    cells = [r for r in rows if r["scenario"] == scenario and r.get("score") not in ("", None)]
    if not cells:
        print(f"  {scenario}: nenhuma célula avaliada ainda")
        return
    scores = {(int(r["weight_seed"]), int(r["eval_seed"])): float(r["score"]) for r in cells}
    seeds = sorted({s for _, s in scores})

    print(f"\n{'=' * 78}\n{scenario}: matriz cruzada ({len(scores)} células)\n{'=' * 78}")
    print("  linha = vetor de origem, coluna = seed avaliada; * = célula nativa")
    header = "".join(f"{s:>9}" for s in seeds)
    print(f"  {'vetor':<8}{header}")
    for ws in seeds:
        cells_txt = ""
        for es in seeds:
            v = scores.get((ws, es))
            mark = "*" if ws == es else " "
            cells_txt += f"{v:>8.1f}{mark}" if v is not None else f"{'—':>8} "
        print(f"  seed{ws:<4}{cells_txt}")

    diag = [scores[(s, s)] for s in seeds if (s, s) in scores]
    off = [v for (ws, es), v in scores.items() if ws != es]
    if not diag or not off:
        return
    print(f"\n  diagonal   : mediana {st.median(diag):>9.1f}  (n={len(diag)})")
    print(f"  fora-diag. : mediana {st.median(off):>9.1f}  (n={len(off)})")
    print(f"  gap        : {st.median(diag) - st.median(off):>+9.1f}")

    # The count a reviewer asks for: on each evaluation seed, did the native
    # vector actually beat the nine foreign ones?
    native_wins = 0
    comparable = 0
    for es in seeds:
        column = {ws: v for (ws, e), v in scores.items() if e == es}
        if (es, es) not in scores or len(column) < 2:
            continue
        comparable += 1
        if scores[(es, es)] >= max(column.values()):
            native_wins += 1
    print(f"  vetor nativo é o melhor da sua coluna: {native_wins}/{comparable} seeds")
    print("\n  H1 (ótimo por realização) prevê gap positivo e nativo vencendo;")
    print("  H2 (paisagem plana) prevê gap ~0 e nativo indistinguível dos demais.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["stressed"])
    ap.add_argument("--meta-root", type=Path, default=DEFAULT_META_ROOT)
    ap.add_argument("--pilot-meta-root", type=Path, default=PILOT_META_ROOT,
                    help="Second search root (the congestion pilot lives apart).")
    ap.add_argument("--output-root", type=Path,
                    default=Path("results_controlled/heuristics_metaheuristics_v2/v2_cross_eval"))
    ap.add_argument("--ns3-dir", type=Path,
                    default=Path("/home/eliothluy/Documentos/artigo_jussi/ns-3-dev"))
    ap.add_argument("--parallel-jobs", type=int, default=10)
    ap.add_argument("--analyze-only", action="store_true")
    # Campaign-identical simulation parameters; changing one breaks comparability.
    ap.add_argument("--intra-algo", default="PF")
    ap.add_argument("--sim-time", type=float, default=5.0)
    ap.add_argument("--app-start", type=float, default=0.4)
    ap.add_argument("--drain-time", type=float, default=0.2)
    ap.add_argument("--period-ms", type=int, default=10)
    ap.add_argument("--tx-power", type=float, default=43.0)
    ap.add_argument("--tdd-pattern", default="D|D|8D|4GB|4U|U|U")
    ap.add_argument("--rlc-mode", default="um")
    ap.add_argument("--run", type=int, default=1)
    args = ap.parse_args()

    # ns-3 runs with cwd=ns3_dir, so a relative --outputDir resolves against the
    # ns-3 tree instead of this one: every result lands somewhere the resume
    # check never looks, and every cell is then recorded as a crash. Resolve
    # before building any command.
    args.output_root = args.output_root.resolve()
    args.ns3_dir = args.ns3_dir.resolve()

    args.output_root.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_root / "cross_eval_results.csv"

    if args.analyze_only:
        if results_csv.exists():
            with results_csv.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        else:
            print(f"({results_csv.name} ainda não existe — varrendo o disco)")
            meta_roots = [args.meta_root.resolve(), args.pilot_meta_root.resolve()]
            rows = []
            for scenario in args.scenarios:
                rows += scan_completed_cells(
                    args.output_root, scenario,
                    best_weights_per_seed(meta_roots, scenario))
        for scenario in args.scenarios:
            analyze(rows, scenario)
        return

    meta_roots = [args.meta_root, args.pilot_meta_root]
    tasks = []
    for scenario in args.scenarios:
        best = best_weights_per_seed(meta_roots, scenario)
        if not best:
            print(f"[AVISO] {scenario}: nenhum candidato encontrado, pulando")
            continue
        print(f"[{scenario}] {len(best)} vetores x {len(best)} seeds = "
              f"{len(best) ** 2} células")
        for weight_seed, weights in best.items():
            print(f"  seed{weight_seed}: "
                  f"[{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]")
            for eval_seed in best:
                tasks.append((scenario, weight_seed, weights, eval_seed))

    if not tasks:
        return
    print(f"\nTotal de células: {len(tasks)} | jobs paralelos: {args.parallel_jobs}")

    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.parallel_jobs) as pool:
        futures = [pool.submit(run_cell, args, *t) for t in tasks]
        for future in futures:
            rows.append(future.result())
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} células concluídas", flush=True)

    rows.sort(key=lambda r: (r["scenario"], r["weight_seed"], r["eval_seed"]))
    with results_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSalvo: {results_csv}")

    for scenario in args.scenarios:
        analyze(rows, scenario)


if __name__ == "__main__":
    main()
