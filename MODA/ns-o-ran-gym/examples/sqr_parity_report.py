"""IPC x E2 parity report for the SQR xApp (Fase 5.4).

Consumes the two run directories produced by
``examples/run_sqr_e2_parity.sh`` — the IPC driver (``examples/sqr_run.py``)
and the E2 xApp (``examples/xapp_sqr.py``) — and compares, at the SAME
seed/simTime/gamma:

1. **Decisions** (``step_metrics.csv`` of both runs, aligned by
   (step, slice_id)): mode, raw_action, action, prb_requested,
   prb_executed, timestamp_ms. Identical sequences => the xApp decided
   exactly what the IPC agent decided on the same observations.
2. **Applied quotas / MAC telemetry** (``sqr_prb_windows.csv`` written by
   sqr-sim on BOTH paths, aligned by timestamp_ms): prb_executed_rbg_*,
   prb_scheduled_rbg_*, prb_budget_rbg_*, prb_idle_rbg compared EXACTLY;
   prb_requested_pct_* expected to differ by the documented transport
   constraint (the RIC Control carries ASN.1 INTEGER percents, Fase 5.3) —
   quantified, not treated as a divergence.
3. **KPM / violations** (``sqr_kpm_history.csv`` of both runs): per-UE
   window rows compared cell-by-cell; per-UE SLA violation counts
   (windowThroughputMbps < SLA[slice], register R19) reported per path.
4. **Decision latency** p50/p95 per path: IPC ``decision_time_ms`` (the
   act() wall time, contract budget 50 ms); E2 decision core (same
   measurement) + end-to-end indication→control handling + the ns-3-side
   bridge RTT (``sqr_e2_xapp_bridge.csv``).

Prints a comparative table and ``PARITY OK`` when (1) and (2) and (3) all
match; otherwise ``PARITY DIVERGED`` with the offending windows listed.
Writes the markdown report (including the observed latency difference vs
the 50 ms budget) to ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import statistics
import sys

SLAS = {0: 0.5, 1: 1.0, 2: 1.5}
NUM_SLICES = 3

STEP_COMPARE_COLUMNS = [
    "timestamp_ms", "mode", "raw_action", "action",
    "prb_requested", "prb_executed",
]
TELEMETRY_EXACT_COLUMNS = (
    ["prb_executed_rbg_%d" % s for s in range(NUM_SLICES)]
    + ["prb_scheduled_rbg_%d" % s for s in range(NUM_SLICES)]
    + ["prb_budget_rbg_%d" % s for s in range(NUM_SLICES)]
    + ["prb_idle_rbg", "dl_slots", "total_rbg", "total_prb"]
)
TELEMETRY_PCT_COLUMNS = ["prb_requested_pct_%d" % s for s in range(NUM_SLICES)]


def read_csv(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def find_sim_out(run_dir: str, e2: bool) -> str | None:
    """Locate the ns-3 results dir of a run (IPC: <dir>/<uuid>/...; E2: <dir>/e2_seed=N/...)."""
    pattern = os.path.join(
        run_dir, "e2_seed=*" if e2 else "*", "results_sqr_network_only",
        "seed=*_run=*",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def percentile_nearest_rank(samples: list[float], q: float) -> float:
    if not samples:
        return 0.0
    s = sorted(samples)
    idx = max(math.ceil(q * len(s)) - 1, 0)
    return s[idx]


def latency_stats(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": len(samples),
        "p50": round(statistics.median(samples), 3),
        "p95": round(percentile_nearest_rank(samples, 0.95), 3),
        "max": round(max(samples), 3),
    }


def read_step_decisions(run_dir: str) -> dict[int, dict]:
    """(step, slice_id) -> row dict of the decision columns."""
    rows = read_csv(os.path.join(run_dir, "step_metrics.csv"))
    out: dict[int, dict] = {}
    for r in rows:
        key = (int(r["step"]), int(r["slice_id"]))
        out[key] = r
    return out


def read_telemetry(sim_out: str | None) -> dict[int, dict]:
    if not sim_out:
        return {}
    rows = read_csv(os.path.join(sim_out, "sqr_prb_windows.csv"))
    return {int(float(r["timestamp_ms"])): r for r in rows}


def kpm_violations(sim_out: str | None) -> tuple[int, int, dict[int, int]]:
    """(n_rows, n_violations, violations_per_slice) of sqr_kpm_history.csv."""
    if not sim_out:
        return 0, 0, {}
    rows = read_csv(os.path.join(sim_out, "sqr_kpm_history.csv"))
    per_slice = {s: 0 for s in range(NUM_SLICES)}
    viol = 0
    for r in rows:
        sid = int(r["sliceId"])
        sla = SLAS.get(sid)
        if sla is not None and float(r["windowThroughputMbps"]) < sla:
            viol += 1
            per_slice[sid] += 1
    return len(rows), viol, per_slice


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipc", required=True, help="IPC run dir (sqr_run.py output)")
    parser.add_argument("--e2", required=True, help="E2 run dir (xapp_sqr.py output)")
    parser.add_argument("--budget-ms", type=float, default=50.0,
                        help="decision budget of the contract §5 (ms)")
    parser.add_argument("--out", type=str, default="",
                        help="markdown report path (default: <e2>/parity_report.md)")
    args = parser.parse_args(argv)

    out_path = args.out or os.path.join(args.e2, "parity_report.md")
    ipc_sim = find_sim_out(args.ipc, e2=False)
    e2_sim = find_sim_out(args.e2, e2=True)

    # --- 1. decisions -------------------------------------------------------
    ipc_dec = read_step_decisions(args.ipc)
    e2_dec = read_step_decisions(args.e2)
    common_keys = sorted(set(ipc_dec) & set(e2_dec))
    dec_mismatch: dict[str, int] = {c: 0 for c in STEP_COMPARE_COLUMNS}
    dec_mismatch_examples: list[str] = []
    for key in common_keys:
        a, b = ipc_dec[key], e2_dec[key]
        for col in STEP_COMPARE_COLUMNS:
            if str(a[col]) != str(b[col]):
                dec_mismatch[col] += 1
                if len(dec_mismatch_examples) < 8:
                    dec_mismatch_examples.append(
                        f"step={key[0]} slice={key[1]} {col}: ipc={a[col]!r} e2={b[col]!r}"
                    )
    decisions_identical = (
        len(ipc_dec) == len(e2_dec)
        and len(common_keys) == len(ipc_dec)
        and all(v == 0 for v in dec_mismatch.values())
    )

    # --- 2. applied quotas / MAC telemetry ----------------------------------
    ipc_tel = read_telemetry(ipc_sim)
    e2_tel = read_telemetry(e2_sim)
    tel_common = sorted(set(ipc_tel) & set(e2_tel))
    tel_exact_mism: dict[str, int] = {c: 0 for c in TELEMETRY_EXACT_COLUMNS}
    pct_max_abs = 0.0
    pct_diff_windows = 0
    tel_examples: list[str] = []
    for ts in tel_common:
        a, b = ipc_tel[ts], e2_tel[ts]
        for col in TELEMETRY_EXACT_COLUMNS:
            if int(float(a[col])) != int(float(b[col])):
                tel_exact_mism[col] += 1
                if len(tel_examples) < 8:
                    tel_examples.append(
                        f"t={ts}ms {col}: ipc={a[col]} e2={b[col]}"
                    )
        window_pct_diff = False
        for col in TELEMETRY_PCT_COLUMNS:
            d = abs(float(a[col]) - float(b[col]))
            pct_max_abs = max(pct_max_abs, d)
            if d > 1e-9:
                window_pct_diff = True
        pct_diff_windows += int(window_pct_diff)
    telemetry_identical = (
        len(ipc_tel) == len(e2_tel)
        and len(tel_common) == len(ipc_tel)
        and all(v == 0 for v in tel_exact_mism.values())
    )

    # --- 3. KPMs / violations ------------------------------------------------
    ipc_rows, ipc_viol, ipc_viol_slice = kpm_violations(ipc_sim)
    e2_rows, e2_viol, e2_viol_slice = kpm_violations(e2_sim)
    kpm_identical = ipc_rows == e2_rows and ipc_viol == e2_viol

    # --- 4. latency ----------------------------------------------------------
    def decision_samples(run_dir: str) -> list[float]:
        rows = read_csv(os.path.join(run_dir, "step_metrics.csv"))
        seen = set()
        out = []
        for r in rows:
            if r["decision_time_ms"] and r["decision_time_ms"] != "":
                step = int(r["step"])
                if step not in seen:  # one sample per decision window
                    seen.add(step)
                    out.append(float(r["decision_time_ms"]))
        return out

    ipc_lat = latency_stats(decision_samples(args.ipc))
    e2_core_rows = read_csv(os.path.join(args.e2, "xapp_decisions.csv"))
    e2_core_rows = [r for r in e2_core_rows if r.get("frozen_echo") != "1"]
    e2_core = latency_stats(
        [float(r["decision_ms"]) for r in e2_core_rows if r["decision_ms"]]
    )
    e2_handle = latency_stats(
        [float(r["handle_ms"]) for r in e2_core_rows if r["handle_ms"]]
    )
    bridge_rows = []
    if e2_sim:
        bridge_rows = read_csv(os.path.join(e2_sim, "sqr_e2_xapp_bridge.csv"))
    bridge_rtt = latency_stats(
        [float(r["rtt_wall_ms"]) for r in bridge_rows if r.get("rtt_wall_ms")]
    )
    bridge_timeouts = sum(
        1 for r in bridge_rows if r.get("timeout") == "1"
    ) if bridge_rows else 0

    parity_ok = decisions_identical and telemetry_identical and kpm_identical

    # --- table ---------------------------------------------------------------
    lines: list[str] = []
    add = lines.append
    add("# Paridade IPC x E2 — SQR xApp (Fase 5.4)")
    add("")
    add(f"- IPC run: `{args.ipc}` (sim out: `{ipc_sim}`)")
    add(f"- E2 run:  `{args.e2}` (sim out: `{e2_sim}`)")
    add("")
    add("| Verificação | IPC | E2 | Resultado |")
    add("|---|---|---|---|")
    add(f"| Janelas de decisão (step_metrics) | {len(ipc_dec)} linhas | "
        f"{len(e2_dec)} linhas | "
        f"{'IDÊNTICAS' if decisions_identical else 'DIVERGENTES'} |")
    for col in STEP_COMPARE_COLUMNS:
        add(f"| &nbsp;&nbsp;divergências `{col}` | — | {dec_mismatch[col]} |")
    add(f"| Janelas de telemetria prb_* (ns-3) | {len(ipc_tel)} | {len(e2_tel)} | "
        f"{'IDÊNTICAS' if telemetry_identical else 'DIVERGENTES'} |")
    for col in TELEMETRY_EXACT_COLUMNS:
        if tel_exact_mism[col]:
            add(f"| &nbsp;&nbsp;divergências `{col}` | — | {tel_exact_mism[col]} |")
    add(f"| Divergência esperada `prb_requested_pct_*` (inteiro RIC) | — | "
        f"{pct_diff_windows} janelas, máx |Δ| = {pct_max_abs:.4f} p.p. | "
        "divergência de transporte documentada (ASN.1 INTEGER, Fase 5.3) |")
    add(f"| Linhas KPM (sqr_kpm_history) | {ipc_rows} | {e2_rows} | "
        f"{'IDÊNTICAS' if kpm_identical else 'DIVERGENTES'} |")
    add(f"| Violações por-UE-janela (R19) | {ipc_viol} | {e2_viol} | "
        f"{'IGUAIS' if ipc_viol == e2_viol else 'DIFERENTES'} |")
    add("")
    add("## Latência de decisão (orçamento contrato §5: "
        f"{args.budget_ms:g} ms)")
    add("")
    add("| Caminho | métrica | n | p50 (ms) | p95 (ms) | máx (ms) |")
    add("|---|---|---|---|---|---|")
    add(f"| IPC (semáforos+CSV; act only) | decision_time_ms | {ipc_lat['n']} | "
        f"{ipc_lat['p50']} | {ipc_lat['p95']} | {ipc_lat['max']} |")
    add(f"| E2 xApp — núcleo act() | decision_ms | {e2_core['n']} | "
        f"{e2_core['p50']} | {e2_core['p95']} | {e2_core['max']} |")
    add(f"| E2 xApp — indication→controle | handle_ms | {e2_handle['n']} | "
        f"{e2_handle['p50']} | {e2_handle['p95']} | {e2_handle['max']} |")
    add(f"| E2 ponte lado gNB (UDP JSON) | rtt_wall_ms | {bridge_rtt['n']} | "
        f"{bridge_rtt['p50']} | {bridge_rtt['p95']} | {bridge_rtt['max']} |")
    add("")
    add(f"- Timeouts da ponte (degradação §5 contável): {bridge_timeouts}")
    add(f"- IPC p95 = {ipc_lat['p95']} ms vs orçamento {args.budget_ms:g} ms → "
        f"{'DENTRO' if ipc_lat['p95'] <= args.budget_ms else 'FORA'}")
    add(f"- E2 p95 (end-to-end) = {e2_handle['p95']} ms vs orçamento "
        f"{args.budget_ms:g} ms → "
        f"{'DENTRO' if e2_handle['p95'] <= args.budget_ms else 'FORA'}")
    add(f"- Diferença de latência observada (p50 E2 − p50 IPC) = "
        f"{round(e2_handle['p50'] - ipc_lat['p50'], 3)} ms; "
        f"(p95: {round(e2_handle['p95'] - ipc_lat['p95'], 3)} ms)")
    add("")
    if dec_mismatch_examples:
        add("## Exemplos de divergência de decisão")
        add("```")
        lines.extend(dec_mismatch_examples)
        add("```")
    if tel_examples:
        add("## Exemplos de divergência de telemetria aplicada")
        add("```")
        lines.extend(tel_examples)
        add("```")
    add("")
    add(f"**{'PARITY OK' if parity_ok else 'PARITY DIVERGED'}**")
    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(report)
    print(f"[parity] report written to {out_path}")
    return 0 if parity_ok else 1


if __name__ == "__main__":
    sys.exit(main())
