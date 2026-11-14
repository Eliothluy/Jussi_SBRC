#!/usr/bin/env python3
"""d3_decide.py — decisões pareadas OFFLINE B0×B1 nos runs de TESTE (seed 2101).

Cuidado 2 (split por RUN): só os 3 runs da seed 2101. Reconstrução
determinística dos estados: SqrKpmParser (watermark) alimentado por um
acumulador crescente do sqr_kpm_history.csv (mesmo formato do IPC — contrato
v1.3), 1 janela por chamada — o mesmo caminho do runtime.

Alinhamento §23 (o mesmo do runtime): na iteração da janela W que FECHA, o
estado pré-decisão é pós-W e o outcome STRICT computado é o DE W — pertence
à ação que GOVERNOU W (decidida no passo anterior). Par correto para resgate:
(a_B0[W]=prev_a, b1[W]=decisão B1 do passo anterior, outcome[W]).

Cuidado 1: hashes do checkpoint antes/depois (modelo congelado).
Saída: d3_test/<run>/decisions_paired.jsonl
"""
import glob
import hashlib
import json
import os
import sys

GYM = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym"
sys.path.insert(0, os.path.join(GYM, "src"))

D1 = os.path.join(GYM, "results/sqr_jc_decision/d1_runs_v9_delayed")
OUT = os.path.join(GYM, "results/sqr_jc_decision/d3_test")
TEST_RUNS = ["het_lam2_s2101", "het_lam5_s2101", "het_lam6_s2101"]


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


D2_STATES = [tuple(x) for x in json.load(open(os.path.join(
    GYM, "results/sqr_jc_decision/d2_states_list.json")))]
D2_EXCL = {}
for _r, _w in D2_STATES:
    for _x in range(_w - 5, _w + 6):
        D2_EXCL.setdefault(_r, set()).add(_x)


def executed_quota(run_dir: str, w: int):
    """Ação EXECUTADA que governou a janela w (autoridade bit-exact D2/A/A):
    quota média por slice da telemetria prb (prb_budget_rbg_i/dl_slots)."""
    import pandas as pd
    pw = sorted(glob.glob(os.path.join(run_dir, "**", "sqr_prb_windows.csv"),
                          recursive=True))
    prb = pd.read_csv(pw[0])
    row = prb[(prb.timestamp_ms // 100 - 1) == w]
    if row.empty or int(row.dl_slots.iloc[0]) == 0:
        return None
    r = row.iloc[0]
    return (int(round(r.prb_budget_rbg_0 / r.dl_slots)),
            int(round(r.prb_budget_rbg_1 / r.dl_slots)),
            int(round(r.prb_budget_rbg_2 / r.dl_slots)))


def strict_of(margins):
    ev = ["COMPLETE" if m is not None else "NO_ACTIVE_DEMAND" for m in margins]
    if "NO_ACTIVE_DEMAND" in ev:
        return "UNKNOWN"
    return "STRICT_PASS" if all(m >= 0 for m in margins) else "STRICT_FAIL"


def process_run(run: str) -> None:
    from nsoran.sqr.sqr_kpis import SqrKpmParser
    from nsoran.sqr_jc.het_runtime import HetRuntime
    from nsoran.sqr_jc.joint_selector import (JointSelector,
                                              load_run_controllers)
    run_dir = os.path.join(D1, run)
    out_path = os.path.join(OUT, run, "decisions_paired.jsonl")
    ck = os.path.join(run_dir, "sqr_checkpoint.pkl")
    if os.path.exists(out_path):
        print(f"[d3] {run}: já pronto")
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ck_before = sha16(ck)
    ctrs, het, meta = load_run_controllers(run_dir)
    js = JointSelector(ctrs, het, c=53)
    kpm = glob.glob(os.path.join(run_dir, "**", "sqr_kpm_history.csv"),
                    recursive=True)[0]
    n_excl = 0
    rt = HetRuntime(os.path.dirname(kpm))
    tmp = os.path.join(OUT, f".kpms_{run}.csv")
    n = 0
    with open(kpm) as src, open(tmp, "w") as acc, \
         open(out_path + ".tmp", "w") as fh:
        header = src.readline()
        acc.write(header)
        parser = SqrKpmParser(num_slices=3)
        cur_ts, buf = None, []
        prev_a_fallback = (0, 0, 0)
        prev_v = (0, 0, 0)
        b1_prev = tr_prev = state_prev = None
        for line in src:
            ts = int(line.split(",", 1)[0])
            if cur_ts is None:
                cur_ts = ts
            if ts != cur_ts:
                acc.writelines(buf)
                acc.flush()
                win = parser.parse_window(tmp)
                w_idx = rt.window_index(win)
                rt.absorb(win)
                prev_a = executed_quota(run_dir, w_idx)  # governou w_idx
                if prev_a is None:
                    prev_a = prev_a_fallback
                margins = rt.outcome(win).margins
                st = rt.joint_state(win, prev_a, prev_v)
                b1_now, tr_now = js.select(st)
                if b1_prev is not None:
                    excl = w_idx in D2_EXCL.get(run, set())
                    n_excl += int(excl)
                    row = {
                        "run": run,
                        "lam": int(run.split("_lam")[1].split("_")[0]),
                        # par da janela w_idx: a_B0 = quota EXECUTADA que a
                        # governou (telemetria, autoridade bit-exact);
                        # b1_prev = decisão B1 do passo anterior (governa
                        # w_idx); outcome = offline completo de w_idx
                        "win": w_idx,
                        "a_B0": list(prev_a),
                        "b1_action": list(b1_prev),
                        "outcome_strict": strict_of(margins),
                        "margins": [None if m is None else float(m)
                                    for m in margins],
                        "n_ues": [st.n_e, st.n_u, st.n_m],
                        "b1_trace": tr_prev,
                        "excluded_d2_pm5": bool(excl),
                    }
                    fh.write(json.dumps(row, default=float) + "\n")
                    n += 1
                prev_a_fallback = (st.prev_a_e, st.prev_a_u, st.prev_a_m)
                prev_v = tuple(0 if (m is None or m >= 0) else 1
                               for m in margins)
                state_prev = st
                b1_prev, tr_prev = b1_now, tr_now
                cur_ts, buf = ts, [line]
            else:
                buf.append(line)
    os.replace(out_path + ".tmp", out_path)
    ck_after = sha16(ck)
    assert ck_before == ck_after, "cuidado 1: checkpoint mudou"
    os.remove(tmp)
    print(f"[d3] {run}: {n} pares ({n_excl} excluídos D2±5) | ckpt hash OK "
          f"({ck_before})", flush=True)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for run in TEST_RUNS:
        process_run(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
