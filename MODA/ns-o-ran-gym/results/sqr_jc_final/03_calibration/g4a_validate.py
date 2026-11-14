"""G4-A — Calibration-Qualified Feasibility Gate (emenda §22).

FFR/FIR nas classes confidentes; split por seed (§12): fit={1001,1002},
validation={1003}. Config FROZEN: ls=2.0, thr=90 (percentil sigma do FIT).
Replica o loop do sqr_jc_calibrate.main; divergências apenas:
 sigma threshold computado no FIT (não no pool); classificação §46/§47 na
 janela de validação ANTES do update; FFR/FIR sobre outcomes observados.
"""
import sys, os, glob, json, time
sys.path.insert(0, "src"); sys.path.insert(0, "examples")
import numpy as np
import pandas as pd
from sqr_jc_calibrate import (FeatureScaler, SliceOutcome, window_features,
                              state_of, load_run, impute_snr_column,
                              _read_kpm_windows, _read_actions,
                              _urllc_packet_windows_from_df,
                              C_PRB, B_MEMORY)
from nsoran.sqr_jc.kernel_models import MultiQuantileKernelModel

LS = 2.0
THR_PCT = 90
CAL = "/home/eliothluy/Documentos/Jussi_SBRC/results/sqr_jc_final/03_calibration"
OUT = f"{CAL}/g4a"
os.makedirs(OUT, exist_ok=True)

def build(tag):
    kpm, prb, delays = load_run(f"{CAL}/cal_runs/{tag}")
    imputed = impute_snr_column(kpm)
    return {"tag": tag,
            "feats": window_features(kpm, imputed, _read_kpm_windows(kpm),
                                     _urllc_packet_windows_from_df(delays)
                                     if delays is not None else
                                     pd.DataFrame(columns=["win", "latency_urllc_p95_ms",
                                                           "latency_urllc_miss_pct", "urllc_n_pkts"])),
            "acts": _read_actions(prb)}

tags = sorted(os.path.basename(p) for p in glob.glob(f"{CAL}/cal_runs/cal_*"))
fit_tags = [t for t in tags if t.endswith(("s1001", "s1002"))]
val_tags = [t for t in tags if t.endswith("s1003")]
print(f"[g4a] fit={fit_tags}\n[g4a] val={val_tags}", flush=True)

# scaler congelado do CAL (§13 permite todo o CAL)
scaler = FeatureScaler.load_manifest(f"{CAL}/feature_scaler_manifest.yaml")

models = tuple(MultiQuantileKernelModel(dim=len(state_of({}, (0, 0, 0), (0, 0, 0)).feature_x((18, 17, 18), C_PRB, scaler)) if False else 0, memory_size=1) for _ in range(0))  # placeholder
from nsoran.sqr_jc.joint_state import JOINT_STATE_DIM
models = tuple(MultiQuantileKernelModel(dim=JOINT_STATE_DIM + 3, length_scale=LS,
                                        memory_size=B_MEMORY) for _ in range(3))

def run_stream(s, learn=True, thr=None, collect=None):
    prev_a, prev_viol = (0, 0, 0), (0, 0, 0)
    acts = s["acts"].set_index("win")
    for w in sorted(s["feats"]):
        st = state_of(s["feats"][w], prev_a, prev_viol)
        out_w = w + 1
        if out_w not in acts.index or out_w not in s["feats"]:
            continue
        a = acts.loc[out_w]
        action = (int(round(a["a_e"])), int(round(a["a_u"])), int(round(a["a_m"])))
        x = st.feature_x(action, C_PRB, scaler)
        preds = [m.predict_raw(x[None, :]) for m in models]
        margins = SliceOutcome(
            embb_throughput_mbps=s["feats"][out_w].get("kpi_e"),
            urllc_p95_ms=s["feats"][out_w].get("kpi_u_p95"),
            urllc_miss_pct=(s["feats"][out_w]["kpi_u_miss"] * 100.0)
            if s["feats"][out_w].get("kpi_u_miss") is not None else None,
            mmtc_pdr_rolling=s["feats"][out_w].get("kpi_m"),
        ).margins
        if collect is not None and thr is not None:
            cls, oos = [], False
            for si in range(3):
                if preds[si]["sigma"][0] > thr:
                    cls.append("MARGINAL"); oos = True; continue
                q25, q75 = preds[si]["q25"][0], preds[si]["q75"][0]
                cls.append("FEASIBLE" if q25 >= 0 else ("INFEASIBLE" if q75 < 0 else "MARGINAL"))
            joint = "INFEASIBLE" if "INFEASIBLE" in cls else \
                    ("FEASIBLE" if all(c == "FEASIBLE" for c in cls) else "MARGINAL")
            obs_ok = all(m is not None and np.isfinite(m) and m >= 0 for m in margins)
            collect.append({"tag": s["tag"], "win": w, "joint": joint, "oos": oos,
                            "obs_all_ok": obs_ok,
                            "m": [None if m is None or not np.isfinite(m) else float(m) for m in margins]})
        if learn:
            for si in range(3):
                m = margins[si]
                if m is None or not np.isfinite(m):
                    continue
                models[si].update(x, float(m), threshold=0.0)
        viol = tuple(0 if (m is None or m >= 0) else 1 for m in margins)
        prev_a, prev_viol = action, viol

# ---- FIT (seeds 1001/1002) + coleta de sigma ----
sig = []
class _SigHook:
    pass
for t in fit_tags:
    s = build(t)
    # coletar sigmas das predições (primeira passada = aprendendo)
    rows = []
    prev_a, prev_viol = (0, 0, 0), (0, 0, 0)
    acts = s["acts"].set_index("win")
    for w in sorted(s["feats"]):
        st = state_of(s["feats"][w], prev_a, prev_viol)
        out_w = w + 1
        if out_w not in acts.index or out_w not in s["feats"]:
            continue
        a = acts.loc[out_w]
        action = (int(round(a["a_e"])), int(round(a["a_u"])), int(round(a["a_m"])))
        x = st.feature_x(action, C_PRB, scaler)
        preds = [m.predict_raw(x[None, :]) for m in models]
        for si in range(3):
            sig.append(float(preds[si]["sigma"][0]))
        margins = SliceOutcome(
            embb_throughput_mbps=s["feats"][out_w].get("kpi_e"),
            urllc_p95_ms=s["feats"][out_w].get("kpi_u_p95"),
            urllc_miss_pct=(s["feats"][out_w]["kpi_u_miss"] * 100.0)
            if s["feats"][out_w].get("kpi_u_miss") is not None else None,
            mmtc_pdr_rolling=s["feats"][out_w].get("kpi_m"),
        ).margins
        for si in range(3):
            m = margins[si]
            if m is None or not np.isfinite(m):
                continue
            models[si].update(x, float(m), threshold=0.0)
        viol = tuple(0 if (m is None or m >= 0) else 1 for m in margins)
        prev_a, prev_viol = action, viol
thr = float(np.percentile(np.asarray(sig), THR_PCT))
print(f"[g4a] thr = p{THR_PCT}(sigma_fit) = {thr:.6f} (n={len(sig)})", flush=True)

# ---- VALIDAÇÃO (seed 1003): classificar ANTES de cada update (online, §29) ----
val = []
for t in val_tags:
    run_stream(build(t), learn=True, thr=thr, collect=val)
df = pd.DataFrame(val)
df.to_csv(f"{OUT}/g4a_validation_windows.csv", index=False)

conf = df[~df.oos]
F = conf[conf.joint == "FEASIBLE"]; I = conf[conf.joint == "INFEASIBLE"]
FFR = float(100 * (~F.obs_all_ok).mean()) if len(F) else None
FIR = float(100 * (I.obs_all_ok).mean()) if len(I) else None
tbl = {
  "n_total": len(df), "n_confident": len(conf),
  "n_FEASIBLE": len(F), "FEASIBLE_violated": int((~F.obs_all_ok).sum()), "FFR_pct": FFR,
  "n_INFEASIBLE": len(I), "INFEASIBLE_all_ok": int((I.obs_all_ok).sum()), "FIR_pct": FIR,
  "n_MARGINAL": int((conf.joint == "MARGINAL").sum()),
  "n_OOS": int(df.oos.sum()),
  "threshold_sigma": thr, "ls": LS, "thr_pct": THR_PCT,
  "split": {"fit": fit_tags, "validation": val_tags},
}
ok = (FFR is not None and FFR <= 10.0) and (FIR is not None and FIR <= 10.0)
tbl["verdict"] = "PASS" if ok else "STOP"
json.dump(tbl, open(f"{OUT}/g4a_verdict.json", "w"), indent=2)
print(json.dumps(tbl, indent=2))
