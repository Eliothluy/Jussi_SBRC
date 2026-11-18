#!/usr/bin/env python3
"""Diagnóstico Exploratório Offline sobre Artefatos Congelados (v1 Limpa).

Executa estritamente em modo somente-leitura sobre os artefatos congelados
de B0-rest (Gate 1 A5) e D1-v9 / D2-v2, gravando todas as saídas exclusivamente
em MODA/ns-o-ran-gym/results/sqr_v2/diag_exploratorio_v1/.

Garantias:
- Zero alterações/gravações fora de results/sqr_v2/diag_exploratorio_v1/.
- Zero carregamentos de sqr_checkpoint.pkl (proibido checkpoint de fim de episódio).
- Zero novas simulações ns-3; zero seeds confirmatórias reservadas.
- Causalidade estrita (observed_at <= decision_at) em todas as reconstruções e índices.
"""

from __future__ import annotations

import csv
import glob
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = "/home/eliothluy/Documentos/Jussi_SBRC"
GYM_DIR = os.path.join(REPO_ROOT, "MODA", "ns-o-ran-gym")
SRC_DIR = os.path.join(GYM_DIR, "src")
DOCS_JC_DIR = os.path.join(REPO_ROOT, "MODA", "docs", "sqr", "jc_decision")
DEC_DIR = os.path.join(GYM_DIR, "results", "sqr_jc_decision")
OUT_DIR = os.path.join(GYM_DIR, "results", "sqr_v2", "diag_exploratorio_v1")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from nsoran.sqr.sqr_controller import SqrSliceController  # noqa: E402
from nsoran.sqr.sqr_kpis import SqrKpmParser, SqrKpmWindow  # noqa: E402
from nsoran.sqr_jc.a5_joint_selector import (  # noqa: E402
    DEFAULT_SIGMA_MAX,
    DEFAULT_SIGMA_SOFT,
    causal_active_slices,
    prop_active_split,
)
from nsoran.sqr_jc.feature_scaler import FeatureScaler  # noqa: E402
from nsoran.sqr_jc.het_runtime import HetRuntime  # noqa: E402
from nsoran.sqr_jc.sla_margin import (  # noqa: E402
    SLA_EMBB_MBPS,
    SLA_MMTC_PDR,
    SLA_URLLC_MISS_PCT_SECONDARY,
    SLA_URLLC_MS,
    embb_margin,
    mmtc_margin,
    urllc_margin,
)
from nsoran.sqr_jc.sqr_het import SqrHetController  # noqa: E402
from nsoran.sqr_jc.yaml_subset import load as yaml_load  # noqa: E402
from nsoran.sqr_v2.manifest_v2 import (  # noqa: E402
    is_reserved_confirmatory_seed,
    verify_b0_frozen_manifest,
)

SLICE_NAMES = {0: "eMBB", 1: "URLLC", 2: "mMTC"}
PKL_OPEN_GUARD_COUNT = 0


def _assert_not_pkl(path: str) -> str:
    if str(path).lower().endswith(".pkl"):
        raise PermissionError(f"Forbidden end-of-episode checkpoint access: {path}")
    return path


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    f = float(v)
    return None if math.isnan(f) else f


def _parse_action_str(s: str) -> tuple[int, int, int]:
    s = str(s).strip()
    if s.startswith("["):
        vals = json.loads(s)
    else:
        vals = [int(x) for x in s.split("-")]
    if len(vals) != 3:
        raise ValueError(f"Invalid 3-slice action string: {s!r}")
    return (int(vals[0]), int(vals[1]), int(vals[2]))


def _find_single(run_dir: str, fname: str) -> str:
    hits = sorted(glob.glob(os.path.join(run_dir, "**", fname), recursive=True))
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected 1 {fname} under {run_dir}, got {len(hits)}")
    return hits[0]


def _pava_nondecreasing(y: np.ndarray) -> np.ndarray:
    """Exact O(n) Pool Adjacent Violators Algorithm for L2 isotonic non-decreasing fit."""
    blocks: list[list[float]] = []  # [sum_y, count]
    for val in y:
        blocks.append([float(val), 1.0])
        while len(blocks) >= 2:
            m1 = blocks[-2][0] / blocks[-2][1]
            m2 = blocks[-1][0] / blocks[-1][1]
            if m1 > m2 + 1e-15:
                b_last = blocks.pop()
                blocks[-1][0] += b_last[0]
                blocks[-1][1] += b_last[1]
            else:
                break
    out = np.empty_like(y, dtype=np.float64)
    idx = 0
    for sum_v, cnt in blocks:
        k = int(cnt)
        out[idx : idx + k] = sum_v / cnt
        idx += k
    return out


def _auroc(scores: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> float | None:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    wins = float(np.sum(diff > 0.0) + 0.5 * np.sum(diff == 0.0))
    return wins / float(pos.size * neg.size)


def _spearman(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> tuple[float | None, float | None]:
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 3 or np.all(xa == xa[0]) or np.all(ya == ya[0]):
        return (None, None)
    r = spearmanr(xa, ya)
    rho = None if np.isnan(r.statistic) else float(r.statistic)
    pval = None if np.isnan(r.pvalue) else float(r.pvalue)
    return (rho, pval)


def _pinball_loss(y: float, f: float, tau: float) -> float:
    e = float(y) - float(f)
    return float(max(tau * e, (tau - 1.0) * e))


# ---------------------------------------------------------------------------
# Tarefa 1: Manifestos, B0-rest Params, 152 Cols Bloco A, e Sub-100ms C++/IPC
# ---------------------------------------------------------------------------

def run_task1() -> dict[str, Any]:
    print("[T1] Extraindo manifestos, hiperparâmetros B0-rest, auditoria 152 colunas e telemetria C++/IPC...")
    sla_path = os.path.join(SRC_DIR, "nsoran", "sqr_jc", "sla_manifest.yaml")
    b0_path = os.path.join(DOCS_JC_DIR, "B0_FROZEN_MANIFEST.json")
    scaler_path = os.path.join(REPO_ROOT, "results", "sqr_jc_final", "03_calibration", "feature_scaler_manifest.yaml")
    states_csv = os.path.join(DOCS_JC_DIR, "GATE_CONTEXTUAL_ESTADOS.csv")

    sla_raw = yaml_load(open(sla_path, encoding="utf-8").read())
    scaler_raw = yaml_load(open(scaler_path, encoding="utf-8").read())
    b0_m = json.loads(open(b0_path, encoding="utf-8").read())

    # Verificar zero seeds reservadas no manifesto B0 ativo (dev vs blocked)
    for s in b0_m["seeds_used_for_dev_and_robustness"]["dev_calibration_and_ablation"] + b0_m["seeds_used_for_dev_and_robustness"]["dev_robustness_check"]:
        assert not is_reserved_confirmatory_seed(int(s)), f"Reserved seed leaked: {s}"

    cfg = b0_m["config_descriptor"]
    param_rows: list[dict[str, Any]] = []

    def _add_p(cat: str, name: str, val: Any, unit_or_rule: str, source: str) -> None:
        param_rows.append({
            "category": cat,
            "parameter": name,
            "value": str(val),
            "unit_or_rule": unit_or_rule,
            "source": source,
        })

    # SLAs & Regras heterogêneas
    _add_p("SLA", "sla_manifest_version", sla_raw["sla_manifest_version"], "versão congelada", "sla_manifest.yaml")
    _add_p("SLA_eMBB", "threshold_mbps", sla_raw["embb"]["threshold"], f"Mbps; {sla_raw['embb']['margin_formula']}; {sla_raw['embb']['per_ue_aggregation']}", "sla_manifest.yaml")
    _add_p("SLA_URLLC", "threshold_p95_ms", sla_raw["urllc"]["threshold"], f"ms; {sla_raw['urllc']['margin_formula']}; chegada floor(txSec*10)", "sla_manifest.yaml")
    _add_p("SLA_URLLC_sec", "secondary_miss_pct", sla_raw["urllc"]["secondary_threshold_pct"], "%; REPORT_ONLY (nunca gate)", "sla_manifest.yaml")
    _add_p("SLA_mMTC", "threshold_pdr", sla_raw["mmtc"]["threshold"], f"fração rolling 1s (10 janelas); {sla_raw['mmtc']['margin_formula']}", "sla_manifest.yaml")
    _add_p("SLA_Lambda", "normalized_margin_boundary", 0.0, "m_i >= 0.0 satisfeito; m_i < 0.0 violado", "sla_margin.py")
    _add_p("Emenda_4", "urllc_pool_closure_rule", "delivered(W)==tx_kpm(W) and tx_kpm(W)>0", "CLOSED => 1 update só-URLLC; tx==0 => NO_ACTIVE_DEMAND; nunca fecha => expira em MAX_AGE=30 janelas (3.0 s) como expired_vacuous_R19", "PROTOCOL_FROZEN_AMENDMENT_4 & sqr_run.py:664")
    _add_p("G4B_Note", "g4b_yaml_block_status", sla_raw["g4b"]["status"], f"urllc_agg_win={sla_raw['g4b']['urllc_aggregation_window_ms']} ms (nota histórica no YAML; Gate 1 usa janela de 100 ms por coorte de chegada)", "sla_manifest.yaml")

    # Hiperparâmetros do regressor & política SQR no B0-rest
    _add_p("B0_Ident", "frozen_baseline_id", b0_m["frozen_baseline_id"], f"arm={b0_m['candidate_arm']} config_sha256={b0_m['config_sha256']}", "B0_FROZEN_MANIFEST.json")
    _add_p("Kernel", "type_nu_length_scale", "Matérn nu=1.5, length_scale=1.0 isotrópico em R^7", "x=[z_i(6), a_i/C]", "sqr_controller.py:214 & kernel.py")
    _add_p("Regressor", "memory_size_B", 2000, "vetores de suporte/slice", "quantile_regressor.py:111")
    _add_p("Regressor", "eta", 0.1, "passo pinball", "quantile_regressor.py:113")
    _add_p("Regressor", "p_penalty", 2.0, "multiplicador falha crítica (f>=0 e y<0)", "quantile_regressor.py:114")
    _add_p("Regressor", "pruning_rule_R10", "argmin_j |alpha_j| * k(x_new, x_j)", "desempate pelo menor índice (mais antigo)", "quantile_regressor.py:206-214")
    _add_p("Policy_SQR", "gamma", cfg["gamma"], "termo de custo -gamma*a/C em A_safe", "B0_FROZEN_MANIFEST.json")
    _add_p("Policy_SQR", "tau_init_min_max_step", f"tau0={cfg['tau_init']}, [{0.01}, {0.05}], step={cfg['tau_step']}", "adapta a cada bloco W=100 passos (delta=0.01; recupera após 2 blocos limpos)", "B0_FROZEN_MANIFEST.json & sqr_controller.py:75-85")
    _add_p("Policy_SQR", "M0_and_M_bounds", f"M0={cfg['margin_init']}, bounds={cfg['margin_bounds']}", "M <- min(50, 2M) se v_W>0.01; M/2 após 2 blocos limpos", "B0_FROZEN_MANIFEST.json & sqr_controller.py:82-83")
    _add_p("Policy_SQR", "epsilon_T_knn_k_succ_mem", "eps=0.1, T=1.0, knn_k=10, success_mem=2000", "exploração incerta", "sqr_controller.py:73-90")
    _add_p("B0_Delta", "virtual_alpha_R12", cfg["virtual_alpha"], "DESATIVADO (vs 0.05 no SQR/D1-v9)", "B0_FROZEN_MANIFEST.json")
    _add_p("B0_Delta", "empty_slice_guard_A2a", cfg["empty_slice_guard"], "zera proposta em slice comprovadamente vazia pré-decisão", "B0_FROZEN_MANIFEST.json")
    _add_p("B0_Delta", "mean_field_A4", cfg["mean_field"], "z_i dims [4..5] = [prev_a_i/C, viol_i] (sem z4/z5 globais)", "B0_FROZEN_MANIFEST.json & sqr_het.py:144-145")
    _add_p("B0_Delta", "fix_scaler_index_Ascaler", cfg["fix_scaler_index"], "indexação 'eum' em slice_state() [antes bug 'emu']", "B0_FROZEN_MANIFEST.json & sqr_het.py:124")
    _add_p("B0_Delta", "urllc_delayed_update", cfg["urllc_delayed_update"], "Emenda 4 ativa", "B0_FROZEN_MANIFEST.json")
    _add_p("B0_Scenario", "c_prbs_cli_vs_mac", f"cli={cfg['c_prbs']} -> MAC C={scaler_raw['c']}", "NumRbPerRbg=1 => 53 RBGs = 53 PRBs/slot DL", "B0_FROZEN_MANIFEST.json")
    _add_p("B0_Scenario", "warmup_s_and_policy", f"{cfg['warmup_s']} s ({int(cfg['warmup_s']*10)} janelas)", "SQR_HET_BOOTSTRAP_PROP (PropDemandController); modelo observa outcomes", "sqr_run.py:791-867")

    for k in ("n_e", "n_u", "n_m", "queue_e", "queue_u", "queue_m", "offered_e", "offered_u", "offered_m"):
        _add_p("FeatureScaler", f"{k}_p1_p99", f"[{scaler_raw[k + '_p1']}, {scaler_raw[k + '_p99']}]", f"rule={scaler_raw['rules'][k]}; n_cal={scaler_raw['n_cal_samples']}", "feature_scaler_manifest.yaml")
    for k in ("kpi_e", "kpi_u", "kpi_m", "snr_e", "snr_u", "snr_m", "prev_a_e", "prev_a_u", "prev_a_m"):
        _add_p("FeatureScaler", f"rule_{k}", scaler_raw["rules"][k], "escala física fixa", "feature_scaler_manifest.yaml")

    pd.DataFrame(param_rows).to_csv(os.path.join(OUT_DIR, "t1_b0_rest_manifest_params.csv"), index=False)

    # Auditoria exata das 152 colunas de GATE_CONTEXTUAL_ESTADOS.csv
    df_gce = pd.read_csv(states_csv)
    assert len(df_gce.columns) == 152, f"Expected 152 cols, got {len(df_gce.columns)}"
    gce_rows: list[dict[str, Any]] = []
    for idx, col in enumerate(df_gce.columns):
        if idx <= 11:
            block = "META_E_INTEGRIDADE"
        elif idx <= 110:
            block = "BLOCO_A_PRE_DECISAO"
        else:
            block = "BLOCO_B_POS_REPLAY_VEDADO"
        if any(x in col for x in ("urllc", "_u_", "_u")) and block == "BLOCO_A_PRE_DECISAO":
            cat = "URLLC_observabilidade_e_backlog"
        elif col.startswith(("n_", "active_mask", "live_floor", "queue_", "offered_", "tx_pkts_", "snr_")):
            cat = "Demanda_Fila_Carga_SNR"
        elif col.startswith(("b0_", "eq_", "prop_", "delta_", "l1_")):
            cat = "Acoes_e_Modelos_B0_PreDecisao"
        elif col.startswith(("prev_", "viol_", "hist10_", "last_closed_")):
            cat = "Historico_Causal_e_KPM_Anterior"
        elif block == "BLOCO_B_POS_REPLAY_VEDADO":
            cat = "Desfecho_Offline_PosReplay"
        else:
            cat = "Chave_ou_Integridade"

        n_na = int(df_gce[col].isna().sum())
        gce_rows.append({
            "col_idx": idx,
            "col_name": col,
            "block": block,
            "category": cat,
            "n_rows": len(df_gce),
            "n_non_null": int(len(df_gce) - n_na),
            "n_nan": n_na,
        })
    pd.DataFrame(gce_rows).to_csv(os.path.join(OUT_DIR, "t1_bloco_a_152_colunas_audit.csv"), index=False)

    # Reconstruir causalmente as duas quantidades ausentes nas 152 colunas (pools abertos e expirados acumulados até sw)
    traced_base = os.path.join(DEC_DIR, "b0_traced_gate1")
    urllc_queue_audit = []
    for row in df_gce.itertuples():
        rdir = os.path.join(traced_base, row.run_tag)
        sw = int(row.state_window_index)
        decs = [json.loads(l) for l in open(os.path.join(rdir, "decisions.jsonl")) if l.strip()]
        udels = [json.loads(l) for l in open(os.path.join(rdir, "urllc_delayed_updates.jsonl")) if l.strip()]
        # Reconstruir fila pendente até sw sem olhar > sw
        open_set = set()
        expired_cum = 0
        closed_delayed_cum = 0
        u_ev_by_obs = defaultdict(list)
        for u in udels:
            if int(u["observed_at_window"]) <= sw:
                u_ev_by_obs[int(u["observed_at_window"])].append(u)
        # Map decision by state_window_index
        for d in decs:
            d_sw = int(d["state_window_index"])
            if d_sw >= sw:
                break
            ow = d.get("outcome_window_index")
            if ow is not None and int(ow) <= sw:
                m_u = d["sla_outcome_margins"][1]
                ev_u = d["sla_evidence"][1] if "sla_evidence" in d else None
                # Stage 2 enqueue ocorre quando margem URLLC foi None mas houve demanda URLLC
                if m_u is None and ev_u == "PENDING_OBSERVATION":
                    open_set.add(int(ow))
        for obs_w, evs in sorted(u_ev_by_obs.items()):
            for u in evs:
                uw = int(u["w"])
                open_set.discard(uw)
                if u.get("updated") is True:
                    closed_delayed_cum += 1
                elif u.get("reason") == "expired_vacuous_R19":
                    expired_cum += 1
        urllc_queue_audit.append({
            "state_id": row.state_id,
            "run_tag": row.run_tag,
            "sw": sw,
            "urllc_pool_open_at_dec_csv": bool(row.urllc_pool_open_at_dec),
            "reconstructed_open_pools_count": len(open_set),
            "reconstructed_expired_vacuous_cum": expired_cum,
            "reconstructed_closed_delayed_cum": closed_delayed_cum,
            "last_closed_lag_csv": row.last_closed_urllc_lag_from_dec,
        })
    pd.DataFrame(urllc_queue_audit).to_csv(os.path.join(OUT_DIR, "t1_urllc_queue_causal_extra.csv"), index=False)

    # Verificação somente leitura no C++/IPC para telemetria sub-100ms
    sub100_rows = [
        {
            "signal": "RLC Head-of-Line (HoL) delay",
            "status_in_code": "AUSENTE",
            "location": "MODA/ns-3-dev/scratch/sqr/{sqr-sim.cc,sqr-mac-scheduler.cc,rslaq-mac-scheduler.cc}",
            "evidence": "Nenhum campo HoL nem consulta ao timestamp do pacote head no NrRlcUm; sqr-sim.cc:1460-1495 só lê GetUeDlBufferSize (bytes totais).",
        },
        {
            "signal": "Derivada intra-janela de fila RLC",
            "status_in_code": "AUSENTE_NO_IPC (apenas snapshot de borda 100ms)",
            "location": "MODA/ns-3-dev/scratch/sqr/rslaq-mac-scheduler.cc:497-516, 938-962; sqr-sim.cc:1460-1505",
            "evidence": "harq_tracking.csv e ue_detail.csv registram dlBufferSize por slot com activeDl em disco local, mas o KPM IPC transmite apenas queueBytes na borda de 100 ms (1 amostra/janela).",
        },
        {
            "signal": "Contadores HARQ (NACKs / activeHarqCount / retransmissões)",
            "status_in_code": "NA / INACESSIVEL_NA_SUBCLASSE",
            "location": "MODA/ns-3-dev/scratch/sqr/rslaq-mac-scheduler.cc:938-962",
            "evidence": "LogHarqState escreve literalmente 'NA,NA,NA' em harq_tracking.csv pois activeHarqCount/harqCapacity/canInsert são privados em NrMacSchedulerNs3; nada vai ao contrato IPC.",
        },
        {
            "signal": "Atraso de pacotes URLLC (sqr_packet_delays.csv)",
            "status_in_code": "POS_ENTREGA_DL_APENAS",
            "location": "MODA/ns-3-dev/scratch/sqr/sqr-sim.cc:478-506",
            "evidence": "WritePacketDelay é acionado na recepção RX; pacotes em fila ou perdidos antes da entrega não aparecem até chegarem (exigiria instrumentação C++ para perda/HoL pré-entrega).",
        },
    ]
    pd.DataFrame(sub100_rows).to_csv(os.path.join(OUT_DIR, "t1_cpp_sub100ms_telemetry_audit.csv"), index=False)

    return {
        "n_manifest_params": len(param_rows),
        "n_gce_cols": len(df_gce.columns),
        "n_bloco_a_cols": sum(1 for r in gce_rows if r["block"] == "BLOCO_A_PRE_DECISAO"),
        "prev_kpi_u_p95_imm_all_nan": bool(df_gce["prev_kpi_u_p95_imm"].isna().all()),
        "urllc_open_queue_max": max(r["reconstructed_open_pools_count"] for r in urllc_queue_audit),
        "urllc_expired_cum_max": max(r["reconstructed_expired_vacuous_cum"] for r in urllc_queue_audit),
    }


# ---------------------------------------------------------------------------
# Tarefa 2: Auditoria BUG-03 nos 9 runs D1-v9 e nos 30 estados D2-v2
# ---------------------------------------------------------------------------

def run_task2() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, Any]]]:
    print("[T2] Auditando dessincronização IPC BUG-03 nos 9 runs D1-v9 e 30 estados D2-v2...")
    d1_dir = os.path.join(DEC_DIR, "d1_runs_v9_delayed")
    run_rows = []
    d1_lag_by_run: dict[str, dict[str, Any]] = {}

    for run_tag in sorted(os.listdir(d1_dir)):
        rdir = os.path.join(d1_dir, run_tag)
        if not os.path.isdir(rdir):
            continue
        decs = [json.loads(l) for l in open(os.path.join(rdir, "decisions.jsonl")) if l.strip()]
        prb_df = pd.read_csv(_find_single(rdir, "sqr_prb_windows.csv"))
        sm_df = pd.read_csv(os.path.join(rdir, "step_metrics.csv"))
        pmap = {
            int(r.timestamp_ms) // 100 - 1: (
                int(r.prb_executed_rbg_0),
                int(r.prb_executed_rbg_1),
                int(r.prb_executed_rbg_2),
            )
            for r in prb_df.itertuples()
        }
        mismatches = []
        for d in decs:
            if not d.get("missing_reason") and d.get("outcome_window_index") is not None:
                ow = int(d["outcome_window_index"])
                if ow <= 0:
                    continue
                act = tuple(int(v) for v in d["action"])
                mac = pmap.get(ow)
                if mac is not None and mac != act:
                    mismatches.append((int(d["decision_index"]), int(d["state_window_index"]), ow, act, mac))
        sm0 = sm_df[sm_df["slice_id"] == 0]
        over50 = sm0[sm0["decision_time_ms"] > 50.0]
        onset_ow = mismatches[0][2] if mismatches else None
        onset_step = mismatches[0][0] if mismatches else None
        info = {
            "run_tag": run_tag,
            "lam": int(run_tag.split("_lam")[1].split("_s")[0]),
            "seed": int(run_tag.split("_s")[1]),
            "n_decisions": len(decs),
            "n_mismatches_ow_ge_1": len(mismatches),
            "onset_outcome_window": onset_ow,
            "onset_decision_index": onset_step,
            "n_decision_ms_gt_50": int(len(over50)),
            "first_over50_step": int(over50["step"].iloc[0]) if len(over50) else None,
            "max_decision_time_ms": float(sm0["decision_time_ms"].max()),
            "clean": len(mismatches) == 0,
        }
        d1_lag_by_run[run_tag] = info
        run_rows.append(info)

    df_runs = pd.DataFrame(run_rows)
    df_runs.to_csv(os.path.join(OUT_DIR, "t2_d1v9_runs_ipc_lag_audit.csv"), index=False)

    d2_states = pd.read_csv(os.path.join(DEC_DIR, "D2v2_states.csv"))
    st_rows = []
    for r in d2_states.itertuples():
        sid = str(r.state_id)
        lam = int(r.lam)
        seed = int(r.seed)
        t_win = int(sid.split("_t")[1])
        rtag = f"het_lam{lam}_s{seed}"
        rinfo = d1_lag_by_run[rtag]
        onset_ow = rinfo["onset_outcome_window"]
        is_pre = (onset_ow is None) or (t_win < int(onset_ow))
        st_rows.append({
            "state_id": sid,
            "run_tag": rtag,
            "lam": lam,
            "seed": seed,
            "stratum": r.stratum,
            "t_win": t_win,
            "onset_outcome_window": onset_ow if onset_ow is not None else -1,
            "desync_status": "PRE_DESYNC" if is_pre else "POST_DESYNC",
            "label": r.label,
            "aa_ok": bool(r.aa_ok),
            "t2064_full_revalidated": (sid == "S1_lam5_s2101_t2064"),
            "n_alt_strict_pass": int(r.n_alt_strict_pass),
        })
    df_d2_st = pd.DataFrame(st_rows)
    df_d2_st.to_csv(os.path.join(OUT_DIR, "t2_d2v2_30_states_desync_status.csv"), index=False)
    return df_runs, df_d2_st, d1_lag_by_run


# ---------------------------------------------------------------------------
# Reconstrução causal forward para D2-v2 PRE_DESYNC (zero checkpoints .pkl!)
# ---------------------------------------------------------------------------

def reconstruct_d2v2_pre_desync_curves(df_d2_st: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Reconstruct exact online A0 (and shadow B0-rest) models causally at t_win."""
    print("[T2/T3] Reconstruindo modelos causais D2-v2 PRE_DESYNC (A0 exato + Shadow B0-rest) sem ler .pkl...")
    scaler_path = os.path.join(REPO_ROOT, "results", "sqr_jc_final", "03_calibration", "feature_scaler_manifest.yaml")
    scaler = FeatureScaler.load_manifest(scaler_path)

    pre_df = df_d2_st[df_d2_st["desync_status"] == "PRE_DESYNC"]
    by_run: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for r in pre_df.itertuples():
        by_run[str(r.run_tag)].append((str(r.state_id), int(r.t_win)))

    reconstructed: dict[str, dict[str, Any]] = {}
    d1_dir = os.path.join(DEC_DIR, "d1_runs_v9_delayed")

    for rtag, tlist in sorted(by_run.items()):
        rdir = os.path.join(d1_dir, rtag)
        decs = [json.loads(l) for l in open(os.path.join(rdir, "decisions.jsonl")) if l.strip()]
        ud_p = os.path.join(rdir, "urllc_delayed_updates.jsonl")
        udels = [json.loads(l) for l in open(ud_p) if l.strip()] if os.path.isfile(ud_p) else []
        kpm_path = _find_single(rdir, "sqr_kpm_history.csv")
        sm_df = pd.read_csv(os.path.join(rdir, "step_metrics.csv"))

        imm_urllc_margin_by_w = {
            int(d["outcome_window_index"]): d["sla_outcome_margins"][1]
            for d in decs
            if d.get("outcome_window_index") is not None
        }
        udels_by_obs: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for u in udels:
            if u.get("updated") is True:
                udels_by_obs[int(u["observed_at_window"])].append(u)

        raw_lines = open(kpm_path, encoding="utf-8").readlines()
        hdr = raw_lines[0]
        by_ts: dict[int, list[str]] = defaultdict(list)
        for ln in raw_lines[1:]:
            by_ts[int(ln.split(",", 1)[0])].append(ln)

        seed = int(rtag.split("_s")[1])
        # Exact A0 (D1-v9): virtual_alpha=0.05, empty_slice_guard=False, mean_field=False, fix_scaler_index=False
        ctls_a0 = tuple(SqrSliceController(state_dim=6, n_prbs=52, gamma=1.0, sla_threshold=0.0, seed=seed + 1000 * i) for i in range(3))
        het_a0 = SqrHetController(ctls_a0, scaler=scaler, c=52, virtual_alpha=0.05, empty_slice_guard=False, mean_field=False, fix_scaler_index=False)

        # Shadow B0-rest counterfactual model on the same physical trajectory: virtual_alpha=0.0, empty_slice_guard=True, fix_scaler_index=True
        ctls_sh = tuple(SqrSliceController(state_dim=6, n_prbs=52, gamma=1.0, sla_threshold=0.0, seed=seed + 1000 * i) for i in range(3))
        het_sh = SqrHetController(ctls_sh, scaler=scaler, c=52, virtual_alpha=0.0, empty_slice_guard=True, mean_field=False, fix_scaler_index=True)

        rt = HetRuntime("/nonexistent_no_poll")
        parser = SqrKpmParser(num_slices=3)
        tmpf = tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False)
        with open(tmpf.name, "w", encoding="utf-8") as wf:
            wf.write(hdr)
        last_win = parser.parse_window(tmpf.name)

        dec_by_idx = {int(d["decision_index"]): d for d in decs}
        target_map = {t_w: sid for sid, t_w in tlist}
        max_t = max(target_map.keys())
        prev_act, prev_viol = (0, 0, 0), (0, 0, 0)
        pending = None
        state_by_gov_w: dict[int, tuple[Any, tuple[int, int, int]]] = {}

        for srow in sm_df[sm_df["slice_id"] == 0].sort_values("step").itertuples():
            step = int(srow.step)
            cb_ts = step * 100
            if cb_ts > (max_t + 1) * 100:
                break
            if cb_ts in by_ts:
                with open(tmpf.name, "w", encoding="utf-8") as wf:
                    wf.write(hdr)
                    wf.writelines(by_ts[cb_ts])
                last_win = parser.parse_window(tmpf.name)
            win = last_win
            _win_ts = int(win.timestamp)
            _is_hold = (_win_ts == getattr(het_a0, "_last_win_ts", -1))
            if not _is_hold:
                het_a0._last_win_ts = _win_ts
                het_sh._last_win_ts = _win_ts
                w = rt.window_index(win)
                tx_m = sum(r.d_tx_packets for r in win.rows if r.slice_id == 2)
                rx_m = sum(r.d_rx_packets for r in win.rows if r.slice_id == 2)
                if rt._mmtc_roll and rt._mmtc_roll[-1][0] == w:
                    rt._mmtc_roll.pop()
                rt._mmtc_roll.append((w, float(tx_m), float(rx_m)))
                _txs = sum(t for _, t, _ in rt._mmtc_roll)
                _rxs = sum(r for _, _, r in rt._mmtc_roll)
                rt._win_cache[w] = {
                    "embb_median": rt.embb_median(win),
                    "mmtc_pdr": (_rxs / _txs) if _txs > 0 else None,
                    "mmtc_demand": _txs > 0,
                    "urllc_tx": sum(r.d_tx_packets for r in win.rows if r.slice_id == 1),
                }
                rt.last_window_index = w
                m_u_imm = imm_urllc_margin_by_w.get(w)
                if m_u_imm is not None:
                    rt.delays.by_win[w] = [10.0 * (1.0 - float(m_u_imm))]

            if pending is not None:
                m_on = pending["dlog"]["sla_outcome_margins"]
                zs_a0 = [het_a0.slice_state(pending["state"], sid) for sid in range(3)]
                zs_sh = [het_sh.slice_state(pending["state"], sid) for sid in range(3)]
                for sid in range(3):
                    if m_on[sid] is not None:
                        ctls_a0[sid].observe(zs_a0[sid], int(pending["action"][sid]), float(m_on[sid]), threshold=0.0)
                        ctls_sh[sid].observe(zs_sh[sid], int(pending["action"][sid]), float(m_on[sid]), threshold=0.0)
                prev_viol = tuple(0 if (m is None or m >= 0) else 1 for m in m_on)
                pending = None

            if not _is_hold and rt.last_window_index in udels_by_obs:
                for u in udels_by_obs.pop(rt.last_window_index):
                    st_u, act_u = state_by_gov_w[int(u["w"])]
                    m_u = (10.0 - float(u["p95_ms"])) / 10.0
                    ctls_a0[1].observe(het_a0.slice_state(st_u, 1), int(act_u[1]), m_u, threshold=0.0)
                    ctls_sh[1].observe(het_sh.slice_state(st_u, 1), int(act_u[1]), m_u, threshold=0.0)

            if step in dec_by_idx:
                dlog = dec_by_idx[step]
                cap_c = int(dlog["capacity_C"])
                if het_a0.c != cap_c:
                    het_a0.set_n_prbs(cap_c)
                    het_sh.set_n_prbs(cap_c)
                st = rt.joint_state(win, prev_act, prev_viol)
                # Populate causal kpm_tx_packets for causal_active_slices / prop_active_split
                cch = rt._win_cache.get(rt.window_index(win)) or {}
                u_tag, _ = rt.urllc_pool_status(rt.window_index(win))
                st.extras["kpm_tx_packets"] = (
                    int(sum(r.d_tx_packets for r in win.rows if r.slice_id == 0)),
                    int(cch.get("urllc_tx") or 0),
                    int(sum(r.d_tx_packets for r in win.rows if r.slice_id == 2)),
                )
                st.extras["urllc_pool_open"] = (u_tag == "OPEN")
                st.extras["mmtc_rolling_active"] = bool(cch.get("mmtc_demand", False))

                zs_a0 = [het_a0.slice_state(st, sid) for sid in range(3)]
                zs_sh = [het_sh.slice_state(st, sid) for sid in range(3)]
                props = tuple(int(v) for v in dlog["a_after_uncertainty_margin"])
                alloc = tuple(int(v) for v in dlog["action"])
                gov_w = int(dlog["state_window_index"]) + 1

                if gov_w in target_map:
                    sid_key = target_map[gov_w]
                    sc_a0 = [ctls_a0[i].score_candidates(zs_a0[i]) for i in range(3)]
                    sc_sh = [ctls_sh[i].score_candidates(zs_sh[i]) for i in range(3)]
                    pb = [float(sc_a0[i]["f"][props[i]]) for i in range(3)]
                    pa = [float(sc_a0[i]["f"][alloc[i]]) for i in range(3)]
                    eb = max(abs(a - b) for a, b in zip(pb, dlog["predicted_safe_before_proposal"]))
                    ea = max(abs(a - b) for a, b in zip(pa, dlog["predicted_safe_after_projection"]))
                    assert eb < 1e-12 and ea < 1e-12, f"Reconstruction mismatch at {sid_key}: eb={eb}, ea={ea}"
                    act_mask, live_mask = causal_active_slices(st)
                    reconstructed[sid_key] = {
                        "state_id": sid_key,
                        "run_tag": rtag,
                        "gov_w": gov_w,
                        "capacity_C": cap_c,
                        "a_b0_exec": alloc,
                        "state": st,
                        "active_mask": act_mask,
                        "live_floor_mask": live_mask,
                        "parity_err_before": eb,
                        "parity_err_after": ea,
                        "a0": {
                            "f": [np.asarray(sc_a0[i]["f"], dtype=np.float64) for i in range(3)],
                            "sigma": [np.asarray(sc_a0[i]["sigma"], dtype=np.float64) for i in range(3)],
                            "tau": [float(ctls_a0[i].tau) for i in range(3)],
                        },
                        "b0rest_shadow": {
                            "f": [np.asarray(sc_sh[i]["f"], dtype=np.float64) for i in range(3)],
                            "sigma": [np.asarray(sc_sh[i]["sigma"], dtype=np.float64) for i in range(3)],
                            "tau": [float(ctls_sh[i].tau) for i in range(3)],
                        },
                    }

                if any(a < p for p, a in zip(props, alloc)):
                    het_a0.apply_virtual_updates(st, props, alloc)
                # Shadow skips R12 virtual updates deliberately
                state_by_gov_w[gov_w] = (st, alloc)
                pending = {"state": st, "action": alloc, "dlog": dlog}
                prev_act = alloc
        os.unlink(tmpf.name)

    return reconstructed


# ---------------------------------------------------------------------------
# Tarefa 3: Curvas Quantílicas (Monotonicidade, Achatamento, Rearranjo)
# ---------------------------------------------------------------------------

def _load_gate1_snapshots() -> dict[str, dict[str, Any]]:
    plan = pd.read_csv(os.path.join(DEC_DIR, "A5_GATE1_SAMPLING_PLAN.csv"))
    by_dir = defaultdict(list)
    for r in plan.itertuples():
        by_dir[str(r.b0_traced_dir)].append(r)

    snaps: dict[str, dict[str, Any]] = {}
    for d_str, rows in by_dir.items():
        wanted = {int(r.governed_window_index): r for r in rows}
        with gzip.open(os.path.join(d_str, "causal_trace.jsonl.gz"), "rt", encoding="utf-8") as gz:
            for idx, ln in enumerate(gz):
                if idx == 0 or not ln.strip():
                    continue
                tw = json.loads(ln)
                gw = int(tw["governed_window_index"])
                if gw in wanted:
                    assert int(tw["causal_horizon_max_window_index"]) == gw - 1
                    r = wanted[gw]
                    psm = sorted(tw["per_slice_model"], key=lambda e: int(e["slice_id"]))
                    snaps[str(r.state_id)] = {
                        "state_id": str(r.state_id),
                        "run_tag": str(r.b0_traced_tag),
                        "lam": int(r.lam),
                        "seed": int(r.seed),
                        "gov_w": gw,
                        "capacity_C": int(tw["capacity_C"]),
                        "a_b0_exec": tuple(int(v) for v in tw["a_exec_per_slice"]),
                        "state": tw["state"],
                        "modes": tw["modes_per_slice"],
                        "f": [np.asarray(psm[i]["f_candidates"], dtype=np.float64) for i in range(3)],
                        "sigma": [np.asarray(psm[i]["sigma_candidates"], dtype=np.float64) for i in range(3)],
                        "tau": [float(psm[i]["tau"]) for i in range(3)],
                    }
    assert len(snaps) == 25
    return snaps


def _curve_diagnostics(f: np.ndarray, sigma: np.ndarray) -> dict[str, Any]:
    df1 = np.diff(f)
    pair_sig = np.maximum(sigma[:-1], sigma[1:])
    inv_mask = df1 < 0.0
    drops = np.maximum(0.0, -df1)

    sig_med = float(np.median(sigma))
    masks_strata = {
        "all": np.ones(len(df1), dtype=bool),
        "trusted_le_015": pair_sig <= DEFAULT_SIGMA_MAX,
        "interm_015_055": (pair_sig > DEFAULT_SIGMA_MAX) & (pair_sig <= DEFAULT_SIGMA_SOFT),
        "oos_gt_055": pair_sig > DEFAULT_SIGMA_SOFT,
        "below_med_sigma": pair_sig <= sig_med,
        "above_med_sigma": pair_sig > sig_med,
    }
    rec: dict[str, Any] = {
        "f_min": float(np.min(f)),
        "f_max": float(np.max(f)),
        "f_range": float(np.max(f) - np.min(f)),
        "sigma_min": float(np.min(sigma)),
        "sigma_med": sig_med,
        "sigma_max": float(np.max(sigma)),
        "n_pts_trusted_le_015": int(np.sum(sigma <= DEFAULT_SIGMA_MAX)),
        "n_pts_oos_gt_055": int(np.sum(sigma > DEFAULT_SIGMA_SOFT)),
    }
    for sname, m in masks_strata.items():
        n_m = int(np.sum(m))
        rec[f"n_pairs_{sname}"] = n_m
        if n_m > 0:
            inv_m = inv_mask[m]
            drops_m = drops[m]
            rec[f"inv_rate_{sname}"] = float(np.mean(inv_m))
            rec[f"inv_count_{sname}"] = int(np.sum(inv_m))
            rec[f"tv_neg_{sname}"] = float(np.sum(drops_m))
            rec[f"drop_mean_on_inv_{sname}"] = float(np.mean(drops_m[inv_m])) if np.any(inv_m) else 0.0
            rec[f"drop_max_{sname}"] = float(np.max(drops_m))
        else:
            rec[f"inv_rate_{sname}"] = None
            rec[f"inv_count_{sname}"] = 0
            rec[f"tv_neg_{sname}"] = 0.0
            rec[f"drop_mean_on_inv_{sname}"] = None
            rec[f"drop_max_{sname}"] = 0.0

    # Achatamento espacial delta in {2, 4, 8}
    for delta in (2, 4, 8):
        dfd = f[delta:] - f[:-delta]
        psd = np.maximum(sigma[delta:], sigma[:-delta])
        cost_thr = 1.0 * float(delta) / 53.0
        for tag, mm in (("all", np.ones(len(dfd), dtype=bool)), ("trusted", psd <= DEFAULT_SIGMA_MAX)):
            vals = dfd[mm]
            rec[f"df{delta}_{tag}_n"] = int(vals.size)
            if vals.size > 0:
                rec[f"df{delta}_{tag}_mean"] = float(np.mean(vals))
                rec[f"df{delta}_{tag}_p50"] = float(np.median(vals))
                rec[f"df{delta}_{tag}_p90"] = float(np.percentile(vals, 90))
                rec[f"df{delta}_{tag}_frac_le0"] = float(np.mean(vals <= 0.0))
                rec[f"df{delta}_{tag}_frac_abs_lt_gamma"] = float(np.mean(np.abs(vals) < cost_thr))
            else:
                rec[f"df{delta}_{tag}_mean"] = None
                rec[f"df{delta}_{tag}_p50"] = None
                rec[f"df{delta}_{tag}_p90"] = None
                rec[f"df{delta}_{tag}_frac_le0"] = None
                rec[f"df{delta}_{tag}_frac_abs_lt_gamma"] = None

    # Rearranjo monotônico offline (sort vs PAVA)
    f_sort = np.sort(f)
    f_pava = _pava_nondecreasing(f)
    pt_trusted = sigma <= DEFAULT_SIGMA_MAX
    pt_nontrusted = ~pt_trusted
    for mname, fm in (("sort", f_sort), ("pava", f_pava)):
        diff_m = np.abs(fm - f)
        rec[f"{mname}_inv_after"] = int(np.sum(np.diff(fm) < -1e-12))
        rec[f"{mname}_linf_all"] = float(np.max(diff_m))
        rec[f"{mname}_l1_mean_all"] = float(np.mean(diff_m))
        rec[f"{mname}_linf_trusted"] = float(np.max(diff_m[pt_trusted])) if np.any(pt_trusted) else None
        rec[f"{mname}_l1_mean_trusted"] = float(np.mean(diff_m[pt_trusted])) if np.any(pt_trusted) else None
        rec[f"{mname}_linf_nontrusted"] = float(np.max(diff_m[pt_nontrusted])) if np.any(pt_nontrusted) else None
        rec[f"{mname}_l1_mean_nontrusted"] = float(np.mean(diff_m[pt_nontrusted])) if np.any(pt_nontrusted) else None
    return rec


def run_task3(
    gate1_snaps: dict[str, dict[str, Any]],
    d2_recon: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[T3] Analisando monotonicidade, achatamento espacial e rearranjo monotônico nas curvas quantílicas...")
    rows = []
    for sid, sn in sorted(gate1_snaps.items()):
        for s in range(3):
            diag = _curve_diagnostics(sn["f"][s], sn["sigma"][s])
            diag.update({
                "dataset": "gate1_b0rest",
                "state_id": sid,
                "slice_id": s,
                "slice_name": SLICE_NAMES[s],
                "tau": sn["tau"][s],
                "mode": sn["modes"][s],
            })
            rows.append(diag)

    for sid, dn in sorted(d2_recon.items()):
        for model_key, ds_label in (("a0", "d2v2_pre_desync_a0_exact"), ("b0rest_shadow", "d2v2_pre_desync_b0rest_shadow")):
            for s in range(3):
                diag = _curve_diagnostics(dn[model_key]["f"][s], dn[model_key]["sigma"][s])
                diag.update({
                    "dataset": ds_label,
                    "state_id": sid,
                    "slice_id": s,
                    "slice_name": SLICE_NAMES[s],
                    "tau": dn[model_key]["tau"][s],
                    "mode": "",
                })
                rows.append(diag)

    df_t3 = pd.DataFrame(rows)
    df_t3.to_csv(os.path.join(OUT_DIR, "t3_curvas_quantilicas_por_estado_slice.csv"), index=False)

    # Resumo agregado por dataset x slice + distribuição exata de df_delta em nível de pontos
    sum_rows = []
    datasets_items = [("gate1_b0rest", gate1_snaps, None)] + [
        (ds_label, d2_recon, mkey)
        for mkey, ds_label in (("a0", "d2v2_pre_desync_a0_exact"), ("b0rest_shadow", "d2v2_pre_desync_b0rest_shadow"))
    ]
    for ds_label, container, mkey in datasets_items:
        for s in range(3):
            # Collect point-level arrays
            all_df1, all_sigpair, all_inv = [], [], []
            df_deltas_all = {2: [], 4: [], 8: []}
            df_deltas_tr = {2: [], 4: [], 8: []}
            sub = df_t3[(df_t3["dataset"] == ds_label) & (df_t3["slice_id"] == s)]
            for _, obj in sorted(container.items()):
                f = obj["f"][s] if mkey is None else obj[mkey]["f"][s]
                sig = obj["sigma"][s] if mkey is None else obj[mkey]["sigma"][s]
                d1 = np.diff(f)
                ps = np.maximum(sig[:-1], sig[1:])
                all_df1.append(d1)
                all_sigpair.append(ps)
                all_inv.append(d1 < 0.0)
                for delta in (2, 4, 8):
                    dfd = f[delta:] - f[:-delta]
                    psd = np.maximum(sig[delta:], sig[:-delta])
                    df_deltas_all[delta].append(dfd)
                    if np.any(psd <= DEFAULT_SIGMA_MAX):
                        df_deltas_tr[delta].append(dfd[psd <= DEFAULT_SIGMA_MAX])
            d1_v = np.concatenate(all_df1)
            ps_v = np.concatenate(all_sigpair)
            inv_v = np.concatenate(all_inv)

            srow: dict[str, Any] = {
                "dataset": ds_label,
                "slice_id": s,
                "slice_name": SLICE_NAMES[s],
                "n_states": int(len(sub)),
                "f_min_mean": float(sub["f_min"].mean()),
                "f_max_mean": float(sub["f_max"].mean()),
                "f_range_median": float(sub["f_range"].median()),
                "inv_rate_all": float(np.mean(inv_v)),
                "inv_rate_trusted_le_015": float(np.mean(inv_v[ps_v <= DEFAULT_SIGMA_MAX])) if np.any(ps_v <= DEFAULT_SIGMA_MAX) else None,
                "n_pairs_trusted_le_015": int(np.sum(ps_v <= DEFAULT_SIGMA_MAX)),
                "inv_rate_interm_015_055": float(np.mean(inv_v[(ps_v > DEFAULT_SIGMA_MAX) & (ps_v <= DEFAULT_SIGMA_SOFT)])) if np.any((ps_v > DEFAULT_SIGMA_MAX) & (ps_v <= DEFAULT_SIGMA_SOFT)) else None,
                "n_pairs_interm": int(np.sum((ps_v > DEFAULT_SIGMA_MAX) & (ps_v <= DEFAULT_SIGMA_SOFT))),
                "inv_rate_oos_gt_055": float(np.mean(inv_v[ps_v > DEFAULT_SIGMA_SOFT])) if np.any(ps_v > DEFAULT_SIGMA_SOFT) else None,
                "n_pairs_oos": int(np.sum(ps_v > DEFAULT_SIGMA_SOFT)),
                "tv_neg_mean": float(sub["tv_neg_all"].mean()),
                "drop_mean_on_inv_trusted": float(np.mean(-d1_v[(ps_v <= DEFAULT_SIGMA_MAX) & inv_v])) if np.any((ps_v <= DEFAULT_SIGMA_MAX) & inv_v) else 0.0,
                "drop_mean_on_inv_oos": float(np.mean(-d1_v[(ps_v > DEFAULT_SIGMA_SOFT) & inv_v])) if np.any((ps_v > DEFAULT_SIGMA_SOFT) & inv_v) else 0.0,
                "sort_linf_trusted_mean": float(sub["sort_linf_trusted"].dropna().mean()) if sub["sort_linf_trusted"].notna().any() else None,
                "sort_l1_trusted_mean": float(sub["sort_l1_mean_trusted"].dropna().mean()) if sub["sort_l1_mean_trusted"].notna().any() else None,
                "pava_linf_trusted_mean": float(sub["pava_linf_trusted"].dropna().mean()) if sub["pava_linf_trusted"].notna().any() else None,
                "pava_l1_trusted_mean": float(sub["pava_l1_mean_trusted"].dropna().mean()) if sub["pava_l1_mean_trusted"].notna().any() else None,
                "pava_linf_nontrusted_mean": float(sub["pava_linf_nontrusted"].dropna().mean()) if sub["pava_linf_nontrusted"].notna().any() else None,
            }
            for delta in (2, 4, 8):
                va = np.concatenate(df_deltas_all[delta])
                vt = np.concatenate(df_deltas_tr[delta]) if df_deltas_tr[delta] else np.empty(0)
                thr = float(delta) / 53.0
                srow[f"df{delta}_all_p25"] = float(np.percentile(va, 25))
                srow[f"df{delta}_all_p50"] = float(np.percentile(va, 50))
                srow[f"df{delta}_all_p75"] = float(np.percentile(va, 75))
                srow[f"df{delta}_all_p90"] = float(np.percentile(va, 90))
                srow[f"df{delta}_all_frac_le0"] = float(np.mean(va <= 0.0))
                srow[f"df{delta}_all_frac_abs_lt_gamma"] = float(np.mean(np.abs(va) < thr))
                srow[f"df{delta}_tr_n"] = int(vt.size)
                srow[f"df{delta}_tr_p50"] = float(np.percentile(vt, 50)) if vt.size else None
                srow[f"df{delta}_tr_p90"] = float(np.percentile(vt, 90)) if vt.size else None
                srow[f"df{delta}_tr_frac_le0"] = float(np.mean(vt <= 0.0)) if vt.size else None
                srow[f"df{delta}_tr_frac_abs_lt_gamma"] = float(np.mean(np.abs(vt) < thr)) if vt.size else None
            sum_rows.append(srow)

    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(os.path.join(OUT_DIR, "t3_resumo_monotonicidade_achatamento_por_slice.csv"), index=False)
    return df_t3, df_sum


# ---------------------------------------------------------------------------
# Tarefa 4: Confronto contra Desfechos Reais (Regimes A e B Separados)
# ---------------------------------------------------------------------------

def _margin_from_kpis(slice_id: int, embb_thr: float | None, urllc_p95: float | None, mmtc_pdr: float | None, urllc_complete: bool) -> float | None:
    if slice_id == 0:
        return embb_margin(embb_thr)
    if slice_id == 1:
        return urllc_margin(urllc_p95) if (urllc_complete and urllc_p95 is not None) else None
    return mmtc_margin(mmtc_pdr)


def _shannon_eff_bits_per_re(snr_db: float) -> float:
    # Aproximação causal simples 3GPP (limitada à faixa realista [QPSK..256QAM])
    if snr_db <= -5.0 or snr_db <= -900.0:
        return 0.15
    lin = 10.0 ** (min(max(float(snr_db), -5.0), 30.0) / 10.0)
    return float(min(5.5, max(0.15, math.log2(1.0 + lin))))


def _physical_donor_floor_est(st: Any, active_mask: tuple[bool, bool, bool], sid: int) -> float:
    """Simple causal pre-decision PRB floor estimate from queue + offered + SNR (no tuning)."""
    if not active_mask[sid]:
        return 0.0
    snr_keys = ("snr_e", "snr_u", "snr_m")
    q_keys = ("queue_e", "queue_u", "queue_m")
    off_keys = ("offered_e", "offered_u", "offered_m")
    n_keys = ("n_e", "n_u", "n_m")
    getf = (lambda k, d=0.0: st.get(k, d) if isinstance(st, dict) else getattr(st, k, d))
    n_i = int(getf(n_keys[sid], 0))
    q_i = float(getf(q_keys[sid], 0.0))
    off_i = float(getf(off_keys[sid], 0.0))
    snr_i = float(getf(snr_keys[sid], 0.0))
    # 10 MHz mu=0: ~50 DL slots/100ms * ~11 data sym/slot * 12 subcarriers = 6600 REs per PRB per 100ms window
    bits_per_prb_100ms = _shannon_eff_bits_per_re(snr_i) * 6600.0
    if sid == 0:
        # precisa sustentar max(3.5 Mbps * n_e em 100ms, drenagem parcial) — pelo menos ~3.5e6 * 0.1 * n_e bits
        req_bits = max(3.5e6 * 0.1 * max(1, n_i), q_i * 8.0)
    elif sid == 1:
        # URLLC exige drenagem imediata em < 10 ms (~5 slots DL -> 10x menor capacidade efetiva no orçamento de 10 ms!)
        # backlog + 10ms de taxa CBR 0.256 Mbps/UE dentro de janela crítica de 10 ms
        req_bits_10ms = max(q_i * 8.0, 1.0) + (0.256e6 * max(1, n_i)) * 0.010
        bits_per_prb_10ms = max(1.0, bits_per_prb_100ms * 0.1)
        return float(min(53.0, math.ceil(req_bits_10ms / bits_per_prb_10ms)))
    else:
        req_bits = max(q_i * 8.0, max(0.0016e6 * max(1, n_i), off_i * 1e6) * 0.1)
    return float(min(53.0, math.ceil(req_bits / max(1.0, bits_per_prb_100ms))))


def run_task4(
    gate1_snaps: dict[str, dict[str, Any]],
    df_d2_st: pd.DataFrame,
    d2_recon: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    print("[T4] Confrontando curvas causais contra desfechos reais (Regime A Ociosidade & Regime B Transferência)...")

    # --- Preparar tabela unificada de braços por estado (separando Gate1 vs D2-v2 PRE_DESYNC) ---
    g1_metrics = pd.read_csv(os.path.join(DEC_DIR, "A5_GATE1_METRICS.csv"))
    d2_res = pd.read_csv(os.path.join(DEC_DIR, "D2v2_results.csv"))
    t2064_full = pd.read_csv(os.path.join(DEC_DIR, "d2v2_t2064full_branches.csv"))

    # Map D2-v2 outcomes (using validated simTime=300 for S1_lam5_s2101_t2064!)
    d2_arm_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in d2_res.itertuples():
        sid = str(r.state_id)
        if sid == "S1_lam5_s2101_t2064":
            continue
        act = _parse_action_str(r.action)
        ev_parts = str(r.evidence).split("|")
        u_complete = (len(ev_parts) >= 2 and ev_parts[1] == "COMPLETE")
        d2_arm_rows[sid].append({
            "arm": f"alt{int(r.alt_index)}:{r.mechanism}",
            "action": act,
            "embb_thr": _safe_float(r.embb_thr),
            "urllc_p95": _safe_float(r.urllc_p95),
            "mmtc_pdr": _safe_float(r.mmtc_pdr),
            "u_complete": u_complete,
            "evidence": str(r.evidence),
            "strict_label": str(r.label),
        })
    for r in t2064_full.itertuples():
        act = _parse_action_str(r.action)
        ev_parts = str(r.evidence).split("|")
        u_complete = (len(ev_parts) >= 2 and ev_parts[1] == "COMPLETE")
        d2_arm_rows["S1_lam5_s2101_t2064"].append({
            "arm": f"alt{int(r.alt)}:{r.mech}",
            "action": act,
            "embb_thr": _safe_float(r.embb),
            "urllc_p95": _safe_float(r.p95),
            "mmtc_pdr": _safe_float(r.pdr),
            "u_complete": u_complete,
            "evidence": str(r.evidence),
            "strict_label": str(r.label),
        })

    # --- Regime A: Aproveitamento de Ociosidade & Chegadas Intra-Janela ---
    def _eval_idle_gain_ordering(
        arms_iter: list[tuple[str, tuple[int, int, int], list[float | None]]],
        a_base: tuple[int, int, int],
        base_margins: list[float | None],
        f_raw_s: list[np.ndarray],
        act_m: tuple[bool, bool, bool],
    ) -> tuple[int, ...]:
        """Score total-gain Δf vs Δy and marginal slope Δf/Δa vs Δy/Δa on deduplicated idle-absorption arms."""
        f_sort_s = [np.sort(f_raw_s[i]) for i in range(3)]
        f_pava_s = [_pava_nondecreasing(f_raw_s[i]) for i in range(3)]
        pure_arms = []
        seen_actions: set[tuple[int, int, int]] = {a_base}
        for arm_n, a_k, m_k in arms_iter:
            if a_k in seen_actions:
                continue
            if all((a_k[i] >= a_base[i]) if act_m[i] else True for i in range(3)) and any(
                a_k[i] > a_base[i] and act_m[i] for i in range(3)
            ):
                seen_actions.add(a_k)
                pure_arms.append((arm_n, a_k, m_k))
        tot = ok_raw = ok_sort = ok_pava = 0
        sl_tot = sl_ok_raw = sl_ok_sort = sl_ok_pava = 0
        for _, a_k, m_k in pure_arms:
            gained = [i for i in range(3) if act_m[i] and a_k[i] > a_base[i] and base_margins[i] is not None and m_k[i] is not None]
            for ia in range(len(gained)):
                for ib in range(ia + 1, len(gained)):
                    sa, sb = gained[ia], gained[ib]
                    da_a = float(a_k[sa] - a_base[sa])
                    da_b = float(a_k[sb] - a_base[sb])
                    dy_a = float(m_k[sa] - base_margins[sa])
                    dy_b = float(m_k[sb] - base_margins[sb])
                    if abs(dy_a - dy_b) >= 1e-12:
                        tot += 1
                        w_real = sa if dy_a > dy_b else sb
                        for ok_flag, ft in ((0, f_raw_s), (1, f_sort_s), (2, f_pava_s)):
                            df_a = float(ft[sa][a_k[sa]] - ft[sa][a_base[sa]])
                            df_b = float(ft[sb][a_k[sb]] - ft[sb][a_base[sb]])
                            if (sa if df_a >= df_b else sb) == w_real:
                                if ok_flag == 0:
                                    ok_raw += 1
                                elif ok_flag == 1:
                                    ok_sort += 1
                                else:
                                    ok_pava += 1
                    sy_a = dy_a / da_a
                    sy_b = dy_b / da_b
                    if abs(sy_a - sy_b) >= 1e-12:
                        sl_tot += 1
                        ws_real = sa if sy_a > sy_b else sb
                        for ok_flag, ft in ((0, f_raw_s), (1, f_sort_s), (2, f_pava_s)):
                            sf_a = float(ft[sa][a_k[sa]] - ft[sa][a_base[sa]]) / da_a
                            sf_b = float(ft[sb][a_k[sb]] - ft[sb][a_base[sb]]) / da_b
                            if (sa if sf_a >= sf_b else sb) == ws_real:
                                if ok_flag == 0:
                                    sl_ok_raw += 1
                                elif ok_flag == 1:
                                    sl_ok_sort += 1
                                else:
                                    sl_ok_pava += 1
        return len(pure_arms), tot, ok_raw, ok_sort, ok_pava, sl_tot, sl_ok_raw, sl_ok_sort, sl_ok_pava

    reg_a_rows = []
    # Gate 1
    for sid, grp in g1_metrics.groupby("state_id"):
        sn = gate1_snaps[sid]
        act_mask, _ = causal_active_slices(sn["state"])
        a_b0 = sn["a_b0_exec"]
        b0_row = grp[grp["arm"] == "b0"].iloc[0]
        idle_unalloc = int(sn["capacity_C"] - sum(a_b0))
        inact_b0 = int(b0_row["inactive_prbs_allocated"])
        # Chegadas intra-janela em slices inicialmente vazias (active_mask[s] == False, mas kpi observado nos replays!)
        intra_arrivals = [False, False, False]
        for rr in grp.itertuples():
            if not act_mask[0] and _safe_float(rr.embb_thr_mbps) is not None:
                intra_arrivals[0] = True
            if not act_mask[1] and int(rr.urllc_tx_pkts) > 0:
                intra_arrivals[1] = True
            if not act_mask[2] and _safe_float(rr.mmtc_pdr_rolling) is not None:
                intra_arrivals[2] = True

        b0_margins = [
            _margin_from_kpis(0, _safe_float(b0_row["embb_thr_mbps"]), None, None, True),
            _margin_from_kpis(1, None, _safe_float(b0_row["urllc_p95_ms"]), None, str(b0_row["urllc_pool_tag"]) == "CLOSED"),
            _margin_from_kpis(2, None, None, _safe_float(b0_row["mmtc_pdr_rolling"]), True),
        ]
        arms_iter_g1 = [
            (
                str(rr.arm),
                (int(rr.action_e), int(rr.action_u), int(rr.action_m)),
                [
                    _margin_from_kpis(0, _safe_float(rr.embb_thr_mbps), None, None, True),
                    _margin_from_kpis(1, None, _safe_float(rr.urllc_p95_ms), None, str(rr.urllc_pool_tag) == "CLOSED"),
                    _margin_from_kpis(2, None, None, _safe_float(rr.mmtc_pdr_rolling), True),
                ],
            )
            for rr in grp.itertuples()
            if rr.arm != "b0"
        ]
        n_pg, i_tot, i_raw, i_sort, i_pava, sl_tot, sl_raw, sl_sort, sl_pava = _eval_idle_gain_ordering(
            arms_iter_g1, a_b0, b0_margins, sn["f"], act_mask
        )

        reg_a_rows.append({
            "dataset": "gate1_b0rest",
            "state_id": sid,
            "active_mask": "".join("1" if x else "0" for x in act_mask),
            "n_active": sum(act_mask),
            "a_b0_e": a_b0[0], "a_b0_u": a_b0[1], "a_b0_m": a_b0[2],
            "b0_unallocated_prbs": idle_unalloc,
            "b0_inactive_prbs_allocated": inact_b0,
            "has_any_b0_idle_capacity": (idle_unalloc > 0 or inact_b0 > 0),
            "intra_win_arrival_e": intra_arrivals[0],
            "intra_win_arrival_u": intra_arrivals[1],
            "intra_win_arrival_m": intra_arrivals[2],
            "n_pure_idle_gain_arms": n_pg,
            "idle_order_pairs_nonzero": i_tot,
            "idle_order_ok_raw": i_raw,
            "idle_order_ok_sort": i_sort,
            "idle_order_ok_pava": i_pava,
            "idle_slope_pairs_nonzero": sl_tot,
            "idle_slope_ok_raw": sl_raw,
            "idle_slope_ok_sort": sl_sort,
            "idle_slope_ok_pava": sl_pava,
        })

    # D2-v2 PRE_DESYNC Regime A (A0 exact + Shadow B0-rest)
    for sid, dn in sorted(d2_recon.items()):
        act_mask = dn["active_mask"]
        a_b0 = dn["a_b0_exec"]
        arms = d2_arm_rows[sid]
        b0_r = [r for r in arms if r["arm"].startswith("alt0:")][0]
        idle_unalloc = int(dn["capacity_C"] - sum(a_b0))
        inact_b0 = int(sum(a_b0[i] for i in range(3) if not act_mask[i]))
        intra_arrivals = [False, False, False]
        for rr in arms:
            if not act_mask[0] and rr["embb_thr"] is not None:
                intra_arrivals[0] = True
            if not act_mask[1] and (rr["urllc_p95"] is not None or rr["evidence"].split("|")[1] != "NO_ACTIVE_DEMAND"):
                intra_arrivals[1] = True
            if not act_mask[2] and rr["mmtc_pdr"] is not None:
                intra_arrivals[2] = True
        b0_margins = [
            _margin_from_kpis(0, b0_r["embb_thr"], None, None, True),
            _margin_from_kpis(1, None, b0_r["urllc_p95"], None, b0_r["u_complete"]),
            _margin_from_kpis(2, None, None, b0_r["mmtc_pdr"], True),
        ]
        arms_iter_d2 = [
            (
                rr["arm"],
                rr["action"],
                [
                    _margin_from_kpis(0, rr["embb_thr"], None, None, True),
                    _margin_from_kpis(1, None, rr["urllc_p95"], None, rr["u_complete"]),
                    _margin_from_kpis(2, None, None, rr["mmtc_pdr"], True),
                ],
            )
            for rr in arms[1:]
        ]
        for ds_tag, model_key in (("d2v2_pre_desync_a0_exact", "a0"), ("d2v2_pre_desync_b0rest_shadow", "b0rest_shadow")):
            n_pg, i_tot, i_raw, i_sort, i_pava, sl_tot, sl_raw, sl_sort, sl_pava = _eval_idle_gain_ordering(
                arms_iter_d2, a_b0, b0_margins, dn[model_key]["f"], act_mask
            )
            reg_a_rows.append({
                "dataset": ds_tag,
                "state_id": sid,
                "active_mask": "".join("1" if x else "0" for x in act_mask),
                "n_active": sum(act_mask),
                "a_b0_e": a_b0[0], "a_b0_u": a_b0[1], "a_b0_m": a_b0[2],
                "b0_unallocated_prbs": idle_unalloc,
                "b0_inactive_prbs_allocated": inact_b0,
                "has_any_b0_idle_capacity": (idle_unalloc > 0 or inact_b0 > 0),
                "intra_win_arrival_e": intra_arrivals[0],
                "intra_win_arrival_u": intra_arrivals[1],
                "intra_win_arrival_m": intra_arrivals[2],
                "n_pure_idle_gain_arms": n_pg,
                "idle_order_pairs_nonzero": i_tot,
                "idle_order_ok_raw": i_raw,
                "idle_order_ok_sort": i_sort,
                "idle_order_ok_pava": i_pava,
                "idle_slope_pairs_nonzero": sl_tot,
                "idle_slope_ok_raw": sl_raw,
                "idle_slope_ok_sort": sl_sort,
                "idle_slope_ok_pava": sl_pava,
            })

    df_reg_a = pd.DataFrame(reg_a_rows)
    df_reg_a.to_csv(os.path.join(OUT_DIR, "t4_regime_a_ociosidade_estados.csv"), index=False)

    # --- Regime B: Pares (a1, a2) com a2 > a1 + Sign Accuracy + Spearman + Segurança da Doadora ---
    pair_rows = []
    donor_rows = []

    def _collect_dataset_pairs(ds_name: str, state_items: list[tuple[str, list[dict[str, Any]], list[np.ndarray], list[np.ndarray], Any, tuple[bool, bool, bool]]]) -> None:
        for sid, arms_list, f_list, sig_list, st_obj, act_mask in state_items:
            f_sort_list = [np.sort(f_list[i]) for i in range(3)]
            f_pava_list = [_pava_nondecreasing(f_list[i]) for i in range(3)]
            prop_vec = prop_active_split(st_obj, 53, act_mask)
            phys_snr_floors = tuple(_physical_donor_floor_est(st_obj, act_mask, i) for i in range(3))

            # Deduplicar por tupla de ação (para evitar viés de braços repetidos no Gate 1) e também guardar por slice
            unique_by_action: dict[tuple[int, int, int], dict[str, Any]] = {}
            for a_rec in arms_list:
                if a_rec["action"] not in unique_by_action:
                    unique_by_action[a_rec["action"]] = a_rec
            u_arms = list(unique_by_action.values())

            for s in range(3):
                # Extrair ações distintas de slice s e seu desfecho real observado;
                # cuidado: várias ações globais podem dar mesmo a_s com ações diferentes nas outras slices;
                # comparamos pares de REPLAYS distintos onde a_2[s] > a_1[s] no mesmo estado.
                obs_pts = []
                for ar in u_arms:
                    a_s = int(ar["action"][s])
                    y_s = _margin_from_kpis(s, ar["embb_thr"], ar["urllc_p95"], ar["mmtc_pdr"], ar["u_complete"])
                    if y_s is not None:
                        obs_pts.append((ar["arm"], ar["action"], a_s, float(y_s)))

                for i in range(len(obs_pts)):
                    for j in range(i + 1, len(obs_pts)):
                        arm1, act1, a1, y1 = obs_pts[i]
                        arm2, act2, a2, y2 = obs_pts[j]
                        if a1 == a2:
                            continue
                        if a1 > a2:
                            arm1, act1, a1, y1, arm2, act2, a2, y2 = arm2, act2, a2, y2, arm1, act1, a1, y1
                        # Agora a2 > a1 sempre
                        dy = float(y2 - y1)
                        df_raw = float(f_list[s][a2] - f_list[s][a1])
                        df_sort = float(f_sort_list[s][a2] - f_sort_list[s][a1])
                        df_pava = float(f_pava_list[s][a2] - f_pava_list[s][a1])
                        s_max = float(max(sig_list[s][a1], sig_list[s][a2]))
                        pair_rows.append({
                            "dataset": ds_name,
                            "state_id": sid,
                            "slice_id": s,
                            "slice_name": SLICE_NAMES[s],
                            "arm1": arm1, "arm2": arm2,
                            "a1": a1, "a2": a2, "da": a2 - a1,
                            "y1": y1, "y2": y2, "dy": dy,
                            "is_tie": abs(dy) < 1e-12,
                            "df_raw": df_raw,
                            "df_sort": df_sort,
                            "df_pava": df_pava,
                            "sigma_max_pair": s_max,
                            "in_support_015": s_max <= DEFAULT_SIGMA_MAX,
                        })

                # Doadora segura: todo par onde na base (a_hi) a slice era SEGURA (y_hi >= 0) e cedeu delta = a_hi - a_lo > 0 para a_lo
                for i in range(len(obs_pts)):
                    for j in range(len(obs_pts)):
                        if i == j:
                            continue
                        arm_hi, act_hi, a_hi, y_hi = obs_pts[i]
                        arm_lo, act_lo, a_lo, y_lo = obs_pts[j]
                        if a_hi <= a_lo or y_hi < 0.0:
                            continue
                        delta = a_hi - a_lo
                        f_lo_raw = float(f_list[s][a_lo])
                        f_lo_sort = float(f_sort_list[s][a_lo])
                        f_lo_pava = float(f_pava_list[s][a_lo])
                        sig_lo = float(sig_list[s][a_lo])
                        donor_harm = bool(y_lo < 0.0)
                        donor_rows.append({
                            "dataset": ds_name,
                            "state_id": sid,
                            "slice_id": s,
                            "slice_name": SLICE_NAMES[s],
                            "arm_base": arm_hi,
                            "arm_don": arm_lo,
                            "a_hi": a_hi,
                            "a_lo": a_lo,
                            "delta": delta,
                            "y_hi": y_hi,
                            "y_lo": y_lo,
                            "donor_harm": donor_harm,
                            "f_lo_raw": f_lo_raw,
                            "f_lo_sort": f_lo_sort,
                            "f_lo_pava": f_lo_pava,
                            "sig_lo": sig_lo,
                            "prop_active_floor": int(prop_vec[s]),
                            "phys_snr_floor": float(phys_snr_floors[s]),
                            "rule_q_raw_safe": f_lo_raw >= 0.0,
                            "rule_q_sort_safe": f_lo_sort >= 0.0,
                            "rule_q_pava_safe": f_lo_pava >= 0.0,
                            "rule_q_minus_sigma_safe": (f_lo_raw - sig_lo) >= 0.0,
                            "rule_q_and_sig015_safe": (f_lo_raw >= 0.0) and (sig_lo <= DEFAULT_SIGMA_MAX),
                            "rule_env_prop_active_safe": a_lo >= int(prop_vec[s]),
                            "rule_env_phys_snr_safe": float(a_lo) >= float(phys_snr_floors[s]),
                        })

    # Construir lista Gate 1
    g1_items = []
    for sid, grp in g1_metrics.groupby("state_id"):
        sn = gate1_snaps[sid]
        act_mask, _ = causal_active_slices(sn["state"])
        arms_l = []
        for rr in grp.itertuples():
            arms_l.append({
                "arm": str(rr.arm),
                "action": (int(rr.action_e), int(rr.action_u), int(rr.action_m)),
                "embb_thr": _safe_float(rr.embb_thr_mbps),
                "urllc_p95": _safe_float(rr.urllc_p95_ms),
                "mmtc_pdr": _safe_float(rr.mmtc_pdr_rolling),
                "u_complete": str(rr.urllc_pool_tag) == "CLOSED",
            })
        g1_items.append((sid, arms_l, sn["f"], sn["sigma"], sn["state"], act_mask))

    d2_a0_items = []
    d2_sh_items = []
    for sid, dn in sorted(d2_recon.items()):
        d2_a0_items.append((sid, d2_arm_rows[sid], dn["a0"]["f"], dn["a0"]["sigma"], dn["state"], dn["active_mask"]))
        d2_sh_items.append((sid, d2_arm_rows[sid], dn["b0rest_shadow"]["f"], dn["b0rest_shadow"]["sigma"], dn["state"], dn["active_mask"]))

    _collect_dataset_pairs("gate1_b0rest", g1_items)
    _collect_dataset_pairs("d2v2_pre_desync_a0_exact", d2_a0_items)
    _collect_dataset_pairs("d2v2_pre_desync_b0rest_shadow", d2_sh_items)

    df_pairs = pd.DataFrame(pair_rows)
    df_pairs.to_csv(os.path.join(OUT_DIR, "t4_regime_b_pares_replays_por_estado_slice.csv"), index=False)
    df_donor = pd.DataFrame(donor_rows)
    df_donor.to_csv(os.path.join(OUT_DIR, "t4_regime_b_falso_doador_pares_brutos.csv"), index=False)

    # Resumo Sign Accuracy & Spearman
    res_b_rows = []
    for (ds, s), g in df_pairs.groupby(["dataset", "slice_id"]):
        nz = g[~g["is_tie"]]
        tr = g[g["in_support_015"]]
        tr_nz = nz[nz["in_support_015"]]
        for curve_col, cname in (("df_raw", "bruto"), ("df_sort", "rearr_sort"), ("df_pava", "rearr_pava")):
            rho_all, p_all = _spearman(g[curve_col].values, g["dy"].values)
            rho_nz, p_nz = _spearman(nz[curve_col].values, nz["dy"].values) if len(nz) >= 3 else (None, None)
            rho_tr, _ = _spearman(tr_nz[curve_col].values, tr_nz["dy"].values) if len(tr_nz) >= 3 else (None, None)
            # Sign accuracy estrita (em dy != 0) e não-estrita (df * dy >= 0)
            sa_strict = float(np.mean((nz[curve_col].values * nz["dy"].values) > 0.0)) if len(nz) else None
            sa_nonstrict = float(np.mean((g[curve_col].values * g["dy"].values) >= 0.0)) if len(g) else None
            sa_tr_strict = float(np.mean((tr_nz[curve_col].values * tr_nz["dy"].values) > 0.0)) if len(tr_nz) else None
            # Spearman dentro de cada estado (para estados com >= 3 pares dy != 0 e variação em ambos os vetores)
            within_rhos: list[float] = []
            for _, sg in nz.groupby("state_id"):
                if len(sg) >= 3 and sg["dy"].nunique() >= 2 and sg[curve_col].nunique() >= 2:
                    r_st, _ = _spearman(sg[curve_col].values, sg["dy"].values)
                    if r_st is not None and math.isfinite(r_st):
                        within_rhos.append(float(r_st))
            res_b_rows.append({
                "dataset": ds,
                "slice_id": int(s),
                "slice_name": SLICE_NAMES[int(s)],
                "curve_variant": cname,
                "n_pairs": int(len(g)),
                "n_pairs_nonzero_dy": int(len(nz)),
                "frac_ties_dy0": float(g["is_tie"].mean()),
                "sign_acc_strict_nonzero": sa_strict,
                "sign_acc_nonstrict_all": sa_nonstrict,
                "spearman_rho_all": rho_all,
                "spearman_p_all": p_all,
                "spearman_rho_nonzero": rho_nz,
                "spearman_p_nonzero": p_nz,
                "n_pairs_in_support_nz": int(len(tr_nz)),
                "sign_acc_in_support_nz": sa_tr_strict,
                "spearman_rho_in_support_nz": rho_tr,
                "n_states_within_spearman_nz": int(len(within_rhos)),
                "mean_within_state_spearman_nz": float(np.mean(within_rhos)) if within_rhos else None,
            })
    df_res_b = pd.DataFrame(res_b_rows)
    df_res_b.to_csv(os.path.join(OUT_DIR, "t4_regime_b_resumo_sign_spearman.csv"), index=False)

    # Resumo Falso Doador Seguro
    rule_cols = [
        ("rule_q_raw_safe", "quantil_bruto_ge_0"),
        ("rule_q_sort_safe", "quantil_sort_ge_0"),
        ("rule_q_pava_safe", "quantil_pava_ge_0"),
        ("rule_q_minus_sigma_safe", "quantil_menos_sigma_ge_0"),
        ("rule_q_and_sig015_safe", "quantil_ge_0_e_sigma_le_015"),
        ("rule_env_prop_active_safe", "envelope_prop_active"),
        ("rule_env_phys_snr_safe", "envelope_fisico_fila_snr"),
    ]
    fd_rows = []
    for (ds, s), g in df_donor.groupby(["dataset", "slice_id"]):
        n_tot = int(len(g))
        n_true_safe = int((~g["donor_harm"]).sum())
        n_true_harm = int(g["donor_harm"].sum())
        for rcol, rname in rule_cols:
            auth = g[g[rcol]]
            n_auth = int(len(auth))
            n_false_safe = int(auth["donor_harm"].sum())
            fdr = float(n_false_safe / n_auth) if n_auth > 0 else None
            recall_safe = float((~auth["donor_harm"]).sum() / n_true_safe) if n_true_safe > 0 else None
            fd_rows.append({
                "dataset": ds,
                "slice_id": int(s),
                "slice_name": SLICE_NAMES[int(s)],
                "rule": rname,
                "n_pairs_base_safe": n_tot,
                "n_true_safe": n_true_safe,
                "n_true_harm": n_true_harm,
                "n_authorized": n_auth,
                "n_false_safe_harm": n_false_safe,
                "false_safe_donor_rate": fdr,
                "safe_donation_recall": recall_safe,
            })
    df_fd = pd.DataFrame(fd_rows)
    df_fd.to_csv(os.path.join(OUT_DIR, "t4_regime_b_falso_doador_seguro.csv"), index=False)

    # Resumo Regime A agregado
    reg_a_sum = {}
    for ds, g in df_reg_a.groupby("dataset"):
        tot_pairs = int(g["idle_order_pairs_nonzero"].sum())
        tot_slope = int(g["idle_slope_pairs_nonzero"].sum())
        reg_a_sum[ds] = {
            "n_states": int(len(g)),
            "n_b0_unalloc_gt0": int((g["b0_unallocated_prbs"] > 0).sum()),
            "n_b0_inactive_alloc_gt0": int((g["b0_inactive_prbs_allocated"] > 0).sum()),
            "n_any_b0_idle": int(g["has_any_b0_idle_capacity"].sum()),
            "intra_win_arrival_counts": {
                "eMBB": int(g["intra_win_arrival_e"].sum()),
                "URLLC": int(g["intra_win_arrival_u"].sum()),
                "mMTC": int(g["intra_win_arrival_m"].sum()),
            },
            "idle_order_pairs_nonzero": tot_pairs,
            "idle_order_acc_raw": float(g["idle_order_ok_raw"].sum() / tot_pairs) if tot_pairs else None,
            "idle_order_acc_sort": float(g["idle_order_ok_sort"].sum() / tot_pairs) if tot_pairs else None,
            "idle_order_acc_pava": float(g["idle_order_ok_pava"].sum() / tot_pairs) if tot_pairs else None,
            "idle_slope_pairs_nonzero": tot_slope,
            "idle_slope_acc_raw": float(g["idle_slope_ok_raw"].sum() / tot_slope) if tot_slope else None,
            "idle_slope_acc_sort": float(g["idle_slope_ok_sort"].sum() / tot_slope) if tot_slope else None,
            "idle_slope_acc_pava": float(g["idle_slope_ok_pava"].sum() / tot_slope) if tot_slope else None,
        }

    return {"regime_a_summary": reg_a_sum, "n_pairs_total": int(len(df_pairs)), "n_donor_pairs_total": int(len(df_donor))}


# ---------------------------------------------------------------------------
# Tarefa 5: Índice de Confiabilidade Prequencial vs. Incerteza Geométrica sigma
# ---------------------------------------------------------------------------

def _eval_run_offline_outcomes(run_dir: str) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    """Compute exact full-episode offline ground-truth outcomes per governed window gov_w."""
    kpm_path = _find_single(run_dir, "sqr_kpm_history.csv")
    del_path = _find_single(run_dir, "sqr_packet_delays.csv")
    kpm_df = pd.read_csv(kpm_path)
    del_df = pd.read_csv(del_path)

    # Arrival-window URLLC offline full aggregates
    u_del = del_df[del_df["sliceId"] == 1].copy()
    u_by_win: dict[int, tuple[float, int]] = {}
    if len(u_del):
        u_del["win"] = np.floor(u_del["txSec"].values * 10.0).astype(int)
        u_del["ms"] = u_del["delaySec"].values * 1000.0
        for w, grp in u_del.groupby("win"):
            arr = grp["ms"].values
            u_by_win[int(w)] = (float(np.percentile(arr, 95.0)), int(arr.size))

    # KPM per-window eMBB, URLLC tx count, and 1s rolling mMTC PDR
    mmtc_roll: deque[tuple[float, float]] = deque(maxlen=10)
    out: dict[int, dict[str, Any]] = {}
    for ts, grp in kpm_df.groupby("timestamp", sort=True):
        w = int(ts) // 100 - 1
        g0_dem = grp[(grp["sliceId"] == 0) & (grp["dTxBytes"] > 0)]
        embb_thr = float(g0_dem["windowThroughputMbps"].median()) if len(g0_dem) else None

        g1 = grp[grp["sliceId"] == 1]
        u_tx = int(g1["dTxPackets"].sum()) if len(g1) else 0
        u_rx_n = u_by_win[w][1] if w in u_by_win else 0
        u_closed = (u_tx > 0 and u_rx_n >= u_tx)
        u_p95 = u_by_win[w][0] if u_closed else None

        g2 = grp[grp["sliceId"] == 2]
        m_tx = float(g2["dTxPackets"].sum()) if len(g2) else 0.0
        m_rx = float(g2["dRxPackets"].sum()) if len(g2) else 0.0
        mmtc_roll.append((m_tx, m_rx))
        t10 = sum(a for a, _ in mmtc_roll)
        r10 = sum(b for _, b in mmtc_roll)
        mmtc_pdr = (r10 / t10) if t10 > 0 else None

        out[w] = {
            "m_e": embb_margin(embb_thr),
            "m_u_closed": urllc_margin(u_p95) if u_closed else None,
            "u_tx": u_tx,
            "u_rx": u_rx_n,
            "u_closed": u_closed,
            "u_p95_allrx": u_by_win[w][0] if w in u_by_win else None,
            "m_m": mmtc_margin(mmtc_pdr),
        }
    return out, kpm_df


def run_task5() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[T5] Avaliando índice prequencial causal vs. incerteza geométrica sigma em b0_traced_gate1...")
    parity = json.loads(open(os.path.join(DEC_DIR, "A5_GATE1_B0_PARITY_REPORT.json"), encoding="utf-8").read())
    traced_base = os.path.join(DEC_DIR, "b0_traced_gate1")

    step_rows = []
    for pr in parity["runs"]:
        tag = str(pr["tag"])
        lam = int(pr["lam"])
        seed = int(pr["seed"])
        traced_lag = pr["traced_first_ipc_lag_step"]
        clean_cut = int(pr["clean_cut_step"])
        rdir = os.path.join(traced_base, tag)

        decs = [json.loads(l) for l in open(os.path.join(rdir, "decisions.jsonl")) if l.strip()]
        udels = [json.loads(l) for l in open(os.path.join(rdir, "urllc_delayed_updates.jsonl")) if l.strip()]
        off_outcomes, _ = _eval_run_offline_outcomes(rdir)

        dec_by_gw = {}
        dec_by_ow = {}
        for d in decs:
            sw = int(d["state_window_index"])
            gw = sw + 1
            dec_by_gw[gw] = d
            if d.get("outcome_window_index") is not None:
                dec_by_ow[int(d["outcome_window_index"])] = d

        udels_by_obs: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for u in udels:
            udels_by_obs[int(u["observed_at_window"])].append(u)

        # Histórico estritamente causal por slice
        # Cada entrada fechada: (f_exec, y_obs, tau, overpred_flag, crit_flag)
        closed_hist: dict[int, list[tuple[float, float, float]]] = {0: [], 1: [], 2: []}
        # Histórico de demanda URLLC enviada para calcular taxa causal de pendência/expiração:
        # map w -> {"status": "CLOSED"|"OPEN"|"EXPIRED", "obs_w": int|None}
        urllc_demand_wins: dict[int, str] = {}
        dec_by_gov_w_meta: dict[int, tuple[float, float]] = {}  # gov_w -> (f_exec_u, tau_u)

        # Carregar causal_trace.jsonl.gz em streaming
        with gzip.open(os.path.join(rdir, "causal_trace.jsonl.gz"), "rt", encoding="utf-8") as gz:
            for idx, ln in enumerate(gz):
                if idx == 0 or not ln.strip():
                    continue
                tw = json.loads(ln)
                step_idx = int(tw["step_index"])
                if traced_lag is not None and step_idx >= int(traced_lag):
                    break
                sw = int(tw["window_index"])
                gw = int(tw["governed_window_index"])
                psm = sorted(tw["per_slice_model"], key=lambda e: int(e["slice_id"]))
                a_exec = tuple(int(v) for v in tw["a_exec_per_slice"])
                st = tw["state"]

                # Incorporar todas as observações causais ocorridas ANTES ou EXATAMENTE em sw
                # (1) decisões anteriores ainda não absorvidas cujo outcome_window_index <= sw
                while True:
                    # Processemos em ordem cronológica os outcome_window_index <= sw ainda não absorvidos
                    break  # faremos via cursor monotônico abaixo!
                # Melhor: usar cursor de decisions.jsonl com sim_time <= dec_at_ms
                break

        # Segunda passagem limpa, cursando decisions.jsonl e causal_trace em paralelo por step_index
        # Ler trace records pré-lag em dict por governed_window_index
        trace_by_gw: dict[int, dict[str, Any]] = {}
        with gzip.open(os.path.join(rdir, "causal_trace.jsonl.gz"), "rt", encoding="utf-8") as gz:
            for idx, ln in enumerate(gz):
                if idx == 0 or not ln.strip():
                    continue
                tw = json.loads(ln)
                step_idx = int(tw["step_index"])
                if traced_lag is not None and step_idx >= int(traced_lag):
                    break
                trace_by_gw[int(tw["governed_window_index"])] = tw

        closed_hist = {0: [], 1: [], 2: []}
        urllc_demand_wins = {}
        dec_exec_by_ow: dict[int, list[tuple[float, float]]] = {}  # ow -> [(f_exec_i, tau_i) for i in 0..2]
        last_closed_urllc_w: int | None = None

        # Avançar sobre decisions em ordem estrita de decision_index
        processed_udels_obs: set[int] = set()
        for d in decs:
            step_idx = int(d["decision_index"])
            if traced_lag is not None and step_idx >= int(traced_lag):
                break
            sw = int(d["state_window_index"])
            gw = sw + 1

            # 1) Primeiro absorver outcome de d anterior? Cuidado: em decisions.jsonl,
            # cada linha `d` tem `outcome_window_index = gw` que só é observado DEPOIS da decisão `d`!
            # Os updates observados ANTES da decisão de `gw` correspondem a:
            # - decisões anteriores `d_prev` cujo `outcome_window_index <= sw`
            # - `urllc_delayed_updates` com `observed_at_window <= sw`
            for ow_prev, d_prev in list(dec_by_ow.items()):
                if ow_prev <= sw and ow_prev not in dec_exec_by_ow:
                    # Extrair predição e tau usados em d_prev
                    gw_prev = int(d_prev["state_window_index"]) + 1
                    tw_p = trace_by_gw.get(gw_prev)
                    if tw_p is not None:
                        psm_p = sorted(tw_p["per_slice_model"], key=lambda e: int(e["slice_id"]))
                        ae_p = [int(v) for v in tw_p["a_exec_per_slice"]]
                        ft_p = [(float(psm_p[i]["f_candidates"][ae_p[i]]), float(psm_p[i]["tau"])) for i in range(3)]
                    else:
                        # Fase de bootstrap antes do trace: usar predicted_safe_after_projection e tau0=0.05
                        ft_p = [(float(d_prev["predicted_safe_after_projection"][i]), 0.05) for i in range(3)]
                    dec_exec_by_ow[ow_prev] = ft_p

                    m_on = d_prev["sla_outcome_margins"]
                    for s in (0, 2):
                        if m_on[s] is not None:
                            closed_hist[s].append((ft_p[s][0], float(m_on[s]), ft_p[s][1]))
                    ev_u = d_prev["sla_evidence"][1] if "sla_evidence" in d_prev else None
                    if m_on[1] is not None:
                        closed_hist[1].append((ft_p[1][0], float(m_on[1]), ft_p[1][1]))
                        urllc_demand_wins[ow_prev] = "CLOSED"
                        last_closed_urllc_w = ow_prev
                    elif ev_u == "PENDING_OBSERVATION":
                        urllc_demand_wins[ow_prev] = "OPEN"

            # Drenar urllc_delayed_updates até sw inclusive
            for obs_w in sorted(k for k in udels_by_obs if k <= sw and k not in processed_udels_obs):
                processed_udels_obs.add(obs_w)
                for u in udels_by_obs[obs_w]:
                    uw = int(u["w"])
                    if u.get("updated") is True and u.get("p95_ms") is not None:
                        urllc_demand_wins[uw] = "CLOSED"
                        last_closed_urllc_w = max(last_closed_urllc_w or uw, uw)
                        if uw in dec_exec_by_ow:
                            f_u, t_u = dec_exec_by_ow[uw][1]
                            m_u = (10.0 - float(u["p95_ms"])) / 10.0
                            closed_hist[1].append((f_u, float(m_u), t_u))
                    elif u.get("reason") == "expired_vacuous_R19":
                        urllc_demand_wins[uw] = "EXPIRED"

            # 2) Agora avaliar exclusivamente se gw tem snapshot causal B0-rest (pós-warmup SQR_HET)
            tw = trace_by_gw.get(gw)
            if tw is None:
                continue
            off = off_outcomes.get(gw)
            if off is None:
                continue

            psm = sorted(tw["per_slice_model"], key=lambda e: int(e["slice_id"]))
            ae = [int(v) for v in tw["a_exec_per_slice"]]
            st = tw["state"]

            # Calcular métricas prequenciais causais para janelas W_preq in (20, 50)
            u_dem_vals = list(urllc_demand_wins.values())
            for W in (20, 50):
                u_recent_dem = u_dem_vals[-W:]
                c_pend = (
                    float(sum(1 for v in u_recent_dem if v in ("OPEN", "EXPIRED")) / len(u_recent_dem))
                    if u_recent_dem else 0.0
                )
                lag_closed = float(sw - last_closed_urllc_w) if last_closed_urllc_w is not None else float(W)
                lag_norm = min(1.0, lag_closed / float(W))

                for s in range(3):
                    # Target offline real de avaliação em gw
                    if s == 0:
                        y_off = off["m_e"]
                        viol_or_unclosed = (y_off < 0.0) if y_off is not None else None
                    elif s == 1:
                        y_off = off["m_u_closed"]
                        # Só avaliar em janelas com demanda URLLC real
                        if off["u_tx"] <= 0:
                            y_off = None
                            viol_or_unclosed = None
                        else:
                            viol_or_unclosed = bool((not off["u_closed"]) or (y_off is not None and y_off < 0.0))
                    else:
                        y_off = off["m_m"]
                        viol_or_unclosed = (y_off < 0.0) if y_off is not None else None

                    if y_off is None and not (s == 1 and viol_or_unclosed is True):
                        continue

                    f_now = float(psm[s]["f_candidates"][ae[s]])
                    sig_now = float(psm[s]["sigma_candidates"][ae[s]])
                    hist_s = closed_hist[s][-W:]
                    if len(hist_s) > 0:
                        pb_list = [_pinball_loss(yo, fo, ta) for fo, yo, ta in hist_s]
                        ov_rate = float(np.mean([1.0 if yo < fo else 0.0 for fo, yo, _ in hist_s]))
                        ov_mag = float(np.mean([max(0.0, fo - yo) for fo, yo, _ in hist_s]))
                        viol_hist_rate = float(np.mean([1.0 if yo < 0.0 else 0.0 for _, yo, _ in hist_s]))
                        pb_mean = float(np.mean(pb_list))
                    else:
                        pb_mean = 0.0
                        ov_rate = 0.0
                        ov_mag = 0.0
                        viol_hist_rate = 0.0

                    if s == 1:
                        u_preq = 0.5 * min(1.0, pb_mean / 0.5) + 0.5 * c_pend
                        u_preq_with_lag = 0.4 * min(1.0, pb_mean / 0.5) + 0.4 * c_pend + 0.2 * lag_norm
                        u_preq_asym = 0.5 * viol_hist_rate + 0.3 * ov_rate + 0.2 * c_pend
                    else:
                        u_preq = 0.5 * min(1.0, pb_mean / 0.5) + 0.5 * ov_rate
                        u_preq_with_lag = u_preq
                        u_preq_asym = 0.6 * viol_hist_rate + 0.4 * ov_rate

                    abs_err = abs(float(y_off) - f_now) if y_off is not None else None
                    sla_viol_closed = bool(y_off < 0.0) if y_off is not None else None

                    step_rows.append({
                        "run_tag": tag,
                        "lam": lam,
                        "seed": seed,
                        "step_index": step_idx,
                        "within_clean_cut": step_idx < clean_cut,
                        "gov_w": gw,
                        "W_preq": W,
                        "slice_id": s,
                        "slice_name": SLICE_NAMES[s],
                        "n_hist_closed": len(hist_s),
                        "sigma_exec": sig_now,
                        "f_exec": f_now,
                        "pinball_roll": pb_mean,
                        "overpred_rate_roll": ov_rate,
                        "overpred_mag_roll": ov_mag,
                        "viol_hist_rate_roll": viol_hist_rate,
                        "urllc_censored_frac_roll": c_pend if s == 1 else 0.0,
                        "urllc_lag_closed": lag_closed if s == 1 else 0.0,
                        "u_preq": u_preq,
                        "u_preq_plus_lag": u_preq_with_lag,
                        "u_preq_asym": u_preq_asym,
                        "y_off": y_off,
                        "abs_err": abs_err,
                        "sla_viol_closed": sla_viol_closed,
                        "sla_viol_or_unclosed": viol_or_unclosed,
                    })

    df_t5 = pd.DataFrame(step_rows)
    df_t5.to_csv(os.path.join(OUT_DIR, "t5_prequencial_vs_sigma_por_slice_run.csv"), index=False)

    res_rows = []
    for W in (20, 50):
        for cut_scope, mask_cut in (("pre_traced_lag", df_t5["W_preq"] == W), ("pre_clean_cut_strict", (df_t5["W_preq"] == W) & df_t5["within_clean_cut"])):
            sub_w = df_t5[mask_cut]
            for s in (0, 1, 2):
                gs = sub_w[sub_w["slice_id"] == s]
                gs_cl = gs[gs["y_off"].notna()].copy()
                # Target 1: erro absoluto |y_off - f_exec| em janelas fechadas
                # Target 2: violação SLA fechada (sla_viol_closed)
                # Target 3 (URLLC): violação OU perda não-fechada (sla_viol_or_unclosed)
                predictors = [
                    ("sigma_exec", "sigma_kernel"),
                    ("pinball_roll", "pinball_rolling"),
                    ("overpred_rate_roll", "overpred_rate_rolling"),
                    ("overpred_mag_roll", "overpred_mag_rolling"),
                    ("viol_hist_rate_roll", "viol_hist_rate_rolling"),
                    ("u_preq", "U_preq_composto"),
                    ("u_preq_asym", "U_preq_assimetrico"),
                ]
                if s == 1:
                    predictors.append(("urllc_censored_frac_roll", "urllc_censorship_frac"))
                    predictors.append(("u_preq_plus_lag", "U_preq_plus_lag"))

                for pcol, pname in predictors:
                    rho_err, p_err = _spearman(gs_cl[pcol].values, gs_cl["abs_err"].values)
                    y_cl = gs_cl["sla_viol_closed"].astype(int).values
                    auroc_cl = _auroc(gs_cl[pcol].values, y_cl)
                    rho_v_cl, _ = _spearman(gs_cl[pcol].values, y_cl)

                    gs_vu = gs[gs["sla_viol_or_unclosed"].notna()]
                    y_vu = gs_vu["sla_viol_or_unclosed"].astype(int).values
                    auroc_vu = _auroc(gs_vu[pcol].values, y_vu) if s == 1 else auroc_cl
                    rho_vu, _ = _spearman(gs_vu[pcol].values, y_vu) if s == 1 else (rho_v_cl, None)

                    res_rows.append({
                        "cut_scope": cut_scope,
                        "W_preq": W,
                        "slice_id": s,
                        "slice_name": SLICE_NAMES[s],
                        "predictor": pname,
                        "n_closed": int(len(gs_cl)),
                        "n_viol_closed": int(y_cl.sum()),
                        "n_dem_all": int(len(gs_vu)),
                        "n_viol_or_unclosed": int(y_vu.sum()),
                        "spearman_abs_err": rho_err,
                        "spearman_abs_err_p": p_err,
                        "auroc_sla_viol_closed": auroc_cl,
                        "spearman_sla_viol_closed": rho_v_cl,
                        "auroc_viol_or_unclosed": auroc_vu,
                        "spearman_viol_or_unclosed": rho_vu,
                    })

    df_t5_res = pd.DataFrame(res_rows)
    df_t5_res.to_csv(os.path.join(OUT_DIR, "t5_prequencial_resumo_auroc_spearman.csv"), index=False)
    return df_t5, df_t5_res


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    t1_sum = run_task1()
    df_t2_runs, df_t2_states, _ = run_task2()
    d2_recon = reconstruct_d2v2_pre_desync_curves(df_t2_states)
    gate1_snaps = _load_gate1_snapshots()
    df_t3, df_t3_sum = run_task3(gate1_snaps, d2_recon)
    t4_sum = run_task4(gate1_snaps, df_t2_states, d2_recon)
    df_t5, df_t5_res = run_task5()

    # Consolidate master JSON summary for exact cross-referencing
    master = {
        "outputs_dir": OUT_DIR,
        "checkpoints_pkl_opened": PKL_OPEN_GUARD_COUNT,
        "task1": t1_sum,
        "task2": {
            "d1v9_total_runs": int(len(df_t2_runs)),
            "d1v9_clean_runs": int(df_t2_runs["clean"].sum()),
            "d1v9_desync_runs": int((~df_t2_runs["clean"]).sum()),
            "d2v2_total_states": int(len(df_t2_states)),
            "d2v2_pre_desync": int((df_t2_states["desync_status"] == "PRE_DESYNC").sum()),
            "d2v2_post_desync": int((df_t2_states["desync_status"] == "POST_DESYNC").sum()),
            "recoverable_strict_status": df_t2_states[df_t2_states["label"] == "RECOVERABLE_STRICT"][
                ["state_id", "t_win", "onset_outcome_window", "desync_status"]
            ].to_dict(orient="records"),
            "d2v2_reconstruction_max_err_before": max(v["parity_err_before"] for v in d2_recon.values()),
            "d2v2_reconstruction_max_err_after": max(v["parity_err_after"] for v in d2_recon.values()),
        },
        "task3": {
            "n_curve_rows": int(len(df_t3)),
            "summary_rows": df_t3_sum.to_dict(orient="records"),
        },
        "task4": t4_sum,
        "task5": {
            "n_step_rows": int(len(df_t5)),
            "summary_rows": df_t5_res.to_dict(orient="records"),
        },
    }
    with open(os.path.join(OUT_DIR, "diag_exploratorio_v1_resumo.json"), "w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=2, default=float)

    print("[DONE] Todas as tarefas T1-T5 concluídas com sucesso!")
    print(f"  T2 D2-v2 PRE_DESYNC={master['task2']['d2v2_pre_desync']}/30, max_recon_err={master['task2']['d2v2_reconstruction_max_err_before']:.3e}")
    print(f"  T3 curve rows={master['task3']['n_curve_rows']}")
    print(f"  T4 regime A summary={t4_sum['regime_a_summary']}")
    print(f"  T5 step rows={master['task5']['n_step_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
