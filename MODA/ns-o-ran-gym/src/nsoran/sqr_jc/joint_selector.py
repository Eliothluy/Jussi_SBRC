"""joint_selector.py — B1: seleção conjunta com os MESMOS modelos SQR-HET.

PROTOCOL_FROZEN §5 (congelado) + D3_execution_declaration (4 cuidados):
- carrega os checkpoints do run B0 (sem retreino — cuidado 1);
- ranking LEXICOGRÁFICO congelado: max critical(a)=min_i g_i(a_i) sobre
  slices STRICT-avaliáveis? NÃO — o seletor é CEGO a rótulos (cuidado 4):
  usa apenas z_i pré-decisão.  critical usa TODAS as 3 slices (o seletor
  não sabe quais serão avaliáveis no futuro — usar o futuro seria vazamento);
- g_i(a_i) = f_τ(z_i, a_i) na escala da MARGEM (maior=melhor; Λ=0);
  utility_i(a_i) = f_τ(z_i,a_i) − γ·a_i/C (o objetivo original do B0);
- A_common = simplex Σ≤53 (cuidado 3); tabela pré-computada 54×3/run;
- desempates determinísticos: Σa ↑, índice lexicográfico ↑;
- traço completo por decisão (sem rótulo de replay em nenhuma entrada).
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

GYM = "/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym"
sys.path.insert(0, os.path.join(GYM, "src"))


class JointSelector:
    """B1 sobre os checkpoints B0 de UM run (freeze: sem updates)."""

    def __init__(self, controllers, het, c: int = 53):
        self.controllers = controllers      # dict sid -> SqrSliceController
        self.het = het                      # SqrHetController (slice_state)
        self.c = int(c)
        self._tbl = None                    # (3, 54) g; (3, 54) utility
        self._z_ref = None

    # -- construção da tabela p/ UM estado (§9.5: 54×3 inferências) ------
    def build_table(self, state):
        zs = [self.het.slice_state(state, sid) for sid in range(3)]
        g = np.zeros((3, self.c + 1))
        u = np.zeros((3, self.c + 1))
        gamma = next(iter(self.controllers.values())).gamma
        for sid in range(3):
            sc = self.controllers[sid].score_candidates(zs[sid])
            g[sid] = sc["f"][: self.c + 1]
            u[sid] = sc["scores"][: self.c + 1]
        self._tbl, self._u_tbl = g, u
        self._zs = zs
        return g, u

    # -- seleção (ranking lexicográfico §5.4) -----------------------------
    def select(self, state):
        """-> (action, trace).  Cego a rótulos: consome apenas o estado e
        os modelos; critical sobre as 3 slices (sem saber o futuro)."""
        g, u = self.build_table(state)
        # enumeração direta sobre as tabelas: 27.720 triplas, custo trivial
        cands = []
        for ae in range(self.c + 1):
            for au in range(self.c + 1 - ae):
                for am in range(self.c + 1 - ae - au):
                    v = (ae, au, am)
                    crit = min(g[0][ae], g[1][au], g[2][am])
                    eff = u[0][ae] + u[1][au] + u[2][am]
                    cands.append((-crit, -eff, ae + au + am, ae, au, am))
        cands.sort()
        ae, au, am = cands[0][3], cands[0][4], cands[0][5]
        action = (ae, au, am)
        trace = {
            "b1_action": list(action),
            "critical": -cands[0][0],
            "efficient": -cands[0][1],
            "sum": ae + au + am,
            "g_at_choice": [float(g[0][ae]), float(g[1][au]), float(g[2][am])],
        }
        return action, trace


def load_run_controllers(run_dir: str):
    """Reconstrói os 3 SqrSliceController + SqrHetController do checkpoint
    do run B0 (freeze; cuidado 1) — build + restore (topologia validada)."""
    ckpt_path = os.path.join(run_dir, "sqr_checkpoint.pkl")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint B0 ausente: {ckpt_path}")
    from nsoran.sqr.checkpoint import load_checkpoint, restore_controllers
    from nsoran.sqr.sqr_controller import (
        SqrSliceController, TAU_INIT, TAU_STEP, MARGIN_INIT, MARGIN_BOUNDS)
    from nsoran.sqr_jc.feature_scaler import FeatureScaler
    from nsoran.sqr_jc.sqr_het import SqrHetController
    payload = load_checkpoint(ckpt_path)
    meta = payload["meta"]
    # o checkpoint congelou o C EFETIVO da telemetria (53 — R01); construir
    # com o MESMO valor senão o restore falha por topologia (lição D3)
    controllers = {
        sid: SqrSliceController(
            state_dim=6, n_prbs=53,
            gamma=float(meta.get("gamma", 1)), sla_threshold=0.0,
            tau=TAU_INIT, tau_step=TAU_STEP, margin=MARGIN_INIT,
            margin_bounds=MARGIN_BOUNDS, violation_window=100,
            epsilon=0.1, temperature=8.0, knn_k=10,
            seed=int(meta.get("seed", 1)) + 1000 * sid)
        for sid in range(3)
    }
    restore_controllers(payload, controllers)
    scaler = FeatureScaler.load_manifest(
        "/home/eliothluy/Documentos/Jussi_SBRC/results/sqr_jc_final/"
        "03_calibration/feature_scaler_manifest.yaml")
    het = SqrHetController(tuple(controllers[sid] for sid in range(3)),
                           scaler, next(iter(controllers.values())).n_prbs)
    return controllers, het, meta
