"""Joint-feasibility oracle (Rodada 2, Etapa 2.4 — Despacho I, offline).

Offline oracle over the frozen Etapa-2.3 campaign data: classifies, per
100 ms decision window, whether the observed network state still admits at
least one PRB-quota allocation that satisfies the eMBB, URLLC and mMTC SLAs
SIMULTANEOUSLY (protocol ``docs/artigo/rodada_2_etapa_2_4_oracle_viabilidade.md``).

Modules
-------
- ``dataset``: per-window dataset builder over the 30 runs of the 2.3 sweep
  (``hetero_sweep`` = equal-split arm, ``hetero_sweep_prop`` = prop arm).
- ``models``: semi-analytic KPI models f_eMBB / f_URLLC / f_mMTC (stratum
  lookups calibrated on the data, leave-one-seed-out validation) + support
  region bookkeeping.
- ``simplex``: admissible-action enumeration (a_e, a_u, a_m), sum <= C = 53,
  feasibility sets F_t, best-degradation action a* (neutral weights w_i = 1).
- ``pipeline``: metrics (P_inf, AVR, UVR, |F_t|, D_min), sanity checks SC1-4,
  hypotheses H2.4-1..4, tables/figures/artifacts of protocol section 28.

NO ns-3 simulation runs here (offline oracle only; executable replay is the
Despacho II ``executable_validation/`` stage).  NO changes to the scheduler,
deadlines or scenario.  New premises enter ``MODA/docs/sqr/assumptions_register.md``
tagged ``[ASSUMIDO-ORACLE]``.
"""

from nsoran.sqr.oracle.dataset import SLA_EMBB_MBPS, SLA_URLLC_MS, SLA_MMTC_PDR, build_window_dataset
from nsoran.sqr.oracle.models import EmmbShareLookup, MmtcConstantPdr, UrllcStrataLookup
from nsoran.sqr.oracle.simplex import enumerate_actions

__all__ = [
    "SLA_EMBB_MBPS",
    "SLA_URLLC_MS",
    "SLA_MMTC_PDR",
    "build_window_dataset",
    "EmmbShareLookup",
    "UrllcStrataLookup",
    "MmtcConstantPdr",
    "enumerate_actions",
]
