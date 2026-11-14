"""Frozen feature normalization (§13 hard gate).

Two normalization families, exactly as the protocol mandates:

1. **Physical/semantic scale** (natural bounds exist — inherited or defined):
   - KPI features  -> /SLA,  clipped to [0, 1]   (eMBB /3.5, URLLC /10 ms)
   - PDR           -> natural [0, 1]
   - SNR           -> /30 dB clip [0, 1]         (audited R09 scale — the
     inherited SQR normalization is kept, protocol §13 "Features herdadas
     devem manter a mesma normalização")
   - allocation    -> /C
   - miss rate, violation flags -> natural [0, 1]

2. **CAL percentile scale** for features without a natural cap (counts and
   queues — §13 rule): fit p1/p99 on CAL data ONLY, clip to [p1, p99],
   map linearly to [0, 1], then FREEZE.  Never refit on PILOT/MAIN.
   p1 == p99 (degenerate feature in CAL) falls back to identity-to-0
   (x/p99 clipped) to avoid division by zero, with the degeneracy flagged.

The frozen artifact is ``feature_scaler_manifest.yaml`` (§13); the scaler
round-trips through it via the deterministic YAML subset module, and
``frozen`` scalers REFUSE to refit (fail loud, never silently).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from nsoran.sqr_jc.joint_state import FEATURE_INDEX, JOINT_STATE_DIM
from nsoran.sqr_jc.sla_margin import SLA_EMBB_MBPS, SLA_URLLC_MS
from nsoran.sqr_jc.yaml_subset import dump, load

SCALER_MANIFEST_FORMAT = "sqrjc_feature_scaler_manifest-1"

#: Feature -> rule table (documented normalization of every z_joint index).
#: keys of FEATURE_INDEX; rule ∈ {"cal_p1p99", "kpi_div_sla", "kpi_div_deadline",
#: "unit01", "snr_r09", "alloc_div_c"}.
PHYSICAL_RULES: dict[str, str] = {
    "kpi_e": "kpi_div_sla",       # Mbps / 3.5
    "kpi_u": "kpi_div_deadline",  # ms / 10
    "kpi_m": "unit01",            # PDR fraction
    "snr_e": "snr_r09",           # dB / 30 clip
    "snr_u": "snr_r09",
    "snr_m": "snr_r09",
    "prev_a_e": "alloc_div_c",    # a / C (§13 allocation rule)
    "prev_a_u": "alloc_div_c",
    "prev_a_m": "alloc_div_c",
    "viol_e": "unit01",
    "viol_u": "unit01",
    "viol_m": "unit01",
    "urllc_miss": "unit01",       # fraction (not %)
    "urllc_p99": "kpi_div_deadline",
}
CAL_RULE_FEATURES = (
    "n_e", "n_u", "n_m",
    "queue_e", "queue_u", "queue_m",
    "offered_e", "offered_u", "offered_m",
)

SNR_SCALE_DB = 30.0  # audited R09 scale (sqr_env.py:80)


class FeatureScaler:
    """Per-feature normalizer; fit on CAL only, then frozen."""

    def __init__(self, c: int, frozen: bool = False) -> None:
        self.c = int(c)
        if self.c <= 0:
            raise ValueError(f"c must be positive, got {c}")
        self.frozen = bool(frozen)
        # CAL p1/p99 per CAL-rule feature (raw physical units).
        self._p1: dict[str, float] = {}
        self._p99: dict[str, float] = {}
        self._degenerate: set[str] = set()
        self.n_cal_samples = 0

    # ------------------------------------------------------------------
    # Fit (CAL only)
    # ------------------------------------------------------------------

    def fit(self, raw_joint_vectors: np.ndarray) -> None:
        """Fit p1/p99 of the CAL-rule features from CAL raw z vectors."""
        if self.frozen:
            raise RuntimeError(
                "FeatureScaler is FROZEN (§13): never refit on PILOT/MAIN"
            )
        X = np.asarray(raw_joint_vectors, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != JOINT_STATE_DIM:
            raise ValueError(
                f"expected (n, {JOINT_STATE_DIM}) raw z vectors, got {X.shape}"
            )
        if X.shape[0] == 0:
            raise ValueError("cannot fit an empty CAL set")
        self._p1.clear()
        self._p99.clear()
        self._degenerate.clear()
        for name in CAL_RULE_FEATURES:
            col = X[:, FEATURE_INDEX[name]]
            p1 = float(np.percentile(col, 1.0))
            p99 = float(np.percentile(col, 99.0))
            self._p1[name] = p1
            self._p99[name] = p99
            if p99 <= p1:
                self._degenerate.add(name)
        self.n_cal_samples = int(X.shape[0])

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _clip01(self, values: np.ndarray) -> np.ndarray:
        return np.clip(values, 0.0, 1.0)

    def _transform_column(self, name: str, col: np.ndarray) -> np.ndarray:
        rule = PHYSICAL_RULES.get(name, "cal_p1p99")
        if rule == "kpi_div_sla":
            return self._clip01(col / SLA_EMBB_MBPS)
        if rule == "kpi_div_deadline":
            return self._clip01(col / SLA_URLLC_MS)
        if rule == "unit01":
            return self._clip01(col)
        if rule == "snr_r09":
            return self._clip01(col / SNR_SCALE_DB)
        if rule == "alloc_div_c":
            return self._clip01(col / float(self.c))
        # cal_p1p99 (counts/queues/offered — §13): clip [p1,p99] -> [0,1]
        p1 = self._p1.get(name)
        p99 = self._p99.get(name)
        if p1 is None or p99 is None:
            raise RuntimeError(
                f"feature {name!r} has no CAL percentile scale; fit on CAL "
                "data first (§13: never use PILOT/MAIN to fit scalers)"
            )
        if name in self._degenerate:  # p1 == p99: identity to 0-safe clip
            return self._clip01(col / p99 if p99 > 0 else np.zeros_like(col))
        return np.clip((col - p1) / (p99 - p1), 0.0, 1.0)

    def transform_joint(self, raw_z: np.ndarray) -> np.ndarray:
        """Normalize one raw z_joint (23,) or a batch (n, 23) -> [0, 1]."""
        Z = np.asarray(raw_z, dtype=np.float64)
        single = Z.ndim == 1
        if single:
            Z = Z[None, :]
        if Z.ndim != 2 or Z.shape[1] != JOINT_STATE_DIM:
            raise ValueError(
                f"expected raw z of dim {JOINT_STATE_DIM}, got {Z.shape}"
            )
        out = np.empty_like(Z)
        for name, idx in FEATURE_INDEX.items():
            out[:, idx] = self._transform_column(name, Z[:, idx])
        return out[0] if single else out

    def transform_feature(self, name: str, values: float | np.ndarray) -> np.ndarray:
        """Normalize ONE named feature by its frozen rule (SQR-HET per-slice
        z blocks reuse the same rules — §55/§27 inheritance)."""
        col = np.asarray(values, dtype=np.float64)
        single = col.ndim == 0
        out = self._transform_column(name, col[None] if single else col)
        return out[0] if single else out

    def transform_x(self, raw_z: np.ndarray, action: tuple[int, int, int]) -> np.ndarray:
        """Full head feature x = [normalized z, a/C] (§5), dim 26."""
        z = self.transform_joint(raw_z)
        a = np.asarray(action, dtype=np.float64) / float(self.c)
        return np.concatenate([z, a])

    # ------------------------------------------------------------------
    # Frozen manifest (§13 feature_scaler_manifest.yaml)
    # ------------------------------------------------------------------

    def to_manifest(self, source: str = "CAL") -> dict:
        payload = {
            "format": SCALER_MANIFEST_FORMAT,
            "source": source,
            "frozen": "true" if self.frozen else "false",
            "c": self.c,
            "n_cal_samples": self.n_cal_samples,
            "rules": {name: PHYSICAL_RULES.get(name, "cal_p1p99")
                      for name in FEATURE_INDEX},
        }
        for name in CAL_RULE_FEATURES:
            payload[f"{name}_p1"] = self._p1.get(name)
            payload[f"{name}_p99"] = self._p99.get(name)
        return payload

    def save_manifest(self, path: str, source: str = "CAL") -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(dump(self.to_manifest(source)))

    @classmethod
    def load_manifest(cls, path: str, freeze: bool = True) -> "FeatureScaler":
        with open(path, "r", encoding="utf-8") as fh:
            data = load(fh.read())
        if data.get("format") != SCALER_MANIFEST_FORMAT:
            raise ValueError(
                f"not a feature scaler manifest (format={data.get('format')!r})"
            )
        scaler = cls(c=int(data["c"]), frozen=bool(freeze))
        for name in CAL_RULE_FEATURES:
            p1 = data.get(f"{name}_p1")
            p99 = data.get(f"{name}_p99")
            if p1 is None or p99 is None:
                continue
            scaler._p1[name] = float(p1)
            scaler._p99[name] = float(p99)
            if p99 <= p1:
                scaler._degenerate.add(name)
        scaler.n_cal_samples = int(data.get("n_cal_samples") or 0)
        return scaler
