"""B0 freeze/resume contract, version/hash manifests, and causal trace instrumentation.

Item 6 preparation for Phase D3-v2 / A5 (without changing B0 policy or touching
historical D1-v9 / A0..A4 artifacts):
  1. Explicit version identifiers and artifact SHA-256 hashes (scaler & SLA
     manifests, canonical config descriptor, controller payload hash).
  2. Strict ``freeze`` / ``resume`` compatibility verification across semantic
     and hyperparameter flags (preserving backward compatibility with legacy
     checkpoints lacking extended metadata).
  3. Opt-in per-window causal trace instrumentation (``CausalTraceWriter``),
     guaranteed side-effect-free (no RNG consumption, no controller mutation)
     so instrumented and uninstrumented runs remain bit-identical.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from typing import Any, Mapping, Sequence

import numpy as np

from nsoran.sqr.checkpoint import CHECKPOINT_FORMAT, CHECKPOINT_VERSION
from nsoran.sqr.sqr_controller import (
    MARGIN_BOUNDS,
    MARGIN_INIT,
    STATE_DICT_VERSION,
    TAU_INIT,
    TAU_STEP,
    VIOLATION_WINDOW,
    SqrSliceController,
)
from nsoran.sqr_jc import SQR_JC_VERSION

FREEZE_CONTRACT_VERSION = "1.0"
CAUSAL_TRACE_VERSION = "1.0"

_DEFAULT_SLA_MANIFEST = os.path.join(os.path.dirname(__file__), "sla_manifest.yaml")


def sha256_file(path: str | None) -> str | None:
    """Return hex SHA-256 of an existing file, or None if unavailable."""
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonicalize(obj: Any) -> Any:
    """Convert numpy/Python containers into a deterministic JSON-serializable tree."""
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {
            "__ndarray__": True,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    return obj


def canonical_json_sha256(obj: Any) -> str:
    """Deterministic SHA-256 of a canonicalized nested structure."""
    encoded = json.dumps(_canonicalize(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Deterministic SHA-256 over the checkpoint's slice states (excludes saved_at)."""
    core = {
        "format": payload.get("format"),
        "version": payload.get("version"),
        "slices": payload.get("slices", {}),
    }
    return canonical_json_sha256(core)


def read_sla_manifest_version(path: str = _DEFAULT_SLA_MANIFEST) -> str:
    """Read sla_manifest_version from the YAML header without external deps."""
    if not os.path.isfile(path):
        return "unknown"
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("sla_manifest_version:"):
                return s.split(":", 1)[1].strip().strip("'\"")
    return "unknown"


def build_version_manifest(
    *,
    scaler_manifest_path: str | None = None,
    sla_manifest_path: str = _DEFAULT_SLA_MANIFEST,
) -> dict[str, Any]:
    """Return explicit code, checkpoint, and manifest versions + SHA-256 digests."""
    return {
        "freeze_contract_version": FREEZE_CONTRACT_VERSION,
        "causal_trace_version": CAUSAL_TRACE_VERSION,
        "sqr_jc_version": SQR_JC_VERSION,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_version": int(CHECKPOINT_VERSION),
        "controller_state_version": int(STATE_DICT_VERSION),
        "sla_manifest_version": read_sla_manifest_version(sla_manifest_path),
        "sla_manifest_sha256": sha256_file(sla_manifest_path),
        "scaler_manifest_path": os.path.abspath(scaler_manifest_path) if scaler_manifest_path else None,
        "scaler_manifest_sha256": sha256_file(scaler_manifest_path),
    }


def build_b0_config_descriptor(
    *,
    controller: str,
    gamma: float,
    c_prbs: int,
    virtual_alpha: float,
    empty_slice_guard: bool,
    fix_scaler_index: bool,
    mean_field: bool,
    margin_init: float = MARGIN_INIT,
    margin_bounds: Sequence[float] = MARGIN_BOUNDS,
    tau_init: float = TAU_INIT,
    tau_step: float = TAU_STEP,
    violation_window: int = VIOLATION_WINDOW,
    urllc_delayed_update: bool = False,
    warmup_s: float = 100.0,
    scaler_manifest_path: str | None = None,
    sla_manifest_path: str = _DEFAULT_SLA_MANIFEST,
) -> tuple[dict[str, Any], str]:
    """Assemble the canonical B0 behavioral config dict and its SHA-256 digest."""
    versions = build_version_manifest(
        scaler_manifest_path=scaler_manifest_path,
        sla_manifest_path=sla_manifest_path,
    )
    descriptor = {
        "controller": str(controller),
        "gamma": float(gamma),
        "c_prbs": int(c_prbs),
        "virtual_alpha": float(virtual_alpha),
        "empty_slice_guard": bool(empty_slice_guard),
        "fix_scaler_index": bool(fix_scaler_index),
        "mean_field": bool(mean_field),
        "margin_init": float(margin_init),
        "margin_bounds": [float(margin_bounds[0]), float(margin_bounds[1])],
        "tau_init": float(tau_init),
        "tau_step": float(tau_step),
        "violation_window": int(violation_window),
        "urllc_delayed_update": bool(urllc_delayed_update),
        "warmup_s": float(warmup_s),
        "versions": {
            "freeze_contract_version": versions["freeze_contract_version"],
            "sqr_jc_version": versions["sqr_jc_version"],
            "checkpoint_format": versions["checkpoint_format"],
            "checkpoint_version": versions["checkpoint_version"],
            "controller_state_version": versions["controller_state_version"],
            "sla_manifest_version": versions["sla_manifest_version"],
        },
        "artifacts_sha256": {
            "sla_manifest_sha256": versions["sla_manifest_sha256"],
            "scaler_manifest_sha256": versions["scaler_manifest_sha256"],
        },
    }
    return descriptor, canonical_json_sha256(descriptor)


def verify_resume_contract(
    payload: Mapping[str, Any],
    requested_meta: Mapping[str, Any],
    controllers: Mapping[int, SqrSliceController],
) -> dict[str, Any]:
    """Validate checkpoint payload + metadata against requested runtime config.

    Checks format/version, bidirectional slice topology, optional payload hash
    integrity, mandatory ``mean_field``/``fix_scaler_index`` guards, and any
    shared config flags present in checkpoint metadata.
    """
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"invalid checkpoint format: {payload.get('format')!r}")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {payload.get('version')!r} "
            f"(expected {CHECKPOINT_VERSION})"
        )

    slices = payload.get("slices", {})
    missing = sorted(set(slices) - set(controllers))
    if missing:
        raise ValueError(f"checkpoint has slices with no controller: {missing}")
    extra = sorted(set(controllers) - set(slices))
    if extra:
        raise ValueError(f"controllers with no slice in checkpoint: {extra}")

    ckpt_meta = dict(payload.get("meta") or {})
    if "payload_sha256" in ckpt_meta:
        actual_hash = checkpoint_payload_sha256(payload)
        if actual_hash != ckpt_meta["payload_sha256"]:
            raise ValueError(
                "checkpoint payload_sha256 mismatch: "
                f"expected {ckpt_meta['payload_sha256']}, got {actual_hash}"
            )

    # Mandatory state semantics & scaler normalization guards (even on legacy meta)
    if bool(ckpt_meta.get("mean_field", False)) != bool(requested_meta.get("mean_field", False)):
        raise ValueError(
            "--resume refused: checkpoint state semantics mismatch "
            f"(checkpoint mean_field={bool(ckpt_meta.get('mean_field', False))}, "
            f"requested mean_field={bool(requested_meta.get('mean_field', False))})"
        )
    if bool(ckpt_meta.get("fix_scaler_index", False)) != bool(requested_meta.get("fix_scaler_index", False)):
        raise ValueError(
            "--resume refused: checkpoint normalization mismatch "
            f"(checkpoint fix_scaler_index={bool(ckpt_meta.get('fix_scaler_index', False))}, "
            f"requested fix_scaler_index={bool(requested_meta.get('fix_scaler_index', False))})"
        )

    # Extended flag & manifest hash checks whenever recorded in ckpt_meta
    bool_keys = ("empty_slice_guard", "urllc_delayed_update")
    float_keys = ("gamma", "virtual_alpha", "margin_init")
    str_keys = ("controller", "config_sha256", "scaler_manifest_sha256", "sla_manifest_sha256")
    for k in bool_keys:
        if k in ckpt_meta and k in requested_meta and bool(ckpt_meta[k]) != bool(requested_meta[k]):
            raise ValueError(f"--resume refused: {k} mismatch ({ckpt_meta[k]} vs {requested_meta[k]})")
    for k in float_keys:
        if k in ckpt_meta and k in requested_meta and float(ckpt_meta[k]) != float(requested_meta[k]):
            raise ValueError(f"--resume refused: {k} mismatch ({ckpt_meta[k]} vs {requested_meta[k]})")
    for k in str_keys:
        if ckpt_meta.get(k) is not None and requested_meta.get(k) is not None:
            if str(ckpt_meta[k]) != str(requested_meta[k]):
                raise ValueError(f"--resume refused: {k} mismatch ({ckpt_meta[k]} vs {requested_meta[k]})")
    if "margin_bounds" in ckpt_meta and "margin_bounds" in requested_meta:
        cb = [float(x) for x in ckpt_meta["margin_bounds"]]
        rb = [float(x) for x in requested_meta["margin_bounds"]]
        if cb != rb:
            raise ValueError(f"--resume refused: margin_bounds mismatch ({cb} vs {rb})")

    return ckpt_meta


def verify_causal_replay_no_future_checkpoint(
    payload: Mapping[str, Any],
    *,
    target_window_index: int,
    target_sim_time_ms: int | None = None,
    target_step_index: int | None = None,
    strict_pre_decision: bool = True,
) -> dict[str, Any]:
    """Refuse reusing a future (or missing-horizon) checkpoint on a past window W.

    Important methodological distinction:
    - Single-window causal/counterfactual replay (``D3-v2`` / ``A5``) evaluates
      an alternative joint action at window ``W`` starting from an instantaneous
      pre-decision snapshot of the controller/state at ``W`` (strictly prior to
      outcomes/virtual-updates at or after ``W``). Reusing a checkpoint saved
      after ``W`` (such as an end-of-episode checkpoint) would leak future data.
    - Closed-loop out-of-sample evaluation (``--resume <ckpt> --freeze``) runs
      a frozen policy from ``step=0`` on an independent realization; trajectories
      are allowed to diverge physically once a different decision changes queueing
      and traffic evolution.
    """
    ckpt_meta = dict(payload.get("meta") or {})
    horizon = ckpt_meta.get("checkpoint_horizon")
    if not isinstance(horizon, Mapping) or "max_window_index" not in horizon:
        raise ValueError(
            "causal replay refused: checkpoint lacks 'checkpoint_horizon' "
            "provenance metadata (cannot prove non-anticipation for window "
            f"{int(target_window_index)})"
        )
    max_win = int(horizon["max_window_index"])
    tgt_win = int(target_window_index)
    violated_win = (max_win >= tgt_win) if strict_pre_decision else (max_win > tgt_win)
    if violated_win:
        raise ValueError(
            "causal replay refused: checkpoint horizon includes window "
            f"{max_win} >= target_window_index={tgt_win} (future/same-window "
            "information leakage prohibited; for out-of-sample closed-loop "
            "evaluation from step=0 use --resume --freeze instead)"
        )
    if target_sim_time_ms is not None and "max_sim_time_ms" in horizon:
        max_ts = int(horizon["max_sim_time_ms"])
        tgt_ts = int(target_sim_time_ms)
        if (max_ts >= tgt_ts) if strict_pre_decision else (max_ts > tgt_ts):
            raise ValueError(
                "causal replay refused: checkpoint sim_time_ms "
                f"{max_ts} violates non-anticipation for target_sim_time_ms={tgt_ts}"
            )
    if target_step_index is not None and "max_step_index" in horizon:
        max_step = int(horizon["max_step_index"])
        tgt_step = int(target_step_index)
        if (max_step >= tgt_step) if strict_pre_decision else (max_step > tgt_step):
            raise ValueError(
                "causal replay refused: checkpoint step_index "
                f"{max_step} violates non-anticipation for target_step_index={tgt_step}"
            )
    return dict(horizon)


def verify_causal_snapshot_matches_controllers(
    trace_record: Mapping[str, Any],
    controllers: Mapping[int, SqrSliceController],
    *,
    capacity_c: int | None = None,
) -> None:
    """Assert that a ``CausalTraceWriter`` snapshot matches live controllers."""
    if capacity_c is not None and int(trace_record["capacity_C"]) != int(capacity_c):
        raise ValueError(
            f"causal snapshot capacity mismatch: trace={trace_record['capacity_C']} vs {capacity_c}"
        )
    entries = {int(e["slice_id"]): e for e in trace_record["per_slice_model"]}
    if sorted(entries) != sorted(controllers):
        raise ValueError(
            f"causal snapshot slice keys mismatch: {sorted(entries)} vs {sorted(controllers)}"
        )
    for sid, ctl in sorted(controllers.items()):
        expected_hash = entries[sid]["model_state_sha256"]
        actual_hash = canonical_json_sha256(ctl.state_dict())
        if actual_hash != expected_hash:
            raise ValueError(
                f"causal snapshot state hash mismatch on slice {sid}: {actual_hash} != {expected_hash}"
            )


def verify_freeze_no_learning(
    before_states: Mapping[int, Mapping[str, Any]],
    after_states: Mapping[int, Mapping[str, Any]],
) -> None:
    """Assert that a --freeze run never mutated regressor, memory, tau, or M."""
    if sorted(before_states) != sorted(after_states):
        raise AssertionError("controller slice keys changed under --freeze")
    for sid in sorted(before_states):
        b, a = before_states[sid], after_states[sid]
        for scalar_key in ("margin_m", "window_count", "clean_window_streak"):
            if b[scalar_key] != a[scalar_key]:
                raise AssertionError(f"slice {sid} {scalar_key} mutated under --freeze")
        if list(b["violations"]) != list(a["violations"]):
            raise AssertionError(f"slice {sid} violations mutated under --freeze")
        for arr_key in ("experience_z", "experience_a", "experience_y", "experience_ok"):
            if not np.array_equal(b[arr_key], a[arr_key]):
                raise AssertionError(f"slice {sid} {arr_key} mutated under --freeze")
        br, ar = b["regressor"], a["regressor"]
        if br["tau"] != ar["tau"]:
            raise AssertionError(f"slice {sid} tau mutated under --freeze")
        for r_key in ("support_points", "alphas"):
            if not np.array_equal(br[r_key], ar[r_key]):
                raise AssertionError(f"slice {sid} regressor.{r_key} mutated under --freeze")


class CausalTraceWriter:
    """Non-invasive per-window causal trace logger for D3-v2 / A5 replay.

    Records strictly pre-decision inputs and instantaneous model predictions
    f_tau(z_i, a) / scores(z_i, a) / sigma(z_i, a) at decision time, without
    calling RNG or mutating any controller/runtime state.
    """

    def __init__(self, path: str, metadata: Mapping[str, Any] | None = None) -> None:
        self.path = str(path)
        directory = os.path.dirname(os.path.abspath(self.path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._is_gz = self.path.endswith(".gz")
        self._fh = (
            gzip.open(self.path, "wt", encoding="utf-8")
            if self._is_gz
            else open(self.path, "w", encoding="utf-8")
        )
        self.records_written = 0
        if metadata is not None:
            header = {
                "record_type": "causal_trace_header",
                "causal_trace_version": CAUSAL_TRACE_VERSION,
                "metadata": _canonicalize(dict(metadata)),
            }
            self._fh.write(json.dumps(header, separators=(",", ":")) + "\n")

    def record_decision(
        self,
        *,
        step_index: int,
        sim_time_ms: int,
        policy: str,
        state: Any,
        zs: Sequence[np.ndarray],
        controllers: Sequence[SqrSliceController] | Mapping[int, SqrSliceController],
        capacity_c: int,
        a_raw_per_slice: Sequence[int],
        a_prop_per_slice: Sequence[int],
        a_exec_per_slice: Sequence[int],
        cut_per_slice: Sequence[bool],
        modes_per_slice: Sequence[str],
        precomputed_scores: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Capture one instantaneous causal decision snapshot (pure/read-only)."""
        ctl_list = (
            [controllers[sid] for sid in range(len(zs))]
            if isinstance(controllers, Mapping)
            else list(controllers)
        )
        per_slice = []
        for sid, (z, ctl) in enumerate(zip(zs, ctl_list)):
            sc = (
                precomputed_scores[sid]
                if (precomputed_scores is not None and sid < len(precomputed_scores))
                else ctl.score_candidates(z)  # pure: zero RNG, zero mutation
            )
            knn_a, knn_mode = ctl.k_nn_action(z)  # pure: zero RNG, zero mutation
            sd = ctl.state_dict()
            per_slice.append({
                "slice_id": int(sid),
                "z": [float(v) for v in z],
                "tau": float(ctl.tau),
                "margin_m": float(ctl.margin),
                "gamma": float(ctl.gamma),
                "n_support": int(ctl.regressor.n_support),
                "n_experiences": int(ctl.n_experiences),
                "sum_alphas": float(np.sum(ctl.regressor.alphas)),
                "model_state_sha256": canonical_json_sha256(sd),
                "knn_action_raw": int(knn_a),
                "knn_mode": str(knn_mode),
                "f_candidates": [float(v) for v in sc["f"][: capacity_c + 1]],
                "scores_candidates": [float(v) for v in sc["scores"][: capacity_c + 1]],
                "sigma_candidates": [float(v) for v in sc["sigma"][: capacity_c + 1]],
            })
        rec = {
            "record_type": "causal_window",
            "causal_trace_version": CAUSAL_TRACE_VERSION,
            "step_index": int(step_index),
            "window_index": int(state.window_index),
            "governed_window_index": int(state.window_index) + 1,
            "causal_horizon_max_window_index": int(state.window_index),
            "causal_horizon_max_sim_time_ms": int(sim_time_ms),
            "sim_time_ms": int(sim_time_ms),
            "policy": str(policy),
            "capacity_C": int(capacity_c),
            "state": {
                "n_e": int(state.n_e),
                "n_u": int(state.n_u),
                "n_m": int(state.n_m),
                "queue_e": float(state.queue_e),
                "queue_u": float(state.queue_u),
                "queue_m": float(state.queue_m),
                "kpi_e_raw": None if state.kpi_e_raw is None else float(state.kpi_e_raw),
                "kpi_u_raw": None if state.kpi_u_raw is None else float(state.kpi_u_raw),
                "kpi_m_raw": None if state.kpi_m_raw is None else float(state.kpi_m_raw),
                "snr_e": float(state.snr_e),
                "snr_u": float(state.snr_u),
                "snr_m": float(state.snr_m),
                "prev_a_e": int(state.prev_a_e),
                "prev_a_u": int(state.prev_a_u),
                "prev_a_m": int(state.prev_a_m),
                "viol_e": int(state.viol_e),
                "viol_u": int(state.viol_u),
                "viol_m": int(state.viol_m),
                "offered_e": float(getattr(state, "offered_e", 0.0)),
                "offered_u": float(getattr(state, "offered_u", 0.0)),
                "offered_m": float(getattr(state, "offered_m", 0.0)),
                "extras": _canonicalize(dict(getattr(state, "extras", None) or {})),
            },
            "a_raw_per_slice": [int(v) for v in a_raw_per_slice],
            "a_prop_per_slice": [int(v) for v in a_prop_per_slice],
            "a_exec_per_slice": [int(v) for v in a_exec_per_slice],
            "cut_per_slice": [bool(v) for v in cut_per_slice],
            "modes_per_slice": [str(m) for m in modes_per_slice],
            "per_slice_model": per_slice,
        }
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.records_written += 1

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
