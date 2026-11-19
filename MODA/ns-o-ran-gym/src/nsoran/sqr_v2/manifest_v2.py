"""Manifest, seed governance, and v1 integrity verification for sqr-v2."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from typing import Any

#: Seeds strictly reserved for future confirmatory studies (PLANO §1.3).
RESERVED_MAIN_SEEDS = frozenset(range(3001, 3011))
RESERVED_G4B_VALIDATION_SEEDS = frozenset(range(1201, 1210))
RESERVED_OOS_MIN_SEED = 3101

DEFAULT_INFRA_REV = "v2.0-etapa2-ipc_v2-metrics_v2"
DEFAULT_PHY_QOS_PROFILE = "P0_legacy_tdd7_uma_mcs_v1"
DEFAULT_LEARNING_ABLATION = "L0_baseline_no_algo_change"


def find_repo_root(start: str | None = None) -> str:
    """Locate repository root containing ``MODA/docs/sqr/jc_decision/B0_FROZEN_MANIFEST.json``."""
    cur = os.path.abspath(start or os.path.dirname(__file__))
    for _ in range(10):
        cand = os.path.join(cur, "MODA", "docs", "sqr", "jc_decision", "B0_FROZEN_MANIFEST.json")
        if os.path.isfile(cand):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError("Could not locate Jussi_SBRC repository root")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_reserved_confirmatory_seed(seed: int) -> bool:
    s = int(seed)
    return (
        s in RESERVED_MAIN_SEEDS
        or s in RESERVED_G4B_VALIDATION_SEEDS
        or s >= RESERVED_OOS_MIN_SEED
    )


def validate_dev_seed(seed: int, *, allow_reserved: bool = False) -> int:
    """Enforce PLANO §1.3 seed governance before launching any v2 run."""
    s = int(seed)
    if s <= 0:
        raise ValueError(f"seed must be positive, got {seed}")
    if not allow_reserved and is_reserved_confirmatory_seed(s):
        raise ValueError(
            f"seed {s} belongs to reserved confirmatory set "
            "(MAIN 3001..3010, G4B 1201..1209, OOS >=3101); "
            "blocked by PLANO_CORRECAO_INFRAESTRUTURA_V2 §1.3"
        )
    return s


def verify_b0_frozen_manifest(repo_root: str | None = None) -> dict[str, Any]:
    """Verify all 17 code/binary hashes and 5 dev checkpoints in B0_FROZEN_MANIFEST.json."""
    root = find_repo_root(repo_root)
    manifest_path = os.path.join(
        root, "MODA", "docs", "sqr", "jc_decision", "B0_FROZEN_MANIFEST.json"
    )
    with open(manifest_path, "r", encoding="utf-8") as f:
        b0 = json.load(f)

    code_entries = b0.get("git_provenance", {}).get("code_sha256", {})
    code_mismatches: list[dict[str, str]] = []
    for rel_path, expected in code_entries.items():
        abs_path = os.path.join(root, rel_path)
        actual = file_sha256(abs_path) if os.path.isfile(abs_path) else "MISSING"
        if actual != expected:
            code_mismatches.append(
                {"path": rel_path, "expected": expected, "actual": actual}
            )

    ckpt_entries = b0.get("verified_dev_checkpoints", {})
    ckpt_mismatches: list[dict[str, str]] = []
    for key, info in ckpt_entries.items():
        rel_path = info["path"]
        expected = info["checkpoint_file_sha256"]
        abs_path = os.path.join(root, rel_path)
        actual = file_sha256(abs_path) if os.path.isfile(abs_path) else "MISSING"
        if actual != expected:
            ckpt_mismatches.append(
                {"key": key, "path": rel_path, "expected": expected, "actual": actual}
            )

    return {
        "manifest_path": manifest_path,
        "code_total": len(code_entries),
        "code_matched": len(code_entries) - len(code_mismatches),
        "code_mismatches": code_mismatches,
        "checkpoints_total": len(ckpt_entries),
        "checkpoints_matched": len(ckpt_entries) - len(ckpt_mismatches),
        "checkpoint_mismatches": ckpt_mismatches,
        "all_ok": (len(code_mismatches) == 0 and len(ckpt_mismatches) == 0),
    }


def _git_info(repo_root: str) -> dict[str, str]:
    info = {"head": "unknown", "branch": "unknown"}
    try:
        info["head"] = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    return info


def build_v2_manifest(
    *,
    run_id: str,
    controller: str,
    seed: int,
    ipc_mode: str,
    ipc_timeout_ms: int,
    ipc_protocol_version: int = 2,
    infra_rev: str = DEFAULT_INFRA_REV,
    phy_qos_profile: str = DEFAULT_PHY_QOS_PROFILE,
    learning_ablation: str = DEFAULT_LEARNING_ABLATION,
    sim_binary_path: str | None = None,
    extra: dict[str, Any] | None = None,
    repo_root: str | None = None,
) -> dict[str, Any]:
    """Construct the tri-axis ``SQR_V2_MANIFEST.json`` record."""
    root = find_repo_root(repo_root)
    if ipc_mode not in ("strict_rt", "causal_sim"):
        raise ValueError(f"ipc_mode must be 'strict_rt' or 'causal_sim', got {ipc_mode!r}")

    rt_50ms_enforced = bool(ipc_mode == "strict_rt" and int(ipc_timeout_ms) <= 50)
    b0_status = verify_b0_frozen_manifest(root)

    tracked_files = [
        "MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc",
        "MODA/ns-o-ran-gym/src/nsoran/sqr_v2/ipc_v2.py",
        "MODA/ns-o-ran-gym/src/nsoran/sqr_v2/env_v2.py",
        "MODA/ns-o-ran-gym/src/nsoran/sqr_v2/controller_v2.py",
        "MODA/ns-o-ran-gym/src/nsoran/sqr_v2/campaign_analysis_v2.py",
        "MODA/ns-o-ran-gym/src/nsoran/sqr_v2/manifest_v2.py",
        "MODA/ns-o-ran-gym/examples/sqr_v2_run.py",
    ]
    hashes: dict[str, str] = {}
    for rel in tracked_files:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            hashes[rel] = file_sha256(p)
    if sim_binary_path and os.path.isfile(sim_binary_path):
        rel_bin = os.path.relpath(os.path.abspath(sim_binary_path), root)
        hashes[rel_bin] = file_sha256(sim_binary_path)

    payload: dict[str, Any] = {
        "schema_version": "sqr_v2_manifest_1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": str(run_id),
        "controller": str(controller),
        "seed": int(seed),
        "seed_class": "reserved" if is_reserved_confirmatory_seed(seed) else "dev_or_smoke",
        "axes": {
            "infra_rev": str(infra_rev),
            "phy_qos_profile": str(phy_qos_profile),
            "learning_ablation": str(learning_ablation),
        },
        "ipc": {
            "protocol_version": int(ipc_protocol_version),
            "ipc_mode": str(ipc_mode),
            "ipc_timeout_ms": int(ipc_timeout_ms),
            "rt_50ms_deadline_enforced": rt_50ms_enforced,
            "causal_validation_separated_from_rt_proof": True,
        },
        "git": _git_info(root),
        "artifacts_sha256": hashes,
        "b0_frozen_integrity": {
            "code_matched": b0_status["code_matched"],
            "code_total": b0_status["code_total"],
            "checkpoints_matched": b0_status["checkpoints_matched"],
            "checkpoints_total": b0_status["checkpoints_total"],
            "all_ok": b0_status["all_ok"],
        },
    }
    if extra:
        payload["runtime_summary"] = dict(extra)
    return payload


def write_v2_manifest(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
        f.flush()
    os.replace(tmp, path)
