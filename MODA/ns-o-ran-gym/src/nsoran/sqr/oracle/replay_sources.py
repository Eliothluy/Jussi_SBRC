"""Locators + log readers for the Despacho-II replay outputs (Etapa 2.4-II-c).

The replay telemetry lives inside ONE run directory per replay
(``<replay>/<uuid>/results_sqr_network_only/seed=<s>_run=1/``) — the same
file family the oracle dataset was built from (``sqr_kpm_history.csv``,
``sqr_prb_windows.csv``, ``sqr_packet_delays.csv``).  These helpers keep the
path conventions in one place.

``rlf_count_from_logs``: the replay runs are NOT instrumented for RLF events
(no NS_LOG traces in stdout/stderr); the count is the number of radio-link
failure markers found in the replay's captured logs.  In the frozen 300 m
scenario this is expected to be 0 everywhere; the column exists so §5 of the
protocol is exportable and the limitation (no direct RLF counter) is
registered in the final report §26.
"""

from __future__ import annotations

import glob
import os
import re

#: Files the adjudication reads from each replay run directory.
_TELEMETRY_FILES = ("sqr_kpm_history.csv", "sqr_prb_windows.csv",
                    "sqr_packet_delays.csv")

#: Case-insensitive markers of radio link failure in the captured logs.
_RLF_RE = re.compile(r"radio\s*link\s*failure|RadioLinkFailure|\bRLF\b",
                     re.IGNORECASE)


def replay_paths(replay_dir: str) -> dict:
    """Locate the telemetry CSVs of one replay directory.

    Returns {"kpm", "prb", "delays", "telemetry_subdir"}; raises
    ``FileNotFoundError`` naming the missing file (§4 'não aceitar run
    parcial').
    """
    hits = sorted(glob.glob(os.path.join(
        replay_dir, "**", "results_sqr_network_only", "seed=*",
        "sqr_kpm_history.csv"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"sqr_kpm_history.csv not found under {replay_dir}")
    sub = os.path.dirname(hits[0])
    out = {"kpm": hits[0], "telemetry_subdir": os.path.relpath(sub, replay_dir)}
    for key, fname in (("prb", "sqr_prb_windows.csv"),
                       ("delays", "sqr_packet_delays.csv")):
        p = os.path.join(sub, fname)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{fname} not found under {replay_dir}")
        out[key] = p
    return out


def rlf_count_from_logs(replay_dir: str) -> int:
    """Count radio-link-failure markers in the replay's stdout/stderr.

    Non-destructive: missing logs count as 0 matches (the logs are captured
    by the runner; both files exist in every completed replay).
    """
    n = 0
    for name in ("stdout", "stderr", "run.log"):
        p = os.path.join(replay_dir, name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", errors="replace") as fh:
                for line in fh:
                    if _RLF_RE.search(line):
                        n += 1
        except OSError:  # pragma: no cover — defensive
            continue
    return n
