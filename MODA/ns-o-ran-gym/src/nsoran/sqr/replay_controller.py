"""Deterministic single-window replay controller (Etapa 2.4, Despacho II-a).

Executable validation of the offline oracle (protocol
``docs/artigo/rodada_2_etapa_2_4_oracle_viabilidade.md`` §15-§16):
re-run a frozen Etapa-2.3 scenario with the SAME seed — so the state
trajectory up to the target window t reproduces the original run (MESMA
seed ⇒ MESMO estado até t) — and OVERRIDE the action of exactly ONE
decision window t with a candidate allocation chosen by the replay plan
(``examples/sqr_oracle_select_replays.py``).  The KPIs of window t, read
from the same telemetry the oracle dataset was built from
(``sqr_kpm_history.csv`` / ``sqr_prb_windows.csv``), adjudicate the oracle
prediction for that (state, action) pair (§16 TP/TN/FP/FN).

Window/act mapping (verified against the code; contract §2/§4):

- the C++ scenario schedules the first ``KpmAndControlCallback`` at
  t = periodMs = 100 ms (sqr-sim.cc ~1810/~2598), so KPM stamps are
  100, 200, ... ms; ``env.reset()`` consumes the FIRST one (win 0 =
  [0, 100) ms, ``ns_env.reset`` → ``_wait_data_availability``) and does
  NOT release the control semaphore;
- act() call k (1-based) therefore decides from win k-1 KPIs; its action
  is written to the action CSV and consumed by the C++ callback at
  t = 100k ms — "governa a janela SEGUINTE" (sqr-sim.cc
  ``ApplyQuotaAction``) — i.e. it is the quota IN FORCE during oracle
  window w = k (win = t//100 - 1; window w covers [100w, 100(w+1)) ms,
  the ``oracle/dataset.py`` convention);
- hence ``replay_at_win = t`` puts the override in force during oracle
  window t: windows < t run the base action, window t runs the override,
  windows > t return to the base (the short run ends soon after).

Bases (reproducing the two frozen 2.3 arms, Despacho I §1):

- ``"equal"``: the network-only arm executed a CONSTANT (18, 17, 18)
  quota at C = 53 — the scenario's INITIAL partition.  Mirrored here
  exactly: the sqr-sim initial weights (0.333333, 0.333333, 0.333334)
  (``sqr-sim.cc`` ``weightsStr``) through the SAME deterministic
  conversion of ``SqrMacScheduler::ComputeRbgQuotas`` (raw_i = frac_i·C;
  target = round(Σraw) capped at C; floor + largest fractional remainder,
  ties by LOWEST slice index, remainder consumed; NO renormalization).
  Pure 1/3 weights would yield (18, 18, 17) and would NOT reproduce the
  original run [ASSUMIDO-REPLAY R47].
- ``"prop"``: delegates the base action to the UNCHANGED
  :class:`~nsoran.sqr.static_controllers.PropDemandController` — the
  original IPC prop arm's controller — fed by the same parsed KPM window
  (``env.builder.last_window``).  Same class + same wiring + same seed ⇒
  the same deterministic proposal sequence as the original run.

IPC contract untouched: this is a plain per-slice CONSUMER like the
static baselines (same :class:`StaticDecision` fields, ``LEARNING =
False``, ``set_n_prbs`` sync for the dynamic C (R01), checkpoint
round-trip); the env writes the action CSV in the current format
(dedicatedPRB = 100·a/C) and the C++ ``ComputeRbgQuotas`` converts it
back to the exact integer quota (verified round-trip for integer a ≤ C).
The override tuple is validated at construction: integer RBGs, ≥ 0,
Σ ≤ C — the projection Π (contract §3.5) stays as the safety net.
"""

from __future__ import annotations

import math

import numpy as np

from nsoran.sqr.static_controllers import (
    MODE_PROP,
    PROP_CAPACITY_PER_RBG_MBPS,
    PROP_FLOOR_FRACS,
    PROP_SLA_TARGETS,
    PropDemandController,
    StaticDecision,
)

#: Checkpoint payload version of the replay controller (own lineage).
REPLAY_STATE_VERSION = 1

#: Decision "mode" logged in step_metrics.csv per window kind.
MODE_REPLAY_EQUAL = "replay_equal"        # base equal window
MODE_REPLAY_PROP = "replay_prop"          # base prop window (== MODE_PROP
                                          # semantics, replay-marked)
MODE_REPLAY_OVERRIDE = "replay_override"  # the single override window t

#: Valid base actions of ``--replay_base``.  ``"seq"`` (D2) replays the
#: RECORDED per-window action sequence of an arbitrary B0 run — the
#: continuation policy is the B0 run's own decisions (PROTOCOL_FROZEN §4.2).
REPLAY_BASES = ("equal", "prop", "seq")

#: Decision "mode" of a seq-base window (D2 replay over B0 trajectories).
MODE_REPLAY_SEQ = "replay_seq"

#: Initial equal-split weights of sqr-sim (``sqr-sim.cc`` ``weightsStr``),
#: order (eMBB, URLLC, mMTC).  Sum to exactly 1.0; the LAST slice carries
#: the +1e-6 residue, which is what makes the scenario's initial partition
#: (18, 17, 18) at C = 53 — the measured network-only equal arm quota
#: (Despacho I §1) [ASSUMIDO-REPLAY R47].
REPLAY_EQUAL_WEIGHTS: tuple[float, ...] = (0.333333, 0.333333, 0.333334)


def initial_equal_split(n_prbs: int, num_slices: int = 3) -> np.ndarray:
    """Python mirror of the scenario's initial partition.

    Applies the EXACT rule of ``SqrMacScheduler::ComputeRbgQuotas``
    (sqr-mac-scheduler.cc ~57-120) to the scenario's initial weights:
    raw_i = w_i·C; target = round(Σraw) capped at C; quota_i = floor(raw_i);
    the (target − Σ floor) units go one each to the LARGEST fractional
    remainders, ties by LOWEST slice index, the chosen remainder consumed
    (frac := −1).  Deterministic; Σ ≤ C guaranteed.

    C = 53 → raw (17.666649, 17.666649, 17.666702), floors (17, 17, 17),
    target 53, extra 2 → slice 2 (largest frac), then slice 0 (tie with
    slice 1, lowest index) → **(18, 17, 18)** — the executed network-only
    equal-arm quota (Despacho I §1).  For reference, C = 52 → (17, 17, 18).
    """
    c = int(n_prbs)
    if c <= 0:
        raise ValueError(f"n_prbs must be positive, got {n_prbs}")
    if int(num_slices) != len(REPLAY_EQUAL_WEIGHTS):
        raise ValueError(
            "replay equal base mirrors the 3-slice sqr-sim initial weights "
            f"{REPLAY_EQUAL_WEIGHTS}; num_slices must be 3, got {num_slices}"
        )
    raw = [w * float(c) for w in REPLAY_EQUAL_WEIGHTS]
    # std::llround = round-half-away-from-zero; raw_sum >= 0 here.
    target = min(int(math.floor(math.fsum(raw) + 0.5)), c)
    quotas = [int(math.floor(r)) for r in raw]
    fracs = [r - q for r, q in zip(raw, quotas)]
    extra = target - sum(quotas)
    while extra > 0:
        best = -1
        for i, f in enumerate(fracs):
            if f >= 0.0 and (best < 0 or f > fracs[best]):
                best = i
        if best < 0:  # pragma: no cover — defensive, unreachable (Σ=1.0)
            break
        quotas[best] += 1
        fracs[best] = -1.0  # consume the remainder (C++ semantics)
        extra -= 1
    return np.asarray(quotas, dtype=np.int64)


def parse_replay_action(
    text: str, num_slices: int = 3, n_prbs: int = 53
) -> tuple[int, ...]:
    """Parse ``--replay_action 'aE,aU,aM'`` into a validated int tuple.

    Integer RBGs, one per slice, each ≥ 0 and Σ ≤ C (the ask's contract;
    the env projection stays as the safety net).  Raises ``ValueError``
    with an explicit message on any violation.
    """
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != int(num_slices):
        raise ValueError(
            f"--replay_action must have exactly {num_slices} comma-separated "
            f"integer RBGs (aE,aU,aM), got {text!r}"
        )
    try:
        values = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(
            f"--replay_action values must be integers, got {text!r}"
        ) from exc
    if any(v < 0 for v in values):
        raise ValueError(f"--replay_action values must be >= 0, got {values}")
    if sum(values) > int(n_prbs):
        raise ValueError(
            f"--replay_action {values} sums to {sum(values)} > C={n_prbs} "
            "(contract: Σa ≤ C)"
        )
    return values


class ReplayController:
    """One slice's share of the replay decision (consumer of the contract).

    Parameters
    ----------
    slice_id:
        Index of this slice (0 = eMBB, 1 = URLLC, 2 = mMTC).
    n_prbs:
        C, total PRBs of the cell (synced to the effective telemetry C by
        :meth:`set_n_prbs`, register R01 as amended).
    num_slices:
        L, number of slices (the equal base mirrors the 3-slice weights).
    base:
        ``"equal"`` (constant initial-partition split, network-only arm
        reproduction) or ``"prop"`` (delegates to the unchanged
        ``PropDemandController``, IPC prop arm reproduction).
    replay_at_win:
        t, the oracle window index whose QUOTA must be the override
        (window w covers [100w, 100(w+1)) ms; the act() call k governs
        window k — see the module docstring).  Must be ≥ 1: window 0 runs
        before any controller decision exists.
    replay_action:
        The override allocation (a_e, a_u, a_m) in integer RBGs, Σ ≤ C.
    window_provider:
        Callable returning the latest parsed :class:`SqrKpmWindow` (wired
        by ``examples/sqr_run.py`` to ``env.builder.last_window``); used
        by the prop base exactly as in the original prop arm.
    """

    LEARNING = False  # no learning state; the run loop logs learned = 0

    def __init__(
        self,
        slice_id: int,
        n_prbs: int,
        num_slices: int = 3,
        base: str = "equal",
        replay_at_win: int = 1,
        replay_action: tuple[int, ...] = (0, 0, 0),
        window_provider=None,
        seq_actions=None,
    ):
        if int(slice_id) not in range(int(num_slices)):
            raise ValueError(
                f"slice_id {slice_id} out of range for num_slices={num_slices}"
            )
        if int(n_prbs) <= 0:
            raise ValueError(f"n_prbs must be positive, got {n_prbs}")
        if base not in REPLAY_BASES:
            raise ValueError(
                f"base must be one of {REPLAY_BASES}, got {base!r}"
            )
        seq_actions = tuple(tuple(int(v) for v in a) for a in (seq_actions or ()))
        if base == "seq":
            if not seq_actions:
                raise ValueError(
                    "base='seq' requires the recorded B0 action sequence "
                    "(seq_actions), one tuple per decision window"
                )
            if len(seq_actions) < int(replay_at_win):
                raise ValueError(
                    f"seq_actions covers {len(seq_actions)} windows; "
                    f"replay_at_win={replay_at_win} is beyond it"
                )
        if int(replay_at_win) < 1:
            raise ValueError(
                "replay_at_win must be >= 1 (window 0 runs before any "
                f"controller decision), got {replay_at_win}"
            )
        action = tuple(int(v) for v in replay_action)
        if len(action) != int(num_slices):
            raise ValueError(
                f"replay_action must have {num_slices} components, got {action}"
            )
        if any(v < 0 for v in action):
            raise ValueError(f"replay_action values must be >= 0, got {action}")
        if sum(action) > int(n_prbs):
            raise ValueError(
                f"replay_action {action} sums to {sum(action)} > C={n_prbs} "
                "(contract: Σa ≤ C)"
            )
        self.slice_id = int(slice_id)
        self.n_prbs = int(n_prbs)
        self.num_slices = int(num_slices)
        self.base = str(base)
        self.replay_at_win = int(replay_at_win)
        self.replay_action = action
        self._seq_actions = seq_actions
        self._window_provider = window_provider
        self._windows_seen = 0  # act() calls so far == windows governed
        # The prop base delegates to the UNCHANGED original controller so
        # the pre/post-t sequence reproduces the IPC prop arm exactly.
        self._prop = (
            PropDemandController(
                slice_id=self.slice_id,
                n_prbs=self.n_prbs,
                num_slices=self.num_slices,
                window_provider=window_provider,
            )
            if base == "prop"
            else None
        )

    # ------------------------------------------------------------------
    # Base + override decision
    # ------------------------------------------------------------------

    def _base_split(self) -> np.ndarray:
        """Base allocation (L,) int64 for a non-override window."""
        if self.base == "seq":
            # act() call k (1-based) governs window k; the recorded B0
            # action of DECISION k (seq[k-1]: decisions.jsonl decision_index
            # is 1-based) reproduces the original trajectory.
            # D2-v2 incident (13/11): the base seq has len(decisions.jsonl)
            # entries — fewer than the replay's decision windows when the
            # base run logged HOLDS (cell-empty windows produce no
            # decision line, lambda5/6: 2298 decisions for 3000 clock
            # windows).  Windows PAST the end of the base are not
            # adjudicated (the target window t and the bit-exact prefix
            # are always inside the range), so clamp to the LAST known
            # action (hold-like) instead of crashing with IndexError.
            _k = min(self._windows_seen - 1, len(self._seq_actions) - 1)
            return np.asarray(
                self._seq_actions[_k], dtype=np.int64)
        if self._prop is not None:
            window = (
                self._window_provider() if self._window_provider else None
            )
            return self._prop.prop_split(window)
        return initial_equal_split(self.n_prbs, self.num_slices)

    def act(self, z: np.ndarray) -> StaticDecision:  # noqa: ARG002 — z unused
        """Decision for this window: base, or the override at window t.

        ``z`` is accepted and ignored (same as the static baselines): the
        base equal is constant and the base prop reads the parsed KPM
        window through the provider, not the normalized features.  The
        base is CONSULTED on every window (including the override one) so
        the provider/state sequence of the base stays in lockstep with the
        original controller's — the override discards the base proposal,
        never skips the window itself.
        """
        self._windows_seen += 1
        base = self._base_split()
        if self._windows_seen == self.replay_at_win:
            action = int(self.replay_action[self.slice_id])
            mode = MODE_REPLAY_OVERRIDE
        else:
            action = int(base[self.slice_id])
            mode = (
                MODE_REPLAY_SEQ if self.base == "seq" else
                (MODE_REPLAY_PROP if self._prop is not None
                 else MODE_REPLAY_EQUAL)
            )
        return StaticDecision(
            action=action,
            raw_action=action,
            mode=mode,
            sigma=float("nan"),
            margin_applied=0,
            tau=float("nan"),
            margin_m=float("nan"),
        )

    # ------------------------------------------------------------------
    # Runtime wiring (dynamic C — register R01 as amended)
    # ------------------------------------------------------------------

    def set_n_prbs(self, n_prbs: int) -> None:
        """Follow the dynamic C: the equal split recomputes and the prop
        delegate rescales on the next :meth:`act`."""
        if int(n_prbs) <= 0:
            raise ValueError(f"n_prbs must be positive, got {n_prbs}")
        self.n_prbs = int(n_prbs)
        if self._prop is not None:
            self._prop.set_n_prbs(int(n_prbs))

    def attach_window_provider(self, provider) -> None:
        """Wire the KPM-window source (prop base); ``None`` detaches."""
        self._window_provider = provider
        if self._prop is not None:
            self._prop.attach_window_provider(provider)

    def observe(self, z: np.ndarray, action: int, y: float) -> None:
        """No learning state — documented no-op (static-style consumer)."""
        return None

    # ------------------------------------------------------------------
    # Checkpoint round-trip (same file-format conventions as "equal")
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "kind": "replay",
            "state_version": REPLAY_STATE_VERSION,
            "slice_id": self.slice_id,
            "n_prbs": self.n_prbs,
            "num_slices": self.num_slices,
            "base": self.base,
            "replay_at_win": self.replay_at_win,
            "replay_action": list(self.replay_action),
            "windows_seen": self._windows_seen,
            "seq_actions": [list(a) for a in self._seq_actions],
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("kind") != "replay":
            raise ValueError(f"not a replay state (kind={state.get('kind')!r})")
        if int(state.get("state_version", -1)) != REPLAY_STATE_VERSION:
            raise ValueError(
                f"unsupported replay state version {state.get('state_version')!r}"
            )
        self.slice_id = int(state["slice_id"])
        self.n_prbs = int(state["n_prbs"])
        self.num_slices = int(state["num_slices"])
        base = str(state["base"])
        if base not in REPLAY_BASES:
            raise ValueError(f"base must be one of {REPLAY_BASES}, got {base!r}")
        self.base = base
        self.replay_at_win = int(state["replay_at_win"])
        if self.replay_at_win < 1:
            raise ValueError(
                f"replay_at_win must be >= 1, got {self.replay_at_win}"
            )
        action = tuple(int(v) for v in state["replay_action"])
        if len(action) != self.num_slices or any(v < 0 for v in action):
            raise ValueError(f"invalid replay_action {action}")
        if sum(action) > self.n_prbs:
            raise ValueError(
                f"replay_action {action} sums to {sum(action)} > "
                f"C={self.n_prbs}"
            )
        self.replay_action = action
        self._windows_seen = int(state.get("windows_seen", 0))
        # Rebuild the delegate on load (the prop controller's own state is
        # parameter-only — no learning state to restore).
        self._prop = (
            PropDemandController(
                slice_id=self.slice_id,
                n_prbs=self.n_prbs,
                num_slices=self.num_slices,
            )
            if base == "prop"
            else None
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ReplayController(slice_id={self.slice_id}, base={self.base!r}, "
            f"n_prbs={self.n_prbs}, replay_at_win={self.replay_at_win}, "
            f"replay_action={self.replay_action}, windows_seen="
            f"{self._windows_seen})"
        )


# Re-exported so the driver/tests can import the prop constants from one
# place without touching the static module further.
__all__ = [
    "MODE_PROP",
    "MODE_REPLAY_EQUAL",
    "MODE_REPLAY_OVERRIDE",
    "MODE_REPLAY_PROP",
    "PROP_CAPACITY_PER_RBG_MBPS",
    "PROP_FLOOR_FRACS",
    "PROP_SLA_TARGETS",
    "REPLAY_BASES",
    "REPLAY_EQUAL_WEIGHTS",
    "REPLAY_STATE_VERSION",
    "ReplayController",
    "initial_equal_split",
    "parse_replay_action",
]
