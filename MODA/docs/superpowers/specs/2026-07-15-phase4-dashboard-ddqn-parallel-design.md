# Phase 4 Dashboard and DDQN Parallelism Design

## Scope

Update the existing Streamlit dashboard to show Phase 4 RSLAQ DDQN progress for only these scenarios: `low_traffic`, `normal`, `stressed`, and `congestion`.

Ensure the active DDQN campaign runs with four parallel DRL jobs, not one, while preserving the current paper scope and excluding `insufficient_resources`.

## Dashboard Design

Add read-only Phase 4 monitoring based on `rslaq_ddqn_paper/` outputs and logs. Expected jobs are `4 scenarios x 3 seeds x 1 replicate = 12` DDQN paper-faithful runs.

The dashboard should show:

- Phase 4 completion metric as `completed/12`.
- Running DDQN process count.
- Phase 4 progress by scenario and seed.
- Per-job status inferred from output directories, `ddqn_training_log.csv`, `ddqn_summary.json`, and process/log activity.
- Recent DDQN training activity when available.

No dashboard action should modify simulations; it remains read-only.

## Execution Design

If the current campaign has not entered Phase 4, stop/relaunch the Phase 3+4 wrapper with `DDQN_DRL_JOBS=4`.

If Phase 4 has already started with `DRL_JOBS=1`, stop the wrapper and DDQN child processes, then relaunch Phase 4-only or Phase 3+4 with resume semantics and `DDQN_DRL_JOBS=4`.

The relaunch must keep:

- `SCENARIOS="low_traffic normal stressed congestion"`.
- `SEEDS="1 2 3"`.
- `RUN_BASELINES=0`.
- `RUN_METAHEURISTICS=0`.
- `RUN_RSLAQ_DDQN_PAPER=1`.
- `BUILD_NS3=0`.

## Verification

Verify that:

- No `insufficient_resources` process is active.
- The new campaign log reports `DRL jobs (parallel): 4` once Phase 4 starts.
- The dashboard Python file passes syntax compilation.
- The dashboard displays Phase 4 counts from the output directory without requiring Streamlit to modify files.
