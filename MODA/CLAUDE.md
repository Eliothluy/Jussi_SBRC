# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read AGENTS.md First

`AGENTS.md` at the repo root is the **authoritative research-scope document** (in Portuguese). Sections 1-13 define the immutable paper scope; Section 14 records the current experimental state, known bugs, and operational rules. On any conflict about research scope or methodology, AGENTS.md wins.

**The (first) paper is about metaheuristics, not DRL.** It compares GA, PSO, SA, and a reformulated hybrid metaheuristic at optimizing the continuous slice PRB-weight vector `[eMBB, URLLC, MTC]` on the weight simplex. Pure schedulers (RR, PF, BCQI) are baselines; RSLAQ (DRL) and AQPS are comparative bases. Do not shift the focus to reinforcement learning, new scheduler design, or a discrete formulation. The DRL/Gymnasium infrastructure below still exists and powers the RSLAQ DDQN baseline (campaign Phase 4), but it is no longer the contribution. (A SECOND, author-authorized work line — the offline DRL xApp, see its own section below — is deliberately separate and does not change these rules.)

Key scope rules from AGENTS.md:
- Decision variables stay the slice allocation weights (`w_i >= 0`, `sum w_i = 1`); the formulation stays continuous.
- The campaign uses 3 seeds (1, 2, 3) by design — do not expand to 30 seeds on your own initiative, and do not claim statistical significance with N=3 (report mean±std, best, median instead).
- Do not claim the hybrid is superior without experimental support (it loses to pure PSO in some scenarios).
- Do not fabricate numeric results or invent experimental parameters.
- Known bugs listed in AGENTS.md §14.3 (window-sampled delay percentiles, ad-hoc SA acceptance scale, PSO simplex re-normalization, hybrid's 84-vs-72 eval budget, shared RNG stream) are **documented paper limitations — do not "fix" them unprompted**, as fixes invalidate the campaign.

## Repository Layout

Two coupled projects plus paper material:

1. **ns-3-dev/** — C++ ns-3 simulator with custom 5G-LENA extensions (`scratch/rslaq/`)
2. **ns-o-ran-gym/** — Python package (`nsoran` + `environments`) with the metaheuristic pipeline, scoring, Gymnasium RL environments, and campaign scripts (`examples/`)

Paper material at the root: `artigo_slices_metaheuristicas_5g.md` (paper draft), reference PDFs, `paper_figures_completed_scenarios_en/`, `figuras_overleaf/`, `resultados_cenarios_finalizados_20260715/` (completed campaign results snapshot), `analysis_rslaq_validation/` (analysis scripts and reports).

## Architecture

### Metaheuristic Search Pipeline (the paper's core)

`ns-o-ran-gym/examples/run_rslaq_metaheuristics.py` runs offline GA / PSO / SA / hybrid search over the weight simplex. Each candidate evaluation spawns one network-only ns-3 run (`--baselineMode=slice_custom --weights=... --intraAlgo=PF`), reads the resulting `summary.csv`, and scores it with `nsoran.scoring.score_summary_rows()`.

The objective (in `ns-o-ran-gym/src/nsoran/scoring.py`) exists in two versions, selectable via `--score_version {v1,v2}` (default `v1`):
- **`score_summary_rows` (v1, campanha original)**: `100 * (0.35*mean_SLA + 0.20*min_SLA + 0.20*mean_offered + 0.15*mean_PDR + 0.10*mean_util)` minus penalties: URLLC delay (`(delay_ms_mean - 10)/20`, weight 25 — uses `delay_ms_mean`, NOT the p95/p99 percentiles), MTC starvation (weight 40), and eMBB buffer (weight 100 × small clamp).
- **`score_summary_rows_v2` (reformulação da auditoria, campanha v2)**: feasibility-first — restrições rígidas (p99_URLLC≤10ms via percentil por-pacote, PDR_URLLC/MTC mínimos, throughput_eMBB≥fração da oferta); infeasível → score<0 proporcional à violação; feasível → média da satisfação de SLA composta por slice. Ver AGENTS.md §14.7. O `summary.csv` v2 traz colunas novas (`delay_ms_p999`, `deadline_violation_pct`, `reliability_in_time_pct`, `sla_thr/pdr/delay_pct`, `plr_detected_pct`) e o `sla_satisfaction_pct` passa a ser composto `min(S_thr,S_pdr,S_delay)`. **Nunca misturar scores v1 e v2**; o sidecar grava `score_version`.

Evaluation budgets per (scenario, seed) pair with the current hyperparameters (`iterations=12`, `population=6`): GA=72, PSO=72, SA=72 (equalized to GA/PSO for fairness), hybrid=84 (the +12 is per-iteration SA local refinement — primary comparison must use the common 72-eval ceiling).

**Checkpointing:** every eval writes a `candidate.json` sidecar; on restart, matching weights (1e-9 tolerance) skip ns-3. Crashed ns-3 runs get a hard-penalty score, are checkpointed as `failed=True` (never retried), and are logged in `evals/.ns3_failures.log`.

Cache rules (critical):
- Changing `scoring.py` → run `examples/rescore_cache.py` to re-score cached sidecars without re-running ns-3.
- Changing `population` or `random_seed` invalidates the cache (RNG stream shifts); changing only `iterations` preserves it.
- A single `random.Random(2026)` is consumed sequentially by GA → PSO → SA → hybrid per pair; keep that ordering.

`examples/run_meta_evaluation.py` re-runs each `best_candidate_<scenario>_seed<seed>.json` as a plain `slice_custom` baseline so metaheuristics can be compared side-by-side with the scheduler baselines.

### Campaign Orchestration

`ns-o-ran-gym/examples/run_all_scenarios.sh` (distinct from `ns-3-dev/run_all_scenarios.sh`) runs the full paper campaign in four phases:
- Phase 1 — baselines (delegates to `ns-3-dev/run_all_scenarios.sh` job pool)
- Phase 2 — metaheuristic search per (scenario, seed) pair
- Phase 3 — meta-evaluation of best candidates
- Phase 4 — RSLAQ DDQN training with the paper-faithful reward (the DRL baseline)

`examples/resume_campaign.sh` resumes the current run tag (`20260626_122326`, under `results_controlled/heuristics_metaheuristics/`) in the background with a PID lock. To kill a campaign, SIGTERM on the PID file may not propagate; use `pkill -KILL -f "ns3.46-rslaq-sim-default|run_rslaq_metaheuristics.py|run_all_scenarios.sh"` — checkpoint sidecars survive SIGKILL.

**Monitoring dashboard:** read-only Streamlit app at `examples/dashboard.py`; run with the dedicated venv: `cd ns-o-ran-gym && .venv-dashboard/bin/streamlit run examples/dashboard.py` (system python3 is PEP 668-blocked).

Note: several campaign scripts default `REPO_ROOT`/`NS3_DIR` to hardcoded home paths (some still say `/home/elioth/...`); override via environment variables when the default is wrong.

### ns-3 / Python Synchronous RL Interface (DRL baseline only)

1. ns-3 runs one step (`periodMs`, typically 10 ms); KPIs written to `rslaq-kpms.txt`
2. Python agent signaled via POSIX semaphore; computes action
3. Action written to `rslaq_actions_for_ns3.csv` (col 1 `sliceId`, col 2 `dedicatedPRB`, col 5 scheduler algo; header optional); ns-3 reads it and continues

### Key Source Files

**ns-3 (`ns-3-dev/scratch/rslaq/`):** `rslaq-sim.cc` (topology, scenarios, CLI, stats/summary output), `rslaq-mac-scheduler.cc/h` (slice-aware MAC scheduler partitioning the RBG/PRB budget by slice weights, with redistribution from inactive slices; also implements the heuristic baseline modes).

**Python (`ns-o-ran-gym/src/`):**
- `nsoran/scoring.py` — the metaheuristic objective function (single source of truth for scoring)
- `nsoran/compute_accounting.py` — compute/cost accounting
- `environments/rslaq_env.py`, `rslaq_reward.py`, `rslaq_kpis.py`, `rslaq_action_spaces.py`, `rslaq_predictive.py`, `rslaq_slice_ids.py` — Gymnasium environment stack for the DRL baseline

### Slices, Topology, Scenarios

Slice IDs (0-based): `0 = eMBB`, `1 = URLLC`, `2 = MTC`. UEs assigned by contiguous ID ranges; each slice gets a UDP downlink flow from remoteHost, with destination port mapping FlowMonitor results back to UE/slice.

Single cell: `remoteHost -> 100Gbps P2P -> PGW/EPC -> 1 NR gNB -> static NR UEs`. PHY: 3.55 GHz, 100 MHz, numerology mu=1, 3GPP UMi, directional gNB panel (65°×7°, bearing 0°, tilt 6°) with UEs on a full 360° disk of radius 100 m — so topology and load are confounded across scenarios (see AGENTS.md §14.4).

Five traffic profiles in `InitScenarios()`: `low_traffic` (57 Mbps offered), `normal` (73), `stressed` (128), `congestion` (245), `insufficient_resources` (295). These are nominal input profiles, not guaranteed operational states — only `congestion` shows unambiguous congestion.

### Baseline Modes (17)

- Pure 5G-LENA schedulers: `pure_rr`, `pure_pf`, `pure_bcqi`
- Slice-aware (via `RslaqMacScheduler`): `slice_rr`, `slice_pf`, `slice_bcqi`, `slice_weighted_*`, `psta_equal`, `slice_custom` (used by the metaheuristics with `--weights=`)
- Heuristic baselines: `slice_demand_greedy`, `slice_sla_greedy`, `slice_least_waste`, `slice_qos_mixed`, `slice_random_vine`, `slice_meta_risk_elastic`, `slice_aqps` (AQPS: integer per-slice RBG budgets with minimum guarantees for active demand plus urgency/priority adjustment)

## Offline DRL xApp (separate work line — second paper)

A parallel line of work, explicitly authorized by the author as **separate from the metaheuristics paper**: 3 policies trained OFFLINE from the logged results in `resultados_cenarios_finalizados_20260715/`, packaged as a conceptual O-RAN xApp, and compared against online-trained RSLAQ. **Scope firewall:** AGENTS.md remains untouched and authoritative for the first paper; the "paper is about metaheuristics, not DRL" rule applies to the FIRST paper only. Do not let this line contaminate the metaheuristics paper's narrative or code paths.

**Honest naming (mandatory in any prose/paper text):** the logged data has no next-state, so the 3 models are contextual-bandit / supervised policies, NOT TD-trained DRL: `ddqn` = Q-regression over 66 simplex bins, `sac` = behavioral cloning of top-20% candidates, `ppo` = reward-weighted regression (tau=0.1). The ddqn/sac/ppo labels are file/CLI identifiers only. Each checkpoint stores an `algo_description` string with the honest description.

### Pipeline (all in `ns-o-ran-gym/`, plain `python3`, CLI on every script)

1. **`examples/build_offline_dataset.py`** → `resultados_cenarios_finalizados_20260715/offline_dataset_v2.parquet` (3,722 rows: 3,591 metaheuristic evals + 131 heuristic runs; columns include `source`, `mode`, `weight_provenance`). The v1 `offline_dataset.parquet` is frozen — never overwrite it. Core logic lives in `src/nsoran/offline_dataset.py` (testable).
2. **`examples/train_offline_rl.py`** → `models/{ddqn,sac,ppo}_offline.pt` + `metrics.json`. Supports `--models`, `--epochs`, `--seed`, and multi-seed (`--seeds 42 43 44` → `models/seed<k>/` + `metrics_multiseed.json` for mean±std). Checkpoints store `norm_params` (normalization travels with the model; legacy v1 checkpoints fall back to dataset-derived norm with a warning). Shared nets/checkpoint I/O in `src/nsoran/offline_models.py`.
3. **`examples/eval_offline_rl.py`** — two modes: `--mode proxy` (nearest dataset candidate, fast, dev only) and `--mode ns3` (re-runs predicted weights as `slice_custom` via `run_meta_evaluation.build_eval_command`, scores with `nsoran.scoring`; resume-friendly, output under `ns-3-dev/results_offline_drl_eval/`). Output CSV carries an `eval_mode` column.
4. **`examples/ingest_online_rslaq.py --results-dir <dir>`** — ingests the online DDQN results trained on the OTHER machine (this repo never trains online); extracts final/best-episode weights per (scenario, seed) into `online_rslaq_summary.csv`, consumed by `eval_offline_rl.py --online-results`.
5. **`examples/xapp_slice_optimizer.py --model {ddqn,sac,ppo}`** — conceptual xApp wrapper (no real E2/A1).

Tests: `tests/test_offline_dataset.py`, `tests/test_offline_models.py` (run with `python3 -m unittest`; pytest is not installed system-wide on this machine, and `.venv-dashboard/` does not exist here).

### Data and comparison rules (on record)

- `pure_rr/pf/bcqi` runs are ALWAYS excluded from the dataset (no slice-weight semantics; attributing their reward to any weight vector would poison the regression). Dynamic-weight heuristic modes (`slice_aqps`, `slice_sla_greedy`, ...) enter as renormalized time-averages of `configured_weight` from `slice_alloc.csv`, tagged `weight_provenance="time_averaged"`; classification uses `metadata.json`'s `slice_weight_policy`, not hardcoded mode names.
- **Never mix score scales unlabeled:** proxy scores inherit the dataset's 5 s search runs; `--mode ns3` defaults to 20 s; the online agent's `total_reward` is on the RSLAQ paper-reward scale and is NOT comparable to `nsoran.scoring` — only its extracted weight vector is, scored through the same path as every other method.
- The fixed vector `[0.3333, 0.4000, 0.2667]` is labeled "P_STA default", not "RSLAQ": the real RSLAQ comparison uses ingested online-training results.

## Build Commands

### ns-3 (C++)
```bash
cd ns-3-dev
./ns3 configure --enable-examples --enable-tests
./ns3 build rslaq-sim
```

### Python package
```bash
cd ns-o-ran-gym
hatch build && pip3 install dist/*.tar.gz
```
`pyproject.toml` does not declare `torch` or `pytest`; install PyTorch (>=2.2) and pytest separately for training and some tests.

## Testing

```bash
cd ns-o-ran-gym
python3 -m pytest tests/test_scoring.py tests/test_rslaq_metaheuristics.py tests/test_meta_evaluation.py \
  tests/test_rslaq_action_spaces.py tests/test_rslaq_slice_ids.py tests/test_rslaq_kpis.py \
  tests/test_rslaq_reward.py tests/test_compute_accounting.py
```
Avoid `tests/test_check_env.py` and `tests/test_time_ts.py` locally (hardcoded `/workspace/` paths). Runner-contract tests (`test_all_scenarios_runner_contract.py`, `test_controlled_runner_contract.py`) validate the campaign shell scripts.

ns-3 tests: `cd ns-3-dev && ./test.py`

## Running Things

### Single ns-3 run
```bash
cd ns-3-dev
./ns3 run "scratch/rslaq/rslaq-sim --scenario=normal --baselineMode=pure_pf --simTime=5 --outputDir=/tmp/rslaq"
```
For a fixed-weight run: `--baselineMode=slice_custom --weights=0.55,0.25,0.20 --intraAlgo=PF`

### Baseline matrix (ns-3 side)
```bash
cd ns-3-dev
SCENARIOS="normal" BASELINE_MODES="pure_pf" SEEDS="1" RUNS="1" SIM_TIME=5 ./run_all_scenarios.sh
```
Parallel job pool (`PARALLEL_JOBS`, default 6), resumable (`RESUME=1` skips manifest entries marked ok).

### Metaheuristic search (single pair, quick)
```bash
cd ns-o-ran-gym
python3 examples/run_rslaq_metaheuristics.py --method ga --scenario normal --seed 1 --iterations 2 --population 4
```
`--method {ga,pso,sa,hybrid,all}`; `--scenarios`/`--seeds` accept space-separated lists.

### Full paper campaign
```bash
cd ns-o-ran-gym
bash examples/run_all_scenarios.sh          # fresh run tag, all 4 phases
bash examples/resume_campaign.sh            # resume run tag 20260626_122326
```
Phase toggles / overrides: `RUN_BASELINES`, `RUN_METAHEURISTICS`, `META_ITERATIONS`, `META_POPULATION`, `PARALLEL_JOBS`, `SEEDS`, etc.

**Campanha v2** (reformulação da auditoria, score v2; ver AGENTS.md §14.7). Output root separado, v1 intacta:
```bash
cd ns-o-ran-gym
REPO_ROOT=$(cd ../ && pwd) RUN_TAG=v2_<tag> \
OUTPUT_ROOT=$PWD/results_controlled/heuristics_metaheuristics_v2/v2_<tag> \
SCENARIOS="congestion" SEEDS="1 2 3 4 5 6 7 8 9 10" SIM_TIME=15 \
META_ITERATIONS=8 META_POPULATION=6 META_SCORE_VERSION=v2 META_PER_SEED_SEARCH=1 \
RUN_BASELINES=0 RUN_METAHEURISTICS=1 RUN_META_EVALUATION=0 PARALLEL_JOBS=6 \
bash examples/run_all_scenarios.sh   # crie o dir do OUTPUT_ROOT antes (o redirect do log exige)
```
Análise v2: `python3 examples/analyze_v2_stats.py --meta-root <OUTPUT_ROOT>/metaheuristics` (Wilcoxon/Friedman, válidos com n≥5) e `python3 examples/generate_v2_audit_figures.py --meta-root <OUTPUT_ROOT>/metaheuristics` → `paper_v2_campaign/`.

### DRL baseline training (Phase 4)
`examples/rslaq_train_ddqn.py` (discrete, 198 actions with scheduler / 66 without), `rslaq_train_sac.py` (continuous), `rslaq_train_predictive_sac.py`. Gotchas: use `--no-apply-p-sta` (not `--apply_p_sta False`); `run_training.sh` is stale; SAC defaults to `reward_mode=paper`; `observation_mode=paper` is the production path (shape `(4,4)`).

### Figures and analysis
`examples/generate_completed_scenario_figures.py`, `generate_pure_vs_meta_ieee.py`, `generate_pure_vs_meta_ieee_bar_cdf.py`, `generate_pure_vs_meta_pareto.py`, `generate_resource_allocation_ieee.py`, `analyze_metaheuristic_latency.py`; plus `analysis_rslaq_validation/` scripts at the repo root.

## Output Structure

- ns-3 baselines: `<outputDir>/results_rslaq_network_only/scenario=<s>/mode=<m>/seed=<n>_run=<r>/` with `timeseries.csv`, `slice_alloc.csv`, `summary.csv`, `metadata.json` (pure modes have header-only `slice_alloc.csv`). Batch manifest: `results_rslaq_network_only/batch_manifest.csv`.
- Metaheuristic search: `<output_root>/scenario=<s>/seed=<n>/metaheuristic_search/` with `evals/<method>_eval_NNNN/{candidate.json,ns3.log,...}`, `metaheuristic_results_*.csv`, `best_candidate_<scenario>_seed<n>.json`, `evals/.ns3_failures.log`.
- Campaign root: `ns-o-ran-gym/results_controlled/heuristics_metaheuristics/<run_tag>/` with `heuristics_ns3/` (Phase 1), `metaheuristics/` (Phase 2), `meta_evaluation/` (Phase 3).

**Never commit** `results_controlled/`, `*.pid`, or campaign logs (`resume_campaign.log`, `.meta_*.log`, `run_*_parallel*.log`) — the `.gitignore` files already protect them.

## Runtime Contracts

- P_STA formula (Python action conversion): `p_final = static_fraction * weights + (1 - static_fraction) * p_opt`, default weights `0.3333, 0.4000, 0.2667`.
- Reward modes in `rslaq_reward.py`: `paper` (RSLAQ-paper-faithful: `alpha=0.3333, beta=0.4000, gamma=0.2667`; terminal SLA outage gives `-sum(slice_weights)`) and `resource_efficient` (adds resource-efficiency shaping). Never mix metrics from the two modes without labeling which produced them.
- Metric interpretation caveats (AGENTS.md §14.4): `budget_utilization_pct=100` means the slice used its granted budget, not that the cell is full; FlowMonitor throughput includes IP/UDP headers and can exceed configured payload; to diagnose congestion prefer PDR, mean delay, and queues over `plr_pct`.
