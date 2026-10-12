# Phase 4 Dashboard DDQN Parallel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add read-only Phase 4 DDQN progress to the Streamlit dashboard and run the paper-faithful DDQN Phase 4 with four parallel jobs for four scenarios.

**Architecture:** Keep all dashboard changes inside the existing `examples/dashboard.py` file, following its current cached loader and section-rendering pattern. Treat Phase 4 as a separate progress model derived from `rslaq_ddqn_paper/` directories, DDQN training logs, summaries, and active process inspection.

**Tech Stack:** Python 3, Streamlit, pandas, Plotly, existing bash campaign scripts.

## Global Constraints

- Track only `low_traffic`, `normal`, `stressed`, and `congestion` for Phase 4 display.
- Expected Phase 4 jobs are `4 scenarios x 3 seeds x 1 replicate = 12`.
- Dashboard remains read-only and must not start, stop, or modify simulations.
- Relaunch must use `DDQN_DRL_JOBS=4`, `RUN_BASELINES=0`, `RUN_METAHEURISTICS=0`, `RUN_RSLAQ_DDQN_PAPER=1`, and `BUILD_NS3=0`.
- Do not commit changes unless explicitly requested by the user.

---

### Task 1: Add Phase 4 Data Loading To Dashboard

**Files:**
- Modify: `ns-o-ran-gym/examples/dashboard.py`

**Interfaces:**
- Consumes: `RSLAQ_DDQN_ROOT`, `REFRESH_SECONDS`, `pd`, `Path`.
- Produces: `PHASE4_SCENARIOS`, `PHASE4_TOTAL_JOBS`, `load_phase4_progress() -> pd.DataFrame`, `count_running_ddqn_processes() -> int`.

- [ ] **Step 1: Add constants near existing configuration**

```python
PHASE4_SCENARIOS = ["low_traffic", "normal", "stressed", "congestion"]
PHASE4_TOTAL_JOBS = len(PHASE4_SCENARIOS) * len(SEEDS)
PHASE4_LOG = RSLAQ_DDQN_ROOT / ".phase4_rslaq_ddqn_paper.log"
```

- [ ] **Step 2: Add helper functions after `phase1_status()`**

```python
@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_phase4_progress() -> pd.DataFrame:
    cols = ["scenario", "seed", "status", "episodes", "steps", "reward", "mtime", "path"]
    rows = []
    for scenario in PHASE4_SCENARIOS:
        for seed in SEEDS:
            run_dir = RSLAQ_DDQN_ROOT / f"ddqn_paper_{scenario}_seed{seed}"
            training_log = run_dir / "ddqn_training_log.csv"
            summary_json = run_dir / "ddqn_summary.json"
            status = "agendada"
            episodes = 0
            steps = 0
            reward = float("nan")
            mtime = 0.0
            if summary_json.exists():
                status = "concluída"
                mtime = summary_json.stat().st_mtime
                try:
                    payload = json.loads(summary_json.read_text(encoding="utf-8"))
                    reward = float(payload.get("mean_reward", payload.get("avg_reward", float("nan"))))
                except (OSError, ValueError, TypeError):
                    reward = float("nan")
            elif training_log.exists():
                status = "em execução"
                mtime = training_log.stat().st_mtime
                try:
                    log_df = pd.read_csv(training_log)
                    episodes = len(log_df)
                    if "steps" in log_df.columns:
                        steps = int(pd.to_numeric(log_df["steps"], errors="coerce").fillna(0).sum())
                    if "reward" in log_df.columns and not log_df.empty:
                        reward = float(pd.to_numeric(log_df["reward"], errors="coerce").dropna().tail(1).iloc[0])
                except (OSError, ValueError, IndexError):
                    episodes = 0
            elif run_dir.exists():
                status = "iniciada"
                mtime = run_dir.stat().st_mtime
            rows.append({
                "scenario": scenario,
                "seed": seed,
                "status": status,
                "episodes": episodes,
                "steps": steps,
                "reward": reward,
                "mtime": mtime,
                "path": str(run_dir),
            })
    return pd.DataFrame(rows, columns=cols)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def count_running_ddqn_processes() -> int:
    try:
        output = os.popen("ps -eo cmd | rg '[r]slaq_train_ddqn.py' || true").read()
    except OSError:
        return 0
    return sum(1 for line in output.splitlines() if "rslaq_train_ddqn.py" in line)
```

- [ ] **Step 3: Run syntax validation**

Run: `python3 -m py_compile ns-o-ran-gym/examples/dashboard.py`

Expected: exit code 0.

---

### Task 2: Render Phase 4 Progress In Dashboard

**Files:**
- Modify: `ns-o-ran-gym/examples/dashboard.py`

**Interfaces:**
- Consumes: `load_phase4_progress() -> pd.DataFrame`, `count_running_ddqn_processes() -> int`, `PHASE4_TOTAL_JOBS`.
- Produces: overview Phase 4 metrics and a dedicated Phase 4 section.

- [ ] **Step 1: Load Phase 4 data inside `main()`**

```python
    phase4_df = load_phase4_progress()
    running_ddqn = count_running_ddqn_processes()
```

- [ ] **Step 2: Replace generic Phase 4 badge count**

```python
    phase4_completed = int(phase4_df["status"].eq("concluída").sum()) if not phase4_df.empty else 0
    pc4.markdown(_phase_badge(4, phase4_completed, PHASE4_TOTAL_JOBS, label="RSLAQ DDQN"), unsafe_allow_html=True)
```

- [ ] **Step 3: Add Phase 4 section before activity feed**

```python
    st.divider()
    st.subheader("Phase 4: RSLAQ DDQN paper-faithful")
    p4c1, p4c2, p4c3 = st.columns(3)
    phase4_running = int(phase4_df["status"].isin(["em execução", "iniciada"]).sum()) if not phase4_df.empty else 0
    p4c1.metric("DDQN concluídos", f"{phase4_completed}/{PHASE4_TOTAL_JOBS}")
    p4c2.metric("Jobs em execução", str(running_ddqn), "processos rslaq_train_ddqn.py")
    p4c3.metric("Saídas iniciadas", str(phase4_running))
    st.progress(phase4_completed / PHASE4_TOTAL_JOBS if PHASE4_TOTAL_JOBS else 0.0)

    if phase4_df.empty:
        st.info("Phase 4 ainda sem dados.")
    else:
        p4_show = phase4_df.copy()
        p4_show["reward"] = p4_show["reward"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        st.dataframe(
            p4_show[["scenario", "seed", "status", "episodes", "steps", "reward"]].rename(columns={
                "scenario": "cenário",
                "episodes": "episódios logados",
                "steps": "steps logados",
                "reward": "último reward",
            }),
            use_container_width=True,
            hide_index=True,
        )
```

- [ ] **Step 4: Run syntax validation**

Run: `python3 -m py_compile ns-o-ran-gym/examples/dashboard.py`

Expected: exit code 0.

---

### Task 3: Relaunch Campaign With Four DDQN Jobs

**Files:**
- Runtime files only: `ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326/phase3_phase4_4scenarios.log`, `phase3_phase4_4scenarios.pid`.

**Interfaces:**
- Consumes: existing `examples/run_all_scenarios.sh` and current results cache.
- Produces: active wrapper process configured with `DDQN_DRL_JOBS=4`.

- [ ] **Step 1: Inspect current process/log state**

Run: `ps -eo pid,ppid,stat,etime,cmd | rg 'run_all_scenarios\.sh|run_meta_evaluation\.py|run_controlled_rslaq_validation\.sh|rslaq_train_ddqn\.py|ns3\.46-rslaq-sim-default'`

Expected: identify whether current wrapper is still in Phase 3 or already in Phase 4.

- [ ] **Step 2: Stop current wrapper if it is still configured with one DDQN job**

Run: `pkill -TERM -f 'phase3_phase4_4scenarios|run_controlled_rslaq_validation.sh|rslaq_train_ddqn.py|run_meta_evaluation.py' || true`

Expected: current wrapper and children stop; no `rslaq_train_ddqn.py` remains.

- [ ] **Step 3: Relaunch with four DDQN jobs**

Run from `ns-o-ran-gym`:

```bash
setsid env RUN_TAG='20260626_122326' OUTPUT_ROOT='/home/elioth/Documentos/artigo_jussi/ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326' REPO_ROOT='/home/elioth/Documentos/artigo_jussi' NS3_DIR='/home/elioth/Documentos/artigo_jussi/ns-3-dev' GYM_DIR='/home/elioth/Documentos/artigo_jussi/ns-o-ran-gym' RESUME='1' BUILD_NS3='0' RUN_BASELINES='0' RUN_METAHEURISTICS='0' RUN_META_EVALUATION='1' RUN_RSLAQ_DDQN_PAPER='1' PARALLEL_JOBS='4' SCENARIOS='low_traffic normal stressed congestion' SEEDS='1 2 3' RUNS='1' SIM_TIME='5' APP_START='0.4' DRAIN_TIME_SEC='0.2' PERIOD_MS='10' TX_POWER='43' TDD_PATTERN='D|D|8D|4GB|4U|U|U' RLC_MODE='um' META_INTRA_ALGO='PF' DDQN_INTERACTION_STEPS='20000' DDQN_EPISODE_STEPS='100' DDQN_TRAINING_REPLICATES='1' DDQN_DRL_JOBS='4' ENABLE_STEP_LOGGING='1' bash examples/run_all_scenarios.sh > '/home/elioth/Documentos/artigo_jussi/ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326/phase3_phase4_4scenarios.log' 2>&1 & printf '%s\n' "$!" > '/home/elioth/Documentos/artigo_jussi/ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326/phase3_phase4_4scenarios.pid'
```

Expected: new PID is written and Phase 3/4 wrapper starts with the four selected scenarios.

- [ ] **Step 4: Verify scope and no insufficient resources**

Run: `rg -n 'Scenarios          : low_traffic normal stressed congestion|Run RSLAQ DDQN paper: 1' ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326/phase3_phase4_4scenarios.log && ! ps -eo cmd | rg -q '[i]nsufficient_resources'`

Expected: command exits 0.

---

### Task 4: Final Verification

**Files:**
- Verify: `ns-o-ran-gym/examples/dashboard.py`
- Verify: campaign log under `results_controlled/.../phase3_phase4_4scenarios.log`

**Interfaces:**
- Consumes: outputs from Tasks 1-3.
- Produces: final status summary.

- [ ] **Step 1: Compile dashboard**

Run: `python3 -m py_compile ns-o-ran-gym/examples/dashboard.py`

Expected: exit code 0.

- [ ] **Step 2: Verify running process**

Run: `pid=$(tr -d '[:space:]' < ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326/phase3_phase4_4scenarios.pid) && ps -p "$pid" -o pid,stat,etime,cmd`

Expected: process exists unless Phase 3 and Phase 4 completed during verification.

- [ ] **Step 3: Verify Phase 4 parallelism after Phase 4 starts**

Run: `rg -n 'DRL jobs \(parallel\): 4|DRL jobs:          4' ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326/phase3_phase4_4scenarios.log ns-o-ran-gym/results_controlled/heuristics_metaheuristics/20260626_122326/rslaq_ddqn_paper/.phase4_rslaq_ddqn_paper.log`

Expected: match appears once Phase 4 has started; if Phase 3 is still running, report that Phase 4 is queued with `DDQN_DRL_JOBS=4` in the wrapper environment.
