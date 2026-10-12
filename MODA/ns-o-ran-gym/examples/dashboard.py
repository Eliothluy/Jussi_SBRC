#!/usr/bin/env python3
"""Real-time monitoring dashboard for the RSLAQ meta-heuristic campaign.

This is a read-only Streamlit app that visualizes the progress and the
comparative quality of the meta-heuristic search (GA, PSO, SA, hybrid) running
on run_tag 20260626_122326. It scans the per-eval ``candidate.json`` sidecars,
the ``best_candidate_*.json`` and ``metaheuristic_results_*.csv`` outputs, and
the Phase 1 ``batch_manifest.csv`` to build a live picture of:

  * overall completion (% of evals, ETA, phase status);
  * per-pair (scenario x seed) progress heatmap;
  * convergence curves and score distributions per method;
  * the current best candidate per scenario (optimized weights).

How to run
----------
Use the dedicated venv (PEP 668 blocks the system python):

    cd /home/elioth/Documentos/artigo_jussi/ns-o-ran-gym
    .venv-dashboard/bin/streamlit run examples/dashboard.py

Then open http://localhost:8501. The page auto-refreshes every 15 s.

Counts (iterations=12, population=6, the current campaign config):
    GA = 72   PSO = 72   SA = 72   hybrid = 84   ->  300 evals / pair
    5 scenarios x 3 seeds = 15 pairs             -> 4500 evals total
Method order within a pair: ga -> pso -> sa -> hybrid.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/elioth/Documentos/artigo_jussi")
RUN_TAG = "20260626_122326"
RESULTS_ROOT = REPO_ROOT / "ns-o-ran-gym" / "results_controlled" / "heuristics_metaheuristics" / RUN_TAG
META_ROOT = RESULTS_ROOT / "metaheuristics"
META_EVAL_ROOT = RESULTS_ROOT / "meta_evaluation"
RSLAQ_DDQN_ROOT = RESULTS_ROOT / "rslaq_ddqn_paper"
BASELINE_MANIFEST = RESULTS_ROOT / "heuristics_ns3" / "results_rslaq_network_only" / "batch_manifest.csv"

SCENARIOS = ["low_traffic", "normal", "congestion", "stressed", "insufficient_resources"]
SEEDS = [1, 2, 3]
METHODS = ["ga", "pso", "sa", "hybrid"]
METHOD_LABELS = {"ga": "GA", "pso": "PSO", "sa": "SA", "hybrid": "Híbrida"}
METHOD_COLORS = {"ga": "#636EFA", "pso": "#EF553B", "sa": "#00CC96", "hybrid": "#AB63FA"}
PHASE4_SCENARIOS = ["low_traffic", "normal", "stressed", "congestion"]
PHASE4_TOTAL_JOBS = len(PHASE4_SCENARIOS) * len(SEEDS)
PHASE4_LOG = RSLAQ_DDQN_ROOT / ".phase4_rslaq_ddqn_paper.log"

# Eval counts per method for iter=12, pop=6. SA budget is equalized to match
# GA/PSO (iterations x population = 72), not the bare iteration count, so the
# comparison across meta-heuristics is fair under an equal evaluation budget.
# The hybrid runs iterations x (population + 1) = 84 (the +1 is the per-iteration
# local SA refinement). Validated against run_rslaq_metaheuristics.py.
EVALS_PER_METHOD = {"ga": 72, "pso": 72, "sa": 72, "hybrid": 84}
EVALS_PER_PAIR = sum(EVALS_PER_METHOD.values())  # 300
TOTAL_EVALS = EVALS_PER_PAIR * len(SCENARIOS) * len(SEEDS)  # 4500
BASELINE_TOTAL = 211  # Phase 1 jobs expected (5 scenarios x 14 modes x 3 seeds + 1)

REFRESH_SECONDS = 15
PARALLEL_WORKERS = 4  # campaign runs with PARALLEL_JOBS=4


# ---------------------------------------------------------------------------
# Data loading (cached with short TTL so the page refreshes live)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_evals() -> pd.DataFrame:
    """Load every candidate.json sidecar under META_ROOT into a DataFrame.

    Falls back to an empty DataFrame (with the expected columns) if nothing
    has been produced yet.
    """
    cols = ["method", "evaluation_id", "scenario", "seed", "run",
            "score", "w_embb", "w_urllc", "w_mtc", "mtime"]
    rows: list[dict] = []
    if not META_ROOT.exists():
        return pd.DataFrame(columns=cols)
    for sidecar in META_ROOT.rglob("candidate.json"):
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            weights = payload.get("weights", [0, 0, 0])
            rows.append({
                "method": payload.get("method", "?"),
                "evaluation_id": int(payload.get("evaluation_id", 0)),
                "scenario": payload.get("scenario", "?"),
                "seed": int(payload.get("seed", 0)),
                "run": int(payload.get("run", 0)),
                "score": float(payload.get("score", 0.0)),
                "w_embb": float(weights[0]) if len(weights) > 0 else 0.0,
                "w_urllc": float(weights[1]) if len(weights) > 1 else 0.0,
                "w_mtc": float(weights[2]) if len(weights) > 2 else 0.0,
                "mtime": sidecar.stat().st_mtime,
            })
        except (OSError, ValueError, KeyError):
            continue
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["mtime_dt"] = pd.to_datetime(df["mtime"], unit="s")
    return df


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_best_candidates() -> pd.DataFrame:
    """Load best_candidate_*.json for every pair that has produced one."""
    cols = ["scenario", "seed", "method", "evaluation_id", "score",
            "w_embb", "w_urllc", "w_mtc"]
    rows: list[dict] = []
    if not META_ROOT.exists():
        return pd.DataFrame(columns=cols)
    for sidecar in META_ROOT.rglob("best_candidate_*_seed*.json"):
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            weights = payload.get("weights", {})
            rows.append({
                "scenario": payload.get("scenario", "?"),
                "seed": int(payload.get("seed", 0)),
                "method": payload.get("method", "?"),
                "evaluation_id": int(payload.get("evaluation_id", 0)),
                "score": float(payload.get("score", 0.0)),
                "w_embb": float(weights.get("eMBB", 0.0)),
                "w_urllc": float(weights.get("URLLC", 0.0)),
                "w_mtc": float(weights.get("MTC", 0.0)),
            })
        except (OSError, ValueError):
            continue
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def phase1_status() -> tuple[int, int, int]:
    """Return (ok, failed, total) for Phase 1 from the batch manifest."""
    if not BASELINE_MANIFEST.exists():
        return 0, 0, BASELINE_TOTAL
    try:
        df = pd.read_csv(BASELINE_MANIFEST)
        total = len(df)
        ok = int(df["status"].str.lower().eq("ok").sum()) if "status" in df.columns else 0
        failed = int(df["status"].str.lower().eq("run_failed").sum()) if "status" in df.columns else 0
        return ok, failed, max(total, BASELINE_TOTAL)
    except (OSError, ValueError):
        return 0, 0, BASELINE_TOTAL


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_phase4_progress() -> pd.DataFrame:
    """Return Phase 4 DDQN progress for the four selected paper scenarios."""
    cols = ["scenario", "seed", "status", "episodes", "steps", "reward", "mtime", "path"]
    rows: list[dict] = []
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
                    reward = float(payload.get("final_avg_100", payload.get("best_avg", float("nan"))))
                    episodes = int(payload.get("episodes", 0) or 0)
                    steps = int(payload.get("interaction_budget", 0) or 0)
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
                    reward_col = "avg_reward" if "avg_reward" in log_df.columns else "total_reward"
                    if reward_col in log_df.columns and not log_df.empty:
                        rewards = pd.to_numeric(log_df[reward_col], errors="coerce").dropna()
                        reward = float(rewards.tail(1).iloc[0]) if not rewards.empty else float("nan")
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
    """Count active DDQN training processes without touching campaign state."""
    try:
        proc = subprocess.run(
            ["ps", "-eo", "cmd"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if "rslaq_train_ddqn.py" in line)


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------
def per_pair_progress(df: pd.DataFrame) -> pd.DataFrame:
    """Build a (scenario x seed) progress matrix with method detail."""
    records = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            sub = df[(df["scenario"] == scenario) & (df["seed"] == seed)] if not df.empty else df
            method_counts = {m: int((sub["method"] == m).sum()) for m in METHODS} if not sub.empty else {m: 0 for m in METHODS}
            total = sum(method_counts.values())
            pct = min(100.0, 100.0 * total / EVALS_PER_PAIR)
            current_method = _current_method(method_counts)
            best_score = float(sub["score"].max()) if not sub.empty else float("nan")
            records.append({
                "scenario": scenario,
                "seed": seed,
                "ga": method_counts["ga"],
                "pso": method_counts["pso"],
                "sa": method_counts["sa"],
                "hybrid": method_counts["hybrid"],
                "total": total,
                "pct": pct,
                "current": current_method,
                "best_score": best_score,
            })
    return pd.DataFrame(records)


def _current_method(counts: dict[str, int]) -> str:
    """Detect which method a pair is currently running, given per-method counts."""
    if counts["ga"] < EVALS_PER_METHOD["ga"]:
        return "ga" if counts["ga"] > 0 else "—"
    if counts["pso"] < EVALS_PER_METHOD["pso"]:
        return "pso"
    if counts["sa"] < EVALS_PER_METHOD["sa"]:
        return "sa"
    if counts["hybrid"] < EVALS_PER_METHOD["hybrid"]:
        return "hybrid"
    return "✓ done"


def estimate_eta(df: pd.DataFrame) -> str:
    """Estimate wall-clock time remaining from recent eval throughput.

    Uses mtimes of evals produced in the last 10 minutes to compute
    evals/second, then projects the remaining evals divided by the parallel
    worker count.
    """
    if df.empty or len(df) < 5:
        return "— (aguardando dados)"
    now = time.time()
    recent = df[df["mtime"] > now - 600]  # last 10 min
    if len(recent) < 3:
        recent = df.sort_values("mtime").tail(10)
    if len(recent) < 2:
        return "—"
    span = recent["mtime"].max() - recent["mtime"].min()
    if span <= 1:
        return "—"
    rate = len(recent) / span  # evals/sec across all workers
    remaining = TOTAL_EVALS - len(df)
    if rate <= 0:
        return "—"
    seconds_left = remaining / rate  # already accounts for all workers in `rate`
    hours = int(seconds_left // 3600)
    minutes = int((seconds_left % 3600) // 60)
    return f"~{hours}h{minutes:02d}m"


def scan_errors() -> list[str]:
    """Return lines of .meta_*.log that mention tracebacks/errors."""
    hits = []
    if not META_ROOT.exists():
        return hits
    for log in META_ROOT.glob(".meta_*.log"):
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            low = line.lower()
            if "traceback" in low or "runtimeerror" in low or "run failed" in low:
                hits.append(f"{log.name}: {line.strip()[:160]}")
    return hits


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="RSLAQ Campaign Monitor", page_icon="📊", layout="wide")
    st.title("📊 RSLAQ — Monitor da Campanha de Meta-heurísticas")
    st.caption(f"run_tag `{RUN_TAG}` · auto-refresh a cada {REFRESH_SECONDS}s · read-only")

    # Auto-refresh
    st_autorefresh(interval=REFRESH_SECONDS * 1000, key="autorefresh")

    df = load_evals()
    best_df = load_best_candidates()
    phase4_df = load_phase4_progress()
    running_ddqn = count_running_ddqn_processes()
    pairs = per_pair_progress(df)
    ok1, fail1, total1 = phase1_status()
    errors = scan_errors()

    # ---- Section A: overview -------------------------------------------------
    n_evals = len(df)
    pct_global = 100.0 * n_evals / TOTAL_EVALS if TOTAL_EVALS else 0.0
    best_score = float(df["score"].max()) if not df.empty else float("nan")

    st.subheader("Visão geral")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Concluído", f"{pct_global:.1f}%", f"{n_evals}/{TOTAL_EVALS} evals")
    c2.metric("ETA restante", estimate_eta(df))
    c3.metric("Melhor score", f"{best_score:.2f}" if df.empty is False else "—")
    c4.metric("Phase 1 (baselines)", f"{ok1}/{total1}", f"{fail1} falhas" if fail1 else "ok")
    phase2_done = "✅" if pct_global >= 99.5 else ("🔄 em execução" if n_evals > 0 else "⏳ agendada")
    c5.metric("Phase 2 (meta-heur)", phase2_done)

    st.progress(pct_global / 100.0, text=f"{n_evals}/{TOTAL_EVALS} avaliações ({pct_global:.1f}%)")

    # Phase status row
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.markdown(_phase_badge(1, ok1, total1), unsafe_allow_html=True)
    pc2.markdown(_phase_badge(2, n_evals, TOTAL_EVALS), unsafe_allow_html=True)
    meta_eval_done = len(list(META_EVAL_ROOT.glob("*"))) if META_EVAL_ROOT.exists() else 0
    pc3.markdown(_phase_badge(3, meta_eval_done, len(SCENARIOS)), unsafe_allow_html=True)
    phase4_completed = int(phase4_df["status"].eq("concluída").sum()) if not phase4_df.empty else 0
    pc4.markdown(_phase_badge(4, phase4_completed, PHASE4_TOTAL_JOBS, label="RSLAQ DDQN"), unsafe_allow_html=True)

    if errors:
        st.error(f"⚠️ {len(errors)} erro(s) detectado(s) nos logs (último: {errors[-1]})")

    st.divider()

    # ---- Section B: per-pair heatmap + table ---------------------------------
    st.subheader("Progresso por par (cenário × seed)")
    col_b1, col_b2 = st.columns([1, 1.3])

    with col_b1:
        pivot = pairs.pivot(index="scenario", columns="seed", values="pct")
        pivot = pivot.reindex(index=SCENARIOS, columns=SEEDS)
        fig = px.imshow(
            pivot,
            text_auto=".1f",
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=100,
            labels=dict(color="% concluído"),
            title="% de conclusão por par",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b2:
        show = pairs.copy()
        show["pct_str"] = show["pct"].map("{:.1f}%".format)
        show["best_score"] = show["best_score"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
        show = show.rename(columns={
            "scenario": "cenário", "current": "método atual",
            "pct_str": "%", "best_score": "melhor score",
        })
        st.dataframe(
            show[["cenário", "seed", "ga", "pso", "sa", "hybrid", "total", "%", "método atual", "melhor score"]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ---- Section C: comparative evaluation -----------------------------------
    st.subheader("Avaliação comparativa das meta-heurísticas")

    if df.empty:
        st.info("Ainda não há avaliações para comparar. Aguarde o início da Phase 2.")
    else:
        scenarios_done = sorted(df["scenario"].unique())
        sel_scenario = st.selectbox("Cenário", scenarios_done, index=0)
        seeds_done = sorted(df[df["scenario"] == sel_scenario]["seed"].unique())
        sel_seed = st.selectbox("Seed", seeds_done, index=0)
        show_cummax = st.checkbox("Mostrar 'melhor até agora' (cummax)", value=True)

        sub = df[(df["scenario"] == sel_scenario) & (df["seed"] == sel_seed)].copy()
        sub = sub.sort_values(["method", "evaluation_id"])

        if sub.empty:
            st.info("Sem dados para este par ainda.")
        else:
            cc1, cc2 = st.columns([1.3, 1])

            with cc1:
                st.markdown("**Curva de convergência** (score por avaliação)")
                plot_df = sub.copy()
                if show_cummax:
                    plot_df = (
                        plot_df.sort_values(["method", "evaluation_id"])
                        .assign(best_so_far=lambda d: d.groupby("method")["score"].cummax())
                    )
                    y_col, y_title = "best_so_far", "melhor score até agora"
                else:
                    y_col, y_title = "score", "score"
                fig_conv = px.line(
                    plot_df, x="evaluation_id", y=y_col, color="method",
                    color_discrete_map=METHOD_COLORS,
                    labels={"evaluation_id": "ID da avaliação", y_col: y_title, "method": "método"},
                    title=f"Convergência — {sel_scenario} / seed {sel_seed}",
                )
                fig_conv.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_conv, use_container_width=True)

            with cc2:
                st.markdown("**Distribuição de scores** (boxplot)")
                fig_box = px.box(
                    sub, x="method", y="score", color="method",
                    color_discrete_map=METHOD_COLORS,
                    labels={"method": "método", "score": "score"},
                    title=f"Distribuição — {sel_scenario} / seed {sel_seed}",
                )
                fig_box.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

            # Ranking table for the selected pair
            st.markdown("**Ranking por melhor score** (par selecionado)")
            ranking = (
                sub.groupby("method")
                .agg(melhor=("score", "max"), média=("score", "mean"),
                     mediana=("score", "median"), desvio=("score", "std"),
                     n=("score", "count"))
                .reset_index()
                .sort_values("melhor", ascending=False)
            )
            ranking["method"] = ranking["method"].map(lambda m: METHOD_LABELS.get(m, m))
            ranking[["melhor", "média", "mediana", "desvio"]] = ranking[["melhor", "média", "mediana", "desvio"]].round(3)
            st.dataframe(ranking, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Section D: best candidate per scenario -----------------------------
    st.subheader("Melhor candidato por cenário (pesos otimizados)")
    if best_df.empty:
        st.info("Nenhum `best_candidate_*.json` gerado ainda. Aparece conforme os pares completam.")
    else:
        # Aggregate: take the best (by score) across seeds for each scenario
        agg = best_df.sort_values("score", ascending=False).drop_duplicates("scenario")
        agg = agg.set_index("scenario").reindex(SCENARIOS).reset_index()
        agg = agg[agg["scenario"].isin(best_df["scenario"].unique())]

        cd1, cd2 = st.columns([1.1, 1])
        with cd1:
            show = agg.copy()
            show["method"] = show["method"].map(lambda m: METHOD_LABELS.get(m, m))
            show["score"] = show["score"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
            for c in ["w_embb", "w_urllc", "w_mtc"]:
                show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
            show = show.rename(columns={
                "scenario": "cenário", "method": "método", "score": "score",
                "w_embb": "w_eMBB", "w_urllc": "w_URLLC", "w_mtc": "w_MTC",
            })
            st.dataframe(show[["cenário", "seed", "método", "score", "w_eMBB", "w_URLLC", "w_MTC"]],
                         use_container_width=True, hide_index=True)
        with cd2:
            plot_w = agg.dropna(subset=["w_embb"]).melt(
                id_vars="scenario", value_vars=["w_embb", "w_urllc", "w_mtc"],
                var_name="slice", value_name="peso")
            plot_w["slice"] = plot_w["slice"].map({"w_embb": "eMBB", "w_urllc": "URLLC", "w_mtc": "MTC"})
            if not plot_w.empty:
                fig_w = px.bar(
                    plot_w, x="scenario", y="peso", color="slice",
                    barmode="stack", title="Pesos ótimos por cenário (melhor candidato)",
                    color_discrete_map={"eMBB": "#636EFA", "URLLC": "#EF553B", "MTC": "#00CC96"},
                    labels={"scenario": "cenário"},
                )
                fig_w.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_w, use_container_width=True)

    st.divider()

    # ---- Section E: Phase 4 DDQN progress -----------------------------------
    st.subheader("Phase 4: RSLAQ DDQN paper-faithful")
    p4c1, p4c2, p4c3 = st.columns(3)
    phase4_started = int(phase4_df["status"].ne("agendada").sum()) if not phase4_df.empty else 0
    p4c1.metric("DDQN concluídos", f"{phase4_completed}/{PHASE4_TOTAL_JOBS}")
    p4c2.metric("Jobs em execução", str(running_ddqn), "processos rslaq_train_ddqn.py")
    p4c3.metric("Saídas iniciadas", str(phase4_started))
    st.progress(
        phase4_completed / PHASE4_TOTAL_JOBS if PHASE4_TOTAL_JOBS else 0.0,
        text=f"{phase4_completed}/{PHASE4_TOTAL_JOBS} treinamentos DDQN concluídos",
    )

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

        active_or_recent = phase4_df[phase4_df["mtime"] > 0].sort_values("mtime", ascending=False).head(6).copy()
        if not active_or_recent.empty:
            active_or_recent["atualizado"] = pd.to_datetime(active_or_recent["mtime"], unit="s").dt.strftime("%H:%M:%S")
            st.caption("Atividade DDQN recente")
            st.dataframe(
                active_or_recent[["atualizado", "scenario", "seed", "status", "path"]].rename(columns={
                    "scenario": "cenário",
                }),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # ---- Section E: activity feed -------------------------------------------
    st.subheader("Atividade recente")
    if df.empty:
        st.info("Nenhuma avaliação registrada ainda.")
    else:
        recent = df.sort_values("mtime", ascending=False).head(12).copy()
        recent["tempo"] = recent["mtime_dt"].dt.strftime("%H:%M:%S")
        recent["method"] = recent["method"].map(lambda m: METHOD_LABELS.get(m, m))
        recent["score"] = recent["score"].map("{:.3f}".format)
        st.dataframe(
            recent[["tempo", "scenario", "seed", "method", "evaluation_id", "score"]].rename(columns={
                "tempo": "hora", "scenario": "cenário", "method": "método",
                "evaluation_id": "eval #",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        f"Dados lidos de `{META_ROOT}` · {n_evals} sidecars · "
        f"config: iter=12, pop=6, 4 workers · "
        f"contagens: GA={EVALS_PER_METHOD['ga']} PSO={EVALS_PER_METHOD['pso']} "
        f"SA={EVALS_PER_METHOD['sa']} híbrida={EVALS_PER_METHOD['hybrid']} = "
        f"{EVALS_PER_PAIR}/par · {TOTAL_EVALS} total"
    )


def _phase_badge(phase: int, done: int, total: int, label: str | None = None) -> str:
    name = label or f"Phase {phase}"
    if total <= 0:
        color, status = "#888", "—"
    elif done >= total:
        color, status = "#28a745", f"✅ {done}/{total}"
    elif done > 0:
        color, status = "#ffc107", f"🔄 {done}/{total}"
    else:
        color, status = "#6c757d", f"⏳ agendada"
    return f"<div style='border-left:4px solid {color};padding:6px 10px;background:#f8f9fa;border-radius:4px;'><b>{name}</b><br>{status}</div>"


def st_autorefresh(*, interval: int, key: str) -> None:
    """Best-effort auto-refresh. Uses streamlit-autorefresh if installed,
    otherwise falls back to a client-side meta-refresh via st.markdown."""
    try:
        from streamlit_autorefresh import st_autorefresh as _sar  # type: ignore
        _sar(interval=interval, key=key)
        return
    except ImportError:
        pass
    # Fallback: inject a meta refresh tag that reloads the page client-side.
    st.markdown(
        f'<meta http-equiv="refresh" content="{interval // 1000}">',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
