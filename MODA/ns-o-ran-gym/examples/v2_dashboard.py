#!/usr/bin/env python3
"""Live dashboard for the v2 campaign — zero dependencies (stdlib only).

No Streamlit / no venv needed (the .venv-dashboard used by the v1 dashboard does
not exist on this machine). Serves a self-refreshing HTML page that scans the v2
campaign roots on every load. Strictly read-only: never writes to the campaign.

Sections
  A. Global state: phase, live processes, progress bars, throughput, ETA
  B. Per-scenario overview: metaheuristic progress, Path C feasibility, best score
  C. Baseline matrix: feasibility of every baseline mode x scenario (the
     non-optimization side of the paper's central comparison)
  D. Optimization vs baselines: feasibility gap per scenario (the key result)
  E. Metaheuristic heatmap: scenario x seed completion, per-method breakdown
  F. Live activity: exactly which sims are running right now (scenario, mode or
     weights, seed, elapsed) + recent completions + failures

Usage:
    cd ns-o-ran-gym
    python3 examples/v2_dashboard.py            # http://localhost:8600
    python3 examples/v2_dashboard.py --port 8700
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V2_ROOT = REPO_ROOT / "ns-o-ran-gym" / "results_controlled" / "heuristics_metaheuristics_v2"
DEFAULT_CAMPAIGNS = ["v2_pilot_congestion", "v2_all_scenarios"]

sys.path.insert(0, str(REPO_ROOT / "ns-o-ran-gym" / "src"))
from nsoran.scoring import read_summary, score_summary_rows_v2  # noqa: E402

# Campaign config (iter=4, pop=6)
EVALS_PER_METHOD = {"ga": 24, "pso": 24, "sa": 24, "hybrid": 28}
EVALS_PER_PAIR = sum(EVALS_PER_METHOD.values())  # 100
METHODS = ["ga", "pso", "sa", "hybrid"]
METHOD_LABEL = {"ga": "GA", "pso": "PSO", "sa": "SA", "hybrid": "Híbrida"}
SEEDS = list(range(1, 11))
SCENARIOS = ["low_traffic", "normal", "stressed", "congestion", "insufficient_resources"]
SCEN_SHORT = {"low_traffic": "Low", "normal": "Normal", "stressed": "Stressed",
              "congestion": "Congestion", "insufficient_resources": "Insuff.Res."}
BASELINE_MODES = [
    "pure_rr", "pure_pf", "pure_bcqi",
    "psta_equal", "slice_weighted_rr", "slice_weighted_pf", "slice_weighted_bcqi",
    "slice_aqps", "slice_demand_greedy", "slice_sla_greedy", "slice_least_waste",
    "slice_qos_mixed", "slice_random_vine", "slice_meta_risk_elastic",
]
BASELINE_LABEL = {
    "pure_rr": "Pure RR", "pure_pf": "Pure PF", "pure_bcqi": "Pure BCQI",
    "psta_equal": "P-STA equal", "slice_weighted_rr": "Weighted RR",
    "slice_weighted_pf": "Weighted PF", "slice_weighted_bcqi": "Weighted BCQI",
    "slice_aqps": "AQPS", "slice_demand_greedy": "Demand-greedy",
    "slice_sla_greedy": "SLA-greedy", "slice_least_waste": "Least-waste",
    "slice_qos_mixed": "QoS-mixed", "slice_random_vine": "Random-vine",
    "slice_meta_risk_elastic": "Meta-risk-elast.",
}
BASELINE_GROUP = {  # for visual grouping
    **{m: "Schedulers puros" for m in ("pure_rr", "pure_pf", "pure_bcqi")},
    **{m: "Pesos fixos" for m in ("psta_equal", "slice_weighted_rr",
                                  "slice_weighted_pf", "slice_weighted_bcqi")},
    **{m: "Heurísticas adaptativas" for m in (
        "slice_aqps", "slice_demand_greedy", "slice_sla_greedy", "slice_least_waste",
        "slice_qos_mixed", "slice_random_vine", "slice_meta_risk_elastic")},
}
BASELINE_TARGET_PER_SCEN = len(BASELINE_MODES) * len(SEEDS)  # 140
REFRESH_S = 30

# --- caches keyed by (path, mtime) so re-scans stay cheap -------------------
_score_cache: dict[str, tuple[float, float]] = {}
_sidecar_cache: dict[str, tuple[float, dict]] = {}


def _score_of(path: str) -> float | None:
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    hit = _score_cache.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        sc = score_summary_rows_v2(read_summary(Path(path)))
    except Exception:  # noqa: BLE001
        return None
    _score_cache[path] = (mt, sc)
    return sc


def _sidecar(path: str) -> dict | None:
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    hit = _sidecar_cache.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    d["_mtime"] = mt
    _sidecar_cache[path] = (mt, d)
    return d


# ---------------------------------------------------------------------------
def scan(campaigns: list[str]) -> dict:
    meta = {}      # scenario -> stats
    pair = defaultdict(lambda: defaultdict(int))        # scenario -> seed -> n evals
    pair_meth = defaultdict(lambda: defaultdict(int))   # (scen,seed) -> method -> n
    # best score per (scenario, seed): the optimization's per-seed outcome, which
    # is what compares apples-to-apples against a baseline's single run per seed.
    pair_best = defaultdict(dict)                       # scenario -> seed -> best score
    base = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # scen -> mode -> [feas, done]
    mtimes: list[float] = []

    for camp in campaigns:
        for f in glob.glob(str(V2_ROOT / camp / "metaheuristics" / "**/candidate.json"),
                           recursive=True):
            d = _sidecar(f)
            if not d:
                continue
            sc, m, seed = d.get("scenario"), d.get("method"), d.get("seed")
            st = meta.setdefault(sc, {"by_method": defaultdict(int), "feasible": 0,
                                      "total": 0, "failed": 0, "best": None,
                                      "best_w": None, "best_method": None})
            mtimes.append(d["_mtime"])
            if d.get("failed"):
                st["failed"] += 1
                continue
            st["by_method"][m] += 1
            st["total"] += 1
            pair[sc][seed] += 1
            pair_meth[(sc, seed)][m] += 1
            s = d.get("score")
            if isinstance(s, (int, float)):
                prev = pair_best[sc].get(seed)
                if prev is None or s > prev:
                    pair_best[sc][seed] = s
                if s >= 0:
                    st["feasible"] += 1
                    if st["best"] is None or s > st["best"]:
                        st["best"], st["best_w"], st["best_method"] = s, d.get("weights"), m

        for f in glob.glob(str(V2_ROOT / camp / "heuristics_ns3" /
                               "results_rslaq_network_only" / "**/summary.csv"),
                           recursive=True):
            parts = Path(f).parts
            sc = next((p.split("=")[1] for p in parts if p.startswith("scenario=")), None)
            mode = next((p.split("=")[1] for p in parts if p.startswith("mode=")), None)
            if not sc or not mode:
                continue
            base[sc][mode][1] += 1
            s = _score_of(f)
            if s is not None and s >= 0:
                base[sc][mode][0] += 1
            try:
                mtimes.append(os.path.getmtime(f))
            except OSError:
                pass

    now = time.time()
    return {"meta": meta, "pair": pair, "pair_meth": pair_meth, "base": base,
            "pair_best": pair_best,
            "rate_1h": sum(1 for t in mtimes if now - t < 3600),
            "rate_10m": sum(1 for t in mtimes if now - t < 600),
            "last_write": max(mtimes) if mtimes else 0}


def running_sims() -> list[dict]:
    """What is executing right now: scenario, mode/weights, seed, elapsed."""
    out = []
    try:
        ps = subprocess.run(["ps", "-eo", "etimes,args"], capture_output=True,
                            text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001
        return out
    for ln in ps.splitlines():
        if "ns3.46-rslaq-sim-default" not in ln or "grep" in ln:
            continue
        try:
            secs = int(ln.strip().split()[0])
        except (ValueError, IndexError):
            secs = 0
        g = lambda p: (re.search(p, ln).group(1) if re.search(p, ln) else "?")  # noqa: E731
        mode = g(r"--baselineMode=(\S+)")
        out.append({
            "scenario": g(r"--scenario=(\S+)"),
            "mode": mode,
            "weights": g(r"--weights=(\S+)") if mode == "slice_custom" else "",
            "seed": g(r"--seed=(\d+)"),
            "elapsed": secs,
            "eval": (re.search(r"evals/([a-z]+_eval_\d+)", ln).group(1)
                     if re.search(r"evals/([a-z]+_eval_\d+)", ln) else ""),
        })
    return sorted(out, key=lambda d: -d["elapsed"])


def n_procs(pat: str) -> int:
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=5).stdout
        return sum(1 for ln in ps.splitlines() if pat in ln and "grep" not in ln)
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def bar(done: int, total: int, w: int = 260, show: bool = True) -> str:
    pct = min(100.0, 100.0 * done / total) if total else 0.0
    col = "#16a34a" if pct >= 99.5 else "#0072B2"
    label = f'<span class="barlab">{done}/{total} · {pct:.0f}%</span>' if show else ""
    return (f'<span class="bar" style="width:{w}px"><span class="fill" '
            f'style="width:{pct:.1f}%;background:{col}"></span></span>{label}')


def heat(v: float | None, vmax: float = 10.0) -> str:
    """Color cell by feasibility count (0..vmax): red -> amber -> green."""
    if v is None:
        return "background:#f1f3f5;color:#adb5bd"
    r = max(0.0, min(1.0, v / vmax if vmax else 0))
    if r < 0.5:
        # red -> amber
        t = r / 0.5
        rr, gg, bb = 220, int(60 + 130 * t), 60
    else:
        t = (r - 0.5) / 0.5
        rr, gg, bb = int(220 - 150 * t), int(190 - 10 * t), int(60 + 20 * t)
    fg = "#fff" if r < 0.35 or r > 0.8 else "#222"
    return f"background:rgb({rr},{gg},{bb});color:{fg};font-weight:600"


def fmt_elapsed(s: int) -> str:
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s%60:02d}s"


def render(st: dict) -> str:
    meta, base, pair, pair_meth = st["meta"], st["base"], st["pair"], st["pair_meth"]
    sims = running_sims()
    n_meta = n_procs("run_rslaq_metaheuristics.py --method")
    n_sim = len(sims)
    scens = [s for s in SCENARIOS if s in meta or s in base]

    meta_done = sum(v["total"] + v["failed"] for v in meta.values())
    meta_target = len(scens) * EVALS_PER_PAIR * len(SEEDS)
    base_done = sum(sum(c[1] for c in modes.values()) for modes in base.values())
    base_target = len(scens) * BASELINE_TARGET_PER_SCEN

    phase = "Fase 2 — metaheurísticas" if n_meta else ("Fase 1 — baselines" if n_sim else "—")
    alive = n_sim > 0 or n_meta > 0
    status = ('<span class="pill live">● rodando</span>' if alive
              else '<span class="pill idle">○ ocioso / concluído</span>')

    # ETA from the last hour's throughput
    remaining = max(0, (meta_target - meta_done) + (base_target - base_done))
    eta = "—"
    if st["rate_1h"] > 3 and remaining:
        hrs = remaining / st["rate_1h"]
        eta = f"~{hrs:.0f} h" if hrs < 48 else f"~{hrs/24:.1f} dias"
    last = (f'{int((time.time()-st["last_write"])//60)} min atrás'
            if st["last_write"] else "—")

    # ---- B: per-scenario overview ----
    ov = []
    for sc in scens:
        m = meta.get(sc, {"by_method": {}, "feasible": 0, "total": 0, "failed": 0,
                          "best": None, "best_w": None, "best_method": None})
        mdone = m["total"] + m["failed"]
        mt = EVALS_PER_PAIR * len(SEEDS)
        feas = f'{m["feasible"]}/{m["total"]}' if m["total"] else "—"
        feas_pct = (100.0 * m["feasible"] / m["total"]) if m["total"] else 0
        bmodes = base.get(sc, {})
        bdone = sum(c[1] for c in bmodes.values())
        best = f'{m["best"]:.1f}' if m["best"] is not None else "—"
        w = m.get("best_w")
        wtxt = (f'[{w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}]'
                if isinstance(w, list) and len(w) == 3 else "—")
        meth = METHOD_LABEL.get(m.get("best_method"), "")
        per_meth = " ".join(
            f'<span class="chip">{METHOD_LABEL[k]} {m["by_method"].get(k,0)}/'
            f'{EVALS_PER_METHOD[k]*len(SEEDS)}</span>' for k in METHODS)
        ov.append(f"""<tr>
          <td><b>{SCEN_SHORT.get(sc, sc)}</b><div class="sub">{sc}</div></td>
          <td>{bar(mdone, mt, 170)}<div class="chips">{per_meth}</div></td>
          <td><b>{feas}</b><div class="sub">{feas_pct:.0f}% viável</div></td>
          <td><b>{best}</b><div class="sub">{meth} {wtxt}</div></td>
          <td>{bar(bdone, BASELINE_TARGET_PER_SCEN, 110)}</td>
          <td class="{'warn' if m['failed'] else ''}">{m['failed']}</td>
        </tr>""")

    # ---- C: baseline matrix (mode x scenario feasibility) ----
    brows, last_group = [], None
    for mode in BASELINE_MODES:
        grp = BASELINE_GROUP[mode]
        if grp != last_group:
            brows.append(f'<tr class="grouprow"><td colspan="{len(scens)+2}">{grp}</td></tr>')
            last_group = grp
        cells, tot_f, tot_d = [], 0, 0
        for sc in scens:
            c = base.get(sc, {}).get(mode)
            if c and c[1]:
                tot_f += c[0]; tot_d += c[1]
                cells.append(f'<td style="{heat(c[0], c[1])}" title="{c[0]} viáveis de {c[1]} seeds">'
                             f'{c[0]}/{c[1]}</td>')
            else:
                cells.append('<td style="background:#f1f3f5;color:#adb5bd">—</td>')
        agg = f"{tot_f}/{tot_d}" if tot_d else "—"
        brows.append(f'<tr><td class="mode">{BASELINE_LABEL[mode]}</td>{"".join(cells)}'
                     f'<td class="agg">{agg}</td></tr>')
    bhead = "".join(f"<th>{SCEN_SHORT.get(s,s)}</th>" for s in scens)

    # ---- D: optimization vs baselines (best-per-seed, apples to apples) ----
    # A baseline contributes ONE run per seed, so its score is "seeds feasible /10".
    # The optimization's equivalent is "seeds whose BEST candidate is feasible /10",
    # NOT the fraction of feasible evals (a search explores bad regions on purpose).
    drows = []
    for sc in scens:
        pb = st["pair_best"].get(sc, {})
        opt_seeds = len(pb)
        opt_feas = sum(1 for v in pb.values() if v >= 0)
        pair_done = sum(1 for seed in SEEDS
                        if pair.get(sc, {}).get(seed, 0) >= EVALS_PER_PAIR)
        bmodes = base.get(sc, {})
        cand = [(c[0], BASELINE_LABEL[mo]) for mo, c in bmodes.items() if c[1]]
        best_base = max(cand, default=(None, "—"))
        n_good = sum(1 for c in bmodes.values() if c[1] and c[0] >= 5)
        if opt_seeds:
            partial = "" if pair_done == opt_seeds else f' <span class="sub">(parcial)</span>'
            opt_txt = f'<b>{opt_feas}/{opt_seeds}</b> seeds{partial}'
            delta = (f'<b style="color:{"#16a34a" if opt_feas > (best_base[0] or 0) else "#6b7280"}">'
                     f'{opt_feas - (best_base[0] or 0):+d}</b>') if best_base[0] is not None else "—"
        else:
            opt_txt, delta = '<span class="sub">aguardando Fase 2</span>', "—"
        drows.append(f"""<tr>
          <td><b>{SCEN_SHORT.get(sc, sc)}</b></td>
          <td>{opt_txt}</td>
          <td>{('<b>'+best_base[1]+'</b> '+str(best_base[0])+'/10') if best_base[0] is not None else '—'}</td>
          <td>{delta}</td>
          <td><b>{n_good}</b>/{len(bmodes) if bmodes else 0}</td>
        </tr>""")

    # ---- E: metaheuristic heatmap scenario x seed ----
    hrows = []
    for sc in scens:
        cells = []
        for seed in SEEDS:
            n = pair.get(sc, {}).get(seed, 0)
            pct = 100.0 * n / EVALS_PER_PAIR
            style = heat(min(n, EVALS_PER_PAIR), EVALS_PER_PAIR) if n else \
                "background:#f1f3f5;color:#adb5bd"
            det = " ".join(f"{METHOD_LABEL[k]}:{pair_meth.get((sc,seed),{}).get(k,0)}"
                           for k in METHODS)
            cells.append(f'<td style="{style}" title="seed {seed} — {det}">{pct:.0f}%</td>')
        hrows.append(f'<tr><td class="mode">{SCEN_SHORT.get(sc,sc)}</td>{"".join(cells)}</tr>')
    hhead = "".join(f"<th>s{seed}</th>" for seed in SEEDS)

    # ---- F: live activity ----
    if sims:
        srows = "".join(
            f'<tr><td>{SCEN_SHORT.get(s["scenario"], s["scenario"])}</td>'
            f'<td>{"<b>"+BASELINE_LABEL.get(s["mode"], s["mode"])+"</b>" if s["mode"]!="slice_custom" else "<i>otimização</i>"}</td>'
            f'<td class="mono">{s["weights"] or s["eval"] or "—"}</td>'
            f'<td>{s["seed"]}</td>'
            f'<td class="{"warn" if s["elapsed"]>3600 else ""}">{fmt_elapsed(s["elapsed"])}</td></tr>'
            for s in sims)
    else:
        srows = '<tr><td colspan="5" class="sub">nenhuma simulação em execução</td></tr>'

    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{REFRESH_S}">
<title>Campanha v2 — {meta_done + base_done} runs</title>
<style>
:root{{--bg:#fff;--fg:#1a1d21;--mut:#6b7280;--line:#e5e7eb;--card:#fafbfc;--acc:#0072B2}}
@media(prefers-color-scheme:dark){{:root{{--bg:#15181c;--fg:#e8eaed;--mut:#9aa0a6;
--line:#2c3138;--card:#1c2025}}}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,Arial;
margin:0;padding:22px 26px;background:var(--bg);color:var(--fg);font-size:14px}}
h1{{font-size:19px;margin:0 0 4px}} h2{{font-size:14px;margin:26px 0 8px;
text-transform:uppercase;letter-spacing:.5px;color:var(--mut)}}
.top{{display:flex;gap:26px;flex-wrap:wrap;align-items:center;margin-bottom:14px}}
.pill{{padding:3px 11px;border-radius:99px;font-size:12px;font-weight:600}}
.live{{background:#dcfce7;color:#166534}} .idle{{background:#f1f3f5;color:#6b7280}}
@media(prefers-color-scheme:dark){{.live{{background:#14532d;color:#bbf7d0}}
.idle{{background:#2c3138;color:#9aa0a6}}}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0 4px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:9px;
padding:9px 14px;min-width:118px}}
.kpi .v{{font-size:19px;font-weight:700}} .kpi .l{{font-size:11px;color:var(--mut)}}
.bar{{display:inline-block;height:9px;background:var(--line);border-radius:5px;
overflow:hidden;vertical-align:middle}}
.fill{{display:block;height:100%;border-radius:5px}}
.barlab{{font-size:11.5px;color:var(--mut);margin-left:7px;white-space:nowrap}}
table{{border-collapse:collapse;width:auto;margin-top:4px}}
td,th{{border:1px solid var(--line);padding:5px 9px;text-align:left;font-size:12.5px}}
th{{background:var(--card);font-size:11.5px;color:var(--mut);font-weight:600;
text-transform:uppercase;letter-spacing:.3px}}
td.mode{{font-weight:600;white-space:nowrap}} td.agg{{font-weight:700;background:var(--card)}}
.grouprow td{{background:var(--card);font-size:11px;font-weight:700;color:var(--mut);
text-transform:uppercase;letter-spacing:.5px;padding:4px 9px}}
.sub{{font-size:11px;color:var(--mut)}} .warn{{color:#dc2626;font-weight:600}}
.chips{{margin-top:3px}} .chip{{display:inline-block;background:var(--card);
border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:10.5px;
color:var(--mut);margin-right:3px}}
.mono{{font-family:ui-monospace,Menlo,monospace;font-size:11px}}
.note{{color:var(--mut);font-size:11.5px;margin-top:8px;max-width:900px;line-height:1.5}}
</style></head><body>

<h1>Campanha v2 — otimização vs baselines</h1>
<div class="top">
  {status}
  <span><b>{phase}</b></span>
  <span class="sub">ns-3: <b>{n_sim}</b> · buscas: <b>{n_meta}</b> · último arquivo: {last}</span>
  <span class="sub">auto-refresh {REFRESH_S}s</span>
</div>

<div class="kpis">
  <div class="kpi"><div class="v">{meta_done}/{meta_target}</div><div class="l">evals metaheurísticos</div></div>
  <div class="kpi"><div class="v">{base_done}/{base_target}</div><div class="l">runs de baseline</div></div>
  <div class="kpi"><div class="v">{st['rate_1h']}</div><div class="l">runs/hora</div></div>
  <div class="kpi"><div class="v">{st['rate_10m']}</div><div class="l">últimos 10 min</div></div>
  <div class="kpi"><div class="v">{eta}</div><div class="l">ETA restante</div></div>
</div>
<div>Metaheurísticas {bar(meta_done, meta_target, 320)}</div>
<div style="margin-top:5px">Baselines &nbsp;&nbsp;&nbsp;&nbsp;{bar(base_done, base_target, 320)}</div>

<h2>A · Visão por cenário</h2>
<table><tr><th>Cenário</th><th>Metaheurísticas (100/seed)</th><th>Viáveis (Path C)</th>
<th>Melhor score</th><th>Baselines</th><th>Falhas</th></tr>
{''.join(ov)}</table>

<h2>B · Matriz de baselines — viabilidade por modo × cenário (nº de seeds com URLLC p99 ≤ 10 ms)</h2>
<table><tr><th>Baseline (sem otimização)</th>{bhead}<th>Total</th></tr>
{''.join(brows)}</table>
<div class="note">Verde = mais seeds latência-viáveis. Esta é a matriz do lado
<b>não-otimizado</b>: mostra quais estratégias sem busca conseguem proteger o URLLC em
cada carga. Comparar com a linha de otimização na seção A/D.</div>

<h2>D · Otimização vs baselines — seeds latência-viáveis (comparação central)</h2>
<table><tr><th>Cenário</th><th>Otimização (melhor por seed)</th><th>Melhor baseline</th>
<th>Δ (otim. − melhor base)</th><th>Baselines ≥5/10</th></tr>
{''.join(drows)}</table>
<div class="note">Comparação pareada correta: um baseline faz <b>1 run por seed</b>, então
vale "seeds viáveis/10"; o equivalente para a busca é o <b>melhor candidato de cada
seed</b> (a busca explora regiões ruins de propósito, logo a fração de evals viáveis
não é comparável). Δ &gt; 0 = a otimização protege o URLLC em mais seeds que qualquer
estratégia sem otimização.</div>

<h2>E · Progresso das buscas — cenário × seed</h2>
<table><tr><th>Cenário</th>{hhead}</tr>{''.join(hrows)}</table>
<div class="note">% do orçamento de 100 evals/seed (GA 24 · PSO 24 · SA 24 · Híbrida 28).
Passe o mouse numa célula para ver a divisão por método.</div>

<h2>F · Em execução agora</h2>
<table><tr><th>Cenário</th><th>Modo</th><th>Pesos / eval</th><th>Seed</th><th>Decorrido</th></tr>
{srows}</table>
<div class="note">Vermelho = simulação acima de 1 h (pesos patológicos com slice faminto
custam muito mais em cenários congestionados). Atualizado {time.strftime('%H:%M:%S')} ·
read-only, nunca escreve na campanha.</div>

</body></html>"""


class Handler(BaseHTTPRequestHandler):
    campaigns = DEFAULT_CAMPAIGNS

    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_response(404); self.end_headers(); return
        try:
            html = render(scan(self.campaigns))
        except Exception as exc:  # noqa: BLE001
            import traceback
            html = f"<pre>dashboard error: {exc}\n\n{traceback.format_exc()}</pre>"
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--campaigns", nargs="+", default=DEFAULT_CAMPAIGNS)
    args = ap.parse_args()
    Handler.campaigns = args.campaigns
    srv = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard v2: http://localhost:{args.port}   (Ctrl+C para parar)")
    print(f"Campanhas: {args.campaigns}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nparado.")


if __name__ == "__main__":
    main()
