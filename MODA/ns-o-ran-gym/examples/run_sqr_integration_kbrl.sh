#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_sqr_integration_kbrl.sh — validação PONTA-A-PONTA do baseline KBRL
# (Fase 3, PLANO_IMPLEMENTACAO_SQR_NORI.md §3-Fase 3) no MESMO loop fechado
# do SQR: ns-3 (sqr-sim, cota absoluta R27) <-> agente Python
# (--controller kbrl: nsoran.sqr.kbrl, port de Alcaraz et al., IEEE TWC 2022)
# via semáforos POSIX + CSV (contrato CONGELADO
# MODA/docs/sqr/contrato_interface.md v1.2).
#
# O que faz (espelha run_sqr_integration.sh):
#   1. Confere o binário ns3.46-sqr-sim-default (rebuild incremental via
#      ns3_gate.sh apenas se fontes em scratch/sqr forem mais novas);
#   2. roda examples/sqr_run.py --controller kbrl --simTime 60 --seed 1
#      contra o sqr-sim REAL (IPC), aguardando o TÉRMINO LIMPO do par;
#   3. verifica os critérios de aceite abaixo e imprime "INTEGRAÇÃO KBRL OK".
#
# Critérios de aceite (ANÁLOGOS aos do run_sqr_integration.sh):
#   C1  ≥100 passos de decisão completos (contíguos 1..N, sem deadlock —
#       timeout global do script) e exit code 0 dos DOIS processos;
#   C2  ≥90% das janelas com ação aplicada (prb_requested_pct_* != -1);
#   C3  concordância prb_scheduled ~= cota (semântica R27 ratificada, igual
#       ao script SQR): teto (cota + 1 RBG/slot) em 100% das janelas (C3a) e
#       utilização MÁXIMA >= 0.99 por slice com demanda (C3b);
#   C4  step_metrics.csv com as colunas do contrato (STEP_METRICS_COLUMNS)
#       e prb_scheduled preenchida pela telemetria ns-3;
#   C5  p95 do tempo de decisão (decision_time_ms) < 50 ms;
#   C6  aprendizado online KBRL: >=100 janelas reais de atualização e
#       checkpoint válido com memória de suporte do Projectron crescida;
#       fatores de segurança finais reportados.
#   C7  conversão a_i -> dedicatedPRB -> cota RBG coerente (±1 RBG,
#       pareamento passo t ↔ telemetria 100·(t+1), contrato v1.2).
#
# Harmonização de granularidade (PLANO Fase 3) — VIOLAÇÕES NOS DOIS NÍVEIS,
# com denominadores explícitos, REPORTADAS ao final ([info], não-bloqueantes;
# os critérios de aceite são os análogos listados acima):
#   - Nível UE-tempo (R19): taxa pooled = Σ(rate_UE × n_UE) / Σ n_UE sobre
#     pares (janela, slice) elegíveis — denominador = nº de observações
#     UE-janela com UE na slice;
#   - Nível slice-tempo: fração de pares (janela, slice) elegíveis com
#     y_t = p5 < Λ_slice — denominador = nº de janelas com UE na slice
#     (o rótulo +1/-1 que o KBRL aprende é exatamente este indicador).
#
# Uso:
#   bash MODA/ns-o-ran-gym/examples/run_sqr_integration_kbrl.sh [simTime] [seed]
# Variáveis de ambiente: TIMEOUT_S (default 3600), C_PRBS (default 52,
#                       register R01 — apenas o C INICIAL; o C EFETIVO vem da
#                       telemetria sqr_prb_windows.csv, R01 emendada).
#                       GAMMA NÃO se aplica ao KBRL (sem termo de custo — a
#                       economia de RBs é estrutural no scan de ação mínima).
# ---------------------------------------------------------------------------
set -euo pipefail

MODA_ROOT="/home/eliothluy/Documentos/Jussi_SBRC/MODA"
GYM_ROOT="${MODA_ROOT}/ns-o-ran-gym"
NS3_ROOT="${MODA_ROOT}/ns-3-dev"
PYBIN="${GYM_ROOT}/.venv/bin/python"
RUN_SCRIPT="${GYM_ROOT}/examples/sqr_run.py"
NS3_GATE="${NS3_ROOT}/scratch/sqr/ns3_gate.sh"
SQR_EXE="${NS3_ROOT}/build/scratch/sqr/ns3.46-sqr-sim-default"

SIMTIME="${1:-60}"
SEED="${2:-1}"
C_PRBS="${C_PRBS:-52}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${GYM_ROOT}/results/sqr_integration_kbrl/sim${SIMTIME}s_seed${SEED}_kbrl_${STAMP}"
LOG="${OUT_DIR}/sqr_run.log"

fail() { echo "INTEGRAÇÃO KBRL FALHOU: $*" >&2; exit 1; }

[[ -x "${PYBIN}" ]] || fail "interpretador Python do projeto ausente: ${PYBIN}"
[[ -f "${RUN_SCRIPT}" ]] || fail "driver ausente: ${RUN_SCRIPT}"
[[ -x "${NS3_GATE}" ]] || fail "gate do ns-3 ausente: ${NS3_GATE}"
mkdir -p "${OUT_DIR}"

# --- 0) binário do cenário em dia? (rebuild incremental só se stale) -------
if [[ ! -x "${SQR_EXE}" ]] \
   || [[ -n "$(find "${NS3_ROOT}/scratch/sqr" -maxdepth 1 \
                 \( -name '*.cc' -o -name '*.h' \) \
                 -newer "${SQR_EXE}" -print -quit 2>/dev/null)" ]]; then
    echo "[integração-kbrl] binário do sqr-sim ausente ou mais velho que as fontes — build"
    bash "${NS3_GATE}" build > "${OUT_DIR}/ns3_build.log" 2>&1 \
        || { tail -30 "${OUT_DIR}/ns3_build.log" >&2; fail "build do ns-3 falhou"; }
fi
[[ -x "${SQR_EXE}" ]] || fail "executável do cenário não encontrado: ${SQR_EXE}"
echo "[integração-kbrl] OK: binário do cenário em dia: ${SQR_EXE}"

# --- 1) loop fechado (par de processos; o driver espera o fim do ns-3) ----
echo "[integração-kbrl] rodando: sqr_run.py --controller kbrl --simTime ${SIMTIME} --seed ${SEED} (timeout global ${TIMEOUT_S}s)"
set +e
timeout --signal=TERM --kill-after=120 "${TIMEOUT_S}" \
    "${PYBIN}" "${RUN_SCRIPT}" \
    --controller kbrl --simTime "${SIMTIME}" --seed "${SEED}" \
    --c_prbs "${C_PRBS}" --output "${OUT_DIR}" > "${LOG}" 2>&1
PY_RC=$?
set -e
echo "[integração-kbrl] sqr_run.py exit code: ${PY_RC} (log: ${LOG})"
if [[ ${PY_RC} -ne 0 ]]; then
    tail -40 "${LOG}" >&2
    fail "processo Python terminou com código ${PY_RC} (esperado 0; 124=timeout do script)"
fi

SIM_PATH="$(sed -n 's/^\[sqr_run\] sim_path: //p' "${LOG}" | tail -1)"
[[ -n "${SIM_PATH}" && -d "${SIM_PATH}" ]] \
    || fail "sim_path não impresso pelo sqr_run — run não chegou ao fim limpo"
NS3_RC="$(sed -n 's/^\[sqr_run\] ns3_exit_code: //p' "${LOG}" | tail -1)"
[[ "${NS3_RC}" == "0" ]] \
    || fail "processo ns-3 terminou com código '${NS3_RC}' (esperado 0)"
echo "[integração-kbrl] OK: término limpo do par — python exit=0, ns-3 exit=0"
echo "[integração-kbrl] sim_path: ${SIM_PATH}"

# --- 2) critérios de aceite -------------------------------------------------
set +e
"${PYBIN}" - "${OUT_DIR}" "${SIM_PATH}" "${SIMTIME}" "${SEED}" "${C_PRBS}" <<'PYEOF'
import csv
import glob
import importlib.util
import math
import os
import pickle
import statistics
import sys

out_dir, sim_path, simtime, seed, c_prbs = (
    sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]),
    int(sys.argv[5]),
)
L = 3
SLAS = {0: 0.5, 1: 1.0, 2: 1.5}  # Mbps por UE [ARTIGO, Tabela I]
failures = []


def check(name, ok, detail=""):
    tag = "OK" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# Colunas do contrato: fonte única = STEP_METRICS_COLUMNS do próprio driver.
# out_dir = <gym>/results/sqr_integration_kbrl/<run> -> raiz do gym = 3 níveis
gym_root = os.path.abspath(os.path.join(out_dir, "..", "..", ".."))
spec = importlib.util.spec_from_file_location(
    "sqr_run", os.path.join(gym_root, "examples", "sqr_run.py"))
sqr_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sqr_run)

# ---------- C4: step_metrics.csv ----------
step_csv = os.path.join(out_dir, "step_metrics.csv")
check("C4a step_metrics.csv existe e não-vazio",
      os.path.isfile(step_csv) and os.path.getsize(step_csv) > 0)
with open(step_csv, newline="") as f:
    rows = list(csv.DictReader(f))
header = list(rows[0].keys()) if rows else []
check("C4b colunas == STEP_METRICS_COLUMNS (contrato)",
      header == list(sqr_run.STEP_METRICS_COLUMNS),
      f"{len(header)} colunas")
filled = sum(1 for r in rows if r["prb_scheduled"] not in ("", None))
check("C4c prb_scheduled preenchida pela telemetria ns-3",
      len(rows) > 0 and filled >= 0.99 * len(rows),
      f"{filled}/{len(rows)} linhas")
modes = {r["mode"] for r in rows}
print(f"[info] modos de decisão KBRL observados: {sorted(modes)} "
      f"(scan = ação mínima prevista segura + fator de segurança; "
      f"cold = classificador prevê falha para todo a — propõe C, Pi corta)")

# ---------- C1: passos completos, sem deadlock ----------
steps = sorted({int(r["step"]) for r in rows})
check("C1a >=100 passos de decisão completos", len(steps) >= 100,
      f"{len(steps)} passos")
check("C1b passos contíguos 1..N (loop sem buracos/deadlock)",
      steps == list(range(1, len(steps) + 1)),
      f"último passo {steps[-1] if steps else 0}")
ts_vals = [int(r["timestamp_ms"]) for r in rows if r["timestamp_ms"] != ""]
last_ts = max(ts_vals) if ts_vals else -1
check("C1c KPM alcançou o fim da simulação",
      bool(ts_vals) and last_ts >= int((simtime - 1.0) * 1000),
      f"último timestamp de KPM = {last_ts} ms (simTime={simtime}s)"
      + ("" if ts_vals else "; NENHUM timestamp de KPM no step_metrics.csv"))

# ---------- C5: tempo de decisão p95 < 50 ms ----------
dt = sorted(float(r["decision_time_ms"]) for r in rows
            if r["decision_time_ms"] not in ("", None))
p95 = dt[max(0, int(math.ceil(0.95 * len(dt))) - 1)] if dt else float("inf")
check("C5 p95 do tempo de decisão < 50 ms", 0 < p95 < 50.0,
      f"p95={p95:.3f} ms em {len(dt)} amostras (máx={dt[-1]:.3f} ms)"
      if dt else "sem amostras")

# ---------- C6: aprendizado online KBRL ----------
learned_by = {
    s: sum(1 for r in rows if r["slice_id"] == str(s) and r["learned"] == "1")
    for s in range(L)
}
total_learned = sum(learned_by.values())
check("C6a >=100 janelas reais de atualização", total_learned >= 100,
      f"janelas com y_t definido por slice: {learned_by}")
ckpt = os.path.join(out_dir, "sqr_checkpoint.pkl")
support = {}
sec_factors = {}
try:
    with open(ckpt, "rb") as f:
        payload = pickle.load(f)
    for sid, st in payload.get("slices", {}).items():
        support[int(sid)] = int(st["projectron"]["counter"])
        sec_factors[int(sid)] = int(st["security_factor"])
    check("C6b checkpoint KBRL válido com memória de suporte crescida",
          bool(support) and sum(support.values()) > 0,
          f"support vectors por slice: {support}")
    check("C6c checkpoint é do controlador kbrl",
          all(st.get("controller") == "kbrl"
              for st in payload.get("slices", {}).values()),
          f"controller tag: { {s: st.get('controller') for s, st in payload.get('slices', {}).items()} }")
except Exception as exc:  # noqa: BLE001
    check("C6b checkpoint KBRL válido com memória de suporte crescida", False,
          f"falha ao carregar {ckpt}: {exc}")
print(f"[info] fator de segurança final por slice (análogo ao M do SQR): "
      f"{sec_factors}")

# ---------- telemetria do C++ (contrato §3.5) ----------
prb_csv = glob.glob(os.path.join(sim_path, "results_sqr_network_only",
                                 "seed=*", "sqr_prb_windows.csv"))
kpm_csv = glob.glob(os.path.join(sim_path, "results_sqr_network_only",
                                 "seed=*", "sqr_kpm_history.csv"))
check("C0a telemetria C++ presente (sqr_prb_windows.csv)", bool(prb_csv))
check("C0b histórico de KPM presente (sqr_kpm_history.csv)", bool(kpm_csv))
if not (prb_csv and kpm_csv):
    print(f"FALHAS: {failures}")
    sys.exit(1)

with open(prb_csv[0], newline="") as f:
    prb = list(csv.DictReader(f))
n_windows = len(prb)
check("C1d >=100 janelas de decisão no C++", n_windows >= 100,
      f"{n_windows} janelas (esperadas ~{int(simtime * 10)})")

# presença de UE por (janela, slice) — demanda do critério C3
ues_by = {}
with open(kpm_csv[0], newline="") as f:
    for r in csv.DictReader(f):
        key = (int(r["timestamp"]), int(r["sliceId"]))
        ues_by[key] = ues_by.get(key, 0) + 1

# ---------- C2: >=90% das janelas com ação aplicada ----------
applied = sum(
    1 for r in prb
    if max(float(r[f"prb_requested_pct_{s}"]) for s in range(L)) >= 0.0
)
frac_applied = applied / n_windows if n_windows else 0.0
check("C2 >=90% das janelas com ação aplicada (prb_requested != -1)",
      frac_applied >= 0.90,
      f"{applied}/{n_windows} = {frac_applied:.2%} das janelas "
      f"(-1 = timeout do semáforo de controle)")

# ---------- C3: prb_scheduled ~= cota (semântica R27 do smoke) ----------
tot_pairs = conc_pairs = cap_viol = 0
utils_by = {}
for r in prb:
    slots = int(r["dl_slots"])
    if slots <= 0:
        continue
    ts = int(r["timestamp_ms"])
    for s in range(L):
        sched = int(r[f"prb_scheduled_rbg_{s}"])
        budget = int(r[f"prb_budget_rbg_{s}"])  # cota x slots DL da janela
        if sched > budget + slots:
            cap_viol += 1
        if budget <= 0 or ues_by.get((ts, s), 0) == 0:
            continue  # sem cota ou sem demanda: par fora do critério
        tot_pairs += 1
        util = sched / budget
        utils_by.setdefault(s, []).append(util)
        if abs(sched - budget) <= slots:
            conc_pairs += 1
check("C3a cota nunca ultrapassada (sched <= cota + 1 RBG/slot)",
      cap_viol == 0, f"{cap_viol} violações de teto em {n_windows} janelas")
per_slice = {}
for s, utils in sorted(utils_by.items()):
    su = sorted(utils)
    per_slice[s] = (su[-1], statistics.median(utils), len(su))
c3b_ok = bool(per_slice) and all(
    mx >= 0.99 for mx, _med, _n in per_slice.values()
)
frac_conc = (conc_pairs / tot_pairs) if tot_pairs else 0.0
check("C3b prb_scheduled ~= cota com demanda (util max>=0.99 por slice; "
      "semântica do smoke R27 ratificado)", c3b_ok,
      "; ".join(
          f"slice {s}: max={mx:.4f} mediana={med:.4f} ({n} pares)"
          for s, (mx, med, n) in per_slice.items()
      ))
print(f"[info] banda estrita |sched-cota| <= 1 RBG/slot (diagnóstico): "
      f"{conc_pairs}/{tot_pairs} = {frac_conc:.2%}")

# ---------- coerência da conversão Python -> C++ (pct -> RBG) ----------
c_eff = c_prbs
try:
    c_eff = int(float(prb[-1]["total_prb"]))
except (KeyError, IndexError, TypeError, ValueError):
    pass
prb_by_ts = {int(r["timestamp_ms"]): r for r in prb}
mism = tot = 0
for r in rows:
    # Contrato v1.2 (RATIFICADO): a ação decidida no passo t governa a janela
    # (t, t+1] — o dreno da telemetria roda ANTES da espera IPC.
    rr = prb_by_ts.get(100 * (int(r["step"]) + 1))
    if rr is None:
        continue
    rbg_total = float(rr["total_rbg"])
    s = int(r["slice_id"])
    a = float(r["prb_executed"])
    rbg = int(rr[f"prb_executed_rbg_{s}"])
    tot += 1
    if abs(rbg - a * rbg_total / float(c_eff)) > 1.0 + 1e-6:
        mism += 1
check("C7 conversão a_i -> dedicatedPRB -> cota RBG coerente (±1 RBG)",
      tot > 0 and mism <= max(1, int(0.01 * tot)),
      f"{mism}/{tot} pares (step,slice) fora de ±1 RBG (C efetivo={c_eff}, "
      f"inicial={c_prbs})")

# ---------- stdout do C++: ACTION logs e término limpo ----------
stdout_txt = stderr_txt = ""
for name, sink in (("stdout", "s"), ("stderr", "e")):
    path = os.path.join(sim_path, name)
    if os.path.isfile(path):
        with open(path, errors="replace") as f:
            if sink == "s":
                stdout_txt = f.read()
            else:
                stderr_txt = f.read()
full_log = stdout_txt + stderr_txt
n_action = full_log.count("[SqrSim] ACTION")
n_window_log = full_log.count("[SqrSim] PRB_WINDOW")
check("R27 logs de auditoria ACTION/PRB_WINDOW presentes",
      n_action >= 100 and n_window_log >= 100,
      f"{n_action} linhas ACTION, {n_window_log} linhas PRB_WINDOW")
crash_markers = [m for m in ("NS_FATAL", "NS_ASSERT", "std::terminate",
                             "Segmentation fault", "Aborted",
                             "Invalid command-line argument")
                 if m in full_log]
check("C++ sem crash/assert/CLI inválida", not crash_markers,
      f"marcadores: {crash_markers}" if crash_markers else "nenhum")
check("C++ terminou limpo (cleanup de semáforos)",
      "IPC semaphores cleaned up" in stdout_txt)

# ---------- harmonização de granularidade: violações nos DOIS níveis -----
# (PLANO Fase 3; métricas [info] com denominadores explícitos — o KBRL
# aprende o rótulo slice-tempo; o SQR impõe o SLA por UE; ambos logados.)
ue_num = ue_den = 0          # UE-tempo: obs UE-janela elegíveis
slice_num = slice_den = 0    # slice-tempo: janelas com UE na slice
per_slice_viol = {s: [0, 0] for s in range(L)}
for r in rows:
    s = int(r["slice_id"])
    n_ues = int(r["n_ues"]) if r["n_ues"] not in ("", None) else 0
    vr = r["violation_rate"]
    y = r["y_t_p5_mbps"]
    if vr not in ("", None) and n_ues > 0:
        ue_num += float(vr) * n_ues
        ue_den += n_ues
    if y not in ("", None) and n_ues > 0:
        slice_den += 1
        per_slice_viol[s][1] += 1
        if float(y) < SLAS[s]:
            slice_num += 1
            per_slice_viol[s][0] += 1
ue_rate = ue_num / ue_den if ue_den else float("nan")
slice_rate = slice_num / slice_den if slice_den else float("nan")
print(f"[info] NÍVEL UE-tempo (R19): taxa pooled = {ue_rate:.4f} "
      f"= {ue_num:.1f}/({ue_den} obs UE-janela elegíveis; janelas sem UE "
      f"excluídas do denominador)")
print(f"[info] NÍVEL slice-tempo (rótulo do KBRL): taxa = {slice_rate:.4f} "
      f"= {slice_num}/({slice_den} pares (janela,slice) com UE; indicador "
      f"1[y_t=p5 < Λ])")
for s in range(L):
    v, d = per_slice_viol[s]
    if d:
        print(f"[info]   slice {s} (Λ={SLAS[s]} Mbps): {v}/{d} = {v / d:.4f} "
              f"(slice-tempo)")
missing_den = (ue_den == 0 and slice_den == 0)
check("C8 denominadores de violação não-vazios (elegibilidade R19)",
      not missing_den,
      "NENHUM par elegível — ver KPM/n_ues" if missing_den else
      f"UE-tempo: {ue_den} obs; slice-tempo: {slice_den} pares")

# ---------- veredito ----------
if failures:
    print(f"\nCRITÉRIOS FALHADOS: {failures}")
    sys.exit(1)
print("\nTodos os critérios de aceite passaram.")
sys.exit(0)
PYEOF
VALIDATE_RC=$?
set -e
[[ ${VALIDATE_RC} -eq 0 ]] || fail "validador dos critérios de aceite retornou ${VALIDATE_RC}"

echo ""
echo "INTEGRAÇÃO KBRL OK"
echo "Artefatos: ${OUT_DIR}"
