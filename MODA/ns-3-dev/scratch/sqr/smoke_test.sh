#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SMOKE TEST REFORÇADO do cenário SQR (Fase 1) — GATE OFICIAL.
# Fase 5.3: acrescenta a fase E2 (--controlPath=e2) com critérios R27 i–iv
# IDÊNTICOS aos do IPC — a origem da ação muda (semáforos/CSV → RIC Control
# do contrib/nori com loopback em-processo), os critérios não.
#
# Modos (o DEFAULT sem argumentos faz TUDO e imprime "SMOKE OK"):
#   bash scratch/sqr/smoke_test.sh            # all = git + network + ipc + e2 + soak
#   bash scratch/sqr/smoke_test.sh network [simTime] [seed]
#   bash scratch/sqr/smoke_test.sh ipc     [simTime] [seed]
#   bash scratch/sqr/smoke_test.sh e2      [simTime] [seed]
#   bash scratch/sqr/smoke_test.sh soak    [simTime] [seeds...]
#
#   (a) network-only: sqr-sim sem agente (baseline) — KPM do contrato §4
#       (sqr-kpms.txt, 13 colunas), chegadas/attach, tráfego ponta a ponta,
#       telemetria prb_* e validação de janela parcial;
#   (b) IPC com DUMMY CONTROLLER (sqr_dummy_controller.py): caminho completo
#       KPM → semáforo → ação → aplicação runtime em fases determinísticas
#       Σa = 0 / 0,5·C / C (λ de chegada elevado para garantir demanda);
#   (b') E2 com LOOPBACK (Fase 5.3, --controlPath=e2): caminho completo
#       KPM → E2SM-KPM indication (E2-interface do contrib/nori) → RIC
#       Control (loopback equal-split, equivalente ao dummy IPC) → aplicação
#       runtime nas MESMAS fases Σa = 0 / 0,5·C / C — SEM semáforos nem
#       rslaq_actions_for_ns3.csv (ausência verificada);
#   (c) validações R27 i–iv (contrato §3.3), iguais para ipc e e2:
#       prb_scheduled ≈ cota (±1 RBG/slot, util ≥ 99% com demanda);
#       Σa=0,5·C ⇒ ~metade dos RBGs ociosa e Σ sched ≤ (0,5·C+L)·slots;
#       nenhum dado em RBG ocioso; log de prb_requested/executed/scheduled/
#       idle (CSV + linhas ACTION/PRB_WINDOW);
#   (d) soak: 600 s simulados × 3 seeds sem crash (exit 0, sem NS_FATAL/ASSERT,
#       ≥1 attach, ≥1 janela com prb_scheduled > 0);
#   (e) R27-iv: git — scratch/rslaq intocado.
#
# Sempre invoca o ns-3 via ns3_gate.sh (Python 3.13; o wrapper ./ns3 quebra
# com o Python 3.14 do sistema).
# ---------------------------------------------------------------------------
set -u

NS3_ROOT="/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev"
MODA_ROOT="/home/eliothluy/Documentos/Jussi_SBRC/MODA"
SQR_DIR="${NS3_ROOT}/scratch/sqr"
NS3_GATE="${SQR_DIR}/ns3_gate.sh"
OUT_DIR="${SQR_DIR}/smoke_out"
VALIDATE="${SQR_DIR}/sqr_smoke_validate.py"
DUMMY="${SQR_DIR}/sqr_dummy_controller.py"
PYTHON="${PYTHON:-python3}"
command -v "${PYTHON}" >/dev/null 2>&1 \
    || PYTHON="/home/eliothluy/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13"
KPM_FILE="${NS3_ROOT}/sqr-kpms.txt"   # contrato §4 (cwd do ns-3)
ACTION_FILE="${NS3_ROOT}/rslaq_actions_for_ns3.csv"  # contrato §3 (cwd do ns-3)
STAGE="${1:-all}"

mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

fail() {
    echo "SMOKE FAILED: $1" >&2
    exit 1
}

[[ -x "${NS3_GATE}" ]] || fail "ns3_gate.sh não encontrado em ${NS3_GATE}"
[[ -f "${VALIDATE}" ]] || fail "validador ausente: ${VALIDATE}"
[[ -f "${DUMMY}" ]] || fail "dummy controller ausente: ${DUMMY}"

check_no_crash() { # $1 = log
    grep -qE "NS_FATAL|NS_ASSERT|Segmentation fault|std::terminate|Aborted" "$1" \
        && fail "crash/assert no log $1"
    return 0
}

# ---------------------------------------------------------------------------
# (e) R27-iv: scratch/rslaq intocado (git)
# ---------------------------------------------------------------------------
check_git_rslaq() {
    local out
    out="$(git -C "${MODA_ROOT}" status --porcelain -- ns-3-dev/scratch/rslaq 2>/dev/null || echo UNTRACKED)"
    if [[ "${out}" == "UNTRACKED" ]]; then
        echo "[smoke] WARN: não foi possível consultar o git do MODA (check R27-iv pulado)"
        return 0
    fi
    if [[ -n "${out}" ]]; then
        fail "scratch/rslaq foi MODIFICADO (regra de ouro AGENTS §5.2 / R27-iv):\n${out}"
    fi
    echo "[smoke] OK (R27-iv): scratch/rslaq intocado no git"
}

# ---------------------------------------------------------------------------
# (a) network-only
# ---------------------------------------------------------------------------
run_network() {
    local SIMTIME="${1:-120}" SEED="${2:-1}"
    local RUN_DIR="${OUT_DIR}/net_${STAMP}_s${SEED}"
    mkdir -p "${RUN_DIR}"
    local LOG="${RUN_DIR}/sqr-sim.log"

    echo "[smoke][network] rodando: simTime=${SIMTIME}s seed=${SEED}"
    rm -f "${KPM_FILE}"
    ( cd "${NS3_ROOT}" && bash "${NS3_GATE}" run sqr-sim -- \
        --simTime="${SIMTIME}" --seed="${SEED}" --run=1 --periodMs=100 \
        --outputDir="${RUN_DIR}" ) > "${LOG}" 2>&1
    local RC=$?
    echo "[smoke][network] exit code: ${RC}"
    if [[ ${RC} -ne 0 ]]; then
        tail -40 "${LOG}" >&2
        fail "network-only: ns-3 retornou ${RC} (esperado 0)"
    fi
    check_no_crash "${LOG}"

    # KPM do contrato (nome novo + 13 colunas)
    [[ -s "${KPM_FILE}" ]] || fail "arquivo de KPMs ausente/vazio: ${KPM_FILE}"
    head -n1 "${KPM_FILE}" | grep -q "windowThroughputMbps,snrDb,queueBytes" \
        || fail "header do KPM sem as colunas novas (esperado contrato §4)"
    echo "[smoke][network] OK: sqr-kpms.txt com header do contrato §4"

    for pat in UE_ARRIVAL UE_DEPARTURE UE_ATTACHED; do
        local n; n=$(grep -c "${pat}" "${LOG}" || true)
        [[ ${n} -ge 1 ]] || fail "nenhum ${pat} no log"
        echo "[smoke][network] OK: ${n}x ${pat}"
    done

    local PRB_NONZERO
    PRB_NONZERO=$(grep "PRB_WINDOW" "${LOG}" | grep -vc "prb_scheduled_rbg=\[0, 0, 0\]" || true)
    [[ ${PRB_NONZERO} -ge 1 ]] || fail "nenhuma janela com prb_scheduled > 0"
    echo "[smoke][network] OK: ${PRB_NONZERO} janelas com prb_scheduled > 0"

    # Telemetria R27-iv no stdout: PRB_WINDOW com prb_idle e dl_slots
    grep -q "PRB_WINDOW.*prb_idle_rbg=" "${LOG}" \
        || fail "PRB_WINDOW sem prb_idle_rbg (log de ociosos ausente)"
    echo "[smoke][network] OK: PRB_WINDOW loga prb_idle_rbg e dl_slots (R27-iv)"

    local RES_DIR="${RUN_DIR}/results_sqr_network_only/seed=${SEED}_run=1"
    local PRB_CSV="${RES_DIR}/sqr_prb_windows.csv"
    local KPM_HIST="${RES_DIR}/sqr_kpm_history.csv"
    [[ -s "${PRB_CSV}" ]] || fail "ausente: ${PRB_CSV}"
    [[ -s "${KPM_HIST}" ]] || fail "ausente: ${KPM_HIST}"

    ${PYTHON} "${VALIDATE}" prb "${PRB_CSV}" || fail "validação prb (network) falhou"
    ${PYTHON} "${VALIDATE}" kpm "${KPM_HIST}" "${LOG}" --period-ms 100 || fail "validação kpm (network) falhou"
    echo "[smoke][network] PASS — evidências em ${RUN_DIR}"
}

# ---------------------------------------------------------------------------
# (b)+(c) IPC com dummy controller + validações R27 i–iv
# ---------------------------------------------------------------------------
run_ipc() {
    local SIMTIME="${1:-30}" SEED="${2:-1}"
    local RUN_DIR="${OUT_DIR}/ipc_${STAMP}_s${SEED}"
    mkdir -p "${RUN_DIR}"
    local LOG="${RUN_DIR}/sqr-sim.log"
    local UUID="smoke${STAMP}_s${SEED}"

    # λ=30 UEs/min (TESTE): garante UEs/demanda em todas as fases dentro de
    # 30 s — não é o cenário do artigo (o soak usa os defaults fiéis).
    local LAMBDA=30

    echo "[smoke][ipc] rodando: simTime=${SIMTIME}s seed=${SEED} lambda=${LAMBDA}/min fases Σa=0/50%/100%"
    rm -f "${KPM_FILE}" "${ACTION_FILE}"

    ( cd "${NS3_ROOT}" \
      && ${PYTHON} "${DUMMY}" --simId "${UUID}" --phases 0,50,100 \
             --windows-per-phase 100 --timeout 120 \
             --log "${RUN_DIR}/dummy_controller_log.csv" \
      ) > "${RUN_DIR}/dummy.log" 2>&1 &
    local DUMMY_PID=$!

    # Espera o dummy criar os semáforos (pré-launch)
    for _ in $(seq 1 50); do
        grep -q "\[dummy\] ready" "${RUN_DIR}/dummy.log" 2>/dev/null && break
        sleep 0.2
    done
    grep -q "\[dummy\] ready" "${RUN_DIR}/dummy.log" || fail "dummy controller não iniciou"

    ( cd "${NS3_ROOT}" && bash "${NS3_GATE}" run sqr-sim -- \
        --simTime="${SIMTIME}" --seed="${SEED}" --run=1 --periodMs=100 \
        --lambdaUesPerMin=${LAMBDA} \
        --outputDir="${RUN_DIR}" --simId="${UUID}" ) > "${LOG}" 2>&1
    local RC=$?
    echo "[smoke][ipc] exit code: ${RC}"
    kill "${DUMMY_PID}" 2>/dev/null
    wait "${DUMMY_PID}" 2>/dev/null

    if [[ ${RC} -ne 0 ]]; then
        tail -40 "${LOG}" >&2
        tail -20 "${RUN_DIR}/dummy.log" >&2
        fail "ipc: ns-3 retornou ${RC} (esperado 0)"
    fi
    check_no_crash "${LOG}"

    # Caminho IPC completo: dummy respondeu, ação escrita, ACTION aplicado
    local RESP; RESP=$(grep -c "action written" "${RUN_DIR}/dummy.log" || true)
    [[ ${RESP} -ge 50 ]] || fail "dummy respondeu só ${RESP} janelas (<50) — IPC degenerado"
    echo "[smoke][ipc] OK: dummy respondeu ${RESP} janelas (KPM→semáforo→ação)"

    grep -q "prb_requested_pct" "${LOG}" || fail "nenhuma linha ACTION no log (ação não aplicada)"
    grep -q "zero_quota_nothing_served" "${LOG}" \
        || fail "nenhuma ACTION com zero_quota_nothing_served (fase Σa=0 não aplicada — R27 exige aplicar, não manter config anterior)"
    grep -qE "PRB_WINDOW.*prb_idle_rbg=" "${LOG}" || fail "PRB_WINDOW sem prb_idle_rbg"
    echo "[smoke][ipc] OK: ACTION com prb_requested/executed, zero_quota e PRB_WINDOW com prb_idle (R27-iv)"

    local RES_DIR="${RUN_DIR}/results_sqr_network_only/seed=${SEED}_run=1"
    local PRB_CSV="${RES_DIR}/sqr_prb_windows.csv"
    local KPM_HIST="${RES_DIR}/sqr_kpm_history.csv"
    [[ -s "${PRB_CSV}" ]] || fail "ausente: ${PRB_CSV}"
    [[ -s "${KPM_HIST}" ]] || fail "ausente: ${KPM_HIST}"

    ${PYTHON} "${VALIDATE}" prb "${PRB_CSV}" --ipc || fail "validações R27 (i-iii) falharam"
    ${PYTHON} "${VALIDATE}" kpm "${KPM_HIST}" "${LOG}" --period-ms 100 || fail "validação kpm (ipc) falhou"
    echo "[smoke][ipc] PASS — evidências em ${RUN_DIR}"
}

# ---------------------------------------------------------------------------
# (b')+(c) E2 (--controlPath=e2, Fase 5.3): loopback equal-split via RIC
# Control do contrib/nori + critérios R27 i–iv IDÊNTICOS aos do IPC
# ---------------------------------------------------------------------------
run_e2() {
    local SIMTIME="${1:-30}" SEED="${2:-1}"
    local RUN_DIR="${OUT_DIR}/e2_${STAMP}_s${SEED}"
    mkdir -p "${RUN_DIR}"
    local LOG="${RUN_DIR}/sqr-sim.log"

    # λ=30 UEs/min (TESTE): idêntico à fase IPC — garante UEs/demanda em
    # todas as fases dentro de 30 s; NÃO é o cenário do artigo.
    local LAMBDA=30

    echo "[smoke][e2] rodando: simTime=${SIMTIME}s seed=${SEED} lambda=${LAMBDA}/min fases Σa=0/50%/100%"
    rm -f "${KPM_FILE}" "${ACTION_FILE}"

    ( cd "${NS3_ROOT}" && bash "${NS3_GATE}" run sqr-sim -- \
        --simTime="${SIMTIME}" --seed="${SEED}" --run=1 --periodMs=100 \
        --lambdaUesPerMin=${LAMBDA} \
        --controlPath=e2 --e2Phases=0,50,100 --e2WindowsPerPhase=100 \
        --outputDir="${RUN_DIR}" ) > "${LOG}" 2>&1
    local RC=$?
    echo "[smoke][e2] exit code: ${RC}"
    if [[ ${RC} -ne 0 ]]; then
        tail -40 "${LOG}" >&2
        fail "e2: ns-3 retornou ${RC} (esperado 0)"
    fi
    check_no_crash "${LOG}"

    # SEM semáforos/CSV no caminho E2 (contraste explícito com o IPC)
    grep -q "IPC semaphores created" "${LOG}" \
        && fail "modo e2 criou semáforos IPC — caminho E2 degenerado"
    [[ ! -e "${ACTION_FILE}" ]] || fail "modo e2 criou ${ACTION_FILE} (deveria ser SEM CSV de ação)"
    echo "[smoke][e2] OK: sem semáforos IPC e sem ${ACTION_FILE##*/} (caminho E2 limpo)"

    # Superfície E2 do contrib/nori instalada + loopback em processo
    grep -q "SqrE2Adapter. installed on gNB" "${LOG}" \
        || fail "adaptador E2 não instalado (E2Interface/E2Termination ausentes)"
    grep -q "SqrE2Loopback. E2-SETUP" "${LOG}" \
        || fail "loopback E2 sem E2-SETUP (E2Term simulado não iniciou)"
    echo "[smoke][e2] OK: E2Interface+E2Termination do contrib/nori + loopback E2-SETUP (DEGRADED)"

    # Caminho E2 completo: indication → RIC Control → ACTION aplicada
    local CTL; CTL=$(grep -c "RIC control sent" "${LOG}" || true)
    [[ ${CTL} -ge 50 ]] || fail "loopback emitiu só ${CTL} RIC Controls (<50) — caminho E2 degenerado"
    local IND; IND=$(grep -oE "indications_sent=[0-9]+" "${LOG}" | tail -1 | cut -d= -f2)
    [[ "${IND}" -ge 50 ]] || fail "só ${IND} E2SM-KPM indications enviadas (<50)"
    grep -q "prb_requested_pct" "${LOG}" || fail "nenhuma linha ACTION no log (RIC Control não aplicado)"
    grep -q "zero_quota_nothing_served" "${LOG}" \
        || fail "nenhuma ACTION com zero_quota_nothing_served (fase Σa=0 não aplicada — R27 exige aplicar, não manter config anterior)"
    grep -qE "PRB_WINDOW.*prb_idle_rbg=" "${LOG}" || fail "PRB_WINDOW sem prb_idle_rbg"
    echo "[smoke][e2] OK: ${IND} indications, ${CTL} RIC Controls, ACTION com prb_requested/executed, zero_quota e PRB_WINDOW com prb_idle (R27-iv)"

    # Fases do loopback = as mesmas do dummy IPC (0/50/100 com ~100 janelas)
    local LOOP_CSV="${RUN_DIR}/results_sqr_network_only/seed=${SEED}_run=1/sqr_e2_ric_loopback.csv"
    [[ -s "${LOOP_CSV}" ]] || fail "ausente: ${LOOP_CSV}"
    for ph in 0 50 100; do
        local n; n=$(awk -F, "NR>1 && \$2==${ph}" "${LOOP_CSV}" | wc -l)
        [[ ${n} -ge 90 ]] || fail "loopback: só ${n} janelas na fase ${ph}% (esperado ~100)"
    done
    echo "[smoke][e2] OK: loopback percorreu as 3 fases (0/50/100, ≥90 janelas cada) — equivalente ao dummy IPC"

    local RES_DIR="${RUN_DIR}/results_sqr_network_only/seed=${SEED}_run=1"
    local PRB_CSV="${RES_DIR}/sqr_prb_windows.csv"
    local KPM_HIST="${RES_DIR}/sqr_kpm_history.csv"
    [[ -s "${PRB_CSV}" ]] || fail "ausente: ${PRB_CSV}"
    [[ -s "${KPM_HIST}" ]] || fail "ausente: ${KPM_HIST}"

    # Mesmos critérios R27 i–iv do IPC (--phased = mesma validação por fases)
    ${PYTHON} "${VALIDATE}" prb "${PRB_CSV}" --phased || fail "validações R27 (i-iii) do caminho e2 falharam"
    ${PYTHON} "${VALIDATE}" kpm "${KPM_HIST}" "${LOG}" --period-ms 100 || fail "validação kpm (e2) falhou"
    echo "[smoke][e2] PASS — evidências em ${RUN_DIR}"
}

# ---------------------------------------------------------------------------
# (d) soak: 600 s simulados × seeds, sem crash
# ---------------------------------------------------------------------------
run_soak() {
    local SIMTIME="${1:-600}"
    shift || true
    local SEEDS=("$@")
    [[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(1 2 3)
    for SEED in "${SEEDS[@]}"; do
        local RUN_DIR="${OUT_DIR}/soak_${STAMP}_s${SEED}"
        mkdir -p "${RUN_DIR}"
        local LOG="${RUN_DIR}/sqr-sim.log"
        echo "[smoke][soak] rodando: simTime=${SIMTIME}s seed=${SEED}"
        rm -f "${KPM_FILE}"
        ( cd "${NS3_ROOT}" && bash "${NS3_GATE}" run sqr-sim -- \
            --simTime="${SIMTIME}" --seed="${SEED}" --run=1 --periodMs=100 \
            --outputDir="${RUN_DIR}" ) > "${LOG}" 2>&1
        local RC=$?
        echo "[smoke][soak] seed=${SEED} exit code: ${RC}"
        if [[ ${RC} -ne 0 ]]; then
            tail -40 "${LOG}" >&2
            fail "soak seed=${SEED}: ns-3 retornou ${RC}"
        fi
        check_no_crash "${LOG}"
        local ATT; ATT=$(grep -c "UE_ATTACHED" "${LOG}" || true)
        [[ ${ATT} -ge 1 ]] || fail "soak seed=${SEED}: nenhum attach"
        local PRB_NONZERO
        PRB_NONZERO=$(grep "PRB_WINDOW" "${LOG}" | grep -vc "prb_scheduled_rbg=\[0, 0, 0\]" || true)
        [[ ${PRB_NONZERO} -ge 1 ]] || fail "soak seed=${SEED}: nenhuma janela com prb_scheduled > 0"
        local RES="${RUN_DIR}/results_sqr_network_only/seed=${SEED}_run=1/sqr_prb_windows.csv"
        [[ -s "${RES}" ]] || fail "soak seed=${SEED}: ausente ${RES}"
        echo "[smoke][soak] OK: seed=${SEED} sem crash, ${ATT} attach(s), ${PRB_NONZERO} janelas com agendamento"
    done
    echo "[smoke][soak] PASS: ${SIMTIME}s × ${#SEEDS[@]} seeds sem crash"
}

# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
case "${STAGE}" in
    network) check_git_rslaq; run_network "${2:-120}" "${3:-1}" ;;
    ipc)     check_git_rslaq; run_ipc "${2:-30}" "${3:-1}" ;;
    e2)      check_git_rslaq; run_e2 "${2:-30}" "${3:-1}" ;;
    soak)    check_git_rslaq; run_soak "${2:-600}" "${3:-1}" "${4:-2}" "${5:-3}" ;;
    all)     check_git_rslaq; run_network 120 1; run_ipc 30 1; run_e2 30 1; run_soak 600 1 2 3 ;;
    *) fail "modo inválido: ${STAGE} (use all|network|ipc|e2|soak)" ;;
esac

echo "SMOKE OK"
exit 0
