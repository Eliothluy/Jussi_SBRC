#!/usr/bin/env bash
# check_infra.sh — Fase 5.1 (NORI): verificação da infraestrutura E2/RIC.
#
# Imprime "INFRA OK" somente se o e2sim foi construído com sucesso (artefato
# presente) e o docker está funcional. O estado do RIC (E2Term) é apenas
# INFORMATIVO — pode ficar PENDENTE sem mudar o resultado (ver preflight.md).
#
# Artefatos verificados (em ordem; basta um):
#   MODA/third_party/e2sim/e2sim-dev_1.0.0_amd64.deb  — build via Docker
#     (estágio e2sim-builder do Dockerfile do NORI, gcc-11), extraído;
#   MODA/third_party/e2sim/e2sim/build/libe2sim.a                    — build nativo;
#   /usr/local/lib/libe2sim.a                                        — build nativo instalado.
#
# Uso: bash MODA/third_party/nori_infra/check_infra.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # .../MODA
E2SIM_DIR="$MODA_DIR/third_party/e2sim"

status="FAIL"
declare -a found=()
declare -a missing_notes=()

# --- 1. Docker ---------------------------------------------------------------
docker_ok="NO"
if command -v docker >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
    docker_ok="YES"
else
    missing_notes+=("docker indisponivel (docker ps falhou) — necessario para o caminho oficial de build (Dockerfile do NORI, gcc-11)")
fi

# --- 2. Artefato do build do e2sim ------------------------------------------
deb_artifact="$E2SIM_DIR/e2sim-dev_1.0.0_amd64.deb"
native_lib_b="$E2SIM_DIR/e2sim/build/libe2sim.a"
native_lib_i="/usr/local/lib/libe2sim.a"

if [ -f "$deb_artifact" ]; then
    found+=("e2sim-dev .deb (docker/gcc-11): $deb_artifact ($(du -h "$deb_artifact" | cut -f1))")
elif [ -f "$native_lib_b" ]; then
    found+=("libe2sim.a (build nativo): $native_lib_b")
elif [ -f "$native_lib_i" ]; then
    found+=("libe2sim.a (instalada): $native_lib_i")
else
    missing_notes+=("artefato do e2sim nao encontrado (procurei: $deb_artifact, $native_lib_b, $native_lib_i)")
fi

# --- 3. Estado do RIC (informativo; NAO afeta o resultado) -------------------
ric_state="PENDENTE"
if command -v kubectl >/dev/null 2>&1 && kubectl get pods -n ricplt >/dev/null 2>&1; then
    ric_state="K8S ricplt acessivel"
elif command -v VBoxManage >/dev/null 2>&1 && VBoxManage list vms 2>/dev/null | grep -qi "nori\|openran"; then
    ric_state="VM blueprint NORI detectada (VirtualBox)"
fi

# --- Veredito ----------------------------------------------------------------
echo "== check_infra (Fase 5.1 NORI) =="
echo "docker..............: $docker_ok"
if [ "${#found[@]}" -gt 0 ]; then
    for f in "${found[@]}"; do echo "e2sim artefato......: $f"; done
    status="OK"
else
    for m in "${missing_notes[@]}"; do echo "FALTANDO............: $m"; done
fi
echo "RIC (E2Term)........: $ric_state (informativo)"

if [ "$status" = "OK" ]; then
    echo "INFRA OK"
    exit 0
else
    echo "INFRA FAIL — ver MODA/third_party/nori_infra/preflight.md"
    exit 1
fi
