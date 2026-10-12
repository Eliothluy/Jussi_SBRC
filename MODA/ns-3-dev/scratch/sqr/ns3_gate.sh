#!/usr/bin/env bash
# Gate wrapper para workflows/orquestração: executa o ./ns3 a partir da raiz do ns-3-dev,
# com Python 3.13 do uv (o wrapper ns3 é incompatível com o argparse do Python 3.14 padrão do sistema).
# Uso: ns3_gate.sh <args do ns3>   ex.: ns3_gate.sh build
set -euo pipefail
cd "$(dirname "$0")/../.."
exec /home/eliothluy/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin/python3.13 ./ns3 "$@"
