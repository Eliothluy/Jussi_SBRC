#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sqr_dummy_controller.py — DUMMY CONTROLLER do smoke reforçado (Fase 1, item 7b).

Controlador Python mínimo que fala o IPC do contrato SQR (semáforos POSIX
nomeados + rslaq_actions_for_ns3.csv) e responde AÇÕES FIXAS em fases
determinísticas — exercita o caminho completo KPM → semáforo → ação →
aplicação runtime no scheduler SqrMacScheduler (cota absoluta, R27):

  fase A: Σ dedicatedPRB = 0   (cota zero ⇒ nada servido)
  fase B: Σ dedicatedPRB = 50  (Σa = 0,5·C ⇒ ~metade dos RBGs ociosa)
  fase C: Σ dedicatedPRB = 100 (partição completa)

Não usa o pacote ns-o-ran-gym nem posix_ipc: os semáforos POSIX são
acessados via ctypes/libc (mesmo mecanismo do sem_post/sem_timedwait do
sqr-sim). Roda no Python 3.13 (ns3_gate) ou 3.14 — sem argparse do ns3.

Protocolo (contrato v1.1 §1/§3):
  1. (pré-launch) cria /sem_metrics_<simId> e /sem_control_<simId>;
  2. bloqueia em sem_wait(sem_metrics) — o sqr-sim posta ao fim da janela;
  3. lê sqr-kpms.txt (valida header do contrato §4);
  4. escreve rslaq_actions_for_ns3.csv com a ação da fase corrente
     (timestamp,sliceId,dedicatedPRB,minPRB,maxPRB,algorithm; min=max=cota,
     algorithm=1=PF);
  5. sem_post(sem_control).

Uso (cwd = raiz do ns-3-dev, igual ao sqr-sim):
  python3 sqr_dummy_controller.py --simId <uuid> \
      [--phases 0,50,100] [--windows-per-phase 100] [--timeout 60]
Escreve um log de respostas: dummy_controller_log.csv (cwd), com
  timestamp_ms,phase,pct_0,pct_1,pct_2
Sai com código 0 após esgotar --max-windows respostas (default: ilimitado —
o smoke mata o processo quando o ns-3 termina).
"""

import argparse
import ctypes
import ctypes.util
import os
import sys
import time

KPM_FILE = "sqr-kpms.txt"              # contrato §4 (nome novo do cenário SQR)
ACTION_FILE = "rslaq_actions_for_ns3.csv"  # contrato §3 (nome congelado)
KPM_HEADER_COLS = [
    "timestamp", "ueImsi", "sliceId", "dTxBytes", "dRxBytes", "plr",
    "resourceSharePct", "dLostPackets", "throughputMbps", "bufferBytes",
    "windowThroughputMbps", "snrDb", "queueBytes",
]
# Divisão da soma de percentuais entre as 3 slices (floats com soma exata).
PHASE_PCTS = {
    0: (0.0, 0.0, 0.0),
    50: (16.6667, 16.6667, 16.6666),   # Σ = 50.0
    100: (33.3333, 33.3333, 33.3334),  # Σ = 100.0
}
NUM_SLICES = 3


def load_libc():
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.sem_open.restype = ctypes.c_void_p
    libc.sem_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_uint]
    libc.sem_wait.argtypes = [ctypes.c_void_p]
    libc.sem_post.argtypes = [ctypes.c_void_p]
    libc.sem_close.argtypes = [ctypes.c_void_p]
    libc.sem_unlink.argtypes = [ctypes.c_char_p]
    return libc


def main():
    ap = argparse.ArgumentParser(description="SQR dummy IPC controller (smoke Fase 1)")
    ap.add_argument("--simId", required=True)
    ap.add_argument("--phases", default="0,50,100",
                    help="somas de %% dedicadas por fase (default 0,50,100)")
    ap.add_argument("--windows-per-phase", type=int, default=100)
    ap.add_argument("--max-windows", type=int, default=0,
                    help="0 = ilimitado (o smoke encerra o processo)")
    ap.add_argument("--timeout", type=int, default=120,
                    help="watchdog: sai com erro se nenhuma métrica chegar em N s")
    ap.add_argument("--log", default="dummy_controller_log.csv")
    args = ap.parse_args()

    phases = [int(p) for p in args.phases.split(",")]
    for p in phases:
        if p not in PHASE_PCTS:
            sys.exit(f"fase {p} inválida (use {[k for k in PHASE_PCTS]})")

    sem_metrics_name = f"/sem_metrics_{args.simId}".encode()
    sem_control_name = f"/sem_control_{args.simId}".encode()
    libc = load_libc()

    # Pré-launch: remove semáforos órfãos de runs anteriores e cria os novos
    # ANTES do sqr-sim abrir (o ns-3 faz sem_open O_CREAT e aguarda ações).
    libc.sem_unlink(sem_metrics_name)
    libc.sem_unlink(sem_control_name)
    SEM_FAILED = ctypes.c_void_p(-1).value  # (sem_t*)-1
    sem_metrics = libc.sem_open(sem_metrics_name, 0o100, 0o660, 0)  # O_CREAT=0o100
    if sem_metrics is None or sem_metrics == SEM_FAILED:
        sys.exit(f"sem_open({sem_metrics_name}) falhou: errno={ctypes.get_errno()}")
    sem_control = libc.sem_open(sem_control_name, 0o100, 0o660, 0)
    if sem_control is None or sem_control == SEM_FAILED:
        sys.exit(f"sem_open({sem_control_name}) falhou: errno={ctypes.get_errno()}")

    # Watchdog: aborta se NENHUMA métrica nova chegar em args.timeout s
    # (relógio reiniciado a cada métrica — um run pode levar minutos de parede).
    import threading

    last_metric = time.time()

    def watchdog():
        while True:
            time.sleep(5)
            if time.time() - last_metric > args.timeout:
                print(f"[dummy] TIMEOUT: nenhuma métrica há {args.timeout}s — abortando",
                      file=sys.stderr, flush=True)
                os._exit(9)

    threading.Thread(target=watchdog, daemon=True).start()

    log = open(args.log, "w", buffering=1)
    log.write("timestamp_ms,phase,pct_0,pct_1,pct_2\n")

    n_windows = 0
    responses = 0
    last_ts = 0
    print(f"[dummy] ready simId={args.simId} phases={phases} wpp={args.windows_per_phase}",
          flush=True)
    while True:
        while libc.sem_wait(sem_metrics) != 0:
            if ctypes.get_errno() != 4:  # EINTR
                sys.exit(f"sem_wait falhou: errno={ctypes.get_errno()}")
        last_metric = time.time()

        # Lê e valida o KPM da janela (contrato §4)
        try:
            with open(KPM_FILE) as f:
                lines = [ln.rstrip("\n") for ln in f if ln.strip()]
        except FileNotFoundError:
            sys.exit(f"{KPM_FILE} não encontrado — o sqr-sim não escreveu o KPM do contrato")
        header = lines[0].split(",")
        if header != KPM_HEADER_COLS:
            sys.exit(f"header do KPM fora do contrato: {header}")
        kpm_rows = len(lines) - 1
        if kpm_rows > 0:
            last_ts = int(lines[1].split(",")[0])
        ts = last_ts

        # Fase corrente (determinística por nº de janelas respondidas)
        phase_idx = min(n_windows // args.windows_per_phase, len(phases) - 1)
        phase_sum = phases[phase_idx]
        pcts = PHASE_PCTS[phase_sum]

        with open(ACTION_FILE, "w", buffering=1) as af:
            for s in range(NUM_SLICES):
                af.write(f"{ts},{s},{pcts[s]},{pcts[s]},{pcts[s]},1\n")

        if libc.sem_post(sem_control) != 0:
            sys.exit(f"sem_post falhou: errno={ctypes.get_errno()}")

        log.write(f"{ts},{phase_sum},{pcts[0]},{pcts[1]},{pcts[2]}\n")
        print(f"[dummy] window={n_windows} ts={ts} phase={phase_sum}% "
              f"kpm_rows={kpm_rows} -> action written", flush=True)

        n_windows += 1
        responses += 1
        if args.max_windows and responses >= args.max_windows:
            break

    libc.sem_close(sem_metrics)
    libc.sem_close(sem_control)
    log.close()
    print(f"[dummy] done after {responses} responses", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
