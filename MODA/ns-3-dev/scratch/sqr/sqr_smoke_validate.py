#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sqr_smoke_validate.py — validações do smoke reforçado do cenário SQR (Fase 1).

Critérios R27 (contrato_interface.md §3.3 + register R27, RATIFICADO 12/10/2026):
  (i)   prb_scheduled ≈ cota por slice (utilização ≥ 99% quando há demanda;
        nunca acima da cota);
  (ii)  teste determinístico com ação Σa = 0,5·C ⇒ Σ prb_scheduled ≤
        (0,5·total_rbg + L)·dl_slots e ~metade dos RBGs ociosa;
  (iii) nenhum dado servido em RBG ocioso (Σ sched ≤ Σ quota + folga de
        arredondamento, e prb_idle consistente em todas as linhas; fase Σa=0
        ⇒ nada servido);
  (iv)  log de prb_requested/prb_executed/prb_scheduled/prb_idle presente
        (header do CSV + linhas ACTION/PRB_WINDOW conferidas no smoke_test.sh).

Fase 5.3: os mesmos critérios R27 i–iii por fases servem ao caminho
--controlPath=e2 (flag --phased, alias de --ipc — o alinhamento de janela e as
fases do dummy são idênticos; a origem da ação muda de CSV/semáforo para RIC
Control).

Janela parcial (contrato §4): a 1ª linha de cada UE aparece na janela seguinte
à entrada (ts ∈ (trafficStart, trafficStart+period]) e o windowThroughputMbps
divide o delta pelo tempo ATIVO (confere implied_active == ts1 − trafficStart).

Saída: prints "OK: ..." por verificação; exit 1 na primeira falha.
"""

import argparse
import re
import sys

L_SLICES = 3  # nº de slices (folga de arredondamento do critério (ii))


def fail(msg):
    print(f"VALIDATE FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def read_csv(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        fail(f"{path} sem linhas de dados")
    header = lines[0].split(",")
    rows = [dict(zip(header, ln.split(","))) for ln in lines[1:]]
    return header, rows


def validate_prb(path, ipc):
    header, rows = read_csv(path)

    # --- R27-iv: colunas de auditoria presentes ---
    need = (["dl_slots", "prb_idle_rbg"]
            + [f"prb_requested_pct_{s}" for s in range(L_SLICES)]
            + [f"prb_executed_rbg_{s}" for s in range(L_SLICES)]
            + [f"prb_scheduled_rbg_{s}" for s in range(L_SLICES)]
            + [f"prb_budget_rbg_{s}" for s in range(L_SLICES)])
    missing = [c for c in need if c not in header]
    if missing:
        fail(f"header de {path} sem colunas de auditoria R27: {missing}")
    print(f"OK: header do CSV de janelas tem prb_requested/executed/scheduled/"
          f"budget/idle + dl_slots (R27-iv)")

    phases = {"PRE": [], "A": [], "B": [], "C": []}
    n_rows = 0
    for r in rows:
        n_rows += 1
        total = int(r["total_rbg"])
        slots = int(r["dl_slots"])
        sched = [int(r[f"prb_scheduled_rbg_{s}"]) for s in range(L_SLICES)]
        budget = [int(r[f"prb_budget_rbg_{s}"]) for s in range(L_SLICES)]
        executed = [int(r[f"prb_executed_rbg_{s}"]) for s in range(L_SLICES)]
        idle = int(r["prb_idle_rbg"])
        req = [float(r[f"prb_requested_pct_{s}"]) for s in range(L_SLICES)]
        ssum = sum(sched)

        # Consistência da coluna ociosa (definição §3.5 agregada na janela)
        expect_idle = total * slots - ssum
        if idle != expect_idle:
            fail(f"linha t={r['timestamp_ms']}: prb_idle={idle} != "
                 f"total_rbg*dl_slots-Σsched={expect_idle}")

        # (iii) agregado: nada é servido além das cotas (+ folga L RBG/slot)
        if ssum > sum(budget) + L_SLICES * slots:
            fail(f"linha t={r['timestamp_ms']}: Σsched={ssum} excede Σquota="
                 f"{sum(budget)} + folga {L_SLICES * slots} — dado em RBG ocioso?")

        # Alinhamento de janela: prb_requested/prb_executed da linha descrevem
        # a ação VIGENTE na janela drenada (aplicada na borda anterior).
        # Sentinela -1 = janela sem ação do agente (não é fase Σa=0!).
        psum = sum(req)
        if ipc:
            if min(req) < 0.0:
                phases["PRE"].append(r)
            elif psum <= 1e-6:
                phases["A"].append(r)
            elif abs(psum - 50.0) <= 0.5:
                phases["B"].append(r)
            elif abs(psum - 100.0) <= 0.5:
                phases["C"].append(r)

    print(f"OK: {n_rows} linhas de janelas consistentes (prb_idle e teto de cota)")

    if not ipc:
        return

    # --- Alinhamento (janela × ação no mesmo passo) ---
    # A 1ª janela dos modos com fases (IPC e E2 — Fase 5.3) necessariamente
    # correu SEM ação do agente: a ação só chega na borda que a inicia e entra
    # no CSV no dreno seguinte. Se nenhuma linha tem a sentinela -1, o log da
    # ação voltou a ser aplicado ANTES do dreno (regressão do alinhamento —
    # ação de uma janela rotulando os contadores da anterior).
    if len(phases["PRE"]) < 1:
        fail("modo com fases (IPC/E2): nenhuma janela com sentinela "
             "prb_requested_pct=-1 — o log da ação não está defasado para a "
             "janela que governa (alinhamento)")
    print(f"OK: alinhamento de janela: {len(phases['PRE'])} janela(s) inicial(is) com "
          f"sentinela -1 (ação recebida na borda t só aparece no CSV do dreno da "
          f"janela que governa)")

    # --- Fase A (Σa=0): nada servido, tudo ocioso ---
    if len(phases["A"]) < 5:
        fail(f"modo com fases (IPC/E2): só {len(phases['A'])} janelas na fase A "
             f"(Σa=0) — fases do dummy insuficientes")
    for r in phases["A"]:
        slots = int(r["dl_slots"])
        total = int(r["total_rbg"])
        ssum = sum(int(r[f"prb_scheduled_rbg_{s}"]) for s in range(L_SLICES))
        idle = int(r["prb_idle_rbg"])
        if ssum != 0:
            fail(f"fase A (Σa=0) t={r['timestamp_ms']}: Σsched={ssum} != 0 — "
                 "nada deveria ser servido com cota zero")
        if idle != total * slots:
            fail(f"fase A t={r['timestamp_ms']}: prb_idle={idle} != {total * slots}")
    print(f"OK: fase A (Σa=0): {len(phases['A'])} janelas com Σ prb_scheduled == 0 "
          f"e 100% dos RBGs ociosos (R27: nada servido, config anterior NÃO mantida)")

    # --- Fase B (Σa=0,5·C): ~metade ociosa, teto do critério (ii) ---
    if len(phases["B"]) < 5:
        fail(f"modo com fases (IPC/E2): só {len(phases['B'])} janelas na fase B "
             f"(Σa=0,5·C)")
    idle_fracs = []
    for r in phases["B"]:
        slots = int(r["dl_slots"])
        total = int(r["total_rbg"])
        ssum = sum(int(r[f"prb_scheduled_rbg_{s}"]) for s in range(L_SLICES))
        idle = int(r["prb_idle_rbg"])
        cap = (0.5 * total + L_SLICES) * slots
        if slots > 0 and ssum > cap:
            fail(f"fase B t={r['timestamp_ms']}: Σsched={ssum} > "
                 f"(0,5·total_rbg+L)·dl_slots = {cap:.0f} (critério ii)")
        if slots >= 10:
            idle_fracs.append(idle / (total * slots))
    if not idle_fracs:
        fail("fase B sem janelas com dl_slots >= 10 (nenhuma demanda?)")
    idle_fracs.sort()
    med = idle_fracs[len(idle_fracs) // 2]
    lo, hi = idle_fracs[0], idle_fracs[-1]
    if not (0.30 <= lo and med <= 0.70):
        fail(f"fase B: fração ociosa min={lo:.3f} mediana={med:.3f} fora de ~[0.3, 0.7] — "
             "cota absoluta Σa=0,5·C não virou ~metade dos RBGs ociosa")
    print(f"OK: fase B (Σa=0,5·C): {len(phases['B'])} janelas, fração ociosa "
          f"min={lo:.3f} mediana={med:.3f} max={hi:.3f} (~metade dos RBGs idle) e "
          f"Σsched ≤ (0,5·C+L)·slots em todas (critério ii)")

    # --- Fase C (partição completa): cota servida até o teto ---
    if len(phases["C"]) < 5:
        fail(f"modo com fases (IPC/E2): só {len(phases['C'])} janelas na fase C "
             f"(Σa=C)")
    max_util = [0.0] * L_SLICES
    for r in phases["C"]:
        slots = int(r["dl_slots"])
        for s in range(L_SLICES):
            executed = int(r[f"prb_executed_rbg_{s}"])
            sched = int(r[f"prb_scheduled_rbg_{s}"])
            budget = int(r[f"prb_budget_rbg_{s}"])
            if executed <= 0:
                fail(f"fase C t={r['timestamp_ms']}: cota executada da slice {s} é 0")
            if sched > budget + slots:
                fail(f"fase C t={r['timestamp_ms']}: slice {s} sched={sched} > "
                     f"quota*slots={budget}+{slots} — cota ultrapassada")
            if budget > 0:
                max_util[s] = max(max_util[s], sched / budget)
    for s in range(L_SLICES):
        if max_util[s] < 0.99:
            fail(f"fase C: utilização máxima da slice {s} = {max_util[s]:.3f} < 0.99 — "
                 "prb_scheduled não atingiu a cota (critério i)")
    print(f"OK: fase C (Σa=C): {len(phases['C'])} janelas, prb_scheduled ≈ cota "
          f"(util máx/slice = {[f'{u:.4f}' for u in max_util]}, critério i) e nenhuma "
          f"slice acima da cota (critério iii)")


def validate_kpm(path, log_path, period_ms, attach_grace):
    header, rows = read_csv(path)
    want = ["timestamp", "ueImsi", "sliceId", "dTxBytes", "dRxBytes", "plr",
            "resourceSharePct", "dLostPackets", "throughputMbps", "bufferBytes",
            "windowThroughputMbps", "snrDb", "queueBytes"]
    if header != want:
        fail(f"header do KPM fora do contrato §4: {header}")
    print("OK: header do KPM com as 13 colunas do contrato §4")

    # Sentinela de SNR (R09/contrato §4): -999 na ausência; -100 é a sentinela
    # ANTIGA (regressão) e é proibida.
    for r in rows:
        v = float(r["snrDb"])
        if v == -100.0:
            fail(f"linha t={r['timestamp']}: snrDb=-100 (sentinela antiga — deveria ser -999)")
        if not (v == -999.0 or v > -150.0):
            fail(f"linha t={r['timestamp']}: snrDb={v} nem é sentinela -999 nem SNR plausível")
    n_sentinel = sum(1 for r in rows if float(r["snrDb"]) == -999.0)
    print(f"OK: sentinela de SNR: {n_sentinel}/{len(rows)} linhas com -999, "
          f"nenhuma com a sentinela antiga -100")

    thr_rows = [r for r in rows if int(r["dRxBytes"]) > 0]
    if not thr_rows:
        fail("nenhuma linha de KPM com dRxBytes > 0 — nenhum byte entregue")
    n_thr_pos = sum(1 for r in rows if float(r["windowThroughputMbps"]) > 0)
    print(f"OK: {n_thr_pos}/{len(rows)} linhas de KPM com windowThroughputMbps > 0")

    # --- Janela parcial (contrato §4) ---
    arrivals = {}
    pat = re.compile(r"UE_ARRIVAL ueId=(\d+).* t=([0-9.]+)s")
    with open(log_path) as f:
        for ln in f:
            m = pat.search(ln)
            if m:
                arrivals[int(m.group(1))] = float(m.group(2))
    if not arrivals:
        fail("nenhuma linha UE_ARRIVAL no log")
    period_s = period_ms / 1000.0
    checked_first = 0
    for r in rows:
        ue = int(r["ueImsi"])
        if ue not in arrivals:
            continue
        ts = int(r["timestamp"]) / 1000.0
        ts_start = arrivals[ue] + attach_grace
        is_first = ts - period_s <= ts_start  # única janela possível de 1ª linha
        if is_first:
            if not (ts > ts_start + 1e-9 and ts <= ts_start + period_s + 1e-3):
                fail(f"UE {ue}: 1ª linha de KPM em t={ts}s, mas entrada+grace="
                     f"{ts_start}s — deveria aparecer na janela seguinte "
                     f"({ts_start}, {ts_start + period_s}]")
            d_rx = int(r["dRxBytes"])
            wthr = float(r["windowThroughputMbps"])
            if d_rx >= 2000 and wthr > 0:
                implied_active = d_rx * 8.0 / (wthr * 1e6)
                expected = ts - ts_start  # activeEnd(=ts, UE vivo) − entrada
                if not (0 < implied_active <= period_s + 1e-3):
                    fail(f"UE {ue}: tempo ativo implícito {implied_active:.4f}s "
                         f"fora de (0, {period_s}] — windowThroughputMbps dividido "
                         "pelo período cheio (bug da janela parcial)?")
                if abs(implied_active - expected) > 0.05:
                    fail(f"UE {ue}: tempo ativo implícito {implied_active:.4f}s != "
                         f"desde a entrada {expected:.4f}s (±0,05) — deltas não "
                         "começam na entrada do UE")
                checked_first += 1
    if checked_first == 0:
        print("WARN: nenhuma 1ª janela de UE com dRxBytes>=2000 para validar "
              "janela parcial (tráfego curto?)")
    else:
        print(f"OK: janela parcial: {checked_first} primeiras janelas de UE "
              f"aparecem na janela seguinte à entrada com tempo ativo desde a "
              f"entrada (contrato §4)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("prb")
    p1.add_argument("csv")
    p1.add_argument("--ipc", action="store_true",
                    help="fases determinísticas do dummy (caminho IPC, Fase 1)")
    p1.add_argument("--phased", action="store_true",
                    help="idêntico a --ipc: critérios R27 i–iii por fases "
                         "(usado pela fase E2 do smoke, Fase 5.3)")
    p2 = sub.add_parser("kpm")
    p2.add_argument("csv")
    p2.add_argument("log")
    p2.add_argument("--period-ms", type=int, default=100)
    p2.add_argument("--attach-grace", type=float, default=0.5)
    args = ap.parse_args()
    if args.mode == "prb":
        validate_prb(args.csv, args.ipc or args.phased)
    else:
        validate_kpm(args.csv, args.log, args.period_ms, args.attach_grace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
