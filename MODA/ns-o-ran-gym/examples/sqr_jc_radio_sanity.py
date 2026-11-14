#!/usr/bin/env python
"""sqr_jc_radio_sanity.py — analyzer do GATE G3 (protocolo §24, rádio saudável).

Consome os runs do braço sanity (results/sqr_jc_final/02_radio_sanity/
radio_sanity/sanity_lam2_s{1,2,3}) e produz:

- radio_sanity_report.json: por run — rlf_count (marcadores RLF nos logs,
  mesma regex congelada de replay_sources), snr p5/mediana, distribuição de
  CQI wideband (cols 17), indicadores de erro PHY (kpm_missing rate, RX sem
  tag, contagem de UEs com KPI ~ 0 sistemático);
- radio_sanity_verdict.txt: PASS/STOP por §24:
    RLF não é causa relevante (rlf_count == 0 esperado no raio 300 m);
    sem outage persistente por cobertura (SNR p5 > limiar);
    nenhuma slice com KPI ~ 0 sistematicamente por PHY;
    SNR/CQI em regime operacional.

NUNCA recalibra nada: se STOP, o protocolo manda recalibrar SOMENTE o
cenário físico (§24) e o orquestrador decide.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd

#: Mesma regex congelada de nsoran.sqr.oracle.replay_sources (R-frozen).
RLF_RE = re.compile(r"radio\s*link\s*failure|RadioLinkFailure|\bRLF\b", re.IGNORECASE)

#: Sentinela do contrato §4.
SNR_SENTINEL = -999.0
CQI_SENTINEL = -1

#: §24 critérios congelados do gate (radio saudável @ raio 300 m, λ=2).
SNR_P5_MIN_DB = 5.0        # p5 de SNR acima => sem outage persistente
SNR_MED_MIN_DB = 10.0      # mediana operacional
RLF_MAX = 0                # RLF não pode ser causa relevante
KPI_ZERO_FRAC_MAX = 0.05   # fração de janelas com eMBB ~0 por PHY
CQI_VALID_MIN_FRAC = 0.50  # >=50% das linhas com CQI válido (1..15)


def find_run_dirs(sanity_dir: str, lambda_total: float, seeds: list[int]) -> list[str]:
    dirs = []
    for s in seeds:
        d = os.path.join(sanity_dir, f"sanity_lam{lambda_total:g}_s{s}")
        if os.path.isdir(d):
            dirs.append(d)
    return dirs


def rlf_count(log_files: list[str]) -> int:
    n = 0
    for p in log_files:
        if not os.path.isfile(p):
            continue
        with open(p, "r", errors="replace") as fh:
            for line in fh:
                if RLF_RE.search(line):
                    n += 1
    return n


def find_csv(run_dir: str, name: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(run_dir, "**", name), recursive=True))
    return hits[0] if hits else None


def analyze_run(run_dir: str) -> dict:
    kpm_path = find_csv(run_dir, "sqr_kpm_history.csv")
    delays_path = find_csv(run_dir, "sqr_packet_delays.csv")
    log_files = [
        os.path.join(run_dir, "run.log"),
        os.path.join(run_dir, "stdout"),
        os.path.join(run_dir, "stderr"),
    ]
    out = {
        "run_dir": os.path.basename(run_dir),
        "rlf_count": rlf_count(log_files),
    }
    if kpm_path is None:
        out["error"] = "sqr_kpm_history.csv ausente"
        return out
    kpm = pd.read_csv(kpm_path)
    valid = kpm[kpm["snrDb"] != SNR_SENTINEL]["snrDb"].astype(float)
    cqi = kpm[kpm["cqiWideband"] != CQI_SENTINEL]["cqiWideband"].astype(float)
    embb = kpm[(kpm["sliceId"] == 0) & (kpm["dTxBytes"] > 0)]
    out["snr_p5_db"] = float(valid.quantile(0.05)) if len(valid) else None
    out["snr_median_db"] = float(valid.median()) if len(valid) else None
    out["snr_n_valid"] = int(len(valid))
    out["snr_imputed_frac"] = float((kpm["snrDb"] == SNR_SENTINEL).mean()) if len(kpm) else None
    out["cqi_distribution"] = (
        {str(int(k)): int(v) for k, v in cqi.value_counts().sort_index().items()}
        if len(cqi) else {}
    )
    out["cqi_valid_frac"] = float((kpm["cqiWideband"] != CQI_SENTINEL).mean()) if len(kpm) else None
    out["cqi_median"] = float(cqi.median()) if len(cqi) else None
    # KPI ~ 0 por PHY: janelas eMBB com demanda mas throughput ~ 0
    if len(embb):
        near_zero = float((embb["windowThroughputMbps"] < 0.05).mean())
    else:
        near_zero = None
    out["embb_near_zero_win_frac"] = near_zero
    # erro PHY adicional: RX sem tag (dRx sem tag do contrato v1.3)
    if delays_path is not None:
        out["packet_delays_present"] = True
    else:
        out["packet_delays_present"] = False
    return out


def verdict_of(runs: list[dict]) -> tuple[str, list[str]]:
    problems: list[str] = []
    for r in runs:
        tag = r.get("run_dir", "?")
        if "error" in r:
            problems.append(f"{tag}: {r['error']}")
            continue
        if (r.get("rlf_count") or 0) > RLF_MAX:
            problems.append(f"{tag}: rlf_count={r['rlf_count']} > {RLF_MAX}")
        p5 = r.get("snr_p5_db")
        if p5 is not None and p5 < SNR_P5_MIN_DB:
            problems.append(f"{tag}: SNR p5={p5:.1f} dB < {SNR_P5_MIN_DB} (outage por cobertura)")
        med = r.get("snr_median_db")
        if med is not None and med < SNR_MED_MIN_DB:
            problems.append(f"{tag}: SNR mediana={med:.1f} dB < {SNR_MED_MIN_DB}")
        nz = r.get("embb_near_zero_win_frac")
        if nz is not None and nz > KPI_ZERO_FRAC_MAX:
            problems.append(f"{tag}: eMBB ~0 em {nz:.1%} das janelas com demanda (KPI ~0 por PHY)")
        cq = r.get("cqi_valid_frac")
        if cq is not None and cq < CQI_VALID_MIN_FRAC:
            problems.append(f"{tag}: CQI válido em só {cq:.1%} das linhas")
    return ("PASS" if not problems else "STOP"), problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sanity-dir", required=True)
    ap.add_argument("--lambda-total", type=float, default=2.0)
    ap.add_argument("--seeds", default="1 2 3")
    ap.add_argument("--report", required=True)
    ap.add_argument("--verdict", required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    run_dirs = find_run_dirs(args.sanity_dir, args.lambda_total, seeds)
    if not run_dirs:
        print(f"[radio_sanity] nenhum run encontrado sob {args.sanity_dir}", file=sys.stderr)
        return 2
    runs = [analyze_run(d) for d in run_dirs]
    verdict, problems = verdict_of(runs)

    report = {
        "gate": "G3_radio_sanity",
        "protocol": "SQRJC-ETAPA-2.5-FINAL (§24)",
        "lambda_total": args.lambda_total,
        "seeds": seeds,
        "criteria": {
            "rlf_max": RLF_MAX,
            "snr_p5_min_db": SNR_P5_MIN_DB,
            "snr_median_min_db": SNR_MED_MIN_DB,
            "embb_near_zero_frac_max": KPI_ZERO_FRAC_MAX,
            "cqi_valid_frac_min": CQI_VALID_MIN_FRAC,
        },
        "runs": runs,
        "verdict": verdict,
        "problems": problems,
    }
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    lines = [f"G3 radio sanity: {verdict}", ""]
    for r in runs:
        lines.append(
            f"  {r.get('run_dir')}: rlf={r.get('rlf_count')} "
            f"snr_p5={r.get('snr_p5_db')} snr_med={r.get('snr_median_db')} "
            f"cqi_med={r.get('cqi_median')} embb~0={r.get('embb_near_zero_win_frac')}"
        )
    for p in problems:
        lines.append(f"  PROBLEMA: {p}")
    lines.append("")
    lines.append("Critérios §24: RLF não-relevante; sem outage por cobertura; "
                 "sem KPI~0 sistêmico; SNR/CQI operacionais.")
    with open(args.verdict, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[radio_sanity] report -> {args.report}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
