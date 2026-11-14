"""D0A audit_r19 — recompute R19 window populations from the source CSV.

Protocolo v2 §5.4. Fonte primária: g4b_validation_windows.csv (detector B,
9 seeds G4B_VALIDATION), que foi gerado dos runs brutos (g4b_validate.py).
Reconcilia: total 24.482 · completas 12.075 · não-completas 12.407 (soma
exata); percentuais por λ vs o parecido (λ2≈86%, λ5≈48%, λ6≈25% — verificar
contra o dado); decomposição por slice da ausência (qual margem faltou).
Também reconcilia o G4-A histórico (8.592 janelas → 1.041 completas).
"""
import pandas as pd

SRC = ("/home/eliothluy/Documentos/Jussi_SBRC/results/sqr_jc_final/"
       "03_calibration/g4b/g4b_validation_windows.csv")
SRC_A = ("/home/eliothluy/Documentos/Jussi_SBRC/results/sqr_jc_final/"
         "03_calibration/g4a/g4a_validation_windows.csv")
OUT = ("/home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/"
       "audit/r19_by_seed_lambda.csv")


def summarize(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    df = df.copy()
    df["seed"] = df.tag.str.extract(r"_s(\d+)$").astype(int)
    df["partial"] = df.obs_partial.astype(bool)
    # qual margem faltou (por slice) nas não-completas
    for s, c in (("e", "m_e"), ("u", "m_u"), ("m", "m_m")):
        df[f"missing_{s}"] = df[c].isna()
    rows = []
    for (lam, seed), g in df.groupby(["lam", "seed"]):
        rows.append({
            "detector_set": tag, "lam": lam, "seed": seed, "n_total": len(g),
            "n_complete": int((~g.partial).sum()),
            "n_not_complete": int(g.partial.sum()),
            "pct_not_complete": 100 * g.partial.mean(),
            "miss_e": int(g.missing_e.sum()), "miss_u": int(g.missing_u.sum()),
            "miss_m": int(g.missing_m.sum()),
            # janelas cuja ÚNICA ausência é eMBB (p/ interpretar o vacuous)
            "only_e_missing": int((g.missing_e & ~g.missing_u & ~g.missing_m).sum()),
            "only_u_missing": int((g.missing_u & ~g.missing_e & ~g.missing_m).sum()),
            "only_m_missing": int((g.missing_m & ~g.missing_e & ~g.missing_u).sum()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    det = pd.read_csv(SRC)
    b = det[det.detector == "B_1s"].copy()
    b["seed"] = b.tag.str.extract(r"_s(\d+)$").astype(int)
    for s, c in (("e", "m_e"), ("u", "m_u"), ("m", "m_m")):
        b[f"missing_{s}"] = b[c].isna()
    per = pd.concat([summarize(b, "G4B_B_1s")], ignore_index=True)
    per.to_csv(OUT, index=False)

    tot, comp = len(b), int((~b.obs_partial.astype(bool)).sum())
    print(f"[R19 G4-B] total={tot} completas={comp} não-completas={tot-comp} "
          f"({100*(tot-comp)/tot:.2f}%) — reportado: 24.482/12.075/12.407 "
          f"({12407/24482*100:.2f}%)")
    for lam in (2, 5, 6):
        g = b[b.lam == lam]
        print(f"  λ{lam}: n={len(g)} não-completas={100*g.obs_partial.astype(bool).mean():.1f}% "
              f"(parecer lê: {86 if lam==2 else 48 if lam==5 else 25}%)")
    miss_cols = ["missing_e", "missing_u", "missing_m"]
    mm = b[b.obs_partial.astype(bool)]
    print("  ausência por slice nas não-completas:",
          {c: int(mm[c].sum()) for c in miss_cols})
    # decomposição por seed (estatística por unidade independente)
    for (lam, seed), g in b.groupby(["lam", "seed"]):
        print(f"    λ{lam} s{seed}: {len(g)} janelas, "
              f"{100*g.obs_partial.astype(bool).mean():.1f}% não-completas")

    a = pd.read_csv(SRC_A)
    if "obs_partial" in a:
        ta, ca = len(a), int((~a.obs_partial.astype(bool)).sum())
        print(f"[R19 G4-A] total={ta} completas={ca} não-completas={ta-ca} "
              f"({100*(ta-ca)/ta:.2f}%) — reportado: 8.592/1.041 (12,1%)")
    print(f"[R19] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
