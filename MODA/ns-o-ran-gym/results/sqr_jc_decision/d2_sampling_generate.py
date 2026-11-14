#!/usr/bin/env python3
"""d2_sampling_generate.py — amostragem FINAL da D2 (Emenda 1 §2-3 + Emenda 2).

Determinístico: nenhuma entrada além dos decisions.jsonl da D1-v3 canônica
e da seed de amostragem 20261108. Produz:

  D2_sampling_FINAL.csv   — 30 estados × 12 alternativas (+ a_B0 por estado),
                            cada linha com o MECANISMO de geração (ajuste 4)
                            e a marca de substituição p/ duplicatas (ajuste 3).
  D2_sampling_manifest.json — hashes (registrados ANTES de qualquer replay).

Regras (congeladas): estratos S1-S4 sobre decisões do B0 (frozen p/ viol);
alocação S1×12 S2×9 S3×6 S4×3; prioridade all3-completa em S1/S3; rodízio de
seeds; distância >=5 janelas dentro de (run, estrato); 12 alternativas
DISTINTAS/admissíveis com substituição determinística de duplicatas.
"""
import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D1 = os.path.join(HERE, os.environ.get("D1_DIR", "d1_runs"))
_SUF = os.environ.get("D2_SUFFIX", "")
OUT_CSV = os.path.join(HERE, f"D2_sampling_FINAL{_SUF}.csv")
OUT_MANIFEST = os.path.join(HERE, f"D2_sampling_manifest{_SUF}.json")
SAMPLING_SEED = 20261108
C = 53
EQUAL = (18, 17, 18)


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def rng_of(state_key: str) -> np.random.Generator:
    h = int(hashlib.sha1(state_key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


def load_universe() -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(f"{D1}/het_lam*_s*/decisions.jsonl")):
        tag = os.path.basename(os.path.dirname(p))
        lam = int(tag.split("_lam")[1].split("_")[0])
        seed = int(tag.split("_s")[1])
        for line in open(p):
            r = json.loads(line)
            if r.get("policy") != "SQR_HET":
                continue
            if not isinstance(r.get("sla_outcome_margins"), list):
                continue
            r["tag"], r["lam"], r["seed"] = tag, lam, seed
            rows.append(r)
    df = pd.DataFrame(rows)
    viol = ~df["joint_frozen_ok"].astype(bool)
    proj = df["projection_triggered"].astype(bool)
    df["S"] = np.where(proj & viol, "S1", np.where(~proj & viol, "S2",
                       np.where(proj, "S3", "S4")))
    df["vslice"] = df["sla_outcome_margins"].apply(
        lambda v: "+".join(n for n, m in zip("eum", v)
                           if m is not None and m < 0) or "none")
    df["n_complete"] = df["sla_evidence"].apply(
        lambda v: sum(1 for e in v if e == "COMPLETE")
        if isinstance(v, list) else 0)
    return df


def select_states(df: pd.DataFrame) -> pd.DataFrame:
    master = np.random.default_rng(SAMPLING_SEED)
    alloc = [("S1", 12, {"lam": 4}), ("S2", 9, {"lam": 3}),
             ("S3", 6, {"lam": 2}), ("S4", 3, {"lam": 1})]
    picks = []
    seed_cycle = {2101: 0, 2102: 1, 2103: 2}
    for S, total, per_lam in alloc:
        per = total // 3
        for lam in (2, 5, 6):
            cell = df[(df.S == S) & (df.lam == lam)]
            if S in ("S1", "S3"):          # prioridade all3-completa
                cell = pd.concat([cell[cell.n_complete == 3],
                                  cell[cell.n_complete < 3]])
            if S == "S2":                  # sub-estratos: 2 e-only + 1 e+m
                sub = ([cell[cell.vslice == "e"],
                        cell[cell.vslice == "e"],
                        cell[cell.vslice == "e+m"]])
            else:
                sub = [cell] * per
            for k, pool in enumerate(sub):
                if pool is None or not len(pool):
                    continue
                # ordenação determinística por hash + rodízio de seed
                want_seeds = [2101, 2102, 2103]
                ws = want_seeds[(len(picks)) % 3]
                pool = pool[pool.seed == ws] if len(pool[pool.seed == ws]) else pool
                hkeys = [hashlib.sha1(
                    f"{r.tag}|{r.decision_index}|{SAMPLING_SEED}"
                    .encode()).hexdigest() for r in pool.itertuples()]
                pool = pool.iloc[np.argsort(hkeys)]
                chosen = None
                used_wins = [p["win"] for p in picks if p["tag"] == pool.iloc[0]["tag"]]
                for _, r in pool.iterrows():
                    if all(abs(int(r.decision_index) - w) >= 5 for w in used_wins):
                        chosen = r
                        break
                if chosen is None:
                    chosen = pool.iloc[0]
                picks.append({"stratum": S, "lam": lam, "sub": k,
                              "tag": chosen["tag"], "seed": int(chosen["seed"]),
                              "win": int(chosen["decision_index"]),
                              "a_B0": tuple(int(v) for v in chosen["action"]),
                              "n_ues": list(chosen["slice_n_ues"]),
                              "backlog": list(chosen["slice_backlog"]),
                              "vslice": chosen["vslice"],
                              "n_complete": int(chosen["n_complete"])})
    return pd.DataFrame(picks)


def alternatives(state: pd.Series) -> list[dict]:
    a = list(state["a_B0"])
    score = [state["n_ues"][i] + (1 if state["backlog"][i] > 0 else 0)
             for i in range(3)]
    pairs = sorted([(0, 1), (0, 2), (1, 2)],
                   key=lambda p: -abs(score[p[0]] - score[p[1]]))
    P = {f"P{i+1}": p for i, p in enumerate(pairs)}

    def transfer(base, pair, delta, to_high):
        i, j = pair
        hi, lo = (i, j) if score[i] >= score[j] else (j, i)
        src, dst = (lo, hi) if to_high else (hi, lo)
        out = list(base)
        move = min(delta, out[src])
        out[src] -= move
        out[dst] += move
        return tuple(out)

    key = f"{state['tag']}|{state['win']}"
    rng = rng_of(key)
    cand = []
    for delta in (2, 4, 8, 16):                      # 4 redistribuições
        p = P["P1"] if delta in (2, 16) else P["P2" if delta == 4 else "P3"]
        cand.append((f"paired:{p}:{delta}:{'hi' if delta != 16 else 'lo'}",
                     transfer(a, p, delta, delta != 16)))
    cand.append(("nbh:P1:+1:hi", transfer(a, P["P1"], 1, True)))   # 3 vizinhanças
    cand.append(("nbh:P2:+1:hi", transfer(a, P["P2"], 1, True)))
    cand.append(("nbh:P3:-1:lo", transfer(a, P["P3"], 1, False)))
    for k in range(4):                               # 4 aleatórias do simplex
        while True:
            v = tuple(int(x) for x in rng.integers(0, C + 1, size=3))
            if sum(v) <= C:
                break
        cand.append((f"rnd:{k+1}:sha1({key})", v))
    cand.append(("ref:equal", EQUAL))
    # dedupe + substituição determinística (ajuste 3)
    surplus = []
    for delta in (1, 32):
        surplus.append((f"sub-paired:P1:{delta}:hi", transfer(a, P["P1"], delta, True)))
        surplus.append((f"sub-paired:P2:{delta}:hi", transfer(a, P["P2"], delta, True)))
    surplus.append(("sub-nbh:P1:+2:hi", transfer(a, P["P1"], 2, True)))
    surplus.append(("sub-nbh:P2:-2:lo", transfer(a, P["P2"], 2, False)))
    ext = rng_of(key + "|surplus")
    for k in range(12):
        while True:
            v = tuple(int(x) for x in ext.integers(0, C + 1, size=3))
            if sum(v) <= C:
                break
        surplus.append((f"sub-rnd:{k+1}", v))
    accepted, out, si = {tuple(a)}, [], 0
    for mech, v in cand:
        if len(out) == 12:
            break
        if v in accepted or sum(v) > C or min(v) < 0:
            while si < len(surplus):
                mech2, v2 = surplus[si]
                si += 1
                if v2 not in accepted and sum(v2) <= C and min(v2) >= 0:
                    accepted.add(v2)
                    out.append({"mechanism": mech2, "action": list(v2),
                                "substituted": True,
                                "replaced": mech})
                    break
            continue
        accepted.add(v)
        out.append({"mechanism": mech, "action": list(v), "substituted": False,
                    "replaced": ""})
    assert len(out) == 12, f"{key}: {len(out)} alternativas"
    return out


def main() -> int:
    df = load_universe()
    states = select_states(df)
    print(f"[d2-sel] {len(states)} estados: "
          f"{states.groupby('stratum').size().to_dict()}")
    rows = []
    for _, s in states.iterrows():
        sid = f"{s['stratum']}_lam{s['lam']}_s{s['seed']}_t{s['win']}"
        rows.append({**{"state_id": sid, "run": s["tag"], "lam": s["lam"],
                        "seed": s["seed"], "win": s["win"],
                        "stratum": s["stratum"], "vslice": s["vslice"],
                        "n_complete": s["n_complete"]},
                     "alt_index": 0, "action": list(s["a_B0"]),
                     "mechanism": "a_B0:parity", "substituted": False})
        for i, alt in enumerate(alternatives(s), 1):
            rows.append({**{"state_id": sid, "run": s["tag"], "lam": s["lam"],
                            "seed": s["seed"], "win": s["win"],
                            "stratum": s["stratum"], "vslice": s["vslice"],
                            "n_complete": s["n_complete"]},
                         "alt_index": i, "action": alt["action"],
                         "mechanism": alt["mechanism"],
                         "substituted": alt["substituted"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    n_sub = int(out.substituted.sum())
    dup_within = (out.groupby("state_id")["action"]
                    .apply(lambda s: len([tuple(x) for x in s]) !=
                           len({tuple(x) for x in s})).sum())
    print(f"[d2-sel] {len(out)} linhas (30 estados × 13) | substituições: "
          f"{n_sub} | estados c/ duplicatas: {dup_within} (deve ser 0)")
    # ---- manifest (hashes ANTES de qualquer replay) ----
    files = {
        f"D2_sampling_FINAL{_SUF}.csv": OUT_CSV,
        "generator": __file__,
        "emenda1": "/home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/"
                   "jc_decision/PROTOCOL_FROZEN_AMENDMENT_1_D2.md",
    }
    for p in sorted(glob.glob(f"{D1}/*/decisions.jsonl")):
        files[f"d1/{os.path.basename(os.path.dirname(p))}/decisions.jsonl"] = p
    m = {"sampling_seed": SAMPLING_SEED, "C": C,
         "n_states": int(len(states)), "n_rows": int(len(out)),
         "substitutions": n_sub, "files": {k: sha16(v) for k, v in files.items()},
         "note": "hashes registrados ANTES da execução de qualquer replay D2"}
    json.dump(m, open(OUT_MANIFEST, "w"), indent=2)
    print(f"[d2-sel] manifest -> {OUT_MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
