#!/usr/bin/env python3
"""Verificação independente dos entregáveis (esta sessão).

Confere: (1) cobertura — toda linha de candidates_eixo?.csv tem entrada no
screening-log.csv; (2) dedup refeito do zero (DOI → título normalizado+1º autor)
vs. linhas duplicate=yes do log; (3) consistência log↔matrix (todo incluído na
matriz, toda linha da matriz incluída, title..doi preenchidos, demais campos
'não reportado'); (4) meta 30–60 incluídos; (5) snowball ≤5 com source=snowball.
Não altera nenhum arquivo. Uso: python3 tools/verify_deliverables.py
"""
import csv
import pathlib
import re
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
LIT = HERE.parent
problems = []


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def norm_doi(d: str) -> str:
    d = (d or "").strip().lower().replace("https://doi.org/", "")
    if d in {"não reportado", "nao reportado", "not reported", "n/a", "-", ""}:
        return ""
    return d


def first_author(a: str) -> str:
    a = re.split(r"[;,]", a or "")
    return a[0].strip().lower() if a and a[0].strip() else ""


# ---- 1. cobertura -----------------------------------------------------------
log = list(csv.DictReader((LIT / "screening-log.csv").open(encoding="utf-8")))
log_ids = {r["id"] for r in log}
if len(log_ids) != len(log):
    problems.append(f"log: ids repetidos ({len(log)} linhas, {len(log_ids)} ids)")
cand_rows = 0
for eixo in "ABCD":
    with (LIT / f"candidates_eixo{eixo}.csv").open(newline="", encoding="utf-8") as f:
        for i, _ in enumerate(csv.DictReader(f), start=2):
            cand_rows += 1
            rid = f"{eixo}{i:02d}"
            if rid not in log_ids:
                problems.append(f"cobertura: candidato {rid} ausente no screening-log")
extra = log_ids - {f"{e}{i:02d}" for e in "ABCD" for i in range(2, cand_rows + 10)}
extra = {x for x in extra if not x.startswith("SB")}
if extra:
    problems.append(f"log: ids sem candidato correspondente: {sorted(extra)}")
print(f"[1] cobertura: {cand_rows} linhas candidatas; log tem {len(log)} entradas "
      f"({len(log_ids)} ids; SB={sum(1 for r in log if r['id'].startswith('SB'))})")

# ---- 2. dedup refeito -------------------------------------------------------
seen: dict = {}
expected_dups = {}
for eixo in "ABCD":
    with (LIT / f"candidates_eixo{eixo}.csv").open(newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            rid = f"{eixo}{i:02d}"
            doi = norm_doi(r.get("doi", ""))
            kt = (norm_title(r["title"]), first_author(r.get("authors", "")))
            hit = ("doi", doi) if doi and ("doi", doi) in seen else None
            if hit is None and kt in seen:
                hit = ("title", kt)
            if hit:
                expected_dups[rid] = (hit[0], seen[hit[1]] if hit[0] == "title" else seen[("doi", doi)])
            else:
                if doi:
                    seen[("doi", doi)] = rid
                seen[kt] = rid
log_dup = {r["id"]: r["reason"] for r in log if r["duplicate"] == "yes"}
if set(expected_dups) != set(log_dup):
    problems.append(f"dedup: esperado {sorted(expected_dups)}; log marca {sorted(log_dup)}")
for rid, (how, of) in expected_dups.items():
    if rid in log_dup and f"duplicata de {of}" not in log_dup[rid]:
        problems.append(f"dedup: {rid} deveria apontar para {of} (match {how}); motivo: {log_dup[rid]!r}")
print(f"[2] dedup refeito: {len(expected_dups)} duplicatas esperadas: "
      f"{sorted(expected_dups)}; log marca {sorted(log_dup)}")

# ---- 3. log ↔ matrix --------------------------------------------------------
HEADER = ["title", "authors", "year", "venue", "doi", "problem", "architecture",
          "algorithm", "state_action_reward", "environment", "traffic_topology",
          "baselines", "metrics", "statistical_method", "key_results", "limitations",
          "reproducibility", "relation_to_our_gaps", "verified"]
with (LIT / "literature-matrix.csv").open(newline="", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    header_actual = rd.fieldnames
    matrix = list(rd)
if header_actual != HEADER:
    problems.append(f"matrix: header divergente do esperado: {header_actual}")
inc = {r["id"] for r in log if r["decision"] == "include"}
mkeys = {(norm_title(r["title"]), norm_doi(r["doi"])) for r in matrix}
if len(mkeys) != len(matrix):
    problems.append("matrix: linhas repetidas (título+doi)")
log_inc_keys = {(norm_title(r["title"]), norm_doi(r["doi"])) for r in log
                if r["decision"] == "include"}
if log_inc_keys != mkeys:
    only_log = log_inc_keys - mkeys
    only_m = mkeys - log_inc_keys
    if only_log:
        problems.append(f"matrix: incluídos do log ausentes na matrix: {only_log}")
    if only_m:
        problems.append(f"matrix: linhas que não são incluídas do log: {only_m}")
for r in matrix:
    for k in ("title", "authors", "year", "venue", "doi"):
        if not (r.get(k) or "").strip():
            problems.append(f"matrix: campo {k} vazio em {r['title'][:40]!r}")
    for k in HEADER[5:]:
        if (r.get(k) or "").strip() != "não reportado":
            problems.append(f"matrix: campo profundo {k} != 'não reportado' em {r['title'][:40]!r}")
n_inc = len(inc)
if not (30 <= len(matrix) <= 60):
    problems.append(f"meta 30–60 incluídos: matriz tem {len(matrix)}")
sb = [r for r in log if r["source"] == "snowball"]
if len(sb) > 5:
    problems.append(f"snowball: {len(sb)} > 5")
if any(r["decision"] != "include" for r in sb):
    problems.append("snowball: entrada com decision != include")
print(f"[3] log↔matrix: {n_inc} incluídos no log; {len(matrix)} linhas na matrix; "
      f"snowball={len(sb)} ({[r['id'] for r in sb]})")

# ---- 4. decisões têm razão e checked_by -------------------------------------
for r in log:
    if not (r["reason"] or "").strip():
        problems.append(f"log: {r['id']} sem razão")
    if "Triador" not in (r["checked_by"] or ""):
        problems.append(f"log: {r['id']} checked_by={r['checked_by']!r}")
    if r["decision"] not in {"include", "exclude"}:
        problems.append(f"log: {r['id']} decision={r['decision']!r}")

print()
if problems:
    print(f"PROBLEMAS ({len(problems)}):")
    for p in problems:
        print(" -", p)
else:
    print("VERIFICAÇÃO OK: todos os 5 grupos de checagens passaram.")
