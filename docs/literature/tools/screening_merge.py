#!/usr/bin/env python3
"""Merge candidates_eixo{A,B,C,D}.csv + dedup (DOI → título normalizado) para o screening.

Saída: tools/screening_merged.json com:
  - unique: lista de obras únicas (primeira ocorrência vence; mantém fontes concatenadas)
  - dup_rows: linhas marcadas duplicata (E4) com referência à obra primeira
Uso: python3 tools/screening_merge.py
"""
import csv
import json
import pathlib
import re
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
LIT = HERE.parent

EIXOS = ["A", "B", "C", "D"]


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def norm_doi(d: str) -> str:
    d = (d or "").strip().lower().replace("https://doi.org/", "").replace("http://doi.org/", "")
    if d in {"não reportado", "nao reportado", "not reported", "n/a", "-", ""}:
        return ""
    return d


def first_author(authors: str) -> str:
    a = re.split(r"[;,]", authors or "")
    return a[0].strip().lower() if a and a[0].strip() else ""


rows = []
for eixo in EIXOS:
    path = LIT / f"candidates_eixo{eixo}.csv"
    with path.open(newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            r["_eixo"] = eixo
            r["_file_line"] = i
            r["_id"] = f"{eixo}{i:02d}"
            rows.append(r)

# dedup: (a) DOI igual (não vazio); (b) título normalizado + primeiro autor iguais
by_key: dict = {}
unique, dup_rows = [], []
for r in rows:
    doi = norm_doi(r.get("doi", ""))
    key_doi = ("doi", doi) if doi else None
    key_title = ("title", norm_title(r["title"]), first_author(r.get("authors", "")))
    hit = None
    if key_doi and key_doi in by_key:
        hit = by_key[key_doi]
    elif key_title in by_key:
        hit = by_key[key_title]
    if hit:
        hit["_sources"].append(r["source"])
        dup_rows.append({
            "id": r["_id"], "eixo": r["_eixo"], "file_line": r["_file_line"],
            "title": r["title"], "doi": r.get("doi", ""),
            "dup_of": hit["_id"], "match": "doi" if key_doi and key_doi in by_key and (not hit.get("_doi") or norm_doi(r.get("doi", "")) == hit.get("_doi")) else "title+author",
        })
        continue
    rec = {
        "_id": r["_id"], "eixo": r["_eixo"], "file_line": r["_file_line"],
        "title": r["title"].strip(), "authors": r.get("authors", "").strip(),
        "year": r.get("year", "").strip(), "venue": r.get("venue", "").strip(),
        "doi": doi, "raw_doi": r.get("doi", "").strip(),
        "_sources": [r["source"]], "why_relevant": r.get("why_relevant", "").strip(),
    }
    if key_doi:
        by_key[key_doi] = rec
    by_key[key_title] = rec
    unique.append(rec)

out = {"n_rows": len(rows), "n_unique": len(unique), "n_dup": len(dup_rows),
       "unique": unique, "dup_rows": dup_rows}
(HERE / "screening_merged.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"linhas lidas: {len(rows)}; únicas: {len(unique)}; duplicatas: {len(dup_rows)}")
for d in dup_rows:
    print(f"  DUP {d['id']} ({d['match']}) -> {d['dup_of']}: {d['title'][:60]}")
