#!/usr/bin/env python3
"""Valida DOIs da literature-matrix.csv (stdlib only).

Cada DOI é verificado na API do Crossref; se lá não estiver registrado,
cai para a API da DataCite — DOIs arXiv (10.48550) e Zenodo (10.5281) são
registrados na DataCite e são tão válidos quanto os do Crossref (correção
da rodada 2: antes o check marcava FALHA para qualquer DOI DataCite).

Uso:
  python3 crossref_check.py ../literature-matrix.csv [--column doi]

Saída: por linha, OK (Crossref) / OK-DATACITE / FALHA (com título do registro);
'SEM DOI' (campo 'não reportado') conta como problema — todo incluído deve ter
DOI verificável quando exista. Exit 1 se houver problema. Registra validações
em search-log.jsonl.
"""
import argparse
import csv
import datetime
import json
import pathlib
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
UA = {"User-Agent": "oran-literature-review/1.0 (mailto:pesquisa@localhost)"}


def check_datacite(doi: str) -> str:
    """Retorna 'Título | publisher' se o DOI estiver registrado na DataCite."""
    url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        a = json.loads(r.read().decode())["data"]["attributes"]
    title = ((a.get("titles") or [{}])[0].get("title") or "")[:70]
    return f"{title} | {a.get('publisher')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--column", default="doi")
    args = ap.parse_args()

    bad = 0
    n_datacite = 0
    with open(args.csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows, start=2):
        doi = (row.get(args.column) or "").strip().replace("https://doi.org/", "")
        if not doi or doi.lower() in {"não reportado", "nao reportado", "not reported", "-"}:
            print(f"L{i}: SEM DOI ({doi!r})")
            bad += 1
            continue
        url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                msg = json.loads(r.read().decode()).get("message", {})
            title = (msg.get("title") or [""])[0][:70]
            print(f"L{i}: OK  {doi}  ->  {title}")
        except Exception:  # noqa: BLE001
            try:
                info = check_datacite(doi)
                print(f"L{i}: OK-DATACITE  {doi}  ->  {info}")
                n_datacite += 1
            except Exception as e2:  # noqa: BLE001
                print(f"L{i}: FALHA  {doi}  (Crossref e DataCite: {e2})")
                bad += 1
    with (HERE / "search-log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source": "crossref+datacite", "action": "doi_check",
            "file": args.csv_path, "rows": len(rows), "failures": bad,
            "datacite_ok": n_datacite,
        }, ensure_ascii=False) + "\n")
    print(f"\n[check] {len(rows)} linhas, {bad} problema(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
