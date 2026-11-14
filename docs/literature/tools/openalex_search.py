#!/usr/bin/env python3
"""Busca reproduzível na API do OpenAlex (stdlib only, sem chave).

Uso:
  python3 openalex_search.py "QUERY" [--from-year 2019] [--out PREFIX] [--max 50]

Registra cada busca em search-log.jsonl (query, fonte, data UTC, contagem) e salva
o JSON bruto em raw/<ts>_<PREFIX>.json para rastreabilidade (protocolo da skill
oran-literature-review, passo 4).
"""
import argparse
import datetime
import json
import pathlib
import urllib.parse
import urllib.request

BASE = "https://api.openalex.org/works"
HERE = pathlib.Path(__file__).resolve().parent
UA = {"User-Agent": "oran-literature-review/1.0 (mailto:pesquisa@localhost)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--from-year", type=int, default=2019)
    ap.add_argument("--out", default="search")
    ap.add_argument("--max", type=int, default=50)
    args = ap.parse_args()

    filt = f"from_publication_date:{args.from_year}-01-01"
    params = urllib.parse.urlencode({
        "search": args.query,
        "filter": filt,
        "per-page": min(args.max, 100),
        "select": "id,doi,title,publication_year,primary_location,cited_by_count,authorships",
    })
    url = f"{BASE}?{params}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        data = json.loads(r.read().decode())

    raw_dir = HERE / "raw"
    raw_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    raw_path = raw_dir / f"{ts}_{args.out}.json"
    raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=1))

    log_path = HERE / "search-log.jsonl"
    count = data.get("meta", {}).get("count", 0)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": ts, "source": "openalex", "query": args.query,
            "filter": filt, "count": count, "returned": len(data.get("results", [])),
            "raw": str(raw_path.relative_to(HERE)),
        }, ensure_ascii=False) + "\n")

    for w in data.get("results", []):
        venue = ""
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        venue = src.get("display_name") or ""
        doi = w.get("doi") or ""
        print(f"{w.get('publication_year')} | cites={w.get('cited_by_count', 0):5d} | {venue[:40]:40s} | {doi.replace('https://doi.org/', ''):38s} | {(w.get('title') or '')[:95]}")
    print(f"\n[log] {count} resultados | bruto: {raw_path} | busca registrada em search-log.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
