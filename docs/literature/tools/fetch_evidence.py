#!/usr/bin/env python3
"""Coleta de evidências por DOI para preenchimento da literature-matrix.csv
(linhas 14–25, extrator 2 — 2026-10-18).

Para cada DOI, tenta em ordem:
  1. OpenAlex  /works/doi:<DOI>   — metadados + abstract_inverted_index reconstruído
                                    + localizações OA (landing/PDF) + is_oa.
  2. Crossref  /works/<DOI>       — metadados + abstract (quando o publisher deposita).
  3. Semantic Scholar /graph/v1/paper/DOI:<DOI> — abstract + openAccessPdf.

Salva tudo em raw/<ts>_evidence_rows14-25.json e registra CADA consulta em
search-log.jsonl (protocolo passo 4 da skill). Nenhum campo é inventado: o que
não vier de uma dessas fontes permanece ausente do JSON.
"""
import argparse
import datetime
import json
import pathlib
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
UA = {"User-Agent": "oran-literature-review/1.0 (mailto:pesquisa@localhost)"}


def get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.loads(r.read().decode())


def log(source: str, action: str, query: str, results: int, outcome: str) -> None:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (HERE / "search-log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": ts, "source": source, "action": action,
                            "query": query, "results": results, "outcome": outcome},
                           ensure_ascii=False) + "\n")


def abstract_from_inverted(inv) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def openalex(doi: str) -> dict:
    url = "https://api.openalex.org/works/" + urllib.parse.quote("doi:" + doi, safe="")
    w = get_json(url)
    locs = []
    for loc in w.get("locations") or []:
        src = (loc.get("source") or {}).get("display_name") or ""
        locs.append({"source": src, "landing": loc.get("landing_page_url") or "",
                     "pdf": loc.get("pdf_url") or "", "is_oa": loc.get("is_oa")})
    return {
        "openalex_id": w.get("id"), "title": w.get("title"),
        "year": w.get("publication_year"),
        "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "is_oa": (w.get("open_access") or {}).get("is_oa"),
        "oa_url": (w.get("open_access") or {}).get("oa_url"),
        "cited_by_count": w.get("cited_by_count"),
        "abstract": abstract_from_inverted(w.get("abstract_inverted_index")),
        "locations": locs,
    }


def crossref(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    msg = get_json(url).get("message", {})
    abs_txt = msg.get("abstract", "")
    # Crossref deposita JATS: remove tags simples
    out = []
    for chunk in abs_txt.replace("<", "\n<").split("\n"):
        if chunk.startswith("<"):
            continue
        out.append(chunk)
    return {"title": (msg.get("title") or [""])[0],
            "container": (msg.get("container-title") or [""])[0],
            "publisher": msg.get("publisher"),
            "abstract": " ".join(" ".join(out).split())}


def semanticscholar(doi: str) -> dict:
    fields = "title,abstract,venue,year,externalIds,openAccessPdf,authors"
    url = ("https://api.semanticscholar.org/graph/v1/paper/DOI:" +
           urllib.parse.quote(doi, safe="") + f"?fields={fields}")
    p = get_json(url)
    pdf = (p.get("openAccessPdf") or {})
    return {"title": p.get("title"), "venue": p.get("venue"), "year": p.get("year"),
            "abstract": p.get("abstract") or "",
            "oa_pdf": pdf.get("url"),
            "arxiv_id": (p.get("externalIds") or {}).get("ArXiv"),
            "authors": [a.get("name") for a in p.get("authors") or []]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dois", help="DOIs separados por vírgula")
    ap.add_argument("--out", default="evidence_rows14-25")
    args = ap.parse_args()

    result = {}
    for doi in [d.strip() for d in args.dois.split(",") if d.strip()]:
        rec = {"doi": doi, "openalex": None, "crossref": None, "semanticscholar": None}
        try:
            rec["openalex"] = openalex(doi)
            log("openalex", "work_fetch", f"api.openalex.org/works/doi:{doi}",
                1, f"ok; abstract={'sim' if rec['openalex']['abstract'] else 'não'}; is_oa={rec['openalex']['is_oa']}")
        except Exception as e:  # noqa: BLE001
            rec["openalex_error"] = str(e)
            log("openalex", "work_fetch", f"api.openalex.org/works/doi:{doi}", 0, f"erro: {e}")
        if not (rec["openalex"] and rec["openalex"]["abstract"]):
            try:
                rec["crossref"] = crossref(doi)
                log("crossref", "work_fetch", f"api.crossref.org/works/{doi}", 1,
                    f"ok; abstract depositado={'sim' if rec['crossref']['abstract'] else 'não'}")
            except Exception as e:  # noqa: BLE001
                log("crossref", "work_fetch", f"api.crossref.org/works/{doi}", 0, f"erro: {e}")
        if not (rec["openalex"] and rec["openalex"]["abstract"]) and not (rec["crossref"] and rec["crossref"]["abstract"]):
            try:
                rec["semanticscholar"] = semanticscholar(doi)
                log("semanticscholar", "work_fetch",
                    f"api.semanticscholar.org/graph/v1/paper/DOI:{doi}", 1,
                    f"ok; abstract={'sim' if rec['semanticscholar']['abstract'] else 'não'}; arxiv={rec['semanticscholar']['arxiv_id']}")
            except Exception as e:  # noqa: BLE001
                log("semanticscholar", "work_fetch",
                    f"api.semanticscholar.org/graph/v1/paper/DOI:{doi}", 0, f"erro: {e}")
        result[doi] = rec
        print(f"[{doi}] openalex={'ok' if rec['openalex'] else 'ERRO'} "
              f"abstract_OA={'sim' if rec['openalex'] and rec['openalex']['abstract'] else 'não'} "
              f"abstract_CR={'sim' if rec['crossref'] and rec['crossref']['abstract'] else 'não'} "
              f"abstract_S2={'sim' if rec['semanticscholar'] and rec['semanticscholar']['abstract'] else 'não'}")

    raw_dir = HERE / "raw"
    raw_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    raw_path = raw_dir / f"{ts}_{args.out}.json"
    raw_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    log("fetch_evidence.py", "save_raw", str(raw_path), len(result), "resumo por DOI salvo")
    print(f"\n[raw] {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
