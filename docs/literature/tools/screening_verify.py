#!/usr/bin/env python3
"""Evidências para a triagem (I2/E3, E2-carve-out, I3-surveys, I1-seminais, snowball).

Consulta (somente leitura, APIs públicas):
  1. arXiv API  — journal-ref/comment/links dos preprints candidatos (I2/E3).
  2. OpenAlex   — referenced_works de incluídos provisórios ∩ IDs dos candidatos
                  de fronteira (carve-out E2, exceção I1, surveys I3).
  3. Crossref/DataCite — DOIs de seminais de snowball e casos sem DOI no arquivo.

Registra tudo em search-log.jsonl. Uso: python3 tools/screening_verify.py
"""
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = pathlib.Path(__file__).resolve().parent
LOG = HERE / "search-log.jsonl"
UA = {"User-Agent": "oran-literature-review/1.0 (mailto:pesquisa@localhost)"}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def log(entry: dict) -> None:
    entry = {"ts": time.strftime("%Y%m%dT%H%M%S"), **entry}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 1. arXiv
ARXIV = {
    "A05": "2208.14877", "A20": "2501.05879", "B24": "2605.02149",
    "D17": "2607.11607", "D18": "1903.02526", "D19": "1906.05819",
    "D20": "2605.22207", "D23": "2509.12833", "D25": "2608.28532",
}
print("=" * 30, "1. arXiv (journal-ref / artefatos)", "=" * 30)
ns = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}
ids = ",".join(ARXIV.values())
xml = get(f"http://export.arxiv.org/api/query?id_list={ids}&max_results=20")
root = ET.fromstring(xml)
arxiv_res = {}
for e in root.findall("a:entry", ns):
    aid_raw = e.find("a:id", ns).text
    aid = aid_raw.rsplit("/", 1)[-1].split("v")[0]
    jr = e.find("ar:journal_ref", ns)
    cm = e.find("ar:comment", ns)
    summ = e.find("a:summary", ns).text or ""
    links = [l.get("href", "") for l in e.findall("a:link", ns)]
    code = sorted(set(re.findall(r"(?:github\.com|gitlab\.com)/[\w\-./]+", summ)))
    res = {"title": e.find("a:title", ns).text.strip().replace("\n", " ")[:80],
           "journal_ref": jr.text if jr is not None else None,
           "comment": (cm.text[:160] if cm is not None else None),
           "code_links": code, "page_link": links}
    arxiv_res[aid] = res
    print(f"[{aid}] {res['title']}\n    journal_ref={res['journal_ref']}\n    comment={res['comment']}\n    code={code}")
log({"source": "arxiv", "action": "preprint_check", "ids": list(ARXIV.values()),
     "result": {k: {"journal_ref": v["journal_ref"], "code_links": v["code_links"],
                    "comment": v["comment"]} for k, v in arxiv_res.items()}})
time.sleep(1)

# ------------------------------------------------------- 2. OpenAlex citações
def oa_id(doi: str):
    try:
        d = json.loads(get("https://api.openalex.org/works/doi:" + urllib.parse.quote(doi)
                           + "?select=id,display_name,referenced_works").decode())
        return d
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}

# incluídos provisórios cujas referências serão inspecionadas
INCLUDES = {
    "B02_KBRL": "10.1109/twc.2022.3195570",
    "B04_NagibSafe": "10.1109/jsac.2023.3336191",
    "B05_SafeSlice": "10.1109/icmlcn64995.2025.11140546",
    "B11_GANdist": "10.1109/jsac.2019.2959185",
    "D17_EldeebEuCNC": "10.1109/eucnc/6gsummit63408.2025.11037178",
    "D15_mmWaveIRS": "10.1109/twc.2021.3107520",
    "A04_ColORAN": "10.1109/tmc.2022.3188013",
    "B06_Zangooei": "10.1109/jsac.2023.3336156",
    "A18_RSLAQ": "10.1109/tmc.2026.3666787",
    "A24_UnderstandingORAN": "10.1109/comst.2023.3239220",
    "B13_Dai": "10.1109/tmc.2024.3476338",
    "A16_ORANSlice": "10.1145/3636534.3701544",
}
refs = {}
print("=" * 30, "2a. referenced_works dos incluídos", "=" * 30)
for name, doi in INCLUDES.items():
    d = oa_id(doi)
    if "error" in d:
        print(f"{name}: ERRO {d['error']}")
        continue
    refs[name] = set(d.get("referenced_works", []))
    print(f"{name}: {len(refs[name])} refs  ({d['display_name'][:60]})")
    time.sleep(0.4)

# candidatos de fronteira -> OpenAlex ID
BORDER = {
    "D12_DabneyQR": "10.1609/aaai.v32i1.11791",
    "D13_YangDistSafety": "10.1007/s10994-022-06187-8",
    "D16_LyleComparative": "10.1609/aaai.v33i01.33014504",
    "D22_LutjensICRA": "10.1109/icra.2019.8793611",
    "D23_KochdumperProj": "10.1109/ojcsys.2023.3256305",
    "A15_TutorialDeploy": "10.1109/comst.2025.3577687",
    "A26_TutorialSlicing": "10.1109/comst.2025.3598406",
    "B26_SurveySensors": "10.3390/s22083031",
}
border_ids = {}
print("=" * 30, "2b. IDs OpenAlex dos candidatos de fronteira", "=" * 30)
for name, doi in BORDER.items():
    d = oa_id(doi)
    if "error" in d:
        print(f"{name}: ERRO {d['error']}")
        continue
    border_ids[name] = d["id"].rsplit("/", 1)[-1]
    print(f"{name}: {border_ids[name]}")
    time.sleep(0.4)

# D14 (não tem DOI): buscar por título no OpenAlex
q = urllib.parse.quote("Non-Crossing Quantile Regression for Distributional Reinforcement Learning")
try:
    d = json.loads(get(f"https://api.openalex.org/works?filter=title.search:{q}&per-page=5").decode())
    for w in d.get("results", [])[:3]:
        print("D14_busca:", w["id"].rsplit("/", 1)[-1], "|", w["display_name"][:70],
              "|", w.get("publication_year"), "|", (w.get("primary_location") or {}).get("source", {}) and ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
              "| doi:", w.get("doi"))
        border_ids.setdefault("D14_NonCrossingQR", w["id"].rsplit("/", 1)[-1])
    time.sleep(0.4)
except Exception as e:  # noqa: BLE001
    print("D14_busca ERRO:", e)

print("=" * 30, "2c. Quem cita os de fronteira (entre os incluídos)", "=" * 30)
cite_map = {}
for bname, bid in border_ids.items():
    citers = [iname for iname, rset in refs.items() if bid in rset]
    cite_map[bname] = citers
    print(f"{bname}: citado por -> {citers if citers else 'NENHUM dos incluídos inspecionados'}")
log({"source": "openalex", "action": "citation_check",
     "includes_inspected": list(INCLUDES), "borderline": cite_map})

# -------------------------------------------- 3. Crossref/DataCite (snowball)
print("=" * 30, "3. DOIs de seminais (Crossref search)", "=" * 30)
def cr_search(title: str):
    url = ("https://api.crossref.org/works?query.bibliographic=" + urllib.parse.quote(title)
           + "&rows=3&select=DOI,title,author,issued,container-title")
    try:
        d = json.loads(get(url).decode())
        out = []
        for it in d["message"]["items"]:
            out.append({"doi": it["DOI"], "title": (it.get("title") or [""])[0][:80],
                        "container": (it.get("container-title") or [""])[0][:50],
                        "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0]})
        return out
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)}]

SNOWBALL_TITLES = [
    "Regression Quantiles Koenker Bassett Econometrica 1978",
    "Energy-Efficient LDPC Acceleration in Open Radio Access Networks Using DRL and QRF",
    "Colosseum Large-Scale Wireless Experimentation Hardware-in-the-Loop Network Emulation",
    "Hierarchical Intelligence Enabled Joint RAN Slicing and MAC Scheduling for SLA Guarantee",
    "Distributional Reinforcement Learning With Quantile Regression Dabney",
]
snow = {}
for t in SNOWBALL_TITLES:
    res = cr_search(t)
    snow[t] = res
    print(f"\nQUERY: {t}")
    for r in res:
        print("   ", r)
    time.sleep(0.5)
log({"source": "crossref", "action": "snowball_doi_search", "queries": snow})

# DataCite: C13 (calibração 5G-LENA) e DOI arXiv do Dabney não precisa (já Crossref)
for doi in ["10.48550/arxiv.2205.03278", "10.48550/arxiv.1903.02526", "10.48550/arxiv.1906.05819"]:
    try:
        d = json.loads(get("https://api.datacite.org/dois/" + urllib.parse.quote(doi)).decode())
        attrs = d["data"]["attributes"]
        print(f"DATACITE OK {doi}: {attrs['titles'][0]['title'][:70]} ({attrs.get('publicationYear')})")
        log({"source": "datacite", "action": "doi_check", "doi": doi, "ok": True,
             "title": attrs["titles"][0]["title"][:80]})
    except Exception as e:  # noqa: BLE001
        print(f"DATACITE FALHA {doi}: {e}")
        log({"source": "datacite", "action": "doi_check", "doi": doi, "ok": False, "err": str(e)})
    time.sleep(0.4)

# Crossref direto para DOIs críticos (SQR/CCNC 2026 e NeurIPS não tem DOI)
for doi in ["10.1109/ccnc65079.2026.11366589", "10.1609/aaai.v32i1.11791"]:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    try:
        msg = json.loads(get(url).decode())["message"]
        print(f"CROSSREF OK {doi}: {(msg.get('title') or [''])[0][:70]}")
        log({"source": "crossref", "action": "doi_check", "doi": doi, "ok": True})
    except Exception as e:  # noqa: BLE001
        print(f"CROSSREF FALHA {doi}: {e}")
        log({"source": "crossref", "action": "doi_check", "doi": doi, "ok": False, "err": str(e)})
    time.sleep(0.4)

print("\n[evidence] concluído — decisões registradas em search-log.jsonl")
