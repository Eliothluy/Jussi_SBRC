#!/usr/bin/env python3
"""Aplica as decisões de triagem e gera screening-log.csv + literature-matrix.csv.

Critérios: search-strategy.md §4 (I1–I5, E1–E7). Evidências: screening_verify.py
(desta sessão) + search-log.jsonl (sessões anteriores).
Uso: python3 tools/screening_finalize.py
"""
import csv
import json
import pathlib
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
LIT = HERE.parent
UA = {"User-Agent": "oran-literature-review/1.0 (mailto:pesquisa@localhost)"}
CHECKER = "Triador (oran-literature-review), 2026-10-18"

merged = json.loads((HERE / "screening_merged.json").read_text(encoding="utf-8"))
unique = {r["_id"]: r for r in merged["unique"]}
dup_rows = merged["dup_rows"]

I = "include"
E = "exclude"
# id -> (decisão, motivo). Motivos de include recebem o why_relevant anexado.
DECISIONS = {
    # ------------------------------------------------ Eixo A (17 incluídos)
    "A02": (I, "I1–I5 OK: ancestral direto do ambiente MODA (fork ns-O-RAN Gym); loop fechado ns-3↔Near-RT RIC (pergunta 5)."),
    "A03": (E, "I5 (corte de curadoria): linhagem ns-3+O-RAN já representada por A02 (fork de origem do nosso ambiente) e A25 (mesmo módulo NR); contribuição incremental da integração NIST marginal às perguntas 1–8."),
    "A04": (I, "I1–I5 OK: referência canônica de xApps ML em loop fechado sobre SDR; métricas/protocolo de avaliação de xApp (pergunta 5)."),
    "A05": (I, "I1–I5 OK: validação de campo (OpenRAN Gym) de xApp de traffic steering (pergunta 5)."),
    "A06": (I, "I2 OK: preprint com artefato público verificado (plataforma OpenRAN Gym, ecossistema do fork ns-o-ran-gym do MODA); DOI arXiv verificado na DataCite em sessão anterior (search-log 16:14)."),
    "A07": (I, "I1–I5 OK: integra FlexRIC + ns-O-RAN — mesma combinação da nossa Fase 5 (paridade/latência, pergunta 5)."),
    "A08": (I, "I1–I5 OK: paper de origem do FlexRIC (RIC alternativo não adotado); DOI corrigido e verificado no Crossref em sessão anterior (search-log 16:14)."),
    "A09": (I, "I2 OK: SBrT 2025 (revisado por pares); sem DOI no documento — confirmado por leitura integral do PDF em sessão anterior (search-log 16:14); paper do módulo NORI portado na nossa Fase 5."),
    "A10": (I, "I1–I5 OK: análise de latência das interfaces O-RAN (incl. E2) — base para contextualizar nossos p50/p95 de 2,39/9,28 ms (pergunta 5)."),
    "A11": (E, "I5: aplicação de segurança (detecção DDoS) fora dos eixos A–D; a contribuição aproveitável (metodologia de medição de latência de loop O-RAN) já está representada por A10 e A13."),
    "A12": (E, "I5: comunicação inter-xApp é mecanismo ortogonal ao nosso loop single-xApp via E2; latência de loops já coberta por A10 (interfaces O-RAN) e A13 (IA dentro de xApp)."),
    "A13": (I, "I1–I5 OK: implanta e mede IA dentro de xApp em Near-RT RIC com latência mensurada (pergunta 5)."),
    "A14": (I, "I1–I5 OK: xApps RL experimentais com OSC RIC + srsRAN — validação real comparável à nossa (pergunta 5)."),
    "A15": (E, "I3: survey/tutorial sem componente mensurável; cláusula de agenda citada por incluído não verificada (nenhum dos 12 incluídos inspecionados no OpenAlex o cita — screening_verify.py 2c, desta sessão)."),
    "A16": (I, "I1–I5 OK: plataforma open source de slicing 5G sobre O-RAN com artefato público (G2; slicing via RIC)."),
    "A17": (I, "I1–I5 OK: slicing O-RAN orientado a SLA via RIC — SOTA a contrastar com SQR/KBRL (perguntas 1 e 5)."),
    "A18": (I, "I1–I5 OK: RSLAQ — linhagem do cenário do ambiente MODA; xApp DRL SLA-driven (perguntas 1 e 5, G5)."),
    "A19": (E, "I5: aplicação de localização, não alocação/slicing; linhagem FlexRIC já representada por A07/A08."),
    "A20": (I, "I1–I5 OK: loop de controle O-RAN fim a fim para alocação de rádio em SDR, com métricas (pergunta 5)."),
    "A21": (E, "E3/I2: preprint sem revisão por pares e sem artefatos verificados (arXiv API nesta sessão: sem journal-ref, nenhum link de código no registro)."),
    "A22": (I, "I1–I5 OK: xApps portáveis entre RICs heterogêneos — discute acoplamento da stack NORI/e2sim e migração para RIC real (G5)."),
    "A23": (E, "I3: artigo de posição/arquitetura (IEEE Comms Standards Mag) sem componente mensurável; agenda de automação preditiva já representada por incluídos validados (A04/A05)."),
    "A24": (I, "I3 (cláusula de agenda citada por incluído): survey canônico citado pelo artigo SQR incluído (tradução.md ref. [3], verificada contra o PDF); referência de fundo do eixo A."),
    "A25": (I, "I1–I5 OK: simulador Open-RAN sobre 5G-LENA (mesmo módulo NR do MODA) — antecedente direto da combinação ns-3.46+v5.2."),
    "A26": (E, "I3: survey/tutorial sem componente mensurável; citação por incluído não verificada (OpenAlex, 12 incluídos inspecionados nesta sessão)."),
    # ------------------------------------------------ Eixo B (15 incluídos)
    "B02": (I, "I1–I5 OK: KBRL — baseline reproduzido em port fiel (Fase 3); origem do método model-based com kernels (pergunta 8)."),
    "B04": (I, "I1–I5 OK: safe RL com transfer learning já aplicado a slicing O-RAN — antecedente central para delimitar a novidade do SQR (pergunta 8)."),
    "B05": (I, "I1–I5 OK: SafeSlice — restrições de latência cumulativa/instantânea com camada de segurança; candidato mais próximo do SQR-JC (pergunta 8)."),
    "B06": (I, "I1–I5 OK: coordenação multi-agente com restrições de SLA em Open RAN (pergunta 8, G8)."),
    "B07": (I, "I1–I5 OK: formulação constrained learning/CMDP para slicing RAN dinâmico — referência de formulação com restrição de SLA (pergunta 8)."),
    "B08": (E, "I5: resource scaling de slices em nível de rede (NOMS); linhagem constrained-RL para slicing já representada por B04–B07 e D07/D08; corte de curadoria."),
    "B09": (I, "I1–I5 OK: CVaR-constrained para alocação dinâmica em RAN slicing — medida de risco de cauda vizinha do quantílico do SQR (pergunta 8, G6)."),
    "B10": (I, "I1–I5 OK: risk-aware RL para O-RAN user-centric, granularidade per-UE (perguntas 7 e 8)."),
    "B11": (I, "I1–I5 OK: GAN-powered deep distributional RL para slicing — antecedente do ramo distributional/quantílico explorado pelo SQR (pergunta 8)."),
    "B12": (E, "I5/sobreposição: mesmos autores e ano que D17 (EuCNC 2025, incluído), foco RRM; títulos e DOIs distintos (E4 não se aplica) — mantida a versão de conferência."),
    "B14": (I, "I1–I5 OK: DRL para slicing O-RAN com SLAs variáveis no tempo, avaliado em OpenRAN Gym/ns-3 — linhagem de ambiente idêntica à nossa (perguntas 1 e 5)."),
    "B15": (I, "I1–I5 OK: PSASlicing — claim de RL perpétuo sem violação de SLA; contraponto direto ao trade-off risco×consumo (perguntas 4 e 6, G6); citado pelo SQR (tradução.md ref. [9])."),
    "B16": (E, "I5: formulação genérica DRL-slicing-SLA sem O-RAN/safe/quantile; SOTA de slicing+DRL já representado por B17/B18/B19/B20; corte de curadoria."),
    "B17": (I, "I1–I5 OK: orquestração cross-slice multi-tenant com DRL — SOTA clássico de coordenação inter-slice (pergunta 8)."),
    "B18": (I, "I1–I5 OK: hierarchical DRL para RAN slicing — arquitetura de política hierárquica comparável à do SQR (pergunta 8)."),
    "B19": (I, "I1–I5 OK: alocação flexível de RBs por slice via DRL — mesma granularidade de ação do SQR (perguntas 3 e 4)."),
    "B20": (I, "I1–I5 OK: resource slicing hierárquico eMBB/URLLC — heterogeneidade de serviços ausente no SQR (G8)."),
    "B21": (E, "I5: alocação online genérica por DRL, sem componente O-RAN/safe/quantile; cluster DRL-slicing já representado por B17/B18/B19/B20; corte de curadoria."),
    "B22": (E, "I5: MARL veicular — domínio veicular representado por B07 e ângulo MARL por B06; sem conexão específica às perguntas 1–8 além da genérica."),
    "B23": (E, "I5: treinamento federado ortogonal às perguntas 1–8; coordenação distribuída já representada por B06."),
    "B24": (I, "I1–I5 OK: DORA — PPO alocando PRBs por slice (URLLC/eMBB/mMTC) em O-RAN; mesma granularidade de ação (perguntas 3 e 4)."),
    "B25": (E, "E3/I2: preprint sem journal-ref nem código (arXiv API nesta sessão); versão publicada do mesmo grupo já incluída (B24, DORA/MECOM 2025)."),
    "B26": (E, "I3: survey sem componente mensurável; citação por incluído não verificada (OpenAlex, 12 incluídos inspecionados nesta sessão)."),
    # ------------------------------------------------ Eixo C (17 incluídos)
    "C02": (I, "I1–I5 OK: extensão do próprio 5G-LENA para RAN slicing flexível/dinâmico — antecedente de plataforma mais próximo do nosso cenário."),
    "C03": (I, "I1–I5 OK: scheduler QoS-aware implementado e avaliado no MAC do 5G-LENA — precedente de instrumentação de ação no MAC (G3)."),
    "C04": (E, "I5: extensão de scheduler com IA no 5G-LENA — padrão já documentado por C02/C03/C05/C06; corte de curadoria dentro do cluster MAC 5G-LENA."),
    "C05": (I, "I1–I5 OK: suporte a QoS por fluxo no módulo NR — base para métricas por slice/UE."),
    "C06": (I, "I1–I5 OK: atribuição de RBs por sub-banda no 5G-LENA — granularidade análoga às cotas de PRB do SQR (G3)."),
    "C07": (I, "I1–I5 OK: configuração flexível de modelo de canal/path loss no 5G-LENA — insumo direto para a matriz de calibração potência/raio/LOS (G1)."),
    "C08": (I, "I1–I5 OK: calibração site-specific via ray tracing no ns-3, no mesmo venue do artigo SQR (G1)."),
    "C09": (I, "I1–I5 OK: implementação e validação do modelo de canal 3GPP TR 38.901 no ns-3 — fundamento da cadeia de propagação que rege RLF de borda (G1)."),
    "C10": (I, "I1–I5 OK: paper de origem do módulo NR (5G-LENA/CTTC) — referência primária da plataforma."),
    "C11": (E, "I5: caso URLLC/configured-grant sem slicing/RL; metodologia de implementação+validação open-source já representada por C14/C16; corte de curadoria."),
    "C12": (E, "I5: avaliação sidelink V2X sem slicing/controle de rádio celular; prática de avaliação system-level 5G-LENA já representada por C10/C13."),
    "C13": (I, "I1–I5 OK: modelos de tráfego misto NGMN no ns-3 — fidelidade da geração de tráfego (chegada Poisson/sessões) nos cenários de slicing."),
    "C14": (I, "I1–I5 OK: calibração do 5G-LENA contra cenários de referência 3GPP (macro outdoor) — precedente central para a calibração realista (G1). I4: DOI da versão arXiv 10.48550/arxiv.2205.03278 verificado na DataCite nesta sessão (título/ano conferem); venue RS-OJICT sem DOI verificado."),
    "C15": (I, "I1–I5 OK: controle hierárquico orientado a SLA avaliado em ns-3/5G-LENA — antecedente de avaliação de violação de SLA na nossa plataforma."),
    "C16": (I, "I1–I5 OK: execução em lote/paralela de simulações ns-3 — infraestrutura de campanha multi-seed reprodutível (G2)."),
    "C17": (E, "I5: ferramental genérico de reprodutibilidade para ns-3; G2 já representado por C16 (execução em lote) e C14 (calibração documentada); corte de curadoria."),
    "C18": (E, "E1: sem componente RAN — validação MAC 802.11ax/WiFi, não rede celular."),
    "C19": (E, "I3: revisão sistemática genérica de simulação de redes, sem componente RAN/slicing mensurável; contexto de G2 já representado por incluídos."),
    "C20": (I, "I1–I5 OK: calibração de modelo de canal 3GPP industrial no ns-3, artefato DataCite verificado (G1)."),
    "C21": (I, "I1–I5 OK: reformulação do link budget para 5G — suporte analítico ao déficit de 25–30 dB identificado na borda da célula (G1)."),
    "C22": (E, "I5: modelagem de path loss consciente de prédios; insumos de propagação/calibração (G1) já representados por C07/C08/C09/C20/C21; corte de curadoria."),
    "C24": (I, "I1–I5 OK: RRM slice-aware com restrições de SLA em simulação system-level — comparação com abordagem de gerência de cotas."),
    "C25": (I, "I1–I5 OK: avaliação de isolamento/prioridade/eficiência entre slices no RAN — conecta ao mecanismo de degradação/isolamento (§5)."),
    "C26": (I, "I1–I5 OK: cobertura uplink em células macro muito grandes (system+link level) — precedente para a inviabilidade de SLA a 2 km / RLF de borda (§4.3)."),
    # ------------------------------------------------ Eixo D (7 incluídos)
    "D03": (I, "I1–I5 OK: artigo-base do projeto (reproduzido/auditado); I4: DOI verificado no Crossref nesta sessão."),
    "D06": (I, "I1–I5 OK: Safe-NORA — safe RL para alocação de recursos em rede móvel com demandas diversas (pergunta 8)."),
    "D07": (I, "I1–I5 OK: formulação CMDP explícita para network slicing — primeiro bloco da linhagem constrained-RL-slicing (pergunta 8)."),
    "D08": (I, "I1–I5 OK: CLARA — constrained RL para alocação multi-dominio em slicing; baseline recorrente da área (pergunta 8)."),
    "D09": (E, "I5: terceira variação da mesma linhagem (Liu/Ding/Xin, constrained RL p/ slicing) já representada por D07/D08; corte de curadoria."),
    "D10": (I, "I1–I5 OK: DRL constraint-aware ponta a ponta (RAN+transporte) — delimita a fronteira do eixo (pergunta 8)."),
    "D12": (E, "I1: seminal pré-2019 (AAAI 2018); a exceção I1 exige entrada via snowballing e nenhum dos 12 incluídos inspecionados o cita (OpenAlex, screening_verify.py 2c desta sessão); a origem do mecanismo quantílico fica representada por Koenker & Bassett 1978 (SB1, citado pelo SQR — tradução.md ref. [13])."),
    "D13": (E, "E2: RL fora de redes sem fio (crítico de segurança distribucional, domínio de controle); carve-out (origem de mecanismo citado por incluído) não verificado (OpenAlex, 12 incluídos)."),
    "D14": (E, "E2: distributional RL fora de redes (NeurIPS 2020); carve-out de citação não verificado (OpenAlex); metadados verificados nesta sessão (OpenAlex W3101710896, 2020, sem DOI)."),
    "D15": (I, "I1–I5 OK: distributional RL aplicado a mmWave/IRS — distributional em wireless fora de slicing, mostra a dispersão da técnica (pergunta 8)."),
    "D16": (E, "E2: análise de distributional vs expected RL em jogos/ML; carve-out de citação não verificado (OpenAlex, 12 incluídos)."),
    "D17": (I, "I1–I5 OK: offline+distributional RL (estilo QR-DQN) para gestão de recursos de rádio — estado da arte quantílico em RRM (pergunta 8)."),
    "D18": (E, "E3/I2: preprint 2026 sem journal-ref e sem artefatos verificados (arXiv API nesta sessão); registrado como possível precedente do nosso ângulo de auditoria — reavaliar se publicado."),
    "D19": (E, "E3+E2: preprint sem journal-ref/artefatos (arXiv API nesta sessão); e RL fora de redes sem fio sem carve-out verificado; linhagem GP-safe representada por D03 (kernel Matérn do SQR) e B02 (KBRL)."),
    "D20": (E, "E2: revisado por pares (L4DC 2020 conforme comentário no arXiv), mas fora de redes sem fio; carve-out de citação não verificado (OpenAlex)."),
    "D21": (E, "E2: aceito no L4DC 2026 (comentário no arXiv), fora de redes sem fio; carve-out não verificado; interseção kernel×safe representada por D03/B02."),
    "D22": (E, "E2: safe RL em robótica (ICRA 2019); carve-out de citação não verificado (OpenAlex, 12 incluídos)."),
    "D23": (E, "E2: projeção de ações com garantia formal em controle/autonomia; carve-out de citação não verificado (OpenAlex)."),
    "D24": (E, "E3+E2: preprint sem journal-ref/artefatos verificados (arXiv API nesta sessão); fora de redes sem fio."),
    "D26": (E, "E3/I2: preprint 'submitted to IEEE' sem journal-ref nem artefatos (arXiv API nesta sessão); relevante ao SQR-JC — reavaliar na versão publicada."),
}

SNOWBALL = [
    {"id": "SB1", "doi": "10.2307/1913643",
     "why": "Seminal da regressão quantílica (pinball), mecanismo central do f_τ do SQR; citado pelo SQR incluído (tradução.md ref. [13], verificada contra o PDF); exceção I1 pré-2019 via snowballing reverso."},
    {"id": "SB2", "doi": "10.1109/wcnc61545.2025.10978275",
     "why": "Trabalho anterior dos mesmos autores do SQR: floresta de regressão quantílica (QRF) para eficiência energética em O-RAN — linhagem quantílica direta no O-RAN; citado pelo SQR (tradução.md ref. [4])."},
    {"id": "SB3", "doi": "10.1109/dyspan53946.2021.9677430",
     "why": "Colosseum (emulação HIL em larga escala) — plataforma da linhagem ColO-RAN/OpenRAN Gym; citado pelo SQR (tradução.md ref. [16])."},
    {"id": "SB4", "doi": "10.1109/tcomm.2024.3519550",
     "why": "RAN slicing hierárquico conjointo com escalonamento MAC sob garantia de SLA — antecedente direto de coordenação conjunta hierárquica (pergunta 8, G3/G7); citado pelo SQR (tradução.md ref. [8]; Crossref: issue 2025, online 2024)."},
]

# ------------------------------------------------------------------ execução
log_rows, matrix_rows = [], []


def q(s: str) -> str:
    s = (s or "").strip()
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n') else s


n_inc = 0
for rid in sorted(DECISIONS, key=lambda x: (x[0], int(x[1:]))):
    rec = unique[rid]
    dec, reason = DECISIONS[rid]
    if dec == I:
        reason = reason
        n_inc += 1
    dup_of = next((d for d in dup_rows if d["id"] == rid), None)
    if dup_of:
        reason = f"E4: duplicata de {dup_of['dup_of']} (mesmo DOI: {dup_of['doi'] or 'não reportado'})."
    log_rows.append({
        "id": rid, "title": rec["title"], "doi": rec["doi"] or "não reportado",
        "source": "; ".join(dict.fromkeys(rec["_sources"])),
        "decision": dec, "duplicate": "yes" if dup_of else "no",
        "reason": reason, "checked_by": CHECKER,
    })

# linhas duplicatas (E4): uma entrada no log para TODA linha candidata
def source_of(eixo: str, line: int) -> str:
    with (LIT / f"candidates_eixo{eixo}.csv").open(newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            if i == line:
                return r.get("source", "").strip()
    return "não reportado"

for d in dup_rows:
    if any(r["id"] == d["id"] for r in log_rows):
        continue
    log_rows.append({
        "id": d["id"], "title": d["title"], "doi": d["doi"] or "não reportado",
        "source": source_of(d["eixo"], d["file_line"]),
        "decision": E, "duplicate": "yes",
        "reason": f"E4: duplicata de {d['dup_of']} (mesmo DOI: {d['doi'] or 'não reportado'}).",
        "checked_by": CHECKER,
    })

# snowball: metadados via Crossref (autores/ano/venue)
def cr_work(doi):
    req = urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(doi), headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())["message"]

for sb in SNOWBALL:
    msg = cr_work(sb["doi"])
    auths = "; ".join(f"{a.get('family','')}, {a.get('given','')}".strip(", ") for a in msg.get("author", []))
    year = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
    venue = (msg.get("container-title") or ["não reportado"])[0]
    title = (msg.get("title") or ["não reportado"])[0]
    log_rows.append({
        "id": sb["id"], "title": title, "doi": sb["doi"], "source": "snowball",
        "decision": I, "duplicate": "no", "reason": sb["why"] + f" Metadados (autores/ano/venue) obtidos do registro Crossref nesta sessão. {CHECKER}",
        "checked_by": CHECKER,
    })
    matrix_rows.append({
        "title": title, "authors": auths, "year": str(year), "venue": venue,
        "doi": sb["doi"], **{k: "não reportado" for k in
                             ["problem", "architecture", "algorithm", "state_action_reward",
                              "environment", "traffic_topology", "baselines", "metrics",
                              "statistical_method", "key_results", "limitations",
                              "reproducibility", "relation_to_our_gaps", "verified"]},
    })
    time.sleep(0.3)

# matrix dos incluídos regulares
# DOIs verificados nesta sessão que não constavam no arquivo de candidatos:
DOI_OVERRIDES = {"C14": "10.48550/arxiv.2205.03278"}  # DataCite OK (screening_verify.py)
for row in log_rows:
    if row["decision"] != I or row["id"].startswith("SB"):
        continue
    rec = unique[row["id"]]
    doi = rec["doi"] or DOI_OVERRIDES.get(row["id"], "")
    if row["id"] in DOI_OVERRIDES:
        row["doi"] = DOI_OVERRIDES[row["id"]]
    matrix_rows.append({
        "title": rec["title"], "authors": rec["authors"], "year": rec["year"],
        "venue": rec["venue"], "doi": doi or "não reportado",
        **{k: "não reportado" for k in
           ["problem", "architecture", "algorithm", "state_action_reward",
            "environment", "traffic_topology", "baselines", "metrics",
            "statistical_method", "key_results", "limitations",
            "reproducibility", "relation_to_our_gaps", "verified"]},
    })

# ---------------------------------------------------------------- gravação
def sort_key(r):
    if r["id"].startswith("SB"):
        return (1, 0, 0)
    return (0, r["id"][0], int(r["id"][1:]))

log_rows.sort(key=sort_key)
with (LIT / "screening-log.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "title", "doi", "source", "decision",
                                      "duplicate", "reason", "checked_by"])
    w.writeheader()
    w.writerows(log_rows)

with (LIT / "literature-matrix.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["title", "authors", "year", "venue", "doi",
                                      "problem", "architecture", "algorithm",
                                      "state_action_reward", "environment",
                                      "traffic_topology", "baselines", "metrics",
                                      "statistical_method", "key_results",
                                      "limitations", "reproducibility",
                                      "relation_to_our_gaps", "verified"])
    w.writeheader()
    w.writerows(matrix_rows)

# ------------------------------------------------------------------- log
with (HERE / "search-log.jsonl").open("a", encoding="utf-8") as f:
    f.write(json.dumps({
        "ts": time.strftime("%Y%m%dT%H%M%S"), "source": "screening",
        "action": "screening_finalize",
        "id_scheme": "id = <Eixo><linha em candidates_eixoX.csv> (A02..D26); SB* = snowball",
        "candidates": 100, "duplicates": 8,
        "include_regular": sum(1 for r in log_rows if r["decision"] == "include" and not r["id"].startswith("SB")),
        "include_snowball": sum(1 for r in log_rows if r["id"].startswith("SB")),
        "exclude": sum(1 for r in log_rows if r["decision"] == "exclude"),
        "matrix_rows": len(matrix_rows),
    }, ensure_ascii=False) + "\n")

inc = [r for r in log_rows if r["decision"] == I]
print(f"screening-log.csv: {len(log_rows)} linhas "
      f"({len(inc)} incluídos [sendo {sum(1 for r in inc if r['id'].startswith('SB'))} snowball], "
      f"{sum(1 for r in log_rows if r['decision'] == E)} excluídos)")
print(f"literature-matrix.csv: {len(matrix_rows)} linhas de dados")
for ax in "ABCD":
    n = sum(1 for r in log_rows if r["decision"] == I and r["id"].startswith(ax))
    print(f"  eixo {ax}: {n} incluídos")
