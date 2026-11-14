# GO / NO-GO — SQR Joint Selection (protocolo v2, 08/11/2026)

## Integridade — D0-A
- Resultado D0-A (PASS/STOP/INCONCLUSIVE): **PASS com resultados científicos revisados** (`audit/D0A_GATE.json`; §5.11 — a integridade dos dados/replays prevaleceu e os labels foram corrigidos honestamente).
- 5×3 linhas auditadas e sustentadas em traces: **15/15** recomputadas com funções congeladas; A/A offline **10/10** (prefixos KPM e delays bit-idênticos); ações executadas = planejadas 15/15.
- 5/5 revisado: **frozen 5/5 confirmado; strict ALL3_COMPLETE = 4/5 + 1 UNKNOWN** (mMTC rolling portado por histórico pré-t; tx in-window = 0 nas 15 células). O "0/5" do in-suporte (37,13,3) era **artefato de métrica do stage1** (eMBB por thr_min + URLLC por miss + mMTC 100 ms + linhas replay_valid=False usadas sem filtro) → **corrigido para 3/5 frozen / 2/5 strict**.
- mMTC a_m=0: evidence, denominator, status: rolling 1 s (congelado R38) = PASS com demanda pré-t; in-window = NO_ACTIVE_DEMAND/UNKNOWN ×5; evidência positiva oposta: t1423 PDR100 = 0; D6 a_m_min = 1.
- R19 total/completas/não completas e denominadores: **24.482/12.075/12.407 exato (50,68%)**; λ2 85,6 / λ5 48,2 / λ6 25,1% — parecer correto; legenda da galeria corrigida.
- Unidade C e rastreio Python→MAC: **C = 53 = 53 RBGs = 53 PRBs** (`NumRbPerRbg=1`; cadeia exata `dedicatedPRB→ComputeRbgQuotas`; telemetria 53/53; narrativa R01 a emendar).
- Divergências de figuras e dados: fig2 **revisada por mudança de dado** (linha A1) em `docs/artigo/figuras_tese/audit_v2/` (original preservado); fig4/6/7 clipping cosmético corrigido; legenda "λ2 ~100%" corrigida para 85,6%.
- Inventário de evidência ausente: **nenhuma** (52/52 itens localizados e hasheados).

## Integridade — D0-B
- Commit e cenário: b9f3c0e + não-commitados inventariados; cenário reconstruído dos runners congelados (flags verificadas).
- Hash dos manifests: `D0A_input_inventory.csv`.
- D0-B: **PASS** (`D0B_baseline_audit.md`) — SQR-HET fiel ao Algoritmo 1 (import do maquinário auditado; Λ=0 na margem comum; projeção+virtuais R12); orientação URLLC correta por construção (quantil baixo da margem); PROTOCOL_FROZEN congelado antes da seleção de estados (com A_common = simplex completo — pisos eram do oracle, não do B0).
- Replays A/A aprovados / total: **offline 10/10 + fresco 1/1 bit-idêntico** (3 CSVs, 442k linhas) sobre o acervo 38× II-b + 813/813 piloto.
- Capacidade e evidência R19: rastreadas (C=53=53RBGs=53PRBs; errata sqr_env aplicada).
- Gap de instrumentação (pré-requisito D1): `--controller sqr_het` a wired + log de decisão schema §4.2; replay base B0 arbitrário (pré-requisito D2).

## D1 — seleção/projeão (PASS v2 CANÔNICA — 10/11, bootstrap-PROP §3.4)
- 26.955 = 9×2.995 decisões; 9 end_of_run sem outcome (última janela além do sim) ⇒ 26.946/26.946 outcomes; fases: 9.000 PROP-bootstrap + 17.955 B0 (17.946 com outcome).
- P_projection 6,29% (1.129 eventos; corte med 9 RBGs; Σ/C p95 1,06); 43,0% das projeções degradam min_i f_τ (Q2).
- Denominadores por λ declarados (eMBB obs 2.435/5.091/5.433, viol 73/77/79%; URLLC viol 2,4–3,5%; mMTC 3–11%); sent=applied 17.946/17.946.
- Estratos D2: S1 563 · S2 9.710 · S3 566 · S4 7.107 (all3-completas 441/7.480/231/1.813).
- Cold-start preservado exploratório (d1_runs_coldstart_exploratory; inversão da associação proj×outcome documentada).
- Emenda 1 D2 CONGELADA antes da execução (30×12; cobertura fixa do simplex; modelos de pré-screening hasheados sem vazamento); proposta detalhada: results/sqr_jc_decision/D2_sampling_proposal.md — AGUARDA GATE DO ORIENTADOR para executar D2 (~390 replays, 20–35 h).

## D1 — seleção/projeção (v1 cold-start — EXPLORATÓRIA, 09/11)
- N decisões válidas: **26.955 decisões / 26.946 com outcome (100%)** — 9/9 runs B0 (SQR-HET) λ{2,5,6}×seeds 2101–2103, decisions.jsonl schema §4.2.
- P(projeção) por carga/seed: **3,87%** global (1,5–6,3% por seed; 1.043 eventos); Σ/C p95 0,98; ||Δ||₁ mediana 7 RBGs.
- Mudanças previstas após projeção: **41,3%** das decisões com projeção degradam min_i f_τ (Q2 diagnosticamente ativo).
- Associação com violação (não causal): sem projeção 55,3% all-OK vs com projeção 39,8% (frozen); eMBB 84,6% / URLLC 2,4% / mMTC 12,1% de violação entre observadas; sent=applied 26.946/26.946.
- Incidentes: D1-v1 (parser posicional → 8 runs quarentena) + D1-v2 (path do stream) — corrigidos com testes de regressão (5 passed); runs v2 íntegros.
- Desvio registrado: cold-start (§3.4 previa bootstrap-PROP) — **decisão pendente do orientador** (implementar bootstrap + re-rodar ~2h × emenda para cold-start).
- Custo D2: orçamento congelado ≈ >100 h de parede — **emenda proposta**: 30 estados × 12 alternativas com pré-screening por modelo + confirmação por replay.
- Gate D1: **PASS** (D2 autorizado metodologicamente; execução aguarda as 2 decisões acima).

## D2 — oportunidade causal
- (pendente — pré-registrado em PROTOCOL_FROZEN §4; 60 estados, ≥3 seeds, busca 24+alternativas, A/A por estado)

## D2 — oportunidade causal (PASS — 11/11)
- N estados elegíveis/testados/excluídos: 30/30 testados, 0 excluídos (1 re-executado sequencialmente após artefato pré-override do paralelismo; 1 A/A perdido no sharding executado à parte — incidentes registrados no relatório).
- N estados recuperáveis dentro de A_common: **3 RECOVERABLE_STRICT** (S1λ5s2102t1706, S1λ5s2103t1974, S1λ6s2102t1420) — STRICT primário (ajuste 2); +2 B0_ALREADY_PASS +1 B0_UNKNOWN_ALTS_PASS +24 NOT_RECOVERED.
- Seeds/episódios representados: recuperação em **≥2 seeds independentes (2102, 2103)** ⇒ **gate D2 PASS (GO científico preliminar)**.
- N recuperáveis somente em A_relaxed: n/a (B3 vazio por construção — A_common = simplex; PROTOCOL_FROZEN §2).
- Cobertura da busca: 12 alternativas/estado de composição fixa (rnd 10 · paired 6 · nbh 4 · equal 4 · sub 3 STRICT_PASS nos 4 estados com alts-pass); busca parcial — NOT_RECOVERED ≠ impossível.
- Falhas/limitações: fração amostrada ≤5,4% por célula (existência, não prevalência); horizonte 1 janela; mMTC strict raramente avaliável in-window.
- 27 avaliações STRICT_PASS; igual (18,17,18) recupera 1 estado violado pela seleção do B0.

## D3 — seletor automático
- (pendente — ranking congelado em PROTOCOL_FROZEN §5; gate EXPLORATÓRIO declarado; conjunto de teste: janelas B0 excl. estados D2 ±5 janelas, seed 20261109; AGUARDA autorização do orientador)

## Decisão
- [ ] GO-JOINT (autorizar pedido de PILOT) — D2 PASS (GO preliminar); D3 pendente de autorização
- [ ] STOP-JOINT (não continuar esta hipótese)
- [ ] INCONCLUSIVE (indicar qual evidência falta)
- Justificativa objetiva: (parcial) a revisão D0-A (A1 3/5 in-suporte frozen) fortalece a motivação da H1 — oportunidades de recuperação existem no MESMO espaço admissível; Q1/Q2/Q3/Q4 ainda não testadas.
- Afirmações que os dados AINDA NÃO permitem: superioridade da seleção conjunta (Q3/Q4); "só fora dos pisos há recuperação" (refutada em D0-A); recuperação causal do mMTC sob a_m=0 (sem evidência in-window).
- Próximo passo permitido: **D3** mediante autorização do orientador (D2 PASS em 11/11; ranking do seletor e isolamento já congelados). D4/PILOT e D5/MAIN permanecem bloqueados.
