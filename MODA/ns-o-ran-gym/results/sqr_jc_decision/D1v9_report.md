# D1v9_report.md — D1-v9 canônica (Emenda 3 Opção B + Emenda 4 Delayed Feedback)

**Data:** 12/11/2026 · **Execução:** 10:48–12:23 (9/9 runs, flags congeladas do run_d1 + `--urllc_delayed_update`) · **D1-v8 preservada** em `d1_runs/` (versão anterior, não misturada) · **Vedados intactos:** regressão quantílica, SLAs, R19, espaço de ações.

## 1. Indicadores por λ×seed (fonte única: `d1_runs_v9_delayed/`)

| run | bootstrap | B0 | B0 c/outcome | P_proj | upd eMBB (s1) | upd mMTC (s1) | **upd URLLC (s2)** | age med | expired |
|---|---|---|---|---|---|---|---|---|---|
| λ2s2101 | 826 | 1.801 | 1.800 | 10,4% | 2.415 | 1.610 | 0 (sem demanda URLLC) | — | — |
| λ2s2102 | 283 | 565 | 564 | 14,2% | 164 | 390 | **309** | 2 | 3 |
| λ2s2103 | 859 | 844 | 843 | 12,6% | 674 | 350 | **1.154** | 2 | 2 |
| λ5s2101 | 928 | 1.995 | 1.994 | 16,8% | 2.921 | 1.695 | **530** | 2 | 2 |
| λ5s2102 | 303 | 1.995 | 1.994 | 16,0% | 1.582 | 1.944 | **1.159** | 2 | 3 |
| λ5s2103 | 975 | 1.948 | 1.947 | 16,7% | 2.167 | 2.002 | **2.709** | 2 | 5 |
| λ6s2101 | 940 | 1.995 | 1.994 | 15,5% | 2.933 | 2.051 | **920** | 2 | 2 |
| λ6s2102 | 460 | 1.995 | 1.994 | 16,0% | 1.845 | 2.084 | **1.569** | 2 | 15 |
| λ6s2103 | 979 | 1.995 | 1.994 | 15,4% | 2.416 | 2.113 | **2.899** | 2 | 5 |
| **TOTAL** | **6.553** | **15.133** | **15.124** | — | — | — | **11.249** | 2 | 37 |

- Universo de decisões idêntico ao v8 (6.553+15.133; bootstrap-PROP determinístico por seed ⇒ fase 0–100 s igual).
- **P_projection 14,2–16,8%** (v8: 10,4–13,5%) — as decisões B0 divergem após os primeiros updates URLLC (esperado; a telemetria física por seed é determinística, as ações não).
- Latência de atualização: age mediana **2 janelas** (200 ms) em todos os runs; expirados (INCOMPLETE/vacuous R19): 37 no total.

## 2. Auditoria de integridade (9/9 runs)

| critério | resultado |
|---|---|
| updates URLLC efetivos | **11.249** (v8: 0) |
| duplicatas de janela | **0** |
| associação estado-ação-outcome | **11.249/11.249** com decisão correspondente |
| vazamento temporal (entrega > janela de observação) | **0** |
| p95 do update ≠ p95 offline completo | **0/11.249** |
| eMBB/mMTC estágio 1 | inalterados (mesma contagem de padrão do v8) |
| gates de completude | 9/9 (KPM ≥290.000, ≥600 decisões, exit 0) |

## 3. Viés de observabilidade (quantificado na Emenda 4 §3, dados v8)

Janelas atualizáveis (pool fecha) vs excluídas (perda real): buffer mediano **69 vs 19.235 B**; resourceShare **24,5% vs 0%**; miss offline **1,5% vs 100%**. O modelo URLLC aprende do regime não-congestionado; janelas de perda permanecem vacuous (R19 correto; sem rótulo inventado). Paridade B0×B1 simétrica (mesmo mecanismo). Categorias NO_ACTIVE_DEMAND e INCOMPLETE/UNKNOWN reportadas separadamente.

## 4. População/amostragem D2 reconstruídas da D1-v9

- `D2_sampling_FINAL_V9.csv` + `D2_sampling_manifest_V9.json` (hashes pré-registrados; CSV sha16 `0e64e507b4969548`): 30 estados × 13 = **390 replays**, estratos S1:12/S2:9/S3:6/S4:3, substituições determinísticas 29, 0 duplicatas, λ balanceado (130/130/130), **seed de amostragem 20261108 e orçamento inalterados** (critérios congelados das Emendas 1–2).
- Estados da D1-v8 **não reutilizados**; replays parciais antigos preservados em `d2v2_pre_audit_pause/` (histórico).

## 5. Recomendação: **GO para D2-v2 (população V9)**

A Emenda 4 remove a limitação "URLLC online cego" sem vazamento (auditoria §2), preserva todos os vedados e a hipótese científica (seleção conjunta vs independente com modelos/espaço equivalentes). O viés de censoring físico-real está quantificado e declarado, com tratamento R19 correto. **D2-v2 liberada para execução mediante avaliação do orientador**; D3/PILOT/MAIN permanecem bloqueados.

---

# ADENDO 13/11 — complementação de auditoria (ofício pós-Emenda 4) — parecer final

## A1. Reconciliação dos updates URLLC v9 (por λ×seed)

Universo: **11.335 janelas com demanda URLLC** (KPM tx>0) = **11.249 updates + 37 expired (INCOMPLETE/vacuous R19) + 16 pendentes-no-fim + 33 com demanda mas sem decisão correspondente** (janelas de gap sob hold). Identidade `elegíveis = updates + expired + pendentes` fecha em **9/9 runs**; NO_ACTIVE_DEMAND = 0 eventos e NO_KPM = 0 (a fila só recebe pools abertos; bordas sem KPM nunca enfileiram).

| run | demanda | elegível | updates | expired | pend.fim | sem decisão |
|---|---|---|---|---|---|---|
| λ2s2101 | 0 | 0 | 0 | 0 | 0 | 0 |
| λ2s2102 | 314 | 312 | 309 | 3 | 0 | 2 |
| λ2s2103 | 1.164 | 1.158 | 1.154 | 2 | 2 | 6 |
| λ5s2101 | 532 | 532 | 530 | 2 | 0 | 0 |
| λ5s2102 | 1.169 | 1.164 | 1.159 | 3 | 2 | 5 |
| λ5s2103 | 2.722 | 2.716 | 2.709 | 5 | 2 | 6 |
| λ6s2101 | 928 | 924 | 920 | 2 | 2 | 4 |
| λ6s2102 | 1.595 | 1.590 | 1.569 | 15 | 6 | 5 |
| λ6s2103 | 2.911 | 2.906 | 2.899 | 5 | 2 | 5 |

**O aumento vs a estimativa da v8 (59,8%) é legítimo e tem causa física documentada:** (i) na v8 o runtime perguntava o outcome 1× e descartava por R19 mesmo com pool fechável (0 updates por desenho da pergunta); a Emenda 4 passa a **utilizar** os pools que fecham; (ii) o TX é idêntico entre v8/v9 (carga determinística por seed), mas o **RX v9 quase iguala o TX** — o B0 treinado aloca URLLC adequadamente e a perda (semRX%) cai de 32–49% (v8) para 0–1% (v9):

| run | semRX% v8 | semRX% v9 |
|---|---|---|
| λ2s2102 / s2103 | 0,3 / 14,3 | 0,03 / 0,00 |
| λ5s2101 / s2102 / s2103 | 45,2 / 44,0 / 32,4 | 0,03 / 0,01 / 0,00 |
| λ6s2101 / s2102 / s2103 | 43,2 / 48,6 / 35,4 | 0,25 / 1,05 / 0,00 |

Fechamento prematuro descartado: o p95 de cada update é **idêntico ao offline completo** do pool (11.249/11.249). Nota de transparência: a fila atualiza tanto janelas do **bootstrap-PROP** quanto do B0 — consistente com o protocolo (o learner sempre observou/aprendeu eMBB/mMTC durante o bootstrap; agora aprende URLLC pelo mesmo fluxo).

**Viés de seleção (dados v9):** com a perda ~0, apenas **86/11.335 (0,8%)** janelas com demanda ficam sem update — todas no regime de congestionamento severo (buffer p95 14.143 B; miss p95 100%; p95 p95 268,5 ms), concentradas em λ6 (46) e λ5 (25). As atualizadas incluem janelas com cauda (miss p95 44%; share p95 61,6%) — o modelo não aprende só o regime saudável. Viés residual declarado: janelas de perda total nunca alimentam o modelo (R19; sem rótulo inventado); paridade B0×B1 simétrica.

## A2. Indicadores de SLA consolidados (offline completo, arquivo íntegro v9)

| run | j eMBB | viol eMBB% | txU→rxU | semRX% | miss med% | p95 janela med | p95 run | j mMTC | PDR | roll viol% | S PASS | S FAIL | S UNK | cob% | e∧u P/F/U |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| λ2s2101 | 2.420 | 0,7 | 0→0 | — | — | — | — | 162 | 0,512 | 7,4 | 0 | 0 | 1.799 | 0 | 0/0/1.799 |
| λ2s2102 | 165 | 1,8 | 31.084→31.074 | 0,03 | 0,0 | 8,1 | 8,1 | 40 | 1,000 | 0,0 | 0 | 0 | 563 | 0 | 26/6/531 |
| λ2s2103 | 675 | 43,6 | 116.100→116.095 | 0,00 | 0,0 | 8,1 | 8,1 | 35 | 0,000 | 2,9 | 0 | 0 | 842 | 0 | 54/79/709 |
| λ5s2101 | 2.926 | 13,2 | 79.102→79.082 | 0,03 | 18,0 | 14,1 | 38,1 | 286 | 0,887 | 6,3 | 0 | 0 | 1.994 | 0 | 176/355/1.463 |
| λ5s2102 | 1.587 | 64,4 | 149.884→149.871 | 0,01 | 0,0 | 8,1 | 8,1 | 426 | 0,884 | 8,2 | 215 | 573 | 1.206 | 39,5 | 216/573/1.205 |
| λ5s2103 | 2.171 | 27,8 | 329.443→329.432 | 0,00 | 4,0 | 9,1 | 12,1 | 217 | 0,922 | 6,9 | 312 | 1.167 | 467 | 76,0 | 315/1.177/454 |
| λ6s2101 | 2.938 | 28,9 | 163.794→163.387 | 0,25 | 5,5 | 10,0 | 54,1 | 353 | 0,917 | 6,5 | 225 | 343 | 1.426 | 28,5 | 332/592/1.070 |
| λ6s2102 | 1.850 | 50,1 | 246.248→243.664 | 1,05 | 1,0 | 8,1 | 57,1 | 448 | 0,922 | 5,8 | 419 | 776 | 799 | 59,9 | 427/800/767 |
| λ6s2103 | 2.420 | 40,8 | 376.427→376.410 | 0,00 | 2,0 | 8,1 | 11,1 | 240 | 0,792 | 10,8 | 443 | 1.177 | 374 | 81,2 | 461/1.298/235 |

Convenções congeladas (§8/Emenda 3): eMBB = mediana windowThroughputMbps ≥3,5 sobre UE-windows com demanda; URLLC = p95 ≤10 ms por janela de chegada (pool completo do arquivo); mMTC = rolling PDR 1 s ≥0,95. STRICT_ALL3 = 3×demanda + all-OK (UNKNOWN quando qualquer slice sem demanda — cobertura reportada); e∧u complementar pré-registrada; janelas incompletas NUNCA contam como sucesso; fontes exclusivamente v9. Artefatos: `d1v9_sla_offline.csv`, `d1v9_strict_by_window.csv`, `d1v9_reconciliation.csv`.

## A3. Validação da amostragem D2-V9

- **Hashes: 12/12 OK** (`D2_sampling_FINAL_V9.csv` `0e64e507b4969548`; gerador; Emenda 1; os 9 `decisions.jsonl` canônicos v9) — nada mudou após a assinatura.
- **390 linhas / 30 estados; 0 duplicatas** (run,win,alt); **0 estados sem decisão v9 correspondente; 0 alternativas inadmissíveis** (Σa≤53, a≥0); `a_B0` das linhas idêntico às decisões canônicas (0 divergências).
- Distribuição: estratos S1:12/S2:9/S3:6/S4:3; λ 10/10/10; seeds 11/10/9.
- **12/30 estados com evidência STRICT completa (3×demanda)**, distribuídos 4/4/3/1 pelos estratos; os demais 18 avaliam por e∧u/parcial conforme o pré-registro (prevalência de 3-slices rara é conhecida).
- Mecanismos: a_B0 30, ref 30, paired 108, nbh 73, rnd 120 (+29 substituições determinísticas registradas na coluna `substituted`).
- Replays parciais da D1-v8 preservados separadamente (`d2v2_pre_audit_pause/`) — não reutilizados.

## A4. Parecer final: **GO para a D2-v2**

Todos os indicadores reconciliados sem inconsistências: identidade de contabilidade fecha em 9/9; 0 duplicações/vazamentos/prematuros; SLA consolidado com denominadores explícitos; amostra íntegra e correspondente às execuções canônicas. O ganho do controlador (perda URLLC 32–49%→0–1%) é consequência legítima da Emenda 4 sobre a mesma carga física determinística (TX idêntico v8×v9) — e não altera a pergunta científica: a seleção conjunta continua sendo avaliada contra um baseline mais forte, com modelos/espaço equivalentes e seleção como única diferença. O delayed feedback permanece **correção metodológica compartilhada**, não contribuição. **D2-v2 apta a prosseguir sem novas alterações de protocolo; D3, PILOT e MAIN continuam bloqueados.**
