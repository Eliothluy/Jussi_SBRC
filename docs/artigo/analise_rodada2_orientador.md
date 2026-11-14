# Análise do MD da Rodada 2 (orientador) × acervo atual — registro do orquestrador

**Fonte:** `docs/artigo/rodada_2_fronteira_viabilidade_sqr.md` (cópia canônica no repo; original em Downloads/Telegram). **Data:** 19/10/2026. **Contexto:** Rodada 1b concluída (24 runs; joelho λ4–5; auditoria de utilização corrigida; SNR dos violadores = não-violadores).

---

## 1. Estrutura recebida × execução imediata

O MD define 6 etapas encadeadas (2.0 estabilidade → 2.1 instrumentação → 2.2 baixa carga → 2.3 sweep heterogêneo → 2.4 oracle → 2.5 episódio transitório) + gate científico (§26: tabela carga×{violação, P_inf, AVR, |F|, severidade mín}) + regra principal (§38: não implementar SQR-JC antes de separar evitável/inevitável).

**Execução imediata desta sessão:** Etapa 2.0 (não exige código novo) — λ∈{2,5,7}×seeds 1–3×**600 s**, mesmo rádio congelado, network-only, equal-split; warm-up 0–180 s tratado na análise. Runner: `results/sqr_characterization_round2/run_stability.sh` (retomável, `.done`). Estimativa de parede ~1,5 h (λ7 600 s ≈ 80–90 min).

## 2. Mapeamento por etapa — o que já existe, o que falta

| Etapa do MD | Situação no acervo | Trabalho necessário |
|---|---|---|
| 2.0 estabilidade (§4) | ⏳ **executando** | análise pós-warm-up (script pandas) |
| 2.1 instrumentação (§5–10) | 🟡 parcial | ver §2.1 abaixo |
| 2.2 baixa carga (§11) | ⬜ | trivial após 2.1 (usa os mesmos runs do sweep em λ baixo) |
| 2.3 sweep heterogêneo (§12–16) | ⬜ | precisa 2.1 + composições A–E (§14) → knobs de chegada por slice no cenário C++ |
| 2.4 oracle (§17–26) | ⬜ | decisão de desenho — ver §2.2 abaixo |
| 2.5 episódio transitório (§27–29) | ⬜ | agenda de λ intra-run no cenário C++ (λ(t) por partes) |

### 2.1 Instrumentação (o que o KPM já tem e o que falta acrescentar)

Existente no `sqr_kpm_history.csv` (13 colunas): throughput de janela normalizado por tempo ativo, plr/dLostPackets (derivados de FlowMonitor), snrDb (sentinela −999), queueBytes, sliceId — por UE por janela. PRB por janela: requested/executed/scheduled/budget/idle + dl_slots (auditoria 1b).

Falta (contrato aditivo — colunas novas, sem quebrar as 13):
- **latency mean por UE-janela**: FlowMonitor já coleta `delaySumSec`/`rxPackets` cumulativos por flow (`sqr-sim.cc:938-965`) mas NÃO exporta; delta(delaySum)/delta(rxPackets) dá a média da janela. **Pequena alteração C++** (colunas `dDelaySumSec,dRxPackets` ou `latencyMeanMs` já resolvida no lado C++).
- **latency p95/p99 e deadline miss**: FlowMonitor só dá soma — exige **tracer por pacote** (Rx com SimTime + timestamp de geração). Custo moderado; registrar decisão: p95/p99 por janela de 100 ms têm poucos pacotes por UE (≈ 8 pacotes de 1500 B @1 Mbps) — **agregar por slice-janela** (nº maior) e/ou janela deslizante de 1 s.
- **PDR**: delta(rxPackets)/delta(txPackets) por janela — exportar contadores cumulativos resolve. Cuidado conceitual: pacotes em fila não são perdidos (PDR cai com atraso, não com drop) — para URLLC o deadline miss é o KPI correto, PDR sozinho mascara.
- **mMTC**: tráfego distinto (pacotes pequenos, periódicos/epsilónicos) — novo perfil de app no cenário C++.
- **Normalização S_i e severidade D_i (§9–10)**: lado Python (análise), sem C++.

### 2.2 Oracle (decisão de desenho — a mais importante da rodada)

O MD §23 exige comparar ações sob estados equivalentes. Nós temos **determinismo bit-a-bit comprovado** (mesma seed ⇒ mesmos arrivals/sessões/canal) — mas a ação em janelas anteriores altera o estado (filas/MC/BL), então runs com sequências de ação distintas divergem após a 1ª diferença. Opções:

- **(a) Oracle executado (por amostragem):** N runs IPC/E2 idênticos (mesma seed, controlador equal) até a janela t, aplicando a ação candidata **só em t** (controlador scriptado). Custo: 1 run por (λ, seed, t, ação) — inviável exaustivo, viável por amostragem de janelas críticas.
- **(b) Modelo offline validado (§23 permite explicitamente):** por UE, bits/RBG observados como função de (MCS, SNR) do KPM → throughput_i(a) ≈ Σ_ue min(demanda_ue, g_ue(a)) com g linear por RBG até saturação de buffer; validar contra runs equal/PF reais (a=1/3,1) já existentes. Dá F_t, |F_t| e D_min em TODAS as janelas por custo de análise.
- **Recomendação do orquestrador:** (b) como primário (cobertura total), (a) como validação em amostra de janelas (λ do joelho) — reportar as duas e a concordância.

## 3. SLAs com fonte (§8 — insumo, congelamento é decisão com o usuário)

A matriz `docs/literature/literature-matrix.csv` (60 artigos) contém valores de referência URLLC (latência 1–10 ms, confiabilidade 99,9–99,999%) e eMBB; mMTC varia por caso de uso (densidade/PDR). Antes do sweep 2.3: extrair da matriz os valores citáveis e montar a tabela §8 (KPI crítico, SLA, sentido, prioridade) **sem inventar valores** — regra da skill oran-literature-review: campo ausente = "não reportado".

## 4. Adaptações necessárias no cenário C++ (lista consolidada, ordem proposta)

1. KPM: colunas de latência média + contadores de pacotes cumulativos (delta-friendly) — pequeno, aditivo.
2. Chegadas por slice: λ_{eMBB}, λ_{URLLC}, λ_{mMTC} + perfis de app distintos (greedy eMBB; burst pequeno URLLC c/ deadline; periódico pequeno mMTC) — médio.
3. Tracer por pacote para p95/p99/deadline miss — médio (só se os SLAs escolhidos exigirem cauda).
4. λ(t) por partes intra-run p/ episódio transitório (2.5) — pequeno (mesma infra de Poisson com taxa escalonada).
5. CQI: MD pede cqi_p05/mean (§32) — o CQI wideband já é lido para snrDb (inversão); exportar o valor bruto é pequeno.

**Regra de ouro respeitada:** tudo em `scratch/sqr/` + análise Python nova; contrato aditivo (versão v1.3 do contrato de interface a emitir antes de codar); testes sintéticos → smoke → campanha (AGENTS §5.5).

## 5. Checklist do MD (§35) — estado após a Etapa 2.0 (19/10)

Estabilidade: ✅ 9/9 runs (λ2/λ5/λ7×3 seeds×600 s) — **veredicto §4.4 misto**: ordenação mantida (λ2 0,1% ≪ λ5 20,6% < λ7 40,3%); estabilidade pós-warm-up só em λ2. **Causa identificada: bursts de chegada + recuperação lenta por backlog** (fila ~3,7 MB/UE) — o regime λ5–7 é episódico, não estacionário; não é warm-up (SNR/E[N] estáveis, RLF=0) nem rádio. Recomendações aplicadas às etapas seguintes: estatística por janela/episódio com N(t) como variável de estado, violação suavizada (MM 1 s) ao lado da de 100 ms, mais seeds na transição. Relatório: `results/sqr_characterization_round2/relatorio_stability.md`. Heterogeneidade: ✅ (2.1 gates verdes: KPM v1.3 + packet delays + --sliceLambdas/--sliceApps; contrato v1.4; SLAs R36–R39; bug do alocador CBR corrigido com gate). Validação: ✅ **2.2 ENCERRADA como sanity qualificado (21/10)** — eMBB ✓ 7/7 (0,15–0,58%), mMTC ✓ PDR 1,0, URLLC 4/7 <5% com cauda mecanística 11–16% (mediana 5–6 ms; decomposição N_urllc×N_embb: monótona em N_urllc, não-monótona em N_embb — observação aberta); scheduler/deadline intocados; "F_t≠∅" → forte evidência de evitabilidade. Sweep: ⏳ 2.3 executando em 2 braços (equal estático + prop witness; Perfil A 15+15 runs). Oracle: ⬜. Transição: ⬜ (ganhou validação empírica: episódios emergem espontaneamente do churn). O gate §26 (tabela carga×P_inf×AVR×|F|×D_min) é o entregável que decide o Caso A/B/C do SQR-JC (§30).

## 6. Decisões tomadas nesta sessão (reversíveis, registradas)

- Etapa 2.0 lançada com seeds 1–3 (mesmas da 1b, comparabilidade) e warm-up 180 s **na análise** (simulação idêntica à 1b, apenas 600 s).
- Oracle: desenho (b) primário + (a) validação — a provar com dados antes de congelar.
- SLAs heterogêneos: extrair da matriz de literatura antes do sweep; nada assumido sem fonte.
- Nada de SQR-JC nesta rodada (§38).
