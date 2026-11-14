# Análise — Plano do orientador (caracterização da fronteira de viabilidade) × acervo do projeto

> Fonte: MD `caracterizacao_fronteira_viabilidade_SQR_JC.md` (33 seções, recebido 18/10/2026).
> Protocolo: caracterizar empiricamente a fronteira de viabilidade da célula ANTES de implementar SQR-JC, separando violação **evitável** (F_t≠∅, política) de **inevitável** (F_t=∅, capacidade). Gate científico: se a fronteira não existir em condições plausíveis, reformular a proposta.
> Status: **integrado ao projeto como Fase 6 do PLANO** (18/10/2026).

## 1. Mapeamento MD ↔ acervo

| Seção do MD | Status | Evidência/observação |
|---|---|---|
| §6 Fase 0 — validar reprodução do SQR | ✅ Feito | Fases 0–2 (testes 190, integração validada, R27 i–iv); coerência qualitativa demonstrada |
| §7 Fase 1 — calibração física saudável | ⏳ Metade | Sweep de raio já É a calibração: 300–800 m = RLF 0, SINR operacional, SLA atingível (0–9,7%); 2 km/30 dBm = cobertura destruída (88%). Falta congelar o cenário primário — **objeto da Rodada 1** |
| §8 — SLAs eMBB/URLLC/mMTC **com fonte** | ⬜ Novo (Rodada 2+) | Hoje 3×eMBB fiel ao artigo; SLAs heterogêneos a definir com fonte — matriz de 60 artigos (`docs/literature/`) fornece referências |
| §9–11 — L₀, varredura, Exp. A–D | ⏳ Tooling quase | `sqr-sim` já expõe `--lambdaUesPerMin` (Exp. A) e `--ueOfferedMbps` (Exp. B); runner/análise multi-seed com IC existem. Falta modo sweep e Exp. C (composição) |
| §12–14 — oracle de viabilidade | ⬜ Novo (Rodada 2+) | Peça central. Determinismo bit-a-bit comprovado (paridade IPC×E2 3585/3585) é o pré-requisito do §14; checkpoint existe. Busca exaustiva C=53/3 slices ≈ 1.431 alocações — viável via janelas curtas reproduzíveis (fallback previsto no próprio MD) |
| §15–16 — P_inf, \|F_t\|, evitável/inevitável, severidade D_i | ⬜ Novo (Rodada 2+) | Extensão do `campaign_analysis.py`; D_i já prevista no GUIA §11 |
| §17 — episódio transitório | ⬜ Novo (Rodada 2+) | Exige agenda de carga intra-run (degraus de λ/offered) — pequena adição ao sqr-sim |
| §18–19 — CSV por janela + \|F_t\| | ⏳ Parcial | `step_metrics.csv` cobre ~80% (PRBs req/exec/sched/idle, KPIs, SLA por slice); faltam `joint_feasible`, `num_feasible_actions`, rótulos evitável/inevitável |
| §20–22 — seeds, busca da fronteira, figuras 1–7 | ⏳ Padrão pronto | Protocolo multi-seed com IC já nosso; figuras = extensões do plot tooling |
| §23–25 — controles e go/no-go | ✅ Alinhado | Controles ("degradação não veio de rádio destruído?") verificáveis com sweep + transição |
| §26–27 — ordem e execução mínima | ✅ Adotado | Rodada 1 = Execução A do MD (varredura ampla, 3 seeds, sem SQR-JC) |

## 2. Observações de orquestrador

1. **A pergunta central do MD (§1/§29) tem a primeira metade respondida com dados nossos**: o regime 2 km/30 dBm do artigo-base é inviabilidade **artificial** (cobertura destruída, RLF dominante — sweep 0→88%) — o que valida a exigência do orientador e define o método correto: infraestrutura fixa e saudável + **estresse por demanda** (λ, taxa/UE), jamais por rádio destruído. A "campanha calibrada pendente" do projeto e as Fases 1/3 do MD são o mesmo experimento.
2. **O MD ressuscita o SQR-JC** (demovido a "trabalho futuro" na Fase 2 do artigo por falta de evidência própria): o protocolo é o gerador dessa evidência. Escopo a decidir: SBRC 2027 (submissão 18/01/2027) com auditoria+caracterização e SQR-JC se o tempo permitir, ou dois artigos.
3. **Lacunas concretas de infra** para Rodadas 2+: KPIs de latência/PDR no contrato de KPM (hoje throughput/PLR/buffer/SNR), agenda de carga intra-run (§17), oracle (§12), SLAs heterogêneos com fonte (§8 — matriz `docs/literature/` cobre).

## 3. Rodada 1 (definição congelada — 18/10/2026)

- **Candidato a cenário saudável**: raio 800 m (evidência prévia: RLF=0, mediana 2,8 Mbps/UE no sweep de raio).
- **Passo 1 — validação de baixa carga** (λ=2 UE/min default, greedy 100 Mbps/UE, 240 s, seeds 1–3, network-only/equal): critérios declarados ANTES de rodar: RLF total = 0; SNR operacional; violação UE-tempo com demanda ≤ 5%; mediana de throughput por UE ≥ 3,0 Mbps (2× SLA máximo). Aprovado ⇒ **congelar TODOS os parâmetros físicos** (30 dBm, 10 MHz/C=53, UMa, omni 15 dBi, TDD 3DL/7, μ=0, antenas 1×1). Reprovado ⇒ testar λ=1 como L₀ (nenhuma mudança de potência/raio/canal).
- **Passo 2 — varredura de carga (Exp. A do MD)**: taxa/UE fixa (greedy 100 Mbps), λ ∈ {1, 2, 4, 8} UE/min (≈0,5×, 1×, 2×, 4× de L₀=2) × 3 seeds × 300 s, network-only, equal-split. **Nada além de λ muda entre pontos.** Se 2,0×L₀ não atingir estresse (utilização <80% e violação <20%), ampliar λ ∈ {16, 32}.
- **Métricas por ponto** (telemetria existente): utilização (prb_scheduled/total), violação UE-tempo com demanda por slice, throughput p5/mediana por UE, filas (queueBytes p95), SNR p5/média, RLF (UE_RNTI_CHANGED), prb_executed/prb_idle.
- **Fora de escopo nesta rodada**: SQR-JC, inviabilidade conjunta (oracle), heterogeneidade.
- Entregáveis: `MODA/ns-o-ran-gym/results/sqr_characterization/` (runs + `characterization_round1.csv` + figs. 1/4/6 do MD preliminares + `relatorio_round1.md`).

## 4. Resultado da validação (Rodada 1, passo 1 — 18/10/2026)

λ=2/min, greedy 100 Mbps, 240 s, 3 seeds, network-only (violação UE-tempo c/ demanda | mediana/UE | SNR p5 | RLF):
- **800 m: 9,7/22,6/28,9% — REPROVADO** (λ=1 também: 16–22%; resíduo 14–17% mesmo com N≤2 → cauda de rádio na borda)
- **500 m: 0,1/16,9/8,5% — REPROVADO** (2/3 seeds > 5%)
- **300 m: 0,0/5,1/2,3% | 6,2–7,3 Mbps | 20,8 dB | RLF 0 — APROVADO** (maior raio que passa; resíduo do seed 2 atribuído a coincidências N≥3 — verificado na varredura)

**CONGELADO (nenhum destes muda na varredura)**: raio 300 m, 30 dBm, 10 MHz/C=53, UMa+shadowing, omni 15 dBi, TDD D|D|8D|4GB|4U|U|U, μ=0, antenas 1×1, greedy 100 Mbps/UE, sessão exp. 60 s, SLA 0,5/1,0/1,5 Mbps/UE, período 100 ms, equal-split (network-only).
**Grade da varredura** (Exp. A): λ ∈ {0,5, 1, 2, 4, 8} UE/min × seeds 1–3 × 300 s (15 runs); ampliar para 16 se λ=8 ainda não estressar.

## 5. Rodada 1b — refinamento λ∈{5,6,7} + auditoria de utilização (19/10/2026)

**Go do orientador:** refinar a transição antes da Rodada 2; adicionar p5/p10/p25 por UE e por slice; auditar a utilização de 38% (denominador, TDD, buffers, scheduler, programado vs executado). 9 runs novos (mesmo rádio congelado), 24 runs no total. Relatório completo: `MODA/ns-o-ran-gym/results/sqr_characterization/relatorio_round1b.md`.

**Auditoria (o 38% era artefato):** o `analyze_round1.py` dividia RBGs somados da janela inteira por 53 (um slot). Denominador correto = 53 × dl_slots (dl_slots = slots DL com ≥1 UE ativo, `sqr-mac-scheduler.cc:356`). Utilização correta (D2): 21/26/41/57/68/76/84/88% para λ=0,5/1/2/4/5/6/7/8 — **monotônica, sem mistério**; em λ8 a célula está 88% ocupada nos slots com demanda e a cota é consumida por inteiro (`scheduled==budget, prb_idle=0`).

**Joelho:** λ4→λ5 triplica a violação (5,6→14,0%) e derruba a mediana thr/UE 34%, mas os intervalos das seeds se sobrepõem (λ4: 0,4–8,2; λ5: 6,8–23,5) — transição **gradual a partir de λ≈4–5**, separação inequívoca em λ6+, λ8 compacto (33–37%) por saturação. **O p5 colapsa primeiro** (1,9→1,4→0,8→0,6→0,04 de λ4 a λ8): cauda é o indicador precoce.

**Quem viola:** slice dominante muda por seed (variância do churn, não propriedade do SLA); violação cresce com a fase da sessão (>10s: 40–45% em λ8) e com o terço do run (λ8: 16→32→46%, acumulação transiente); **SNR mediana dos violadores = 24,0 dB = não-violadores** — violação por demanda, não por rádio.

**Congelamento proposto p/ Rodada 2/3:** L_N=λ2 · joelho λ4–5 (marginal, ~35% do DL ocioso ⇒ evitável) · **L_O=λ7** (24%, p5=0,64, D2=84% — severo não-degenerado; λ8 vira só limite superior: p5≈0, violação majoritariamente inevitável). Pergunta do oracle: violação evitável (F_t≠∅) em λ4–5 vs inevitável (F_t=∅) em λ7–8?

**Incidente operacional:** dois runners coexistiram na troca JOBS=3→9 e corromperam 3 runs lam7 (repetidos e validados do zero); lição registrada no relatório (setsid devolve PID do wrapper; nunca 2 runners no mesmo sweep).
