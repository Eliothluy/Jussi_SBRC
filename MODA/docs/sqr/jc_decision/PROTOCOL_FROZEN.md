# PROTOCOL_FROZEN.md — critérios D1/D2/D3 congelados ANTES da seleção de estados

**Protocolo:** `docs/artigo/SQR_JC_Protocolo_Implementacao_Auditoria_v2.md` (§6.7) · **Congelado em:** 08/11/2026 (D0-B) pelo orquestrador, após D0-A PASS com revisões · **Regra:** nada aqui muda após a primeira execução de D2; mudança ⇒ novo protocolo.

## 0. Inputs congelados (pós D0-A — motivação, NUNCA conjunto de teste)

- Matriz 5×3 auditada (`audit/D0A_replay_5x3.csv`): A2 (48,5,0) 5/5 frozen (4/5 strict; mMTC rolling portado por histórico); **A1 (37,13,3) 3/5 frozen** (correção do "0/5"); A0 0/5. Os 15 estados-ação do D0-A ficam **EXCLUÍDOS do conjunto de teste de D3** (são motivação/treino de decisão de desenho).
- G4-B STOP FINAL (FFR 11,08% > 10%): o classificador 3-way **não** é condição de execução de B1 (§0.6).
- C=53=53RBGs=53PRBs; `ALLOCATED_USAGE` Σ≤C (27.720 ações); SLAs R36–R38 (primária URLLC p95≤10 ms/janela; rolling 1 s só em rótulos de aprendizado); R19 (vacuous ≠ sucesso).

## 1. Braços (§2.2) — inalterados

B0 = SQR-HET (`sqr_jc/sqr_het.py`: 3×SqrSliceController + projeção `project_prbs` + virtuais R12) · B1 = B0 + **JointSelector** (muda SÓ a escolha final; sem projeção corretiva pois só seleciona admissível) · B2 = oracle/replay (só diagnóstico) · B3 = relaxação (diagnóstico separado).

## 2. A_common — ESCLARECIMENTO ESTRUTURAL (consequência do D0-A)

Os pisos a_u≥13/a_m≥3 pertencem ao **desenho do plano de replay do oracle** (R50/R58 — testemunhas), **não ao controlador B0**: `SqrSliceController` propõe a_i∈[0,C] livre e a projeção só corta Σ>C. Portanto:

```text
A_common(B0) = A_common(B1) = { (a_e,a_u,a_m) ∈ Z³≥0 : Σa ≤ 53 }   (27.720 ações)
```

B3 (relaxação de pisos) fica **vazio por construção** nesta campanha — a ação (48,5,0) do D0-A É admissível a B0/B1 (Σ=53). A distinção relevante passa a ser **seleção dentro do mesmo simplex**, exatamente a H1. INV-11 satisfeito trivialmente.

## 3. D1 — instrumentação do B0 (pré-registrada)

1. **Pré-requisito (instrumentação permitida, sem campanha de otimização):** wired `--controller sqr_het` no `sqr_run.py` + log por decisão no schema §4.2 (a_raw, a_after_uncertainty_margin, a_after_projection, a_sent_to_mac, a_applied do prb_windows, projection_triggered/delta_l1, predições f_τ/σ antes e depois da projeção, demanda/filas/UEs, R19 flags) — JSONL versionado.
2. Runs D1: **mesmas flags congeladas** das campanhas (PROP/CAL): 300 s, slas 3,5, c_prbs 52 (telemetria→53), warmup 100 s, cellRadius 300, greedy/urllc/mmtc, λ∈{2,5,6} × 3 seeds (2001–2003 são PILOT — usar seeds próprias D1: {2101,2102,2103} por λ, disjuntas de tudo).
3. Estatísticas §7.2: P_projection, Σa_raw/C, ||a_raw−a_proj||₁, violação estratificada por projeção×carga×demanda×seed, matriz projeção×outcome — **sem inferência causal** em D1.
4. Bootstrap-PROP 0–100 s compartilhado (§29 herdado): B0 e B1 iniciam idênticos (mesmas seeds).

## 4. D2 — replays causais pré-registrados (orçamento exploratório)

1. **Estados:** até 60, de runs B0 (D1), ≥3 seeds independentes, λ∈{2,5,6}; estratos S1 projeção+violação, S2 violação sem projeção, S3 projeção sem violação, S4 sem-e-sem; **seleção programática com seed de amostragem pré-fixada `20261108`**; sem duplicar janelas sobrepostas; quantidades/razões de exclusão reportadas.
2. **Pré-requisito de instrumentação:** estender o replay para base = run B0 arbitrário (hoje `--replay_base {equal,prop}`); override único em t; política de continuação = **a própria sequência de ações B0 registrada** (idêntica em todos os ramos); A/A obrigatório por estado.
3. **Busca por estado (orçamento 24 alternativas no A_common + a_B0 + equal/prop se admissíveis):** (i) redistribuições pareadas ±Δ entre pares de slices (Δ∈{2,4,8,16}, determinístico); (ii) vizinhança da ação B0 (±1..4 por eixo); (iii) 6 amostras aleatórias reprodutíveis (RNG seed = hash(run_id,win) mod 2³¹); (iv) enumeração exaustiva apenas num subconjunto pré-escolhido de 5 estados (marcados EXHAUSTIVE). Linguagem obrigatória: "recuperação encontrada/não recuperada nas alternativas testadas" — nunca "impossível/ótimo".
4. **Rótulos §8.4** inalterados. **Métricas §8.5** inalteradas (unidade de inferência = seed/run).
5. **Gate D2 (pré-registrado):** GO preliminar ⇔ recuperações em A_common replicadas em **≥2 seeds independentes** com denominadores/cobertura reportados; RETHINK ⇔ recuperações só fora do simplex (impossível aqui) ou ausentes; INCONCLUSIVE ⇔ replay/R19/busca insuficientes.

## 5. D3 — JointSelector offline (ranking CONGELADO agora)

1. **Modelos:** MESMOS checkpoints B0 (SqrSliceController por slice), mesmas features z_i/normalização/τ/M/γ/memória; **nenhum** acesso a replay/oracle (INV-12); mesma política de incerteza/fallback do B0 para ações mal suportadas.
2. **g_i(s,a_i)** = saída conservadora do próprio B0: f_τ(z_i,a_i) na escala da margem (orientação correta: quantil BAIXO da margem = conservador; jamais `limiar − quantil inferior do miss`).
3. **Ranking (lexicográfico, determinístico):** 1º max `critical(a)=min_{i ativo} g_i(s,a_i)` · 2º max `efficient(a)=Σ_i utility_i(s,a_i)` (utility = o termo de custo/ganho original do B0) · 3º min Σa · 4º desempate pelo índice lexicográfico do vetor.
4. **Busca:** pré-computar g_i/utility_i por (slice, quantidade 0..53) uma vez (54×3 inferências), avaliar triplas por consulta à tabela; incluir a_B0 como candidata; orçamento de tempo medido (p50/p95/p99).
5. **Teste pareado:** estados reservados disjuntos dos usados p/ qualquer escolha (split por SEED); incluir estados B0=PASS e B0=FAIL; ambos os braços re-executados em replay controlado; UNKNOWN nunca vira PASS.
6. **Métricas:** Δ_allOK pareado (principal), RescueRate (condicional a recuperação confirmada em A_common), HarmRate, NetRescues, por slice/carga, OOD, latência.
7. **Critério D3 (GATE EXPLORATÓRIO declarado — §9.8):** avanço ⇔ (a) Δ_allOK>0 com IC bootstrap por seed não cruzando 0 fortemente contra, (b) HarmRate < RescueRate, (c) sem degradação sistemática de URLLC/mMTC nos estratos, (d) p95 latência de decisão ≤ 80 ms. Sem power estatístico p/ comprovação final — D4/D5 (bloqueados) seriam a confirmação.
8. **Proibições:** tuning pós-hoc; métrica tautológica (comparar rescues só em B0=FAIL); misturar B0 restrito × B1 livre (moot: mesmo simplex).

## 6. Vedações herdadas

Sem PILOT/MAIN (D4/D5) sem autorização explícita pós-D3; sem mudar SLA/TDD/scheduler/kernel/τ/η/memória/γ; sem tratar predição como verdade; sem contar vacuous como sucesso; sem apresentar regeneração de runs ausentes como reprodução bit-a-bit.
