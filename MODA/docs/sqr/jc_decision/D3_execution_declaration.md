# D3_execution_declaration.md — conformidade com os 4 cuidados do orientador

**Data:** 11/11/2026 · **Base:** PROTOCOL_FROZEN §5 + §9 (ranking congelado; gate exploratório declarado) · **Nada executado ainda além da leitura de estrutura.**

## Cuidado 1 — Modelos congelados: CUMPRIDO por construção

O JointSelector **carrega os checkpoints SQR-HET de cada run D1** (mesmos `SqrSliceController` com o aprendizado online da própria trajetória). **Nenhum retreino, nenhum ajuste, nenhuma informação da D2 entra nos modelos** (os resultados D2 são rótulos de avaliação, nunca features — INV-12). Verificação executável: hash dos checkpoints antes/depois da execução D3 (devem ser idênticos — o seletor roda em modo freeze/leitura).

## Cuidado 2 — Independência do conjunto de teste: split POR RUN (mais forte que ±5 janelas)

A exclusão D2 ±5 janelas permanece, mas o ponto do orientador é correto e vai além: janelas vizinhas de um mesmo run são autocorrelacionadas (mesma trajetória, mesmo aprendizado online). **Decisão (mais estrita que o congelado):**

- **Unidade de split = RUN inteiro.** Conjunto de TESTE = os 3 runs da seed **2101** (λ2/λ5/λ6) — nenhuma janela de um run de teste participa de qualquer decisão de desenho, ranking-calibração ou estatística de desenvolvimento. Conjunto de DESENVOLVIMENTO (apenas para sanity/plumbing do seletor, sem ajuste de nada — ranking já congelado) = seeds 2102/2103, usados também para relatório secundário.
- **Transparência estrutural (limitação inerente, declarada):** os modelos B0 de cada run aprenderam ONLINE na própria trajetória do run (protocolo SQR original — não há "treino separado"). É impossível avaliar um checkpoint que não viu a trajetória sem retreinar (proibido pelo cuidado 1). **Mitigação pareada:** B0 e B1 são avaliados nos MESMOS estados com o MESMO checkpoint — todo aprendizado-na-trajetória afeta os dois braços igualmente; o pareamento Δ_allOK isola a DIFERENÇA de seleção. O viés residual possível é sobre a magnitude absoluta das taxas (não sobre o sinal do Δ pareado). Declarado no relatório como condição do método SQR online.
- Sobreposição de chegadas/tráfego entre runs: mesmos λ e configuração, seeds diferentes ⇒ processos de Poisson/canal independentes por seed (construção do cenário); não há partilha de traços exógenos entre runs de seeds distintas.

## Cuidado 3 — Mesmo espaço admissível: CUMPRIDO (A_common idêntico)

B0 e B1 operam sobre `A_common = {(a_e,a_u,a_m) ∈ Z³≥0 : Σa ≤ 53}` (27.720 ações; PROTOCOL_FROZEN §2). B1 não aplica projeção corretiva (só seleciona admissível); B0 mantém sua arquitetura (margem + projeção) **intacta** — a única diferença entre braços é a regra de seleção final. Mesmos checkpoints, features z_i, normalização, τ/M/γ/memória, bootstrap-PROP, flags de campanha.

## Cuidado 4 — STRICT primário; seletor cego a rótulos: CUMPRIDO

- Comparação primária: **Δ_allOK_STRICT = P(ALL3_COMPLETE | B1) − P(ALL3_COMPLETE | B0)** nos mesmos estados; UNKNOWN (não-3×COMPLETE) reportado separadamente e nunca convertido em PASS/FAIL; ausência de recuperação no conjunto testado ≠ inviabilidade.
- O JointSelector **não lê rótulos de replay/oracle em momento algum**: consome apenas (i) os modelos congelados do próprio run, (ii) o estado pré-decisão z_i, (iii) o A_common. Ranking lexicográfico congelado §5.4: max critical(a)=min_i g_i(a_i) → max efficient(a)=Σ utility_i → min Σa → índice lexicográfico. g_i = f_τ da margem (orientação correta verificada no D0-B).

## Plano de execução (pré-registrado; orçamento)

1. `joint_selector.py`: tabela g_i/utility_i por (slice, quantidade 0..53) pré-computada por checkpoint (54×3 inferências/run), ranking lexicográfico com desempate determinístico; traço de decisão completo.
2. Decisões pareadas OFFLINE em TODAS as janelas STRICT-avaliáveis dos runs de teste seed 2101 (≈4.400 janelas × 3 λ) — B0 (ação registrada) × B1 (ação do seletor), sem simulação nesta fase.
3. **Replays confirmatórios pareados** (B0 já existe = o próprio run; 1 replay por estado para a ação B1) numa amostra pré-registrada: TODAS as janelas onde B0 STRICT_FAIL ∧ B1 propõe ação ≠ a_B0 (recuperadoras potenciais) + amostra aleatória estratificada de 60 estados-controle (B0 pass/B1 diferente; B0 pass/B1 igual não precisa replay) — orçamento ≈ 200–400 replays (~1–2 dias paralelizado ×3).
4. Métricas: Δ_allOK_STRICT pareado (principal), RescueRate/HarmRate/NetRescues, por λ/seed, OOD, latência de decisão do seletor.
5. Gate exploratório (congelado §5.7): (a) Δ_allOK>0, (b) Harm<Rescue, (c) sem degradação sistemática URLLC/mMTC, (d) p95 ≤80 ms.
6. `GO_NO_GO.md` final: recomendação única GO-JOINT/STOP-JOINT/INCONCLUSIVE; D4/PILOT e D5/MAIN permanecem bloqueados.
