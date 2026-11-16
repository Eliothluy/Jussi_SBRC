# A5 — Especificação Matemática Revisada, Implementação Modular e Plano de Testes (`SQR-JC` / Pré-`D3-v2`)

**Status:** **PREPARAÇÃO E IMPLEMENTAÇÃO MODULAR CONCLUÍDAS** ([`a5_joint_selector.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/a5_joint_selector.py) + [`test_sqr_jc_a5_selector.py`](../../../ns-o-ran-gym/tests/test_sqr_jc_a5_selector.py)). **ZERO alterações em `B0-rest`** ([`B0_FROZEN_MANIFEST.json`](B0_FROZEN_MANIFEST.json), congelado no commit `8375f4da3713006449036ace09b7c0579e38fade`) e **ZERO alterações em [`joint_selector.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/joint_selector.py)**.  
**Bloqueios ativos:** Campanha **`D3-v2` confirmatória, `PILOT`, `MAIN`, `A2b`, `A3b` e *seeds* independentes `3101–3103` permanecem ESTRITAMENTE BLOQUEADOS** aguardando autorização explícita.

---

## 1. Formulação Matemática Final das Políticas (`B1-raw`, `B1-active`, `B1-support`)

Todas as variantes alteram **exclusivamente a regra de seleção** sobre os três modelos quantílicos congelados de **`B0-rest`** (`A1+A2a+A_scaler`), sem retreinar pesos, sem alterar `R10`/`R11`/`R12`, sem alterar SLAs ou regras `R19`, e sem consumir números aleatórios (*RNG = 0*).

### 1.1 Estado Causal Disponível no Instante $W$

No instante decisório $W$ (após fechar a janela KPM $W$ e **antes** de aplicar a alocação $a(W) = (a_0, a_1, a_2)$ que governará a janela $W+1$), o seletor observa exclusivamente:

1. **Estado físico pré-decisão $S_W$ (`JointState`):** contagem de UEs ativos $n_i(W) \in \mathbb{N}_0$, filas $q_i(W) \in \mathbb{R}_{\ge 0}$, demanda oferecida $\text{off}_i(W) \in \mathbb{R}_{\ge 0}$, pacotes transmitidos na janela fechada $\text{tx}_i(W) \in \mathbb{N}_0$ (`extras["kpm_tx_packets"]`), flags causais de trabalho pendente $\text{pending}_i(W) \in \{0,1\}$, KPIs prévios fechados, SNR média, alocação prévia executada $a_i(W-1)$, indicador recente de violação $v_i(W) \in \{0,1\}$ e capacidade efetiva $C \in \mathbb{Z}_+$ (`totalRbg = 53`). **Nenhum rótulo ou outcome de $W+1$ (`outcome(W+1)`, `STRICT_ALL3_R19`, `UNKNOWN`) é acessível.**
2. **Representação normalizada congelada:** $z_i(W) = \text{slice\_state}(S_W, i) \in [0,1]^6$ via `SqrHetController` congelado (`fix_scaler_index=True` $\to$ `"eum"`, `mean_field=False`, `empty_slice_guard=True`, `virtual_alpha=0.0`).
3. **Tabelas de predição e incerteza congeladas** para todo $i \in \{0,1,2\}$ e candidato $a \in \{0,\dots,C\}$ com entrada $x_{i,a} = [z_i(W),\, a/C] \in [0,1]^7$:
   $$f_i(a) := \hat{f}_{\tau_i(W)}(x_{i,a}) = \sum_{j=1}^{B_i(W)} \alpha_{i,j}\, k_i(x_{i,a}, x_{i,j}), \qquad u_i(a) := f_i(a) - \gamma \frac{a}{C} \quad (\gamma = 1),$$
   $$\sigma_i(a) := 1 - \max_{j=1..B_i(W)} k_i(x_{i,a}, x_{i,j}) \in [0,1] \quad (\sigma_i(a)=1 \text{ se } B_i(W)=0),$$
   além da ação determinística $k$-NN congelada $a_i^{\text{knn}}(W) \in \{0,\dots,C\}$ (`SqrSliceController.k_nn_action`, sem RNG), capturadas de modo imutável em `CausalTraceWriter` ou via avaliação *read-only* sobre checkpoint com horizonte estritamente $< W$ (`verify_causal_replay_no_future_checkpoint`).

### 1.2 Normalização e Comparabilidade entre as Margens dos Três SLAs

Por construção de [`sla_margin.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/sla_margin.py) (`sla_manifest.yaml` `v1.1-g4b`, $\Lambda = 0$), todas as predições $f_i(a)$ estão na mesma escala relativa adimensional:
$$m_0 = \frac{T_{\text{eMBB}} - 3{,}5}{3{,}5}, \qquad m_1 = \frac{10{,}0 - D_{\text{URLLC}, p_{95}}}{10{,}0}, \qquad m_2 = \frac{\text{PDR}_{\text{mMTC}, 1\text{s}} - 0{,}95}{0{,}95}.$$
Logo, $m_i > 0 \iff \text{SLA}_i$ atendido ($m_i = 0$ na fronteira $\Lambda=0$), tornando $\min_i g_i(a_i)$ comparável entre fatias heterogêneas.

**Causa matemática da degenerescência sob $\Lambda=0$ sem controle de suporte (`D3-v1`):** como $k_i(x, x_j) \to 0$ longe do suporte observado, a regressão reverte para $f_i(a) \to 0$. Em alta contenção, onde margens reais observadas são tipicamente negativas ($\text{med } f_i \approx -0{,}25$ a $-0{,}78$), valores extrapolados próximos a $0$ aparentam ser superiores aos estados suportados negativos ($0 > -0{,}25$), levando seletores sem controle de incerteza a colapsar para extremos espúrios como $(0,0,0)$ ou $(0,53,0)$.

### 1.3 Tratamento de Demanda Nula e Observações Pendentes

Adota-se **exatamente** o predicado causal congelado de `B0-rest` (`SqrHetController.is_slice_provably_empty`, [`causal_active_slices`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/a5_joint_selector.py)):
$$\text{Empty}_i(W) \iff \big(n_i(W)=0\big) \land \big(q_i(W) \le 0\big) \land \big(\text{off}_i(W) \le 0\big) \land \big(\text{tx}_i(W) \le 0\big) \land \neg\text{pending}_i(W),$$
onde:
- $\text{pending}_0(W) = \text{embb\_pending\_pipeline}$;
- $\text{pending}_1(W) = \text{urllc\_pool\_open}$ (pacotes em voo do *pool* URLLC ainda aberto online no instante $W$);
- $\text{pending}_2(W) = (\text{kpi\_m\_raw} \neq \text{None}) \lor \text{mmtc\_rolling\_active}$ (janela deslizante de 1 s ainda ativa para mMTC).

Define-se o conjunto ativo causal $\mathcal{I}_{\text{act}}(W) = \{ i \in \{0,1,2\} : \neg \text{Empty}_i(W) \}$ e o subconjunto com carga viva $\mathcal{I}_{\text{live}}(W) = \{ i \in \{0,1,2\} : n_i(W) > 0 \lor q_i(W) > 0 \}$.

### 1.4 Restrições de Capacidade e Ações Admissíveis

- **Espaço comum completo (`B1-raw`):** $\mathcal{A}_{\text{common}}(C) = \{a \in \mathbb{Z}_{\ge 0}^3 : a_0 + a_1 + a_2 \le C\}$ ($27\,720$ triplas para $C=53$).
- **Subespaço ativo (`B1-active`):** $\mathcal{A}_{\text{act}}(W, C) = \{a \in \mathcal{A}_{\text{common}}(C) : a_i = 0 \;\forall i \notin \mathcal{I}_{\text{act}}(W)\}$. Se $\mathcal{I}_{\text{act}}(W) = \emptyset$, retorna imediatamente $(0, 0, 0)$ (modo `all_empty`).
- **Subespaço ativo não degenerado (`B1-support`):** além de $\mathcal{A}_{\text{act}}(W, C)$, impõe piso mínimo de não inanição $a_i \ge \underline{a}_i(W)$ onde $\underline{a}_i(W) = 1$ se $i \in \mathcal{I}_{\text{live}}(W)$ e $|\mathcal{I}_{\text{live}}(W)| \le C$ (e $0$ caso contrário), denotado $\mathcal{A}_{\text{act}}^+(W, C)$, tornando $(0,0,0)$ impossível sempre que houver UEs ou filas vivas.

### 1.5 Critério de Suporte Confiável e Incerteza (`B1-support`)

Com o kernel padrão de `B0-rest` (Matérn $\nu=1{,}5$, $\ell = 1{,}0$ em $[z_i, a_i/C]$), deslocar apenas a ação de $a_j=25$ para $a=0$ sob o mesmo estado ($d_z = 0$) produz $\sigma = 1 - k(25/53) = 0{,}197$, e o máximo deslocamento possível com $d_z = 0$ é $\sigma(1{,}0) = 0{,}5167$, enquanto ~90% das ações bem visitadas nos checkpoints reais `B0-rest` apresentam $\sigma \le 0{,}05$ (e `p95` até `~0,07–0,21`). Consequentemente:

1. **Margem penalizada por incerteza (LCB):**
   $$g_i^{\text{supp}}(a_i) = f_i(a_i) - \beta\,\sigma_i(a_i), \qquad u_i^{\text{supp}}(a_i) = g_i^{\text{supp}}(a_i) - \gamma \frac{a_i}{C}, \qquad (\beta = 1{,}0).$$
2. **Envelope confiável primário ($\sigma_{\max} = 0{,}15$):**
   $$\mathcal{F}_{\text{supp}}(W; \sigma_{\max}) = \big\{ a \in \mathcal{A}_{\text{act}}^+(W, C) : \max_{i \in \mathcal{I}_{\text{act}}(W)} \sigma_i(a_i) \le \sigma_{\max} \big\}.$$
3. **Envelope relaxado de contenção ($\sigma_{\text{soft}} = 0{,}55$):** cobre extrapolações moderadas na ação dentro da distribuição de $z_i$ antes de classificar o estado como *out-of-distribution* (OOD) no espaço de estado ($\min_a \sigma_i(a) > 0{,}55$).

> *Calibração pré-confirmatória:* os hiperparâmetros $(\sigma_{\max}=0{,}15,\; \sigma_{\text{soft}}=0{,}55,\; \beta=1{,}0)$ poderão ter sensibilidade verificada **exclusivamente** no conjunto de desenvolvimento (`2101–2103`) e deverão ser congelados com hash antes de qualquer abertura das *seeds* `3101–3103`.

### 1.6 Regras de Seleção, Desempate e Hierarquia sem Solução / Fallback

Para qualquer conjunto candidato não vazio $\mathcal{F} \neq \emptyset$, escore $g$ e utilidade $u$ em escopo $\mathcal{S}$, o operador determinístico $\text{LexMax}(\mathcal{F}, g, u, \mathcal{S})$ escolhe o único ótimo na seguinte ordem estrita:
1. $\max_{a \in \mathcal{F}} \min_{i \in \mathcal{S}} g_i(a_i)$ *(maximiza pior margem — gargalo)*;
2. $\max \sum_{i \in \mathcal{S}} u_i(a_i)$ *(maximiza eficiência/custo $\gamma$)*;
3. $\min \sum_{i \in \mathcal{S}} a_i$ *(conserva RBGs excedentes)*;
4. Menor tupla lexicográfica $(a_0, a_1, a_2) \in \mathbb{N}_0^3$.

Definição exata de cada política:

- **`B1-raw`:** $\text{LexMax}(\mathcal{A}_{\text{common}}(C), f, u, \{0,1,2\})$.
- **`B1-active`:** se $\mathcal{I}_{\text{act}}(W)=\emptyset \to (0,0,0)$; senão $\text{LexMax}(\mathcal{A}_{\text{act}}(W, C), f, u, \mathcal{I}_{\text{act}}(W))$.
- **`B1-support` (hierarquia operacional em 4 níveis — zero RNG):**
  - **Nível 0 (`all_empty`):** se $\mathcal{I}_{\text{act}}(W)=\emptyset \to (0,0,0)$.
  - **Nível 1 (`joint_support`):** se $\mathcal{F}_{\text{supp}}(W; \sigma_{\max}) \neq \emptyset \to \text{LexMax}(\mathcal{F}_{\text{supp}}(W; \sigma_{\max}), g^{\text{supp}}, u^{\text{supp}}, \mathcal{I}_{\text{act}}(W))$.
  - **Nível 2 (`joint_lcb_relaxed`):** se Nível 1 for vazio (ex.: regiões individuais de suporte conflitantes com soma $> C$), mas $\mathcal{F}_{\text{supp}}(W; \sigma_{\text{soft}}) \neq \emptyset \to \text{LexMax}(\mathcal{F}_{\text{supp}}(W; \sigma_{\text{soft}}), g^{\text{supp}}, u^{\text{supp}}, \mathcal{I}_{\text{act}}(W))$, onde $-\beta\,\sigma_i$ arbitra o compromisso longe da extrapolação pura.
  - **Nível 3 (`OOD fallback`, se mesmo $\mathcal{F}_{\text{supp}}(W; \sigma_{\text{soft}}) = \emptyset$):** protege a operação contra predições sem suporte:
    - *(a)* Se a memória $k$-NN congelada possuir proposta positiva em todas as slices vivas ($a_i^{\text{knn}} > 0 \;\forall i \in \mathcal{I}_{\text{live}}$), aplica projeção com preservação de piso `bounded_project_active(a^knn, C, I_act, floor)` $\to$ `fallback_knn`;
    - *(b)* Caso contrário (memória insuficiente ou $k$-NN degenerado), aplica divisão proporcional determinística por UEs/fila ativos `prop_active_split` respeitando os pisos $\underline{a}_i$ (ou `equal_active_split` configurável) $\to$ `fallback_prop_active` / `fallback_equal_active`.

### 1.7 Complexidade de Execução vs. Período de Decisão de 100 ms

- Construção das tabelas $(54 \times 3)$ via `SqrSliceController.score_candidates` com memória cheia $B=2000$: ~`32–36 ms` (mesma ordem da inferência do `B0-rest`, que registrou `p50=17,9–20,4 ms`, `p95=34,9–38,3 ms` online em `seed=2103`).
- Busca combinatória vetorizada em NumPy sobre o grid pré-cacheados em $\mathcal{A}_{\text{common}}(53)$ ($|\mathcal{F}| \le 27\,720$, $O(|\mathcal{F}|)$ sem laços Python): medida empiricamente em **`0,42–0,52 ms` média (`p95 <= 0,61 ms`)**.
- **Latência total ponta a ponta (`B=2000` + busca vetorizada em `select_from_het`):** **`p50 = 32,07–36,41 ms`, `p95 = 32,34–39,60 ms` (< 40% do orçamento de `100 ms` e bem abaixo do gate histórico de `80 ms`).**

---

## 2. Suíte de Testes Unitários e de Integração (`test_sqr_jc_a5_selector.py`)

Implementada em [`test_sqr_jc_a5_selector.py`](../../../ns-o-ran-gym/tests/test_sqr_jc_a5_selector.py) e integrada a [`run_sqr_tests.sh`](../../../ns-o-ran-gym/tests/run_sqr_tests.sh), verificando obrigatoriamente:

1. **Equivalência e desempate determinístico:** 20 tabelas aleatórias com empates forçados em gargalo, utilidade, $\sum a$ e vetor vs. enumeração pura em Python (`O(C^3)`).
2. **Paridade de demanda nula e pendências:** igualdade exata entre `causal_active_slices` (sobre `JointState` e dicts de trace) e `SqrHetController.is_slice_provably_empty` (incl. `urllc_pool_open`, `mmtc_rolling_active`, `embb_pending_pipeline`, `kpm_tx_packets`).
3. **Legalidade irrestrita:** todas as combinações de 0/1/2/3 slices ativas e $C \in \{10, 52, 53\}$ respeitando $a_i \in \{0..C\}$, $\sum a_i \le C$, zero em inativas e $a_i \ge 1$ em slices vivas em `B1-support`.
4. **Regressão anti-degenerescência `D3-v1`:** comprovação com `KernelQuantileRegressor` real sob $\Lambda=0$ de que `B1-raw`/`B1-active` colapsam para extrapolação zero enquanto `B1-support` seleciona alocações suportadas e positivas.
5. **Hierarquia e fallback determinístico:** cobertura explícita de Nível 1 (`joint_support`), Nível 2 (`joint_lcb_relaxed`), Nível 3 (`fallback_knn`, `fallback_prop_active`, `fallback_equal_active`) e `bounded_project_active`.
6. **Causalidade, determinismo, zero RNG e não-antecipação:** paridade `select_from_het` / `select_from_trace_record`, imutabilidade integral de `_rng.bit_generator.state` e `state_dict()`, invariância total da decisão em $W=22$ frente a futuros divergentes pós-22, e bloqueio de checkpoint futuro em `load_b0_rest_het_from_checkpoint`.
7. **Integração de replay `A/A` bit-exato:** comparação ponta a ponta na mesma janela alvo entre `B0_rest`, `equal`, `prop`, `b1_raw`, `b1_active` e `b1_support` com prefixo físico `step_metrics.csv` idêntico até $t-1$.

---

## 3. Plano de Replays Contrafactuais (`D3-v2`) vs. Avaliação *Closed-Loop* e Portas (*Gates*)

### 3.1 Dois Regimes Estritamente Separados

1. **Regime I — Replay contrafactual de janela única (`D3-v2`, pós-GO):**
   - Usa exclusivamente *snapshots* pré-decisão gravados com `--causal_trace_path` durante runs `B0-rest` (sem nenhuma informação futura de $W+1$) + `ReplayController(base="seq")` no ns-3, garantindo trajetória física **bit-idêntica (`A/A`)** até $W-1$ e intervenção isolada na janela $W$.
   - Compara lado a lado **6 braços**: `B0-rest`, `equal` `(18,17,18)`, `prop` (proporcional ativo determinístico), `B1-raw`, `B1-active` e `B1-support`.
   - **Amostragem sem viés de seleção:** o conjunto de janelas avaliadas na `D3-v2` deve ser estratificado por carga ($\lambda \in \{5, 6\}$) e estrato operacional (`1`, `2`, `=3` slices ativas; com e sem projeção no `B0`), **contendo obrigatoriamente estados onde `B0-rest` obteve `PASS` E estados onde obteve `FAIL`**, nunca filtrando apenas `B0=FAIL` ou rótulos `D2`.
   - **Métricas obrigatórias:** taxas `PASS`, `FAIL` e cobertura `UNKNOWN` em `STRICT_ALL3_R19`, `STRICT_DEMAND_R19`, `e∧u` e por slice; **`RescueRate`** $= \mathbb{P}(\text{Arm PASS} \mid B0\text{ FAIL})$, **`HarmRate`** $= \mathbb{P}(\text{Arm FAIL} \mid B0\text{ PASS})$ e **`NetGain`** $= \text{RescueRate} - \text{HarmRate}$ (com intervalo bootstrap e estratificação por contenção entre *seeds* `2101–2103`).

2. **Regime II — Avaliação *closed-loop out-of-sample* (`--resume <ckpt_B0> --freeze`, somente pós-Gate 2):**
   - Executa cada política desde `step=0` em episódios completos nas *seeds* independentes (`3101–3103`), mantendo `verify_freeze_no_learning` (ou comparando também adaptação online se especificado no futuro protocolo de `MAIN`).
   - Diferença metodológica: em malha fechada, decisões divergentes modificam filas, chegadas efetivas e dinâmica futura da célula; por isso avaliam-se distribuições por episódio entre *seeds*, nunca pareamento janela-a-janela com a trajetória de `B0-rest`.

### 3.2 Portas de Decisão (*Gates*) e Condição para Primeira Execução Exploratória

| Gate | Escopo | Pré-requisitos Objetivos de Avanço | Status Atual |
|---|---|---|---|
| **Gate 0 (Implementação modular `A5`)** | `a5_joint_selector.py` + `test_sqr_jc_a5_selector.py` + `B0_FROZEN_MANIFEST.json` | Todos os testes unitários e de integração (`run_sqr_tests.sh`) passando (`0` falhas), zero alterações em `B0-rest`, formulação matemática aprovada. | **CONCLUÍDO (`360 passed`)** |
| **Gate 1 (Primeira execução exploratória em dev)** | Gravar `causal_trace` de `B0-rest` em `dev` (`2102`, `2103`) + mini-replay `A5` exploratório no ns-3 **somente em *seeds* de desenvolvimento** | Aprovação explícita do usuário após revisar a formulação `A5` + auditoria `2102–2103` + verificação de paridade `A/A` bit-exata em `ReplayController`. | **EM EXECUÇÃO (GO exclusivo Gate 1)** |
| **Gate 2 (`D3-v2` em dev `2101–2103`)** | Ablação pareada contrafactual completa (`B0`, `equal`, `prop`, `B1-*`) em dev | `A/A` 100% bit-exato; zero ações ilegais/degeneradas `(0,0,0)` com demanda; `NetGain > 0` (`HarmRate < RescueRate`); superioridade demonstrada além de `equal`/`prop` inclusive em `=3` ativas (caso `seed=2103`). | **BLOQUEADO** |
| **Gate 3 (`PILOT` / `MAIN` / `3101–3103`)** | Abertura das *seeds* independentes | Hiperparâmetros de `A5` (`sigma_max`, `sigma_soft`, `beta_uncertainty`), critérios de aceitação e protocolo de avaliação integralmente congelados antes de rodar `3101–3103`. | **BLOQUEADO** |
