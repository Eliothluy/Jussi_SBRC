# Relatório Exploratório — Gate 1 da Fase `A5` (`SQR-JC`) em *Seeds* de Desenvolvimento (`2102`, `2103`)

**Data:** 2026-11-16  
**Baseline Congelado (`B0-rest`):** `A1+A2a+A_scaler` ([`B0_FROZEN_MANIFEST.json`](B0_FROZEN_MANIFEST.json), commit de congelamento `8375f4da3713006449036ace09b7c0579e38fade`; correção de latência do *writer* causal em `da19af3`).  
**Políticas Avaliadas (`A5`):** `B1-raw`, `B1-active`, `B1-support` ([`a5_joint_selector.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/a5_joint_selector.py), parâmetros congelados `sigma_max=0.15`, `sigma_soft=0.55`, `beta_uncertainty=1.0`, `gamma=1.0`, `min_active_floor=1`), comparadas no mesmo prefixo físico contra `B0-rest`, `equal=(18,17,18)` e `prop` (proporcional ativo determinístico).  
**Veredito Científico do Gate 1:** **NO-GO para avançar para `Gate 2` (`D3-v2`) com a formulação `max-min` / LCB atual.** Todos os bloqueios em `Gate 2`, `D3-v2`, `PILOT`, `MAIN`, `A2b`, `A3b` e *seeds* independentes `3101–3103` permanecem **rigorosamente mantidos**.

---

## 1. Integridade da Instrumentação Causal (`B0-rest` + `--causal_trace_path`)

A reexecução dos 5 runs de desenvolvimento de `B0-rest` (`s2102` em $\lambda \in \{2,5,6\}$; `s2103` em $\lambda \in \{5,6\}$) com `--causal_trace_path` para `MODA/ns-o-ran-gym/results/sqr_jc_decision/b0_traced_gate1/` exigiu a correção documentada no commit `da19af3`: mover a serialização do `CausalTraceWriter` para depois de `controlSemaphore.release()` (via `_wait_data_availability` em `sqr_run.py`) com reuso de `score_candidates` (`precomputed_scores` em `freeze_contract.py`), evitando que o custo de `+37,07 ms` por decisão invadisse o *timeout* IPC em tempo real de `50 ms` (`sem_timedwait` em `sqr-sim.cc:1674`).

Mesmo sem `--causal_trace_path`, observou-se que os runs históricos sem trace já ultrapassavam pontualmente o orçamento de `50 ms` sob $n_{\text{support}}=2000$ devido ao segundo cálculo de `score_candidates` para log antes de `env.step()`. Por isso, o relatório de paridade ([`A5_GATE1_B0_PARITY_REPORT.json`](../../../ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_B0_PARITY_REPORT.json)) verifica rigorosamente o corte estrito anterior ao primeiro *shift* IPC de `50 ms` (`cut_step = min(orig_lag, traced_lag)`):

| Run `B0-rest` | `orig_first_lag` | `traced_first_lag` | `cut_step` limpo | `pre_dec` | `pre_kpm` | `pre_prb` | `pre_del_delivered` | `pre_step_excl_dt` | `causal_ok` / `future_ckpt_blocked` | `seq_mis` pré-corte |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `het_lam2_s2102` | `None` | `None` | `999999` (100%) | `True` | `True` | `True` | `True` (`45.017`) | `True` | `True` / `564` (1 boundary `2995`) | `0` |
| `het_lam5_s2102` | `None` | `2836` | `2836` (94,7%) | `True` | `True` | `True` | `True` (`252.547`) | `True` | `True` / `1994` | `0` |
| `het_lam6_s2102` | `2891` | `2416` | `2416` (80,7%) | `True` | `True` | `True` | `True` (`244.945`) | `True` | `True` / `1994` | `0` |
| `het_lam5_s2103` | `1692` | `2219` | `1692` (56,5%) | `True` | `True` | `True` | `True` (`270.850`) | `True` | `True` / `1947` | `0` |
| `het_lam6_s2103` | `1473` | `1788` | `1473` (49,2%) | `True` | `True` | `True` | `True` (`280.026`) | `True` | `True` / `1994` | `0` |

* Nota metodológica transparente: `het_lam2_s2102` preservou paridade integral de ponta a ponta (inclusive `checkpoint_payload_sha256` idêntico; única diferença em `step_metrics.csv` foi a coluna de tempo de relógio `decision_time_ms`). Nos demais 4 runs, `decisions.jsonl`, KPM, PRBs e pacotes URLLC entregues até `cut_step` são **100% bit-idênticos** ao baseline histórico congelado, e todos os candidatos do Gate 1 foram extraídos exclusivamente de `step < cut_step` e validados com `verify_causal_replay_no_future_checkpoint`.
* Observação técnica para preparação futura: `s2103` em $\lambda=2$ não integrava os runs congelados em [`B0_FROZEN_MANIFEST.json`](B0_FROZEN_MANIFEST.json) e permaneceu fora deste gate.

---

## 2. Plano Estratificado Pré-Registrado e Integridade dos Replays Contrafactuais

Antes de qualquer chamada ao `A5JointSelector`, [`gate1_generate_sampling_plan.py`](../../../ns-o-ran-gym/results/sqr_jc_decision/gate1_generate_sampling_plan.py) gerou deterministicamente (passo por quantil dentro de cada estrato) **25 estados** distribuídos em **3 universos separados** ([`A5_GATE1_SAMPLING_PLAN.csv`](../../../ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_SAMPLING_PLAN.csv), manifesto SHA-256 `771b9a2c69e10bbbbdcd4c505684552c3523cb1edaa01c30c308b145f8b4650c`, commit `da19af3`):

1. **Universo A — `ALL3_COMPLETE` ($N = 18$ estados):** Todos os 3 SLAs com evidência `COMPLETE` (`STRICT_ALL3_R19`), compreendendo **8 `B0=PASS`** e **10 `B0=FAIL`**, cobrindo $\lambda \in \{5,6\}$, *seeds* `2102` (`9`) e `2103` (`9`), com excesso `sum(a_prop) > 53` (`10`) e sem excesso (`8`), `16` estados com `3` slices ativas e `2` estados com `2` slices causais ativas onde o *rolling* mMTC + chegada URLLC fecham os 3 SLAs em $T$.
2. **Universo B — `LT3_DEMAND_COMPLETE` ($N = 5$ estados):** Regimes com **1 ou 2 slices ativas** ($\lambda \in \{2,5,6\}$, *seeds* `2102` e `2103`, `2 PASS` e `3 FAIL` em `STRICT_DEMAND_R19`; `STRICT_ALL3_R19 = UNKNOWN` por ausência de demanda em slice ociosa).
3. **Universo C — `UNKNOWN_INTEGRITY` ($N = 2$ estados):** Janelas com pool URLLC `u_unclosed_online` (`lam2_s2102_w2448` e `lam5_s2103_w1351`), mantidas isoladas exclusivamente para auditoria de cobertura/integridade e **excluídas dos denominadores** de desempenho.

### 2.1 Verificação Executável dos Replays (`ReplayController(base="seq")` + horizonte `+3 s`)
* **Plano de execução:** `25 estados × 6 políticas = 150` avaliações lógicas consolidadas em **`127` replays únicos no ns-3** (`--c_prbs 53`, cauda de drenagem `+3 s` pós-$T$, `--simTime <= 300 s`).
* **Ocorrência operacional registrada (`WORKERS=12` -> `WORKERS=4`):** A primeira onda paralela (`WORKERS=12`) produziu `12/127` falhas de paridade de prefixo causadas por contenção transitória de CPU ultrapassando `sem_timedwait(50 ms)` em passos do prefixo (e no passo alvo de `lam2_s2102_w2442__b0`). Reexecutados com `WORKERS=4`, todos os `12` passaram.
* **Integridade final antes da adjudicação:** **`127/127` (`100%`) replays válidos**, **`25/25` (`100%`) paridade `A/A` `B0-rest` bit-exata** (prefixos KPM, PRB e atrasos URLLC entregues `< T·100 ms` idênticos, cota executada `==` plano, e KPM/PRB/delays da própria janela `T` bit-idênticos no ramo `A/A`).

---

## 3. Resultados Contrafactuais com Denominadores Explícitos

### 3.1 Universo A — `ALL3_COMPLETE` (`STRICT_ALL3_R19`, $N = 18$, $0$ `UNKNOWN`)

> **Denominadores explícitos:** `B0 PASS` tem $N_{\text{pass}} = 8$ (`HarmRate = harmed / 8`); `B0 FAIL` tem $N_{\text{fail}} = 10$ (`RescueRate = rescued / 10`); `NetGain = (Rescued - Harmed) / 18`.

| Política | `PASS / Known` (`%`) | `UNK` | `Rescued / 10` (`RescueRate`) | `Harmed / 8` (`HarmRate`) | `Net (R - H) / 18` | `util_dl` média | `sum(a)` médio |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`B0-rest`** (`A/A`) | `8 / 18` (**44,4%**) | `0` | `0 / 10` (0,0%) | `0 / 8` (0,0%) | `+0 / 18` (+0,0%) | `46,2%` | `46,6` |
| **`equal`** `(18,17,18)` | **`12 / 18` (66,7%)** | `0` | **`6 / 10` (60,0%)** | `2 / 8` (25,0%) | **`+4 / 18` (+22,2%)** | `37,6%` | `53,0` |
| **`prop`** (ativo) | **`11 / 18` (61,1%)** | `0` | `5 / 10` (50,0%) | `2 / 8` (25,0%) | **`+3 / 18` (+16,7%)** | `31,7%` | `53,0` |
| **`B1-raw`** | `3 / 18` (**16,7%**) | `0` | `2 / 10` (20,0%) | **`7 / 8` (87,5%)** | **`-5 / 18` (-27,8%)** | `26,5%` | `53,0` |
| **`B1-active`** | `2 / 18` (**11,1%**) | `0` | `1 / 10` (10,0%) | **`7 / 8` (87,5%)** | **`-6 / 18` (-33,3%)** | `27,0%` | `53,0` |
| **`B1-support`** | `2 / 18` (**11,1%**) | `0` | `1 / 10` (10,0%) | **`7 / 8` (87,5%)** | **`-6 / 18` (-33,3%)** | `28,1%` | `49,1` |

**Estratificação dentro do Universo A (`PASS / Known`, `Rescue`, `Harm`, `Net`):**

| Estrato (denominador) | `B0-rest` | `equal` | `prop` | `B1-raw` | `B1-active` | `B1-support` |
|---|---:|---:|---:|---:|---:|---:|
| **`=3` ativas** ($N=16$; 8F/8P) | `8/16 (50,0%)` | `11/16` (`R=5/8, H=2/8, +3`) | `11/16` (`R=5/8, H=2/8, +3`) | `2/16` (`R=1/8, H=7/8, -6`) | `2/16` (`R=1/8, H=7/8, -6`) | `2/16` (`R=1/8, H=7/8, -6`) |
| **`λ=5`** ($N=9$; 5F/4P) | `4/9 (44,4%)` | `5/9` (`R=3/5, H=2/4, +1`) | `6/9` (`R=3/5, H=1/4, +2`) | `3/9` (`R=2/5, H=3/4, -1`) | `2/9` (`R=1/5, H=3/4, -2`) | `2/9` (`R=1/5, H=3/4, -2`) |
| **`λ=6`** ($N=9$; 5F/4P) | `4/9 (44,4%)` | **`7/9`** (`R=3/5, H=0/4, +3`) | `5/9` (`R=2/5, H=1/4, +1`) | `0/9` (`R=0/5, H=4/4, -4`) | `0/9` (`R=0/5, H=4/4, -4`) | `0/9` (`R=0/5, H=4/4, -4`) |
| **`seed=2102`** ($N=9$; 5F/4P) | `4/9 (44,4%)` | `4/9` (`R=1/5, H=1/4, 0`) | `3/9` (`R=1/5, H=2/4, -1`) | `3/9` (`R=2/5, H=3/4, -1`) | `2/9` (`R=1/5, H=3/4, -2`) | `2/9` (`R=1/5, H=3/4, -2`) |
| **`seed=2103`** ($N=9$; 5F/4P) | `4/9 (44,4%)` | **`8/9`** (`R=5/5, H=1/4, +4`) | **`8/9`** (`R=4/5, H=0/4, +4`) | **`0/9` (`R=0/5, H=4/4, -4`)** | **`0/9` (`R=0/5, H=4/4, -4`)** | **`0/9` (`R=0/5, H=4/4, -4`)** |
| **`excess=False`** ($N=8$; 4F/4P) | `4/8 (50,0%)` | `7/8` (`R=3/4, H=0/4, +3`) | `7/8` (`R=3/4, H=0/4, +3`) | `2/8` (`R=1/4, H=3/4, -2`) | `2/8` (`R=1/4, H=3/4, -2`) | `2/8` (`R=1/4, H=3/4, -2`) |
| **`excess=True`** ($N=10$; 6F/4P) | `4/10 (40,0%)` | `5/10` (`R=3/6, H=2/4, +1`) | `4/10` (`R=2/6, H=2/4, 0`) | `1/10` (`R=1/6, H=4/4, -3`) | `0/10` (`R=0/6, H=4/4, -4`) | `0/10` (`R=0/6, H=4/4, -4`) |

### 3.2 Universo B — `LT3_DEMAND_COMPLETE` (`STRICT_DEMAND_R19`, $N = 5$, $0$ `UNKNOWN`)

> Todos os 5 estados têm `STRICT_ALL3_R19 = UNKNOWN` (slice ociosa sem demanda) e `STRICT_DEMAND_R19 COMPLETE` (`2 B0=PASS`, `3 B0=FAIL`).

| Política | `PASS / Known` (`%`) | `Rescued / 3` | `Harmed / 2` | Desperdício em slice inativa (`inactive_waste`) | Slice viva zerada (`starved_live`) |
|---|---:|---:|---:|---:|---:|
| **`B0-rest`** | `2 / 5` (40,0%) | `0 / 3` | `0 / 2` | `0 / 5` | `1 / 5` (`lam2_w2685=[0,0,0]`) |
| **`equal`** `(18,17,18)` | `3 / 5` (60,0%) | `1 / 3` | `0 / 2` | **`5 / 5`** | `0 / 5` |
| **`prop`** (ativo) | `3 / 5` (60,0%) | `1 / 3` | `0 / 2` | `0 / 5` | `0 / 5` |
| **`B1-raw`** | `2 / 5` (40,0%) | `0 / 3` | `0 / 2` | **`3 / 5`** (`(0,53,0)`/`(39,14,0)`) | **`3 / 5`** |
| **`B1-active`** | **`4 / 5` (80,0%)** | **`2 / 3`** | **`0 / 2`** | **`0 / 5`** | **`1 / 5`** |
| **`B1-support`** | `3 / 5` (60,0%) | `2 / 3` | `1 / 2` | `0 / 5` | `0 / 5` |

### 3.3 Universo C — `UNKNOWN_INTEGRITY` ($N = 2$ estados)
* `lam2_s2102_w2448` (`tx_u=86, rx_u=81`) e `lam5_s2103_w1351` (`tx_u=178, rx_u=176`) permaneceram com `urllc_pool_tag = "OPEN"` e `strict_demand_r19 = "UNKNOWN"` em **12/12 braços (`100%`)** mesmo após a cauda estendida de `+3 s`, confirmando perdas reais (`5` e `2` pacotes perdidos) e ausência de vazamento para os denominadores de desempenho.

---

## 4. Isolamento dos Efeitos e Diagnóstico Causal das Patologias

### 4.1 Efeito da Filtragem de Demanda (`B1-raw -> B1-active`) vs. Suporte/LCB/Piso (`B1-active -> B1-support`)
1. **`B1-raw -> B1-active` (efeito positivo em `< 3` ativas, limitação de fronteira em chegadas limítrofes):**
   * No **Universo B**, a filtragem causal de slices ociosas eliminou `3/3` desperdiçamentos em slices inativas e elevou `STRICT_DEMAND_R19` de **`40,0%` (`2/5`) para `80,0%` (`4/5`)**, superando `equal` e `prop` (`60,0%`) ao resgatar `lam2_s2102_w2685` (`mMTC` ativa, `(0,0,41):PASS` vs `(0,53,0):FAIL` de `B1-raw`) e `lam6_s2102_w2071` (`eMBB+mMTC`, `(53,0,0):PASS` vs `(39,14,0):FAIL`).
   * Porém, em `lam5_s2102_w2552` (`ALL3`, `n_act=2` no instante $W$ com `active_mask=(1,0,1)`), `B1-raw` alocara `(47,6,0):PASS` por acidente (extrapolando para `URLLC`), enquanto `B1-active` e `B1-support` zeraram a slice URLLC inativa no instante $W$ (`(53,0,0)` e `(52,0,1)`), falhando quando pacotes URLLC chegaram já dentro da janela $T=2552$.
2. **`B1-active -> B1-support` (`min_floor=1`, `sigma_max=0.15`, `beta=1.0`):**
   * O seletor operou `25/25` no nível primário (`joint_raw=25`, `joint_active=25`, `joint_support=25`, **`0` fallbacks** e **`0` violações** $\sigma > 0{,}15$ em slices ativas).
   * Contudo, no **Universo A**, `B1-support` **não recuperou nenhum dano** do `B1-active` (`2/18 = 11,1%`, `Rescue=1/10`, `Harm=7/8`) e ainda produziu um **dano adicional no Universo B** (`lam2_s2102_w2442`: `B0=(31,22,0):PASS` e `B1-active=(38,15,0):PASS`, mas **`B1-support=(6,19,0):FAIL[e]`**, desperdiçando `28/53` PRBs!).
   * Como esperado conceitualmente, a restrição $\sigma \le 0{,}15$ e o piso $a_i \ge 1$ **não conferem garantia alguma de atendimento de SLA**: no modelo observado, $n_{\text{trusted}} \ge 44\text{–}54$ cobria quase todo $[0..53]$, e `1` PRB para eMBB rendeu `0,49–0,98 Mbps < 3,5 Mbps` (`FAIL[e]` em 100% dos colapsos `(1, 51, 1)`).

### 4.2 Causa Raiz 1 — Distorção de Escala e Dominância Sistemática de URLLC no `max-min`
As margens relativas de [`sla_margin.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/sla_margin.py) vivem em faixas incompatíveis entre serviços:
* **eMBB (`s0`):** $m_0 = (T_{\text{eMBB}} - 3{,}5)/3{,}5 \in [-1{,}0, +\infty)$, atingindo $f_0 \in [+0{,}39, +1{,}51]$ em `s2103`.
* **URLLC (`s1`):** $m_1 = (10{,}0 - p_{95})/10{,}0 \in (-\infty, +1{,}0]$, raramente excedendo `+0,19` e caindo para **$f_1 \in [-0{,}87, -0{,}28]$ em todo $[0..53]$** sob concorrência (mesmo onde $f_1(51) - f_1(13) \approx +0{,}04$).
* **mMTC (`s2`):** $m_2 = (\text{PDR}_{1\text{s}} - 0{,}95)/0{,}95 \in [-1{,}0, +0{,}0526]$, com teto absoluto de `+0,0526`.

Sob seleção lexicográfica `max-min` ($\arg\max \min_{i \in \mathcal{I}_{\text{act}}} g_i(a_i)$), **URLLC foi o gargalo (`bottleneck_slice=1`) em `17/25` estados no `B1-raw`, `16/25` no `B1-active` e `20/25` (`80%`) no `B1-support`** (inclusive `16/16` em todas as janelas `n_act=3` de $\lambda=6$ e `s2103`). Em **8 de 8 (`100%`)** estados `n_act=3` de `seed=2103`, mesmo com eMBB em $+0{,}39\text{–}+0{,}81$ e URLLC saturada em $-0{,}29\text{–}-0{,}48$, a regra `max-min` drenou todos os `51–53` PRBs para `URLLC`:
* `B1-raw` / `B1-active`: `(0, 53, 0)` $\implies$ `eMBB = 0,0–0,61 Mbps` (`FAIL[e]` em `8/8`, `PASS = 0%`).
* `B1-support`: `(1, 51, 1)` $\implies$ `eMBB = 0,49–0,98 Mbps` (`FAIL[e]` em `8/8`, **`HarmRate = 4/4 = 100%`, `RescueRate = 0/4 = 0%`**).  
Em contraste direto nos mesmos estados de `2103`, `equal=(18,17,18)` e `prop=(13,27,13)` atingiram **`8/9 (88,9%)` de `PASS`**.

### 4.3 Causa Raiz 2 — Encolhimento da Alocação ABAIXO de `C=53` pelo termo LCB (`-beta_uncertainty * sigma` com `beta=1.0`)
Nos casos com inclinação fraca de $f_i(a)$ em quantis extremos ($\tau \in [0{,}01, 0{,}04]$), subtrair $-1{,}0 \cdot \sigma_i(a)$ diretamente na utilidade lexicográfica primária $g_i(a) = f_i(a) - \beta\sigma_i(a)$ faz o crescimento de $\sigma_i(a)$ com o afastamento dos pontos mais densos superar o ganho físico $\Delta f_i(a)$:
* `lam6_s2102_w1406`: `b0=(20,23,1):PASS` vs `b1_act=(8,45,0):FAIL[e]` vs **`b1_sup=(6,13,1):FAIL[e]`** (`sum(a) = 20/53`, deixando **33 PRBs ociosos**! Em `a_u=13`, $\sigma_1=0{,}0005, g_1=-0{,}1819$, vencendo `a_u=44` onde $\sigma_1 \approx 0{,}10$, e `s0` fechou em `a_e=6` com $\sigma_0=0{,}014$).
* `lam6_s2102_w1396`: **`b1_sup=(1,13,1):FAIL[e]`** (`sum(a) = 15/53`, **38 PRBs ociosos**!).
* `lam2_s2102_w2442`: `b0=(31,22,0):PASS` e `b1_act=(38,15,0):PASS` ($\sigma_0=0{,}153$ marginalmente acima de `0,15`), porém **`b1_sup=(6,19,0):FAIL[e]`** (`sum(a) = 25/53`, **28 PRBs ociosos**!), gerando o único dano do `B1-support` no Universo B.
* No total dos 25 estados, `B1-support` desperdiçou capacidade ($a_{\text{sum}} < 53$) em **`6/25` (`24%`)** das decisões (contra `1/25` no `B1-raw` e `2/25` no `B1-active`).

---

## 5. Latência Computacional (Orçamento IPC `50 ms` e RIC `100 ms`)

Medições registradas em [`A5_GATE1_DECISIONS.json`](../../../ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_DECISIONS.json) (`25` estados, $n_{\text{support}}=2000$ no benchmark *end-to-end* com `het_lam6_s2103`):

| Variante | Otimizador isolado `select_from_trace_record` (`p50 / p95 / p99 / max`) | Online completo `select_from_het` (`3×score_candidates + A5`, `p50 / p95 / p99 / max`) |
|---|---|---|
| `B1-raw` | `0,31 / 0,35 / 4,15 / 5,35 ms` | `36,21 / 37,19 / 37,37 / 37,41 ms` |
| `B1-active` | `0,30 / 0,34 / 3,56 / 4,57 ms` | `36,37 / 40,69 / 45,38 / 46,59 ms` |
| `B1-support` | `0,43 / 2,98 / 4,17 / 4,35 ms` | `36,35 / 36,94 / 37,10 / 37,13 ms` |

O otimizador conjunto em si custa **$< 0{,}5\text{ ms}$ mediano**; o gargalo computacional online decorre integralmente das `3 × score_candidates` sobre buffers `2000 × 7` (`~36 ms`), mantendo folga frente ao período de `100 ms` do RIC/E2, porém próximo ao teto de `50 ms` do `sem_timedwait` IPC quando não houver reutilização de tabelas.

---

## 6. Decisão do Gate 1 e Recomendações para Revisão da Formulação (Antes de Novo Gate)

* **Decisão formal:** **NO-GO para Gate 2 (`D3-v2`).** A formulação atual de `A5` (`max-min` sobre margens quantílicas assimétricas $f_i$ com penalização LCB $-\beta\sigma_i$ na utilidade primária) degrada causalmente o `B0-rest` (`44,4% -> 11,1%`, `HarmRate = 87,5%`, `NetGain = -33,3%`), sendo superada por `equal` (`66,7%`), `prop` (`61,1%`) e pelo próprio `B0-rest`.
* **Regra preservada:** Nenhuma calibração ou ajuste de parâmetros foi aplicada neste gate (`sigma_max=0.15, sigma_soft=0.55, beta=1.0` mantidos intocados); `Gate 2`, `D3-v2`, `PILOT`, `MAIN`, `A2b`, `A3b` e *seeds* `3101–3103` permanecem **estritamente bloqueados**.
* **Caminho científico recomendado para nova proposta conceitual (sujeita à aprovação explícita antes de codar):**
  1. **Corrigir a distorção de escala no objetivo conjunto:** substituir `min_i g_i` sobre margens não limitadas e assimétricas por objetivo lexicográfico de **viabilidade primeiro** (ex.: minimizar déficit positivo em relação ao limiar $\Lambda=0$, $\max_i \max(0, -f_i)$ ou soma de déficits/probabilidade de satisfação $\min(0, f_i)$ saturado em $f_i \ge 0$), de modo que uma slice já inviável sob carga extrema (`URLLC` em `-0.45` com inclinação `0.04`) **não destrua** o SLA das demais slices com alocações extremas `(1, 51, 1)` quando ganhos adicionais acima de certo $a_u$ são nulos;
  2. **Separar incerteza da monotonia de alocação de recursos:** usar $\sigma \le \sigma_{\max}$ apenas como **restrição de confiança / conjunto admissível** (ou termo de empate secundário) **sem** penalizar $-\beta\sigma$ dentro da utilidade primária, somado à exigência de eficiência de Pareto / uso completo da capacidade disponível quando há demanda ativa ($\sum a_i = C$), eliminando os colapsos ociosos `(6, 13, 1)` e `(6, 19, 0)`;
  3. **Preservar o ganho comprovado da filtragem de demanda (`B1-active`)** no Universo B (`40% -> 80%`) e verificar tratamento seguro de fronteira para tráfego pendente/recém-chegado (`lam5_s2102_w2552`).

---

## 7. Artefatos e Verificação Final
* Resultados tabulares: [`A5_GATE1_METRICS.csv`](../../../ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_METRICS.csv) (`150` linhas), [`A5_GATE1_DIAGNOSTICS.json`](../../../ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_DIAGNOSTICS.json), [`A5_GATE1_DECISIONS.json`](../../../ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_DECISIONS.json), [`A5_GATE1_REPLAY_INTEGRITY.json`](../../../ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_REPLAY_INTEGRITY.json).
* Scripts de auditoria e replay: [`gate1_verify_traced_b0_parity.py`](../../../ns-o-ran-gym/results/sqr_jc_decision/gate1_verify_traced_b0_parity.py), [`gate1_generate_sampling_plan.py`](../../../ns-o-ran-gym/results/sqr_jc_decision/gate1_generate_sampling_plan.py), [`gate1_decide_and_plan_replays.py`](../../../ns-o-ran-gym/results/sqr_jc_decision/gate1_decide_and_plan_replays.py), [`gate1_adjudicate_and_analyze.py`](../../../ns-o-ran-gym/results/sqr_jc_decision/gate1_adjudicate_and_analyze.py).
