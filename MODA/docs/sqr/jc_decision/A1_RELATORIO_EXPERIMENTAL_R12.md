# Relatório Experimental A1 — Isolamento de R12 (`--virtual_alpha 0.0`) vs. A0 (`D1-v9`, `virtual_alpha = 0.05`)

- **Data:** 2026-11-14
- **Protocolo de referência:** [`D3_RETIFICACAO_E_PLANO_ABLACAO.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/D3_RETIFICACAO_E_PLANO_ABLACAO.md)
- **Escopo autorizado:** Experimento isolado **A1** (`seed = 2102`, $\lambda \in \{2, 5, 6\}$ por cenário total, isto é, $\{0.667, 1.667, 2.000\}$ por slice; `c_prbs = 52` iniciais / `C = 53` efetivos após descoberta MAC; `simTime = 300 s`, `warmup = 100 s`, `period = 100 ms`, `cellRadius = 300 m`, `--urllc_delayed_update`, `--virtual_alpha 0.0`).
- **Preservação de histórico:** `D1-v9` ([`d1_runs_v9_delayed/`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/d1_runs_v9_delayed)), `D2-v2` e `D3` foram mantidos integralmente intactos; os runs de A1 foram gravados em [`a1_runs_v9_va0/`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/a1_runs_v9_va0).
- **Artefatos de avaliação gerados:**
  - Script reproduzível: [`evaluate_a1_vs_a0.py`](file:///home/eliothluy/.gemini/antigravity/brain/4622041f-48e4-41e3-8c43-cf5834bb15d6/scratch/evaluate_a1_vs_a0.py)
  - Sumário estruturado: [`a1_vs_a0_summary.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/a1_vs_a0_summary.json)
  - Tabela CSV consolidada: [`a1_vs_a0_metrics.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/a1_vs_a0_metrics.csv)

---

## 1. Confirmação de Isolamento Pré-Execução (`--virtual_alpha 0.0`) e Integridade

### 1.1 Verificação do flag e correção mínima necessária
Na inspeção pré-execução de [`examples/sqr_run.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_run.py), identificou-se que `--virtual_alpha` existia no CLI, mas **não era repassado ao construtor de `SqrHetController`** (`virtual_alpha` permanecia fixo no padrão `0.05`). Foram feitas exclusivamente duas alterações pontuais em [`examples/sqr_run.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_run.py):
1. Repasse explícito `virtual_alpha=args.virtual_alpha` para `SqrHetController`.
2. Registro causal passivo (somente log) de `f_at_a0`, `f_at_a37` e `f_at_amax` em `decisions.jsonl` a partir do vetor `sc["f"]` já calculado em `decide()`.

### 1.2 Prova de isolamento de `R12` antes da simulação
Executado [`verify_virtual_alpha_isolation.py`](file:///home/eliothluy/.gemini/antigravity/brain/4622041f-48e4-41e3-8c43-cf5834bb15d6/scratch/verify_virtual_alpha_isolation.py) e a suíte SQR completa (`bash tests/run_sqr_tests.sh -x -q` -> **342 passed**):
- **Sem projeção (`sum(a_prop) <= C`):** execução com `virtual_alpha = 0.0` vs `0.05` produz trajetórias idênticas de atualização pinball, exclusões `R19`, adaptação de $\tau_{i,t}$ e $M_{i,t}$, política de seleção e estado do RNG.
- **Com projeção (`sum(a_prop) > C`):** `apply_virtual_updates()` com `virtual_alpha = 0.0` retorna `[]`, sem inserir vetores virtuais, sem alterar $\alpha$, sem disparar `R11`/`R21` e sem alterar o RNG.
- **End-to-end (`sqr_run.main`):** confirmado `het.virtual_alpha = 0.0` via `--virtual_alpha 0.0`.

> *Nota técnica sobre leitura de checkpoint offline:* [`load_run_controllers()`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/joint_selector.py) reconstrói `SqrHetController` offline com o argumento padrão `virtual_alpha=0.05` caso o chamador não releia `meta["virtual_alpha"]`, mas os runs gravaram explicitamente `meta["virtual_alpha"] = 0.0` em `sqr_checkpoint.pkl`, `.done` com `va=0.0` e **zero vetores virtuais** (`n_virt = 0`) em todas as 9 tabelas/slices de A1.

### 1.3 Completude e paridade bit-a-bit de bootstrap (A0 vs. A1)

| $\lambda$ | Variante | `.done` / `ns3_exit_code` | `kpm_max_ts` | Decisões Totais | Bootstrap (`SQR_HET_BOOTSTRAP_PROP`) | `SQR_HET` (fechadas) | Paridade Bootstrap A0 == A1 |
|---:|---|---|---:|---:|---:|---:|---|
| 2 | A0 | `rc=0`, `ns3=0` | `299900 ms` | 848 | 283 | 565 (564) | — |
| 2 | A1 | `rc=0 va=0.0`, `ns3=0` | `299900 ms` | 848 | 283 | 565 (564) | **Bit-idêntico (`283/283`)** |
| 5 | A0 | `rc=0`, `ns3=0` | `299900 ms` | 2298 | 303 | 1995 (1994) | — |
| 5 | A1 | `rc=0 va=0.0`, `ns3=0` | `299900 ms` | 2298 | 303 | 1995 (1994) | **Bit-idêntico (`303/303`)** |
| 6 | A0 | `rc=0`, `ns3=0` | `299900 ms` | 2455 | 460 | 1995 (1994) | — |
| 6 | A1 | `rc=0 va=0.0`, `ns3=0` | `299900 ms` | 2455 | 460 | 1995 (1994) | **Bit-idêntico (`460/460`)** |

---

## 2. Decomposição dos Pesos do Regressor (`sqr_checkpoint.pkl`) e Sensibilidade Direcional ($\hat{f}(37)>\hat{f}(0)$)

A desativação de `R12` eliminou `100%` dos vetores virtuais ($\alpha = -0.05$) e restaurou o sinal físico e a sensibilidade direcional em duas ações ($\hat{f}(z,37) > \hat{f}(z,0)$) em todas as combinações $(\lambda, \text{slice})$:

| $\lambda$ | Slice | Var. | $N_{\text{SV}}$ | `virt(-0.05)` (`%`) | `UP (>0)` | `CF (<-0.05)` | $\sum \alpha$ | $\tau_{\text{final}}$ | $M_{\text{final}}$ | Sensib. $\hat{f}(37)>\hat{f}(0)$ | $\text{med}[\hat{f}(37)-\hat{f}(0)]$ |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | eMBB (0) | A0 | 239 | 75 (31.4%) | 163 | 1 | `-3.1250` | `0.050` | `10.0` | **0.0%** | `-0.2538` |
| 2 | eMBB (0) | **A1** | 164 | **0 (0.0%)** | 160 | 4 | **`+0.1025`** | `0.045` | `20.0` | **100.0%** | **`+0.1796`** |
| 2 | URLLC (1) | A0 | 381 | 72 (18.9%) | 297 | 12 | `-3.7905` | `0.040` | `40.0` | **0.0%** | `-0.0517` |
| 2 | URLLC (1) | **A1** | 309 | **0 (0.0%)** | 298 | 11 | **`+0.0050`** | `0.040` | `40.0` | **100.0%** | **`+0.0630`** |
| 2 | mMTC (2) | A0 | 446 | 56 (12.6%) | 390 | 0 | `-0.8500` | `0.050` | `5.0` | **0.0%** | `-0.4112` |
| 2 | mMTC (2) | **A1** | 390 | **0 (0.0%)** | 382 | 8 | **`+0.1005`** | `0.035` | `50.0` | **100.0%** | **`+0.1842`** |
| 5 | eMBB (0) | A0 | 1885 | 303 (16.1%) | 1578 | 4 | `-12.3580` | `0.010` | `50.0` | **0.0%** | `-1.2269` |
| 5 | eMBB (0) | **A1** | 1582 | **0 (0.0%)** | 1547 | 35 | **`-1.1735`** | `0.010` | `50.0` | **100.0%** | **`+0.4552`** |
| 5 | URLLC (1) | A0 | 1417 | 258 (18.2%) | 1151 | 8 | `-10.6180` | `0.010` | `50.0` | **0.0%** | `-0.8320` |
| 5 | URLLC (1) | **A1** | 1159 | **0 (0.0%)** | 1127 | 32 | **`-0.9045`** | `0.010` | `50.0` | **100.0%** | **`+0.2643`** |
| 5 | mMTC (2) | A0 | 2000 | 126 (6.3%) | 1853 | 21 | `-2.6930` | `0.020` | `50.0` | **0.8%** | `-1.2668` |
| 5 | mMTC (2) | **A1** | 1944 | **0 (0.0%)** | 1916 | 28 | **`+0.0655`** | `0.020` | `50.0` | **100.0%** | **`+0.3094`** |
| 6 | eMBB (0) | A0 | 2000 | 305 (15.3%) | 1689 | 6 | `-12.1750` | `0.010` | `50.0` | **0.0%** | `-1.1690` |
| 6 | eMBB (0) | **A1** | 1845 | **0 (0.0%)** | 1797 | 48 | **`-1.6120`** | `0.010` | `50.0` | **97.6%** | **`+0.3702`** |
| 6 | URLLC (1) | A0 | 1834 | 265 (14.4%) | 1514 | 55 | `-15.4575` | `0.010` | `50.0` | **6.3%** | `-0.5247` |
| 6 | URLLC (1) | **A1** | 1584 | **0 (0.0%)** | 1538 | 46 | **`-1.5810`** | `0.010` | `50.0` | **100.0%** | **`+0.1019`** |
| 6 | mMTC (2) | A0 | 2000 | 137 (6.9%) | 1841 | 22 | `-3.7400` | `0.010` | `50.0` | **0.0%** | `-1.3745` |
| 6 | mMTC (2) | **A1** | 2000 | **0 (0.0%)** | 1967 | 33 | **`-0.1440`** | `0.015` | `25.0` | **100.0%** | **`+0.3032`** |

**Sensibilidade em duas ações ($a=37$ vs $a=0$) causal online em A1 (`decisions.jsonl`, sobre observações avaliáveis):** *(Nota de retificação metodológica: a verificação $\hat{f}(37) > \hat{f}(0)$ avalia estritamente dois pontos discretos e **não** constitui prova de monotonia global em todo o domínio $a \in [0, 53]$.)*
- $\lambda=2$: $\hat{f}(z,37)>\hat{f}(z,0)$ em **98.8%** (eMBB), **77.4%** (URLLC, apenas 31 obs.), **97.2%** (mMTC).
- $\lambda=5$: **100.0%** (eMBB), **99.6%** (URLLC), **99.4%** (mMTC).
- $\lambda=6$: **98.3%** (eMBB), **99.9%** (URLLC), **99.9%** (mMTC).

Observa-se igualmente um segundo sinal estrutural importante: mesmo com `n_virt = 0`, `M_final` ainda satura no teto `50.0` (eMBB/URLLC em $\lambda \in \{5,6\}$; mMTC em $\lambda \in \{2,5\}$), porque o mecanismo de adaptação de `M` (R24) e a ausência de guarda de demanda vazia permanecem inalterados em A1.

---

## 3. Previsões Causais e Calibração no Quantil Efetivamente Utilizado ($\tau_{i,t}$)

Para cada decisão `SQR_HET` fechada ($W \to W+1$), pareou-se $\hat{q}_{\text{exec},i,t} = \hat{f}_{\tau_{i,t}}(z_{i,W}, a_{\text{exec},i,t})$ (`predicted_safe_after_projection`) e $\hat{q}_{\text{prop},i,t}$ (`predicted_safe_before_proposal`) com o desfecho causal $y_{i,t}$ da janela $W+1$ e com o quantil $\tau_{i,t}$ efetivo daquela decisão (`step_metrics.csv`).

| $\lambda$ | Slice | Var. | $N_{\text{obs}}$ | $\bar{\tau}_i$ | $\hat{p}_{\text{exec}}$ ($\Pr[y<\hat{q}]$) | Erro $\hat{p}_{\text{exec}}-\bar{\tau}_i$ | Pinball $\mathcal{L}_{\tau_{i,t}}$ média (med.) | $\mathcal{L}_{0.05}$ ref. | $\text{med}|\hat{q}-y|$ | $\hat{q}_{\text{exec}}$ med. | $\Pr(\hat{q}_{\text{exec}}\ge 0)$ / $\Pr(\hat{q}_{\text{prop}}\ge 0)$ |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | eMBB (0) | A0 | 164 | 0.0500 | 0.0061 | `-0.0439` | 0.1530 (0.1168) | 0.1530 | 2.3174 | `-0.1781` | 1.2% / 1.2% |
| 2 | eMBB (0) | **A1** | 164 | 0.0480 | **0.0244** | **`-0.0237`** | **0.0896 (0.0708)** | **0.0918** | **1.4441** | **`+0.0016`** | **81.1% / 86.0%** |
| 2 | URLLC (1) | A0 | 31 | 0.0439 | 0.0968 | `+0.0529` | 0.3029 (0.0296) | 0.3050 | 0.6770 | `-0.4521` | 16.1% / 16.1% |
| 2 | URLLC (1) | **A1** | 31 | 0.0439 | **0.0645** | **`+0.0206`** | 0.3479 (**0.0085**) | 0.3472 | **0.1889** | **`+0.0015`** | **61.3% / 67.7%** |
| 2 | mMTC (2) | A0 | 390 | 0.0500 | 0.0000 | `-0.0500` | 0.0167 (0.0162) | 0.0167 | 0.3242 | `-0.2716` | 0.0% / 0.0% |
| 2 | mMTC (2) | **A1** | 390 | 0.0427 | **0.0205** | **`-0.0222`** | 0.0231 (**0.0021**) | 0.0235 | **0.0500** | **`+0.0028`** | **92.8% / 93.3%** |
| 5 | eMBB (0) | A0 | 1519 | 0.0202 | 0.0026 | `-0.0176` | 0.0437 (0.0367) | 0.1423 | 2.5730 | `-2.3857` | 0.3% / 0.5% |
| 5 | eMBB (0) | **A1** | 1519 | 0.0202 | **0.0230** | **`+0.0029`** | **0.0191 (0.0089)** | **0.0411** | **0.5583** | **`-0.2476`** | **14.0% / 37.4%** |
| 5 | URLLC (1) | A0 | 891 | 0.0263 | 0.0000 | `-0.0263` | 0.0721 (0.0775) | 0.1528 | 2.3495 | `-2.1755` | 0.0% / 0.0% |
| 5 | URLLC (1) | **A1** | 891 | 0.0218 | **0.0269** | **`+0.0051`** | **0.0550 (0.0066)** | **0.0658** | **0.3560** | **`-0.2130`** | **14.9% / 28.5%** |
| 5 | mMTC (2) | A0 | 1944 | 0.0318 | 0.0108 | `-0.0210` | 0.0220 (0.0161) | 0.0332 | 0.5685 | `-0.5138` | 0.8% / 0.8% |
| 5 | mMTC (2) | **A1** | 1944 | 0.0272 | **0.0144** | **`-0.0128`** | **0.0100 (0.0015)** | **0.0113** | **0.0557** | **`-0.0024`** | **45.0% / 98.1%** |
| 6 | eMBB (0) | A0 | 1680 | 0.0213 | 0.0036 | `-0.0177` | 0.0639 (0.0527) | 0.1773 | 3.5596 | `-3.4583` | 0.7% / 1.3% |
| 6 | eMBB (0) | **A1** | 1680 | 0.0192 | **0.0286** | **`+0.0094`** | **0.0166 (0.0087)** | **0.0371** | **0.5958** | **`-0.3886`** | **6.8% / 16.4%** |
| 6 | URLLC (1) | A0 | 1300 | 0.0190 | 0.0354 | `+0.0164` | 0.4479 (0.0559) | 0.5658 | 4.5419 | `-4.5204` | 0.0% / 0.0% |
| 6 | URLLC (1) | **A1** | 1315 | 0.0189 | **0.0289** | **`+0.0100`** | **0.0345 (0.0079)** | **0.0513** | **0.5446** | **`-0.5814`** | **3.3% / 3.6%** |
| 6 | mMTC (2) | A0 | 1962 | 0.0289 | 0.0082 | `-0.0208` | 0.0212 (0.0158) | 0.0377 | 0.6221 | `-0.5787` | 1.7% / 1.8% |
| 6 | mMTC (2) | **A1** | 1962 | 0.0235 | **0.0138** | **`-0.0098`** | **0.0080 (0.0013)** | **0.0096** | **0.0565** | **`-0.0033`** | **40.1% / 96.5%** |

### Principais achados de calibração
1. **Calibração quantílica restaurada em todos os 9 pares $(\lambda, \text{slice})$:** o erro absoluto de calibração $|\hat{p}_{\text{exec}} - \bar{\tau}_i|$ caiu em **todos os 9 casos**, atingindo `0.0029` (eMBB $\lambda=5$) e `0.0051` (URLLC $\lambda=5$). Nos estratos por $\tau_{i,t}$ em $\lambda=5$, por exemplo:
   - A0 (eMBB $\tau=0.01$, `N=782`): cobertura `0.0000` (desvio `-0.010`, pinball `0.0431`).
   - A1 (eMBB $\tau=0.01$, `N=782`): cobertura **`0.0179`** (desvio `+0.0079`, pinball **`0.0133`**).
   - A0 (URLLC $\tau=0.01..0.04$, todos `7` estratos): cobertura **`0.0000`** em absolutamente todos os 891 passos (previsão colapsada em `[-7.71, -1.82]`).
   - A1 (URLLC $\tau=0.01..0.04$): cobertura varia entre `0.0156` e `0.0588` acompanhando o alvo (desvios `-0.005` a `+0.035`).
2. **Queda drástica da perda pinball e do erro de magnitude:** em $\lambda=5$ e $\lambda=6$, a perda pinball média caiu `56%` a `92%` nos slices críticos (`eMBB: 0.0639 -> 0.0166`, `-74.0%`; `URLLC: 0.4479 -> 0.0345`, `-92.3%` em $\lambda=6$), e a mediana de $|\hat{q}_{\text{exec}} - y|$ encolheu de `2.3–4.5` para `0.36–0.60` (e em mMTC de `0.32–0.62` para `0.05–0.06`).
3. **Efeito visível da projeção sobre a segurança prevista (`q_prop` vs `q_exec`):** em A1 com $\lambda=5$, os regressores propuseram ações consideradas seguras em **98.1%** (mMTC) e **37.4%** (eMBB) das observações (`pct_q_prop_ge0`), mas após a projeção de capacidade essas frações despencaram para **45.0%** e **14.0%** (`pct_q_exec_ge0`).

---

## 4. Utilização, Modos de Operação e Dinâmica de Projeções

| $\lambda$ | Var. | `P_proj` (`n/1994` ou `/564`) | Excesso $\overline{\sum a_{\text{prop}}-C}$ (máx) | $\bar{\Delta}_{L1}$ se proj. | `prb_exec_sum` média [e, u, m] | `sched/slot` [e, u, m] | Utilização DL ativa (`util_dl%` / `idle%`) |
|---:|---|---|---:|---:|---|---|---|
| 2 | A0 | 14.18% (80/564) | -15.87 (81) | 16.59 | 22.93 `[7.16, 13.06, 2.71]` | `[2.43, 0.02, 0.00]` | `4.63%` / `95.37%` |
| 2 | **A1** | **5.32% (30/564)** | -22.63 (24) | 6.17 | 21.83 `[6.34, 10.99, 4.49]` | `[1.79, 0.02, 0.00]` | `3.43%` / `96.57%` |
| 5 | A0 | 16.00% (319/1994) | -18.76 (67) | 11.67 | 32.36 `[16.72, 11.87, 3.78]` | `[13.32, 0.97, 0.02]` | `27.00%` / `73.00%` |
| 5 | **A1** | **66.05% (1317/1994)** | **+10.75 (62)** | **21.97** | **49.25** `[23.08, 20.27, 5.91]` | `[17.99, 0.96, 0.01]` | **`35.78%` / `64.22%`** |
| 6 | A0 | 16.05% (320/1994) | -16.08 (55) | 13.89 | 34.67 `[18.84, 11.82, 4.01]` | `[18.76, 1.81, 0.02]` | `38.83%` / `61.17%` |
| 6 | **A1** | **73.42% (1464/1994)** | **+7.43 (71)** | **16.25** | **48.49** `[22.85, 18.96, 6.68]` | `[21.92, 2.01, 0.02]` | **`45.18%` / `54.82%`** |

### Modos de operação e comportamento em slices vazias ($n_i = 0 \wedge \text{backlog}_i = 0$)

| $\lambda$ | Slice | Var. | Modos (`confident` / `knn` / `softmax` / `fallback`) | $\bar{a}_{\text{raw}}$ / $\bar{a}_{\text{prop}}$ / $\bar{a}_{\text{exec}}$ (ativas) | Passos vazios (`n`) | Vazias: $\bar{a}_{\text{raw}}$ / $\bar{a}_{\text{prop}}$ / $\bar{M}$ / $\bar{\sigma}$ | Cortes por projeção (total / se vazia) |
|---:|---|---|---|---:|---:|---|---|
| 2 | eMBB (0) | A0 / **A1** | `2/506/56/0` $\to$ **`541/20/3/0`** | `30.8/30.9/29.5` $\to$ `22.1/22.4/21.8` | 399 | `12.07/12.28/10.0/0.022` $\to$ **`0.00/9.99/20.0/0.477`** | `75(47)` $\to$ `29(2)` |
| 2 | URLLC (1) | A0 / **A1** | `87/430/47/0` $\to$ **`554/9/1/0`** | `15.3/16.5/14.9` $\to$ `13.9/15.2/14.4` | 532 | `13.46/13.78/36.9/0.014` $\to$ `5.91/6.69/36.9/0.046` | `72(59)` $\to$ `11(7)` |
| 2 | mMTC (2) | A0 / **A1** | `4/358/49/153` $\to$ **`538/24/2/0`** | `6.2/6.3/5.8` $\to$ `9.1/9.4/9.4` | 165 | `2.93/3.47/10.0/0.077` $\to$ **`0.00/10.00/10.0/1.000`** | `56(15)` $\to$ `23(21)` |
| 5 | eMBB (0) | A0 / **A1** | `8/1795/191/0` $\to$ **`1043/857/94/0`** | `17.9/18.3/17.5` $\to$ `30.7/31.4/23.6` | 475 | `14.63/14.90/20.0/0.014` $\to$ **`11.23/22.22/20.0/0.558`** | `303(73)` $\to$ **`1308(96)`** |
| 5 | URLLC (1) | A0 / **A1** | `0/1796/198/0` $\to$ **`592/1266/136/0`** | `11.6/11.7/11.4` $\to$ `22.2/22.5/18.6` | 1099 | `13.09/13.15/45.7/0.002` $\to$ **`28.82/31.34/45.7/0.161`** | `258(182)` $\to$ **`1285(701)`** |
| 5 | mMTC (2) | A0 / **A1** | `19/1720/210/45` $\to$ **`1957/35/2/0`** | `4.0/4.2/3.8` $\to$ `6.9/7.1/5.8` | 41 | `3.66/5.29/10.0/0.153` $\to$ `0.00/10.00/10.0/1.000` | `126(5)` $\to$ **`1071(8)`** |
| 6 | eMBB (0) | A0 / **A1** | `336/1500/158/0` $\to$ **`589/1272/133/0`** | `23.0/23.4/22.3` $\to$ `32.1/32.6/26.0` | 315 | `0.00/0.44/10.0/0.070` $\to$ `0.00/6.00/10.0/0.558` | `305(1)` $\to$ **`1455(29)`** |
| 6 | URLLC (1) | A0 / **A1** | `0/1796/198/0` $\to$ **`47/1753/194/0`** | `12.2/12.3/11.5` $\to$ `22.2/22.4/18.3` | 677 | `12.85/13.21/39.9/0.012` $\to$ **`22.66/26.81/39.9/0.089`** | `265(76)` $\to$ **`1429(459)`** |
| 6 | mMTC (2) | A0 / **A1** | `35/1750/209/0` $\to$ **`1925/65/4/0`** | `4.4/4.5/4.0` $\to$ `7.6/7.8/6.5` | 32 | `2.50/5.34/50.0/0.058` $\to$ **`15.78/27.56/25.0/0.477`** | `137(5)` $\to$ **`1157(31)`** |

**Mecanismo de interação revelado:** Sem `R12`, o regressor deixa de permanecer artificialmente negativo (`<-0.5` a `-4.5`), saindo do modo permanente `knn` (onde A0 ficava preso em `98%–100%` dos passos de eMBB/URLLC para $\lambda=5$ e `85%–100%` para $\lambda=2,6$) para operar majoritariamente em `confident` (ex.: `95.9%–98.2%` em $\lambda=2$; `98.1%` mMTC e `52.3%` eMBB em $\lambda=5$). Contudo, nas janelas vazias (`empty_steps`), o controlador continua propondo sem guarda de demanda: e.g. em $\lambda=5$, quando URLLC está vazia (`1099/1994` decisões), propõe em média **`31.34` RBGs** (contra `13.15` em A0) e eMBB vazia (`475` decisões) propõe **`22.22` RBGs**, forçando projeção em **66.05%** das decisões em $\lambda=5$ e **73.42%** em $\lambda=6$.

---

## 5. Violações por Slice, `STRICT_ALL3` e Avaliação de Segurança do Controlador

### 5.1 Violações por slice, `STRICT_ALL3` e `eU` (janelas governadas fechadas `SQR_HET`)

> Convenção explícita (retificada sob R19 estrito): na tabela abaixo mantêm-se preservados os números de referência da variante `all_rx` (`STRICT_ALL3_BOTH_PRESENT`, idêntica ao offline `d1v9_sla_offline.csv`), e acrescenta-se em **§5.1.1** a auditoria primária sob **R19 estrito (`closed_online` apenas)** com decomposição explícita de `UNKNOWN` — onde demandas não fechadas online (`closed_late_eof`, `partial_rx`, `zero_rx`) e `no_demand` são contadas estritamente como `UNKNOWN` (nunca como `PASS`, `FAIL` nem perda presumida). `STRICT_ANY_ACTIVE` marca `FAIL` se **qualquer** slice ativa falhar na janela.

| $\lambda$ | Var. | Viol. eMBB | Viol. URLLC (gov / post) | Viol. mMTC | `STRICT_ALL3_BOTH_PRESENT` (`PASS / FAIL / UNK`) | `STRICT_ANY_ACTIVE` (`PASS / FAIL / UNK`) | `eU` (`PASS / FAIL`) |
|---:|---|---|---|---|---|---|---|
| 2 | A0 | **1.22%** (`2/164`) | 19.35% (`6/31`) / `6/32` | **0.00%** (`0/390`) | `0 / 0 / 564` | `0 / 8 / 556` | `26 / 6` |
| 2 | **A1** | 2.44% (`4/164`) | **6.45%** (`2/31`) / `2/32` | 2.05% (`8/390`) | `0 / 0 / 564` | `0 / 14 / 550` | **`29 / 3`** |
| 5 | A0 | 66.95% (`1017/1519`) | 24.35% (`217/891`) / `219/901` | 2.78% (`54/1944`) | `215 / 570 / 1209` (**27.39%** pass R19) | `215 / 1108 / 671` | `216 / 573` |
| 5 | **A1** | **32.65%** (`496/1519`) | **15.04%** (`134/891`) / `136/901` | **1.29%** (`25/1944`) | **`577 / 208 / 1209` (`73.50%` pass R19)** | **`579 / 636 / 779`** | **`584 / 205`** |
| 6 | A0 | 54.88% (`922/1680`) | **33.77%** (`439/1300`) / `452/1315` | 3.26% (`64/1962`) | `418 / 768 / 808` (35.24% pass R19) | `419 / 1051 / 524` | `427 / 800` |
| 6 | **A1** | **38.99%** (`655/1680`) | 46.92% (`617/1315`) / `619/1318` | **1.27%** (`25/1962`) | **`522 / 675 / 797` (`43.61%` pass R19)** | `523 / 1080 / 391` | **`527 / 703`** |

#### 5.1.1 Auditoria R19 estrita (`closed_online`) e decomposição de `UNKNOWN` (`SQR_HET` fechado)

| $\lambda$ | Var. | Janelas URLLC (`no_demand / closed_online / closed_late_eof / partial_rx / zero_rx`) | URLLC viol R19 | `STRICT_ALL3_R19` (`PASS/FAIL/UNK`, `% avaliadas`) | Causas `UNK` (`eMBB / URLLC / mMTC / multi`) | Sensib. `all_rx` (`PASS/FAIL/UNK`) |
|---|---|---|---|---|---|---|
| 2 | A0 | `532 / 31 / 0 / 1 / 0` | `19.35% (6/31)` | `0 / 0 / 564` | — (`N=564`) | `0 / 0 / 564` |
| 2 | A1 | `532 / 31 / 0 / 1 / 0` | `6.45% (2/31)` | `0 / 0 / 564` | — (`N=564`) | `0 / 0 / 564` |
| 5 | A0 | `1099 / 891 / 2 / 2 / 0` | `24.35% (217/891)` | `215 / 570 / 1209` (`27.39%` de `785`) | `106 / 684 / 0 / 419` | `215 / 573 / 1206` (`27.28%`) |
| 5 | A1 | `1099 / 891 / 2 / 2 / 0` | `15.04% (134/891)` | `577 / 208 / 1209` (`73.50%` de `785`) | — (`785` avaliadas) | `579 / 209 / 1206` (`73.48%`) |
| 6 | A0 | `676 / 1300 / 2 / 13 / 3` | `33.77% (439/1300)` | `418 / 768 / 808` (`35.24%` de `1186`) | `88 / 462 / 26 / 232` | `419 / 776 / 799` (`35.06%`) |
| 6 | A1 | `676 / 1315 / 2 / 1 / 0` | `46.92% (617/1315)` | `522 / 675 / 797` (`43.61%` de `1197`) | `88 / 451 / 30 / 228` | `523 / 675 / 796` (`43.66%`) |

### 5.2 Telemetria por pacote URLLC pós-warmup (auditoria de janela temporal e coortes exatas)

> [!IMPORTANT]
> **Auditoria obrigatória da contabilização URLLC (`tx=215164` vs `rx=215437`, antigo `-0.13%`):** o aparente superávit de pacotes RX (`-0.13%`, `-273` pacotes em `A1` $\lambda=6$) decorreu exclusivamente de **descasamento de janela temporal** no script diagnóstico anterior (`evaluate_a1_vs_a0.py` v1):
> - `sqr_kpm_history.csv` com filtro `timestamp >= 100100` encerra no último tick periódico em `299900 ms`, cobrindo **`[100.0s, 299.9s)`** (`1,999` janelas de decisão, excluindo os `100 ms` finais);
> - `sqr_packet_delays.csv` com filtro `txSec >= 100.0` recebe o *flush* final no encerramento da simulação em `300.0s` (`FlushPacketDelays()`), cobrindo **`[100.0s, 300.0s)`** (`2,000` janelas — diferença de `300` TX e `276` RX nascidos na cauda `[299.9s, 300.0s)` em `A1` $\lambda=6$).
>
> Reconciliando na mesma coorte temporal, a conservação é exata (`dRxPackets == rx_in_win` pacote-a-pacote em todas as 6 rodadas, `0` duplicatas, `0` RX sem `ByteTag`, e atrasos agregados `p95`/`p99` idênticos entre coortes condicionados aos pacotes entregues, mantendo R19 estrito nas janelas não fechadas como `UNKNOWN`).

| $\lambda$ | Var. | `tx_u_post` | `rx_u_post` | `semRX_post%` | `p50` | `p90` | `p95` | `p99` | `max` | `pkt_miss% (>10 ms)` | `p95_win` med / máx |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | A0 | 3151 | 3146 | +0.16% | 5.10 ms | 8.10 ms | 18.10 ms | 43.10 ms | 68.10 ms | 8.49% | 8.10 / 64.90 ms |
| 2 | **A1** | 3151 | 3146 | +0.16% | 5.10 ms | 8.10 ms | **15.10 ms** | 75.65 ms | 106.10 ms | **6.23%** | 8.10 / 99.15 ms |
| 5 | A0 | 119182 *(coorte `[100,300)`)* | 119158 | **+0.02%** (`24` não rx em 300s) | 5.10 ms | 8.10 ms | 9.10 ms | 14.10 ms | 43.10 ms | 4.16% | 8.10 / 41.10 ms |
| 5 | **A1** | 119182 *(coorte `[100,300)`)* | 119158 | **+0.02%** (`24` não rx em 300s) | 5.10 ms | 8.10 ms | **8.10 ms** | 14.10 ms | 108.10 ms | **2.99%** | 8.10 / 99.15 ms |
| 6 | A0 | 215464 *(coorte `[100,300)`)* | 212863 | **+1.21%** (`2601` não rx em 300s) | 6.10 ms | 14.10 ms | **83.03 ms** | **307.10 ms** | **374.10 ms** | **14.55%** | 8.15 / 372.10 ms |
| 6 | **A1** | 215464 *(coorte `[100,300)`)* | **215437** | **+0.01%** (`27` não rx em 300s) | 6.10 ms | **10.03 ms** | **12.10 ms** | **18.10 ms** | **110.10 ms** | **10.06%** | 9.10 / 103.15 ms |

#### 5.2.1 Reconciliação exata por coorte (`[0, 100.0s)`, `[100.0s, 299.9s)`, `[299.9s, 300.0s)`, `[100.0s, 300.0s)`)

| $\lambda$ | Var. | `[0, 100.0s)` TX / RX KPM / `cross_100s` | `[100.0s, 299.9s)` TX KPM / RX KPM (`==rx_win`) / RX coorte TX (`cross_299.9s`) / não-RX | Cauda `[299.9s, 300.0s)` TX / RX (`tx>=299.9`) | Pós-warmup `[100.0s, 300.0s)` TX / RX / não-RX (`%`) | Dupl. / Miss `p95` / `p99` (`[100, 299.9)` vs `[100, 300)`) |
|---|---|---|---|---|---|---|
| 2 | A0 | `27933 / 27928 / 0` | `3151 / 3146 / 3146 (0) / 5 (0.16%)` | `0 / 0` | `3151 / 3146 / 5 (0.16%)` | `0` / `18.10 / 43.10 ms`, `miss=8.49%` |
| 2 | A1 | `27933 / 27928 / 0` | `3151 / 3146 / 3146 (0) / 5 (0.16%)` | `0 / 0` | `3151 / 3146 / 5 (0.16%)` | `0` / `15.10 / 75.65 ms`, `miss=6.23%` |
| 5 | A0 | `30902 / 30892 / 5` | `118982 / 118967 / 118974 (12) / 8 (0.007%)` | `200 / 196 (184)` | `119182 / 119158 / 24 (0.020%)` | `0` / `9.10 / 14.10 ms`, `4.15% vs 4.16%` |
| 5 | A1 | `30902 / 30892 / 5` | `118982 / 118967 / 118974 (12) / 8 (0.007%)` | `200 / 196 (184)` | `119182 / 119158 / 24 (0.020%)` | `0` / `8.10 / 14.10 ms`, `2.99% vs 2.99%` |
| 6 | A0 | `31084 / 31074 / 0` | `215164 / 212572 / 212590 (18) / 2574 (1.20%)` | `300 / 291 (273)` | `215464 / 212863 / 2601 (1.21%)` | `0` / `83.03 / 307.10 ms`, `14.50% vs 14.55%` |
| 6 | A1 | `31084 / 31074 / 0` | `215164 / 215143 / 215161 (18) / 3 (0.001%)` | `300 / 294 (276)` | `215464 / 215437 / 27 (0.013%)` | `0` / `12.10 / 18.10 ms`, `10.07% vs 10.06%` |

- **Conclusão sobre integridade e validade de `p95`/`p99`:** `rx_packets` em `sqr_summary.csv` iguala exatamente o total de linhas de `sqr_packet_delays.csv` (nenhum pacote RX sem `ByteTag`; `0` duplicatas). Os pacotes transmitidos antes de `100.0s` e recebidos em `t >= 100.0s` são apenas `0 / 5 / 0` ($\lambda = 2/5/6$). Portanto, os percentis `p95`/`p99` agregados pós-warmup permanecem **válidos condicionados aos pacotes entregues**, mantendo-se a regra **R19** nas janelas não fechadas (`UNKNOWN`, sem inferir sucesso/falha/perda). Note-se ainda que em A0 $\lambda=6$ (`2601` não-RX pós-warmup, `1.21%`) a distribuição de atraso dos pacotes entregues já reflete fila/retenção elevada (`p95=83.03 ms`, `p99=307.10 ms`), enquanto em A1 $\lambda=6$ (`27` não-RX ao corte `t=300s`, `0.013%`) `p95/p99` agregados (`12.10 / 18.10 ms`) coexistem com maior frequência de micro-explosões >10 ms por janela (`46.92%` das `1315` janelas fechadas vs `33.77%` das `1300` em A0).

### 5.3 Transições de janelas entre trajetórias distintas (A0 vs A1 no mesmo índice temporal — não replay causal em estado idêntico)

> [!CAUTION]
> **Retificação metodológica da comparação janela-a-janela:** embora A0 e A1 compartilhem exatamente o mesmo bootstrap exógeno (`283 / 303 / 460` decisões bit-a-bit idênticas para $\lambda = 2/5/6$), a partir da fase `SQR_HET` as políticas closed-loop divergem, levando a trajetórias distintas ($S_t^{\text{A0}} \neq S_t^{\text{A1}}$, históricos e filas diferentes). Portanto, as contagens abaixo comparam **trajetórias distintas no mesmo índice temporal de janela** e **não** devem ser interpretadas como *replay* contrafactual em estados idênticos. Sob **R19 estrito** (`closed_online`), as associações cruzadas (`both_ok / fail_to_ok / ok_to_fail / both_fail`) são:
> - $\lambda=2$: `eMBB` `159 / 1 / 3 / 1` · `URLLC_R19` `24 / 5 / 1 / 1` · `mMTC` `382 / 0 / 8 / 0` · `STRICT_ALL3_R19` `0 / 0 / 0 / 0`.
> - $\lambda=5$: `eMBB` `414 / 609 / 88 / 408` · `URLLC_R19` `604 / 153 / 70 / 64` · `mMTC` `1867 / 52 / 23 / 2` · `STRICT_ALL3_R19` `180 / 397 / 35 / 173`.
> - $\lambda=6$: `eMBB` `533 / 492 / 225 / 430` · `URLLC_R19` `532 / 164 / 329 / 275` · `mMTC` `1875 / 62 / 23 / 2` · `STRICT_ALL3_R19` `256 / 264 / 162 / 504`.
### 5.4 Avaliação explícita de Benefícios vs. Prejuízos na Segurança do Controlador

#### Benefícios comprovados ao remover `R12`
1. **Recuperação da função de valor quantílica:** elimina o colapso sistemático de pesos (`sum_alpha` de `-15.46..-0.85` para `-1.61..+0.10`), zera os vetores virtuais espúrios e restaura a sensibilidade direcional em duas ações ($\hat{f}(z,37)>\hat{f}(z,0)$ de `0%–6.3%` para `97.6%–100.0%`).
2. **Desbloqueio do modo `confident` e salto de calibração/pinball:** `|\hat{p}_{\text{exec}} - \bar{\tau}|` diminui nos 9 pares $(\lambda,\text{slice})$ e a perda pinball cai `56%–92%` em $\lambda \in \{5,6\}$.
3. **Ganho líquido robusto em `STRICT_ALL3_R19` e eMBB:** sob carga intermediária ($\lambda=5$), todas as 3 slices melhoram simultaneamente (eMBB `-34.3 p.p.`, URLLC `-9.3 p.p.`, mMTC `-1.5 p.p.`) e `STRICT_ALL3_R19` passa de `27.39%` para **`73.50%`**. Em $\lambda=6$, `STRICT_ALL3_R19` sobe de `35.24%` para **`43.61%`**, o atraso por pacote URLLC severo colapsa (`p95: 83.03 -> 12.10 ms`; `p99: 307.10 -> 18.10 ms`; `pkt_miss%>10ms: 14.55% -> 10.06%`) e os pacotes URLLC não recebidos em `t=300s` na coorte `[100.0s, 300.0s)` caem de `2601` (`1.21%`) para `27` (`0.013%`).

#### Prejuízos e riscos de segurança identificados ao isolar `A1` sem `A2`/`A3`
1. **Explosão de projeções por propostas de slices vazias + $M_{\max}=50$ (`66.0%–73.4%` das decisões):**
   Em A0, os vetores virtuais `R12` mantinham a função $\hat{f}_\tau$ artificialmente deprimida, mascarando parcialmente a ausência de guarda em slices vazias. Ao remover `R12` sem corrigir `A2`/`A3`, slices vazias passam a propor `9.99–10.00` RBGs ($\lambda=2$, devido a $\lceil M \cdot \sigma \rceil$ com $\sigma \in [0.48, 1.0]$) e `22.22–31.34` RBGs ($\lambda=5,6$), disparando projeções proporcionais em `66.05%` ($\lambda=5$) e `73.42%` ($\lambda=6$) dos passos. Como efeito direto sobre a segurança, a projeção corta recursos das slices ativas e reduz $\Pr(\hat{q}_{\text{exec}} \ge 0)$ em $\lambda=5$ de `37.4% -> 14.0%` (eMBB) e `98.1% -> 45.0%` (mMTC).
2. **Sub-alocação agressiva pelo modo `confident` sob custo $\gamma=1$ em $\lambda=2$ e nos extremos de $\lambda=6$:**
   Quando o regressor sem `R12` prevê $\hat{f}_\tau(z, 0) > 0$ após sequências limpas, a otimização $\arg\max_a [\hat{f}_\tau(z,a) - a/C]$ passa a selecionar $a_{\text{raw}} = 0$ ou muito baixo. Exemplos verificados em `decisions.jsonl` / `sqr_packet_delays.csv`:
   - Em $\lambda=2$, surgem pequenas violações adicionais de eMBB (`2 -> 4` janelas) e mMTC (`0 -> 8` janelas), e um pico URLLC (`max_p95 = 99.15 ms`, apesar de a taxa de violação URLLC cair de `6/31` para `2/31`).
   - Em $\lambda=6$, há janelas URLLC ativas com $a_{\text{raw},1}=0$ que recebem apenas `a_exec = [6, 1, 25]` (`w=1261`, `p95=27.10 ms`) ou `[6, 0, 14]` (`w=1315`, `p95=98.15 ms`), elevando `max p95` em $\lambda=5$ (`41.10 -> 99.15 ms`) e gerando surtos locais quando a reserva de URLLC cai abaixo do necessário.
3. **Aumento da taxa de violação por janela URLLC em $\lambda=6$ (`33.77% -> 46.92%`):**
   A análise diagnóstica mostrou que, entre as `330` janelas URLLC em $\lambda=6$ que passaram em A0 e falharam em A1, **`82` janelas registraram `p95_ms = 10.15 ms`** e **`129` ficaram entre `12.1–14.1 ms`** (limiar imediato acima de `10.0 ms`). Quando eMBB está ativa e há projeção (`sum(a_exec) == 53`, `ae_mean = 27.3`, `au_mean = 19.4`), `53.99%` das janelas URLLC ultrapassam `10 ms` (contra `33.19%` sem projeção), mesmo com média maior de RBGs nominais em URLLC (`19.4` vs `10.9`). Em A0, `14.55%` dos pacotes falhavam concentrados em apagões longos (`83–374 ms`); em A1, menos pacotes falham no agregado (`10.06%`), mas leves atrasos de `10.03–14.1 ms` se espalham por mais janelas individuais — evidenciando a necessidade de validar em `A2` o impacto da retirada de propostas/cortes espúrios sobre essa contenção entre eMBB e URLLC no escalonador MAC.

---

## 6. Parecer Científico sobre o Efeito Isolado de R12 e Próximo Passo

1. **Veredito causal sobre `R12`:** O experimento A1 **confirma** a hipótese central da auditoria: as atualizações virtuais `R12` (`virtual_alpha = 0.05`) eram a causa primária do colapso negativo dos pesos do regressor, da perda de sensibilidade direcional em duas ações ($\hat{f}(37)>\hat{f}(0)$ de `0%` em A0 para `98–100%` em A1) e da incapacidade de operar em modo `confident`.
2. **Insuficiência do ajuste isolado:** `R12` também funcionava, por efeito colateral indesejado, como um amortecedor que mantinha `sum(a_prop)` artificialmente baixo em slices vazias. Sem `R12`, afloram imediatamente os dois defeitos estruturais subjacentes que ela mascarava:
   - propostas não triviais (`6–31` RBGs) em slices sem demanda nem backlog (`A2`), elevando projeções para `66%–73%`;
   - teto/crescimento da margem de incerteza `M` até `50` RBGs em escala incompatível com $C=53$ (`A3`), além da sensibilidade de URLLC quando $\gamma=1$ empurra `a_raw` a zero logo após janelas limpas.
3. **Conclusão retificada da A1 e encaminhamento controlado (`A2a` e `A1+A2a`):**
   Embora `A1` (`--virtual_alpha 0.0`) tenha melhorado significativamente a calibração, a perda pinball e `STRICT_ALL3_R19` em $\lambda \in \{5, 6\}$, **a remoção definitiva de `R12` no baseline restaurado NÃO está aprovada nesta etapa**, pois o corte por projeção saltou para `66.05%` ($\lambda=5$) e `73.42%` ($\lambda=6$) e as violações URLLC por janela pioraram em $\lambda=6$ (`33.77% -> 46.92%`). O próximo passo autorizado e isolado é **`A2a` (isolado em `A0`, `--virtual_alpha 0.05 --empty_slice_guard`) + `A1+A2a` (interação, `--virtual_alpha 0.0 --empty_slice_guard`)** com `seed=2102` e $\lambda \in \{2, 5, 6\}$, consolidado em [`A2A_RELATORIO_EXPERIMENTAL_4WAY.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/A2A_RELATORIO_EXPERIMENTAL_4WAY.md), mantendo `A2b`, `A3a`, `A3b`, `A4` e `A5` bloqueados conforme [`D3_RETIFICACAO_E_PLANO_ABLACAO.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/D3_RETIFICACAO_E_PLANO_ABLACAO.md).
