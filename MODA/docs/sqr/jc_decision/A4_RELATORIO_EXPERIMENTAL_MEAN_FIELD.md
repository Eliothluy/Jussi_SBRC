# Relatório Experimental $A_4$: Auditoria do `FeatureScaler` e Restauração do Estado *Mean-Field* ($\bar{s}_t$)

**Data:** 2026-11-14  
**Escopo:** Fase exploratória pós-$A_{3\text{a}}$ na seed de desenvolvimento `2102`, $\lambda \in \{2, 5, 6\}$ (`300 s`, `warmup_s=100`, `periodMs=100`, `cellRadius=300`, `slas=3.5,3.5,3.5`, `--urllc_delayed_update`, $C = 53$ PRBs).  
**Artefatos preservados:** `D1-v9`, $A_0$, $A_1$, $A_{2\text{a}}$, $A_1+A_{2\text{a}}$, $A_{3\text{a}}$, $A_1+A_{2\text{a}}+A_{3\text{a}}$ e todo o estado histórico permanecem intactos. As seeds independentes `3101–3103`, $A_{2\text{b}}$, $A_{3\text{b}}$, $A_5$, `D3-v2`, `PILOT` e `MAIN` permanecem **bloqueados**.

---

## 1. Auditoria Pontual do `FeatureScaler` (`letter = "emu"[slice_id]`)

### 1.1 Evidência no código e no manifesto congelado

Em [`sqr_het.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/sqr_het.py) dentro de `SqrHetController.slice_state()`, a resolução do sufixo de slice usa:

```python
letter = "emu"[slice_id]
```

Entretanto, a convenção de `slice_id` em todo o sistema é `0 = eMBB`, `1 = URLLC`, `2 = mMTC` (ordem `e, u, m`), como refletido nas tuplas `(n_e, n_u, n_m)`, `(q_e, q_u, q_m)` e `_KPI_FEATURE = ("kpi_e", "kpi_u", "kpi_m")`. Como resultado, `letter = "emu"[slice_id]` realiza o mapeamento trocado `0 -> 'e'`, `1 -> 'm'`, `2 -> 'u'`.

Análise por dimensão de `slice_state()`:
- **Dimensões afetadas (`n_{letter}` e `queue_{letter}`):** consultam os quantis `_p1` / `_p99` trocados entre URLLC (`1`) e mMTC (`2`).
- **Dimensões não afetadas (`snr_{letter}` e `kpi_{letter}`):** `PHYSICAL_RULES` em [`feature_scaler.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/feature_scaler.py) mapeia `snr_e`, `snr_u` e `snr_m` para a mesma regra `"snr_r09"` (divisão por `30.0`, clip `[0, 1]`), e o KPI usa a tupla explícita `_KPI_FEATURE = ("kpi_e", "kpi_u", "kpi_m")`.

No manifesto congelado usado pelos experimentos (`results/sqr_jc_final/03_calibration/feature_scaler_manifest.yaml`, `n_cal_samples = 23720`, `c = 53`, `frozen = True`), os limites relevantes são:
- **`_p1`:** `n_e = 0.0`, `n_u = 0.0`, `n_m = 0.0`, `queue_e = 0.0`, `queue_u = 0.0`, `queue_m = 0.0`, `offered_e = 0.0`, `offered_u = 0.0`, `offered_m = 0.0`.
- **`_p99`:** `n_e = 4.0`, `n_u = 3.0`, `n_m = 2.0`, `queue_e = 9844456.460000005`, `queue_u = 709.0`, `queue_m = 1422.0`, `offered_e = 16.5024`, `offered_u = 1.4256`, `offered_m = 0.0182`.

### 1.2 Prova via teste unitário com valores distintos por slice

O teste unitário `test_sqr_het_scaler_indexing_audit_emu_bug` em [`test_sqr_jc_controller.py`](../../../ns-o-ran-gym/tests/test_sqr_jc_controller.py) e o script de verificação `verify_a4_mean_field_and_scaler.py` confirmam o impacto quantitativo aplicando o vetor sonda `(n=(2, 2, 2), queue=(1000, 709, 709))` sobre o manifesto real congelado:

| Slice (`slice_id`) | Letra esperada | Letra aplicada (`"emu"`) | Dimensão `n` obtida vs. esperada | Dimensão `queue` obtida vs. esperada | Distorção comprovada |
|---|---|---|---|---|---|
| **eMBB (`0`)** | `'e'` | `'e'` | `0.500000` vs. `0.500000` | `0.000102` vs. `0.000102` | Sem distorção |
| **URLLC (`1`)** | `'u'` | `'m'` | **`1.000000`** vs. `0.666667` | **`0.498594`** vs. `1.000000` | **`n_u` superescalado `1.5×`** (usa `n_m_p99=2.0`, satura em `1.0` já com `n_u=2`); **`queue_u` subescalado `0.4986×`** (usa `queue_m_p99=1422.0`) |
| **mMTC (`2`)** | `'m'` | `'u'` | **`0.666667`** vs. `1.000000` | **`1.000000`** vs. `0.498594` | **`n_m` subescalado `0.6667×`** (usa `n_u_p99=3.0`); **`queue_m` superescalado `2.0056×`** (usa `queue_u_p99=709.0`, satura acima de `709 B`) |

**Regra de isolamento respeitada:** conforme determinado, o bug **não foi corrigido silenciosamente** durante $A_4$. A indexação `letter = "emu"[slice_id]` permaneceu congelada tanto em $A_4$ quanto em $A_1+A_{2\text{a}}+A_4$, sendo recomendada sua avaliação isolada em experimento próprio (`A_scaler`) na Seção 4.

---

## 2. Implementação Isolada da $A_4$ (`--mean_field`) e Integridade Causal

### 2.1 Mudanças implementadas e isolamento de escopo

Em [`sqr_het.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/sqr_het.py) e [`sqr_run.py`](../../../ns-o-ran-gym/examples/sqr_run.py):
- Adicionado o parâmetro explícito `mean_field: bool = False` em `SqrHetController.__init__` e a flag CLI `--mean_field` em `sqr_run.py`.
- Quando `mean_field=True`, as dimensões 4–5 de `slice_state()` passam de `(prev_a_i / C, viol_i)` para o estado *mean-field* causal do artigo original (Seção III-B):
  - Dimensão 4: `clip_norm(prev_a_others, C)`, onde `prev_a_others` é a soma causal dos PRBs executados nas demais slices na janela anterior ($t-1$).
  - Dimensão 5: `clip_norm(n_e + n_u + n_m, N_MAX_CELL)` com `N_MAX_CELL = 20`, reutilizando `clip_norm` de [`sqr_env.py`](../../../ns-o-ran-gym/src/nsoran/sqr/sqr_env.py).
- **Cold start obrigatório e guarda de `--resume`:** o campo `mean_field` é gravado no metadado do checkpoint (`meta["mean_field"]`) e `--resume` recusa com `ValueError` qualquer tentativa de retomar checkpoint com `mean_field` incompatível. Todos os runs de $A_4$ partiram de `cold start`.
- **Auditoria linha a linha:** `z_dim4_dim5_per_slice` e `mean_field` passaram a ser gravados nos registros `_het_pending` (`SQR_HET_BOOTSTRAP_PROP` e `SQR_HET`).
- **Componentes inalterados:** R12 (`--virtual_alpha`), R11, R10, guarda `A2a` (`--empty_slice_guard`), atualização pinball, kernel Matérn, `M_0=10`, `M ∈ [2, 50]` e lei de adaptação, SLAs (`3.5,3.5,3.5`), `R19`, bootstrap, política de seleção, `"emu"` e seed `2102`.

### 2.2 Verificação de integridade dos 6 runs $A_4$ (Onda 1 e Onda 2)

- **Onda 1 (`A4` isolado sobre `A0`):** `MODA/ns-o-ran-gym/results/sqr_jc_decision/a4_runs_v9_mf` (`--virtual_alpha 0.05`, guarda OFF, `M_0=10`, `M∈[2,50]`, `--mean_field`).
- **Onda 2 (`A1+A2a+A4` sobre `A1+A2a`):** `MODA/ns-o-ran-gym/results/sqr_jc_decision/a1_a2a_a4_runs_v9_va0_guard_mf` (`--virtual_alpha 0.0`, `--empty_slice_guard`, `M_0=10`, `M∈[2,50]`, `--mean_field`).

Todas as 6 execuções concluíram na primeira tentativa com `done=True`, `rc0=True` (`ns3_exit_code: 0`), `ts=299900`:

| Métrica de integridade | $\lambda = 2$ | $\lambda = 5$ | $\lambda = 6$ |
|---|---|---|---|
| Decisões totais (`decisions.jsonl`) | `848` | `2298` | `2455` |
| Paridade de bootstrap (`boot == baseline`) | `283 == 283` (`True`) | `303 == 303` (`True`) | `460 == 460` (`True`) |
| Consistência `rx == kpm` & `mf_meta` | `True` / `True` | `True` / `True` | `True` / `True` |
| Registros com `z_dim4_dim5_per_slice` (`zlog`) | `847` | `2297` | `2454` |
| Erro máximo de reconstrução causal (`dim5` / `dim4`) | `0.00e+00` / `4.91e-07` | `0.00e+00` / `4.91e-07` | `0.00e+00` / `4.91e-07` |
| Média `dim5` (`A4` e `A1+A2a+A4`) | `0.0518` | `0.2214` | `0.2499` |
| Médias `dim4` por slice (`A4`) | `[0.3072, 0.2643, 0.3489]` | `[0.2173, 0.3626, 0.4814]` | `[0.2942, 0.4972, 0.5116]` |
| Médias `dim4` por slice (`A1+A2a+A4`) | `[0.2229, 0.2292, 0.2365]` | `[0.3236, 0.5481, 0.5899]` | `[0.4130, 0.5562, 0.6911]` |

---

## 3. Avaliação Causal Completa: `A4` vs. `A0` e `A1+A2a+A4` vs. `A1+A2a`

### 3.1 Calibração quantílica, perda pinball, MAE/viés e fração conservadora (`q >= 0`)

#### $\lambda = 2$ (sem janelas com 3 slices ativas simultâneas)

| Braço | Slice 0 (eMBB) | Slice 1 (URLLC) | Slice 2 (mMTC) |
|---|---|---|---|
| `A0` | `cov=0.0061` (`t=0.0500`, `|c-t|=0.0439`), `pinb=0.1530`, `mae=3.0072`, `bias=-3.0014`, `q>=0=1.2%` | `cov=0.0968` (`t=0.0439`, `|c-t|=0.0529`), `pinb=0.3029`, `mae=0.8169`, `bias=-0.2299`, `q>=0=16.1%` | `cov=0.0000` (`t=0.0500`), `pinb=0.0167`, `mae=0.3334`, `bias=-0.3334` |
| `A4` | `cov=0.0061` (`t=0.0480`), `pinb=0.0408`, `mae=0.7765`, `bias=-0.7706` | `cov=0.0000` (`t=0.0439`), `pinb=0.0168`, `mae=0.3831`, `bias=-0.3831` | `cov=0.0026` (`t=0.0500`), `pinb=0.0127`, `mae=0.2078`, `bias=-0.2027` |
| `A1+A2a` | `cov=0.0244` (`t=0.0480`, `|c-t|=0.0237`), `pinb=0.1039`, `mae=2.0457`, `bias=-2.0369`, `q>=0=86.6%` | `cov=0.0968` (`t=0.0439`), `pinb=0.2950`, `mae=0.5625`, `bias=+0.0309`, `q>=0=16.1%` | `cov=0.0205` (`t=0.0427`), `pinb=0.0232`, `mae=0.0926`, `bias=-0.0506`, `q>=0=93.3%` |
| `A1+A2a+A4` | `cov=0.0244` (`t=0.0480`), `pinb=0.1230`, `mae=2.4346`, `bias=-2.4245`, `q>=0=64.6%` | `cov=0.0968` (`t=0.0439`), `pinb=0.5131`, `mae=0.7672`, `bias=+0.2851`, `q>=0=25.8%` | `cov=0.0205` (`t=0.0427`), `pinb=0.0233`, `mae=0.0920`, `bias=-0.0498`, `q>=0=92.6%` |

#### $\lambda = 5$

| Braço | `sv` `[e, u, m]` | `virt` `[e, u, m]` | `pinb` `[e, u, m]` | `cov` / `|c-t|` | `mae` `[e, u, m]` | `q>=0` `[e, u, m]` |
|---|---|---|---|---|---|---|
| `A0` | `[1885, 1417, 2000]` | `[303, 258, 126]` | `[0.0437, 0.0721, 0.0220]` | `cov=[0.0026, 0.0000, 0.0108]` / `|c-t|=[0.0176, 0.0263, 0.0210]` | `[2.8341, 3.0561, 0.5574]` | — |
| `A4` | `[1815, 1342, 2000]` | `[233, 183, 111]` | `[0.0405, 0.0507, 0.0243]` | `cov=[0.0013, 0.0034, 0.0139]` / `|c-t|=[0.0212, 0.0218, 0.0054]` | `[2.0238, 1.9924, 1.2921]` | — |
| `A1+A2a` | `[1582, 1159, 1944]` | `0` | `[0.0206, 0.0595, 0.0101]` | `cov=[0.0197, 0.0213, 0.0149]` (`t=[0.0228, 0.0207, 0.0275]`, `|c-t|=[0.0030, 0.0006, 0.0125]`) | `[0.7377, 0.4000, 0.0685]` | `[55.4%, 26.2%, 81.6%]` |
| `A1+A2a+A4` | `[1582, 1159, 1944]` | `0` | `[0.0188, 0.0415, 0.0102]` | `cov=[0.0171, 0.0325, 0.0144]` (`t=[0.0202, 0.0246, 0.0273]`, `|c-t|=[0.0031, 0.0079, 0.0129]`) | `[0.7938, 0.4225, 0.0734]` | **`[22.2%, 16.3%, 64.9%]`** |

#### $\lambda = 6$

| Braço | `pinb` `[e, u, m]` | `cov` / `|c-t|` | `mae` `[e, u, m]` | `q>=0` `[e, u, m]` |
|---|---|---|---|---|
| `A0` | `[0.0639, 0.4479, 0.0212]` | `cov=[0.0036, 0.0354, 0.0082]` | `[3.4986, 4.1189, 0.6792]` | — |
| `A4` | `[0.0547, 0.2821, 0.0240]` | `cov=[0.0030, 0.0351, 0.0066]` | `[2.8908, 2.2570, 0.6751]` | — |
| `A1+A2a` | `[0.0157, 0.0274, 0.0092]` | `cov=[0.0286, 0.0266, 0.0127]` (`|c-t|=[0.0073, 0.0078, 0.0100]`) | `[0.5623, 0.4774, 0.0846]` | `[19.2%, 1.4%, 41.8%]` |
| `A1+A2a+A4` | `[0.0142, 0.0324, 0.0088]` | `cov=[0.0250, 0.0281, 0.0133]` (`|c-t|=[0.0058, 0.0093, 0.0123]`) | `[0.5335, 0.5996, 0.0755]` | **`[2.5%, 1.8%, 41.1%]`** |

**Síntese de calibração:**
- Sobre `A0` (com `R12 = 0.05` ativo), `A4` reduz o MAE e a perda pinball em eMBB e URLLC (`λ=2, 5, 6`), mas aumenta o MAE em mMTC em `λ=5` (`0.5574 -> 1.2921`).
- Sobre `A1+A2a`, adicionar `A4` **não melhora** a calibração empírica (`|c-t|` em URLLC piora de `0.0006` para `0.0079` em `λ=5` e de `0.0078` para `0.0093` em `λ=6`), eleva `mae` em `λ=2` (`2.0457 -> 2.4346` em eMBB; `0.5625 -> 0.7672` em URLLC) e em URLLC `λ=6` (`0.4774 -> 0.5996`), e **derruba drasticamente `q>=0`** (fração de previsões quantílicas não negativas): de `55.4% -> 22.2%` (eMBB) e `26.2% -> 16.3%` (URLLC) em `λ=5`, e de `19.2% -> 2.5%` (eMBB) em `λ=6` — sinalizando que o regressor prevê violações com maior frequência sob o estado *mean-field*.

### 3.2 Competição inter-slice: `sum(a_raw)` e `sum(a_prop) > 53` estratificados por slices ativas

| $\lambda$ | Braço | Global: `sumR`, `raw>53`, `excR`, `sumP`, `proj` | Estrato `=3` ativas (`n=0 / 787 / 1197`) | Estrato `<3` (`1` vs `2` ativas: `raw>53 / proj`) |
|---|---|---|---|---|
| **2** | `A0` | `36.32`, `13.65%`, `2.22`, `37.13`, `14.18%` | `n=0` | `1: 12.03% / 12.59%`, `2: 40.62% / 40.62%` |
| **2** | `A4` | `28.13`, `9.57%`, `1.13`, `28.96`, `11.35%` | `n=0` | `1: 9.59% / 11.47%`, `2: 9.38% / 9.38%` |
| **2** | `A1+A2a` | `15.24`, `0.71%`, `0.11`, `16.14`, `0.89%` | `n=0` | `1: 0.19% / 0.38%`, `2: 9.38% / 9.38%` |
| **2** | `A1+A2a+A4` | `18.43`, `5.50%`, `0.56`, `19.33`, `6.91%` | `n=0` | `1: 2.82% / 3.95%`, `2: 50.00% / 56.25%` |
| **5** | `A0` | `33.55`, `15.10%`, `1.67`, `34.24`, `16.00%` | `sumR=28.74`, `raw>53=7.62%`, `excR=0.90`, `proj=7.88%` | `<3: 19.97% / 21.29%` (`1: 18.14%/18.87%`, `2: 20.90%/22.53%`) |
| **5** | `A4` | `28.99`, `10.53%`, `1.34`, `30.17`, `12.24%` | `sumR=22.72`, `raw>53=5.08%`, `excR=0.56`, `proj=6.48%` | `<3: 14.08% / 15.99%` (`1: 19.61%/20.59%`, `2: 11.26%/13.64%`) |
| **5** | `A1+A2a` | `40.57`, `24.52%`, `3.86`, `41.27`, `26.18%` | `sumR=56.05`, `raw>53=43.20%`, `excR=9.01`, `sumP=56.94`, `proj=43.96%` | `<3: 12.34% / 14.58%` (`1: 0.49%/0.74%`, `2: 18.40%/21.65%`) |
| **5** | **`A1+A2a+A4`** | `46.70`, `46.64%`, `6.21`, `47.78`, `47.59%` | **`sumR=64.76`, `raw>53=83.35%`, `excR=12.99`, `sumP=65.95`, `proj=83.74%`** | `<3: 22.70% / 24.03%` (`1: 0.00%/0.25%`, `2: 34.29%/36.17%`) |
| **6** | `A0` | `36.07`, `14.84%`, `2.04`, `36.92`, `16.05%` | `sumR=37.87`, `raw>53=17.38%`, `excR=2.34`, `sumP=38.64`, `proj=18.63%` | `<3: 11.04% / 12.17%` |
| **6** | `A4` | `36.27`, `12.69%`, `1.74`, `37.52`, `14.34%` | `sumR=36.48`, `raw>53=15.71%`, `excR=2.08`, `sumP=37.31`, `proj=17.29%` | `<3: 8.16% / 9.91%` |
| **6** | `A1+A2a` | `55.83`, `63.69%`, `11.22`, `56.86`, `64.14%` | `sumR=68.06`, `raw>53=86.30%`, `excR=16.46`, `sumP=69.27`, `proj=86.63%` | `<3: 29.74% / 30.36%` (`1: 0.00%/0.00%`, `2: 41.58%/42.46%`) |
| **6** | **`A1+A2a+A4`** | `57.17`, `70.31%`, `10.11`, `58.24`, `70.96%` | **`sumR=67.80`, `raw>53=94.65%`, `excR=15.18`, `sumP=68.95`, `proj=95.07%`** | `<3: 33.75% / 34.76%` (`1: 0.00%/0.00%`, `2: 47.19%/48.60%`) |

**Achado central sobre a hipótese da $A_4$:**
- A restauração de $\bar{s}_t$ **refuta a hipótese** de que features *mean-field* sozinhas reduzam a competição inter-slice no SQR-HET desacoplado.
- Sobre `A1+A2a`, `--mean_field` **quase dobra a taxa de excesso bruto com 3 slices ativas em $\lambda=5$** (`raw>53` sobe de `43.20%` para `83.35%`; `sum_raw` médio sobe de `56.05` para `64.76`; projeção sobe de `43.96%` para `83.74%`) e a eleva ainda mais em $\lambda=6$ (`raw>53` de `86.30%` para `94.65%`; projeção de `86.63%` para `95.07%`). Mesmo com 2 slices ativas (`λ=2, 5, 6`), `A1+A2a+A4` explode `raw>53`: `9.38% -> 50.00%` (λ=2), `18.40% -> 34.29%` (λ=5) e `41.58% -> 47.19%` (λ=6).
- Mecanismo: sem coordenação conjunta no espaço admissível $\sum a_i \le C$, cada regressor quantílico independente interpreta maior carga global / alocação das outras slices como sinal de risco compartilhado e eleva seu lance simultaneamente, fazendo a soma bruta `sum(a_raw)` ultrapassar `53` PRBs na grande maioria das janelas multi-slice.

### 3.3 Efeitos de SLA, segurança URLLC, interações e regressões negativas

**Interações aditivas em `STRICT_ALL3_R19`:**
- **$\lambda = 5$:** $\Delta A_4 = -1.15\text{ pp}$ (`26.24%` vs `27.39%`); $\Delta(A_1+A_{2\text{a}}+A_4 \text{ vs } A_1+A_{2\text{a}}) = -10.45\text{ pp}$ (`73.76%` vs `84.20%`); **Interação $I = -9.30\text{ pp}$** (antagonismo).
- **$\lambda = 6$:** $\Delta A_4 = +14.71\text{ pp}$ (`49.96%` vs `35.24%`); $\Delta(A_1+A_{2\text{a}}+A_4 \text{ vs } A_1+A_{2\text{a}}) = +0.58\text{ pp}$ (`42.36%` vs `41.77%`); **Interação $I = -14.13\text{ pp}$**.

**Regressões e diagnóstico causal da degradação URLLC em $\lambda = 5$ (`A1+A2a+A4` vs. `A1+A2a`):**
- Em `λ=5`, adicionar `--mean_field` a `A1+A2a` derruba `STRICT_ALL3_R19` em **`-10.45 pp`** (`84.20% -> 73.76%`), mais que dobra as violações URLLC por janela (`9.20% -> 20.09%`), eleva violações eMBB (`5.99% -> 7.83%`) e quase dobra a taxa de *miss* em pacotes URLLC (`2.22% -> 4.03%`).
- Diagnóstico janela a janela entre trajetórias (`A1+A2a+A4` vs `A1+A2a` em `λ=5`): **`ok2fail = 149`** janelas contra apenas **`fail2ok = 52`**. O desvio se inicia com **picos precoces de inanição URLLC** nas janelas `w = 1482, 1484, 1578, 1625`, quando a projeção e a competição inter-slice derrubam `a_exec_u` para `1, 0, 6, 1` PRBs (`p95 = 27.1, 101.15, 10.15, 27.1 ms`). A partir daí ocorre **divergência de trajetória**: falhas marginais na faixa `(10, 11] ms` saltam de `28 -> 62` e na faixa `(11, 15] ms` saltam de `42 -> 95`, mesmo com `a_exec_u` médio global superior (`19.41` vs `16.41` PRBs) — mostrando que alocações mais instáveis e picos de subalocação sob projeção anulam o ganho de PRBs médios.
- Em `A4` isolado (`λ=2` e `λ=5`), apesar de ganho em `λ=6` (`49.96%`), ocorrem fortes regressões de segurança e SLA: em `λ=2`, violação eMBB explode `+23.17 pp` (`1.22% -> 24.39%`) com queda de utilização (`4.63% -> 1.79%`); em `λ=5`, violação mMTC quadruplica (`2.78% -> 11.42%`) e `miss` URLLC sobe (`4.16% -> 5.38%`); em `λ=6`, `A4` mantém **1293 pacotes URLLC sem recebimento** (`tx=215464`, `rx=214171`), com `p95=56.10 ms` e `p99=229.03 ms` (`miss=13.65%`).

---

## 4. Matriz Consolidada 8-Way e Proposta Científica para `B0-rest`

### 4.1 Tabela comparativa dos 8 braços exploratórios (`seed = 2102`)

#### $\lambda = 2$ (`STRICT_ALL3_R19 = 0/0/564` em todos; `u_r19_known = 31`; `tx = 3151`, `rx = 3146`, `unrx = 5`)

| Braço | `proj` | `util` | `viol [eMBB, URLLC, mMTC]` | URLLC `p95` / `p99` (ms) | `miss` URLLC |
|---|---|---|---|---|---|
| `A0` | `14.18%` | `4.63%` | `[1.22%, 19.35%, 0.00%]` | `18.10` / `43.10` | `8.49%` |
| `A1` | `5.32%` | `3.43%` | `[2.44%, 6.45%, 2.05%]` | `15.10` / `75.65` | `6.23%` |
| `A2a` | `0.71%` | `3.92%` | `[2.44%, 16.13%, 2.05%]` | `18.10` / `43.10` | `8.30%` |
| **`A1+A2a`** | `0.89%` | `3.81%` | `[2.44%, 16.13%, 2.05%]` | `18.10` / `43.10` | `8.42%` |
| `A3a` | `11.70%` | `4.54%` | `[2.44%, 16.13%, 5.64%]` | `12.80` / `24.10` | `6.36%` |
| `A1+A2a+A3a` | `0.53%` | `3.98%` | `[3.05%, 19.35%, 2.05%]` | `20.10` / `51.65` | `9.57%` |
| `A4` | `11.35%` | `1.79%` | `[24.39%, 6.45%, 0.26%]` | `8.10` / `8.61` | `0.60%` |
| `A1+A2a+A4` | `6.91%` | `4.35%` | `[2.44%, 16.13%, 2.05%]` | `24.80` / `76.10` | `9.35%` |

#### $\lambda = 5$ (`u_r19_known = 891`, `unk = 1103`; `tx = 119182`, `rx = 119158`, `unrx = 24`)

| Braço | `raw>53` (global / `=3`) | `proj` (global / `=3`) | `util` | `viol [eMBB, URLLC, mMTC]` | `STRICT_ALL3_R19` (`P/F/U`) | URLLC `p95`/`p99` / `miss` |
|---|---|---|---|---|---|---|
| `A0` | `15.10%` / `7.62%` | `16.00%` / `7.88%` | `27.00%` | `[66.95%, 24.35%, 2.78%]` | `27.39%` (`215/570/1209`) | `9.10`/`14.10` / `4.16%` |
| `A1` | `62.04%` / `72.55%` | `66.05%` / `73.44%` | `35.78%` | `[32.65%, 15.04%, 1.29%]` | `73.50%` (`577/208/1209`) | `8.10`/`14.10` / `2.99%` |
| `A2a` | `8.12%` / `14.74%` | `8.83%` / `15.76%` | `33.59%` | `[37.00%, 17.40%, 8.49%]` | `50.57%` (`397/388/1209`) | `9.10`/`15.10` / `3.84%` |
| **`A1+A2a`** | `24.52%` / `43.20%` | `26.18%` / `43.96%` | `44.94%` | `[5.99%, 9.20%, 1.29%]` | **`84.20%`** (`661/124/1209`) | `8.10`/`13.10` / `2.22%` |
| `A3a` | `14.54%` / `7.62%` | `14.94%` / `7.62%` | `24.72%` | `[63.66%, 21.21%, 3.70%]` | `39.11%` (`307/478/1209`) | `9.10`/`14.10` / `4.04%` |
| `A1+A2a+A3a` | `42.03%` / `87.04%` | `42.13%` / `87.04%` | `44.17%` | `[9.15%, 5.84%, 1.23%]` | `80.00%` (`628/157/1209`) | `8.10`/`10.10` / `1.24%` |
| `A4` | `10.53%` / `5.08%` | `12.24%` / `6.48%` | `25.50%` | `[68.40%, 22.56%, 11.42%]` | `26.24%` (`206/579/1209`) | `10.10`/`15.10` / `5.38%` |
| `A1+A2a+A4` | `46.64%` / `83.35%` | `47.59%` / `83.74%` | `45.97%` | `[7.83%, 20.09%, 1.29%]` | `73.76%` (`579/206/1209`) | `9.10`/`14.10` / `4.03%` |

#### $\lambda = 6$ (`tx = 215464`)

| Braço | `raw>53` (global / `=3`) | `proj` (global / `=3`) | `util` | `viol [eMBB, URLLC, mMTC]` | `STRICT_ALL3_R19` (`P/F/U`) | `rx` / `unrx` | URLLC `p95`/`p99` / `miss` |
|---|---|---|---|---|---|---|---|
| `A0` | `14.84%` / `17.38%` | `16.05%` / `18.63%` | `38.83%` | `[54.88%, 33.77%, 3.26%]` | `35.24%` (`418/768/808`) | `212863` / `2601` | `83.03`/`307.10` / `14.55%` |
| `A1` | `71.77%` / `80.03%` | `73.42%` / `80.45%` | `45.18%` | `[38.99%, 46.92%, 1.27%]` | `43.61%` (`522/675/797`) | `215437` / `27` | `12.10`/`18.10` / `10.06%` |
| `A2a` | `14.74%` / `20.97%` | `16.85%` / `22.64%` | `44.89%` | `[43.04%, 42.18%, 3.41%]` | `36.02%` (`429/762/803`) | `212634` / `2830` | `42.10`/`252.10` / `16.31%` |
| **`A1+A2a`** | `63.69%` / `86.30%` | `64.14%` / `86.63%` | `47.19%` | `[29.70%, 43.27%, 1.27%]` | `41.77%` (`500/697/797`) | `215437` / `27` | `11.10`/`15.10` / `8.42%` |
| `A3a` | `14.09%` / `17.29%` | `14.44%` / `17.63%` | `41.26%` | `[66.13%, 41.06%, 9.68%]` | `21.47%` (`257/940/797`) | `215437` / `27` | `13.10`/`27.10` / `11.71%` |
| `A1+A2a+A3a` | `57.62%` / `88.14%` | `57.67%` / `88.14%` | `45.52%` | `[28.39%, 22.13%, 1.48%]` | **`56.89%`** (`681/516/797`) | `215437` / `27` | `9.10`/`14.10` / `3.50%` |
| `A4` | `12.69%` / `15.71%` | `14.34%` / `17.29%` | `40.38%` | `[44.17%, 28.09%, 1.63%]` | `49.96%` (`596/597/801`) | `214171` / `1293` | `56.10`/`229.03` / `13.65%` |
| `A1+A2a+A4` | `70.31%` / `94.65%` | `70.96%` / `95.07%` | `48.80%` | `[29.11%, 42.36%, 1.27%]` | `42.36%` (`507/690/797`) | `215437` / `27` | `12.10`/`15.10` / `8.83%` |

*(Nota: `u_known` em $\lambda=6$ varia ligeiramente por término/cauda do tráfego: `1300 (unk=694)` em `A0`, `1304 (690)` em `A2a`, `1310 (684)` em `A4` e `1315 (679)` nos demais braços com `rx=215437`.)*

### 4.2 Revisão Pós-$A_{\text{scaler}}$, Proposta Científica para `B0-rest` e Próximo Encaminhamento *(Atualizado em 2026-11-14)*

> **Adendo pós-$A_{\text{scaler}}$ (2026-11-14):** Todos os 8 braços analisados nas Seções 1–4.1 deste relatório foram executados sob o bug de indexação `"emu"` (`URLLC` e `mMTC` trocados em `dim0..1`, com distorção de até `2.0×`). O experimento isolado [`A_SCALER_RELATORIO_EXPERIMENTAL.md`](A_SCALER_RELATORIO_EXPERIMENTAL.md) (`--fix_scaler_index`, `"emu" -> "eum"` sobre `A1+A2a`) demonstrou posteriormente que essa troca altera `STRICT_ALL3_R19` em `-12.48 pp` ($\lambda=5$) e `+13.53 pp` ($\lambda=6$), deslocando a linha de base de comparação e invalidando o descarte definitivo de $A_4$ baseado apenas em runs contaminados por `"emu"`.

1. **Decisão revisada sobre $A_4$ (`--mean_field`):** **Concluída como experimento exploratório sob `"emu"` — NÃO rejeitada definitivamente.** Quando combinada com `A1+A2a` sob `"emu"`, piorou a competição inter-slice (`raw>53` em `=3` ativas: `43.20% -> 83.35%` em $\lambda=5$ e `86.30% -> 94.65%` em $\lambda=6$), reduziu `STRICT_ALL3_R19` em `-10.45 pp` em $\lambda=5$ (`84.20% -> 73.76%`), dobrou violações URLLC em $\lambda=5$ (`9.20% -> 20.09%`) e produziu interação negativa ($I = -9.30\text{ pp}$ e $-14.13\text{ pp}$); em forma isolada sobre `A0`, elevou $\lambda=6$ (`35.24% -> 49.96%`), porém com regressões (`+23.17 pp` em eMBB $\lambda=2$, `+8.64 pp` em mMTC $\lambda=5$, `1293` pacotes URLLC sem recebimento em $\lambda=6$). Contudo, como `dim4..5` (`--mean_field`) interage diretamente com `dim0..1` no kernel Matérn $\nu=1{,}5$, uma decisão definitiva sobre $A_4$ depende de reavaliação limpa sobre `A1+A2a+A_scaler` (**não executada** nesta rodada).
2. **Posição sobre $A_{3\text{a}}$ (`M_0=3, M\in[1,10]`):** **Não incluir automaticamente** no $B_0^{\text{rest}}$ primário apenas pelo ganho em $\lambda=6$ (`56.89%`). Embora ajude sob carga saturada em $\lambda=6$, $A_1+A_{2\text{a}}+A_{3\text{a}}$ perde `-4.20 pp` de `STRICT_ALL3_R19` em $\lambda=5$ (`84.20% -> 80.00%`), piora levemente $\lambda=2$ (`miss` `8.42% -> 9.57%`) e mantém projeção em 3 ativas elevada (`87.04%–88.14%`). Deve ser preservado como **variante secundária de sensibilidade** para comparação transparente em $\lambda=6$.
3. **Recomendação atualizada para $B_0^{\text{rest}}$ (após $A_{\text{scaler}}$):** **`A1 + A2a + A_scaler`** (`--virtual_alpha 0.0 --empty_slice_guard --fix_scaler_index`, `M_0 = 10`, `M ∈ [2, 50]`, `mean_field = False`) passa a ser o **candidato primário recomendado para `B0`** — por eliminar o defeito objetivo `"emu"`, melhorar $\lambda=2$ (`97.11% STRICT_DEMAND`, `miss` `5.59%`) e $\lambda=6$ (`55.30% STRICT_ALL3_R19`, `+13.53 pp`) e expor honestamente a contenção de `81.58%–82.62%` em `=3` ativas que motiva `SQR-JC`. `A1+A2a` (`"emu"`) **não** deve ser congelado como `B0` definitivo, sendo mantido apenas como controle histórico. Detalhes completos e lista mínima de 3 verificações pré-congelamento constam em [`A_SCALER_RELATORIO_EXPERIMENTAL.md`](A_SCALER_RELATORIO_EXPERIMENTAL.md).
4. **Escopo mantido bloqueado:** nenhuma alteração no `contrib/nr`, nem execução de `A1+A2a+A_scaler+A4`, `A2b`, `A3b`, `A5`, `D3-v2`, `PILOT`, `MAIN` ou das seeds independentes `3101–3103` deve ocorrer sem novo `GO` explícito.
