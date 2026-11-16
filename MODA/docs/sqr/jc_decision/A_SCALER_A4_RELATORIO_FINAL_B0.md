# Relatório Final de Fidelidade ($A_4$ sobre $A_{\text{scaler}}$), Auditoria de SLAs e Proposta de Fechamento de $B_0$ (`SQR-HET`)

**Data:** 15/11/2026 · **Seed de desenvolvimento:** `2102`, $\lambda \in \{2, 5, 6\}$, `simTime=300 s`, `warmup_s=100 s` (*cold start*).  
**Regra de Preservação e Escopo:** Todos os artefatos históricos (`D1-v9`, `D2-v2`, `D3-v1`, $A_0\text{–}A_4$ sob `"emu"` e $A_{\text{scaler}}$) permanecem intocados. `A5`, `D3-v2`, `PILOT`, `MAIN`, a checagem proposta na `seed=2103` e as seeds independentes `3101–3103` **não foram executados**.

---

## 1. Experimento Controlado: $A_1 + A_{2\text{a}} + A_{\text{scaler}} + A_4$ vs. $A_1 + A_{2\text{a}} + A_{\text{scaler}}$

### 1.1 Isolamento Estrito e Saúde de Execução
- **Controle:** `A1+A2a+A_scaler` (`--gamma 1 --virtual_alpha 0.0 --empty_slice_guard --fix_scaler_index --margin_init 10.0 --margin_bounds 2.0,50.0 --urllc_delayed_update`, `mean_field=False`, saída em `MODA/ns-o-ran-gym/results/sqr_jc_decision/a1_a2a_ascaler_runs_v9_va0_guard_eum`).
- **Variante:** `A1+A2a+A_scaler+A4`, adicionando **exclusivamente** `--mean_field` (`dims 4..5` em `SqrHetController.slice_state()`: `[clip_norm(prev_others, C), clip_norm(total_ues, N_MAX_CELL)]`), mantendo idênticos `R12=0.0`, guarda de demanda, `M_0=10`, `M ∈ [2,50]`, $\tau$, `R10`, `R19`, kernel Matérn $\nu=1{,}5$, SLAs, bootstrap e política de seleção (`cold start`, `--resume` bloqueado entre semânticas/escaladores via [`verify_a_scaler_a4_isolation.py`](file:///home/eliothluy/.gemini/antigravity/brain/4622041f-48e4-41e3-8c43-cf5834bb15d6/scratch/verify_a_scaler_a4_isolation.py)). Saída em `MODA/ns-o-ran-gym/results/sqr_jc_decision/a1_a2a_ascaler_a4_runs_v9_va0_guard_eum_mf`.
- **Integridade dos 6 runs (`seed=2102`, $\lambda \in \{2,5,6\}$):**
  - Todos encerraram com `ns3_exit_code: 0`, `.done=True`, timestamp final KPM `299900 ms`, e contagens idênticas de decisões online (`848`, `2298`, `2455`) e janelas de bootstrap (`283`, `303`, `460`), com **paridade bit-a-bit de bootstrap = `True`** e **`RX == KPM` = `True`** em todos os runs.
  - Reconstrução causal das features: `max_abs_err` em `dims 0..1` $\le 5{,}00\times 10^{-7}$; `dims 4..5` zeradas no controle (`err4=0.00e+00`, `err5=0.00e+00`) e reconstruídas exatamente na variante (`err4 <= 4.91e-07`, `err5=0.00e+00`).

---

### 1.2 Desempenho Global, Compliance dos 3 SLAs, Cobertura `UNKNOWN` e Caudas URLLC

> **Regra metodológica:** `STRICT_ALL3_R19` aplica-se apenas onde há demanda simultânea conhecida nas 3 slices ($\lambda \in \{5, 6\}$). Em $\lambda = 2$, não existe nenhuma janela com as 3 slices simultaneamente conhecidas (`known=0`, `unk=564`), razão pela qual `STRICT_ALL3_R19 = NaN` e o acompanhamento de baixa carga usa exclusivamente `STRICT_DEMAND_R19` (separado).

| $\lambda$ | Braço | Util. DL | `eMBB` viol. (`n`) | `URLLC` R19 viol. (`known` / `unk`) | `mMTC` viol. (`n`) | `STRICT_ALL3_R19` (`pass/fail/unk`) | `STRICT_DEMAND_R19` (`pass/fail/unk`) | URLLC `uTX` `rx/tx` (`unrx @300s`) | `uTX` `p95 / p99 / max` (ms) | `uTX` `miss >10ms` | R19 `win_p95_max` / `closed_miss` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2** | `A1+A2a+A_scaler` | `3.73%` | `2.44%` (`164`) | `12.90%` (`31` / `533`) | `2.05%` (`390`) | `N/A` (`0/0/564`) | **`97.11%`** (`537/16/1`) | `3146/3151` (`5`) | `11.10 / 43.10 / 68.10` | `5.59%` | `64.90 ms` / `5.74%` |
| **2** | `A1+A2a+A_scaler+A4` | `4.33%` | `2.44%` (`164`) | `12.90%` (`31` / `533`) | `2.05%` (`390`) | `N/A` (`0/0/564`) | **`97.11%`** (`537/16/1`) | `3146/3151` (`5`) | `20.10 / 77.07 / 106.10` | `6.07%` | `99.15 ms` / `6.23%` |
| **5** | `A1+A2a+A_scaler` | `46.75%` | `11.92%` (`1519`) | `17.51%` (`891` / `1103`) | `1.18%` (`1944`) | **`71.72%`** (`563/222/1209`) | `83.22%` (`1656/334/4`) | `119158/119182` (`24`) | `8.10 / 14.10 / 108.10` | `3.38%` | `99.15 ms` / `3.36%` |
| **5** | `A1+A2a+A_scaler+A4` | `44.38%` | `13.56%` (`1519`) | `18.74%` (`891` / `1103`) | `1.34%` (`1944`) | **`67.26%`** (`528/257/1209`) | `83.67%` (`1665/325/4`) | `119158/119182` (`24`) | `9.10 / 15.10 / 108.10` | `4.49%` | `100.15 ms` / `4.50%` |
| **6** | `A1+A2a+A_scaler` | `48.01%` | `22.26%` (`1680`) | `32.78%` (`1315` / `679`) | `1.33%` (`1962`) | **`55.30%`** (`662/535/797`) | `66.75%` (`1329/662/3`) | `215430/215464` (`34`) | `10.10 / 15.10 / 89.10` | `5.50%` | `84.85 ms` / `5.48%` |
| **6** | `A1+A2a+A_scaler+A4` | `49.08%` | `28.04%` (`1680`) | `38.63%` (`1315` / `679`) | `1.27%` (`1962`) | **`44.78%`** (`536/661/797`) | `59.87%` (`1192/799/3`) | `215437/215464` (`27`) | `11.10 / 15.10 / 108.10` | `8.08%` | `100.15 ms` / `8.09%` |

**Detalhamento de cobertura `UNKNOWN` (`R19`):**
- A cobertura `unknown` de `STRICT_ALL3_R19` resulta quase integralmente da ausência estrutural de demanda simultânea (`no_demand`), com baixíssima não-fechamento real em URLLC (`unknown_unclosed_demand`):
  - $\lambda=2$: `532 no_demand`, `1 partial_rx`, `0 zero_rx`, `0 closed_late_eof` (`STRICT_ALL3` `unk_reasons`: `533 multi_slice`, `31 mmtc_only`; `STRICT_DEMAND` `unk=1`, `no_demand=10`). Análise de sensibilidade URLLC (`sensitivity_all_rx`): `12.50%` (`4/32`) em ambos os braços.
  - $\lambda=5$: `1099 no_demand`, `2 partial_rx`, `2 closed_late_eof`, `0 zero_rx` (`unknown_unclosed_demand=4`; `STRICT_ALL3` `unk=1209`: `684 urllc_only`, `106 embb_only`, `419 multi_slice`). Sensibilidade `all_rx`: `17.65%` (`158/895`) no controle vs. `18.77%` (`168/895`) com `+A4`.
  - $\lambda=6$: `676 no_demand`, `1 partial_rx`, `2 closed_late_eof`, `0 zero_rx` (`unknown_unclosed_demand=3`; `STRICT_ALL3` `unk=797`: `451 urllc_only`, `88 embb_only`, `30 mmtc_only`, `228 multi_slice`). Sensibilidade `all_rx`: `32.85%` (`433/1318`) no controle vs. `38.69%` (`510/1318`) com `+A4`.
- Os pacotes `unreceived_at_300s` (`5`, `24`, `27–34`) representam `0.013%–0.159%` do tráfego TX pós-bootstrap e correspondem aos pacotes finais próximo a $t=300\text{ s}$ em ambos os braços.

---

### 1.3 Calibração Quantílica, Perda Pinball e Cobertura por Quantil Utilizado ($\tau_t$)

| $\lambda$ | Slice | Braço | `n_obs` / `sv` | $\bar{\tau}$ | `cov_exec` | `|cov - tau|` | `pb(tau)` | `pb(0.05)` | `MAE` / `Bias` | `q_exec >= 0` | `conf_obs` (`cov` / `pb`) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2** | 0 (eMBB) | `A_scaler` | `164` / `164` | `0.0480` | `0.0244` | `0.0237` | `0.1010` | `0.1031` | `1.9816` / `-1.9728` | `70.1%` | `147` (`0.0272` / `0.0843`) |
| **2** | 0 (eMBB) | `+A4` | `164` / `164` | `0.0480` | `0.0244` | `0.0237` | **`0.1248`** | **`0.1283`** | `2.4713` / `-2.4607` | `84.8%` | `143` (`0.0280` / `0.1073`) |
| **2** | 1 (URLLC) | `A_scaler` | `31` / `309` | `0.0439` | `0.0645` | `0.0206` | `0.2379` | `0.2379` | `0.4398` / `+0.0400` | `41.9%` | `14` (`0.1429` / `0.5137`) |
| **2** | 1 (URLLC) | `+A4` | `31` / `309` | `0.0439` | `0.0968` | **`0.0529`** | **`0.4754`** | **`0.4744`** | `0.7192` / `+0.2551` | `29.0%` | `9` (`0.3333` / `1.6062`) |
| **2** | 2 (mMTC) | `A_scaler` | `390` / `390` | `0.0427` | `0.0205` | `0.0222` | `0.0231` | `0.0234` | `0.0928` / `-0.0511` | `93.3%` | `364` (`0.0220` / `0.0241`) |
| **2** | 2 (mMTC) | `+A4` | `390` / `390` | `0.0427` | `0.0205` | `0.0222` | `0.0233` | `0.0236` | `0.0927` / `-0.0506` | `92.6%` | `361` (`0.0222` / `0.0244`) |
| **5** | 0 (eMBB) | `A_scaler` | `1519` / `1582` | `0.0202` | `0.0230` | `0.0029` | `0.0201` | `0.0454` | `0.8479` / `-0.8412` | `20.2%` | `581` (`0.0327` / `0.0343`) |
| **5** | 0 (eMBB) | `+A4` | `1519` / `1582` | `0.0212` | `0.0197` | `0.0014` | `0.0178` | `0.0335` | `0.6187` / `-0.6130` | `32.2%` | `964` (`0.0218` / `0.0235`) |
| **5** | 1 (URLLC) | `A_scaler` | `891` / `1159` | `0.0224` | `0.0281` | `0.0057` | `0.0482` | `0.0600` | `0.4490` / `-0.3654` | `12.9%` | `174` (`0.0402` / `0.0873`) |
| **5** | 1 (URLLC) | `+A4` | `891` / `1159` | `0.0224` | `0.0325` | `0.0102` | **`0.0577`** | **`0.0699`** | `0.4664` / `-0.3629` | `9.5%` | `229` (`0.0480` / **`0.1604`**) |
| **5** | 2 (mMTC) | `A_scaler` | `1944` / `1944` | `0.0250` | `0.0129` | `0.0122` | `0.0090` | `0.0104` | `0.0692` / `-0.0539` | `68.2%` | `1921` (`0.0130` / `0.0090`) |
| **5** | 2 (mMTC) | `+A4` | `1944` / `1944` | `0.0269` | `0.0139` | `0.0131` | `0.0106` | `0.0119` | `0.0739` / `-0.0557` | `62.8%` | `1911` (`0.0141` / `0.0106`) |
| **6** | 0 (eMBB) | `A_scaler` | `1680` / `1845` | `0.0213` | `0.0250` | `0.0037` | `0.0158` | `0.0302` | `0.5202` / `-0.5108` | `22.1%` | `1047` (`0.0220` / `0.0174`) |
| **6** | 0 (eMBB) | `+A4` | `1680` / `1845` | `0.0192` | `0.0280` | **`0.0088`** | `0.0139` | `0.0316` | `0.5389` / `-0.5285` | `8.8%` | **`554`** (`0.0415` / `0.0245`) |
| **6** | 1 (URLLC) | `A_scaler` | `1315` / `1584` | `0.0189` | `0.0274` | `0.0085` | `0.0293` | `0.0442` | `0.4972` / `-0.4541` | `1.0%` | `16` (`0.1875` / `0.1167`) |
| **6** | 1 (URLLC) | `+A4` | `1315` / `1584` | `0.0189` | `0.0259` | `0.0070` | **`0.0389`** | **`0.0596`** | `0.6211` / `-0.5576` | `2.5%` | `333` (`0.0300` / `0.0817`) |
| **6** | 2 (mMTC) | `A_scaler` | `1962` / `2000` | `0.0218` | `0.0133` | `0.0085` | `0.0086` | `0.0111` | `0.0959` / `-0.0819` | `40.2%` | `1891` (`0.0137` / `0.0085`) |
| **6** | 2 (mMTC) | `+A4` | `1962` / `2000` | `0.0243` | `0.0138` | `0.0105` | `0.0091` | `0.0107` | `0.0768` / `-0.0617` | `43.4%` | `1943` (`0.0139` / `0.0091`) |

---

### 1.4 Modos de Seleção, Ações Brutas/Executadas, Excesso $>53$ e Projeções por Slices Ativas

| $\lambda$ | Braço | `raw>53` global | `sum_raw` (`=3` ativas) | `raw>53` (`=3` ativas) | Projeção global | Proj. `1` ativa | Proj. `2` ativas | Proj. `=3` ativas |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **2** | `A1+A2a+A_scaler` | `5.50%` (`31/564`) | `N/A` (`n=0`) | `N/A` (`n=0`) | `5.85%` (`33/564`) | `5.26%` (`28/532`) | `15.63%` (`5/32`) | `N/A` (`0/0`) |
| **2** | `A1+A2a+A_scaler+A4` | `0.71%` (`4/564`) | `N/A` (`n=0`) | `N/A` (`n=0`) | `0.89%` (`5/564`) | `0.19%` (`1/532`) | `12.50%` (`4/32`) | `N/A` (`0/0`) |
| **5** | `A1+A2a+A_scaler` | `41.68%` (`831/1994`) | `64.24` (`min=15`, `p95=88.7`) | **`81.58%`** | `42.78%` (`853/1994`) | `0.74%` (`3/408`) | `25.53%` (`204/799`) | **`82.08%`** (`646/787`) |
| **5** | `A1+A2a+A_scaler+A4` | `45.39%` (`905/1994`) | `67.29` (`min=23`, `p95=87.7`) | **`90.85%`** | `46.44%` (`926/1994`) | `0.25%` (`1/408`) | `25.41%` (`203/799`) | **`91.74%`** (`722/787`) |
| **6** | `A1+A2a+A_scaler` | `61.63%` (`1229/1994`) | `71.47` (`min=14`, `p95=99.0`) | **`82.62%`** | `62.29%` (`1242/1994`) | `0.00%` (`0/227`) | `42.98%` (`245/570`) | **`83.29%`** (`997/1197`) |
| **6** | `A1+A2a+A_scaler+A4` | `71.72%` (`1430/1994`) | `68.23` (`min=32`, `p95=89.0`) | **`94.07%`** | `72.07%` (`1437/1994`) | `0.00%` (`0/227`) | `54.04%` (`308/570`) | **`94.32%`** (`1129/1197`) |

Modos (`confident`/`knn`/`softmax`) e ações médias `(a_raw / a_exec)` em todas as janelas, janelas ativas e `=3` ativas:

| $\lambda$ | Slice | `A1+A2a+A_scaler` (`c/k/sm`) | `A_scaler` `all (r/p/e)` · `act (r/e)` · `eq3 (r/e)` | `A1+A2a+A_scaler+A4` (`c/k/sm`) | `+A4` `all (r/p/e)` · `act (r/e)` · `eq3 (r/e)` |
|---:|---:|---|---|---|---|
| **2** | 0 | `547/15/2` | `7.24/7.31/6.97` · `24.76/23.83` · `—` | `543/18/3` | `8.12/8.17/8.11` · `27.75/27.71` · `—` |
| **2** | 1 | `98/420/46` | `3.20/3.70/2.93` · `17.16/18.41` · `—` | `91/427/46` | `1.65/1.96/1.87` · `13.81/14.28` · `—` |
| **2** | 2 | `538/24/2` | `6.70/6.93/6.93` · `9.47/9.79` · `—` | `535/27/2` | `7.82/8.02/8.02` · `11.05/11.33` · `—` |
| **5** | 0 | `1056/845/93` | `26.57/26.95/23.78` · `34.87/31.21` · `30.52/24.25` | `1439/501/54` | `25.60/25.99/22.47` · `33.61/29.50` · `31.18/24.09` |
| **5** | 1 | `174/1636/184` | `12.12/12.31/9.89` · `25.88/21.10` · `27.33/21.77` | `699/1170/125` | `13.62/14.11/11.02` · `29.35/23.42` · `29.12/22.67` |
| **5** | 2 | `1971/22/1` | `6.54/6.69/6.09` · `6.68/6.22` · `6.39/5.22` | `1961/31/2` | `8.28/8.46/7.62` · `8.46/7.78` · `6.98/5.42` |
| **6** | 0 | `1361/571/62` | `29.79/30.30/23.60` · `35.38/28.03` · `33.80/24.67` | `868/1021/105` | `28.70/29.34/23.91` · `34.09/28.40` · `30.45/23.67` |
| **6** | 1 | `16/1782/196` | `12.37/12.58/9.14` · `18.29/13.45` · `18.48/13.27` | `455/1388/151` | `19.72/20.20/15.62` · `29.13/22.97` · `29.02/22.46` |
| **6** | 2 | `1923/66/5` | `16.12/16.41/12.33` · `16.38/12.52` · `19.20/13.31` | `1975/18/1` | `9.53/9.69/8.00` · `9.67/8.12` · `8.76/6.52` |

### 1.5 Diagnóstico Mecanístico: Por que `--mean_field` (`A4`) Piora Mesmo com Scaler Corrigido (`"eum"`)

Mesmo corrigindo `"emu" -> "eum"`, a inclusão das duas dimensões globais (`prev_others/C`, `total_ues/N_MAX_CELL`) **degrada sistematicamente o controlador nos três regimes de carga**:
1. **Em $\lambda=2$:** embora reduza projeções em janelas vazias (`5.85% -> 0.89%`, via redução de `emp_araw` URLLC `2.36 -> 0.92`), a expansão para 6D **dobra a perda pinball em URLLC** (`0.2379 -> 0.4754`), dobra o `|cov-tau|` (`0.0206 -> 0.0529`), eleva a perda pinball de eMBB (`0.1031 -> 0.1283`), reduz `a_exec` ativo de URLLC (`18.41 -> 14.28`) e degrada a cauda de atraso (`p95`: `11.10 -> 20.10 ms`; `p99`: `43.10 -> 77.07 ms`; `miss`: `5.59% -> 6.07%`).
2. **Em $\lambda=5$:** URLLC e mMTC passam a propor lances mais agressivos (`act a_raw` URLLC `25.88 -> 29.35`, mMTC `6.68 -> 8.46`), com `pb(tau)` URLLC subindo (`0.0482 -> 0.0577`, e `conf_pb` quase dobrando `0.0873 -> 0.1604`). Isso empurra `raw>53` em `=3` ativas de `81.58%` para **`90.85%`** (projeção **`82.08% -> 91.74%`**, `+9.66 pp`), comprimindo eMBB ativo (`a_exec`: `31.21 -> 29.50`) e piorando **as 3 slices**: eMBB (`11.92% -> 13.56%`), URLLC (`17.51% -> 18.74%`), mMTC (`1.18% -> 1.34%`), fazendo `STRICT_ALL3_R19` cair **`71.72% -> 67.26%` (`-4.46 pp`)** e elevando `uTX miss` (`3.38% -> 4.49%`).
3. **Em $\lambda=6$:** o efeito é ainda mais severo:
   - Na eMBB (`s=0`), o espaço 6D com variáveis não-estacionárias de outras slices colapsa o regime `confident` pela metade (`1361 -> 868`, `conf_obs`: `1047 -> 554`), empurrando eMBB para `knn` (`571 -> 1021`), derrubando `q>=0` (`22.1% -> 8.8%`) e elevando `|c-t|` (`0.0037 -> 0.0088`).
   - Em URLLC (`s=1`), a perda pinball `pb(0.05)` sobe `+34.8%` (`0.0442 -> 0.0596`) e `MAE` sobe `0.4972 -> 0.6211`, enquanto URLLC trava um piso de proposta altíssimo (`act a_raw` `p10/p50/p90` salta de `[11, 15, 29]` para **`[22, 28, 40]`**; `a_raw >= 15` sobe de `51.5%` para **`95.0%`**!).
   - Consequentemente, em `=3` ativas, o piso de `sum_raw` salta de `14` para `32`, disparando `raw>53` de `82.62%` para **`94.07%`** (projeção **`94.32%`**) e gerando forte oscilação inter-janela na alocação e filas (`queue_u` médio ativo sobe `187.5 -> 200.1 B`; `queue_e` sobe `4.995M -> 5.019M B`). Em `trajectory_transitions` (`STRICT_ALL3_R19` em $\lambda=6$), `ok_to_fail = 317` supera largamente `fail_to_ok = 191`: **eMBB** piora `+5.78 pp` (`22.26% -> 28.04%`), **URLLC** piora `+5.86 pp` (`32.78% -> 38.63%`), `uTX miss` sobe `5.50% -> 8.08%`, **`STRICT_ALL3_R19` cai `-10.53 pp` (`55.30% -> 44.78%`)** e **`STRICT_DEMAND_R19` cai `-6.88 pp` (`66.75% -> 59.87%`)**.

**Conclusão do experimento $A_4$ limpo:** com a hipótese devidamente testada sob `"eum"`, o *mean-field* com controle independente sem seleção conjunta agrava a contenção por capacidade em vez de resolvê-la. O resultado em $\lambda \in \{2,5,6\}$ refuta sua inclusão no `B0` primário.

---

## 2. Auditoria Final de SLAs (`--slas 3.5,3.5,3.5` e SLAs Heterogêneos Reais)

### 2.1 Rastreamento Completo do Fluxo de `--slas 3.5,3.5,3.5`
1. **Origem CLI (`sqr_run.py`):** `--slas` (default `"3.5,3.5,3.5"`) passa por `parse_slas` -> `SqrEnv(..., sla_thresholds=slas)` -> `SqrObservationBuilder` -> `SqrKpmParser` ([`sqr_run.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_run.py#L1290-L1328)).
2. **Onde `--slas` tem efeito:** apenas no parser legado `SqrKpmParser.parse_window` (`violation_rate = count(thr < lam) / n` em [`sqr_kpis.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr/sqr_kpis.py#L385-L427)), gravando:
   - coluna auxiliar `violation_rate` em `step_metrics.csv`;
   - `reward` apenas informativo em `SqrEnv._compute_reward` (`-pooled_violation_rate`);
   - campo informativo `meta["slas"]` no checkpoint.
3. **Invariância estrutural no `SQR-HET`:**
   - Em `build_controllers` ([`sqr_run.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_run.py#L347-L364)), todos os 3 controladores `SqrSliceController` recebem **`sla_threshold = 0.0`** ($\Lambda = 0$).
   - Em `SqrHetController.__init__` ([`sqr_het.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/sqr_het.py#L85-L115)), há guarda explícita recusando `ctl.sla_threshold != 0.0` (`ValueError`).
   - Tanto os alvos do regressor ($y_i$), o estado $z_i$ (`FeatureScaler`), a perda pinball, quanto as atualizações de $\tau_i$ e $M_i$ consomem exclusivamente as **margens heterogêneas de `sla_margin.py` / `HetRuntime`**.
4. **Prova dinâmica executada ([`audit_sla_invariance.py`](file:///home/eliothluy/.gemini/antigravity/brain/4622041f-48e4-41e3-8c43-cf5834bb15d6/scratch/audit_sla_invariance.py)):**
   - Execução pareada de `120` janelas (`--controller sqr_het --warmup_s 2 --virtual_alpha 0.0 --empty_slice_guard --fix_scaler_index --urllc_delayed_update`) comparando `--slas 3.5,3.5,3.5` contra `--slas 99.0,0.01,500.0`:
     - `violation_rate` legado em `step_metrics.csv` mudou de `['0.500000', '1.000000']` para `['0.000000', '1.000000']`.
     - `SqrSliceController.sla_threshold` permaneceu `[0.0, 0.0, 0.0]` em ambos.
     - Todas as `120` linhas de `decisions.jsonl`, todas as decisões/modos por passo, todas as colunas de `step_metrics.csv` (exceto `violation_rate`/`reward`/`decision_time_ms`), `final_tau={0: 0.05, 1: 0.045000000000000005, 2: 0.05}`, `final_margin={0: 10.0, 1: 20.0, 2: 10.0}`, `support_points` e `alphas` foram **bit-idênticos (`True`)**.

### 2.2 Documentação dos 3 SLAs Heterogêneos Reais (`sla_manifest.yaml` `v1.1-g4b` + `sla_margin.py` + `het_runtime.py`)

| Slice | Métrica Primária e Critério | Escala de Margem Normalizada $y_i$ ($\Lambda=0$) | Janela de Avaliação | Regra de Ausência / Não-Fechamento (`R19`) |
|---|---|---|---|---|
| **0 — eMBB** | Mediana dos `windowThroughputMbps` dos UEs ativos (`dTxBytes > 0`) $\ge \mathbf{3{,}5\text{ Mbps}}$ | $(x_e - 3{,}5) / 3{,}5 \ge 0$, limitado inferiormente a $-1{,}0$ | Por janela de decisão de **`100 ms`** | Sem UE ativo / sem `dTxBytes > 0` $\Rightarrow$ `None` / `UNKNOWN` (excluído do aprendizado eMBB). |
| **1 — URLLC** | Percentil $p_{95}$ dos atrasos dos pacotes da janela de chegada $W$ $\le \mathbf{10{,}0\text{ ms}}$ *(secundária de diagnóstico: `miss_rate` $\le 5{,}0\%$)* | $(10{,}0 - p_{95}) / 10{,}0 \ge 0$, com piso $-1{,}0$ e teto $+1{,}0$ | Pacotes agrupados por **janela de chegada** $W = \lfloor \text{txSec} \cdot 10 \rfloor$ (`100 ms`), monitorada até fechamento online (`delivered == tx`) ou expiração (`_URLLC_MAX_AGE = 30` janelas) | Janela sem pacotes TX (`tx == 0`) ou pool ainda aberto/sem fechamento (`closed_online == False`) $\Rightarrow$ `None` / `UNKNOWN` (`PENDING_OBSERVATION`). **Nunca** trata falta de RX como sucesso ou falha presumida. |
| **2 — mMTC** | Taxa de entrega de pacotes (PDR = $\sum \text{dRxPackets} / \sum \text{dTxPackets}$) $\ge \mathbf{0{,}95}$ | $(\text{PDR} - 0{,}95) / 0{,}05 \ge 0$ (faixa $[-19, +1]$ normalizada com piso $-1{,}0$ em `mmtc_margin`) | **Janela móvel de `1 s`** (`10` janelas de `100 ms`) sobre as janelas com $\text{dTxPackets} > 0$ | Se $\sum \text{dTxPackets} == 0$ na janela móvel de `1 s` $\Rightarrow$ `None` / `UNKNOWN` (excluído do aprendizado mMTC). |

---

## 3. Proposta Objetiva para Fechamento do Baseline $B_0$ (`B0-rest`)

### 3.1 Recomendação Única
Recomenda-se fechar e congelar como **configuração única do baseline $B_0$ corrigido (`B0-rest`)** a configuração:

> **`B0 = A1 + A2a + A_scaler`** (`mean_field=False`, `M_0=10`, `M ∈ [2, 50]`, `--virtual_alpha 0.0`, `--empty_slice_guard`, `--fix_scaler_index`, `--urllc_delayed_update`, `--gamma 1`).

**Por que esta decisão é objetiva e encerra ajustes de componentes:**
1. **`A1+A2a` (`"emu"`) não pode ser congelado:** mantém um bug objetivo de indexação em `FeatureScaler` que trocava `n_{u,m}` e `queue_{u,m}` com distorção de até `2.0×`, inflacionando artificialmente `lambda=5` enquanto degradava `lambda=6` (`41.77% -> 55.30%`) e `lambda=2` (`16.13% -> 12.90%` URLLC).
2. **`A1+A2a+A_scaler+A4` foi refutado sob `"eum"` limpo:** reduziu `STRICT_ALL3_R19` em $\lambda=5$ (`-4.46 pp`) e em $\lambda=6$ (`-10.53 pp`), elevou projeções em `=3` ativas para `91.74%–94.32%`, piorou perda pinball em $\lambda=2,5,6$ e agravou violações URLLC e caudas de atraso nos três regimes. Mantém-se `--mean_field` **desativado** no `B0` primário (registrado apenas como controle de sensibilidade teórica/histórica).
3. **`A3a` (`M_0=3, [1,10]`):** mantém-se apenas como variante secundária de sensibilidade (não foi aprovada para o baseline principal e já havia perdido `-4.20 pp` em $\lambda=5$ e dobrado projeções em `=3` ativas `43.96% -> 87.04%` sobre `A1+A2a`).

### 3.2 Enquadramento Transparente em Três Camadas (`B0-rest` vs. Artigo Publicado)
Para rigor científico na redação e defesa metodológica (sem reivindicar fidelidade literal indevida ao publicado e sem atribuir falhas de adaptação ao artigo original):

| Camada | Componentes Cobertos | Relação Explícita com Tuerxun & Nakao (IEEE CCNC 2026) |
|---|---|---|
| **1. Núcleo Publicado (`[ARTIGO]`)** | Regressor quantílico online kernelizado com kernel Matérn ($\nu=1{,}5$), incerteza $\sigma(x) = 1 - \max_j k(x, x_j)$, perda pinball assimétrica com $P_{\text{penalty}}=2$, adaptação de $(\tau_t, M_t)$, seleção hierárquica (`A_safe` $\to$ `knn`/`ε-softmax`) com margem $+\text{round}(\sigma M)$ e custo $-\gamma a_i / C$. | Reprodução direta do núcleo algorítmico publicado (Seções III–IV). |
| **2. Adaptação Heterogênea (`[ADAPTAÇÃO]`)** | Normalização para margem comum ($\Lambda = 0$) dos 3 SLAs reais (`eMBB >= 3.5 Mbps`, `URLLC p95 <= 10 ms`, `mMTC PDR >= 0.95`), regra causal `R19` para janelas sem demanda/fechamento, *delayed feedback* online URLLC (`--urllc_delayed_update`), bootstrap `PropDemandController` (`0–100 s`), unidade física $C=53$ PRBs/RBGs (`1 RBG = 1 PRB/slot DL`), e **remoção do vetor global *mean-field* ($\bar{s}_t$)** em `B0` primário. | **Divergência explícita e documentada frente ao artigo:** além da adaptação multi-serviço, a remoção de $\bar{s}_t$ decorre de evidência empírica sob controle independente (`A4` degrada $\lambda \in \{2,5,6\}$ quando combinado à projeção independente $\Pi$). |
| **3. Correções Locais (`[CORREÇÃO]`)** | Desativação da atualização virtual incondicional `R12=-0.05` (`A1`: `--virtual_alpha 0.0`), supressão de proposta em slices sem demanda efetiva/pipeline (`A2a`: `--empty_slice_guard`), e correção de indexação do scaler `"emu" -> "eum"` (`A_scaler`: `--fix_scaler_index`). | Saneamento de parâmetros locais superespecificados/defeituosos da implementação base, antes da avaliação da contribuição central conjunta (`SQR-JC`). |

---

## 4. Proposta de Checagem Curta de Robustez (`seed=2103`, $\lambda \in \{5, 6\}$) — **SOMENTE PROPOSTA (NÃO EXECUTADA)**

Antes de destravar `A5`/`D3-v2` ou tocar as seeds independentes reservadas (`3101–3103`), propõe-se **apenas 1 checagem curta na família de desenvolvimento já aberta (`seed=2103`)**, com o objetivo estrito de verificar que o comportamento de `B0 = A1+A2a+A_scaler` em $\lambda \in \{5, 6\}$ não é artefato exclusivo de uma única trajetória da `seed=2102`:

- **Escopo mínimo proposto (somente após autorização explícita):** `2` runs de `B0 = A1+A2a+A_scaler` na `seed=2103` para $\lambda \in \{5, 6\}$ (`simTime=300 s`, `warmup_s=100 s`, exatamente as mesmas flags congeladas do `B0`, `cold start`, `--causal_trace_path` ativado apenas se autorizado ou validado em paridade separada). *(Opcionalmente, incluir `seed=2101` se desejar comparar contra o histórico `D1-v9`, mantendo `3101–3103` estritamente intactas.)*
- **Critérios pré-registrados de aceitação:**
  1. **Integridade de execução:** `ns3_exit_code == 0`, `RX == KPM`, `errors dim0..1 <= 1e-6`, `dims 4..5 == 0`, zero erro de contrato.
  2. **Saúde de calibração:** `|cov_exec - tau_mean| <= 0.03` nas 3 slices e ausência de deriva negativa extrema ($f_\tau$ não afunda abaixo de $-1.0$ por colapso de suporte; `bias` estável na mesma ordem da `seed=2102`).
  3. **Estabilidade de regime (vs. `seed=2102`):** `STRICT_ALL3_R19` em $\lambda=5$ acima de `65%` e em $\lambda=6$ acima de `45%`, com diferença $\le 15\text{ pp}$ em relação a `seed=2102` (`71.72%` e `55.30%`), mantendo taxa de `unreceived_at_300s < 0.5%` do TX pós-bootstrap.
  4. **Consistência de diagnóstico:** confirmação de que a contenção em `=3` ativas (`raw>53` e projeção elevados, faixa `~75%–90%`) persiste como limitação estrutural do controle desacoplado (`B0`), motivando a etapa `A5` (`JointSelector`).
- **Política de proteção de dados:** `seed=2103` permanece rotulada como família de desenvolvimento aberta; **`3101–3103` permanecem integralmente preservadas e intocadas** para avaliação confirmatória futura.

---

## 5. Contrato de Congelamento, Versionamento, Hashes e Rastreamento Causal (`D3-v2` / `A5` Prep)

Sem iniciar `A5` ou `D3-v2`, foi preparado e verificado o módulo [`freeze_contract.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/freeze_contract.py) integrado a [`sqr_run.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_run.py) e coberto por [`test_sqr_jc_freeze_contract.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/tests/test_sqr_jc_freeze_contract.py):

1. **Identificadores explícitos e hashes criptográficos (SHA-256):**
   - `FREEZE_CONTRACT_VERSION = "1.0"`, `CAUSAL_TRACE_VERSION = "1.0"`, `SQR_JC_VERSION = "0.1.0-preflight"`, `CHECKPOINT_FORMAT = "sqr-checkpoint"`, `CHECKPOINT_VERSION = 2`, `STATE_DICT_VERSION = 2`, `sla_manifest_version = "1.1-g4b"`.
   - `sla_manifest_sha256`: `12ffd2cfc4769b8c7d63c2ec501b755fe8a0038a2bc89687231dd3605304be2c` ([`sla_manifest.yaml`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/sla_manifest.yaml)).
   - `scaler_manifest_sha256`: `08db8784c0097df08d242bff4340448d33b096c86df8befc6026ba1965af5c16` ([`feature_scaler_manifest.yaml`](file:///home/eliothluy/Documentos/Jussi_SBRC/results/sqr_jc_final/03_calibration/feature_scaler_manifest.yaml)).
   - `B0 config_sha256` (`A1+A2a+A_scaler`, `mean_field=False`): `c3a768e8f3c0267acc8b8ba1872a6ebb70d7d138fd086d2d0b0539984d8178e7` (distinto de `+A4` `mean_field=True`: `ab90fcde8087a37b2b33438baf0ea7b087f0cbf523fdfa75dd972e929f5a1f8e`).
   - `_save_checkpoint_with_hash()` em `sqr_run.py` injeta `payload_sha256` determinístico no `meta` de cada checkpoint salvo.
2. **Guarda estrita de `--freeze` e `--resume`:**
   - `verify_resume_contract()` valida formato, versão, topologia bidirecional de slices, `payload_sha256` (detecta adulteração de pesos/margem), e recusa incompatibilidades em `mean_field`, `fix_scaler_index`, `virtual_alpha`, `empty_slice_guard`, `urllc_delayed_update`, `gamma`, `margin_init`, `margin_bounds`, `controller`, `config_sha256` e hashes de manifestos — preservando compatibilidade retroativa com checkpoints legados que não carreguem chaves estendidas.
   - Quando `--freeze` e `--resume` são usados juntos, `sqr_run.py` preserva integralmente o arquivo de checkpoint fonte em disco (`ep_checkpoint_path = None`) e `verify_freeze_no_learning()` garante estado invariante.
3. **Trace causal por janela sem efeitos colaterais (`--causal_trace_path`):**
   - `CausalTraceWriter.record_decision()` é invocado no exato instante pós-bootstrap da decisão `SQR_HET`, **antes** de `apply_virtual_updates()` e sem acesso a outcomes futuros, utilizando apenas `ctl.score_candidates(z)` (100% puro, zero RNG, zero mutação).
   - Registra por janela: `step_index`, `window_index`, `sim_time_ms`, `state` completo, `a_raw`, `a_prop`, `a_exec`, `cut`, `modes` e, para cada slice, `z`, `tau`, `margin_m`, `n_support`, `n_experiences`, `sum_alphas`, `model_state_sha256`, `f_candidates[0..C]`, `scores_candidates[0..C]` e `sigma_candidates[0..C]`.

---

## 6. Verificações Executadas após a Edição Final

- `bash tests/run_sqr_tests.sh -x -q` (incluindo os 5 testes novos de `tests/test_sqr_jc_freeze_contract.py` que validam paridade bit-a-bit entre execução com e sem `--causal_trace_path`, detecção de adulteração `payload_sha256`, bloqueios `--resume`, preservação em `--freeze --resume` e invariância sem aprendizado): **`345 passed, 2 warnings in 55.31s`** (`0` falhas, `0` skipped).
- `.venv/bin/python .../verify_a_scaler_a4_isolation.py`: **`ISOLATION VERIFICATION PASSED`**.
- `.venv/bin/python .../audit_sla_invariance.py`: **`=== SLA INVARIANCE AUDIT PASSED ===`**.
