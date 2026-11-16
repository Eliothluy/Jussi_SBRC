# Relatório Experimental $A_{\text{scaler}}$: Correção Isolada do `FeatureScaler` (`"emu"` $\to$ `"eum"`), Esclarecimento de SLAs e Fechamento de $B_0$

**Data:** 2026-11-14  
**Escopo:** Experimento isolado `A_scaler` (`--fix_scaler_index`) sobre `A1+A2a` na seed de desenvolvimento `2102`, $\lambda \in \{2, 5, 6\}$ (`300 s`, `warmup_s=100`, `periodMs=100`, `cellRadius=300`, `c_prbs=52 -> 53`, `--gamma 1`, `--virtual_alpha 0.0`, `--empty_slice_guard`, `--margin_init 10.0`, `--margin_bounds 2.0,50.0`, `--urllc_delayed_update`, *cold start*).  
**Artefatos preservados:** `D1-v9`, $A_0$, $A_1$, $A_{2\text{a}}$, $A_1+A_{2\text{a}}$, $A_{3\text{a}}$, $A_1+A_{2\text{a}}+A_{3\text{a}}$, $A_4$, $A_1+A_{2\text{a}}+A_4$ e todo o histórico permanecem intactos. `A2b`, `A3b`, `A5`, `D3-v2`, `PILOT`, `MAIN`, um eventual reteste `A1+A2a+A_scaler+A4` e as seeds independentes `3101–3103` permanecem **bloqueados** e não foram executados.

---

## 1. Correção Isolada (`"emu"` $\to$ `"eum"`) e Verificação Unitária

### 1.1 Mudanças implementadas e isolamento estrito

- [`MODA/ns-o-ran-gym/src/nsoran/sqr_jc/sqr_het.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/sqr_het.py): adicionado o parâmetro explícito `fix_scaler_index: bool = False` em `SqrHetController.__init__` (`self.fix_scaler_index = bool(fix_scaler_index)`) e corrigido o mapeamento em `slice_state()` apenas quando a flag está ativa:
  ```python
  letter = ("eum" if self.fix_scaler_index else "emu")[slice_id]
  ```
  - Com `fix_scaler_index=False` (padrão histórico): `0 -> 'e'`, `1 -> 'm'`, `2 -> 'u'`.
  - Com `fix_scaler_index=True` (`A_scaler`): `0 -> 'e'` (eMBB), `1 -> 'u'` (URLLC), `2 -> 'm'` (mMTC), alinhando `n_{letter}` e `queue_{letter}` às tuplas `(n_e, n_u, n_m)`, `(q_e, q_u, q_m)` e `_KPI_FEATURE = ("kpi_e", "kpi_u", "kpi_m")`.
- [`MODA/ns-o-ran-gym/examples/sqr_run.py`](../../../ns-o-ran-gym/examples/sqr_run.py): adicionada a flag CLI `--fix_scaler_index`, persistência em `meta["fix_scaler_index"]`, guarda de incompatibilidade em `--resume` (`ValueError` em divergência nas duas direções) e registro por janela de `"fix_scaler_index"` e `"z_dim0_dim1_per_slice"` nos registros `_het_pending` (`SQR_HET_BOOTSTRAP_PROP` e `SQR_HET`).
- **Componentes preservados sem alteração:** kernel Matérn ($\nu=1{,}5$), atualização pinball, regra `R10`, `R12` (`--virtual_alpha 0.0`), guarda `A2a` (`--empty_slice_guard`), parâmetros de `R11` (`M_0=10`, `M ∈ [2, 50]`), `mean_field=False`, SLAs, política de seleção, espaço de ações e atualização atrasada `R19` (`--urllc_delayed_update`).

### 1.2 Evidência unitária no manifesto congelado e nos testes

No manifesto congelado da calibração (`results/sqr_jc_final/03_calibration/feature_scaler_manifest.yaml`, `n_cal_samples = 23720`, `c = 53`), os limites de interesse são `p1 = 0.0` e `p99`: `n_e=4.0`, `n_u=3.0`, `n_m=2.0`, `queue_e=9844456.46`, `queue_u=709.0`, `queue_m=1422.0`.

Os testes [`test_sqr_het_scaler_indexing_audit_emu_bug`](../../../ns-o-ran-gym/tests/test_sqr_jc_controller.py) e [`test_sqr_het_scaler_indexing_fix_eum_a_scaler`](../../../ns-o-ran-gym/tests/test_sqr_jc_controller.py#L548-L608), somados ao script `verify_a_scaler_and_slas.py` (`VERIFICATION PASSED`) e à suíte `bash tests/run_sqr_tests.sh -x -q` (`345 passed, 2 warnings`), comprovaram com a sonda `n=(2,2,2)` e `queue=(1000,709,709)`:

| Slice (`slice_id`) | `"emu"` (`fix=False`) `(dim0, dim1)` | `"eum"` (`fix=True`) `(dim0, dim1)` | `dims 2..5` bit-idênticas | Efeito da correção |
|---|---|---|---|---|
| **eMBB (`0`)** | `(0.500000, 0.000102)` | `(0.500000, 0.000102)` | `True` | Sem alteração (`'e' -> 'e'`). |
| **URLLC (`1`)** | `(1.000000, 0.498594)` | `(0.666667, 1.000000)` | `True` | Remove superescala `1.5×` de `n_u` (saturação precoce) e subescala `0.4986×` de `queue_u`. |
| **mMTC (`2`)** | `(0.666667, 1.000000)` | `(1.000000, 0.498594)` | `True` | Remove subescala `0.6667×` de `n_m` e superescala `2.0056×` de `queue_m` (saturação `>709 B`). |

### 1.3 Integridade das 3 execuções `A1+A2a+A_scaler`

Diretório: `MODA/ns-o-ran-gym/results/sqr_jc_decision/a1_a2a_ascaler_runs_v9_va0_guard_eum/het_lam{2,5,6}_s2102`.

| Verificação | $\lambda = 2$ | $\lambda = 5$ | $\lambda = 6$ |
|---|---|---|---|
| `done` / `ns3_exit_code` | `True` / `0` | `True` / `0` | `True` / `0` |
| Decisões totais (`decisions.jsonl`) / `SQR_HET` | `848` / `565` | `2298` / `1995` | `2455` / `1995` |
| Bootstrap (`boot`) & paridade vs `A1+A2a` e `A0` | `283` (`bit_identical=True`) | `303` (`bit_identical=True`) | `460` (`bit_identical=True`) |
| `virtual_updates` / `kpm_missing` / `C_final` | `0` / `0` / `53` (`52->53`) | `0` / `0` / `53` (`52->53`) | `0` / `0` / `53` (`52->53`) |
| Consistência `rx == kpm` / `meta["fix_scaler_index"]` | `True` / `True` | `True` / `True` | `True` / `True` |
| Linhas com `z_dim0_dim1_per_slice` (`err` máx. reconstrução) | `848` (`4.98e-07`) | `2298` (`5.00e-07`) | `2455` (`5.00e-07`) |
| Latência de inferência `xApp` (`p50` / `p95` ms) | `2.42` / `3.80` | `11.55` / `17.96` | `12.73` / `22.12` |

---

## 2. Esclarecimento Obrigatório de SLAs e Unidades (`--slas 3.5,3.5,3.5` vs. SLAs Heterogêneos Reais)

Nos scripts experimentais, a linha de comando passa `--slas 3.5,3.5,3.5` ao `sqr_run.py`. A auditoria direta do código confirma que **não há** aplicação homogênea de `3.5 Mbps` às três slices no `SQR-HET`:

| Camada / Arquivo | Onde `--slas 3.5,3.5,3.5` atua | Onde vigoram os SLAs heterogêneos efetivos |
|---|---|---|
| **CLI & Meta (`examples/sqr_run.py`)** | Salvo em `meta["slas"] = {0: 3.5, 1: 3.5, 2: 3.5}` e repassado ao `SqrEnv.sla_thresholds`. | Nos 3 `SqrSliceController` sob `--controller sqr_het`, `sqr_run.py` fixa `sla_threshold = 0.0` ($\Lambda = 0$). |
| **Parser legado (`src/nsoran/sqr/sqr_env.py`)** | Alimenta exclusivamente `SqrKpmParser.violation_rate` sobre `lowest_thr_mbps` (legado homogêneo `--controller sqr` e coluna auxiliar em `step_metrics.csv`). | Não entra no regressor, na margem $y_i$, no aprendizado online nem nos avaliadores do `SQR-HET`. |
| **Normalização de margens (`src/nsoran/sqr_jc/sla_margin.py`)** | Não lê `--slas`. | Define `SLA_EMBB_MBPS = 3.5`, `SLA_URLLC_MS = 10.0`, `SLA_MMTC_PDR = 0.95` e `SLA_URLLC_MISS_PCT_SECONDARY = 5.0`. |
| **Runtime heterogêneo (`src/nsoran/sqr_jc/het_runtime.py`)** | Não lê `--slas`. | `HetRuntime.outcome_for_window()` e `DelayStream` avaliam eMBB, URLLC e mMTC nas escalas físicas reais abaixo. |
| **Scaler (`src/nsoran/sqr_jc/feature_scaler.py`)** | Não lê `--slas`. | `PHYSICAL_RULES` usa `kpi_e -> "kpi_div_sla"`, `kpi_u -> "kpi_div_deadline"` e `kpi_m -> "unit01"`. |
| **Manifesto e Oráculo (`results/sqr_jc_final/sla_manifest.yaml`, `oracle/dataset.py`, `oracle/adjudication.py`, avaliadores)** | Não lê `--slas`. | Mesmas constantes (`3.5` Mbps, `10.0` ms, `0.95` PDR) e mesmas regras de vácuo/fechamento (`R19`). |

**Definições operacionais exatas usadas nos controladores, margens e avaliações (`R36–R38` + `R19`):**

| Slice (`slice_id`) | Métrica física e janela | Limiar SLA efetivo | Predicado de satisfação | Margem normalizada $y_i$ enviada ao `SQR-HET` | Regra de vácuo / observação |
|---|---|---|---|---|---|
| **eMBB (`0`, R36)** | Mediana de `windowThroughputMbps` nos UEs com demanda (`dTxBytes > 0`) na janela de `100 ms` | $\ge 3{,}5\text{ Mbps}$ | $T_e \ge 3{,}5$ | $m_e = \frac{T_e - 3{,}5}{3{,}5}$ | Sem demanda (`dTxBytes == 0`) $\to$ excluída do aprendizado/falha (`None`). |
| **URLLC (`1`, R37)** | $p_{95}$ do atraso (`delaySec * 1000` em ms) dos pacotes agrupados por janela de chegada $\lfloor \text{txSec}\cdot 10\rfloor$ | $\le 10{,}0\text{ ms}$ | $D_{p95} \le 10{,}0$ | $m_u = \frac{10{,}0 - D_{p95}}{10{,}0}$ | Só conta e atualiza quando o pool fecha online (`delivered == tx > 0`). Métrica secundária *report-only* (nunca *gate*): `miss_rate` ($> 10{,}0\text{ ms}$, ref. `5,0%`). |
| **mMTC (`2`, R38)** | PDR rolante de $1\text{ s}$ (`10` janelas): $\sum dRxPackets / \sum dTxPackets$ | $\ge 0{,}95$ (`95%`) | $\text{PDR} \ge 0{,}95$ | $m_m = \frac{\text{PDR} - 0{,}95}{0{,}95}$ | Sem TX na janela rolante $\to$ excluída (`None`). |

---

## 3. Avaliação Experimental: `A1+A2a+A_scaler` vs. `A1+A2a` e Matriz de 9 Braços

*(Fonte: `a_scaler_evaluation_summary.json`, `a_scaler_evaluation_comparison.csv` e `a_scaler_diagnostics.json` em `MODA/ns-o-ran-gym/results/sqr_jc_decision/`.)*

### 3.1 Verificação de `z_dim0_dim1_per_slice` e impacto na geometria do kernel Matérn

Abaixo comparam-se as coordenadas normalizadas `(dim0 = n_norm, dim1 = queue_norm)` nas janelas pós-warmup avaliadas (`het_closed`), a incerteza causal online $\sigma_t = 1 - \max_j k(x_t, x_j)$ com distância causal invertida $d_{\text{caus}}$ no kernel Matérn $\nu=1{,}5$, e a geometria *leave-one-out* do buffer de suporte final (`N`, distância Matérn `d_nn`, distância euclidiana em `dim0..1` `d01` e similaridade máxima `k_nn`):

| $\lambda$ | Slice (letra usada) | Braço | `n_norm` (`mean` / `sat=1.0`) | `queue_norm` (`mean` / `sat=1.0`) | `sigma` causal (`mean` / `p95` / `>0.5`) | `d_caus` (`mean` / `p95`) | Buffer (`N` / `sp n_sat1` / `sp q_sat1` / `d_nn` / `d01`) |
|---|---|---|---|---|---|---|---|
| **2** | `1` URLLC (`m`) | `A1+A2a` | `0.0284` / `0.0%` | `0.0089` / `0.2%` | `0.1250` / `0.2888` / `0.2%` | `0.3360` / `0.6159` | `309` / `0.0%` / `0.3%` / `0.0122` / `0.0018` |
| **2** | `1` URLLC (`u`) | `A1+A2a+A_scaler` | `0.0189` / `0.0%` | `0.0111` / `0.4%` | `0.1266` / `0.4155` / `0.0%` | `0.3429` / `0.8206` | `309` / `0.0%` / `0.6%` / `0.0152` / `0.0014` |
| **2** | `2` mMTC (`u`) | `A1+A2a` | `0.2358` / `0.0%` | `0.2166` / `0.0%` | `0.3155` / `1.0000` / `31.4%` | `6.8663` / `22.0592` | `390` / `0.0%` / `0.0%` / `0.0150` / `0.0002` |
| **2** | `2` mMTC (`m`) | `A1+A2a+A_scaler` | `0.3537` / `0.0%` | `0.1102` / `0.0%` | `0.3154` / `1.0000` / `31.4%` | `6.8671` / `22.0592` | `390` / `0.0%` / `0.0%` / `0.0146` / `0.0001` |
| **5** | `0` eMBB (`e`) | `A1+A2a` | `0.4283` / `3.4%` | `0.3366` / `2.6%` | `0.1255` / `0.4839` / `0.2%` | `0.2738` / `0.9395` | `1582` / `4.2%` / `3.3%` / `0.0335` / `0.0011` |
| **5** | `0` eMBB (`e`) | `A1+A2a+A_scaler` | `0.4283` / `3.4%` | `0.3399` / `0.9%` | `0.1265` / `0.4839` / `0.2%` | `0.2805` / `0.9395` | `1582` / `4.2%` / `1.1%` / `0.0410` / `0.0009` |
| **5** | `1` URLLC (`m`) | `A1+A2a` | `0.2961` / **`14.3%`** | `0.0455` / `0.4%` | `0.1030` / `0.2675` / `0.1%` | `0.2544` / `0.5825` | `1159` / `28.0%` / `0.7%` / `0.0089` / `0.0002` |
| **5** | `1` URLLC (`u`) | `A1+A2a+A_scaler` | `0.1974` / **`0.0%`** | `0.0874` / `0.5%` | `0.0463` / `0.0890` / `0.0%` | `0.1593` / `0.2864` | `1159` / `0.0%` / `0.8%` / `0.0111` / `0.0003` |
| **5** | `2` mMTC (`u`) | `A1+A2a` | `0.7566` / `55.3%` | `0.6766` / **`40.0%`** | `0.0299` / `0.0378` / `2.7%` | `0.5883` / `0.1754` | `1944` / `56.7%` / `41.0%` / `0.0115` / `0.0001` |
| **5** | `2` mMTC (`m`) | `A1+A2a+A_scaler` | `0.8583` / `73.7%` | `0.3622` / **`0.0%`** | `0.0294` / `0.0301` / `2.7%` | `0.5864` / `0.1546` | `1944` / `75.6%` / `0.0%` / `0.0109` / `0.0001` |
| **6** | `0` eMBB (`e`) | `A1+A2a` | `0.5260` / `7.9%` | `0.4120` / `5.5%` | `0.1047` / `0.5585` / `15.9%` | `0.2523` / `1.0814` | `1845` / `8.6%` / `5.9%` / `0.0441` / `0.0009` |
| **6** | `0` eMBB (`e`) | `A1+A2a+A_scaler` | `0.5260` / `7.9%` | `0.4246` / `3.1%` | `0.1014` / `0.5585` / `15.9%` | `0.2450` / `1.0814` | `1845` / `8.6%` / `3.3%` / `0.0396` / `0.0009` |
| **6** | `1` URLLC (`m`) | `A1+A2a` | `0.5148` / **`36.9%`** | `0.0856` / `0.4%` | `0.0464` / `0.1467` / `0.1%` | `0.1390` / `0.3890` | `1584` / `48.9%` / `0.4%` / `0.0109` / `0.0003` |
| **6** | `1` URLLC (`u`) | `A1+A2a+A_scaler` | `0.3574` / **`4.3%`** | `0.1569` / `1.7%` | `0.0613` / `0.2246` / `0.0%` | `0.1627` / `0.5150` | `1584` / `5.2%` / `2.1%` / `0.0096` / `0.0005` |
| **6** | `2` mMTC (`u`) | `A1+A2a` | `0.7340` / `55.3%` | `0.7057` / **`49.7%`** | `0.0134` / `0.0347` / `1.6%` | `0.0461` / `0.1672` | `2000` / `51.1%` / `45.5%` / `0.0114` / `0.0000` |
| **6** | `2` mMTC (`m`) | `A1+A2a+A_scaler` | `0.8245` / `66.5%` | `0.4146` / **`0.0%`** | `0.0161` / `0.0419` / `1.6%` | `0.0615` / `0.1858` | `2000` / `62.2%` / `0.0%` / `0.0113` / `0.0000` |

**Leitura geométrica:** A correção elimina completamente o teto artificial de `n_u` em $\lambda=5$ (`14.3% -> 0.0%`, `28.0% -> 0.0%` no buffer) e derruba-o em $\lambda=6$ (`36.9% -> 4.3%`, `48.9% -> 5.2%` no buffer), enquanto dobra a resolução de `queue_u` (`0.0455 -> 0.0874` em $\lambda=5$; `0.0856 -> 0.1569` em $\lambda=6$) e zera a saturação artificial de `queue_m` em ambos os regimes (`40.0% -> 0.0%` e `49.7% -> 0.0%`). Em URLLC, a distância espacial em `d01` aumenta (`0.0002 -> 0.0003` em $\lambda=5$; `0.0003 -> 0.0005` em $\lambda=6$).

### 3.2 Calibração quantílica e perda pinball ($\tau_{\text{adapt}}$, $\tau=0{,}05$, $\tau=0{,}5$, $\tau=0{,}95$)

*(Nota metodológica: o regressor `SQR-HET` prevê o quantil de segurança inferior adaptativo $\tau_{i,t} \in [0{,}01, 0{,}05]$ sobre a margem $y_i$; a cobertura empírica é $\text{cov} = \mathbb{P}(y < q_{\text{exec}})$, comparada a $\tau_{\text{adapt}}$. Os valores de `pb(0.5)` e `pb(0.95)` também são reportados sobre `q_exec` conforme solicitado para auditoria numérica, mas não correspondem ao nível quantílico visado pelo controlador.)*

| $\lambda$ | Slice | Braço | `n_obs` | $\tau_{\text{adapt}}$ | `cov` | `|c-t|` | `pb(tau_t)` | `pb(0.05)` | `pb(0.5)` | `pb(0.95)` | `mae` | `bias` | `q>=0` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **2** | `0` eMBB | `A1+A2a` | `164` | `0.0480` | `0.0244` | `0.0237` | `0.1039` | `0.1063` | `1.0228` | `1.9394` | `2.0457` | `-2.0369` | `86.6%` |
| **2** | `0` eMBB | `+A_scaler` | `164` | `0.0480` | `0.0244` | `0.0237` | `0.1010` | `0.1031` | `0.9908` | `1.8786` | `1.9816` | `-1.9728` | `70.1%` |
| **2** | `1` URLLC | `A1+A2a` | `31` | `0.0439` | `0.0968` | `0.0529` | `0.2950` | `0.2952` | `0.2812` | `0.2673` | `0.5625` | `+0.0309` | `16.1%` |
| **2** | `1` URLLC | `+A_scaler` | `31` | `0.0439` | **`0.0645`** | **`0.0206`** | **`0.2379`** | `0.2379` | `0.2199` | `0.2019` | **`0.4398`** | `+0.0400` | **`41.9%`** |
| **2** | `2` mMTC | `A1+A2a` | `390` | `0.0427` | `0.0205` | `0.0222` | `0.0232` | `0.0235` | `0.0463` | `0.0691` | `0.0926` | `-0.0506` | `93.3%` |
| **2** | `2` mMTC | `+A_scaler` | `390` | `0.0427` | `0.0205` | `0.0222` | `0.0231` | `0.0234` | `0.0464` | `0.0694` | `0.0928` | `-0.0511` | `93.3%` |
| **5** | `0` eMBB | `A1+A2a` | `1519` | `0.0228` | `0.0197` | `0.0030` | `0.0206` | `0.0398` | `0.3689` | `0.6979` | `0.7377` | `-0.7312` | `55.4%` |
| **5** | `0` eMBB | `+A_scaler` | `1519` | `0.0202` | `0.0230` | `0.0029` | `0.0201` | `0.0454` | `0.4239` | `0.8025` | `0.8479` | `-0.8412` | **`20.2%`** |
| **5** | `1` URLLC | `A1+A2a` | `891` | `0.0207` | `0.0213` | `0.0006` | `0.0595` | `0.0696` | `0.2000` | `0.3304` | `0.4000` | `-0.2898` | `26.2%` |
| **5** | `1` URLLC | `+A_scaler` | `891` | `0.0224` | `0.0281` | `0.0057` | **`0.0482`** | `0.0600` | `0.2245` | `0.3889` | `0.4490` | `-0.3654` | **`12.9%`** |
| **5** | `2` mMTC | `A1+A2a` | `1944` | `0.0275` | `0.0149` | `0.0125` | `0.0101` | `0.0113` | `0.0342` | `0.0572` | `0.0685` | `-0.0510` | `81.6%` |
| **5** | `2` mMTC | `+A_scaler` | `1944` | `0.0250` | `0.0129` | `0.0122` | `0.0090` | `0.0104` | `0.0346` | `0.0588` | `0.0692` | `-0.0539` | `68.2%` |
| **6** | `0` eMBB | `A1+A2a` | `1680` | `0.0213` | `0.0286` | `0.0073` | `0.0157` | `0.0330` | `0.2811` | `0.5293` | `0.5623` | `-0.5514` | `19.2%` |
| **6** | `0` eMBB | `+A_scaler` | `1680` | `0.0213` | `0.0250` | **`0.0037`** | `0.0158` | `0.0302` | `0.2601` | `0.4900` | **`0.5202`** | `-0.5108` | `22.1%` |
| **6** | `1` URLLC | `A1+A2a` | `1315` | `0.0189` | `0.0266` | `0.0078` | `0.0274` | `0.0420` | `0.2387` | `0.4354` | `0.4774` | `-0.4372` | `1.4%` |
| **6** | `1` URLLC | `+A_scaler` | `1315` | `0.0189` | `0.0274` | `0.0085` | `0.0293` | `0.0442` | `0.2486` | `0.4529` | `0.4972` | `-0.4541` | `1.0%` |
| **6** | `2` mMTC | `A1+A2a` | `1962` | `0.0228` | `0.0127` | `0.0100` | `0.0092` | `0.0111` | `0.0423` | `0.0734` | `0.0846` | `-0.0692` | `41.8%` |
| **6** | `2` mMTC | `+A_scaler` | `1962` | `0.0218` | `0.0133` | **`0.0085`** | **`0.0086`** | `0.0111` | `0.0479` | `0.0848` | `0.0959` | `-0.0819` | `40.2%` |

### 3.3 Quadro Consolidado de 9 Braços (`seed = 2102`, $\lambda \in \{2, 5, 6\}$)

#### $\lambda = 2$ (`STRICT_ALL3_R19 = 0/0/564` por ausência de janelas com `=3` slices ativas simultâneas; `u_tx = 3151`, `rx = 3146`, `unrx = 5`)

| Braço | `a_raw` `[e, u, m]` | `raw>53` / `proj` (global) | `util` | `a_exec` `[e, u, m]` | `viol [eMBB, URLLC, mMTC]` | `STRICT_DEMAND_R19` (`P/F/U`) | URLLC TX `p95`/`p99`/`max` / `miss` (`r19win_max` / `r19miss`) |
|---|---|---|---|---|---|---|---|
| `A0` | `[17.53, 13.57, 5.21]` | `13.65%` / `14.18%` | `4.63%` | `[7.16, 13.06, 2.71]` | `[1.22%, 19.35%, 0.00%]` | `98.55%` (`545/8/1`) | `18.10`/`43.10`/`68.10` / `8.49%` (`64.90` / `8.71%`) |
| `A1` | `[6.48, 6.36, 6.44]` | `0.71%` / `5.32%` | `3.43%` | `[6.34, 10.99, 4.49]` | `[2.44%, 6.45%, 2.05%]` | `97.47%` (`539/14/1`) | `15.10`/`75.65`/`106.10` / `6.23%` (`99.15` / `6.39%`) |
| `A2a` | `[7.33, 1.56, 6.37]` | `0.53%` / `0.71%` | `3.92%` | `[6.10, 9.25, 3.20]` | `[2.44%, 16.13%, 2.05%]` | `96.93%` (`536/17/1`) | `18.10`/`43.10`/`68.10` / `8.30%` (`64.90` / `8.52%`) |
| `A1+A2a` | `[7.14, 1.73, 6.37]` | `0.71%` / `0.89%` | `3.81%` | `[5.27, 9.32, 3.20]` | `[2.44%, 16.13%, 2.05%]` | `96.93%` (`536/17/1`) | `18.10`/`43.10`/`68.10` / `8.42%` (`64.90` / `8.65%`) |
| `A3a` | `[19.94, 4.71, 6.51]` | `10.82%` / `11.70%` | `4.54%` | `[7.89, 10.08, 3.22]` | `[2.44%, 16.13%, 5.64%]` | `94.39%` (`522/31/1`) | `12.80`/`24.10`/`29.10` / `6.36%` (`27.10` / `6.53%`) |
| `A1+A2a+A3a` | `[7.46, 1.49, 6.25]` | `0.53%` / `0.53%` | `3.98%` | `[4.92, 9.11, 3.13]` | `[3.05%, 19.35%, 2.05%]` | `96.75%` (`535/18/1`) | `20.10`/`51.65`/`70.10` / `9.57%` (`68.10` / `9.82%`) |
| `A4` | `[11.01, 9.24, 7.87]` | `9.57%` / `11.35%` | `1.79%` | `[5.42, 13.32, 3.57]` | `[24.39%, 6.45%, 0.26%]` | `92.59%` (`512/41/1`) | `8.10`/`8.61`/`15.10` / `0.60%` (`10.15` / `0.62%`) |
| `A1+A2a+A4` | `[8.53, 2.53, 7.36]` | `5.50%` / `6.91%` | `4.35%` | `[5.17, 9.43, 3.47]` | `[2.44%, 16.13%, 2.05%]` | `96.93%` (`536/17/1`) | `24.80`/`76.10`/`107.10` / `9.35%` (`98.15` / `9.43%`) |
| **`A1+A2a+A_scaler`** | `[7.24, 3.20, 6.70]` | `5.50%` / `5.85%` | `3.73%` | `[5.17, 9.51, 3.29]` | **`[2.44%, 12.90%, 2.05%]`** | **`97.11%`** (`537/16/1`) | **`11.10`**/`43.10`/`68.10` / **`5.59%`** (`64.90` / `5.74%`) |

#### $\lambda = 5$ (`active_3 = 787` janelas; `u_tx = 119182`, `rx = 119158`, `unrx = 24`)

| Braço | `raw>53` (global / `=3`) | `proj` (global / `=3`) | `util` | `a_exec` `[e, u, m]` | `viol [eMBB, URLLC, mMTC]` | `STRICT_ALL3_R19` (`P/F/U`) | `STRICT_DEMAND_R19` (`P/F/U`) | URLLC `p95`/`p99`/`max` / `miss` (`r19win_max` / `r19miss`) |
|---|---|---|---|---|---|---|---|---|
| `A0` | `15.10%` / `7.62%` | `16.00%` / `7.88%` | `27.00%` | `[16.72, 11.87, 3.78]` | `[66.95%, 24.35%, 2.78%]` | `27.39%` (`215/570/1209`) | `44.52%` (`886/1104/4`) | `9.10`/`14.10`/`43.10` / `4.16%` (`41.10` / `4.19%`) |
| `A1` | `62.04%` / `72.55%` | `66.05%` / `73.44%` | `35.78%` | `[23.08, 20.27, 5.91]` | `[32.65%, 15.04%, 1.29%]` | `73.50%` (`577/208/1209`) | `68.14%` (`1356/634/4`) | `8.10`/`14.10`/`108.10` / `2.99%` (`99.15` / `3.00%`) |
| `A2a` | `8.12%` / `14.74%` | `8.83%` / `15.76%` | `33.59%` | `[16.85, 4.53, 5.53]` | `[37.00%, 17.40%, 8.49%]` | `50.57%` (`397/388/1209`) | `63.12%` (`1256/734/4`) | `9.10`/`15.10`/`107.10` / `3.84%` (`98.15` / `3.82%`) |
| `A1+A2a` | `24.52%` / `43.20%` | `26.18%` / `43.96%` | `44.94%` | `[22.87, 7.95, 6.19]` | `[5.99%, 9.20%, 1.29%]` | **`84.20%`** (`661/124/1209`) | **`90.90%`** (`1809/181/4`) | `8.10`/`13.10`/`108.10` / `2.22%` (`101.15` / `2.20%`) |
| `A3a` | `14.54%` / `7.62%` | `14.94%` / `7.62%` | `24.72%` | `[15.58, 13.81, 2.90]` | `[63.66%, 21.21%, 3.70%]` | `39.11%` (`307/478/1209`) | `44.52%` (`886/1104/4`) | `9.10`/`14.10`/`108.10` / `4.04%` (`98.15` / `3.96%`) |
| `A1+A2a+A3a` | `42.03%` / `87.04%` | `42.13%` / `87.04%` | `44.17%` | `[22.47, 10.82, 6.26]` | `[9.15%, 5.84%, 1.23%]` | `80.00%` (`628/157/1209`) | `90.00%` (`1791/199/4`) | `8.10`/`10.10`/`107.10` / `1.24%` (`100.15` / `1.25%`) |
| `A4` | `10.53%` / `5.08%` | `12.24%` / `6.48%` | `25.50%` | `[17.65, 8.28, 2.67]` | `[68.40%, 22.56%, 11.42%]` | `26.24%` (`206/579/1209`) | `41.71%` (`830/1160/4`) | `10.10`/`15.10`/`34.10` / `5.38%` (`31.10` / `5.39%`) |
| `A1+A2a+A4` | `46.64%` / `83.35%` | `47.59%` / `83.74%` | `45.97%` | `[23.39, 9.21, 8.26]` | `[7.83%, 20.09%, 1.29%]` | `73.76%` (`579/206/1209`) | `85.08%` (`1693/297/4`) | `9.10`/`14.10`/`108.10` / `4.03%` (`101.15` / `4.03%`) |
| **`A1+A2a+A_scaler`** | `41.68%` / **`81.58%`** | `42.78%` / **`82.08%`** | `46.75%` | `[23.79, 9.92, 6.08]` | **`[11.92%, 17.51%, 1.18%]`** | **`71.72%`** (`563/222/1209`) | **`83.22%`** (`1656/334/4`) | `8.10`/`14.10`/`108.10` / `3.38%` (`99.15` / `3.36%`) |

*(Nota: em `sched_per_slot` de $\lambda=5$, `A1+A2a+A_scaler` executou `[23.78, 0.98, 0.02]` RBGs/slot vs. `[22.86, 0.94, 0.02]` em `A1+A2a`.)*

#### $\lambda = 6$ (`active_3 = 1197` janelas; `u_tx = 215464`)

| Braço | `raw>53` (global / `=3`) | `proj` (global / `=3`) | `util` | `a_exec` `[e, u, m]` | `viol [eMBB, URLLC, mMTC]` | `STRICT_ALL3_R19` (`P/F/U`) | `STRICT_DEMAND_R19` (`P/F/U`) | `unrx` | URLLC `p95`/`p99`/`max` / `miss` (`r19win_max` / `r19miss`) |
|---|---|---|---|---|---|---|---|---|---|
| `A0` | `14.84%` / `17.38%` | `16.05%` / `18.63%` | `38.83%` | `[18.84, 11.82, 4.01]` | `[54.88%, 33.77%, 3.26%]` | `35.24%` (`418/768/808`) | `47.62%` (`941/1035/18`) | `2601` | `83.03`/`307.10`/`374.10` / `14.55%` (`357.10` / `13.68%`) |
| `A1` | `71.77%` / `80.03%` | `73.42%` / `80.45%` | `45.18%` | `[22.85, 18.96, 6.68]` | `[38.99%, 46.92%, 1.27%]` | `43.61%` (`522/675/797`) | `45.86%` (`913/1078/3`) | `27` | `12.10`/`18.10`/`110.10` / `10.06%` (`103.15` / `9.89%`) |
| `A2a` | `14.74%` / `20.97%` | `16.85%` / `22.64%` | `44.89%` | `[22.03, 6.86, 4.21]` | `[43.04%, 42.18%, 3.41%]` | `36.02%` (`429/762/803`) | `50.40%` (`998/982/14`) | `2830` | `42.10`/`252.10`/`512.10` / `16.31%` (`487.03` / `15.69%`) |
| `A1+A2a` | `63.69%` / `86.30%` | `64.14%` / `86.63%` | `47.19%` | `[22.89, 12.49, 9.38]` | `[29.70%, 43.27%, 1.27%]` | `41.77%` (`500/697/797`) | `57.76%` (`1150/841/3`) | `27` | `11.10`/`15.10`/`89.10` / `8.42%` (`84.85` / `8.41%`) |
| `A3a` | `14.09%` / `17.29%` | `14.44%` / `17.63%` | `41.26%` | `[20.09, 10.12, 2.97]` | `[66.13%, 41.06%, 9.68%]` | `21.47%` (`257/940/797`) | `35.61%` (`709/1282/3`) | `27` | `13.10`/`27.10`/`113.10` / `11.71%` (`104.10` / `11.44%`) |
| `A1+A2a+A3a` | `57.62%` / `88.14%` | `57.67%` / `88.14%` | `45.52%` | `[22.31, 10.92, 11.27]` | `[28.39%, 22.13%, 1.48%]` | **`56.89%`** (`681/516/797`) | **`67.30%`** (`1340/651/3`) | `27` | `9.10`/`14.10`/`107.10` / `3.50%` (`99.15` / `3.47%`) |
| `A4` | `12.69%` / `15.71%` | `14.34%` / `17.29%` | `40.38%` | `[19.87, 7.07, 8.57]` | `[44.17%, 28.09%, 1.63%]` | `49.96%` (`596/597/801`) | `55.94%` (`1111/875/8`) | `1293` | `56.10`/`229.03`/`361.10` / `13.65%` (`356.03` / `12.76%`) |
| `A1+A2a+A4` | `70.31%` / `94.65%` | `70.96%` / `95.07%` | `48.80%` | `[23.77, 14.91, 8.51]` | `[29.11%, 42.36%, 1.27%]` | `42.36%` (`507/690/797`) | `58.61%` (`1167/824/3`) | `27` | `12.10`/`15.10`/`107.10` / `8.83%` (`100.15` / `8.71%`) |
| **`A1+A2a+A_scaler`** | `61.63%` / **`82.62%`** | `62.29%` / **`83.29%`** | `48.01%` | `[23.62, 9.15, 12.31]` | **`[22.26%, 32.78%, 1.33%]`** | **`55.30%`** (`662/535/797`) | **`66.75%`** (`1329/662/3`) | `34` | **`10.10`**/`15.10`/`89.10` / **`5.50%`** (`84.85` / **`5.48%`**) |

*(Em `sched_per_slot` de $\lambda=6$, `A1+A2a+A_scaler` executou `[23.61, 1.82, 0.02]` RBGs/slot vs. `[22.88, 2.11, 0.02]` em `A1+A2a`; `unrx=34` em `A_scaler` vs `27` ocorre no encerramento da cauda de `300 s`, ambos `>99.98%` de entregas concluídas `215430/215464` vs `215437/215464`.)*

### 3.4 Diagnóstico causal do contraste $\lambda=5$ (`-12.48 pp`) vs. $\lambda=6$ (`+13.53 pp`)

1. **Em $\lambda=2$ (`564` janelas):** os dois braços permanecem idênticos até `w=2420` (primeira divergência em `action[1]: 9 -> 7`). Com `A_scaler`, a calibração URLLC melhora claramente (`|c-t|` cai `0.0529 -> 0.0206`, `pinball` `0.2950 -> 0.2379`, `q>=0` sobe `16.1% -> 41.9%`), derrubando violações URLLC (`16.13% -> 12.90%`), `p95` TX (`18.10 -> 11.10 ms`), `miss` (`8.42% -> 5.59%`) e elevando `STRICT_DEMAND_R19` (`96.93% -> 97.11%`).
2. **Em $\lambda=5$ (`ok_to_fail=163` vs. `fail_to_ok=65` em `STRICT_ALL3_R19`):**
   - A divergência começa logo após o bootstrap, em `w=1002` (`[1, 1, 0]`, `action[1]: 13 -> 14`).
   - Sob a escala limpa `"eum"` (sem o colapso artificial de `n_u` e `queue_u`), as previsões de margem do regressor URLLC caem para valores negativos (`q>=0` cai `26.2% -> 12.9%`; `sum_alpha` final cai `-0.60 -> -0.95`), reduzindo drasticamente o modo `confident` em URLLC de **`1009` para apenas `174` janelas** e elevando os fallbacks **`knn` (`892 -> 1636`)** e **`softmax` (`94 -> 185`)**.
   - Como `knn` seleciona alocações históricas altas que passaram no passado, a proposta ativa média URLLC salta **`+6.32 PRBs` (`19.56 -> 25.88`, e em `=3` ativas `20.1 -> 27.3`)**, fazendo `raw>53` em `=3` ativas quase dobrar (`43.20% -> 81.58%`, projeção `43.96% -> 82.08%`).
   - Essa explosão de contenção comprime eMBB nas `787` janelas `=3` ativas (`a_exec[0]`: `25.3 -> 24.3`), desloca eMBB (`confident`: `1588 -> 1056`, `knn`: `371 -> 846`, `q>=0`: `55.4% -> 20.2%`) — respondendo por `103/142` transições `ok->fail` de eMBB por subalocação (`37.55 -> 31.50` PRBs) — e introduz episódios agudos de exploração `softmax` ou inanição em URLLC (como `w=1499`, `1562`, `1627`, `1629` com `0–1` PRB e `w=1728` com `raw=exec=[27, 0, 9]` em `mode=softmax` imediatamente após `w=1727` com `29` PRBs, causando transbordamento `p95=12.90 ms`). Resultado líquido: `STRICT_ALL3_R19` cai `84.20% -> 71.72%` (`-12.48 pp`) e `STRICT_DEMAND_R19` cai `90.90% -> 83.22%`.
3. **Em $\lambda=6$ (`fail_to_ok=315` vs. `ok_to_fail=153` em `STRICT_ALL3_R19`):**
   - Sob carga saturada, o efeito de trajetória inverte-se: enquanto `A1+A2a` sob `"emu"` propunha em média `24.4` PRBs ativos para URLLC nas `1197` janelas `=3` ativas (`viol_u = 43.27%`), `A1+A2a+A_scaler` modera URLLC para `18.5` PRBs (`a_exec_act`: `18.38 -> 13.45` PRBs) e amplia `sum_alpha` menos negativa (`-1.48 -> -0.88` em URLLC; `-1.65 -> -1.15` em eMBB).
   - Com isso, eMBB recupera `confident` (`996 -> 1361`), `raw>53` em `=3` ativas **recua** (`86.30% -> 82.62%`), as violações eMBB caem **`29.70% -> 22.26%` (`-7.44 pp`)**, as violações URLLC caem **`43.27% -> 32.78%` (`-10.49 pp`)**, `miss` URLLC cai `8.42% -> 5.50%`, `STRICT_ALL3_R19` sobe **`+13.53 pp` (`41.77% -> 55.30%`)** e `STRICT_DEMAND_R19` sobe `+8.99 pp` (`57.76% -> 66.75%`).

**Leitura conjunta de $\lambda \in \{2, 5, 6\}$:** `A1+A2a` mantinha `84.20%` em $\lambda=5$ às custas de um **bug de indexação (`"emu"`)** que saturava artificialmente `n_u` (`14.3%–36.9%`) e comprimía `queue_u` pela metade (`0.5×`), mascarando a transição para `knn`/`softmax` em $\lambda=5$ e prejudicando fortemente $\lambda=6$ (`41.77%` vs `55.30%`) e $\lambda=2$ (`16.13%` vs `12.90%` URLLC). A média simples de `STRICT_ALL3_R19` em `{5, 6}` é inclusive ligeiramente superior com a correção (`63.51%` em `A1+A2a+A_scaler` vs `62.99%` em `A1+A2a`), com código fisicamente consistente.

---

## 4. Reavaliação de Status de $A_4$ (*Mean-Field*) e Análise sobre Reteste sobre `A_scaler`

1. **Status corrigido de $A_4$ (`--mean_field`):** Concluída como experimento exploratório sob o indexador antigo `"emu"`, com desempenho **dependente da configuração** — **não definitivamente rejeitada**:
   - **Sobre `A0` (`R12=0.05`, guarda OFF):** melhorou $\lambda=6$ (`STRICT_ALL3_R19`: `35.24% -> 49.96%`, `+14.71 pp`; `STRICT_DEMAND_R19`: `47.62% -> 55.94%`), mas apresentou regressões em $\lambda=2$ (eMBB `1.22% -> 24.39%`), $\lambda=5$ (mMTC `2.78% -> 11.42%`) e manteve `1293` pacotes URLLC sem recebimento em $\lambda=6$ (`p95=56.10 ms`).
   - **Sobre `A1+A2a` (`R12=0`, guarda ON):** reduziu `STRICT_ALL3_R19` em $\lambda=5$ (`84.20% -> 73.76%`, `-10.45 pp`), dobrou violações URLLC em $\lambda=5$ (`9.20% -> 20.09%`) e elevou projeções em `=3` ativas (`43.96% -> 83.74%` em $\lambda=5$; `86.63% -> 95.07%` em $\lambda=6$).
2. **Por que o veredito sob `"emu"` não basta para descartar `A4` permanentemente:**
   - Todos os runs de `A4` rodaram com `URLLC` e `mMTC` trocados em `dim0..1` (`"emu"`). Acabamos de demonstrar em `A_scaler` que corrigir apenas `dim0..1` desloca `STRICT_ALL3_R19` em `-12.48 pp` ($\lambda=5$) e `+13.53 pp` ($\lambda=6$) e move a linha de base de comparação para `71.72%` ($\lambda=5$) e `55.30%` ($\lambda=6$).
   - Como o kernel Matérn combina todas as 6 dimensões do estado (`dim0..5`), testar `dim4..5` (`--mean_field`) enquanto `dim0..1` estava trocado em até `2.0×` entre URLLC e mMTC impede atribuir integralmente os efeitos observados à ausência de mérito de $\bar{s}_t$.
3. **Indicação técnica fundamentada sobre retestar `A4` sobre `A1+A2a+A_scaler` (sem executar neste passo):**
   - **Expectativa mecânica:** estruturalmente, mesmo sob `"eum"`, regressores independentes sem seleção conjunta tendem a manter lances competitivos elevados em $\sum a_i > C$ quando informados do congestionamento global $\bar{s}_t$ (vimos `A1+A2a+A_scaler` já atingir `81.58%–82.62%` de `raw>53` em `=3` ativas e `A1+A2a+A4` chegar a `94.65%` em $\lambda=6$).
   - **Valor científico do reteste pontual (`1` onda de `3` runs `A1+A2a+A_scaler+A4`, `seed=2102`, $\lambda \in \{2,5,6\}$):** **justificado apenas se autorizado explicitamente antes do congelamento final de `B0`**, pois fecha a célula fatorial `A_scaler × A4` com custo baixo (`~15 min`), responde empiricamente se `[prb_others_norm, total_ues_norm]` ajuda ou não o colapso de `knn` em $\lambda=5$ sob `"eum"` limpo, e elimina qualquer crítica de revisão sobre rejeitar o *mean-field* com base em runs contaminados por `"emu"`. **Não foi executado nesta rodada** por estrita observância do bloqueio vigente.

---

## 5. Enquadramento Metodológico do Baseline (`SQR-HET` Adaptado/Corrigido vs. SQR Publicado)

Para rigor científico na redação do artigo (evitando tanto supervalorizar falhas locais de implementação quanto reivindicar fidelidade literal indevida ao artigo original), o baseline deve ser enquadrado em **três camadas explícitas** (alinhadas ao registro [`D3_RETIFICACAO_E_PLANO_ABLACAO.md`](D3_RETIFICACAO_E_PLANO_ABLACAO.md) §4.2):

1. **Núcleo SQR Publicado (`[ARTIGO]`, Tuerxun & Nakao, IEEE CCNC 2026):** regressor quantílico online com kernel Matérn ($\nu=1{,}5$), métrica de incerteza $\sigma(x) = 1 - \max_j k(x, x_j)$, perda pinball assimétrica com penalização crítica $P_{\text{penalty}}=2$, adaptação $(\tau_t, M_t)$ e política hierárquica `A_safe` $\to$ `knn`/`ε-softmax` com margem $+\text{round}(\sigma M)$ e termo de custo $-\gamma a_i / C$ sobre **3 slices eMBB homogêneas em throughput**.
2. **Adaptação `SQR-HET` ao Cenário Multi-Serviço (`[ADAPTAÇÃO]`):** extensão para **eMBB ($\ge 3{,}5\text{ Mbps}$), URLLC ($p_{95} \le 10{,}0\text{ ms}$) e mMTC ($\text{PDR} \ge 0{,}95$)** via normalização de margens comuns ($\Lambda = 0$), exclusão causal `R19` para janelas sem demanda/fechamento, *delayed feedback* online do pool URLLC (`--urllc_delayed_update`), bootstrap compartilhado `0–100 s` (`PropDemandController`) e discretização física em $C=53$ PRBs/RBGs (`1 RBG = 1 PRB/slot DL`).
3. **Depuração de Artefatos Locais de Implementação (`[CORREÇÃO]`):** remoção da atualização virtual incondicional excessiva `R12=-0.05` (`A1`), supressão de proposta em slices comprovadamente sem demanda/pipeline (`A2a`) e correção do bug de indexação `"emu" -> "eum"` (`A_scaler`), distinguindo falhas estruturais da arquitetura desacoplada (ex.: `~82%` de excesso `raw>53` em `=3` ativas que motiva a seleção conjunta `SQR-JC`) de defeitos acidentais de codificação.

---

## 6. Recomendação Objetiva para Fechamento de $B_0$ e Lista Mínima Pré-Congelamento

### 6.1 Comparação das opções de fechamento

| Opção | Avaliação técnica | Decisão recomendada |
|---|---|---|
| **`B0 = A1+A2a` (`"emu"`)** | Mantém erro objetivo de código (`"emu"` trocando `n_{u,m}` e `queue_{u,m}` em até `2.0×`), que inflacionou `lambda=5` (`84.20%`) por saturação artificial e degradou `lambda=6` (`41.77%`) e `lambda=2` (`16.13%` URLLC). | **Não recomendar como `B0` definitivo** (preservar apenas como controle histórico). |
| **`B0 = A1+A2a+A_scaler` (`"eum"`)** | Código íntegro, sem viés de indexador; melhora `lambda=2` (`97.11% STRICT_DEMAND`, `miss` `5.59%`) e `lambda=6` (`55.30% STRICT_ALL3_R19`, `+13.53 pp`); expõe honestamente a contenção em 3 slices (`82%` projeção em `=3` ativas, `71.72%` em `lambda=5`) que o seletor conjunto precisa resolver. | **Candidato primário recomendado para `B0`.** |
| **Reteste pontual `A4` sobre `A_scaler`** | Resolve a única hipótese estrutural do artigo testada anteriormente sob `"emu"`. Custo baixo (`1` onda, `3` runs na seed `2102`). Nota: `A2b` torna-se inócua sob `virtual_alpha=0.0` (sem atualizações virtuais). | **Recomendado como verificação pontual antes do congelamento final**, se autorizado pelo pesquisador. |

### 6.2 Lista mínima de verificações antes do congelamento definitivo de $B_0$

1. **Verificação 1 — Reteste pontual de `A4` sobre `A1+A2a+A_scaler` (1 onda, `seed=2102`, $\lambda \in \{2, 5, 6\}$):** confirmar sob `"eum"` limpo se `--mean_field` melhora ou degrada a contenção em $\lambda=5$ (`82.08%` de projeção em `=3` ativas); manter `mean_field=False` salvo ganho líquido claro sem regressão de URLLC/contenção. *(Opcional: registrar sensibilidade de `M_0=3, [1,10]` (`A3a`) apenas se `A4` também não estabilizar $\lambda=5$, lembrando que `A2b` é identicamente nula sob `virtual_alpha=0.0`.)*
2. **Verificação 2 — Checagem de generalização na família de desenvolvimento já aberta (`seed=2101` ou `2103`, $\lambda \in \{5, 6\}$):** confirmar que a divergência entre $\lambda=5$ e $\lambda=6$ não decorre de uma única bifurcação de trajetória da `seed=2102` (iniciada em `w=1002`), **mantendo estritamente intocadas as seeds independentes `3101–3103`**.
3. **Verificação 3 — Prontidão de trace causal e regressão de contrato (`D3-v2`):** validar `--freeze` e `--resume` com `fix_scaler_index=True`, verificar captura de estado causal online sem vazamento pós-warmup (`policy == "SQR_HET"`), reconciliação `RX == KPM` e execução verde de `tests/run_sqr_tests.sh` e `verify_a_scaler_and_slas.py`.
