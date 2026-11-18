# Auditoria Técnica Linha a Linha — SQR Base e Baselines (ns-3 / 5G-LENA + Python RL)

**Data:** 2026-11-17  
**Auditor:** `ns3-sqr-auditor` (Especialista ns-3 / 5G-LENA e Caçador de Bugs)  
**Escopo:** Implementação C++ (`MODA/ns-3-dev/scratch/sqr/`, `contrib/nr/`, `docs/patches_nr/`) e Python (`MODA/ns-o-ran-gym/src/nsoran/sqr/`, `examples/sqr_run.py`, `examples/xapp_sqr.py`, `src/nsoran/action_controller.py`, `src/nsoran/ns_env.py`) referente ao artigo base *Tuerxun & Nakao, IEEE CCNC 2026* (`SQR γ=0`, `SQR γ=1`, `KBRL`, `equal`), além dos artefatos de campanha (`results/sqr_campaign/`, `results/sqr_e2_parity/`, `results/sqr_characterization/`).  
**Modo de operação:** Somente-leitura no código/dados do repositório; correções apenas propostas cirurgicamente (não aplicadas).

---

## Sumário Executivo

A auditoria linha a linha associada a experimentos diagnósticos em memória e inspeção direta dos logs/checkpoints da campanha de 40 runs (2 km, 30 dBm) revelou **3 bugs críticos confirmados**, **3 distorções altas de fidelidade/métrica**, **4 achados médios** e validou **10 subsistemas sãos**.

Os três achados críticos explicam mecanisticamente três anomalias fundamentais observadas nos resultados existentes:
1. **`BUG-01 [CRÍTICO]` (`rslaq-mac-scheduler.cc:1553-1556`)**: `BeforeDlSched` sobrescreve **`m_dlMcs == 0` para `MCS = 15` (16-QAM)** em todo slot DL, destruindo todas as transmissões legítimas de **`CQI = 1`** (`100%` com throughput `0,0 Mbps` na `seed_1`) e `CQI = 0`/CQI expirado (`8,4%–17,8%` de todas as alocações MAC na `seed_1`), amplificando perdas e trocas RLF (`43–65` por run).
2. **`BUG-02 [CRÍTICO]` (`coordinator.py:169-212`, `quantile_regressor.py:254-331`, `kernel.py:72-101`)**: A atualização virtual negativa incondicional (`−0,05`, **10×–50× mais forte** que $\alpha_{UP} = \eta\tau$) combinada ao kernel largo `l=1.0` e à poda assimétrica `I_j = |α_j|·k` (que expulsa pesos positivos pequenos) colapsou `f_τ` para valores inteiramente negativos (**`-3,5` a `-12,4 Mbps`** em `sqr-g0/seed_1`), mantendo **`A_safe = ∅` em 100% das 179.850 decisões** da campanha de 40 runs e **impedindo que `γ=0` vs `γ=1` executasse sequer uma única vez**. O mesmo colapso ocorre também em simulações sintéticas 100% viáveis.
3. **`BUG-03 [CRÍTICO]` (`sqr-sim.cc:1680-1802`, `sqr_controller.py:300-301`, `action_controller.py:34-41`)**: A dupla avaliação de `predict()` + `uncertainty()` em `score_candidates` e a reescrita/leitura $O(T^2)$ do CSV levaram `decision_time_ms` até `97,3 ms`, estourando silenciosamente o timeout de `50 ms` do `sem_timedwait` IPC e gerando **dessincronização causal permanente de 1 passo em 18 de 20 runs de SQR** (até `2.103` passos divergentes por run). Toda diferença observada entre `sqr-g0` e `sqr-g1` na campanha de 40 runs foi artefato desse race condition de timeout.

Entre os achados de severidade alta estão: a triplicação aritmética de `rb_cell_total_idle_est_mean` (`~75–77` de `53` PRBs) e mistura de escalas por slot vs por janela na Fig. 4 (`BUG-04`), a dessincronia `C=52→53` no Passo 1 do `sqr_run.py` vs `xapp_sqr.py` (`BUG-05`), e o mascaramento de UEs `CQI = 0` (`snrDb = -999`, `23,8%–35,1%` das linhas KPM) para `15 dB`/mediana positiva via imputação (`BUG-06`).

---

## Tabela Resumo dos Achados

| ID | Severidade | Componente | Impacto Principal |
|---|---|---|---|
| `BUG-01` | `[CRÍTICO - BUG CONFIRMADO]` | `RslaqMacScheduler::BeforeDlSched` (`C++`) | Força `MCS=15` quando `MCS=0` (`CQI=0/1`/expirado); `0/68` janelas `CQI=1` com vazão >0 |
| `BUG-02` | `[CRÍTICO - BUG CONFIRMADO]` | `KernelQuantileRegressor` + `apply_virtual_updates` (`Py`) | `f_τ` colapsa para `-3,5..-12,4 Mbps`; `confident = 0,00%`; `γ` nunca atuou na campanha |
| `BUG-03` | `[CRÍTICO - BUG CONFIRMADO]` | `sem_timedwait(50ms)` + `score_candidates` + CSV (`C++/Py`) | Desync causal permanente em `18/20` runs SQR; gera diferença espúria `g0` vs `g1` |
| `BUG-04` | `[ALTO - DISTORÇÃO DE FIDELIDADE/MÉTRICA]` | `campaign_analysis.py` + `sqr_run.py` (`Py`) | Triplica `rb_cell_total_idle_est_mean` (`76,7 > 53`); mistura RBG·slots/janela com PRBs/slot na Fig. 4 |
| `BUG-05` | `[ALTO - DISTORÇÃO DE FIDELIDADE/MÉTRICA]` | `sqr_run.py` + `xapp_sqr.py` (`Py`) | Passo 1 usa `C=52` no `sqr_run.py` vs `C=53` no `xapp_sqr.py`; `learned`/`virtual_updates` errados no E2 |
| `BUG-06` | `[ALTO - DISTORÇÃO DE FIDELIDADE/MÉTRICA]` | `sqr_kpis.py` + `sqr-sim.cc` (`C++/Py`) | `snrDb=-999` (`24–35%` KPMs) imputado como mediana positiva/`15 dB`, ocultando outage `CQI=0` |
| `BUG-07` | `[MÉDIO]` | `sqr-sim.cc` ciclo de vida UE (`C++`) | `clientApp` dura min `0,6 s` > `UeDeparture` em sessões `<0,6 s`; KPM não checa `ue.attached` |
| `BUG-08` | `[MÉDIO]` | `SqrMacScheduler::AssignDLRBG` (`C++`) | Ordena UEs 1× por slot (quase-RR intra-slot sob PF); `dl_slots` omite slots vazios |
| `BUG-09` | `[MÉDIO]` | `sqr-sim.cc` canal 3GPP (`C++`) | `UpdatePeriod=0` congela estado LOS/NLOS e LSPs (Doppler continua ativo) |
| `BUG-10` | `[MÉDIO]` | `docs/patches_nr/` vs `contrib/nr/` + `sqr_run.py` (`Docs/Py`) | Patches 1–3 divergem do v5.2-dev; coluna `virtual_updates` mistura cortes de $\Pi$ com updates |

---

## Achados Detalhados

### `BUG-01` `[CRÍTICO - BUG CONFIRMADO]` — Sobrescrita forçada `m_dlMcs == 0 -> 15` (16-QAM) em `BeforeDlSched`

- **Localização exata:**
  - `MODA/ns-3-dev/scratch/sqr/rslaq-mac-scheduler.cc:1547-1556` (`RslaqMacScheduler::BeforeDlSched`, herdada diretamente por `SqrMacScheduler` e invocada em `MODA/ns-3-dev/scratch/sqr/sqr-mac-scheduler.cc:504`)
  - Interação direta com `MODA/ns-3-dev/contrib/nr/model/nr-mac-scheduler-cqi-management.cc:137-139, 210-215` (`DlCqiReported` e `RefreshDlCqiMaps`), `MODA/ns-3-dev/contrib/nr/model/nr-amc.cc:79-99, 247-250` e `MODA/ns-3-dev/contrib/nr/model/nr-lte-mi-error-model.cc:872-892`.

- **Mecanismo exato:**
  No modelo de erro padrão (`NrLteMiErrorModel`), as eficiências espectrais são `SpectralEfficiencyForCqi[0] = 0.0`, `[1] = 0.15`, `[2] = 0.23` e `SpectralEfficiencyForMcs[0] = 0.15`, `[1] = 0.19`, `[2] = 0.23`. Consequentemente, `NrAmc::GetMcsFromCqi(wbCqi)` retorna **`mcs = 0`** tanto para **`CQI = 0`** (abaixo do limiar de `CQI 1`, `SNR < ~−5 dB`) quanto para **`CQI = 1`** (canal legítimo em QPSK `0,15 bits/RE`, `SNR ≈ −2,18 dB`). Além disso, quando o timer de CQI expira, `RefreshDlCqiMaps` define `m_wbCqi = 1` e `m_dlMcs = GetStartMcsDl() = 0`.
  Em todo slot DL, `BeforeDlSched` executa:
  ```cpp
  if (ueInfo->m_dlMcs == 0)
  {
      ueInfo->m_dlMcs = 15;
  }
  ```
  Em vez de inicializar apenas um UE sem CQI prévio, o código substitui **todo `MCS = 0` válido (`CQI = 1`) ou `CQI = 0` / expirado** por **`MCS = 15`** (`16-QAM`, `SE = 2,16 bits/RE` — **14,4× maior** que `MCS = 0`, exigindo `SNR ~ 9–10 dB`). O comportamento correto no 5G-LENA (`NrMacSchedulerOfdmaPF::BeforeDlSched`) não altera `ueInfo->m_dlMcs`.

- **Impacto concreto nos resultados já obtidos:**
  Inspeção direta de `ue_detail.csv` e `sqr_kpm_history.csv` na `seed_1` da campanha de 40 runs (`results/sqr_campaign/<metodo>/seed_1/.../seed=1_run=1/`):
  - **`mcs == 0` teve 0 ocorrências** em todos os métodos; todos os demais MCSs observados são pares (`2, 4, ..., 28`), sendo `15` o único ímpar.
  - **`mcs == 15` respondeu por:** `27.378 / 153.418` (**17,8%**) das alocações MAC em `equal`, `11.266 / 133.452` (**8,4%**) em `kbrl`, `12.942 / 96.780` (**13,4%**) em `sqr-g0` e `12.954 / 96.302` (**13,5%**) em `sqr-g1`.
  - **100% (68 / 68) das janelas com `CQI = 1` (`snrDb = -2.18 dB`)** na `seed_1` (`23` em `equal`, `19` em `kbrl`, `13` em `sqr-g0`, `13` em `sqr-g1`) tiveram **`windowThroughputMbps = 0.0000 Mbps`** (BLER ~100% ao transmitir `16-QAM` num canal QPSK-mínimo), impedindo que UEs na faixa de transição de 1–2 km recuperassem o enlace, desperdiçando RBGs agendados e alimentando o churn de RLF (`43–65` eventos `UE_RNTI_CHANGED` por run em `seed_1`, elevando para `72–82` RNTIs distintos para apenas `23` UEs).

- **Proposta de correção cirúrgica:**
  Em `MODA/ns-3-dev/scratch/sqr/rslaq-mac-scheduler.cc:1553-1556`, remover o bloco `if (ueInfo->m_dlMcs == 0) { ueInfo->m_dlMcs = 15; }`. Caso se deseje um MCS inicial apenas antes do primeiro CQI reportado, condicionar estritamente ao estado sem CQI (ou configurar atributo `StartMcsDl` via `NrMacSchedulerNs3` apenas quando `m_dlCqi.m_wbCqi == 0` pré-CQI e nunca sobrescrever `CQI = 1`).

---

### `BUG-02` `[CRÍTICO - BUG CONFIRMADO]` — Colapso negativo de `f_τ` por atualização virtual (`−0,05`), poda assimétrica `I_j` e ausência de decaimento NORMA (`A_safe ≡ ∅`, `γ` inativo)

- **Localização exata:**
  - `MODA/ns-o-ran-gym/src/nsoran/sqr/coordinator.py:169-212` (`apply_virtual_updates`)
  - `MODA/ns-o-ran-gym/src/nsoran/sqr/sqr_controller.py:420-438` (`insert_virtual_support`)
  - `MODA/ns-o-ran-gym/src/nsoran/sqr/quantile_regressor.py:206-214, 254-331` (`_prune_index`, `update`, `insert_support_vector`)
  - `MODA/ns-o-ran-gym/src/nsoran/sqr/kernel.py:72-101` (`MaternKernel`, `length_scale=1.0` isotrópico sobre $\mathbb{R}^7$).

- **Mecanismo exato:**
  Três mecanismos interligados destroem a calibração de $f_\tau(z, a)$:
  1. **Atualização virtual desproporcional e incondicional:** Quando $\sum a_i > C$, `apply_virtual_updates` chama `insert_support_vector(..., -0.05)` para cada slice cortada. Enquanto uma observação real bem-sucedida ($y_t > f_\tau$) acrescenta apenas $\alpha_{UP} = \eta\tau = +0.005$ (para $\tau=0.05$) ou **$+0.001$** (quando $\tau$ adapta para `0.01`), cada atualização virtual insere **$-0.05$** — **10× a 50× mais forte** — independentemente do valor corrente de $f_\tau$ (mesmo quando $f_\tau$ já é negativo!).
  2. **Vazamento espacial global do kernel Matérn ($\nu=1.5, \ell=1.0$):** Em $[0,1]^7$, variar $a/C$ do máximo ($50/53$) para $0$ gera distância $d = 50/53 = 0.943$, logo $k = (1 + \sqrt{3}d)e^{-\sqrt{3}d} = 0.514$. Assim, um único vetor virtual $\alpha_{\text{virt}} = -0.05$ em $a=50$ deprime $f_\tau(z, 0)$ em `−0.0257 Mbps` e $f_\tau(z, 25)$ em `−0.0401 Mbps`, arrastando **todo** o espaço de ações para baixo.
  3. **Poda assimétrica `I_j = |α_j|·k(x_new, x_j)` sem reprojeção nem decaimento NORMA:** Em $\tau \in [0.01, 0.05]$, $|\alpha_{OP}| \in [0.095, 0.099]$, $|\alpha_{CF}| \in [0.190, 0.198]$ e $|\alpha_{\text{virt}}| = 0.05$ são **10× a 198× maiores** que $\alpha_{UP} \in [0.001, 0.005]$. Ao atingir $B = 2000$ (~200 s do episódio de 600 s), `argmin_j |α_j|·k` expulsa sistematicamente vetores positivos enquanto preserva vetores negativos e virtuais, sem o encolhimento $(1 - \eta\lambda)\alpha_j$ do algoritmo NORMA original [ref. 13].

  ```
  Ciclo vicioso: f_tau < Lambda -> A_safe = vazio -> modo k-NN/fallback + M=50
       ^                                                    |
       |                                             sum(a_prop) > C
       |                                                    v
  f_tau cai (-3.5 a -12.4 Mbps) <--- insere alpha_virt = -0.05 (50x alpha_UP)
  ```

- **Impacto concreto nos resultados já obtidos:**
  - **Checkpoint real `sqr-g0/seed_1` (`sqr_checkpoint.pkl`):**
    - *Slice 0* (`Λ=0.5`, `n_sv=2000`, `n_ok=380`, `437` janelas reais $\ge \Lambda$): `virt(-0.05)=338`, `sum_alpha=-13.4390`, `f_tau` nos pontos de suporte entre **`-9.7200` e `-3.5726 Mbps`** (máximo sobre grade `a=0..53` em 200 estados: **`-3.6618 Mbps`**).
    - *Slice 1* (`Λ=1.0`, `n_sv=1278 < 2000` — **antes mesmo da poda iniciar!**): `virt(-0.05)=367`, `sum_alpha=-15.7390`, `f_tau` entre **`-12.4375` e `-3.5376 Mbps`** (máx na grade: **`-5.5848 Mbps`**).
    - *Slice 2* (`Λ=1.5`, `n_sv=2000`, `n_ok=551`, `769` janelas reais $\ge \Lambda$): `virt(-0.05)=400`, `sum_alpha=-17.1240`, `f_tau` entre **`-10.4845` e `-5.5091 Mbps`** (máx na grade: **`-6.1408 Mbps`**).
  - **Campanha de 40 runs (`modes_by_method.csv`):** `mode == "confident"` teve **0 ocorrências em 179.850 decisões** (0,00%) tanto em `sqr-g0` quanto em `sqr-g1`. Portanto, **o mecanismo `f_tau(z,a) - γ·a/C` nunca foi ativado nenhuma única vez** na campanha; `sqr-g0` e `sqr-g1` foram funcionalmente idênticos e consumiram apenas ~28 PRBs/slot de 53 (presos em repetições históricas de `k-NN`).
  - **Comprovação diagnóstica em regimes 100% viáveis (Python em memória):**
    1. *Regressão estacionária linear* ($y = 2.0\,a + \mathcal{N}(0, 0.2^2)$, $\tau=0.05$, $P_{\text{penalty}}=1.0$, **sem** atualizações virtuais): em $t=1999$ (`B=2000`), `err = [-0.064, -0.007, -0.211]` para $a \in \{0.1, 0.5, 0.9\}$; após iniciar a poda ($t=6000$), `sum_alpha` caiu para `-0.300` e o erro degradou para **`[-0.558, -0.223, -0.808]`**. Com $B=200, P_{\text{penalty}}=2.0$ em $t=3000$, erro médio chegou a **`-0.878 Mbps`**.
    2. *Loop multi-slice viável de 500 passos* ($y = 0.15\,a$, onde 18 PRBs dão `2.70 Mbps` > todos SLAs `[0.5, 1.0, 1.5]`): com `virtual_alpha = 0.05`, houve **`0 / 1500` decisões `confident`** (`max_f = [0.167, 0.301, 0.327]`); desligando para `virtual_alpha = 0.0`, subiu para **`196 / 1500` decisões `confident`** (`max_f = [0.609, 0.881, 0.995]`).

- **Proposta de correção cirúrgica:**
  1. Em `KernelQuantileRegressor.update`: aplicar regularização/decaimento NORMA $\alpha_j \leftarrow (1 - \eta\lambda)\alpha_j$ a cada passo (com $\lambda > 0$ pequeno, ex. $10^{-3}\text{–}10^{-2}$), removendo vetores evanescentes/antigos (ou FIFO com decaimento) em vez de expurgar seletivamente `alpha > 0` via `|α_j|·k`.
  2. Em `coordinator.py` / `sqr_controller.py`: restringir a atualização virtual negativa a casos onde $f_\tau(z, a_{\text{prop}}) \ge \text{piso}$ (ex.: $0$ ou $\Lambda$) e/ou usar escala compatível com $\eta\tau$ (não $-0.05$ fixo quando $\tau \approx 0.01$), além de calibrar `length_scale` menor (ex. `0.2–0.3`) na coordenada `a/C` para localizar a penalidade em torno de $a_{\text{prop}}$ sem destruir $a$ baixos.
  3. Adicionar termo de viés/offset inicial (ou centralização em $\Lambda$ / média exponencial) para que $f_\tau$ nunca permaneça travado em valores negativos espúrios.

---

### `BUG-03` `[CRÍTICO - BUG CONFIRMADO]` — Timeout silencioso de 50 ms no `sem_timedwait` IPC e dessincronização permanente de fase em 18/20 runs SQR

- **Localização exata:**
  - `MODA/ns-3-dev/scratch/sqr/sqr-sim.cc:1680-1723, 1800-1802` (`sem_timedwait` de `50 ms`, leitura completa de `rslaq_actions_for_ns3.csv` e timeout sem contador)
  - `MODA/ns-o-ran-gym/src/nsoran/sqr/sqr_controller.py:300-301` (`score_candidates` duplamente custoso)
  - `MODA/ns-o-ran-gym/src/nsoran/action_controller.py:34-41` (append acumulativo).

- **Mecanismo exato:**
  1. `score_candidates` (`sqr_controller.py:300-301`) invoca `self.regressor.predict(X)` e `self.regressor.uncertainty(X)` separadamente, recalculando **duas vezes** por slice a matriz de distâncias e kernel `(54 × 2000 × 7)`. Em paralelo, `ActionController.create_control_action` anexa 3 linhas por passo em `rslaq_actions_for_ns3.csv` e `sqr-sim.cc:1685-1723` relê o arquivo inteiro do byte 0 a cada passo ($O(T^2)$ ao longo de 6.000 passos).
  2. Quando o ciclo completo em Python (`parse_kpm` + `observe` + `act` + escrita CSV + agendamento SO) excede **50 ms** (mesmo com `decision_time_ms` isolado de `32,7 ms` em `sqr-g1/seed_1`), `sem_timedwait(g_semControl, 50ms)` no C++ expira, mantém a cota antiga e emite apenas `NS_LOG_WARN` (não aparece em `IPC_DEGRADATION` nem no stdout padrão).
  3. Segundos/milissegundos depois, o Python publica no semáforo `semControl`. Na borda seguinte $t+100\text{ ms}$, o C++ encontra um token sobrando no semáforo e **consome imediatamente a ação atrasada do passo anterior sem aguardar o Python** — instalando uma defasagem permanente de 1 passo: o ns-3 passa a executar $a_{k-1}$ na janela $(k, k+1]$ enquanto o Python observa $y_{k+1}$ e aprende sobre $(z_k, a_k)$.

- **Impacto concreto nos resultados já obtidos:**
  - Latência medida na campanha (`summary_by_method.csv` e `step_metrics.csv`): `equal` máx `1,050 ms` (`0` > 50 ms); `kbrl` máx `10,660 ms` (`0` > 50 ms); **`sqr-g0` máx `97,276 ms` (`696` passos > 50 ms); `sqr-g1` máx `70,853 ms` (`819` passos > 50 ms)**.
  - Cruzamento exato `step_metrics.csv` (`prb_executed`) vs `sqr_prb_windows.csv` (`prb_executed_rbg`) em todos os 20 runs SQR (`task-114`): **18 de 20 runs sofreram dessincronização permanente** (`9/10` em `sqr-g0`, `9/10` em `sqr-g1`; apenas `sqr-g0/seed_1` e `sqr-g1/seed_8` escaparam):
    - `sqr-g0`: `seed_2` (1.284 mismatches, início passo 3934), `seed_3` (1.186), `seed_4` (1.391), **`seed_5` (2.103 mismatches, início passo 3000)**, `seed_6` (120), `seed_7` (1.239), `seed_8` (74), `seed_9` (1.248), `seed_10` (589).
    - `sqr-g1`: `seed_1` (154 mismatches, início passo 5724 c/ `decision_time_ms=32,71 ms`), `seed_2` (1.348), `seed_3` (1.312), `seed_4` (1.141), **`seed_5` (1.970)**, `seed_6` (22), `seed_7` (1.311), `seed_9` (810), `seed_10` (458).
  - Como `confident` foi `0%` nos dois variantes, **toda diferença numérica entre `sqr-g0` e `sqr-g1` nos relatórios da campanha de 40 runs decorreu exclusivamente dessa corrida de timeout IPC**.

- **Proposta de correção cirúrgica:**
  1. Criar `KernelQuantileRegressor.predict_and_uncertainty(X)` que calcula `K = kernel(X, sx)` **uma única vez**, reduzindo pela metade o custo de `score_candidates`.
  2. Tornar o timeout `sem_timedwait` configurável via CLI (ex. infinito/`5000 ms` para campanhas offline determinísticas vs `50 ms` estrito para tempo real), drenar/validar `timestamp` em `rslaq_actions_for_ns3.csv` caso ocorra timeout (para não consumir token atrasado), e contabilizar todo timeout em `[SqrSim] IPC_DEGRADATION`.
  3. Trocar append/leitura total de `rslaq_actions_for_ns3.csv` por escrita do comando corrente (`trunc` + `rename` atômico) ou leitura a partir do último offset.

---

### `BUG-04` `[ALTO - DISTORÇÃO DE FIDELIDADE/MÉTRICA]` — Triplicação de `rb_cell_total_idle_est_mean` (`75–77` de `53`) e mistura de unidades na Fig. 4

- **Localização exata:**
  - `MODA/ns-o-ran-gym/examples/sqr_run.py:1112`
  - `MODA/ns-o-ran-gym/src/nsoran/sqr/campaign_analysis.py:76-81, 210-211, 260-274, 433-470`.

- **Mecanismo exato:**
  1. `sqr_run.py:1112` escreve o valor **da célula** `total_prbs - int(np.sum(executed))` repetido nas 3 linhas de slice de cada passo na coluna `prb_idle_est`. Em `campaign_analysis.py:271-274`, o loop sobre `RB_CATEGORIES` executa `df.groupby("step")[col].sum(min_count=1)` para todas as 4 colunas — **somando 3 vezes** a ociosidade da célula.
  2. A coluna `prb_scheduled` em `step_metrics.csv` (copiada de `sqr_prb_windows.csv`) contém RBGs acumulados **em toda a janela de 100 ms** (~46–58 slots DL com demanda, unidade `RBG·slots/janela`), enquanto `prb_requested`, `prb_executed` e `prb_idle_est` estão em **PRBs por slot** (`0..53`). `fig4_prb_distribution` plota as 4 categorias no mesmo eixo y (`PRBs per decision`).

- **Impacto concreto nos resultados já obtidos (`results/sqr_campaign/tables/summary_by_method.csv`):**
  - `sqr-g0`: `rb_cell_total_executed_mean = 28,0138`, mas **`rb_cell_total_idle_est_mean = 74,9585`** ($= 3 \times (53 - 28,0138)$, impossível > 53!).
  - `sqr-g1`: `rb_cell_total_executed_mean = 27,4177`, mas **`rb_cell_total_idle_est_mean = 76,7468`** ($= 3 \times (53 - 27,4177)$).
  - `equal`: **`rb_cell_total_idle_est_mean = 0,0005004`** ($= 3 \times 1/5995$).
  - `rb_scheduled_mean` reportado como `326,77` (`equal`), `300,58` (`kbrl`), `210,32` (`sqr-g0`), `205,32` (`sqr-g1`) ao lado de `rb_executed_mean` `17,67 / 17,65 / 9,34 / 9,14` no mesmo boxplot da Fig. 4.

- **Proposta de correção cirúrgica:**
  Em `campaign_analysis.py:271-274`, usar `.first()` quando `col == "prb_idle_est"` (em vez de `.sum()`); para `prb_scheduled`, dividir pelo `dl_slots` da janela correspondente ou plotar em eixo/painel separado com rótulo explícito `RBG·slots / janela`.

---

### `BUG-05` `[ALTO - DISTORÇÃO DE FIDELIDADE/MÉTRICA]` — Dessincronia de `C` (`52 -> 53`) no Passo 1 em `sqr_run.py` vs `xapp_sqr.py` e contabilidade de `learned`/`virtual_updates` no E2

- **Localização exata:**
  - `MODA/ns-o-ran-gym/examples/sqr_run.py:615-616, 974-979, 1006-1021`
  - `MODA/ns-o-ran-gym/examples/xapp_sqr.py:518-538, 558-568, 658-660, 665`.

- **Mecanismo exato:**
  1. Em `sqr_run.py:615`, `env.reset()` bloqueia na 1ª borda (`100 ms`), lê `sqr_prb_windows.csv` e atualiza `env.total_prbs` de `52` para `53`. Entretanto, `controllers` só têm `n_prbs` atualizado em `1006-1021` — **após** `act()` e `env.step()` do Passo 1. Assim, o Passo 1 inteiro decide sobre `C = 52` e é projetado/convertido pelo ambiente sobre `C = 53`.
  2. Em `xapp_sqr.py:558-567`, a sincronização ocorre **antes** de `act()` no Passo 1 (`C = 53`), criando divergência estrutural entre IPC e E2 para qualquer política dependente de `C` no Passo 1 (`equal`, `kbrl`, etc.; só não apareceu na paridade SQR porque SQR em memória vazia dá `raw=0, margin=10`).
  3. Em `xapp_sqr.py:658-660`, as colunas `learned` (índice 23) e `virtual_updates` (índice 24) recebem estaticamente `int(not args.freeze)` e `0`, não integrando `DEFERRED_COLS = {2, 17, 18, 19, 20, 21, 26}` em `flush_pending()`. Assim, janelas sem UEs recebem `learned=1` e cortes da projeção $\Pi$ nunca atualizam `virtual_updates`. Além disso, `xapp_decisions.csv` (`665`) registra apenas `decisions[0].mode`.

- **Impacto concreto nos resultados já obtidos:**
  Comprovado em `results/sqr_campaign/equal/seed_1/step_metrics.csv`: no Passo 1, `equal` pediu `[18, 17, 17]` (`soma = 52`) sobre célula de `53` PRBs (`prb_idle_est = 1`), passando para `[18, 18, 17]` somente no Passo 2 (afetando todas as 10 seeds: `rb_executed_mean_slice1 = 17,999833` e `rb_cell_total_executed_mean = 52,999833`).

- **Proposta de correção cirúrgica:**
  Em `sqr_run.py`, sincronizar `controllers` com `env.total_prbs` logo após `env.reset()` (linha 616); em `xapp_sqr.py`, incluir as colunas 23–24 em `DEFERRED_COLS` e preencher no `flush_pending()` conforme o resultado real de `observe`/`apply_virtual_updates`, logando os modos das 3 slices em `xapp_decisions.csv`.

---

### `BUG-06` `[ALTO - DISTORÇÃO DE FIDELIDADE/MÉTRICA]` — Imputação de `snrDb = -999,0` para mediana positiva/`15 dB` oculta UEs fora de alcance (`CQI = 0`) no estado `z_t`

- **Localização exata:**
  - `MODA/ns-3-dev/scratch/sqr/rslaq-mac-scheduler.cc:1066-1079` e `sqr-mac-scheduler.cc:423-436`
  - `MODA/ns-3-dev/scratch/sqr/sqr-sim.cc:1484-1498`
  - `MODA/ns-o-ran-gym/src/nsoran/sqr/sqr_kpis.py:345-372` (`_impute_snr`).

- **Mecanismo exato:**
  Quando `wbCqi == 0` (UE já conectado porém com SINR abaixo de CQI 1 — "out of range") ou pós-RLF com RNTI novo antes do 1º relatório CQI, o scheduler exporta a sentinela `-999,0 dB`. Como `SqrKpmParser._impute_snr` aplica `carried -> previous_cell_median -> 15,0 dB` indistintamente sobre `-999,0`, um UE em outage profundo (`CQI = 0`) ou que nunca alcançou CQI válido (`UE 22` em `equal/seed_1`: 10/10 janelas em `-999`, `0 Mbps`) recebe `snr_db` positivo (mediana de UEs centrais ou `15 dB`) em `z_t[3]`, confundindo estados de outage com estados de canal bom.

- **Impacto concreto nos resultados já obtidos:**
  Na `seed_1`, **`35,1%` (`2.142/6.111`) das linhas KPM em `equal`**, **`23,8%` (`1.456/6.111`) em `kbrl`** e **`24,4%` (`1.492/6.111`) em `sqr-g0/sqr-g1`** chegaram com `snrDb = -999,0`, contaminando a feature `snr_norm` de uma fração expressiva das janelas com valores positivos imputados.

- **Proposta de correção cirúrgica:**
  Distinguir no C++ "sem relatório pós-attach" (`wbCqi == 0` após attach completo -> exportar piso negativo explícito, ex. `-10,0 dB`, ou `cqiWideband = 0`) de "pré-CQI inicial" (`-999`), mapeando `CQI = 0` para `snr_norm = 0.0` em vez de imputar para a mediana da célula.

---

### `BUG-07` `[MÉDIO]` — `clientApp->SetStopTime` >= `0,6 s` sobrevive a `UeDeparture` em sessões `< 0,6 s` e ausência de guarda `ue.attached` no KPM

- **Localização exata:** `MODA/ns-3-dev/scratch/sqr/sqr-sim.cc:1018-1019, 1033, 1063, 1081-1105, 1410-1413`.
- **Mecanismo exato:** `clientApp->SetStopTime(Seconds(std::max(departureSec - nowSec, g_attachGraceSec + 0.1)))` impõe mínimo de `0,6 s`, enquanto `UeDeparture` dispara em `departureSec - nowSec`. Para `sessionSec < 0.6 s` (~1% sob `Exp(60 s)`), `UeDeparture` remove o RNTI de `g_sliceRntis` antes do término do `OnOffApplication`, enviando tráfego sem slice (`not_in_any_slice`). Além disso, `KpmAndControlCallback` verifica apenas `tEndSec > ue.trafficStartSec`, sem testar `ue.attached` — se o RACH exceder `0,5 s` sob alta carga, emitiria linhas de KPM com `0 Mbps` pré-conexão (não ocorreu na `seed_1`, onde o atraso máximo de attach foi `0,105 s`).
- **Proposta de correção cirúrgica:** Garantir `departureSec >= nowSec + g_attachGraceSec + 0.1` ou parar `clientApp` dentro de `UeDeparture`; adicionar `if (!ue.attached) continue;` no loop KPM (e ajustar `trafficStartSec` se o attach atrasar além de `0,5 s`).

---

### `BUG-08` `[MÉDIO]` — `SqrMacScheduler::AssignDLRBG` ordena UEs apenas 1 vez por slot (vs por RBG no 5G-LENA upstream) e `dl_slots` depende de `activeDl`

- **Localização exata:** `MODA/ns-3-dev/scratch/sqr/sqr-mac-scheduler.cc:361, 534, 545-640` vs `MODA/ns-3-dev/contrib/nr/model/nr-mac-scheduler-ofdma.cc:458-463`.
- **Mecanismo exato:** O `NrMacSchedulerOfdma` upstream reordena o vetor de UEs a cada RBG após atualizar a métrica PF (`UpdateUeDlMetric`). O `SqrMacScheduler` chama `SortUeVectorByAlgorithm` **uma única vez antes do `while`** (`534`) e depois distribui 1 RBG por UE em round-robin dentro da slice (`for (auto& ue : sliceUeVec[s])`), tornando inócua intra-slot a chamada `AssignedDlResources`/`NotAssignedDlResources` (`631-639`). Adicionalmente, `m_windowDlSlots` (`361`) só incrementa quando `AssignDLRBG` é chamado (slots com `activeDl` não vazio; em `seed_1`, média `46,3–47,9` vs ~57 slots TDD nominais).
- **Proposta de correção cirúrgica:** Se desejar PF estrito intra-slice idêntico ao 5G-LENA, selecionar/reordenar o melhor UE da slice a cada RBG atribuído; registrar também `total_dl_slots_tdd` além de `active_dl_slots` no `sqr_prb_windows.csv`.

---

### `BUG-09` `[MÉDIO]` — `ThreeGppChannelModel::UpdatePeriod = 0` e `ChannelConditionModel::UpdatePeriod = 0` congelam LOS/NLOS e LSPs da sessão

- **Localização exata:** `MODA/ns-3-dev/scratch/sqr/sqr-sim.cc:2383, 2452`; `MODA/ns-3-dev/src/spectrum/model/three-gpp-channel-model.cc:2431-2445`; `MODA/ns-3-dev/src/propagation/model/channel-condition-model.cc:414`.
- **Mecanismo exato:** Com `UpdatePeriod = 0`, tanto o estado `LOS/NLOS` quanto os parâmetros de larga escala e clusters (`ThreeGppChannelParams`) são gerados uma única vez no nascimento do UE e congelados pela sessão inteira mesmo com `RandomWalk2dMobilityModel` (~1 m/s). (**Ressalva importante verificada:** o desvanecimento rápido de pequena escala **continua evoluindo** a cada transmissão via termo Doppler de `ThreeGppSpectrumPropagationLossModel::CalcBeamformingGain`, `three-gpp-spectrum-propagation-loss-model.cc:215-268`, usando `Simulator::Now()` e velocidade do UE.)
- **Proposta de correção cirúrgica:** Definir `UpdatePeriod` finito (ex. `1–5 s` no canal/condição) em novos experimentos onde transições LOS/NLOS por mobilidade sejam relevantes.

---

### `BUG-10` `[MÉDIO]` — Divergência entre `MODA/docs/patches_nr/` e `contrib/nr/` v5.2-dev + coluna `virtual_updates` em `step_metrics.csv`

- **Localização exata:** `MODA/docs/patches_nr/{README.md,nr-pdcp.cc,nr-net-device.cc,nr-rlc-um.cc}` vs `MODA/ns-3-dev/contrib/nr/model/nr-{pdcp,net-device,rlc-um,qos-rule-classifier}.cc`; `MODA/ns-o-ran-gym/examples/sqr_run.py:1124`.
- **Mecanismo exato:** Diff direto mostra que apenas o **Patch 4** (`nr-qos-rule-classifier.cc`, dedup QoS idêntico) está ativo no `contrib/nr/` atual. Os backups 1–3 em `docs/patches_nr/` são de uma base mais antiga do 5G-LENA: o v5.2-dev em `contrib/nr/` já contém correção superior no `NrRlcUm` (`nr-rlc-um.cc:750-764, 931-947`, descarte em `t-Reordering` e ressincronização de FI) e drop de IP desconhecido no `NrNetDevice`, porém não inclui as checagens `p->GetSize() < 2`/`D/C=0` (`nr-pdcp.cc`) nem `p->GetSize() >= 20` (`nr-net-device.cc`). Já em `sqr_run.py:1124`, a coluna `virtual_updates` registra `len(proj.virtual_updates)` (cortes de $\Pi$), imprimindo `17.982` no `kbrl/seed_1` embora `apply_virtual_updates` nunca rode em KBRL.
- **Proposta de correção cirúrgica:** Atualizar `docs/patches_nr/README.md` documentando exatamente o estado v5.2-dev + Patch 4; separar no CSV `proj_cuts` de `virtual_updates_applied`.

---

## Subsistemas Auditados e Confirmados Corretos (`[VERIFICADO SÃO / SEM BUG]`)

1. **Sinal da perda *pinball* e amplificação de falha crítica (`quantile_regressor.py:261-273`):** Subpredição (`y > f_before`) aplica $+\eta\tau$; sobrepredição (`y <= f_before`) aplica $\eta(\tau-1) < 0$; falha crítica (`f_before >= Λ and y < Λ`) aplica $\eta(\tau-1)P_{\text{penalty}}$ (com $P_{\text{penalty}}=2$ ativo para ambos $\gamma=0$ e $\gamma=1$). Em total conformidade com a Seção IV-A/IV-D de `tradução.md:91-96, 150`.
2. **Direção da adaptação dinâmica de $\tau$ e $M$ (`sqr_controller.py:453-484`):** Quando $v_W > \delta = 0.01$, reduz $\tau \leftarrow \max(0.01, \tau - 0.005)$ (quantil menor = mais conservador para throughput) e dobra $M \leftarrow \min(50, 2M)$; após 2 blocos completos sem violação, eleva $\tau$ e reduz $M/2$. Direção exata da Seção IV-C (`tradução.md:140`).
3. **Kernel Matérn normalizado e incerteza $\sigma$ (`kernel.py:107-167`, `quantile_regressor.py:190-200`):** Fórmulas fechadas exatas ($\nu \in \{0.5, 1.5, 2.5, \infty\}$), diferenciação direta em blocos preservando $k(x,x)=1.0$ exato e $\sigma(x) = 1 - \max_j k(x, x_j) \in [0, 1]$.
4. **Política hierárquica `A_safe`, k-NN Safety-First e `ε`-softmax (`sqr_controller.py:292-383`):** Seleção em `A_safe` (`f >= Λ`) maximiza `f - γ·a/C`, isolando `γ` apenas no termo de custo; no modo incerto, `ε=0.1` softmax sobre `f/T` e `1-ε` k-NN (`K=10`) filtrando sucessos antes de ordenar por distância euclidiana em `z`.
5. **Operador de projeção inteira $\Pi$ (`coordinator.py:120-166`):** Método de Hamilton por maiores restos fracionários com desempate determinístico por índice de slice; garante $\sum a_{\text{alloc}} \le C$ e $0 \le a_{\text{alloc}} \le a_{\text{prop}}$.
6. **Unidades RBG/PRB (`C=53`, `NumRbPerRbg=1`), `ComputeRbgQuotas`, porta `tbCoversBuffer` e cota absoluta (`sqr-mac-scheduler.cc:56-263, 473-662`, `resource_units_trace.md`):** Em 10 MHz $\mu=0$, `NumRbPerRbg = 1` (`53 RBGs = 53 PRBs`), eliminando erro de RBG parcial na borda de banda. `ComputeRbgQuotas` aplica piso + maiores restos sem renormalização; `Σa=0` suspende o serviço sem redistribuir RBGs ociosos; `tbCoversBuffer` evita padding artificial.
7. **Padrão de antena omnidirecional 15 dBi (`sqr-sim.cc:2421-2434`, `cosine-antenna-model.cc:73-75, 137-156`):** `HorizontalBeamwidth = 360.0` zera o expoente azimutal (`exponent = 0.0`), mantendo ganho de `15 dBi` uniforme no plano horizontal.
8. **Alinhamento temporal nominal IPC e paridade de quantização E2 (`ns_env.py:420-488`, `sqr_env.py:562-579`, `xapp_sqr.py:169-197`, `sqr-e2-adapter.cc:689-757`):** Sem estouro de timeout IPC, a dupla `(z_k, a_k^{\text{exec}}, y_{k+1})` alinha exatamente ação e janela governada. Verificação exaustiva de `integer_pcts_for_quotas` sobre todas as **27.720 alocações viáveis** de $C=53$ resultou em **0 falhas**.
9. **Port do baseline `KBRL` (`kbrl_controller.py`, `projectron.py`):** Crescimento matricial de `Kinv` (`(1,) -> (2,2) -> ...`), `GaussianKernelSV(gamma=1.0)`, `alfa=0.05`, `accuracy_range=(0.99, 0.999)`, `eta=0.1` e augmentação monotônica correspondem fielmente à referência (com o congelamento esperado de `security_factor` quando `over_capacity=True`, como ocorre em `5.994/5.995` passos da campanha congestionada de 2 km). Suíte unitária executada nesta auditoria (`bash MODA/ns-o-ran-gym/tests/run_sqr_tests.sh -x -q`): **`360 passed, 2 warnings in 56.94s`**.
10. **Métricas de violação `R19` e alinhamento de chegadas ao período TDD (`sqr_kpis.py:397-416`, `campaign_analysis.py:198-204, 225-256`, `sqr-sim.cc:864-879, 1475-1479`):** `windowThroughputMbps` normaliza pelo tempo ativo real em janelas parciais; janelas sem UEs são excluídas corretamente do denominador de violação; o alinhamento das chegadas a múltiplos de `7 ms` elimina a colisão `Cannot TX while RX` no `NrUePhy`.
