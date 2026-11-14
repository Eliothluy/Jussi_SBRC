# Relatório Experimental Comparativo 4-Way: `A0` × `A1` × `A2a` × `A1+A2a` (Guarda de Slice Vazia + Interação com `R12`)

- **Data:** `2026-11-14`
- **Escopo estrito:** Ablação exploratória controlada na seed de desenvolvimento **`2102`** ($\lambda \in \{2, 5, 6\}$, `300 s`, `warmup 100 s`, `--urllc_delayed_update`), avaliando isoladamente a guarda `A2a` (`--empty_slice_guard`) sobre `A0` (`R12` ativo) e sua interação com `A1` (`--virtual_alpha 0.0`).
- **Artefatos de avaliação gerados:**
  - Script comparativo: [`evaluate_a2a_4way.py`](file:///home/eliothluy/.gemini/antigravity/brain/4622041f-48e4-41e3-8c43-cf5834bb15d6/scratch/evaluate_a2a_4way.py) (importando a reconciliação auditada de [`evaluate_a1_vs_a0.py`](file:///home/eliothluy/.gemini/antigravity/brain/4622041f-48e4-41e3-8c43-cf5834bb15d6/scratch/evaluate_a1_vs_a0.py))
  - Consolidação estruturada: [`a2a_4way_summary.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/a2a_4way_summary.json) e [`a2a_4way_metrics.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_jc_decision/a2a_4way_metrics.csv)
- **Política de contorno:** `A2b`, `A3a`, `A3b`, `A4`, `A5`, `D3-v2`, `PILOT`, `MAIN` e as seeds reservadas `3101–3103` **não foram executados**. Os baselines históricos (`d1_runs_v9_delayed/`, `a1_runs_v9_va0/`, `D1v9_report.md`, `D2v2_*`, `D3_report.md`) e `contrib/nr/` permaneceram intactos.

---

## 1. Configuração dos 4 Braços, Isolamento Pré-Execução e Paridade de Bootstrap

### 1.1 Definição exata da alteração única (`--empty_slice_guard`)
Nos arquivos [`het_runtime.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/het_runtime.py), [`sqr_het.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/sqr_het.py) e [`sqr_run.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_run.py), adicionou-se a guarda `is_slice_provably_empty()` aplicada **exclusivamente** às propostas em `SqrHetController.decide()` quando uma slice satisfaz simultaneamente:
1. $n_i == 0$ (nenhum UE associado na janela KPM pré-decisão);
2. $\text{queue}_i == 0$ (zero bytes em fila RLC/PDCP);
3. $\text{tx\_packets}_i == 0$ (`dTxPackets == 0` na janela pré-decisão);
4. $\neg \text{pending\_pipeline}_i$: para URLLC ($i=1$), o pool online de atrasos `_urllc_delayed` / `urllc_pool_open` precisa estar fechado; para mMTC ($i=2$), a janela móvel `mmtc_rolling_active` precisa estar inativa.

Quando verdadeira, a proposta pós-margem (`action` / `a_after_uncertainty_margin`) e `a_raw` são fixados em `0` (mantendo exatamente a mesma sequência de chamadas PRNG de exploração por controlador, e **sem alterar** `apply_virtual_updates()`, atualização pinball, bootstrap, SLAs ou regra `R19`). Quando `--empty_slice_guard` está desligado (`False`), o comportamento é idêntico ao código anterior.

| Braço | `--virtual_alpha` (`R12`) | `--empty_slice_guard` (`A2a`) | Diretório dos runs (`het_lam{2,5,6}_s2102`) |
|---|---|---|---|
| **`A0`** *(controle)* | `0.05` (ativo) | `False` | `results/sqr_jc_decision/d1_runs_v9_delayed/` |
| **`A1`** *(R12 off)* | `0.0` (desativado) | `False` | `results/sqr_jc_decision/a1_runs_v9_va0/` |
| **`A2a`** *(guarda isolada em A0)* | `0.05` (ativo) | `True` | `results/sqr_jc_decision/a2a_runs_v9_guard/` |
| **`A1+A2a`** *(interação)* | `0.0` (desativado) | `True` | `results/sqr_jc_decision/a1_a2a_runs_v9_va0_guard/` |

### 1.2 Integridade operacional e paridade exata de bootstrap
- **Verificação pré-execução:** [`verify_a2a_guard_isolation.py`](file:///home/eliothluy/.gemini/antigravity/brain/4622041f-48e4-41e3-8c43-cf5834bb15d6/scratch/verify_a2a_guard_isolation.py) (`7/7` `[OK]`) e `bash tests/run_sqr_tests.sh -x -q` (`342 passed`).
- **Saúde dos 6 novos runs (`A2a` e `A1+A2a`, $\lambda \in \{2,5,6\}$):** todos concluíram com `.done`, `ns3_exit_code: 0`, `steps = 2995`, `kpm_missing = 0`, `total_prbs_final = 53`, último KPM periódico em `299900 ms`, e totais exatos de decisões (`848` em $\lambda=2$, `2298` em $\lambda=5$, `2455` em $\lambda=6$, sendo `564`, `1994` e `1994` decisões `SQR_HET` fechadas).
- **Paridade de bootstrap contra `A0` (`check_bootstrap_parity`):** **100% bit-idêntica** em ações propostas, margens pós-incerteza, ações executadas e margens de desfecho nas três cargas (`283==283`, `303==303`, `460==460`, `0` divergências em ações/desfechos também frente a `A1`).

---

## 2. Auditoria da Guarda de Slice Vazia e Proteção de Tráfego Pendente

| $\lambda$ | Slice | Passos vazios (`n=0,q=0`) | Suprimidos (`A2a` / `A1+A2a`) | `prot0` (`n_ues=0` protegido p/ pendência) | Prop. irrestrita média q/ vazia (`A0` / `A1` / `A2a` / `A1+A2a`) | Prop. com guarda q/ vazia (`A2a` / `A1+A2a`) | Supressões com obs. pós-decisão / falhas |
|---:|---|---:|---:|---:|---|---|---|
| 2 | eMBB (0) | 399 | 399 / 399 | **0** / **0** | `12.28 / 9.99 / 51.77 / 49.94` | `0.00 / 0.00` | `0 / 0` |
| 2 | URLLC (1) | 532 | 504 / 504 | **28** / **28** | `13.78 / 6.69 / 16.50 / 16.90` | `1.27 / 1.56` | `1 / 0` *(pós-decisão)* |
| 2 | mMTC (2) | 165 | 165 / 165 | **0** / **0** | `3.47 / 10.00 / 10.00 / 10.00` | `0.00 / 0.00` | `0 / 0` |
| 5 | eMBB (0) | 475 | 475 / 475 | **0** / **0** | `14.90 / 22.22 / 52.99 / 33.99` | `0.00 / 0.00` | `1 / 1` *(pós-decisão)* |
| 5 | URLLC (1) | 1099 | 1043 / 1043 | **56** / **56** | `13.15 / 31.34 / 10.53 / 37.31` | `0.65 / 1.12` | `2 / 0` *(pós-decisão)* |
| 5 | mMTC (2) | 41 | 41 / 41 | **0** / **0** | `5.29 / 10.00 / 10.00 / 10.00` | `0.00 / 0.00` | `0 / 0` |
| 6 | eMBB (0) | 315 | 315 / 315 | **0** / **0** | `0.44 / 6.00 / 6.00 / 6.00` | `0.00 / 0.00` | `1 / 1` *(pós-decisão)* |
| 6 | URLLC (1) | 677 | 649 / 649 | **28** / **28** | `13.21 / 26.81 / 8.32 / 28.99` | `0.43 / 1.36` | `2 / 0` *(pós-decisão)* |
| 6 | mMTC (2) | 32 | 31 / 31 | **1** / **1** | `5.34 / 27.56 / 24.97 / 41.69` | `0.19 / 0.22` | `0 / 0` |

### Verificação dos casos `suppressed_with_outcome_obs > 0` (ausência de falsos positivos)
Os poucos eventos em que uma decisão suprimida produziu observação não nula na janela seguinte (`lam=2` URLLC `1`, `lam=5` eMBB `1` e URLLC `2`, `lam=6` eMBB `1` e URLLC `2`, idênticos em `A2a` e `A1+A2a`) foram auditados diretamente em `sqr_kpm_history.csv` e `sqr_packet_delays.csv`:
- **Pré-decisão:** em **todos os 5 casos distintos**, `pre KPM rows = 0, tx = 0, q = 0`, `slice_n_ues = 0` e pool/janela móvel fechados (`241.7s`, `147.1s`, `157.7s`, `255.2s`, `122.7s`, `131.5s`, `212.8s`).
- **Pós-decisão:** o primeiro pacote/chegada Poisson ocorreu **estritamente após** o instante de decisão (`241.735s > 241.7s`, `147.116s > 147.1s`, `255.238s > 255.2s`, `122.714s > 122.7s`, `212.829s > 212.8s`), correspondendo a chegadas aleatórias imprevisíveis no intervalo `[W, W+1)` e **não** a falha de detecção de backlog ou trabalho pendente. Em contraste, a proteção de trabalho pendente atuou exatamente nos **`28 / 56 / 28`** passos URLLC com pool aberto e **`1`** passo mMTC em $\lambda=6$ (`prot0`).

---

## 3. Pesos do Regressor, Sensibilidade em Duas Ações, Calibração por $\tau_{i,t}$ e Pinball

> Nota metodológica: `Sensib. f(37)>f(0)` avalia exclusivamente dois pontos discretos ($a=37$ vs $a=0$) e **não** constitui prova de monotonia global em todo $[0, 53]$. Os modos de decisão são reportados sobre todas as janelas `SQR_HET` fechadas.

| $\lambda$ | Slice | Var. | $N_{\text{SV}}$ (`n_virt`) | $\sum \alpha$ | $M_{\text{final}}$ / $\tau_{\text{final}}$ | `Sensib. f(37)>f(0)` | $\bar{\tau}_i$ | $\hat{p}_{\text{exec}}$ | $\|\hat{p}_{\text{exec}}-\bar{\tau}_i\|$ | Pinball $\mathcal{L}_{\tau_{i,t}}$ | `q_exec` med. (`Pr>=0`) | Modos (`conf/knn/soft`) |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0 | A0 | 239 (`75`) | `-3.125` | `10.0 / 0.050` | `0.0%*` | 0.0500 | 0.0061 | 0.0439 | 0.1530 | `-0.178` (`1.2%`) | `2 / 506 / 56` |
| 2 | 0 | A1 | 164 (`0`) | `+0.103` | `20.0 / 0.045` | `98.8%` | 0.0480 | 0.0244 | **0.0237** | **0.0896** | `+0.002` (`81.1%`) | `541 / 20 / 3` |
| 2 | 0 | A2a | 168 (`4`) | `-0.193` | `20.0 / 0.045` | `98.8%` | 0.0480 | 0.0244 | **0.0237** | 0.1085 | `+0.002` (`86.6%`) | `147 / 380 / 37` |
| 2 | 0 | A1+A2a | 164 (`0`) | `+0.007` | `20.0 / 0.045` | `98.8%` | 0.0480 | 0.0244 | **0.0237** | 0.1039 | `+0.002` (`86.6%`) | `547 / 15 / 2` |
| 2 | 1 | A0 | 381 (`72`) | `-3.791` | `40.0 / 0.040` | `0.0%*` | 0.0439 | 0.0968 | 0.0529 | 0.3029 | `-0.452` (`16.1%`) | `87 / 430 / 47` |
| 2 | 1 | A1 | 309 (`0`) | `+0.005` | `40.0 / 0.040` | `77.4%` | 0.0439 | **0.0645** | **0.0206** | 0.3479 | `+0.002` (`61.3%`) | `554 / 9 / 1` |
| 2 | 1 | A2a | 313 (`4`) | `-0.391` | `40.0 / 0.040` | `90.3%` | 0.0439 | 0.0968 | 0.0529 | 0.2960 | `-0.161` (`16.1%`) | `87 / 430 / 47` |
| 2 | 1 | A1+A2a | 309 (`0`) | `-0.191` | `40.0 / 0.040` | `90.3%` | 0.0439 | 0.0968 | 0.0529 | **0.2950** | `-0.125` (`16.1%`) | `87 / 430 / 47` |
| 2 | 2 | A0 | 446 (`56`) | `-0.850` | `5.0 / 0.050` | `0.0%*` | 0.0500 | 0.0000 | 0.0500 | **0.0167** | `-0.272` (`0.0%`) | `4 / 358 / 49` (+153 fb) |
| 2 | 2 | A1 | 390 (`0`) | `+0.101` | `50.0 / 0.035` | `97.2%` | 0.0427 | 0.0205 | **0.0222** | 0.0231 | `+0.003` (`92.8%`) | `538 / 24 / 2` |
| 2 | 2 | A2a | 390 (`0`) | `+0.101` | `50.0 / 0.035` | `97.2%` | 0.0427 | 0.0205 | **0.0222** | 0.0232 | `+0.003` (`93.3%`) | `538 / 24 / 2` |
| 2 | 2 | A1+A2a | 390 (`0`) | `+0.101` | `50.0 / 0.035` | `97.2%` | 0.0427 | 0.0205 | **0.0222** | 0.0232 | `+0.003` (`93.3%`) | `538 / 24 / 2` |
| 5 | 0 | A0 | 1885 (`303`) | `-12.358` | `50.0 / 0.010` | `0.0%*` | 0.0202 | 0.0026 | 0.0176 | 0.0437 | `-2.386` (`0.3%`) | `8 / 1795 / 191` |
| 5 | 0 | A1 | 1582 (`0`) | `-1.174` | `50.0 / 0.010` | `100.0%` | 0.0202 | 0.0230 | **0.0029** | **0.0191** | `-0.248` (`14.0%`) | `1043 / 857 / 94` |
| 5 | 0 | A2a | 1741 (`159`) | `-5.103` | `50.0 / 0.010` | `8.0%` | 0.0225 | 0.0039 | 0.0185 | 0.0337 | `-0.926` (`5.7%`) | `571 / 1289 / 134` |
| 5 | 0 | **A1+A2a** | 1582 (`0`) | **`-0.382`** | **`25.0 / 0.015`** | **`100.0%`** | 0.0228 | 0.0197 | **0.0030** | 0.0206 | **`+0.001`** (`55.4%`) | **`1587 / 371 / 36`** |
| 5 | 1 | A0 | 1417 (`258`) | `-10.618` | `50.0 / 0.010` | `0.0%*` | 0.0263 | 0.0000 | 0.0263 | 0.0721 | `-2.176` (`0.0%`) | `0 / 1796 / 198` |
| 5 | 1 | A1 | 1159 (`0`) | `-0.905` | `50.0 / 0.010` | `99.6%` | 0.0218 | 0.0269 | 0.0051 | **0.0550** | `-0.213` (`14.9%`) | `592 / 1266 / 136` |
| 5 | 1 | A2a | 1268 (`109`) | `-4.565` | `50.0 / 0.010` | `4.8%` | 0.0207 | 0.0090 | 0.0117 | 0.0551 | `-1.446` (`0.3%`) | `3 / 1793 / 198` |
| 5 | 1 | **A1+A2a** | 1159 (`0`) | **`-0.603`** | `50.0 / 0.010` | **`100.0%`** | 0.0207 | 0.0213 | **`0.0006`** | 0.0595 | `-0.118` (`26.2%`) | **`1009 / 891 / 94`** |
| 5 | 2 | A0 | 2000 (`126`) | `-2.693` | `50.0 / 0.020` | `0.8%*` | 0.0318 | 0.0108 | 0.0210 | 0.0220 | `-0.514` (`0.8%`) | `19 / 1720 / 210` (+45 fb) |
| 5 | 2 | A1 | 1944 (`0`) | `+0.066` | `50.0 / 0.020` | `99.4%` | 0.0272 | 0.0144 | 0.0128 | **0.0100** | `-0.002` (`45.0%`) | `1957 / 35 / 2` |
| 5 | 2 | A2a | 2000 (`117`) | `-4.471` | `50.0 / 0.010` | `25.8%` | 0.0205 | 0.0077 | 0.0128 | 0.0182 | `-1.081` (`21.9%`) | `481 / 1355 / 158` |
| 5 | 2 | **A1+A2a** | 1944 (`0`) | **`+0.017`** | **`25.0 / 0.020`** | **`99.4%`** | 0.0275 | 0.0149 | **`0.0125`** | 0.0101 | **`+0.004`** (`81.6%`) | **`1971 / 22 / 1`** |
| 6 | 0 | A0 | 2000 (`305`) | `-12.175` | `50.0 / 0.010` | `0.0%*` | 0.0213 | 0.0036 | 0.0177 | 0.0639 | `-3.458` (`0.7%`) | `336 / 1500 / 158` |
| 6 | 0 | A1 | 1845 (`0`) | `-1.612` | `50.0 / 0.010` | `98.3%` | 0.0192 | 0.0286 | 0.0094 | 0.0166 | `-0.389` (`6.8%`) | `589 / 1272 / 133` |
| 6 | 0 | A2a | 2000 (`327`) | `-13.648` | `50.0 / 0.010` | `3.6%` | 0.0192 | 0.0036 | 0.0156 | 0.0651 | `-3.477` (`0.7%`) | `335 / 1501 / 158` |
| 6 | 0 | **A1+A2a** | 1845 (`0`) | `-1.652` | `50.0 / 0.010` | **`100.0%`** | 0.0213 | 0.0286 | **`0.0073`** | **`0.0157`** | `-0.323` (`19.2%`) | **`996 / 899 / 99`** |
| 6 | 1 | A0 | 1834 (`265`) | `-15.458` | `50.0 / 0.010` | `6.3%*` | 0.0190 | 0.0354 | 0.0164 | 0.4479 | `-4.520` (`0.0%`) | `0 / 1796 / 198` |
| 6 | 1 | A1 | 1584 (`0`) | `-1.581` | `50.0 / 0.010` | `99.9%` | 0.0189 | 0.0289 | 0.0100 | 0.0345 | `-0.581` (`3.3%`) | `47 / 1753 / 194` |
| 6 | 1 | A2a | 1799 (`226`) | `-13.900` | `50.0 / 0.010` | `15.3%` | 0.0189 | 0.0368 | 0.0179 | 0.3337 | `-4.624` (`0.5%`) | `7 / 1791 / 196` |
| 6 | 1 | **A1+A2a** | 1584 (`0`) | **`-1.481`** | `50.0 / 0.010` | **`99.9%`** | 0.0189 | 0.0266 | **`0.0078`** | **`0.0274`** | `-0.459` (`1.4%`) | `22 / 1776 / 196` |
| 6 | 2 | A0 | 2000 (`137`) | `-3.740` | `50.0 / 0.010` | `0.0%*` | 0.0289 | 0.0082 | 0.0208 | 0.0212 | `-0.579` (`1.7%`) | `35 / 1750 / 209` |
| 6 | 2 | A1 | 2000 (`0`) | **`-0.144`** | `25.0 / 0.015` | `99.9%` | 0.0235 | 0.0138 | **`0.0098`** | **`0.0080`** | `-0.003` (`40.1%`) | `1925 / 65 / 4` |
| 6 | 2 | A2a | 2000 (`133`) | `-4.120` | `50.0 / 0.010` | `18.4%` | 0.0269 | 0.0092 | 0.0177 | 0.0262 | `-0.994` (`12.9%`) | `257 / 1554 / 183` |
| 6 | 2 | A1+A2a | 2000 (`0`) | `-0.149` | `25.0 / 0.015` | `99.9%` | 0.0228 | 0.0127 | 0.0100 | 0.0092 | `-0.006` (`41.8%`) | `1926 / 64 / 4` |

*(Nota `*`: em `A0`, `f_at_a37` não estava instrumentado em `decisions.jsonl`; o valor de `A0` na coluna `Sensib. f(37)>f(0)` provém de `ckpt_f37_gt_f0_pct` no checkpoint final. Nos demais braços, é a métrica causal online `sens_f37_gt_f0_pct` sobre observações avaliáveis.)*

- **Estratos por quantil efetivo $\tau_{i,t}$:** em `A2a` (`R12` ativo), a sub-cobertura sistemática persiste quase como em `A0` (ex.: em $\lambda=5$ `S=0`, `gap` negativo entre `-0.009` e `-0.040` de $\tau=0.010$ a $0.045$; em $\lambda=6$ `S=0`, `cov = 0.000` em todos os estratos $\tau \in [0.010, 0.045]$). Já **`A1+A2a`** acompanha os estratos $\tau_{i,t}$ de perto tanto em $\lambda=5$ (`S=1` em $\tau=0.010$: `cov=0.012, gap=+0.002`; `S=0` em $\tau \ge 0.020$: `gap` entre `-0.015` e `+0.031`) quanto em $\lambda=6$ (`S=0` `abs_err=0.0073`, `S=1` `abs_err=0.0078`).

---

## 4. Projeções (Estratificadas `<3` vs `=3` Slices Ativas), PRBs Executados e Utilização

| $\lambda$ | Braço | `proj_rate` geral | `proj` em `<3` slices ativas | `proj` em `=3` slices ativas | Excesso $\sum a_{\text{prop}}-C$ (`mean / max`) | `dl1` médio q/ proj. | `a_exec` médio `(eMBB / URLLC / mMTC)` | `prb_exec_sum` médio | `util_dl%` |
|---:|---|---:|---:|---:|---|---:|---|---:|---:|
| 2 | A0 | 14.18% (`80/564`) | 14.18% (`80/564`) | — (`0/0`) | `-15.87 / 81` | 16.59 | `16.67 / 13.26 / 4.85` | 22.93 | 4.63% |
| 2 | A1 | 5.32% (`30/564`) | 5.32% (`30/564`) | — (`0/0`) | `-22.63 / 24` | 6.17 | `13.44 / 7.11 / 9.49` | 21.83 | 3.43% |
| 2 | **A2a** | **0.71%** (`4/564`) | **0.71%** (`4/564`) | — (`0/0`) | `-36.94 / 14` | 9.00 | `7.38 / 2.00 / 6.61` | 18.55 | 3.92% |
| 2 | **A1+A2a** | **0.89%** (`5/564`) | **0.89%** (`5/564`) | — (`0/0`) | `-36.86 / 37` | 17.20 | `7.14 / 2.24 / 6.61` | 17.78 | 3.81% |
| 5 | A0 | 16.00% (`319/1994`) | 21.29% (`257/1207`) | 7.88% (`62/787`) | `-18.76 / 67` | 11.67 | `16.72 / 11.86 / 3.78` | 32.36 | 27.00% |
| 5 | A1 | 66.05% (`1317/1994`) | 61.23% (`739/1207`) | 73.44% (`578/787`) | `+10.75 / 62` | 21.97 | `23.05 / 20.28 / 5.91` | 49.25 | 35.78% |
| 5 | **A2a** | **8.83%** (`176/1994`) | **4.31%** (`52/1207`) | **15.76%** (`124/787`) | `-24.88 / 64` | 14.28 | `16.84 / 4.51 / 5.50` | 26.92 | 33.59% |
| 5 | **A1+A2a** | **26.18%** (`522/1994`) | **14.58%** (`176/1207`) | **43.96%** (`346/787`) | `-11.73 / 70` | 16.27 | `22.88 / 7.94 / 6.19` | 37.02 | **44.94%** |
| 6 | A0 | 16.05% (`320/1994`) | 12.17% (`97/797`) | 18.63% (`223/1197`) | `-16.08 / 55` | 13.89 | `18.85 / 11.83 / 4.01` | 34.67 | 38.83% |
| 6 | A1 | 73.42% (`1464/1994`) | 62.86% (`501/797`) | 80.45% (`963/1197`) | `+7.43 / 71` | 16.25 | `22.86 / 18.97 / 6.67` | 48.49 | 45.18% |
| 6 | **A2a** | **16.85%** (`336/1994`) | **8.16%** (`65/797`) | 22.64% (`271/1197`) | `-17.50 / 78` | 14.33 | `22.01 / 6.85 / 4.22` | 33.11 | 44.89% |
| 6 | **A1+A2a** | 64.14% (`1279/1994`) | **30.36%** (`242/797`) | **86.63%** (`1037/1197`) | **`+3.86 / 80`** | 18.87 | `22.88 / 12.47 / 9.40` | 44.76 | **47.19%** |

**Achado estrutural crítico:** `A2a` reduz fortemente as projeções no regime `<3` slices ativas (`61.23% -> 14.58%` em $\lambda=5$ e `62.86% -> 30.36%` em $\lambda=6$ sobre `A1`), eliminando o desperdício de propostas em slices vazias (`empty_steps` zerados em eMBB e quase zerados em URLLC/mMTC). Porém, sob carga $\lambda=6$ quando **todas as 3 slices estão ativas**, `A1+A2a` ainda projeta em **`86.63%`** das janelas (`+3.86` PRBs médios acima de $C=53$, máximo `+80`), porque a escala/teto $M_{\max}=50$ (`A3a/A3b`) continua somando até `50` PRBs por slice quando há incerteza não nula.

---

## 5. Violações por Slice, `STRICT_ALL3_R19`, Decomposição `UNKNOWN`, Contabilização URLLC e Efeitos Negativos

### 5.1 Desempenho SLA sob regra estrita **R19** (`closed_online`) e reconciliação de pacotes URLLC

| $\lambda$ | Braço | Viol. eMBB | Viol. URLLC R19 | Viol. mMTC | `STRICT_ALL3_R19` (`P/F/U`, `% aval.`) | Causas `UNK` (`e / u / m / multi`) | URLLC `[100,300)` `TX / RX / não-RX` | `p95` / `p99` / `miss>10ms` |
|---:|---|---:|---:|---:|---|---|---|---:|
| 2 | A0 | **1.22%** (`2/164`) | 19.35% (`6/31`) | **0.00%** (`0/390`) | `0 / 0 / 564` | `0 / 0 / 31 / 533` | `3151 / 3146 / 5 (0.16%)` | `18.10 / 43.10 ms / 8.49%` |
| 2 | A1 | 2.44% (`4/164`) | **6.45%** (`2/31`) | 2.05% (`8/390`) | `0 / 0 / 564` | `0 / 0 / 31 / 533` | `3151 / 3146 / 5 (0.16%)` | `15.10 / 75.65 ms / 6.23%` |
| 2 | A2a | 2.44% (`4/164`) | 16.13% (`5/31`) | 2.05% (`8/390`) | `0 / 0 / 564` | `0 / 0 / 31 / 533` | `3151 / 3146 / 5 (0.16%)` | `18.10 / 43.10 ms / 8.30%` |
| 2 | A1+A2a | 2.44% (`4/164`) | 16.13% (`5/31`) | 2.05% (`8/390`) | `0 / 0 / 564` | `0 / 0 / 31 / 533` | `3151 / 3146 / 5 (0.16%)` | `18.10 / 43.10 ms / 8.42%` |
| 5 | A0 | 66.95% (`1017/1519`) | 24.35% (`217/891`) | 2.78% (`54/1944`) | `215 / 570 / 1209` (`27.39%` de `785`) | `106 / 684 / 0 / 419` | `119182 / 119158 / 24 (0.02%)` | `9.10 / 14.10 ms / 4.16%` |
| 5 | A1 | 32.65% (`496/1519`) | 15.04% (`134/891`) | **1.29%** (`25/1944`) | `577 / 208 / 1209` (`73.50%` de `785`) | `106 / 684 / 0 / 419` | `119182 / 119158 / 24 (0.02%)` | `8.10 / 14.10 ms / 2.99%` |
| 5 | A2a | 37.00% (`562/1519`) | 17.40% (`155/891`) | 8.49% (`165/1944`) | `397 / 388 / 1209` (`50.57%` de `785`) | `106 / 684 / 0 / 419` | `119182 / 119158 / 24 (0.02%)` | `9.10 / 15.10 ms / 3.84%` |
| 5 | **A1+A2a** | **`5.99%`** (`91/1519`) | **`9.20%`** (`82/891`) | **1.29%** (`25/1944`) | **`661 / 124 / 1209` (`84.20%` de `785`)** | `106 / 684 / 0 / 419` | `119182 / 119158 / 24 (0.02%)` | **`8.10 / 13.10 ms / 2.22%`** |
| 6 | A0 | 54.88% (`922/1680`) | **33.77%** (`439/1300`) | 3.26% (`64/1962`) | `418 / 768 / 808` (`35.24%` de `1186`) | `88 / 462 / 26 / 232` | `215464 / 212863 / 2601 (1.21%)` | `83.03 / 307.10 ms / 14.55%` |
| 6 | A1 | 38.99% (`655/1680`) | 46.92% (`617/1315`) | **1.27%** (`25/1962`) | **`522 / 675 / 797` (`43.61%` de `1197`)** | `88 / 451 / 30 / 228` | `215464 / 215437 / 27 (0.01%)` | `12.10 / 18.10 ms / 10.06%` |
| 6 | A2a | 43.04% (`723/1680`) | 42.18% (`550/1304`) | 3.41% (`67/1962`) | `429 / 762 / 803` (`36.02%` de `1191`) | `88 / 457 / 25 / 233` | `215464 / 212634 / 2830 (1.31%)` | `42.10 / 252.10 ms / 16.31%` |
| 6 | A1+A2a | **29.70%** (`499/1680`) | 43.27% (`569/1315`) | **1.27%** (`25/1962`) | `500 / 697 / 797` (`41.77%` de `1197`) | `88 / 451 / 30 / 228` | `215464 / 215437 / 27 (0.01%)` | **`11.10 / 15.10 ms / 8.42%`** |

- **Conservação exata de pacotes URLLC nos novos braços (`A2a` e `A1+A2a`):** `rx_kpm == rx_delays_in_win` em todas as 6 novas simulações (`3146 == 3146` em $\lambda=2$; `118967 == 118967` em $\lambda=5$; `212340 == 212340` em `A2a` $\lambda=6$ e `215143 == 215143` em `A1+A2a` $\lambda=6$), com `0` duplicatas e `cross_100s = 0 / 5 / 0`. As janelas URLLC `no_demand / closed_online / closed_late_eof / partial_rx / zero_rx` são idênticas em `A2a` ($\lambda=6$: `676 / 1304 / 2 / 5 / 7`) e `A1+A2a` ($\lambda=6$: `676 / 1315 / 2 / 1 / 0`), preservando integralmente a regra `R19`.

### 5.2 Transições entre trajetórias distintas (associações cruzadas no mesmo índice de janela — NÃO replay causal)

*(Como após o bootstrap os estados $S_t$, filas e históricos divergem em malha fechada, os números `both_ok / fail_to_ok / ok_to_fail / both_fail` abaixo descrevem apenas associações entre trajetórias distintas sob R19 estrito.)*

| Comparação | $\lambda$ | `eMBB` (`ok / f→o / o→f / ff`) | `URLLC_R19` (`ok / f→o / o→f / ff`) | `STRICT_ALL3_R19` (`ok / f→o / o→f / ff`) |
|---|---:|---|---|---|
| `A0 -> A2a` *(guarda isolada com R12)* | 5 | `378 / 579 / 124 / 438` | `598 / 138 / 76 / 79` | `150 / 247 / 65 / 323` |
| `A0 -> A2a` *(guarda isolada com R12)* | 6 | `552 / 405 / 206 / 517` | `565 / 187 / 296 / 245` | `227 / 202 / 191 / 564` |
| `A1 -> A1+A2a` *(guarda sobre va=0)* | 5 | **`992 / 436 / 31 / 60`** | `699 / 110 / 58 / 24` | **`510 / 151 / 67 / 57`** |
| `A1 -> A1+A2a` *(guarda sobre va=0)* | 6 | `862 / 319 / 163 / 336` | `478 / 268 / 220 / 349` | `288 / 212 / 234 / 463` |
| `A2a -> A1+A2a` *(desativar R12 c/ guarda)* | 5 | `917 / 511 / 40 / 51` | `686 / 123 / 50 / 32` | `351 / 310 / 46 / 78` |
| `A2a -> A1+A2a` *(desativar R12 c/ guarda)* | 6 | `793 / 388 / 164 / 335` | `500 / 245 / 254 / 305` | `253 / 247 / 176 / 515` |
| `A0 -> A1+A2a` *(efeito combinado vs A0)* | 5 | **`489 / 939 / 13 / 78`** | `635 / 174 / 39 / 43` | **`194 / 467 / 21 / 103`** |
| `A0 -> A1+A2a` *(efeito combinado vs A0)* | 6 | `636 / 545 / 122 / 377` | `571 / 175 / 290 / 264` | `255 / 245 / 163 / 523` |

### 5.3 Registro explícito dos efeitos negativos por slice
1. **`A2a` isolado em `A0` (`R12` ativo) piora mMTC em $\lambda=5$ e URLLC em $\lambda=6$:**
   - Em $\lambda=5$ `S=2` (mMTC), as violações sobem de `2.78%` para **`8.49%` (`+5.71 p.p.`)** porque `R12` continua injetando `117` vetores virtuais em mMTC e a eliminação dos `1043` passos em que URLLC pedia ~13 PRBs vazia deixa o modo `confident` sub-calibrado de mMTC assumir ações baixas sem colchão.
   - Em $\lambda=6$ `S=1` (URLLC), as violações por janela sobem de `33.77%` para **`42.18%` (`+8.41 p.p.`)**, os pacotes URLLC não recebidos em `t=300s` aumentam de `2601` (`1.21%`) para **`2830` (`1.31%`)**, e `miss > 10 ms` piora (`14.55% -> 16.31%`). Ademais, em $\lambda=2$, eMBB/mMTC variam `+1.22 / +2.05 p.p.` (`4/164` e `8/390`).
2. **`A1+A2a` vs `A1`:**
   - Em $\lambda=2$ `S=1`, URLLC sobe de `6.45%` (`2/31`) para `16.13%` (`5/31`, `N=31`, `+9.68 p.p.`), com piora pontual de pinball em `S=0` (`0.0896 -> 0.1039`) e `abs_cal_err` em `S=1` (`0.0206 -> 0.0529`). Causa observada: ao suprimir propostas espúrias de eMBB/mMTC em regime quase ocioso, URLLC muda precocemente de trajetória de suporte (`knn=430` vs `9` em `A1`).
   - Em $\lambda=5$, todas as 3 slices melhoram em violações (`eMBB: 32.65% -> 5.99%`, `URLLC: 15.04% -> 9.20%`, `mMTC: 1.29% -> 1.29%`), com diferenças mínimas de pinball (`< 0.005`).
   - Em $\lambda=6$, `A1+A2a` melhora todas as 3 slices em relação a `A1` (`eMBB: 38.99% -> 29.70%`; `URLLC: 46.92% -> 43.27%`; `mMTC: 1.27% -> 1.27%`; `p95: 12.10 -> 11.10 ms`; `miss: 10.06% -> 8.42%`), porém `STRICT_ALL3_R19` cai discretamente de `43.61%` para **`41.77%` (`-1.84 p.p.`)**, pois o ganho de eMBB e o de URLLC acontecem parcialmente em janelas disjuntas sob contenção de projeção (`86.63%` em `=3` slices ativas).
3. **`A1+A2a` vs `A0` (resíduo não resolvido em $\lambda=6$):**
   - Apesar de reduzir a retenção URLLC pós-warmup de `2601` (`1.21%`) para `27` (`0.013%`), cortar `p95/p99` agregados (`83.03/307.10 ms -> 11.10/15.10 ms`) e reduzir eMBB (`54.88% -> 29.70%`), **`URLLC` por janela em $\lambda=6$ permanece `+9.50 p.p.` pior que `A0` (`43.27%` vs `33.77%`)**. Em `A1+A2a` $\lambda=6$, `URLLC` recebe menos média executada (`12.47` vs `18.97` RBGs em `A1`) porque quando as 3 slices competem com `M_max=50`, a projeção proporcional favorece propostas infladas de eMBB (`22.88`) e mMTC (`9.40`).

---

## 6. Síntese Científica e Recomendações para a Próxima Ablação

1. **A guarda `A2a` sozinha (`A0 + guard`) não basta e não resolve `R12`:** enquanto `R12` continuar ativo, os pesos permanecem negativos, a sensibilidade em duas ações fica destruída (`3.6%–25.8%` em $\lambda \in \{5,6\}$) e URLLC piora em $\lambda=6$ (`33.77% -> 42.18%`).
2. **A combinação `A1+A2a` apresenta o melhor desempenho conjunto em $\lambda=5$ (`84.20%`), sem constituir sinergia aditiva positiva:** desabilitar `R12` + suprimir propostas vazias derruba projeções em `<3` slices (`61.23% -> 14.58%`), leva `eMBB` de `66.95%` para **`5.99%`**, `URLLC` de `24.35%` para **`9.20%`**, `STRICT_ALL3_R19` de `27.39%` para **`84.20%`**, e mantém calibração estrita (`|p_exec - tau| <= 0.0125`). *(Retificação de interação: em `STRICT_ALL3_R19` frente a `A0`, a interação aditiva calculada é **negativa** tanto em $\lambda=5$, `-12.48 p.p.` [`+56.82 p.p.` observado vs `+46.11 + 23.18 = +69.29 p.p.` aditivo], quanto em $\lambda=6$, `-2.62 p.p.` [`+6.53 p.p.` observado vs `+8.37 + 0.78 = +9.15 p.p.` aditivo]; logo, os ganhos refletem o desempenho superior da combinação em $\lambda=5$, e não sinergia aditiva positiva demonstrada.)*
3. **Diagnóstico exato do gargalo remanescente em $\lambda=6$:** mesmo em `A1+A2a`, **`86.63%` das janelas com 3 slices ativas sofrem projeção**, empurrando violações URLLC para `43.27%`. Isso aponta diretamente para os dois fatores estruturais seguintes ainda intactos na matriz [`D3_RETIFICACAO_E_PLANO_ABLACAO.md`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/docs/sqr/jc_decision/D3_RETIFICACAO_E_PLANO_ABLACAO.md):
   - **`A2b`:** verificar se a remoção da injeção `R12` em slices vazias explica o efeito de `A2a` sobre `n_virt` quando `R12` estiver ativo (ou se `A1` já domina `A2b`);
   - **`A3a` (prioritário se o orçamentador aprovar `va=0.0 + guard` como base):** reescalar `M_0 = 3` e `M in [1, 10]` mantendo a lei multiplicativa intacta, para impedir que `M_max = 50` sature propostas em `53` quando 3 slices estão ativas em $\lambda=6$.

**Nenhuma etapa subsequente (`A2b`, `A3a`, `A3b`, `A4`, `A5`) foi executada.** Aguardando autorização explícita para o próximo passo único.
