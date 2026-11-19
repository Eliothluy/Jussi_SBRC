# Auditoria de Conformidade Física e Normativa 3GPP (`v2` — Campanha Fase B)

**Data:** 2026-11-18  
**Escopo:** Auditoria matemática independente dos resultados reais da campanha `Fase B` ([`fase3_closed_loop/`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_method_summary.csv)) e da configuração do simulador [`sqr-v2-sim.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc) frente às especificações **3GPP TS 38.104, TS 38.214, TS 38.306, TS 38.331 e TR 38.901**.  
**Script de verificação reprodutível:** [`sqr_v2_3gpp_physical_audit.py`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_v2_3gpp_physical_audit.py)  
**Resumo auditado em JSON:** [`3gpp_physical_audit_summary.json`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/3gpp_physical_audit_summary.json)

---

## 1. Prova Matemática Passo-a-Passo

### 1.1 Limite Físico de Vazão DL (`3GPP TS 38.306 §4.1.2`) e Conservação de Espectro

A taxa máxima teórica de dados em *downlink* na camada física NR (sem agregação de portadoras e em operação SISO 1 camada) é dada pela **3GPP TS 38.306 §4.1.2**:

$$
\text{DataRate}_{\text{FDD}}\,[\text{Mbps}] = 10^{-6} \cdot \sum_{j=1}^{J} \left( v_{\text{layers}}^{(j)} \cdot Q_m^{(j)} \cdot f^{(j)} \cdot R_{\max} \cdot \frac{N_{\text{PRB}}^{BW(j),\mu} \cdot 12}{T_s^\mu} \cdot \left(1 - \text{OH}^{(j)}\right) \right)
$$

Parâmetros exatos de [`sqr-v2-sim.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L2728-L2745) e [`nr-amc.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-amc.cc#L67-L73):
- \(J = 1\) portadora componente, \(v_{\text{layers}} = 1\) (`1×1 SISO`), \(Q_m = 6\) bits/símbolo (`64-QAM` via `NrLteMiErrorModel`), \(f = 1{,}0\), \(R_{\max} = \frac{948}{1024} = 0{,}92578125\).
- Numerologia \(\mu = 0\) (\(\Delta f = 15\text{ kHz}\)): duração média de símbolo OFDM \(T_s^0 = \frac{10^{-3}}{14 \cdot 2^0} \approx 7{,}142857\,\mu\text{s}\).
- Overhead de controle FR1 DL normativo: \(\text{OH} = 0{,}14\).
- Quadro TDD normativo de `5 ms` (`"DL|DL|DL|S|UL|"`, `DDDSU` em `sqr-v2-sim.cc:2744`), totalizando \(5 \times 14 = 70\) símbolos OFDM por período, com fator de ciclo útil DL+S de referência \(\eta_{\text{TDD}} = \frac{54}{70} = 0{,}771429\) (`3` slots `DL` completos = `42` símbolos + `12` símbolos DL no slot `S`).

**Substituição numérica direta (`TS 38.306 §4.1.2`):**

1. **Banda normativa (`N_PRB = 52`, `TS 38.104 Table 5.3.2-1`, `10 MHz`, `15 kHz`):**
   - \(\text{DataRate}_{\text{FDD}}(52) = 10^{-6} \times 1 \times 1 \times 6 \times 1{,}0 \times \frac{948}{1024} \times (52 \times 12 \times 14 \times 1000) \times (1 - 0{,}14) = \mathbf{41{,}732145\text{ Mbps}}\)
   - \(\text{DataRate}_{\text{TDD},54/70}(52) = 41{,}732145 \times \frac{54}{70} = \mathbf{32{,}193369\text{ Mbps}}\)

2. **Grade implementada pelo 5G-LENA (`N_PRB = 53`, [`nr-helper.cc:155`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/helper/nr-helper.cc#L155) `RbOverhead = 0.04`, [`nr-phy.cc:576-584`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-phy.cc#L576-L584) \(\lfloor 10^7 \cdot 0{,}96 / 180\times 10^3 \rfloor = 53\)):**
   - \(\text{DataRate}_{\text{FDD}}(53) = 41{,}732145 \times \frac{53}{52} = \mathbf{42{,}534686\text{ Mbps}}\)
   - \(\text{DataRate}_{\text{TDD},54/70}(53) = 42{,}534686 \times \frac{54}{70} = \mathbf{32{,}812472\text{ Mbps}}\)
   - Eficiência espectral máxima por PRB em DL TDD `54/70`: \(\frac{32{,}812472}{53} = \frac{32{,}193369}{52} = \mathbf{0{,}619103\text{ Mbps/PRB}}\).

3. **Cross-check exato do teto MAC/PHY TBS no código do 5G-LENA ([`nr-amc.cc:116-163`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-amc.cc#L116-L163), [`nr-lte-mi-error-model.cc:80-82, 893-898, 1257-1278`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-lte-mi-error-model.cc#L1257-L1278)):**
   - Em MCS máximo `28` (`Qm = 6`, `McsEcrTable[28] = 0.92`, `usefulSC = 12 - 1 = 11` REs de dados/PRB/símbolo, `K_cb = 768 B`, `m_crcLen = 3 B`) e alocação PDSCH efetiva do 5G-LENA ([`nr-mac-scheduler-ns3.cc:1961-1984`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-mac-scheduler-ns3.cc#L1961-L1984): `13` símbolos PDSCH por slot `DL` e `12` símbolos PDSCH no slot `S`, totalizando \(3 \times 13 + 12 = 51\) símbolos PDSCH em `5 ms`):
   - Para **`N_PRB = 52`**: `TB_DL = 5109 B` (7 CBs) e `TB_S = 4715 B` (7 CBs) \(\implies \frac{(3\cdot 5109 + 4715)\times 8}{0{,}005\cdot 10^6} = \frac{20042 \times 8}{5000} = \mathbf{32{,}0672\text{ Mbps}}\).
   - Para **`N_PRB = 53`**: `TB_DL = 5208 B` (7 CBs) e `TB_S = 4806 B` (7 CBs) \(\implies \frac{(3\cdot 5208 + 4806)\times 8}{0{,}005\cdot 10^6} = \frac{20430 \times 8}{5000} = \mathbf{32{,}6880\text{ Mbps}}\) (contra \(\frac{20430 \times 8}{7000} = \mathbf{23{,}3486\text{ Mbps}}\) no padrão legado `7 ms`).

**Verificação automática em 100% dos dados da `Fase B`:**
- **Conservação estrita por janela ([`fase3_post_windows.jsonl`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_post_windows.jsonl)):** inspecionadas todas as **`68.040` janelas pós-warmup** (`105` episódios × `648` janelas); `0` alocações negativas, `0` violações de capacidade (`max(a_exec_0 + a_exec_1 + a_exec_2) = 53 <= 53`).
- **Médias de métodos (`14/14` linhas de [`fase3_method_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_method_summary.csv)) e episódios (`105/105` linhas de [`fase3_per_seed_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_per_seed_summary.csv)):** `rb_win_conservation_abs_err = 0.0` e `prb_util_pct <= 100.0%` em 100% dos casos.
- **Teto composto `(eMBB + URLLC + mMTC)` abaixo de `32,193 Mbps` (`N_PRB = 52`):**
  - Pela Lei de Little (\(\lambda = 3\text{ UE/min}\), duração média \(60\text{ s} \implies \mathbb{E}[N_i] = 3{,}0\text{ UEs/slice}\)), o tráfego nominal auxiliar é \(3 \times 0{,}256 = 0{,}768\text{ Mbps}\) (URLLC) + \(3 \times 0{,}0016 \cdot \text{PDR} \le 0{,}0048\text{ Mbps}\) (mMTC). Mesmo no **pior caso `p99`** do manifesto [`scaler_manifest_v2_tdd5ms.yaml`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_calibration/scaler_manifest_v2_tdd5ms.yaml) (`offered_u_p99 + offered_m_p99 = 3.811824 + 0.0182 = 3.830024 Mbps`), **100% dos métodos e 100% dos `105` episódios** ficam bem abaixo de `32,193 Mbps`:
    - Maior média por método: **`OOS/prop`** com `eMBB = 10,597 Mbps`, total nominal `11,365 Mbps`, total pior caso `p99` **`14,427 Mbps`** (folga de `17,767 Mbps`); **`MAIN/sqr_cer`** obtém `eMBB = 7,019 Mbps`, total nominal `7,791 Mbps`, total `p99` **`10,849 Mbps`**.
    - Maior episódio individual: **`MAIN/prop/s3008`** com `eMBB = 15,972 Mbps`, total nominal `16,740 Mbps`, total `p99` **`19,802 Mbps`** (\(\le 32{,}193\text{ Mbps}\)).
- **Conservação por PRB (`Mbps/PRB`) e nota estatística sobre `priority_sqr/s3008`:**
  - Como `embb_thr_mean_mbps` agrega *somente* janelas com demanda ativa (`embb_thr_mbps is not None`, [`sqr_v2_run.py:343`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/examples/sqr_v2_run.py#L343), [`het_runtime.py:166-170`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/src/nsoran/sqr_jc/het_runtime.py#L166-L170)), uma divisão ingênua pelo `mean(a_exec_0)` de *todas* as `648` janelas produz `0,6305 Mbps/PRB` apenas no episódio `MAIN/priority_sqr/s3008` (`embb_thr = 4,434 Mbps`, `mean_a0_all = 7,032 PRBs`) porque `104` das `648` janelas tiveram fatia eMBB vazia (`sla_ok_e == 1 & a_exec_0 == 0`) e **não entraram no numerador**. Ao parear o denominador às `544` janelas não-vácuas (`mean_a0_non_vac = 8,377 PRBs`), a taxa cai para **`0,5293 Mbps/PRB`** (e nas janelas com alocação `a_exec_0 > 0`, `mean = 18,449 PRBs`, resulta **`0,2403 Mbps/PRB`**). Across **105/105 episódios**, o máximo real não-vácuo é **`0,5411 Mbps/PRB`** (`MAIN/equal/s3008`, `9,740 Mbps / 18,00 PRBs`), ou seja, **`87,4%`** do teto físico `0,6191 Mbps/PRB`.

---

### 1.2 Orçamento de Enlace `3GPP TR 38.901 UMa` (`300 m` a `2000 m`)

Parâmetros físicos exatos de [`sqr-v2-sim.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L3179-L3225), [`three-gpp-propagation-loss-model.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/src/propagation/model/three-gpp-propagation-loss-model.cc#L997-L1168), [`channel-condition-model.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/src/propagation/model/channel-condition-model.cc#L692-L714), [`cosine-antenna-model.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/src/antenna/model/cosine-antenna-model.cc#L72-L156) e [`nr-ue-phy.cc`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-ue-phy.cc#L220-L233):
- \(f_c = 2{,}0\text{ GHz}\), \(h_{\text{BS}} = 25{,}0\text{ m}\), \(h_{\text{UT}} = 1{,}5\text{ m}\), \(h_E = 1{,}0\text{ m} \implies d'_{\text{BP}} = \frac{4 (25 - 1)(1{,}5 - 1)(2\times 10^9)}{3\times 10^8} = \mathbf{320{,}0\text{ m}}\).
- \(P_{\text{TX}} = 30{,}0\text{ dBm}\), ganho máximo gNB \(G_{\text{TX,max}} = 15{,}0\text{ dBi}\) (`CosineAntennaModel` com feixe horizontal `360°`, vertical `65°`, tilt `0°`: atenuação vertical de `-0,056 dB` a `300 m` \(\implies G_{\text{TX}} = 14{,}94\text{ dBi}\), e `-0,001 dB` a `2000 m` \(\implies G_{\text{TX}} = 15{,}00\text{ dBi}\)), \(G_{\text{RX}} = 0{,}0\text{ dBi}\), \(\text{NF} = 9{,}0\text{ dB}\).
- Piso de ruído térmico com \(k_B T_0 = 1{,}38064852\times 10^{-23} \times 290\text{ W/Hz}\) (`nr-ue-phy.cc:1720`):
  - Em `BW = 10 MHz`: \(N_{10\text{M}} = 10\log_{10}(k_B T_0 \cdot 10^7 \cdot 10^3) + 9{,}0 = \mathbf{-94{,}98\text{ dBm}}\);
  - Em banda útil `53 PRBs` (`9,54 MHz`): \(N_{53} = \mathbf{-95{,}18\text{ dBm}}\);
  - Em banda nominal `52 PRBs` (`9,36 MHz`): \(N_{52} = \mathbf{-95{,}26\text{ dBm}}\).
- Limiares de RLF no 5G-LENA (`TS 36.213 §4.2.1` / `TS 36.133 §7.6` em `nr-ue-phy.cc:224,231`): \(Q_{\text{out}} = \mathbf{-5{,}0\text{ dB}}\) e \(Q_{\text{in}} = \mathbf{-3{,}9\text{ dB}}\), com sombreamento *log-normal* NLOS \(\sigma_{\text{SF,NLOS}} = \mathbf{6{,}0\text{ dB}}\) (LOS \(\sigma_{\text{SF,LOS}} = 4{,}0\text{ dB}\)).

**Equações `TR 38.901 Table 7.4.1-1` avaliadas nos raios de [`fase3_radius_sweep_summary.csv`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_v2/fase3_closed_loop/fase3_radius_sweep_summary.csv):**

| Raio \(d_{\text{2D}}\) (m) | \(P_{\text{LOS}}\) | \(\text{PL}_{\text{LOS}}\) (dB) | \(\text{PL}_{\text{NLOS}}\) (dB) | \(\text{SNR}_{\text{LOS}}\) (`53 PRB`, dB) | \(\text{SNR}_{\text{NLOS}}\) (`53 PRB`, dB) | Margem NLOS vs \(Q_{\text{out}} / Q_{\text{in}}\) (dB) | \(P(\text{SNR} < Q_{\text{out}})\) Mista (%) | Sim `het_sqr_cer`: Viol (%) / URLLC `p95` (ms) | Sim `embb_article`: Viol (%) / `thr_p5` (Mbps) |
|---|---|---|---|---|---|---|---|---|---|
| **`300`** | `6,80%` | `88,55` | `116,42` | `+51,58` (`+51,37` em `10 MHz`) | **`+23,71`** (`+23,50` em `10 MHz`) | **`+28,71` / `+27,61`** | **`0,000%`** | `46,87%` / `12,38 ms` | `44,59%` / `1,873` |
| `600` | `3,01%` | `100,05` | `128,14` | `+40,12` | `+12,02` | `+17,02` / `+15,92` | `0,22%` | `50,34%` / `20,34 ms` | `52,24%` / `1,439` |
| `1000` | `1,80%` | `108,91` | `136,81` | `+31,26` | `+3,37` | `+8,37` / `+7,27` | `8,01%` | `55,59%` / `77,62 ms` | `61,68%` / `1,159` |
| `1500` | `1,20%` | `115,95` | `143,68` | `+24,22` | `-3,51` | `+1,49` / `+0,39` | `39,69%` | `80,76%` / `135,75 ms` | `68,80%` / `0,923` |
| **`2000`** | `0,90%` | `120,95` | `148,57` | `+19,23` | **`-8,39`** (`-8,59` em `10 MHz`) | **`-3,39` / `-4,49`** | **`70,74%`** (`71,38%` NLOS) | **`83,67%` / `389,15 ms`** | **`78,64%` / `0,771`** |

**Conclusão do orçamento de enlace:**
- A **`300 m`** (geometria UMa canônica próximo de \(\text{ISD}/\sqrt{3} \approx 289\text{ m}\)), a SNR mediana NLOS de **`+23,71 dB`** garante mais de `27 dB` de margem sobre \(Q_{\text{out}} = -5{,}0\text{ dB}\) e \(Q_{\text{in}} = -3{,}9\text{ dB}\) (probabilidade teórica de outage de borda `< 0,0001%`), viabilizando altas ordens MCS sem RLF de borda.
- A **`2000 m`** com apenas `30 dBm` (`1 W`), a perda NLOS sobe para **`148,57 dB`** (`d_3D = 2000,14 m > d'_BP = 320 m`), derrubando a SNR mediana de borda para **`-8,39 dB`** (`3,39 dB` *abaixo* de \(Q_{\text{out}}\) e `4,49 dB` *abaixo* de \(Q_{\text{in}}\)). Sob sombreamento NLOS \(\sigma_{\text{SF}} = 6{,}0\text{ dB}\), **`71,38%` dos UEs NLOS (`77,27%` abaixo de `Q_in`)** entram na região de RLF — explicando quantitativamente por que a violação salta para `83,67%` (`het_sqr_cer`, URLLC `p95 = 389,15 ms`) e `78,64%` (`embb_article`) no sweep real.

---

### 1.3 Piso Físico Determinístico de Latência E2E (`remoteHost -> UE`)

Somatório exato dos atrasos sem enfileiramento de tráfego na pilha simulada (`sqr-v2-sim.cc` + `contrib/nr`):

1. **Backhaul P2P + EPC:**
   - `PointToPoint` `remoteHost <-> PGW` (`100 Gbps`, [`sqr-v2-sim.cc:3257-3259`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/scratch/sqr_v2/sqr-v2-sim.cc#L3257-L3259)): **`1,000 ms`** de propagação (+ `4,8 ns` de serialização para `60 B @ 100 Gbps`);
   - `S1uLinkDelay` (`PGW <-> SGW <-> gNB`, [`nr-point-to-point-epc-helper.cc:64-66`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/helper/nr-point-to-point-epc-helper.cc#L64-L66)): **`0,000 ms`** por padrão.
2. **Atraso de alinhamento TDD (`DDDSU`, `5 ms`, \(\mu = 0\), `1 ms`/slot):**
   - Em `4/5` do período (slots `DL, DL, DL, S`), a quantização de fronteira de TTI (`1 ms`) introduz espera uniforme \(U(0, 1)\text{ ms}\) (média `0,50 ms`); em `1/5` do período (slot `UL`), soma-se o bloqueio até o próximo slot `DL`, resultando em \(U(1, 2)\text{ ms}\) (média `1,50 ms`).
   - Logo: \(W_{\text{TDD,min}} = \mathbf{0{,}00\text{ ms}}\), \(\mathbb{E}[W_{\text{TDD}}] = \frac{4}{5}(0{,}5) + \frac{1}{5}(1{,}5) = \mathbf{0{,}70\text{ ms}}\), \(p_{95}(W_{\text{TDD}}) = \mathbf{1{,}75\text{ ms}}\) e \(W_{\text{TDD,max}} = \mathbf{2{,}00\text{ ms}}\) (redução expressiva frente ao legado `7 ms DDDSUUU`, que tinha média `1,357 ms` e máximo `4,00 ms`).
3. **Pipeline MAC/PHY do 5G-LENA (`nr-phy.cc:703`, `nr-gnb-phy.cc:133-152`, `nr-ue-phy.cc:128`, `nr-mac-scheduler-ns3.cc:1913-1919`):**
   - `L1L2CtrlLatency = 2 slots` \(\implies \mathbf{2{,}00\text{ ms}}\) entre preparação/agendamento e transmissão no ar;
   - `N0Delay = 0 slots` na primeira transmissão DL;
   - Duração do transporte aéreo no slot: `13/14 = 0,929 ms` (símbolos PDSCH até fim do dado) a `1,000 ms` (slot inteiro);
   - `TbDecodeLatency = 100 us` \(\implies \mathbf{0{,}100\text{ ms}}\).
   - *Retransmissão HARQ (`10%` BLER alvo em [`nr-amc.cc:235`](file:///home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev/contrib/nr/model/nr-amc.cc#L235)):* `N1Delay = 2 slots (2 ms)` + espera pelo próximo slot `UL/S` (`0–3 ms`) + `L1L2 = 2 ms` + reenvio geram passo adicional de **`5–7 ms`**, produzindo o patamar de `11,1–13,1 ms` em `1ª retransmissão` (`N2Delay = 2 slots` governa a janela de preparação UL).

**Pisos determinísticos resultantes e comparação empírica (`14/14` linhas e `105/105` episódios):**

| Regime de Piso Determinístico E2E | Soma exata (ms) | Resultado (ms) | Observado na Campanha `Fase B` |
|---|---|---|---|
| Mínimo físico absoluto (`W_TDD=0`, `13/14` slot) | `1,00 + 0,00 + 2,00 + 0,929 + 0,10` | **`4,029 ms`** | **100% (`14/14` médias e `105/105` episódios) `>= 6,100 ms > 4,029 ms`** |
| Piso médio sem fila (`E[W_TDD]=0,70 ms`) | `1,00 + 0,70 + 2,00 + 0,929 + 0,10` | **`4,729 ms`** (`4,80 ms` slot cheio) | — |
| **Piso de pior fase TDD (`p95`/`max` sem fila, colisão `U`)** | `1,00 + 2,00 + 2,00 + 1,00 + 0,10` | **`6,100 ms`** | **`6,100 ms`** exato no menor `urllc_p95_ms_p95` (`MAIN/equal/s3009`) e **`6,120 ms`** no menor `urllc_p95_ms_mean` (`MAIN/prop/s3009`)! |
| 1ª retransmissão HARQ (`BLER=10%`) | `6,100 + [5,0 .. 7,0]` | **`11,10 – 13,10 ms`** | `equal MAIN`: `8,91 ms` (média) / `13,21 ms` (`p95`); `sqr_cer MAIN`: `11,60 ms` (média); `priority_sqr MAIN`: **`7,63 ms`** (`11,45 ms p95`). |

A coincidência exata até o milissegundo centesimal (`6,100 ms` medido vs `6,100 ms` teórico em `seed=3009`) confirma que o atraso URLLC observado não contém artefatos artificiais de relógio e reflete integralmente a física da pilha `P2P + DDDSU + L1/L2 + HARQ` do 5G-LENA.

---

## 2. Tabela Comparativa Didática (Markdown e LaTeX `booktabs`)

### 2.1 Versão Markdown

| Parâmetro / Métrica | Expectativa 3GPP / Limite Físico | Valor Obtido na Simulação (`v2`) | Conformidade Física | Referência Normativa (TS/TR 3GPP) |
|---|---|---|---|---|
| **Banda & Largura de Canal** | `10 MHz` FR1, \(\mu = 0\) (`15 kHz`); `N_PRB = 52` (`9,36 MHz`); TDD real em `2 GHz`: bandas `n34`/`n39` (`n1` é FDD) | `f_c = 2,0 GHz`, `BW = 10 MHz`, \(\mu = 0\), `N_PRB = 53` (`9,54 MHz` por `RbOverhead = 0,04`, `+1,92%`) | **Simplificação documentada** (`+1 PRB`; portadora `2,0 GHz` em banda matemática TR 38.901) | TS 38.104 §5.2, Table 5.3.2-1; TS 38.101-1 Table 5.3.2-1 |
| **Classe de Potência & Geometria UMa** | Medium-Range BS (\(\le 38\text{ dBm}\)); UMa canônico \(\text{ISD} = 500\text{ m}\) (\(R \approx 289\text{ m}\)), \(h_{\text{BS}} = 25\text{ m}\), \(h_{\text{UT}} = 1,5\text{ m}\) | `P_TX = 30 dBm` (`1 W`), `G_TX = 15 dBi`, `R = 300 m` na campanha principal (`600–2000 m` em sweep) | **Conforme (`300 m`)**; **déficit físico esperado (`2000 m`)**; `1×1` SISO e célula única (`I=0`) documentados | TS 38.104 §6.2 Table 6.2.1-1; TR 38.901 Table 7.2-1 & Table 7.3-1 |
| **Orçamento de Enlace & Limiares RLF (`300 m` vs `2000 m`)** | `d'_BP = 320 m`; NLOS a `300 m`: `SNR = +23,71 dB` (`outage = 0,000%`); NLOS a `2000 m`: `SNR = -8,39 dB < Q_out (-5 dB)` (`outage = 70,74%`) | `300 m`: `u_p95 = 12,38 ms`, `46,87%` viol.; `2000 m`: `u_p95 = 389,15 ms`, `83,67%` viol. (`het_sqr_cer`) | **Totalmente conforme** (colapso a `2 km` previsto por `TR 38.901` + `Q_out`/`Q_in`) | TR 38.901 §7.4.1–7.4.2; TS 36.213 §4.2.1; TS 36.133 §7.6 |
| **Padrão TDD & Periodicidade** | `dl-UL-TransmissionPeriodicity = 5 ms` (`DDDSU`, `20` ciclos / `100 ms`), fator DL+S de `54/70` (`77,1%`) | `"DL\|DL\|DL\|S\|UL\|"` (`5 ms`), `dl_slots_max = 80/100 ms`, `0` abortos `AssignCsiRsOffset` | **Conforme** (elimina `7 ms` não-normativo e batimento \(\pm 7\%\)) | TS 38.331 §6.3.2 (`TDD-UL-DL-Pattern`); TS 38.213 §4.1 |
| **Teto de Vazão DL & Conservação de PRB** | `TS 38.306`: `32,193 Mbps` (`52 PRB`) / `32,812 Mbps` (`53 PRB`), `0,619 Mbps/PRB`; 5G-LENA TB (`51` símb.): `32,067` / `32,688 Mbps` | Máx. método `p99`: `14,427 Mbps` (`OOS/prop`), máx. episódio `p99`: `19,802 Mbps`; máx. não-vácuo: `0,5411 Mbps/PRB`; `0` violações em `68.040` janelas | **100% conforme** (`cons_err = 0,0`, sem criação de espectro/energia) | TS 38.306 §4.1.2; TS 38.214 §5.1.3.1; 5G-LENA `NrLteMiErrorModel` |
| **Piso Físico de Latência E2E (`remoteHost -> UE`)** | Abs. mín.: `4,03 ms`; média sem fila: `4,73 ms`; pior fase TDD `DDDSU`: `6,10 ms`; 1ª retx HARQ (`N1=2 ms`): `11,1–13,1 ms` | Mínimo observado `p95`: **`6,100 ms`** (`MAIN/equal/s3009`); média mínima: `7,63 ms` (`priority_sqr MAIN`), `11,60 ms` (`sqr_cer MAIN`) | **100% conforme** (limite inferior teórico reproduzido exatamente; abstração `5QI=9` & \(\mu=0\)) | TS 38.211 §4.3.2; TS 38.214; TS 23.501 Table 5.7.4-1; TR 38.913 §7.5 |

### 2.2 Código LaTeX (`booktabs`) Pronto para o Artigo

```latex
\begin{table*}[t]
\centering
\caption{Physical and Normative Conformance Audit of the Sanitized \texttt{v2} Closed-Loop Campaign Against 3GPP Specifications.}
\label{tab:3gpp_physical_audit}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}p{3.4cm}p{4.3cm}p{4.1cm}p{2.9cm}p{2.8cm}@{}}
\toprule
\textbf{Par\^ametro / M\'etrica} & \textbf{Expectativa 3GPP / Limite F\'isico} & \textbf{Valor Obtido na Simula\c{c}\~ao (\texttt{v2})} & \textbf{Conformidade F\'isica} & \textbf{Refer\^encia Normativa (TS/TR 3GPP)} \\
\midrule
Banda, largura e grade PRB &
$BW=10\text{ MHz}$, $\mu=0$ ($15\text{ kHz}$), $N_{\text{PRB}}=52$ ($9{,}36\text{ MHz}$); TDD em $2\text{ GHz}$ pertence a n34/n39 (n1 \'e FDD) &
$f_c=2{,}0\text{ GHz}$, $BW=10\text{ MHz}$, $\mu=0$, $N_{\text{PRB}}=53$ ($9{,}54\text{ MHz}$ via $\text{RbOverhead}=0{,}04$, $+1{,}92\%$) &
Simplifica\c{c}\~ao documentada ($+1$ PRB; $f_c=2\text{ GHz}$ de refer\^encia TR~38.901) &
TS~38.104 \S5.2, Table~5.3.2-1; TS~38.101-1 Table~5.3.2-1 \\
\midrule
Classe BS, pot\^encia e geometria UMa &
Medium-Range BS ($\le 38\text{ dBm}$); UMa can\^onico $\text{ISD}=500\text{ m}$ ($R\approx 289\text{ m}$), $h_{\text{BS}}=25\text{ m}$, $h_{\text{UT}}=1{,}5\text{ m}$ &
$P_{\text{TX}}=30\text{ dBm}$ ($1\text{ W}$), $G_{\text{TX}}=15\text{ dBi}$, $R=300\text{ m}$ principal ($600$--$2000\text{ m}$ sweep); $1{\times}1$ SISO sem ICI ($I=0$) &
Conforme ($300\text{ m}$); d\'eficit esperado em $2000\text{ m}$; SISO e $I=0$ documentados &
TS~38.104 \S6.2 Table~6.2.1-1; TR~38.901 Tables~7.2-1 \& 7.3-1 \\
\midrule
Or\c{c}amento de enlace e limiares RLF &
$d'_{\text{BP}}=320\text{ m}$; NLOS $300\text{ m}$: $\text{SNR}=+23{,}71\text{ dB}$ ($\approx 0\%$ outage); NLOS $2000\text{ m}$: $\text{SNR}=-8{,}39\text{ dB} < Q_{\text{out}}=-5\text{ dB}$ ($70{,}74\%$ outage) &
Sweep \texttt{het\_sqr\_cer}: $300\text{ m}\rightarrow 46{,}87\%$ viol. ($p_{95}=12{,}38\text{ ms}$); $2000\text{ m}\rightarrow 83{,}67\%$ viol. ($p_{95}=389{,}15\text{ ms}$) &
Totalmente conforme (RLF de borda previsto por TR~38.901 e $Q_{\text{out}}/Q_{\text{in}}$) &
TR~38.901 \S7.4.1--7.4.2; TS~36.213 \S4.2.1; TS~36.133 \S7.6 \\
\midrule
Estrutura de quadro TDD &
\texttt{dl-UL-TransmissionPeriodicity} $=5\text{ ms}$ (\texttt{DDDSU}, $20$ ciclos/$100\text{ ms}$), ciclo \'util DL+S $=54/70$ ($77{,}1\%$) &
\texttt{DL|DL|DL|S|UL|} ($5\text{ ms}$), $80$ slots DL+S/$100\text{ ms}$, $0$ abortos em \texttt{AssignCsiRsOffset} &
Conforme (elimina ciclo $7\text{ ms}$ inv\'alido e batimento $\pm 7\%$) &
TS~38.331 \S6.3.2 (\texttt{TDD-UL-DL-Pattern}); TS~38.213 \S4.1 \\
\midrule
Teto de vaz\~ao DL e conserva\c{c}\~ao PRB &
TS~38.306: $32{,}193\text{ Mbps}$ ($52$ PRB) / $32{,}812\text{ Mbps}$ ($53$ PRB), $0{,}619\text{ Mbps/PRB}$; teto TB 5G-LENA: $32{,}067$ / $32{,}688\text{ Mbps}$ &
M\'ax. m\'etodo ($p_{99}$): $14{,}427\text{ Mbps}$; m\'ax. epis\'odio ($p_{99}$): $19{,}802\text{ Mbps}$; m\'ax. n\~ao-v\'acuo: $0{,}541\text{ Mbps/PRB}$; $0/68{.}040$ janelas $>53$ PRB &
100\% conforme (\texttt{cons\_err} $=0{,}0$ e conserva\c{c}\~ao estrita de espectro) &
TS~38.306 \S4.1.2; TS~38.214 \S5.1.3.1; 5G-LENA \texttt{NrLteMiErrorModel} \\
\midrule
Piso f\'isico de lat\^encia E2E &
M\'in. absoluto: $4{,}03\text{ ms}$; m\'edia s/ fila: $4{,}73\text{ ms}$; pior fase \texttt{DDDSU}: $6{,}10\text{ ms}$; $1^{\text{a}}$ retx HARQ ($N_1=2\text{ ms}$): $11{,}1$--$13{,}1\text{ ms}$ &
Menor $p_{95}$ medido: $6{,}100\text{ ms}$ (\texttt{MAIN/equal/s3009}); m\'edia \texttt{priority\_sqr}: $7{,}63\text{ ms}$, \texttt{sqr\_cer}: $11{,}60\text{ ms}$ (\texttt{MAIN}) &
100\% conforme (piso $6{,}10\text{ ms}$ reproduzido; abstra\c{c}\~ao 5QI=9 e $\mu=0$) &
TS~38.211 \S4.3.2; TS~38.214; TS~23.501 Table~5.7.4-1; TR~38.913 \S7.5 \\
\bottomrule
\end{tabular}%
}
\end{table*}
```

---

## 3. Parágrafo Pronto para *System Model & Simulation Setup* (Inglês Acadêmico IEEE)

> We evaluate the proposed closed-loop inter-slice resource controller in `ns-3` (`5G-LENA v5.2-dev`) using a physically calibrated single-cell Medium-Range FR1 urban macro (UMa) deployment (`3GPP TR 38.901`, `3GPP TS 38.104`). In accordance with `TS 38.104` §6.2 (Table 6.2.1-1) and the canonical `TR 38.901` UMa cell layout ($\text{ISD} = 500\text{ m}$, cell radius $R \approx 289\text{ m}$), the gNB is positioned at $h_{\text{BS}} = 25\text{ m}$ and serves UEs at $h_{\text{UT}} = 1.5\text{ m}$ over a $300\text{ m}$ cell radius with $P_{\text{TX}} = 30\text{ dBm}$ transmit power, $15\text{ dBi}$ omnidirectional-azimuth antenna gain, and a $9\text{ dB}$ receiver noise figure, yielding an analytical median NLOS edge SNR of $+23.71\text{ dB}$ ($>27\text{ dB}$ margin above the RLF out-of-sync threshold $Q_{\text{out}} = -5\text{ dB}$) while capturing realistic `TR 38.901` path loss and log-normal shadowing ($\sigma_{\text{SF}} = 6\text{ dB}$ NLOS). To isolate single-layer RAN slicing dynamics under bounded computational complexity, the channel operates over a $10\text{ MHz}$ carrier centered at $2.0\text{ GHz}$ with numerology $\mu = 0$ ($\Delta f = 15\text{ kHz}$, $C = 53$ usable PRBs under the `5G-LENA` $4\%$ guard overhead, closely matching the normative $N_{\text{PRB}} = 52$ of `TS 38.104`), $1\times 1$ SISO rank-1 transmission, `64-QAM` (`MCS Table 1` via `NrLteMiErrorModel`), and a `TS 38.331`-compliant $5\text{ ms}$ `DDDSU` TDD frame (`80` DL/S slots per $100\text{ ms}$ Near-RT RIC decision window). Under `TS 38.306` §4.1.2 ($54/70$ DL symbol duty factor), this configuration establishes an analytical downlink capacity ceiling of $32.19\text{ Mbps}$ ($32.81\text{ Mbps}$ for $53$ PRBs; `5G-LENA` MAC TB ceiling $32.69\text{ Mbps}$) and a deterministic E2E latency floor of $4.03\text{--}6.10\text{ ms}$ (combining the $1\text{ ms}$ backhaul link, `DDDSU` frame alignment wait, $2\text{ ms}$ `L1L2CtrlLatency`, air slot transmission, and $0.1\text{ ms}$ TB decoding), with all traffic mapped onto a shared default `5QI = 9` bearer to evaluate pure MAC PRB isolation without intra-MAC priority shortcuts.
