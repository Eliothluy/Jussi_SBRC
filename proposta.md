# Relatório Técnico de Pesquisa: SQR-JC (Safe Quantile Regression with Joint Coordination & Regime Awareness)

**Projeto:** Evolução do `Jussi_SBRC` (MODA: ns-3.46 + 5G-LENA v5.2 + O-RAN/NORI)  
**Objetivo:** Superar a falha estrutural do SQR original (*Tuerxun & Nakao, CCNC 2026* [1]) sob regimes de sobrecarga física e corrigir a quebra de segurança causada pela projeção proporcional ingênua ($\Pi$), fundamentando o problema e as soluções na literatura de Estado da Arte.

---

## 1. Formulação do Problema e Fontes no Estado da Arte

A auditoria empírica realizada sobre o SQR original [1] no ambiente MODA (ns-3.46 + 5G-LENA [2]) expôs três falhas fundamentais que conectam lacunas abertas na literatura recente de *Safe Reinforcement Learning* e *O-RAN Slicing*:

### 1.1. Problema 1: O Gap "Sim-to-Real" entre Modelos Analíticos e Pilha 3GPP Real
* **Fonte do Problema no Estado da Arte:** Grande parte da literatura de *Model-Based RL* (MBRL) para RAN Slicing — incluindo o baseline **KBRL** de *Alcaraz et al. (IEEE TWC, 2022)* [3] e o próprio **SQR** de *Tuerxun & Nakao (IEEE CCNC, 2026)* [1] — avalia seus controladores em simuladores sintéticos baseados em capacidade de Shannon contínua ou modelos LTE legados (**3GPP TS 36.942** [4] e **TS 36.104** [5]). Como destacado por *Polese et al. (IEEE COMST, 2023)* [6] e *Bonati et al. (Colosseum, 2021)* [7], abstrair as camadas MAC/RLC/RRC omite fenômenos não-lineares críticos do 5G NR (**3GPP TR 38.901** [8] e **TR 38.104** [9]), como quedas de enlace (*Radio Link Failure* — RLF), esvaziamento abrupto de buffers RLC durante reestabelecimento RRC e quantização de RBGs.
* **Evidência na Auditoria (`Jussi_SBRC`):** Ao parametrizar uma macrocélula de $2\text{ km}$ com potência de *small cell* ($30\text{ dBm}$) em vez dos $43\text{ dBm}$ padronizados no **3GPP TR 38.104** [9], o cenário de [1] impõe um déficit de *link budget* de $25\text{–}30\text{ dB}$. No 5G-LENA [2], isso resulta em $56,6\%$ de janelas com entrega nula por RLF e violação de SLA de $\sim 88\%$ em todos os métodos.

### 1.2. Problema 2: Colapso do Conjunto Seguro ($\mathcal{A}_{\text{safe}} = \emptyset$) e Cegueira de Inviabilidade em CMDP
* **Fonte do Problema no Estado da Arte:** A formulação de *Constrained Markov Decision Processes* (CMDPs) via conjuntos de segurança baseados em incerteza (como em *Safe Exploration* de *Turchetta et al., NeurIPS* [10] e *Berkenkamp et al., NeurIPS* [11], ou em *Perpetual SLA-Aware RL — PSASlicing* de *Yin et al., GLOBECOM 2024* [12]) assume a premissa de **viabilidade estrita** (*Slater's condition* / existência de política segura a priori: $\mathcal{A}_{\text{safe}, t} \neq \emptyset$). Na literatura de RL para redes (*Yungaicela-Naula et al., RSLAQ, 2025* [13]; *Hua et al., GAN-DDQN, JSAC 2019* [14]), quando a demanda excede temporariamente a capacidade física da célula, métodos baseados em penalidades sofrem instabilidade de gradiente.
* **Mecanismo da Falha em [1]:** No Algoritmo 1 de *Tuerxun & Nakao* [1], a função objetivo regularizada por custo $U(a) = f_{\tau_t^{(i)}}(z_t^{(i)}, a) - \gamma(a/C)$ reside exclusivamente dentro da condicional `if` $\mathcal{A}_{\text{safe}, t}^{(i)} \neq \emptyset$. Sob inviabilidade física ou congestionamento, a regressão quantílica assimétrica de *Koenker & Bassett* [15] amplificada por $P_{\text{penalty}}$ desloca $f_{\tau_t^{(i)}}$ para baixo de $\Lambda^{(i)}$. Com $\mathcal{A}_{\text{safe}, t}^{(i)} = \emptyset$ permanentemente:
  1. O termo de eficiência $-\gamma(a/C)$ torna-se **código morto** ($\gamma=0$ e $\gamma=1$ ficam estatisticamente idênticos);
  2. Ocorre **integral windup** no controlador heurístico ($\tau_t^{(i)} \to \tau_{\min}$ e $M_t^{(i)} \to M_{\max}$).

### 1.3. Problema 3: Insegurança da Projeção Proporcional Desacoplada em Multi-Agent RL
* **Fonte do Problema no Estado da Arte:** Em arquiteturas multi-agente onde aprendizes por slice decidem alocações independentes $a_{\text{req}, t}^{(i)}$ e um coordenador aplica uma projeção centralizada $\Pi$ para satisfazer $\sum_{i=1}^L a^{(i)} \le C$ (*Jia et al., IEEE TCOMM 2024* [16]; *Sulaiman et al., MicroOpt, IEEE TNSM 2025* [17]), o escalonamento proporcional uniforme ($\Pi(a^{(i)}) \propto a^{(i)}$) não preserva restrições não-lineares de QoS.
* **Mecanismo da Falha em [1]:** O corte proporcional cego retira PRBs igualmente de slices que possuem alta margem ociosa ($\sigma_t^{(i)} M_t^{(i)}$ elevado) e de slices que estão operando exatamente no limiar crítico $a_{\min,\text{safe}}^{(i)} = \min \mathcal{A}_{\text{safe}, t}^{(i)}$, induzindo violações simultâneas em múltiplos slices mesmo quando havia capacidade global suficiente ($\sum_i a_{\min,\text{safe}}^{(i)} \le C$).

---

## 2. Cenário de Simulação Calibrado e Aplicação O-RAN

Para mitigar o problema físico sem perder a comparabilidade com [1] e [3], propomos uma aplicação de **Controle de Fatiamento Tolerante a Sobrecarga via xApp no Near-RT RIC**, avaliada no framework **MODA** (ns-3.46 + 5G-LENA v5.2 [2]) e integrada ao barramento E2 via **NORI** [18].

### 2.1. Matriz de Calibração Física (Padrão 3GPP 5G NR)

| Parâmetro | Artigo Original (*CCNC 2026* [1]) | Nosso Cenário Calibrado (*MODA / 5G-LENA* [2]) | Fonte / Justificativa no Estado da Arte |
| :--- | :--- | :--- | :--- |
| **Pilha de Protocolos** | Simulador abstrato não especificado | **ns-3.46 + 5G-LENA v5.2 (`SqrMacScheduler`)** | Pilha 3GPP NR completa com MAC/RLC/RRC reais [2, 6]. |
| **Modelo de Canal** | 3GPP TS 36.942 (LTE RF) [4] | **3GPP TR 38.901 UMa (5G NR)** [8] | Padrão oficial 3GPP para frequências de $0.5\text{–}100\text{ GHz}$ [8]. |
| **Banda e Orçamento ($C$)** | Omitido na Tab. I ($C=100$ na Fig. 4) [1] | **$10\text{ MHz}$ ($C=53\text{ PRBs}$) e $20\text{ MHz}$ ($C=106\text{ PRBs}$), $\mu=0$** | Alinhado à tabela de PRBs do **3GPP TS 38.104** [9] e ao baseline KBRL [3]. |
| **Potência TX / Raio** | $30\text{ dBm}$ @ $2000\text{ m}$ (Inviável) [1] | **Varredura de Regimes:** $R \in \{300, 800, 1500, 2000\}\text{ m}$ e calibração Macro $43\text{ dBm}$ | **3GPP TR 38.104** define $43\text{–}46\text{ dBm}$ para Macro BS e $30\text{ dBm}$ para Micro/Picocell [9]. |
| **Interface RIC** | Conceitual (sem protocolo E2) [1] | **IPC POSIX (baseline ideal) + E2SM-KPM / RIC Control via NORI** [18] | Conformidade com a arquitetura O-RAN ALLIANCE WG3 [6, 18]. |

---

## 3. Nossa Proposta Metodológica: O Algoritmo **SQR-JC** e Suas Fontes Teóricas

Mantendo a leveza computacional do regressor quantílico online com kernel Matérn [1, 15] (essencial para operar abaixo de $100\text{ ms}$ no Near-RT RIC), introduzimos três mecanismos fundamentados na teoria de controle e otimização:

### 3.1. Inovação 1: *Soft-Safe Fallback* via Desacoplamento de Incerteza Epistêmica vs. Aleatória
* **Fundamentação Teórica (Fontes):** Inspirado na decomposição de incerteza epistêmica e aleatória em aprendizado de máquina seguro (*Kendall & Gal, NeurIPS 2017* [19]) e em técnicas de **Relaxamento de Restrições (Constraint Relaxation / Exact Penalty Functions)** em Controle Preditivo Baseado em Modelo — *MPC* (*Kerrigan & Maciejowski, 2000* [20]). Em MPC, quando o conjunto viável é vazio devido a perturbações físicas externas, relaxa-se o limiar $\Lambda^{(i)}$ pelo menor valor de folga $\nu_t^{(i)} \ge 0$ que restaura a viabilidade, mantendo a função custo ativa.
* **Formulação no SQR-JC:** Quando $\mathcal{A}_{\text{safe}, t}^{(i)} = \emptyset$, verificamos a incerteza média do kernel $\bar{\sigma}_t^{(i)} = \frac{1}{C+1}\sum_{a=0}^C \sigma_t^{(i)}(z_t^{(i)}, a)$.
  * Se $\bar{\sigma}_t^{(i)} > \sigma_{\text{th}}$ (**Incerteza Epistêmica Alta** — fase inicial de aprendizado), mantemos a exploração original (*Softmax / Safety-First k-NN* [1]).
  * Se $\bar{\sigma}_t^{(i)} \le \sigma_{\text{th}}$ (**Inviabilidade Física / Saturação de Canal** — o modelo já conhece a curva, mas $\max_a f_{\tau_t^{(i)}}(z, a) < \Lambda^{(i)}$), definimos o limiar efetivo atingível $\Lambda_{\text{eff}, t}^{(i)} = \max_{a} f_{\tau_t^{(i)}}(z_t^{(i)}, a) - \xi^{(i)}$ e construímos o conjunto quase-seguro:
    $$\tilde{\mathcal{A}}_{\text{safe}, t}^{(i)} = \left\{ a \in \{0, \dots, C\} \;\middle\vert{}\; f_{\tau_t^{(i)}}(z_t^{(i)}, a) \ge \Lambda_{\text{eff}, t}^{(i)} \right\}$$
    A ação é então escolhida reativando o termo de eficiência espectral:
    $$a_{\text{base}, t}^{(i)} = \arg\max_{a \in \tilde{\mathcal{A}}_{\text{safe}, t}^{(i)}} \left( f_{\tau_t^{(i)}}(z_t^{(i)}, a) - \gamma \frac{a}{C} \right)$$
* **Impacto:** Elimina o desperdício de PRBs em UEs limitados por SINR/RLF (onde alocar mais PRBs não eleva a vazão), reativando o parâmetro $\gamma$ em qualquer regime operacional.

### 3.2. Inovação 2: Projeção Conjunta Consciente de Risco ($\Pi_{\text{JC}}$ — *Lexicographic Slack Compression*)
* **Fundamentação Teórica (Fontes):** Baseado em **Alocação de Recursos com Garantia de Piso (*Minimum-Guarantee Water-Filling / Lexicographic Minimax Optimization*)** (*Bertsekas, Nonlinear Programming* [21]; *Boyd & Vandenberghe, Convex Optimization* [22]) e coordenação hierárquica inter-slice (*Jia et al., 2024* [16]). Em vez de projetar o vetor bruto na norma $\ell_1$/proporcional sobre a origem, projeta-se apenas o **excedente de margem** acima do piso de segurança $a_{\min, t}^{(i)} = \min \mathcal{A}_{\text{safe}, t}^{(i)}$.
* **Formulação no SQR-JC:** Seja $a_{\text{req}, t}^{(i)} = a_{\text{base}, t}^{(i)} + \text{round}(\sigma_t^{(i)} M_t^{(i)})$ e a folga de margem $s_t^{(i)} = a_{\text{req}, t}^{(i)} - a_{\min, t}^{(i)} \ge 0$. Quando $\sum_{i=1}^L a_{\text{req}, t}^{(i)} > C$:
  1. **Regime Conjuntamente Viável ($\sum_{i=1}^L a_{\min, t}^{(i)} \le C$):** Garante-se integralmente $a_{\min, t}^{(i)}$ para todos os slices e comprime-se apenas a folga $s_t^{(i)}$:
     $$a_{\text{final}, t}^{(i)} = a_{\min, t}^{(i)} + \left\lfloor \left(C - \sum_{j=1}^L a_{\min, t}^{(j)}\right) \frac{s_t^{(i)}}{\sum_{j=1}^L s_t^{(j)}} \right\rfloor$$
  2. **Regime Conjuntamente Inviável ($\sum_{i=1}^L a_{\min, t}^{(i)} > C$):** Alocação marginal gulosa (*Greedy Submodular/Marginal Utility Allocation*) incrementando $1\text{ RBG}$ por passo ao slice que apresenta maior derivada discreta de satisfação normalizada $\Delta_a \left[ \min\left(1, f_{\tau_t^{(i)}}(z_t^{(i)}, a) / \Lambda^{(i)}\right) \right]$.

### 3.3. Inovação 3: Mecanismo *Anti-Windup* de Quantil/Margem e Filtragem Cross-Layer (MAC $\to$ RIC)
* **Fundamentação Teórica (Fontes):** Baseado na teoria clássica de **Integrator Anti-Windup em Controle Adaptativo/PID** (*Åström & Murray, Feedback Systems* [23]) combinada com *Conformal Quantile Calibration* (*Romano et al., NeurIPS 2019* [24]). Quando o atuador físico atinge saturação ($a^{(i)} = C$ ou queda física de enlace no MAC onde `prb_executed == 0`), continuar integrando o erro de SLA corrompe o estado interno do controlador ($\tau_t^{(i)}$ e $M_t^{(i)}$), causando longo tempo de recuperação (*overshoot* reverso) quando o canal normaliza.
* **Formulação no SQR-JC:** Utilizando as métricas expostas pelo nosso `SqrMacScheduler` (Gate R27: `prb_requested`, `prb_scheduled`, `prb_executed`, `prb_idle`), introduzimos uma guarda *cross-layer*:
  * Se a violação de SLA ocorre em uma janela onde os UEs do slice sofreram anomalia de camada física/RLC (`prb_executed == 0` com UEs desconectados/RLF) ou saturação global de capacidade ($\sum_i a_{\min, t}^{(i)} > C$), o integrador de $\tau_t^{(i)}$ e $M_t^{(i)}$ é **congelado** (*clamping anti-windup*) e a atualização do kernel usa o peso padrão $\alpha_{\text{OP}}^{(i)} = \eta(\tau_t^{(i)} - 1)$ sem o fator destrutivo $P_{\text{penalty}}$.

---

## 4. Plano de Experimentos e Hipóteses Falsificáveis

| Experimento | Configuração no MODA (`ns-3.46` + `5G-LENA`) | Hipótese Científica Testada |
| :--- | :--- | :--- |
| **Exp. 1 — Auditoria do Estado da Arte e Varredura de Regimes** | $R \in \{300, 800, 1500, 2000\}\text{ m}$ a $30\text{ dBm}$ + Cenário Macro Calibrado ($43\text{ dBm}$ @ $2000\text{ m}$), 10 seeds $\times$ $600\text{ s}$. | Demonstrar o déficit de $25\text{–}30\text{ dB}$ de [1] (G1) e provar que no regime viável/marginal o **SQR-JC** reduz violações induzidas por projeção a zero. |
| **Exp. 2 — Resiliência a Transição Não-Estacionária (*Anti-Windup*)** | Episódio de $600\text{ s}$ com surto de tráfego/degradação entre $200\text{–}400\text{ s}$ (Viável $\to$ Inviável $\to$ Viável). | Provar que o SQR original [1] sofre *windup* (mantendo alto consumo de PRBs entre $400\text{–}600\text{ s}$), enquanto o **SQR-JC** recupera a economia de PRBs imediatamente após $t=400\text{ s}$. |
| **Exp. 3 — Fidelidade O-RAN Real (IPC vs. E2/NORI)** | Execução do **SQR-JC** como xApp real via interface E2 (E2SM-KPM + RIC Control no **NORI** [18]). | Quantificar o custo de latência/jitter da pilha SCTP/ASN.1 do protocolo E2 real frente ao acoplamento ideal via memória compartilhada/IPC (G5). |

---

## 5. Referências Bibliográficas (Base para o Artigo)

1. **A. Tuerxun and A. Nakao**, "Safe RAN Slicing in O-RAN: Minimizing SLA Violations via Model-Based Reinforcement Learning," in *Proc. IEEE 23rd Consumer Communications & Networking Conference (CCNC)*, 2026, doi: 10.1109/CCNC65079.2026.11366589.
2. **N. Patriciello, S. Lagen, B. Bojovic, and L. Giupponi**, "An E2E simulator for 5G NR networks," *Simulation Modelling Practice and Theory (5G-LENA)*, vol. 96, 101933, 2019.
3. **J. J. Alcaraz, F. Losilla, A. Zanella, and M. Zorzi**, "Model-Based Reinforcement Learning With Kernels for Resource Allocation in RAN Slices," *IEEE Transactions on Wireless Communications*, vol. 22, no. 1, pp. 486–501, Jan. 2022.
4. **3GPP**, "Evolved Universal Terrestrial Radio Access (E-UTRA); Radio Frequency (RF) system scenarios," *3GPP TR 36.942*, v15.0.0, 2018.
5. **3GPP**, "Evolved Universal Terrestrial Radio Access (E-UTRA); Base Station (BS) radio transmission and reception," *3GPP TS 36.104*, v15.15.0, 2022.
6. **M. Polese, L. Bonati, S. D'Oro, S. Basagni, and T. Melodia**, "Understanding O-RAN: Architecture, Interfaces, Algorithms, Security, and Research Challenges," *IEEE Communications Surveys & Tutorials*, vol. 25, no. 2, pp. 1376–1411, 2023.
7. **L. Bonati et al.**, "Colosseum: Large-Scale Wireless Experimentation Through Hardware-in-the-Loop Network Emulation," in *Proc. IEEE DySPAN*, 2021, pp. 105–113.
8. **3GPP**, "Study on channel model for frequencies from 0.5 to 100 GHz," *3GPP TR 38.901*, v16.1.0, 2020.
9. **3GPP**, "NR; Base Station (BS) radio transmission and reception," *3GPP TS 38.104*, v17.0.0, 2022.
10. **M. Turchetta, F. Berkenkamp, and A. Krause**, "Safe Exploration in Finite Markov Decision Processes with Gaussian Processes," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 29, 2016.
11. **F. Berkenkamp, M. Turchetta, A. Schoellig, and A. Krause**, "Safe Model-Based Reinforcement Learning with Stability Guarantees," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
12. **M. Yin, Y. Deng, A. Kak, N. Choi, and T. Han**, "PSASlicing: Perpetual SLA-Aware Reinforcement Learning for O-RAN Slice Management," in *Proc. IEEE GLOBECOM*, 2024, pp. 4534–4539.
13. **N. M. Yungaicela-Naula, V. Sharma, and S. Scott-Hayward**, "RSLAQ—A Robust SLA-Driven 6G O-RAN QoS xApp Using Deep Reinforcement Learning," *arXiv preprint arXiv:2504.09187*, 2025.
14. **Y. Hua, R. Li, Z. Zhao, X. Chen, and H. Zhang**, "GAN-Powered Deep Distributional Reinforcement Learning for Resource Management in Network Slicing," *IEEE Journal on Selected Areas in Communications (JSAC)*, vol. 38, no. 2, pp. 334–349, 2019.
15. **R. Koenker and G. Bassett Jr.**, "Regression Quantiles," *Econometrica*, vol. 46, no. 1, pp. 33–50, 1978.
16. **Y. Jia, C. Zhang, N. Li, Y. Huang, and T. Q. S. Quek**, "Hierarchical Intelligence Enabled Joint RAN Slicing and MAC Scheduling for SLA Guarantee," *IEEE Transactions on Communications*, 2024.
17. **M. Sulaiman et al.**, "MicroOpt: Model-Driven Slice Resource Optimization in 5G and Beyond Networks," *IEEE Transactions on Network and Service Management*, 2025.
18. **LASSE / UFPA**, "NORI: ns-3 O-RAN Interface for Near-RT RIC integration," GitHub Repository (`https://github.com/lasseufpa/nori`).
19. **A. Kendall and Y. Gal**, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
20. **E. C. Kerrigan and J. M. Maciejowski**, "Soft constraints and exact penalty functions in model predictive control," in *Proc. UKACC International Conference on Control*, 2000.
21. **D. P. Bertsekas**, *Nonlinear Programming*, 3rd ed. Belmont, MA: Athena Scientific, 2016.
22. **S. Boyd and L. Vandenberghe**, *Convex Optimization*. Cambridge University Press, 2004.
23. **K. J. Åström and R. M. Murray**, *Feedback Systems: An Introduction for Scientists and Engineers*, 2nd ed. Princeton University Press, 2021.
24. **Y. Romano, E. Patterson, and E. Candès**, "Conformalized Quantile Regression," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.