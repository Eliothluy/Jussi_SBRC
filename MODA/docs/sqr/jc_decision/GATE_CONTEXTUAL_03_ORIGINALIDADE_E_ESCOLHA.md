# GATE CONTEXTUAL — E3: Originalidade frente à Literatura e Formulações Alternativas (Gate G2 Atualizado)

**Data-base do Ofício:** 16/11/2026 · **Atualização G2:** 17/11/2026 · **Horizonte:** 19/11/2026  
**Regra crítica aplicada (§4):** *declarar com exatidão o nível de acesso de cada trabalho (texto integral lido vs abstract/metadados vs bloqueado por login) e não presumir sobreposição total sem comprovação técnica.*

---

## 1. Resumo Executivo e Decisão G2 Atualizada

**Parecer G2:** **`PARCIALMENTE DIFERENCIADO (`H-RLS` esclarecido pelo texto integral) / `INCONCLUSIVE` específico para `Hoashi et al. (2026)` e `SARA (2026)` por ausência de PDF integral.**

1. **H-RLS (`10.3390/electronics15184083`) — Texto integral (25 páginas) obtido e lido:** **NÃO há sobreposição total.** A leitura completa de [electronics-15-04083.pdf](file:///home/eliothluy/.gemini/antigravity/brain/4622041f-48e4-41e3-8c43-cf5834bb15d6/browser_sandbox/downloads/electronics-15-04083.pdf) demonstra que a semelhança se restringe ao padrão arquitetural geral (*política nominal lenta + corretor rápido limitado*), com divergência profunda em objetivo (energia de BS vs contenção SLA multi-slice sem folga), atuador (boost unilateral contínuo em URLLC + modo de energia vs seleção discreta entre políticas heterogêneas), modelo físico (*fluid-flow* 2 fatias sem fading vs `5G-LENA` 3 fatias) e observabilidade (feedback instantâneo sem censura vs `lag >= 200 ms` e censura RLC-UM na Emenda 4).
2. **Hoashi, Ono & Nakao (`10.1109/INFOCOM59046.2026.11571673`) — Status real de acesso:** tentativa direta no IEEE Xplore retornou **`AUTH_REQUIRED`** (apenas título, abstract integral, estrutura de seções e 1º parágrafo da Seção I visíveis sem login institucional/IEEE; OpenAlex `W7166543768` confirma `pdf_url: null`, `any_repository_has_fulltext: false`; sem cópia local em disco). **Sem presumir sobreposição total** a partir do abstract, `Hoashi` permanece **`INCONCLUSIVE`** até obtenção do PDF autenticado.
3. **Separação explícita de escopo:** o **`NO-GO` imediato em G1** aplica-se à **heurística contextual simples calibrada sobre os 25 estados atuais**, e **não** refuta a viabilidade geral da linha de pesquisa em *identificação + alocação inter-slice* sob as duas formulações alternativas analisadas abaixo (sem implementação nesta rodada).

---

## 2. Análise Detalhada dos Trabalhos-Chave e Matriz de Diferenciação (§4)

### 2.1 Auditoria do Texto Integral de H-RLS (`Electronics` 2026, 15(18), 4083)
Verificado diretamente nas Seções 3–7, Eqs. (1)–(28), Algoritmos 1–2 e Tabelas 3–4 de [`electronics-15-04083.pdf`](file:///home/eliothluy/.gemini/antigravity/brain/4622041f-48e4-41e3-8c43-cf5834bb15d6/browser_sandbox/downloads/electronics-15-04083.pdf):
* **Modelo e cenário (Seções 3.1 e 6.1):** simulador discreto com apenas **`2` fatias (`eMBB`, `URLLC` — sem `mMTC`, deixada para trabalhos futuros na Sec. 7)**, $B_{\text{total}} = 100$ PRBs ($100\text{ Mbps}$, eficiência constante $\eta_e = \eta_u = 1\text{ Mbps/PRB}$), que **abstrai explicitamente *fast fading* e interferência inter-célula** e usa proxy determinístico de atraso de fila $D_u(t) = \frac{Q_u^+(t)}{\max(C_u(t),\epsilon)}\Delta t$ (Eq. 5, alvo $D_u^{\max} = 5\text{ ms}$).
* **Arquitetura e atuador (Seções 4.1–4.3, Eqs. 12–13, 20, 23–28):**
  * *Non-RT PPO rApp* ($T_N = 100$ passos = `1 s`) gera orçamento nominal $\bar{b}_u(k)$ e modo discreto de energia $\bar{m}(k) \in \{0,1,2\}$ minimizando potência $P(t) = P_{ba} + \alpha_{m(t)} P_{\text{load}} \rho_{\text{load}}(t)$ (Eqs. 9–14).
  * *Near-RT RLS xApp* (`10 ms`, $\lambda_{\text{RLS}} = 0{,}92$) prediz $\hat{y}(t) = W^T(t-1)X(t)$ com 8/10 métricas normalizadas instantâneas (Eq. 23). Se $\hat{y}(t) \le \Gamma_{\text{risk}} = 0{,}45$, mantém $\Delta b_u = 0, \Delta m = 0$; se $0{,}45 < \hat{y}(t) \le 0{,}60$, aplica **acréscimo unidirecional em URLLC** $\Delta b_u(t) = \min(\lceil \eta(\hat{y}(t)-\Gamma_{\text{risk}})\rceil, \Delta b_{\max})$ ($\Delta b_{\max} = 25$ PRBs); se $> 0{,}60$, satura $\Delta b_u = 25$ e degrada o modo de energia $\Delta m(t) \in \{1,2\}$ (Eqs. 12–13, Alg. 2).
* **Diferenças técnicas objetivas (por que não há sobreposição total):**
  1. H-RLS ajusta **apenas margem unilateral positiva de PRB URLLC + modo de potência** sobre folga de `100 PRBs` (na Tabela 4, `eMBB` mantém satisfação `1,00 ± 0,00`), **não** arbitra políticas concorrentes (`B0 / prop / equal / piso`) nem enfrenta interferência não-monotônica entre **3 fatias** (`eMBB+URLLC+mMTC`) sob `C = 53` RBGs saturados em `5G-LENA`.
  2. H-RLS assume **observação instantânea, exata e não censurada** de $D_u(t), Q(t), A(t)$ e $y(t)$ em cada passo de `10 ms` (Alg. 2), sem feedback assíncrono atrasado nem censura por perdas RLC-UM (`Emenda 4`).
  3. Inatividade em H-RLS ($\hat{y} \le 0{,}45$) é apenas zona morta de limiar linear, sem critério de **abstenção protetora contra dano cruzado (`HARM`) sob incerteza/censura**.

### 2.2 Análise de Hoashi, Ono & Nakao (`IEEE INFOCOM 2026`, `10.1109/INFOCOM59046.2026.11571673`)
* **Nível de verificação real:** página IEEE Xplore (`AUTH_REQUIRED` para PDF/texto completo); verificados **título, autores, venue, abstract completo (OpenAlex `W7166543768` / IEEE)** e metadados públicos (`99,8%` de proteção à fatia de alta confiabilidade; `+58,5%` de utilização nas não prioritárias vs reserva estática).
* **Por que NÃO presumir sobreposição total:** embora seja do mesmo laboratório do SQR original (`Nakao Lab`) e separe RL *model-based* de uma *regra determinística de realocação ponderada por prioridade*, o foco declarado no abstract é **isolamento hierárquico que protege uma fatia prioritária designada frente a fatias secundárias**. No nosso problema, `STRICT_ALL3` exige conformidade **simultânea** das 3 fatias e evita `HARM` cruzado em `eMBB`; além disso, sem ler o PDF autenticado não se sabe se Hoashi usa regressão quantílica SQR, se contempla feedback atrasado/censurado (`Emenda 4`), abstenção probabilística ou escalonamento em `ns-3/5G-LENA`. Logo, o risco permanece **`INCONCLUSIVE (MODERADO–ALTO a checar no PDF)`**, sem presumir equivalência.

### 2.3 Matriz Consolidada dos 7 Antecedentes (§4)

| Trabalho (ID / Venue) | Nível de verificação | Sobreposição / Antecedente real | Diferença técnica observada | Lacuna remanescente | Risco atualizado |
|---|---|---|---|---|---|
| **H-RLS** — Riel et al. (*Electronics* 15(18):4083, 2026) | **Texto integral lido (25 pp. PDF)** | Arquitetura em 2 tempos: política nominal + delta limitado Near-RT (`RLS`). | Foco em energia (`m∈{0,1,2}`), 2 fatias *fluid-flow* (`100 PRB`), boost unilateral URLLC, telemetria instantânea sem censura, sem abstenção entre políticas discretas. | Nenhuma sobre o texto principal (lido integralmente). | **BAIXO–MODERADO** (diferenciado) |
| **Hoashi, Ono & Nakao** (*IEEE INFOCOM 2026*, `10.1109/INFOCOM59046.2026.11571673`) | **Abstract completo + metadados (`AUTH_REQUIRED` p/ PDF)** | RL *model-based* + regra determinística *priority-weighted* que isola fatia crítica (`99,8%` proteção, `+58,5%` uso secundárias). | Sem presumir sobreposição total: isolamento por prioridade fixa difere de arbitragem `STRICT_ALL3` anti-`HARM` sob feedback atrasado/censurado (`Emenda 4`). | Obter e ler o **PDF completo via acesso IEEE autorizado**: verificar se usa SQR, quais fatias/SLAs, regra de corte e tratamento de atraso/abstenção. | **INCONCLUSIVE (MODERADO–ALTO)** |
| **SafeSlice** — Nagib et al. (*IEEE ICMLCN 2025*, `arXiv:2503.12753`) | **Abstract completo arXiv** | Recompensa *risk-sensitive* + *safety layer* supervisionada que projeta ação DRL para a ação segura mais próxima. | Opera em espaço contínuo de DRL assumindo modelo de custo imediato sem censura; não faz seleção discreta com abstenção sobre controlador nominal sob Emenda 4. | Ler PDF `arXiv:2503.12753` para detalhes da *safety layer* e baselines. | **MODERADO** |
| **xApp Fuzzy** — Ertuğrul et al. (*ComCom* 258:108656, 2026) | Metadados + resumo público | xApp em lógica fuzzy no Near-RT RIC para alocação dinâmica de PRBs inter-slice. | Regras fuzzy contínuas diferem de arbitragem discreta anti-`HARM` com censura; porém colidiria com heurísticas manuais de fila/PRB. | Ler PDF para checar regras fuzzy, pool elástico e métricas de heterogeneidade. | **MODERADO** |
| **Online Meta-Learning** — Zoulkarni et al. (*MeditCom 2026*, `10.1109/MeditCom67211.2026.11641095`) | Metadados DBLP/resumo público | Adaptação rápida online (*meta-learning*) de alocação RAN sob não-estacionariedade. | Adaptação contínua de política/pesos difere de seletor discreto com abstenção explícita sob feedback atrasado/censurado. | Ler PDF MeditCom 2026 (nível de decisão, métricas e proteção a dano). | **MODERADO** |
| **DARIO** — Ajayi et al. (*NOMS 2024*, `10.1109/NOMS59830.2024.10575766`) | Metadados + resumo público | MAB *drift-aware* para seleção adaptativa de políticas em **admissão de fatias (SAC)**. | Problema distinto (admissão de NSRs vs PRBs a cada `100 ms`), sem abstenção anti-`HARM` nem censura de latência de cauda. | Conferir PDF NOMS 2024 apenas para referência metodológica de *drift*. | **BAIXO–MODERADO** |
| **SARA** — Alagheh Band et al. (*CNSM 2026*, Best Poster) | Título/autores/prêmio apenas | Possível antecedente em alocação SLA-aware de RAN slicing. | Indeterminado (sem abstract público acessível). | Localizar abstract/pôster/artigo CNSM 2026. | **INCONCLUSIVE** |

---

## 3. Separação Explícita: `NO-GO` da Heurística Atual vs Viabilidade da Contribuição Geral

* **O que recebe `NO-GO` agora:** ajustar ou implementar uma **heurística contextual simples** (regras de limiar em Bloco A) sobre os `25` estados de Gate 1, bem como retomar o seletor `A5 max-min/LCB` quantílico bruto.
* **Por que a contribuição geral *Identificação + Alocação Inter-Slice* NÃO está invalidada:**
  1. As evidências contrafactuais provam ganho real de realocação inter-slice (`B0=8/18` vs `equal=12/18`, `prop=11/18` e oráculo restrito $O_3=14/18=77,8\%$ em `ALL3_COMPLETE`; além de `4/9` recuperáveis na `D2-v2`).
  2. Existem sinais pré-decisão legítimos em `eMBB` (`AUROC(f_0)=0,7283`), em colapsos extremos de `B0` (`w2667, w1283, w2685`) e no risco eMBB do `prop` (`w1890, w2229`).
  3. A leitura integral de **H-RLS** confirma que a literatura aberta analisada **não** resolve arbitragem discreta multi-slice com proteção assimétrica a dano sob feedback atrasado e censurado em `5G-LENA` — restando verificar o PDF de `Hoashi (2026)` e `SARA (2026)` e endereçar as duas formulações conceituais abaixo.

---

## 4. As Duas Formulações Alternativas de Decisão (Conceitual — Sem Implementação)

| Dimensão | Formulação 1 — Árbitro Discreto Protetor sobre Políticas Estruturais + Piso | Formulação 2 — Decisão Sequencial Contextual de Baixo Espaço de Ações sob Censura |
|---|---|---|
| **Mecanismo central** | Árbitro de intervenção binária/discreta em $\mathcal{A}_1(s_t) = \{\text{MANTER\_B0 (abster-se)}, \text{INTERVIR\_PROP\_ATIVA}, \text{INTERVIR\_EQUAL\_ATIVA}\}$ (incluindo variante explícita com **piso mínimo preventivo** p/ chegadas intra-janela à luz de `w1254/w2552`), com **penalização assimétrica de `HARM`** sobre `B0=PASS`. | Aprendizado contextual (bandit contextual / classificação sensível a custo — **algoritmo não escolhido agora**) sobre $\le 3$–$4$ macro-ações discretas, incorporando explicitamente no modelo de decisão e atualização o **atraso e a censura do feedback URLLC (`Emenda 4`)**. |
| **Informação usada no instante `t`** | Exclusivamente Bloco A: demanda $(n_s, q_s, d_s)$, proposta $a_{B0}$ e projeção, predições/incertezas locais $(f_i, \sigma_i)$, KPI eMBB anterior e histórico URLLC fechado (`lag >= 200 ms`). | Mesmo Bloco A **estendido** (caso novo ciclo experimental exponha idade do pacote URLLC em fila / taxa recente de pools `OPEN`) + atualização de recompensa/custo apenas quando o pool fecha ou com correção explícita de censura. |
| **Evidências existentes que apoiam/limitam** | • Apoio: $O_3=14/18$ (`0` danos) vs `8/18` (`B0`); colapsos `w2667, w1283, w2685` e risco `prop` em `w1890, w2229` visíveis em Bloco A; em `D2-v2`, `equal` teve `2R/0H/3B/7NR`.<br>• Limite: `R1/R2` manuais sem ajuste resgatam no máx. `2` casos limpos sem confundir seed. | • Apoio: `eMBB` (`AUROC(f_0)=0,7283`) e `mMTC` (`0,8187`) são previsíveis; `11.249/11.335` pools fecham em `D1-v9` (mediana `200 ms`, `37` expirados); `B0` vs `equal/prop` concentra contraste em `<= 3` ações.<br>• Limite: URLLC `f_1` tem `AUROC=0,5607` e `rho(sigma_1, |err|)=-0,2614`; `N=18` (`2` seeds) impede qualquer treino agora. |
| **Limitações de observabilidade determinantes** | `prev_kpi_u_p95_imm = NaN` em `18/18`; `w1621 (HARM[u])` vs `w1546/w1588` indistinguíveis por fase TDD intra-janela de `100 ms`; trade-off entre `active_mask` (`w1254` falha por chegada futura) vs piso fixo (`17–36` PRBs ociosos em `9/25`). | Além do `lag >= 200 ms` e da censura em congestão (`69 vs 19.235 B`), há feedback parcial (*bandit*): online só se observa o braço executado, e falhas URLLC puramente intra-janela podem permanecer irredutíveis sem telemetria MAC sub-`100 ms`. |
| **Status nesta rodada** | **Conceitual apenas — NÃO implementar nem ajustar sobre os 25 estados.** | **Conceitual apenas — NÃO escolher algoritmo nem iniciar treino sem novos dados/PDFs.** |

---

## 5. Síntese Final do Gate G2

* **H-RLS (`electronics15184083`):** **verificado integralmente — sem sobreposição total.**
* **Hoashi et al. (`INFOCOM59046.2026.11571673`) e SARA (`CNSM 2026`):** **`INCONCLUSIVE` específico (sem presumir sobreposição total)** enquanto o acesso ao texto completo autenticado não for concluído.
* **Protocolo futuro (§5.1–§5.2):** caso venha a ocorrer avaliação futura após fechamento de G2 e amostra independente em `seeds >= 3101`, mantêm-se `B0-rest`, `equal_ativo`, `prop_ativo`, intervenções não seletivas (`always-*`), `A5` como referência negativa, e métrica primária congelada `STRICT_ALL3` em `3×COMPLETE` com `RESCUE`, `HARM` e `NET_RESCUE`.
