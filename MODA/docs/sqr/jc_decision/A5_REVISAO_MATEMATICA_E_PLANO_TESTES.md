# Revisão Matemática da A5 (`SQR-JC-A5v2`), Escopo Alternativo Enxuto, Matriz Sintética e Proposta de Mini-Replay (Apenas Especificação)

**Status:** Documento pré-gate (**ZERO** código novo do seletor implementado, **ZERO** replays executados).  
**Base diagnóstica:** [`A5_GATE1_DIAGNOSTICO_RETROSPECTIVO.md`](A5_GATE1_DIAGNOSTICO_RETROSPECTIVO.md) (127 replays únicos válidos do Gate 1, commit `3afe72c`) e [`B0_FROZEN_MANIFEST.json`](B0_FROZEN_MANIFEST.json) (`8375f4d`, intocado).  
**Bloqueios mantidos:** `Gate 2`, `D3-v2`, `PILOT`, `MAIN`, `A2b`, `A3b` e seeds `3101–3103` permanecem estritamente bloqueados.

---

## 1. Decisão Metodológica Prévia: Formulação Revisada vs Escopo Científico Alternativo Enxuto

Antes de apresentar a especificação matemática revisada `A5-v2`, o diagnóstico retrospectivo impõe uma advertência científica transparente (**Item 3 do protocolo pré-gate**):

- Nos modelos congelados do `B0-rest` ($\tau_i \in [0{,}01,\,0{,}05]$), a cabeça `f_1` do **URLLC** estima a cauda de `~1,8%` pior caso e possui **`AUROC = 0,5607`** (`Spearman rho = 0,1168`, `TPR = 2,67%` em `f_1 >= 0`, correlação invertida $\rho(\sigma_1, |err|) = -0{,}2614$).
- Mesmo um **oráculo retrospectivo em *hindsight*** restrito a `{B0-rest, equal, prop}` (`O_3`, não causal) atinge **`14/18 (77,8%)`** no Universo A — apenas **`+2/18` (`+11,1 pp`)** acima do baseline cego estático **`equal = (18,17,18)` (`12/18 = 66,7%`)**. E nenhuma regra causal sobre `f_i` e `sigma_i` pré-decisão superou `12/18`.

Consequentemente, a revisão matemática `A5-v2` abaixo resolve **todos** os defeitos internos do seletor (assimetria de escala do `max-min`, colapso de capacidade por $-\beta\sigma - \gamma a/C$, insuficiência do piso de `1` PRB e violação de slices satisfazíveis), **porém seu sucesso empírico futuro continua condicionado à qualidade do preditor de viabilidade de entrada (`phi_i`)**:

1. **Opção A — Escopo Alternativo Mais Enxuto (Recomendado se `B0-rest` permanecer como único modelo sem nova cabeça de previsão):**
   - Não gastar novos replays tentando extrair ganho líquido sobre `equal` a partir de curvas `f_1` com `AUROC = 0,56` e $\tau \approx 0{,}018$.
   - Assumir o Gate 1 + Diagnóstico Retrospectivo + Análise do Oráculo `{B0, eq, prop}` como **resultado científico negativo informativo** completo sobre acoplamento pós-hoc em `B0-rest` congelado: demonstrando por que `max-min` sobre margens quantílicas adaptativas por R11 falha e quantificando o teto estrutural de `O_3`, reservando eventual novo controlador conjunto apenas após resolver a calibragem/observabilidade de URLLC em experimento separado autorizado.
2. **Opção B — Gate 1b Condicional com Formulação Revisada `A5-v2` (se o orquestrador autorizar validar a correção algorítmica do seletor):**
   - Implementar e validar primeiro **100% em testes sintéticos** (§4) a especificação `A5-v2` abaixo (com sinal sintético calibrado e com sinal ruidoso), e **só então** decidir se há autorização para um mini-replay exploratório mínimo em dev (§5), exigindo explicitamente calibragem/substituição do critério de viabilidade URLLC (`f_1 >= 0` a $\tau=0{,}018$ é formalmente inválido como gate de viabilidade).

A Seção 2 especifica integralmente a **formulação candidata única revisada (`A5-v2`)**.

---

## 2. Formulação Matemática Revisada: Seletor Conjunto Orientado a Viabilidade e Proteção Condicional (`A5-v2`)

Abandona-se integralmente a comparação direta $\min_i (f_i - \beta\sigma_i)$ como prioridade absoluta. O seletor opera em **seis módulos explicitamente separados** sobre o estado causal pré-decisão $s_W$ em janela governada $W$ com orçamento físico $C \in \mathbb{Z}_{>0}$ (ex.: $C = 53$):

### 2.1 Escopo causal de demanda e referências candidatas admissíveis (sem acesso futuro)

Define-se o indicador causal de atividade de cada fatia $i \in \{0:\text{eMBB},\, 1:\text{URLLC},\, 2:\text{mMTC}\}$ idêntico ao já auditado (`causal_active_slices(s_W)`):
$$\text{act}_i(W) = \mathbb{I}\!\left[n_i(W) > 0 \;\lor\; q_i(W) > 0 \;\lor\; \text{pending}_i(W)\right], \qquad \mathcal{I}_{\text{act}}(W) = \{i \in \{0,1,2\} : \text{act}_i(W) = 1\}.$$
O conjunto de alocações válidas no subespaço causal é:
$$\mathcal{A}_{\text{act}}(W, C) = \left\{ a = (a_0, a_1, a_2) \in \mathbb{Z}_{\ge 0}^3 \;\middle|\; \sum_{i=0}^{2} a_i \le C,\; \forall i \notin \mathcal{I}_{\text{act}}(W): a_i = 0 \right\}.$$
*(Se $\mathcal{I}_{\text{act}}(W) = \emptyset$, retorna-se imediatamente $a = (0,0,0)$.)*

Define-se o **conjunto causal de referências candidatas** $\mathcal{R}(W) \subseteq \mathcal{A}_{\text{act}}(W, C)$ (calculado estritamente com estado disponível antes de $W$):
$$a_{\text{eq,act}}(W) = \text{equal\_active\_split}(C,\, \text{act}(W)), \qquad a_{\text{prop,act}}(W) = \text{prop\_active\_split}(s_W,\, C,\, \text{act}(W)),$$
$$a_{B0,\text{act}}(W) = \Pi_{\mathcal{A}_{\text{act}}(W,C)}\big(a_{B0,\text{exec}}(W)\big),$$
onde $\Pi_{\mathcal{A}_{\text{act}}}$ zera fatias inativas em $\mathcal{I}_{\text{act}}^c$ e aplica projeção inteira proporcional caso a soma exceda $C$. Em adição, quando $|\mathcal{I}_{\text{act}}(W)| = 3$, $a_{\text{eq,act}} = (18, 17, 18)$ para $C=53$. Nenhuma referência utiliza qualquer desfecho observado da janela $W$.

*(Nota de contorno para chegada latente em $W$, ex. `w2552`/`w1254`: quando a folga residual de capacidade existir após satisfazer $\mathcal{I}_{\text{act}}$, pode-se opcionalmente avaliar o candidato estático cego $(18,17,18)$ apenas como fallback se nenhuma ação ativa atingir segurança, preservando `0` desperdício inativo nos regimes normais `|act| < 3` como `w2071` e `w2685`.)*

### 2.2 Separação (i): Predição de satisfação de SLA (desacoplada de margens brutas heterogêneas e de $\tau \approx 0{,}02$)

Para cada fatia ativa $i \in \mathcal{I}_{\text{act}}$ e alocação candidata $a_i \in \{0,\dots,C\}$, o seletor consome um par de saída **explicitamente calibrado**:
$$\hat{m}_i(W, a_i) \in \mathbb{R} \quad \text{(margem mediana / centrada em } \Lambda_i = 0\text{)}, \qquad \phi_i(W, a_i) \in [0, 1] \quad \text{(escore de viabilidade estimada de SLA)}.$$

- **Restrição formal obrigatória de interface:** **é proibido** alimentar $\hat{m}_i$ diretamente com a saída bruta $f_i^{(\tau_i)}$ de um modelo adaptado por R11 com $\tau_i \le 0{,}05$ tratando-a como fronteira $\text{SLA OK} \iff f_i \ge 0$ (pois o Gate 1 provou $TPR_{\text{URLLC}} = 2{,}67\%$, viés `-0,4469`).
- A camada de predição fornece:
  1. Um preditor de margem centrada na mediana/limiar (ex.: regressão quantílica fixa a $\tau_{\text{feas}} = 0{,}50$ ou calibração isotônica/offset empírico fora da janela de avaliação $\hat{m}_i = f_i + \Delta_{\text{cal},i}$ quando $\tau_i < 0{,}50$, com fallback para razão calibrada de demanda em `eMBB` quando não calibrada);
  2. Um mapeamento **limitado e simétrico entre perfis heterogêneos** para comparação conjunta:
     $$\tilde{m}_i(W, a_i) = \operatorname{clip}\!\left(\hat{m}_i(W, a_i),\, -1,\, +1\right),$$
     impedindo que $+\infty$ no eMBB ou $-\infty$ no URLLC esmaguem a escala das demais slices;
  3. Um predicado de viabilidade discreta em três faixas com histerese $\delta_h > 0$ (ex.: $\delta_h = 0{,}05$):
     $$S_i(W, a_i) = \begin{cases} \text{SAFE}, & \hat{m}_i(W, a_i) \ge +\delta_h, \\ \text{MARGINAL}, & 0 \le \hat{m}_i(W, a_i) < +\delta_h, \\ \text{INFEASIBLE}, & \hat{m}_i(W, a_i) < 0. \end{cases}$$

### 2.3 Separação (ii) e (iii): Proteção anti-inanição condicional não unitária e restrição de capacidade

Em vez do piso ineficaz $a_i \ge 1$ (que produziu `0,49–0,98 Mbps` de eMBB em `8/8` estados `s2103`), define-se para cada fatia $i \in \mathcal{I}_{\text{act}}(W)$ um **piso condicional de adequação de serviço** $a_{\min,i}(W)$ ancorado nas referências causais e na demanda observada:
$$a_{\text{ref},i}(W) = \min\!\big(a_{\text{eq,act},i}(W),\, a_{\text{prop,act},i}(W)\big), \qquad a_{\min,i}(W) = \operatorname{floor}_{\le C}\!\Big(\lfloor \lambda_{\text{floor},i} \cdot a_{\text{ref},i}(W) \rfloor\Big),$$
sujeito à invariante estrita $\sum_{i \in \mathcal{I}_{\text{act}}} a_{\min,i}(W) \le C$:
- Caso $\sum_{i \in \mathcal{I}_{\text{act}}} \lfloor \lambda_{\text{floor},i} \cdot a_{\text{ref},i}(W) \rfloor > C$ (ex.: capacidade comprimida $C$ pequeno), aplica-se escalonamento proporcional inteiro para baixo preservando $\ge 1$ PRB por fatia ativa enquanto $|\mathcal{I}_{\text{act}}| \le C$.
- Para perfis orientados a taxa contínua (`eMBB`, $\lambda_{\text{floor},0} \in [0{,}5,\, 0{,}7]$ quando ativo), $C=53$ e $|\mathcal{I}_{\text{act}}|=3$ garantem piso mínimo significativo (ex.: $\ge 6\text{–}9$ PRBs com base em $a_{\text{ref},0} = 13\text{–}18$, nunca `0` ou `1` PRB!); para `URLLC` e `mMTC`, $\lambda_{\text{floor},i}$ impede zeramento total sem autorizar captura monopolística (`51–53` PRBs).
- O **subconjunto admissível anti-inanição** é:
  $$\mathcal{A}_{\text{adm}}(W, C) = \Big\{ a \in \mathcal{A}_{\text{act}}(W, C) \;\Big|\; \forall i \in \mathcal{I}_{\text{act}}(W): a_i \ge a_{\min,i}(W) \Big\} \;\cup\; \Big(\mathcal{R}(W) \cap \text{guarda ativa}\Big).$$

### 2.4 Separação (iv): Incerteza $\sigma_i$ apenas como filtro de suporte em camadas e desempate secundário (ZERO $-\beta\sigma$)

Remove-se **completamente** $-\beta\sigma_i$ da utilidade ($\beta_{\text{util}} \equiv 0$). A incerteza máxima entre fatias ativas:
$$\Sigma_{\text{act}}(W, a) = \max_{i \in \mathcal{I}_{\text{act}}(W)} \sigma_i(W, a_i)$$
atua exclusivamente em **dois papéis ortogonais**:
1. **Hierarquia de admissibilidade de suporte (com preservação estrita de não-inanição e não-degradação de Pareto):**
   - **Camada S1 (Confiável):** $\mathcal{A}_{S1} = \{a \in \mathcal{A}_{\text{adm}} : \Sigma_{\text{act}}(W, a) \le \sigma_{\max}\}$, aceita **somente** se for não vazia **e** contiver ao menos uma alocação que não deixe fatia ativa com benefício marginal positivo ociosa abaixo da fronteira de Pareto. Caso contrário, nunca bloqueia ações de alta capacidade apenas porque $\sigma > \sigma_{\max}$ (eliminando encolhimentos artificiais como `(37,13,0)`/`(0,0,41)` quando $53$ tem $\sigma > 0{,}15$!).
   - **Camada S2 (Suporte moderado):** $\mathcal{A}_{S2} = \{a \in \mathcal{A}_{\text{adm}} : \Sigma_{\text{act}}(W, a) \le \sigma_{\text{soft}}\}$.
   - **Camada S3 (Irrestrita admissível / Fallback causal):** $\mathcal{A}_{S3} = \mathcal{A}_{\text{adm}}$ contendo sempre as âncoras causais $\mathcal{R}(W)$.
2. **Desempate secundário de risco:** após empatar todos os critérios primários de viabilidade, déficit, âncora `B0` e eficiência de Pareto (§2.5), prefere-se menor $\Sigma_{\text{act}}(W, a)$.

### 2.5 Critério lexicográfico orientado a viabilidade, preservação e degradação controlada

Sobre o conjunto selecionado $\mathcal{A}^* \in \{\mathcal{A}_{S1}, \mathcal{A}_{S2}, \mathcal{A}_{S3}\}$ (em ordem de confiança):

1. **Nível 1 — Maximizar número de fatias ativas previstas viáveis (`Viabilidade & Preservação`):**
   $$N_{\text{sat}}(W, a) = \sum_{i \in \mathcal{I}_{\text{act}}(W)} \mathbb{I}\!\big[\hat{m}_i(W, a_i) \ge 0\big].$$
   *Efeito:* se URLLC é previsto inviável mesmo com muitos PRBs (`Contraexemplo #1`), $N_{\text{sat}}$ vale no máximo `2` quando `eMBB` e `mMTC` são preservados, e **`0` ou `1`** se `eMBB` for esmagado para tentar salvar `URLLC`. Logo, **é matematicamente impossível sacrificar serviços satisfazíveis para melhorar marginalmente uma fatia inviável**.
2. **Nível 2 — Degradação controlada entre fatias não satisfeitas (sem peso explosivo):**
   $$D(W, a) = \sum_{i \in \mathcal{I}_{\text{act}}(W)} \min\!\big(0,\, \tilde{m}_i(W, a_i)\big) \quad (\text{maximizar } D \in [-|\mathcal{I}_{\text{act}}|,\, 0]).$$
   Quando $N_{\text{sat}} < |\mathcal{I}_{\text{act}}|$ (nenhuma ação atinge viabilidade conjunta, **Requisito (vi)**), maximizar $D(W, a)$ sobre $\mathcal{A}_{\text{adm}}$ reduz a severidade total de violação de forma limitada ($[-1, 0]$ por fatia) mantendo todas as fatias protegidas pelo piso $a_{\min,i}(W)$. Se houver desempate dentro da tolerância numérica $\epsilon_D$, prefere-se o candidato com maior margem média $\frac{1}{|\mathcal{I}_{\text{act}}|}\sum \tilde{m}_i$.
3. **Nível 3 — Guarda conservadora de âncora `B0-rest` (`Contraexemplo #7`):**
   Se $a_{B0,\text{act}} \in \mathcal{A}_{\text{adm}}$ e atinge $N_{\text{sat}}(a_{B0,\text{act}}) = \max_{a \in \mathcal{A}^*} N_{\text{sat}}(a)$ (por ex., todas as fatias ativas já estão `SAFE` em $B_0$), o seletor **retém $a_{B0,\text{act}}$** salvo se outra candidata superar estritamente $a_{B0,\text{act}}$ em $N_{\text{sat}}$ **ou** produzir ganho líquido de déficit/margem limitada acima de uma banda morta de histerese $\Delta_{\text{switch}} > 0$ **e** respeitar confiança $\Sigma_{\text{act}} \le \sigma_{\max}$.
4. **Nível 4 — Eficiência de Pareto justificada por demanda e benefício marginal (§4 — sem forçar $\sum a_i = 53$ cegamente):**
   Define-se a utilidade de sobra benéfica por:
   $$B_{\text{marg}}(W, a) = \sum_{i \in \mathcal{I}_{\text{act}}(W)} \Big( \operatorname{clip}(\hat{m}_i(W, a_i), 0, M_{\text{cap}}) + \epsilon_{\text{head}} \cdot \mathbb{I}[\Delta^+ \hat{m}_i(W, a_i) > \varepsilon_m \lor q_i(W) > q_{\text{sat},i}] \cdot \frac{a_i}{C} \Big),$$
   onde $\Delta^+ \hat{m}_i(W, a_i) = \hat{m}_i(W, a_i + 1) - \hat{m}_i(W, a_i)$ e $\epsilon_{\text{head}} > 0$ é infinitesimal ($10^{-3}$).
   - **Sem $-\beta\sigma$ e sem $-\gamma a_i/C$ na utilidade:** aumentar $a_i$ nunca é penalizado.
   - **Quando $\sum a_i < C$ é permitido e justificado:** se todas as fatias ativas já estão em platô sem benefício marginal ($\Delta^+ \hat{m}_i \le \varepsilon_m$ e sem fila ativa pendente além do já atendido, p. ex. `1` fatia `mMTC` com `237 B` como em `w2685`/`w2731`), alocar menos de `53` PRBs é explícito e ótimo pelo desempate final de parcimônia (evita desperdício cego sem necessidade de empurrar `53` PRBs para 1 pacote `mMTC`).
   - **Quando $\sum a_i < C$ é proibido:** sempre que houver fatia ativa com demanda contínua/fila (`eMBB` com $q_0 > 0$ ou $\Delta^+ \hat{m}_i > \varepsilon_m$), adicionar PRB aumenta $B_{\text{marg}}$, proibindo colapsos como `(6,13,1)` (`33` ociosos), `(1,13,1)` (`38` ociosos) e `(6,19,0)` (`28` ociosos).
5. **Nível 5 — Desempate final de risco/ancoragem e fallback explícito:**
   Entre empates residuais dentro de $\epsilon_{\text{tie}}$ nos Níveis 1–4: (a) minimizar $\Sigma_{\text{act}}(W, a)$; (b) menor distância $\ell_1$ à âncora causal preferida ($a_{\text{prop,act}}$ ou $a_{\text{eq,act}}$); (c) se todo o modelo preditivo estiver indisponível/não confiável ($\Sigma_{\text{act}} > \sigma_{\text{soft}}$ em toda a grade ou $n_{\text{support}} < n_{\min}$), **fallback determinístico explícito** para $a_{\text{fallback}}(W) = a_{\text{prop,act}}(W)$ (com registro `fallback_used = True` e `fallback_reason`).

---

## 3. Verificação Explícita Contra Todos os 7 Contraexemplos Obrigatórios (§5)

| # | Contraexemplo exigido (§5) | Testemunha / falha no Gate 1 anterior | Comportamento formal da `A5-v2` | Invariante garantida |
|---:|---|---|---|---|
| **1** | **URLLC previsto inviável mesmo com muitos PRBs** | `lam5_s2103_w1241`: `(0,53,0)` falhou `[u]` (`10,15ms`); `s2103` `act3`: `f_1 < 0` em `[0..53]`, drenou `(0,53,0)`/`(1,51,1)` falhando `[e]` em `8/8`. | Nível 1 (`max N_sat`) reconhece que URLLC não muda viabilidade ao subir de `18` para `51` (`0` ganho em $N_{\text{sat}}$), enquanto preservar `eMBB` e `mMTC` assegura $N_{\text{sat}} = 2 > 1$. Nível 2 limita perda do URLLC em $[-1, 0]$ sem violar o piso condicional de `eMBB`/`mMTC`. | Nunca sacrifica fatias ativas satisfazíveis por tentativa fútil de salvar fatia inviável; URLLC fica limitado sem esmagar `eMBB`. |
| **2** | **Risco de inanição de eMBB por captura de PRBs por URLLC** | `b1_act` deu `a_e = 0` em `9/18` Univ A (`0,0–0,61 Mbps`); `b1_sup` deu `a_e = 1` em `9/18` (`0,49–0,98 Mbps << 3,5 Mbps`). | Piso condicional $a_{\min,0}(W) = \lfloor \lambda_{\text{floor},0} \cdot \min(a_{\text{eq},0}, a_{\text{prop},0}) \rfloor$ garante reserva proporcional real (ex. `~7–9` RBGs em 3-ativas `C=53`, nunca `1` PRB simbólico), além da proteção de $N_{\text{sat}}$ no Nível 1. | $a_e \ge a_{\min,0}(W) \gg 1$ sempre que `eMBB` ativa e $C$ suportar; impossível `(0,53,0)` ou `(1,51,1)` com 3 ativas. |
| **3** | **Três slices simultaneamente ativas (`n_act = 3`)** | Universo A (`18` estados): `b1_sup = 2/18 (11,1%)`, `Harm = 7/8`. | $\mathcal{A}_{\text{adm}}$ impõe pisos proporcionais às 3 fatias; $N_{\text{sat}}$ busca viabilidade conjunta (`3/3`), preservando $B_0$ seguro pelo Nível 3 e usando $\mathcal{R}(W) = \{a_{B0}, a_{\text{eq,act}}, a_{\text{prop,act}}\}$ como âncoras admissíveis. | Todos os 3 pisos ativos respeitados, $\sum a_i \le C$, sem drenagem unilateral. |
| **4** | **Uma ou duas slices ativas (`n_act in {1, 2}`)** | Universo B (`5` estados): `equal` desperdiçou `212` PRBs; `b1_raw` falhou `w2071 [T,F,T]` (`14` p/ URLLC vazia); `w2442 [T,T,F]` colapsou `(6,19,0)`; contraponto `w2552/w1254` (chegada latente dentro de `W`). | $\text{act}_i(W)=0 \implies a_i=0$ elimina desperdício em slices vazias (`w2071 -> >=45` p/ eMBB, `w2685 -> a_u=a_e=0`), mantendo `sum a_i <= C` e piso não nulo só para ativas. Risco residual de chegada latente em `W` (`2/25` no Gate 1) é explicitado na camada de escopo causal. | Zero PRBs desperdiçados em fatias causalmente inativas; 100% da folga útil disponível às `1` ou `2` fatias ativas. |
| **5** | **Conjunto confiável vazio ou insuficiente ($\mathcal{A}_{S1} = \emptyset$ ou ralo)** | Se $\sigma_{\max}=0{,}15$ filtra alta alocação (`w2442`: `(38,15,0)` tinha $\sigma_0=0{,}153$), encolhia para `(37,13,0)` ou `beta=1` encolhia para `(6,19,0)`. | Camada S1 só restringe se não violar não-inanição nem gerar deficiência de Pareto; senão relaxa automaticamente para S2 ($\sigma_{\text{soft}}$) e S3 ($\mathcal{A}_{\text{adm}}$), caindo em fallback explícito para $a_{\text{prop,act}}$ se toda a grade exceder $\sigma_{\text{soft}}$. | Seleção sempre definida, determinística, dentro do orçamento e sem colapso por escassez de suporte. |
| **6** | **Predições pessimistas ou discrepância de escala entre slices** | `B0-rest` $\tau_1 \approx 0{,}018$ (`f_1` viés `-0,45`, `max f_1 < 0` em `85,7%`), `eMBB in [-1, +3]` vs `mMTC <= +0,0526` vs `URLLC in (-inf, +0,19]`. | Separação estrita: (a) $\hat{m}_i$ exige calibração centrada no limiar (rejeitando `tau<=0.05` não calibrado como gate de viabilidade), (b) $\tilde{m}_i = \operatorname{clip}(\hat{m}_i, -1, +1)$ equaliza escalas antes de somar $D(W, a)$, e (c) Nível 1 conta satisfações discretas em vez de minimizar $\min_i \hat{m}_i$. | Escalas heterogêneas e pessimismo extremo em cauda não distorcem pesos nem invertem prioridade de serviços viáveis. |
| **7** | **`B0-rest` já seguro onde desviar traz risco** | `7/8` estados seguros do `B0` prejudicados (`w1890`, `w1588`, `w1621`, `w1406`, `w2229`, `w1378`, `w1361`). | Nível 3 retém $a_{B0,\text{act}}$ sempre que já é maximizador de $N_{\text{sat}}$ com $\Sigma_{\text{act}}(a_{B0,\text{act}}) \le \sigma_{\max}$, exigindo benefício líquido $\ge \Delta_{\text{switch}}$ e risco controlado $\sigma \le \sigma_{\max}$ para justificar qualquer desvio. | Zero dano gratuito sobre estados já seguros do `B0-rest` quando o modelo não tem evidência forte e segura de ganho. |

---

## 4. Matriz de Testes Sintéticos e Critérios GO/NO-GO Pré-Gate (Item 4)

Qualquer eventual implementação futura da `A5-v2` (somente após autorização) **deve** passar integralmente por esta matriz determinística **antes** de executar qualquer simulação no `ns-3`:

| ID Teste | Cenário sintético | Verificação obrigatória (`assert`) | Critério GO |
|---|---|---|---|
| `T1_CX1_URLLC_INFEAS` | `act=[T,T,T]`, $\hat m_1(a_1) \in [-0{,}60, -0{,}25]$ para todo $a_1 \in [0..53]$; `eMBB` viável p/ $a_0 \ge 16$; `mMTC` p/ $a_2 \ge 2$. | $a_0 \ge 16, a_2 \ge 2, N_{\text{sat}} = 2$; nunca escolhe $(0,53,0)$ nem $(1,51,1)$. | Passa 100% |
| `T2_CX2_NO_EMBB_STARVE` | `URLLC` negativa com inclinação rasa $+0{,}001$/PRB; `eMBB` negativa até $a_0=14$, depois $+0{,}05$/PRB. | $a_0 \ge a_{\min,0}(W) \gg 1$ e nunca cai abaixo do piso condicional. | Passa 100% |
| `T3_CX3_ALL3_FEAS` | `act=[T,T,T]` com região conjunta segura; `a_{eq}` vs `a_{prop}` vs ótimo interno. | Escolhe ação em $\mathcal{A}_{\text{adm}}$ com $N_{\text{sat}}=3$ e respeita todos pisos. | Passa 100% |
| `T4_CX4_1OR2_ACTIVE` | Subcasos `[F,F,T]`, `[T,T,F]`, `[T,F,T]` e `[F,F,F]`. | Fatias inativas recebem `0` PRB; `[F,F,F]` retorna `(0,0,0)`; `[T,F,T]` entrega sobra à ativa sem desperdiçar na inativa. | Passa 100% |
| `T5_CX5_EMPTY_SUPPORT` | (a) $\sigma > 0{,}15$ para $a > 25$ quando demanda pede $40$; (b) $\sigma > 0{,}55$ em toda a grade. | (a) Não encolhe capacidade útil artificialmente; (b) cai em fallback determinístico para $a_{\text{prop,act}}$ com `fallback_used=True`. | Passa 100% |
| `T6_CX6_SCALE_CLIP` | `eMBB` com $\hat m_0 \in [-1, +3]$, `mMTC` limitada em $+0{,}05$, `URLLC` com cauda $-5{,}0$. | Nível 1 não compara margens brutas entre si; Nível 2 usa $\tilde m_i \in [-1, +1]$ sem explosão por `-5.0`. | Passa 100% |
| `T7_CX7_B0_ANCHOR_KEEP` | `a_{B0,act}` já seguro ($N_{\text{sat}}=3$), alternativa tem $\Delta < \Delta_{\text{switch}}$ ou $\sigma > \sigma_{\max}$. | Retém exatamente `a_{B0,act}` sem regressão gratuita. | Passa 100% |
| `T8_IDLE_JUSTIFIED_VS_FORBIDDEN` | (a) `[F,F,T]` mMTC saturada ($\Delta^+ \hat m_2 = 0$ p/ $a_2 \ge 12$, $q_0=q_1=0$); (b) `[T,T,T]` `eMBB` com $q_0 \gg 0$ e $\Delta^+ \hat m_0 > \varepsilon_m$, mesmo com $\sigma_1$ menor em `a_u=13`. | (a) Pode parar abaixo de `53` sem forçar lixo; (b) **proibido** deixar folga ociosa tipo `(6,13,1)`/`(1,13,1)`/`(6,19,0)`. | Passa 100% |
| `T9_INVARIANT_NO_BETA_SIGMA` | Aumentar $\sigma_i(a)$ monotonicamente com $a$ enquanto $\hat m_i$ é constante/crescente. | Ação selecionada nunca encolhe só para diminuir $\sigma$ às custas de $N_{\text{sat}}$, $D$ ou $B_{\text{marg}}$. | Passa 100% |
| `T10_CALIBRATION_CHECK` | Preditor de entrada com $\tau \le 0{,}05$ não calibrado. | Camada de contrato/calibração recusa operar sem calibração explícita ou aplica mapeamento documentado (zero uso silencioso de $\tau=0{,}018$ como $P(\text{SLA})\ge 0{,}5$). | Passa 100% |

**Critérios GO/NO-GO para autorizar eventual Mini-Replay (Gate 1b):**
1. **GO sintético obrigatório:** `10/10` testes sintéticos acima passando (`100%`), inclusive com curvas extraídas dos snapshots reais do Gate 1 (sem tocar em `ns-3`).
2. **GO de sinal causal prévio (condição bloqueante para Opção B):** demonstrar nos dados congelados (ou em calibração fora da amostra de teste) que a camada de predição $\hat{m}_i$ **melhora** a separação de URLLC além de `AUROC = 0,56` (ou, alternativamente, que uma política ancorada em `{B0, eq_act, prop_act}` com seleção guiada por `eMBB`/demanda supera `equal = 12/18` nos traces existentes sem acesso a resultado futuro). **Se nem essa pré-condição for atingida, aplicar diretamente a Opção A (Escopo Científico Enxuto) = `NO-GO` para novos replays no `B0-rest`.**

---

## 5. Proposta Mínima de Mini-Replay em Desenvolvimento (Aguardando Autorização Separada — NÃO EXECUTADA)

Caso o orquestrador aprove a **Opção B** após aprovação integral da Seção 4:
- **Escopo estritamente restrito a dev:** **somente** seeds `2102` e `2103` (`λ ∈ {2, 5, 6}`), **proibindo** `3101–3103`, `Gate 2`, `D3-v2`, `PILOT` e `MAIN`.
- **Subconjunto mínimo de estados (`N = 25` idêntico ao plano pré-registrado [`A5_GATE1_SAMPLING_PLAN.csv`](../../../ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_SAMPLING_PLAN.csv) `sha256=771b9a2c...`, reusando 100% dos replays `b0/eq/prop` já executados!):**
  - Como `b0`, `eq` e `prop` **já possuem replays validados** nos `25` estados, avaliar `A5-v2` (apenas 1 braço novo + eventual 1 ablação de calibração = **`25` a `50` replays únicos** no máximo, dedupando empates exatos com `{b0, eq, prop}`!).
  - Alternativa ainda menor de fumaça inicial (**Gate 1b-mini, `N = 12` estados**): os `9` estados de discordância `{b0, eq, prop}` do Universo A (`w1890, w2667, w1621, w1546, w1560, w2229, w1283, w1397, w1254`) + `2` estados com colapso ocioso em `s2102` (`w1406, w1396`) + `1` estado crítico do Universo B (`w2442`) = **`<= 12` novos replays** (~3–5 min com `WORKERS=4`), com parada imediata (`STOP-EARLY`) se:
    - qualquer estado em `w1406/w1396/w2442` apresentar `sum(a) < 53` injustificado,
    - `HarmRate` sobre `{w1890, w1621, w1406, w2229}` exceder `1/4` (`25%`), ou
    - taxa conjunta em Universo A ficar $\le$ `equal` (`12/18`) sem vantagem clara em Universo B (`0` desperdício inativo).
- **Critério de aprovação pós-mini-replay para só então discutir `D3`/`Gate 2`:** superar `B0-rest` (`>= 44,4%`) **e** igualar ou superar `equal` (`>= 66,7%`) no Universo A mantendo `HarmRate <= 12,5%` (`<= 1/8`), `0%` de colapsos ociosos injustificados e `0` desperdício inativo no Universo B.
