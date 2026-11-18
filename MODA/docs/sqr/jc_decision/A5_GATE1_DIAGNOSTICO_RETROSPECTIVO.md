# Diagnóstico Retrospectivo do Gate 1 A5 e Análise do Oráculo Restrito (Sem Novos Replays)

**Referências congeladas e fontes exclusivas deste relatório:**
- Commit de preservação do resultado negativo do Gate 1: `3afe72c` ([`A5_GATE1_RELATORIO_EXPLORATORIO.md`](A5_GATE1_RELATORIO_EXPLORATORIO.md))
- Tabela e agregados do Gate 1: `A5_GATE1_METRICS.csv` (`150` linhas lógicas, `127` replays únicos válidos, `25/25` paridade `A/A` `B0-rest`), `A5_GATE1_DIAGNOSTICS.json`, `A5_GATE1_DECISIONS.json`
- Script e tabelas geradas estritamente sobre artefatos existentes (`new_replays_executed = 0`): [`gate1_retrospective_analysis.py`](../../../ns-o-ran-gym/results/sqr_jc_decision/gate1_retrospective_analysis.py), `A5_GATE1_RETROSPECTIVE_ANALYSIS.json` e `A5_GATE1_RETROSPECTIVE_TABLES.csv`
- Seletor avaliado preservado intocado (`29989f1`): [`a5_joint_selector.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/a5_joint_selector.py)

> **Regra de integridade:** **nenhum** novo replay no `ns-3`, **nenhum** ajuste de hiperparâmetros e **nenhuma** alteração no `B0-rest` congelado (`8375f4d`, `17/17` SHA-256 intactos) foram realizados nesta análise. Todas as comparações contrafactuais $\beta = 0$ além das ações já executadas são explicitamente marcadas como *checagens offline de trace (não executadas no `ns-3`)*.

---

## 1. Resumo Executivo das Descobertas Empíricas

1. **Separação causal dos dois mecanismos de falha do Gate 1:**
   - **(a) Dominância de escala do URLLC no `max-min` (`B1-raw` e `B1-active`, que na implementação `29989f1` já rodam sem $-\beta\sigma$):** mesmo com $\beta = 0$, `B1-active` (`g = f`, sem filtro $\sigma_{\max}$) sofreu colapso no **Universo A** (`2/18 = 11,1%`, `Harm = 7/8`, `Net = -6/18`), drenando recursos para URLLC (`bottleneck_slice = 1` em `13/18` estados do Universo A e `16/25` no total) e zerando eMBB (`a_e <= 1`) em **`10/18` (`55,6%`)** estados do Universo A (`(0,53,0)` em todos os `8/8` estados ativos da `s2103` + `lam6_s2102_w1396` + `lam6_s2103_w1254`). Na checagem offline `off_sup_beta0` (`active=T, floor=1, sigma<=0.15, beta=0`), eliminar $-\beta\sigma$ manteve exatamente **`10/18`** estados do Universo A com `a_e <= 1` (`14/18` casaram ações já executadas no Gate 1, com apenas **`2/14` `PASS`**). Logo, a dominância do URLLC no `max-min` destrói o seletor **mesmo sem $-\beta\sigma$**.
   - **(b) Penalização $-\beta\sigma$ (`beta = 1.0` no `B1-support`) combinada a $u_i = g_i - \gamma a_i/C$ (`gamma=1.0`) e desempate `min sum(a)`:** é a responsável específica pelos colapsos ociosos com `sum(a) < 53` em `6/25` (`24%`) estados (`153` RBGs ociosos no total; contra `1/25` no `B1-raw` e `2/25` no `B1-active`):
     - `lam6_s2102_w1406`: `(6, 13, 1)` $\to$ **33 recursos não alocados** (contra `(8, 44, 1)` com `0` ociosos quando $\beta=0$);
     - `lam6_s2102_w1396`: `(1, 13, 1)` $\to$ **38 recursos não alocados** (contra `(1, 51, 1)` quando $\beta=0$);
     - `lam2_s2102_w2442` (Universo B): `(6, 19, 0):FAIL[e]` $\to$ **28 recursos não alocados** (contra `(37, 13, 0)` com `3` ociosos quando $\beta=0$ — sendo este o único dano adicional do `B1-support` em `B0=PASS` no Universo B);
     - além de `(45, 0, 1)` (`7` ociosos em `lam6_s2102_w2071`, vs `(52,0,1)` quando $\beta=0$), `(0, 0, 29)` (`24` ociosos em `w2685`, vs `(0,0,41)`) e `(0, 0, 30)` (`23` ociosos em `w2731`, vs `(0,0,43)`).
2. **Descolamento severo de calibragem e baixo sinal preditivo pré-decisão no URLLC:**
   - Nos `25/25` snapshots do `B0-rest`, a heurística dinâmica R11 manteve $\tau_i \le 0{,}05$ em todas as fatias (**eMBB:** mediana `0,040`, média `0,0327`; **URLLC:** mediana `0,015`, média `0,0180`; **mMTC:** mediana `0,035`, média `0,0300`). Assim, `f_i(a_i)` estima a cauda extrema de `1%–5%` pior caso, **não** a probabilidade ou mediana de atendimento de SLA.
   - Nas `93` combinações únicas `(state_id, slice=1, a_u)` com desfecho conhecido (`Universo C` e janelas vacuous excluídos): o URLLC atingiu SLA em **`80,65%` (`75/93`)**, porém `f_1 >= 0` previu viabilidade em apenas **`2,15%` (`2/93`)** (`TPR = 2,67%`, `Acurácia = 21,51%`, viés médio `-0,4469`). Mais grave: mesmo de forma **livre de limiar**, **`AUROC(f_1) = 0,5607`** (`Spearman rho = 0,1168`) e `sigma_1` tem correlação **invertida** com o erro absoluto (`rho(sigma_1, |f_1 - m_1|) = -0,2614`, com `sigma_1` médio maior nos acertos `0,0192` que nas falhas `0,0087`).
   - Em contraste, `eMBB` (`AUROC = 0,7283`, `rho = 0,3596`, `sigma_0` positivamente associado ao erro `rho = +0,2588`) carrega sinal real moderado, enquanto `mMTC` já passa em **`95,05%` (`96/101`)** das observações (`AUROC = 0,8187`).
3. **Teto apertado do Oráculo Restrito `{B0-rest, equal, prop}` (limite superior retrospectivo não causal):**
   - No **Universo A (`N=18`)**, o oráculo não causal `O_3 = {B0, eq, prop}` alcança **`14/18 (77,8%)`** (`Rescue = 6/10`, `Harm = 0/8`), apenas **`+2/18` (`+11,1 pp`)** acima do baseline cego estático **`equal = (18, 17, 18)` (`12/18 = 66,7%`)** e `+3/18` acima de `prop` (`11/18 = 61,1%`). `O_6` (todos os 6 braços) soma só mais 1 estado (`15/18 = 83,3%`, exclusivamente `lam5_s2102_w2552` via `b1_raw=(47,6,0)`).
   - No **Universo B (`N=5`)**, `O_3` atinge **`3/5 (60,0%)`** — **empate exato** com `equal` (`3/5`) e `prop` (`3/5`), embora `equal` desperdice `212` PRBs em slices inativas em `5/5` estados do Universo B (`9/25` no total) enquanto `prop` tem **`0` desperdício inativo**. `O_6` chega a `4/5 (80,0%)` igualando `b1_act` (que falhou apenas em `lam5_s2103_w1241`, onde até `(0,53,0)` falhou URLLC).
   - Nenhuma regra puramente causal avaliada retrospectivamente sobre `{b0, eq, prop}` usando `f_i` e `sigma_i` pré-decisão superou `equal` (`12/18 = 66,7%`).

---

## 2. Decomposição Causal: Dominância URLLC (`max-min`) vs Penalização $-\beta\sigma$

Inspeção direta do código congelado de [`a5_joint_selector.py`](../../../ns-o-ran-gym/src/nsoran/sqr_jc/a5_joint_selector.py) confirma um fato metodológico crucial que facilita a separação exata:
- **`b1_raw`** (linhas 509–528) e **`b1_act`** (`variant="b1_active"`, linhas 549–568) **não aplicam $-\beta\sigma$ nem filtro $\sigma_{\max}$** (`g_tbl = f_arr`), usando `u_tbl = f_arr - gamma * a/C` ($\gamma = 1{,}0$) e desempate #3 `min sum(a)`. A única diferença entre `b1_raw` e `b1_act` é o mascaramento de slices causalmente inativas (`causal_active_slices`).
- **`b1_sup`** (`variant="b1_support"`, linhas 570–660) acrescenta simultaneamente: (i) $g_i = f_i - 1{,}0\cdot\sigma_i$, (ii) $u_{\text{supp}} = g_i - 1{,}0\cdot a_i/C$, (iii) piso `a_i >= 1` para slices com `live_floor_mask[i]=True` e (iv) restrição de suporte `max sigma_i <= 0.15`.

### 2.1 Comparação executada + checagem offline $\beta = 0$ (`25` estados)

| Configuração / Variante | Modo de Execução | $\beta$ | Filtro $\sigma_{\max}$ | Piso `1` | Ativas | `sum < 53` (`25` est.) | Ociosos Totais | `bn = URLLC` (`25`) | `a_e <= 1` Univ A (`18`) | Univ A `PASS` | Univ B `PASS` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`b1_raw`** | Replay `ns-3` | `0` | Não | Não | Todas `3` | `1/25` | `11` | `17/25` | **`10/18`** | `3/18 (16,7%)` | `2/5 (40,0%)` |
| **`b1_act`** | Replay `ns-3` | `0` | Não | Não | Causais | `2/25` | `22` | `16/25` | **`10/18`** | `2/18 (11,1%)` | **`4/5 (80,0%)`** |
| **`b1_sup`** | Replay `ns-3` | `1.0` | `0.15` | `>= 1` | Causais | **`6/25`** | **`153`** | **`20/25`** | **`10/18`** | `2/18 (11,1%)` | `3/5 (60,0%)` |
| `off_act_beta0` | *Offline trace (`eq` a `b1_act`)* | `0` | Não | Não | Causais | `2/25` | — | `16/25` | `10/18` | `2/18` | `4/5` |
| `off_sup_beta0` | *Offline trace (14/18 casam replays)* | `0` | `0.15` | `>= 1` | Causais | `3/25` (só Univ B) | — | `18/25` | **`10/18`** | `2/14` nos replays casados | `2/2` nos replays casados |

### 2.2 Por que `(6, 13, 1)`, `(1, 13, 1)` e `(6, 19, 0)` ocorreram com $C = 53$

Em `B1-support`, a combinação de:
1. $f_1(a_u)$ achatada sob concorrência (ganho médio total $f_1(53) - f_1(0) = +0{,}176$, com `85,7%` dos snapshots ativos tendo $\max_a f_1(a) < 0$),
2. $g_1(a_u) = f_1(a_u) - 1{,}0\cdot\sigma_1(a_u)$ criando máximos locais artificiais próximos aos valores mais visitados pelo `B0-rest` (onde $\sigma_1 \approx 0{,}0005$, enquanto $\sigma_1$ sobe para `0,10–0,15` em `a_u = 44..53`),
3. uma vez que `g_1` trava o gargalo `min(g_0, g_1, g_2)` num pico interno baixo (`a_u = 13` ou `19`), aumentar `a_e` na fatia eMBB **não altera `min(g_i)`**, mas **reduz $u_{\text{supp}} = \sum (g_i - a_i/53)$** (penalidade $\gamma/53 = -0{,}0189$/RBG) e **perde o desempate #3 `min sum(a)`**:

| Estado | Ação `b1_act` ($\beta=0$, sem piso) | Ação `off_sup_beta0` ($\beta=0$, `sigma<=0.15`, piso `1`) | Ação `b1_sup` ($\beta=1.0$, `sigma<=0.15`, piso `1`) | Recursos não alocados ($C=53$) | Mecanismo exato em `b1_sup` |
|---|---|---|---:|---:|---|
| `lam6_s2102_w1406` (`[T,T,T]`) | `(8, 45, 0):FAIL[e]` | `(8, 44, 1)` (`0` ociosos) | **`(6, 13, 1):FAIL[e]`** | **`33`** | `a_u=13` ($\sigma_1=0{,}0005, g_1=-0{,}1819$) vence `a_u=44` ($\sigma_1 \approx 0{,}11$); `eMBB` fecha em `a_e=6` ($g_0=-0{,}1778 > g_1$) e `gamma=1` + `min sum(a)` descartam os `33` RBGs restantes (`2,44 Mbps`). |
| `lam6_s2102_w1396` (`[T,T,T]`) | `(0, 53, 0):FAIL[e]` | `(1, 51, 1)` (`0` ociosos) | **`(1, 13, 1):FAIL[e]`** | **`38`** | Mesmo efeito na `s1` (`a_u=13`, $g_1=-0{,}1479$); `a_e=1` já supera $g_1$ e o desempate descarta `38` RBGs (`0,49 Mbps`). |
| `lam2_s2102_w2442` (`[T,T,F]`) | `(38, 15, 0):PASS` | `(37, 13, 0)` (`3` ociosos por $\sigma \le 0{,}15$) | **`(6, 19, 0):FAIL[e]`** | **`28`** | $\sigma_1$ desloca gargalo para $g_1(19)=-0{,}1507$; $g_0(6)=-0{,}1442 > -0{,}1507$, derrubando `eMBB` de `16,38 Mbps` (`38` RBGs) para `2,57 Mbps` (`6` RBGs) — dano único de `b1_sup` no Universo B. |
| `lam6_s2102_w2071` (`[T,F,T]`) | `(53, 0, 0):PASS` | `(52, 0, 1)` (`0` ociosos) | **`(45, 0, 1):PASS`** | `7` | $-\beta\sigma_0$ limita `a_e=45` (`53` tinha $\sigma_0=0{,}0975$). |
| `lam2_s2102_w2685` (`[F,F,T]`) | `(0, 0, 41):PASS` | `(0, 0, 41):PASS` (`12` ociosos por $\sigma \le 0{,}15$) | **`(0, 0, 29):PASS`** | `24` | Única slice ativa `mMTC` (`237 B`); $-\beta\sigma_2$ encolhe `41 -> 29`. |
| `lam2_s2102_w2731` (`[F,F,T]`) | `(0, 0, 43):PASS` | `(0, 0, 43):PASS` (`10` ociosos por $\sigma \le 0{,}15$) | **`(0, 0, 30):PASS`** | `23` | Mesmo perfil de demanda única `mMTC`; $-\beta\sigma_2$ encolhe `43 -> 30`. |

Além disso, a comparação entre `off_act_beta0` e `b1_sup` prova que o **piso unitário (`a_i >= 1`) é completamente ineficiente para proteger SLA de slices ativas**: nos `8/8` estados com 3 slices ativas da seed `2103`, `(0,53,0)` (`b1_act`) deu `eMBB = 0,0–0,61 Mbps` e `(1,51,1)` (`b1_sup`) deu `eMBB = 0,49–0,98 Mbps`, falhando `[e]` em **`8/8` (`100%`)** frente ao SLA de `3,5 Mbps`.

---

## 3. Atribuição Completa Estado a Estado: Ações, SLAs Prejudicadas, Preservadas e Resgatadas

### 3.1 Universo A (`ALL3_COMPLETE`, `N=18`, `STRICT_ALL3_R19`; `B0 PASS=8`, `B0 FAIL=10`)

| `state_id` | $\lambda$ | Seed | `act_mask` | `B0-rest` | `equal=(18,17,18)` | `prop` | `B1-raw` | `B1-act` | `B1-sup` |
|---|---:|---:|---|---|---|---|---|---|---|
| `lam5_s2102_w1654` | 5 | 2102 | `[T,T,T]` | `(10,25,7)` **PASS** | **PASS** | `(11,10,32)` **PASS** | `(12,31,10)` **PASS** | `(12,31,10)` **PASS** | `(12,31,10)` **PASS** |
| `lam5_s2102_w1890` | 5 | 2102 | `[T,T,T]` | `(32,17,4)` **PASS** | FAIL`[e]` (`2,57`) | `(20,7,26)` FAIL`[e]` (`2,81`) | `(17,36,0)` FAIL`[e]` (`2,44`) | `(17,36,0)` FAIL`[e]` (`2,44`) | `(18,34,1)` FAIL`[e]` (`2,57`) |
| `lam5_s2103_w1588` | 5 | 2103 | `[T,T,T]` | `(28,13,7)` **PASS** | **PASS** | `(13,27,13)` **PASS** | `(0,53,0)` FAIL`[e]` (`0,00`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(1,51,1)` FAIL`[e]` (`0,49`) |
| `lam5_s2103_w1621` | 5 | 2103 | `[T,T,T]` | `(27,13,13)` **PASS** | FAIL`[u]` (`10,10ms`) | `(13,27,13)` **PASS** | `(0,53,0)` FAIL`[e]` (`0,61`) | `(0,53,0)` FAIL`[e]` (`0,61`) | `(1,51,1)` FAIL`[e]` (`0,98`) |
| `lam6_s2102_w1406` | 6 | 2102 | `[T,T,T]` | `(20,23,1)` **PASS** | **PASS** | `(11,10,32)` **PASS** | `(8,45,0)` FAIL`[e]` (`3,30`) | `(8,45,0)` FAIL`[e]` (`3,30`) | `(6,13,1)` FAIL`[e]` (`2,44`) |
| `lam6_s2102_w2229` | 6 | 2102 | `[T,T,T]` | `(21,11,21)` **PASS** | **PASS** | `(15,8,30)` FAIL`[e]` (`3,18`) | `(33,20,0)` FAIL`[u+m]` (`10,15ms;0,50`) | `(33,20,0)` FAIL`[u+m]` (`10,15ms;0,50`) | `(33,19,1)` FAIL`[u]` (`10,15ms`) |
| `lam6_s2103_w1378` | 6 | 2103 | `[T,T,T]` | `(29,9,0)` **PASS** | **PASS** | `(13,27,13)` **PASS** | `(0,53,0)` FAIL`[e]` (`0,00`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(1,51,1)` FAIL`[e]` (`0,49`) |
| `lam6_s2103_w1361` | 6 | 2103 | `[T,T,T]` | `(20,28,5)` **PASS** | **PASS** | `(13,27,13)` **PASS** | `(0,53,0)` FAIL`[e+m]` (`0,12;0,00`) | `(0,53,0)` FAIL`[e+m]` (`0,12;0,00`) | `(1,51,1)` FAIL`[e]` (`0,49`) |
| `lam5_s2102_w2667` | 5 | 2102 | `[T,T,T]` | `(7,33,4)` FAIL`[e]` (`1,53`) | **PASS** (`3,91`) | `(18,9,26)` **PASS** (`3,91`) | `(26,27,0)` **PASS** (`5,62`) | `(26,27,0)` **PASS** (`5,62`) | `(26,26,1)` **PASS** (`5,62`) |
| `lam5_s2102_w2578` | 5 | 2102 | `[T,T,T]` | `(27,20,6)` FAIL`[e]` (`3,30`) | FAIL`[e]` (`2,44`) | `(23,7,23)` FAIL`[e]` (`2,93`) | `(28,25,0)` FAIL`[e]` (`3,42`) | `(28,25,0)` FAIL`[e]` (`3,42`) | `(28,24,1)` FAIL`[e]` (`3,42`) |
| `lam5_s2102_w2552` | 5 | 2102 | `[T,F,T]` | `(45,0,8)` FAIL`[u]` (`64,05ms`) | FAIL`[e]` (`2,32`) | `(27,0,26)` FAIL`[e+u]` (`3,42;64,05ms`) | `(47,6,0)` **PASS** (`5,75;9,05ms`) | `(53,0,0)` FAIL`[u]` (`64,05ms`) | `(52,0,1)` FAIL`[u]` (`64,05ms`) |
| `lam5_s2103_w1546` | 5 | 2103 | `[T,T,T]` | `(27,13,2)` FAIL`[u]` (`10,10ms`) | **PASS** (`7,82;8,10ms`) | `(13,27,13)` **PASS** (`5,38;8,10ms`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(1,51,1)` FAIL`[e]` (`0,49`) |
| `lam5_s2103_w1560` | 5 | 2103 | `[T,T,T]` | `(27,12,14)` FAIL`[u]` (`12,10ms`) | **PASS** (`7,58;8,10ms`) | `(13,27,13)` **PASS** (`5,62;8,10ms`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(1,51,1)` FAIL`[e]` (`0,49`) |
| `lam6_s2102_w1396` | 6 | 2102 | `[T,T,T]` | `(10,5,4)` FAIL`[u]` (`13,10ms`) | FAIL`[u]` (`10,15ms`) | `(11,10,32)` FAIL`[u]` (`13,10ms`) | `(0,53,0)` FAIL`[e]` (`0,12`) | `(0,53,0)` FAIL`[e]` (`0,12`) | `(1,13,1)` FAIL`[e]` (`0,49`) |
| `lam6_s2102_w2153` | 6 | 2102 | `[T,T,T]` | `(23,11,19)` FAIL`[e+u]` (`2,08;10,15ms`) | FAIL`[e+u]` (`1,65;10,15ms`) | `(21,5,27)` FAIL`[e+u]` (`1,96;10,15ms`) | `(40,13,0)` FAIL`[e+u+m]` | `(40,13,0)` FAIL`[e+u+m]` | `(41,11,1)` FAIL`[e+u+m]` |
| `lam6_s2103_w1283` | 6 | 2103 | `[T,T,T]` | `(28,2,2)` FAIL`[u]` (`27,10ms`) | **PASS** (`7,82;8,10ms`) | `(13,27,13)` **PASS** (`5,62;8,10ms`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(1,51,1)` FAIL`[e]` (`0,49`) |
| `lam6_s2103_w1397` | 6 | 2103 | `[T,T,T]` | `(19,24,10)` FAIL`[u]` (`14,15ms`) | **PASS** (`7,58;8,10ms`) | `(13,27,13)` **PASS** (`5,62;8,10ms`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(1,51,1)` FAIL`[e]` (`0,49`) |
| `lam6_s2103_w1254` | 6 | 2103 | `[F,T,T]` | `(0,40,13)` FAIL`[e]` (`0,00`) | **PASS** (`6,86;8,10ms`) | `(0,35,18)` FAIL`[e]` (`0,00`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(0,53,0)` FAIL`[e]` (`0,00`) | `(0,52,1)` FAIL`[e]` (`0,00`) |

**Síntese das SLAs prejudicadas e resgatadas no Universo A (`N=18`):**
- **`B0-rest` (`8/18`):** falhou `6` por `[u]`, `3` por `[e]` (`w2667` com `a_e=7`, `w2578` com `a_e=27`, `w1254` com `a_e=0` porque eMBB estava vazia em `W`) e `1` por `[e+u]` (`w2153`). Deixou capacidade `< 53` em `7/18` estados no Universo A (`mean sum = 46,61`).
- **`equal=(18,17,18)` (`12/18`):** preservou **`6/8`** estados seguros (prejudicou `w1890` por `[e]` e `w1621` por `[u]`) e resgatou **`6/10`** falhas (`w2667`, `w1546`, `w1560`, `w1283`, `w1397`, `w1254`).
- **`prop` (`11/18`):** preservou **`6/8`** seguros (prejudicou `w1890` por `[e]` e `w2229` por `[e]`) e resgatou **`5/10`** falhas (`w2667`, `w1546`, `w1560`, `w1283`, `w1397`); falhou em `w1254` por `causal_active_slices[0]=False`.
- **`B1-raw / B1-act / B1-sup` (`3/18`, `2/18`, `2/18`):** todos prejudicaram **`7/8`** estados seguros (`6` por `[e]` ou `[e+m]` ao esmagar eMBB; `1` — `w2229` — por `[u+m]` ou `[u]` ao dar `19–20` PRBs ao URLLC que teve `p95=10,15ms`). O único resgate preservado pelos 3 braços foi `w2667` (`a_e=26`); `B1-raw` resgatou adicionalmente `w2552` por acidente (não zerar `u` quando `act[1]=False`).

### 3.2 Universo B (`LT3_DEMAND_COMPLETE`, `N=5`, `STRICT_DEMAND_R19`) e Universo C (`UNKNOWN_INTEGRITY`, `N=2`)

| `state_id` | Universo | `act_mask` | `B0-rest` | `equal` | `prop` | `B1-raw` | `B1-act` | `B1-sup` |
|---|---|---|---|---|---|---|---|---|
| `lam2_s2102_w2685` | Univ B | `[F,F,T]` | `(0,0,0)` **FAIL`[m]`** (`0,0`) | **PASS** (`1,0`) [35 inat.] | `(0,0,53)` **PASS** (`1,0`) | `(0,53,0)` FAIL`[m]` (`0,0`) | `(0,0,41)` **PASS** (`1,0`) | `(0,0,29)` **PASS** (`1,0`) |
| `lam2_s2102_w2442` | Univ B | `[T,T,F]` | `(31,22,0)` **PASS** | **PASS** (`7,70`) [18 inat.] | `(27,26,0)` **PASS** (`11,61`) | `(38,15,0)` **PASS** (`16,38`) | `(38,15,0)` **PASS** (`16,38`) | **`(6,19,0)` FAIL`[e]`** (`2,57`) |
| `lam2_s2102_w2731` | Univ B | `[F,F,T]` | `(0,0,3)` **PASS** (`1,0`) | **PASS** (`1,0`) [35 inat.] | `(0,0,53)` **PASS** (`1,0`) | `(0,53,0)` **PASS** (`1,0`) | `(0,0,43)` **PASS** (`1,0`) | `(0,0,30)` **PASS** (`1,0`) |
| `lam5_s2103_w1241` | Univ B | `[F,T,F]` | `(0,13,0)` **FAIL`[u]`** (`10,15ms`) | FAIL`[u]` (`10,15ms`) [36 inat.] | `(0,53,0)` FAIL`[u]` (`10,15ms`) | `(0,53,0)` FAIL`[u]` (`10,15ms`) | `(0,53,0)` FAIL`[u]` (`10,15ms`) | `(0,53,0)` FAIL`[u]` (`10,15ms`) |
| `lam6_s2102_w2071` | Univ B | `[T,F,T]` | `(37,0,16)` **FAIL`[e]`** (`2,81`) | FAIL`[e]` (`1,59`) [17 inat.] | `(24,0,29)` FAIL`[e]` (`2,08`) | `(39,14,0)` FAIL`[e]` (`3,36`) | **`(53,0,0)` PASS** (`5,01`) | **`(45,0,1)` PASS** (`3,85`) |
| `lam2_s2102_w2448` | Univ C | `[T,T,F]` | `UNK` (`pool=OPEN`) | `UNK` | `UNK` | `UNK` | `UNK` | `UNK` |
| `lam5_s2103_w1351` | Univ C | `[F,T,T]` | `UNK` (`pool=OPEN`) | `UNK` | `UNK` | `UNK` | `UNK` | `UNK` |

**Observação crítica em `lam5_s2103_w1241`:** única fatia ativa é URLLC, e mesmo alocando **`(0, 53, 0)` (100% da célula)** em `prop`/`b1_*`, o `p95` observado foi `10,15 ms > 10,0 ms` — testemunha direta de inviabilidade estrutural de URLLC independente de cota PRB.

---

## 4. Análise do Oráculo Restrito `{B0-rest, equal, prop}` (Limite Superior Retrospectivo Não Causal)

> ⚠️ **AVISO METODOLÓGICO EXPLÍCITO:** os números `O_3`, `O_2` e `O_6` abaixo têm **acesso retrospectivo ao resultado de cada replay em `ns-3`**. Representam estritamente **limites superiores em *hindsight* (não causais)** para quantificar espaço de ganho, **nunca** uma política online implementável.

| Oráculo (Hindsight Não Causal) | Universo A (`N=18`, `ALL3_COMPLETE`) | Universo B (`N=5`, `LT3_DEMAND`) | Resgates s/ `B0=FAIL` | Danos s/ `B0=PASS` | Distribuição dos subconjuntos vencedores em Univ A |
|---|---|---|---:|---:|---|
| `B0-rest` (baseline causal congelado) | `8/18 (44,4%)` | `2/5 (40,0%)` | `0/10` | `0/8` | — |
| `prop` (referência estática ativa) | `11/18 (61,1%)` | `3/5 (60,0%)` | `5/10` | `2/8` | — |
| `equal = (18,17,18)` (referência cega) | **`12/18 (66,7%)`** | `3/5 (60,0%)` | `6/10` | `2/8` | — |
| **`O_2 = {eq, prop}`** *(hindsight)* | `13/18 (72,2%)` | `3/5 (60,0%)` | `6/10` | `1/8` | `NONE:5`, `eq+prop:10`, `eq:2`, `prop:1` |
| **`O_3 = {B0, eq, prop}`** *(hindsight)* | **`14/18 (77,8%)`** | `3/5 (60,0%)` | `6/10` | `0/8` | `NONE:4`, `b0+eq+prop:5`, `eq+prop:5`, `b0:1` (`w1890`), `b0+prop:1` (`w1621`), `b0+eq:1` (`w2229`), `eq:1` (`w1254`) |
| **`O_6 = {todos os 6 braços}`** *(hindsight)* | `15/18 (83,3%)` | `4/5 (80,0%)` | `7/10` | `0/8` | Ganho `O_6 - O_3` em Univ A = **+1 estado** (`w2552` via `b1_raw=(47,6,0)`); em Univ B = **+1 estado** (`w2071` via `b1_act=(53,0,0)`) |

**Estratificação do Oráculo `O_3`:**
- **Por $\lambda$ (Univ A):** $\lambda=5$: `7/9 (77,8%)` vs `B0=4/9`; $\lambda=6$: `7/9 (77,8%)` vs `B0=4/9`.
- **Por seed (Univ A):** **`seed 2103`: `O_3 = 9/9 (100,0%)`** (e até `O_2 = {eq, prop}` acerta **`9/9`**, com `prop = 8/9` e `eq = 8/9` contra `B0 = 4/9`!); **`seed 2102`: `O_3 = 5/9 (55,6%)`** vs `B0 = 4/9` (4 estados em `s2102` — `w2578`, `w2552`, `w1396`, `w2153` — não foram resolvidos por nenhum membro de `{B0, eq, prop}`).
- **Por capacidade estrutural:** o espaço máximo de ganho causal de escolher dentre `{B0, eq, prop}` sobre o baseline estático `equal=(18,17,18)` é de apenas **`+2/18` (`+11,1 pp`)** no Universo A (`w1890` via `B0` e `w1621` via `B0/prop`) e **`+0/5`** no Universo B (embora `prop` elimine 100% dos `212` PRBs desperdiçados em fatias inativas).

---

## 5. Teste Empírico de Separabilidade Pré-Decisão (`f_i(a)` e `sigma_i(a)` antes de `W`)

### 5.1 Calibragem e discriminação por fatia (observações únicas `(state_id, slice_id, a_slice)` com desfecho conhecido)

| Escopo | $n$ | `tau` médio | Pass Real | `f_i >= 0` | Acurácia | Bal. Acc | TPR | TNR | **AUROC($f_i$)** | **Spearman $\rho(f_i, m_i)$** | MAE($m_i$) | Viés ($f_i - m_i$) | $\rho(\sigma_i, \lvert err \rvert)$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **eMBB (`s0`)** | 93 | `0,0327` | `54,84%` | `53,76%` | `62,37%` | `62,11%` | `64,71%` | `59,52%` | **`0,7283`** | **`+0,3596`** | `0,7483` | `-0,1246` | `+0,2588` |
| **URLLC (`s1`)** | 93 | **`0,0180`** | `80,65%` | **`2,15%`** | **`21,51%`** | `51,33%` | **`2,67%`** | `100,0%` | **`0,5607`** | **`+0,1168`** | `0,5844` | **`-0,4469`** | **`-0,2614`** |
| **mMTC (`s2`)** | 101 | `0,0300` | `95,05%` | `49,50%` | `52,48%` | `65,52%` | `51,04%` | `80,00%` | `0,8187` | `+0,1817` | `0,1197` | `-0,0488` | — |
| **JOINT_ACTIVE** (`min f`) | 116 | `0,0278` | `43,97%` | **`11,21%`** | `63,79%` | `59,25%` | `21,57%` | `96,92%` | **`0,5959`** | `+0,0993` | `0,3926` | `-0,1460` | — |

*(Checagens de sensibilidade por deduplicação `unique_replay` e `all_138_known_rows` confirmam o mesmo diagnóstico: em `unique_replay`, `n=98`, `URLLC AUROC = 0,5461`, `rho = 0,0927`; e em `eMBB` `n=101`, `AUROC = 0,7025`, `rho = 0,3142`. Para subgrupo confiável $\sigma \le 0{,}15$, quase todas as observações já estão nele — `92/93` em `eMBB`, `92/93` em `URLLC`, `100/101` em `mMTC` — portanto $\sigma \le 0{,}15$ **não** filtra erros de `URLLC`: $\sigma_1$ médio foi `0,0192` quando passou e `0,0087` quando falhou).*

### 5.2 Diagnóstico causal em `pool_3 = {B0, eq, prop}` e `pool_all` no Universo A (`N=18`)

Avaliando regras puramente causais (usando **somente** `f_i(a)` e `sigma_i(a)` anteriores a `W`) para ranquear candidatos já executados no `ns-3`:

| Regra Causal de Ranqueamento (sem acesso a resultado futuro) | Pool `{B0, eq, prop}` (`N=18`) | Pool `{todos os 6 braços}` (`N=18`) |
|---|---|---|
| Baseline cego `equal=(18,17,18)` | **`12/18 (66,7%)`** (`R=6, H=2, Net=+4`) | `12/18` |
| `R_maxmin_f_beta0`: `argmax min f_i` | `12/18 (66,7%)` (`R=5, H=1, Net=+4`; `b0:9, eq:2, prop:7`) | **`2/18 (11,1%)`** (`R=1, H=7, Net=-6`; `b1_raw:17, b1_act:1`) |
| `R_feas_count_then_sum_clip`: maximizar `#(f_i>=0)` + `sum min(0, f_i)` | `11/18 (61,1%)` (`R=5, H=2, Net=+3`) | `9/18 (50,0%)` (`R=4, H=3, Net=+1`) |
| `R_feas_count_lcb_1sig`: `#(f_i - sigma_i >= 0)` | `11/18 (61,1%)` (`R=5, H=2, Net=+3`) | `9/18 (50,0%)` (`R=4, H=3, Net=+1`) |
| `R_stay_b0_if_all_f_ge_0_else_feas_count` | `11/18 (61,1%)` (`R=5, H=2, Net=+3`) | `9/18 (50,0%)` (`R=4, H=3, Net=+1`) |
| `R_min_sigma_then_feas` | `11/18 (61,1%)` (`R=5, H=2, Net=+3`) | `9/18 (50,0%)` (`R=4, H=3, Net=+1`) |

**Achado central:** sobre os modelos `B0-rest` congelados ($\tau \approx 0{,}018\text{–}0{,}040$), **nenhuma regra causal baseada em `f_i` e `sigma_i` conseguiu superar o baseline estático `equal=(18,17,18)`**. A razão é estrutural: nos estados de discordância em `seed 2102` (`w1890`, `w2667`, `w2229`), a predição `f_0` (**eMBB**) ranqueou corretamente a melhor ação (`b0 > prop > eq` em `w1890`; `eq=prop > b0` em `w2667`; `b0 > eq > prop` em `w2229`), mas o sinal ruidoso/descalibrado `f_1` (**URLLC**, `AUROC=0,56`, `p95` descontínuo por alinhamento de slot HARQ/TDD — ex. `w1397`: `a_u=24` teve `14,15ms` enquanto `a_u=17` teve `8,10ms`!) dominou ou empatou decisões conjuntas.

---

## 6. Frequências de Fallback, Uso do Suporte Confiável e Desperdício de Capacidade (`25` estados)

| Braço | `fallback_used` (`25` est.) | Tiers (`25` est.) | $\max_{\text{act}} \sigma_i > 0{,}15$ (observado nos traces) | $\max_{\text{all3}} \sigma_i > 0{,}15$ | Estados `sum < 53` (RBGs ociosos totais) | Estados c/ desperdício inativo (RBGs inativos totais) | Estados c/ slice viva zerada (`starved_live`, total slices) |
|---|---:|---|---:|---:|---:|---:|---:|
| **`b0`** | `0/25` | `b0_or_static: 25` | `0/25` (`max=0,1473`) | `5/25` | **`13/25`** (`308`) | `0/25` (`0`) | `3/25` (`3`, incl. `w2685=[0,0,0]`) |
| **`eq`** | `0/25` | `b0_or_static: 25` | `0/25` (`max=0,1405`) | `5/25` | `0/25` (`0`) | **`9/25`** (`212`) | `0/25` (`0`) |
| **`prop`** | `0/25` | `b0_or_static: 25` | `2/25` (`max=0,1762`) | `5/25` | `0/25` (`0`) | `0/25` (`0`) | `0/25` (`0`) |
| **`b1_raw`** | `0/25` | `joint_raw: 25` | `1/25` (`max=0,1528`) | `5/25` | `1/25` (`11`) | `4/25` (`126`) | **`21/25`** (`30`) |
| **`b1_act`** | `0/25` | `joint_active: 25` | `1/25` (`max=0,1528`) | `5/25` | `2/25` (`22`) | `0/25` (`0`) | **`19/25`** (`28`) |
| **`b1_sup`** | `0/25` | **`joint_support: 25`** | **`0/25`** (`max=0,1416`) | `5/25` | **`6/25`** (`153`) | `0/25` (`0`) | `0/25` (`0`; mas `8/8` `s2103` com `a_e=1`) |

- **Cobertura da grade `[0..53]` por $\sigma \le 0{,}15$:** em janelas ativas, o envelope confiável $\sigma \le 0{,}15$ já cobria `38–54/54` ações (mediana `52`) no eMBB, `11–54/54` (mediana `54`) no URLLC e `35–54/54` (mediana `50`) no mMTC (`<= 0,55` em `54/54` em todas as slices ativas). Portanto, **`B1-support` nunca acionou relaxamento Tier 2 nem fallback (`0/25`)** — a degradação veio inteiramente de dentro do próprio Tier 1 confiável.
