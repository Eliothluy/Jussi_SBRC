# Rastreabilidade de Fontes — Artigo SQR → Parâmetro (Fase 0)

> **Versão:** 1.1 — 11/10/2026 · **Responsável:** Agente B · **Status:** proposto, sujeito a aprovação do orquestrador.
> **Artigo-fonte:** A. Tuerxun, A. Nakao, "Safe RAN Slicing in O-RAN: Minimizing SLA Violations via Model-Based Reinforcement Learning", IEEE CCNC 2026, DOI [10.1109/CCNC65079.2026.11366589](https://doi.org/10.1109/CCNC65079.2026.11366589), 6 páginas.
> **Âncoras:** cada linha cita a **seção/tabela/equação do artigo** e a linha correspondente em `tradução.md` (tradução verificada contra o PDF de 6 páginas — `AGENTS.md` §2). A Tabela I aparece em `tradução.md` nas linhas 188–199.
> **Regra:** toda linha desta tabela é um valor **publicado** (`[ARTIGO]`). Quando uma linha carrega um detalhe operacional **não** publicado, esse detalhe fica explicitamente marcado `[ASSUMIDO]` dentro da própria linha (nunca sob a tag [ARTIGO]). Tudo que o artigo não publica está consolidado no `assumptions_register.md` — nunca misturar os dois.

---

## 1. Sistema e formulação (Seção III)

| Parâmetro | Valor publicado | Fonte no artigo | Âncora `tradução.md` | Observações |
|---|---|---|---|---|
| Estações-base | 1 macrocélula | Seção III-A; Seção V | L48; L186 | "uma única estação-base" |
| Capacidade total | C PRBs (símbolo; **valor não publicado**) | Seção III-A | L48 | Valor de C é `[ASSUMIDO]` (register R01/R02) |
| Nº de slices | L = 3 | Seção III-A; Seção V | L48; L186, Tabela I | As 3 são eMBB (Seção V) |
| Período de decisão | 100 TTIs | Seção III-A | L48 | "cada passo corresponde a 100 TTIs" |
| Duração do TTI | 1 ms | Seção III-A | L48 | Base da premissa μ=0 (R03) |
| Estado local s^(i) | KPIs agregados: menor throughput dos usuários, nº de usuários, tamanhos das filas, SNR | Seção III-B | L52 | Publicado como exemplo ("tais como"); composição exata `[ASSUMIDO]` (R20) |
| Estado global s̄ | uso total de PRBs das outras fatias, nº total de UEs | Seção III-B | L52 | "por exemplo"; composição exata `[ASSUMIDO]` (R20) |
| Estado completo | z_t^(i) = [s_t^(i), s̄_t] | Seção III-B | L52 | — |
| Ação | a_t^(i) ∈ {0, 1, …, C} PRBs por slice | Seção III-B | L54 | Inteiros; semântica de execução no MAC é `[ASSUMIDO]` (R27 — cota absoluta com RBGs ociosos) |
| Restrição de capacidade | Σ_i a_t^(i) ≤ C | Seção III-B | L54 | Projeção Π garante (Seção IV-B.3) |
| KPI realizado y_t^(i) | "por exemplo, o 5º percentil do throughput dos usuários" | Seção III-B | L56 | Adotado como definição operacional (R18) |
| Objetivo (CMDP) | min uso médio de PRBs s.t. taxa de violação ≤ δ^(i), ∀i | Seção III-C, Eq. (1) | L62-63 | — |
| Orçamento de violação δ | "por exemplo, 1%" | Seção III-C | L70; Fig. 2 (meta 99%, L203); Seção V | Adotado δ = 1% |
| Limiar de SLA | Λ^(i) = valor mínimo aceitável de KPI | Seção III-C | L68 | Valores na Tabela I (linha SLAs) |

## 2. Regressor quantílico online (Seção IV-A)

| Parâmetro | Valor publicado | Fonte no artigo | Âncora `tradução.md` | Observações |
|---|---|---|---|---|
| Modelo | f_τ^(i)(x) = Σ_{j=1..B} α_j · k(x, x_j) | Seção IV-A | L84 | Aproximador kernelizado |
| Incerteza/novidade | σ_t^(i)(x) = 1 − max_j k(x, x_j) | Seção IV-A | L85 | Exige kernel normalizado (R08) |
| Kernel | **Matérn** (ν, ℓ não publicados) | Seção IV-A | L87 | ν/ℓ `[ASSUMIDO]` (R08) |
| Entrada do modelo | x_t^(i) = [z_t^(i), a_t^(i)/C] | Seção IV-A | L80 | — |
| Memória | B vetores de suporte com orçamento fixo; **B = 2000** na avaliação | Seção IV-A ("orçamento fixo"); Seção V ("memória … limitada a 2000 entradas") | L82; L209 | Poda só qualitativa (R10) |
| Regra de atualização (pinball) | subpredição (y > f): α_UP = η·τ; sobrepredição (y ≤ f): α_OP = η·(τ−1) | Seção IV-A | L91-92 | — |
| Taxa de aprendizado | η = 0,1 | Tabela I; Seção IV-A | L198 | — |
| Falha crítica | α_CF = α_OP · P_penalty, P_penalty = 2 | Seção IV-A | L96; Tabela I (L196) | P_penalty ativo nas variantes γ=0 e γ=1 (AGENTS §5.7; register R25) |
| Novo vetor de suporte | adicionado a cada passo com coeficiente do gradiente da pinball | Seção IV-A | L98 | Online a cada passo |
| Poda | "os vetores de suporte menos informativos são descartados, priorizando aqueles que mais contribuem" | Seção IV-A | L98 | Qualitativo; regra exata `[ASSUMIDO]` (R10); reforçado na Seção V ("amostras de baixo impacto ou redundantes", L209) |
| Piso do quantil | fronteira de decisão mais avessa ao risco que o quantil-τ nominal | Seção IV-D | L144-154 | Propriedade provada; não é parâmetro |

## 3. Política hierárquica e execução (Seção IV-B + Algoritmo 1)

| Parâmetro | Valor publicado | Fonte no artigo | Âncora `tradução.md` | Observações |
|---|---|---|---|---|
| Conjunto seguro | A_safe = { a ∈ {0..C} : f_τ(z,a) ≥ Λ } | Seção IV-B.1 | L108 | — |
| Modo Confiante | a = argmax_{a∈A_safe} ( f_τ(z,a) − γ·(a/C) ) | Seção IV-B.2 | L114 | γ alterna SÓ o termo de custo (AGENTS §5.7; register R25) |
| Variantes de γ | γ = 0 ("w/o Cost") e γ = 1 ("w/ Cost") | Seção V | L201 | Base da comparação Figs. 2–4; redação do artigo sobre γ=0 é ambígua quanto a P_penalty — interpretação congelada em R25 |
| Modo Incerto (ramo 1) | com probabilidade ε: ação ~ Softmax ∝ exp( f_τ(z,a) / T_softmax ) | Seção IV-B.2 | L118-120 | ε = 0,1 (Tabela I, L197); **T_softmax não publicada** (R07) |
| Modo Incerto (ramo 2) | com probabilidade 1−ε: k-NN Safety-First nos K estados vizinhos, argmax f_τ sobre ações com y ≥ Λ | Seção IV-B.2 | L122-126 | K = 10 (Tabela I, L199) |
| Margem de segurança | a ← a + round(σ · M) | Seção IV-B.3 | L132 | σ avaliado em a_base `[ASSUMIDO]` (R22); M adaptado (R11) |
| Projeção | se Σ_i a_i > C: operador Π reduz proporcionalmente | Seção IV-B.3 | L136 | Forma exata (inteira, restos, desempates) `[ASSUMIDO]` (R24) |
| **Alvo da atualização real do regressor** | **(z_t^(i), a_final,t^(i), y_t^(i))** — a ação FINAL, pós-margem e pós-projeção | **Algoritmo 1, Seção IV** | **L177**: "Atualizar fₜ₊₁⁽ⁱ⁾ com (zₜ⁽ⁱ⁾, a(final,t)⁽ⁱ⁾, yₜ⁽ⁱ⁾)" | Fato publicado (correção v1.1, revisor): fecha o quadro com R12 (atualização virtual usa a pré-projeção) e R22 (σ na margem avaliado em a_base) |
| Atualização virtual | vetor de suporte da ação pré-projeção com "pequeno peso negativo" | Seção IV-B.3 | L136 | Valor do peso `[ASSUMIDO]` (R12) |
| Adaptação τ/M | mais violações ⇒ τ↓ e M↑; inverso quando quase perfeito | Seção IV-C | L138-140 | Equações/ganhos/janela e sinal exato de violação não publicados (R11) |

## 4. Adaptação dinâmica (Seção IV-C)

| Parâmetro | Valor publicado | Fonte no artigo | Âncora `tradução.md` | Observações |
|---|---|---|---|---|
| Direção da adaptação | mais violações ⇒ τ↓ e M↑; SLA quase perfeito ⇒ τ↑ cauteloso e M↓ | Seção IV-C | L138-140 | Equações/ganhos/janela **não publicados** (R11) |
| Faixa do quantil τ | [0,01; 0,05] | Seção V | L186 | Inicialização τ₀ `[ASSUMIDO]` (R11) |

## 5. Cenário de avaliação (Seção V + Tabela I)

| Parâmetro | Valor publicado | Fonte no artigo | Âncora `tradução.md` | Observações |
|---|---|---|---|---|
| Alcance da célula | 2 km | Tabela I; Seção V | L196; L186 | — |
| Frequência portadora | 2 GHz | Tabela I; Seção V | L199; L186 | — |
| Propagação | Macrocelular urbano (3GPP TS 36.942, ref. [14] do artigo) | Tabela I; Seção V | L192; L186; ref. L259 | — |
| Fading | Pedestrian A (TS 36.104, ref. [15] do artigo) | Tabela I; Seção V | L193; L186; ref. L261 | Não existe no 5G-LENA v5.2 (verificado: sem classe de fading em `contrib/nr`; apenas 2 comentários — register R06) — aproximação `[ASSUMIDO]` (R06) |
| Potência transmitida | 30 dBm | Tabela I | L194 | — |
| Ganho de antena da BS | 15 dBi | Tabela I | L195 | — |
| Ruído | 9 dB (campo "Noise") | Tabela I | L197 | Interpretação NF `[ASSUMIDO]` (R15); piso térmico resultante: −95 dBm @10 MHz |
| Interferência mais ruído | −110 dBm | Tabela I | L198 | Interpretação por PRB `[ASSUMIDO]` (R16); leitura por banda total inexequível (~15 dB abaixo do piso térmico) |
| Chegadas de UEs | Poisson, 2 usuários/min | Tabela I; Seção V | L192; L186 | Escopo global×por-slice não publicado (R13) |
| Tempo de conexão | Exponencial, média 60 s | Tabela I; Seção V | L193; L186 | — |
| Mobilidade | posições e trajetórias aleatórias por episódio | Seção V | L186 | Velocidade/modelo `[ASSUMIDO]` (R05) |
| Slices | 3 eMBB com limiares distintos | Seção V | L186 | — |
| SLAs Λ | 0,5 / 1,0 / 1,5 Mbps **por UE** | Tabela I | L194 | Granularidade per-UE explícita na Seção V (L207) |
| Atribuição de slice | cada UE atribuído aleatoriamente a uma das fatias | Seção V | L186 | Uniforme `[ASSUMIDO]` (R13) |
| **Agregação das métricas** | **"todas as métricas são agregadas nas três fatias para permitir comparação consistente entre métodos"** | Seção V | **L201** | Fato publicado (correção v1.1, revisor); forma exata da agregação (média de taxas por slice vs pooled UE-janela) não especificada — `[ASSUMIDO]` (R26) |
| Memória B | 2000 entradas | Seção V | L186, L209 | — |
| Faixa de quantis | τ ∈ [0,01; 0,05] | Seção V | L186 | — |
| ε (exploração) | 0,1 | Tabela I | L197 | — |
| K (vizinhos) | 10 | Tabela I | L199 | — |
| P_penalty | 2 | Tabela I | L196 | — |
| η | 0,1 | Tabela I | L198 | — |
| Frequência de atualização do modelo | a cada 100 slots; cada atualização < 100 ms | Seção V | L186 | "atualizado a cada 100 slots e cada atualização leva menos de 100 ms"; consistente com período de 100 ms (100 TTIs de 1 ms) |
| Métodos avaliados | KBRL [12]; SQR w/o Cost (γ=0); SQR w/ Cost (γ=1) | Seção V | L201 | KBRL = Alcaraz et al., IEEE TWC 2022 (repo: jjalcaraz-upct/network-slicing) |
| Seeds/execuções | "múltiplas execuções para garantir confiabilidade estatística" — **nº não especificado** | Seção V | L201 | Protocolo próprio `[ASSUMIDO]` (R23) |
| Simulador | **Não identificado** (nenhuma menção a ns-3/MATLAB/OMNeT; sem código/commit) | — (ausência) | — | Reimplementação fiel, não reprodução bit a bit (PLANO §2.1; GUIA §2.2 critério editorial) |

## 6. Resultados publicados (apenas referência — NÃO são critérios de aceite)

| Alegação | Valor | Fonte | Âncora `tradução.md` | Status |
|---|---|---|---|---|
| Meta de SLA pós-convergência | < 1% de violação (99%) | Resumo; Fig. 2; Seção V | L8; L203 | Referência qualitativa; recalculada no nosso ambiente |
| Excesso de violação do SQR com custo | 0,003–0,008% | Seção V | L211 | Não é aceite (AGENTS §5.6) |
| Economia de RBs | ">60%" | Seção V (discussão Fig. 4) | L213 | **Inconsistente** com a Conclusão (20%) — AGENTS §5.6 |
| Redução de violações acumuladas / recursos | 60% / 20% | Seção VI (Conclusão) | L223 | Inconsistência V×VI documentada (GUIA §1.2; PLANO §2.1) |
| Figuras-alvo | Fig. 2 (taxa de violação), Fig. 3 (violação acumulada), Fig. 4 (RBs usados) | Seção V | L203; L207; L213 | Réplicas equivalentes com IC 95% (PLANO Fase 4; GUIA §11.4); agregação conforme R26 |

## 7. Parâmetros NÃO publicados (resumo → register)

C, banda, SCS/numerologia, duplexação; tráfego aplicativo (direção/protocolo/pacotes); velocidade/modelo de mobilidade; nº inicial de UEs; composição/ordem exata e normalização do estado (incl. escalas e tratamento de missing de SNR); ν e ℓ do Matérn; amplitude/escala do kernel; regra exata de poda; equações do controlador τ/M (τ₀, M₀, ganhos, janela, sinal de violação v_W); T_softmax; distância/representação do k-NN e regras de borda; valor do peso da atualização virtual; local de avaliação de σ na margem; **forma exata do operador Π (arredondamento/restos/desempates)**; **forma exata da agregação das três fatias**; nº de seeds/episódios/duração/warm-up; definição exata de violação per-UE vs per-slice e harmonização com KBRL; semântica de execução da cota no MAC (RBGs ociosos); simulador/código.

Cada um deles está `[ASSUMIDO]` no `assumptions_register.md` (R01–R27) e compõe a lista de perguntas aos autores (GUIA §15, itens 1–12) — respostas futuras **substituem** as premissas correspondentes (nunca o contrário).

---
*Histórico: v1.0 (11/10/2026, Agente B) — criação na Fase 0, conforme modelo do GUIA §2.1 e regra de rastreabilidade do AGENTS §5.3. **v1.1 (11/10/2026, Agente B)** — correções dos apontamentos do revisor: nova linha [ARTIGO] do alvo da atualização real (a_final — Algoritmo 1, L177, §3); nova linha [ARTIGO] da agregação das métricas nas três fatias (Seção V, L201, §5); higiene de tags (detalhes não publicados marcados [ASSUMIDO] dentro da linha); lista §7 ampliada (Π, agregação, escalas/missing, semântica da cota no MAC).*
