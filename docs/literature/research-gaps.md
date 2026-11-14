# Gaps de pesquisa — literatura × G1–G8, com perguntas testáveis

> **Skill:** oran-literature-review · **Artefato:** 6/6 (`research-gaps.md`) · **Data:** 2026-10-18
> **Citações:** `[Lxx]` = linha de `literature-matrix.csv` (60 entradas) + DOI. Evidências próprias = resultados verificados nesta equipe (campanha 40 runs em `MODA/ns-o-ran-gym/results/sqr_campaign/tables/summary_by_method.csv`; sweep de raio em `README.md` §3.2; paridade e latência E2 em `MODA/docs/sqr/fase5.4-xapp-sqr-paridade.md` §4–§5 — linhas 54 e 61–72 — e `AGENTS.md` §8), distinguidas de claims de autores. *[CORRIGIDO na sessão de revisão 2026-10-18: o ponteiro anterior — "sweep de raio e paridade E2 em `README.md` §3–§4" — estava quebrado para a paridade E2: o README não contém os números 3585/3585, p50 2,39/p95 9,28 ms nem +1,23 ms (verificado por grep nesta sessão); eles estão no doc de paridade (linhas 54, 68, 71–72) e em AGENTS.md:179.]*
> **Regra anti-lacuna-falsa (skill):** nenhum gap abaixo foi declarado por "poucos resultados de query". Antes de cada declaração houve expansão de sinônimos (4 eixos, 19 queries booleanas), snowballing backward (Crossref) e forward-check em 12 incluídos — registrado em `search-strategy.md` §4.3 e `tools/search-log.jsonl` (137 entradas — 132 da sessão de busca + 5 da sessão de revisão). Onde o antecedente mais próximo **existe e enfraquece** nossa novidade, ele é citado explicitamente.
> **Status:** este artefato aguarda revisão e aprovação explícita do pesquisador. Nenhum manuscrito será escrito antes (regra da skill).

---

## Legenda dos rótulos de evidência

- **[MATRIZ-full]** — evidência de full-text lido nesta sessão (27 entradas; *[CORRIGIDO: era 31]* — recontagem do campo `verified` na sessão de revisão, ver `evidence-report.md` §1).
- **[MATRIZ-abs]** — evidência de abstract/metadados somente; **não** prova omissão do autor (full-text paywalled).
- **[EQUIPE]** — resultado experimental nosso, verificado nesta sessão.
- **[BUSCA-LIM]** — observação "não encontrada na amostra de busca"; sujeita às limitações de `evidence-report.md` §7 (top-N do OpenAlex, inglês, 1 dia de busca). **Não constitui prova de inexistência.**

---

## G1 — Inviabilidade física dos parâmetros publicados (cenário)

**Definição** (`README.md` §4): 30 dBm + raio 2 km + urban macro do SQR [L55]+10.1109/ccnc65079.2026.11366589 é inconsistente com link budget de macro real; o artigo alega 99% de SLA sem publicar simulador, C, banda, MCS ou scheduler.

**Evidência na literatura:**
- Nenhum trabalho da amostra que avalia controlador de slice com RL **declara e valida** link budget/potência/raio/antena antes da avaliação — incluindo os de stack real, onde a física é dada pelo hardware (ex.: [L15]+10.1109/iccworkshops67674.2025.11162144 limita-se a reportar saturação; [L37]+10.1109/mecom67453.2025.11439664 usa FSPL com offset fixo em RFsim). **[BUSCA-LIM, 60 entradas]**
- A metodologia de calibração existe e está madura **na comunidade ns-3/5G-LENA**, mas isolada dos controladores RL: calibração 3GPP IMT-2020 do 5G-LENA [L47]+10.1016/j.simpat.2022.102580; calibração do ancestral NR [L45]+10.1016/j.simpat.2019.101933; canal industrial InF [L50]+10.5281/zenodo.6553700; canal espacial [L44]+10.1145/3389400.3389401; ray-tracing site-specific [L43]+10.1109/ccnc65079.2026.11366300; rework de link budget com padrões de antena efetivos [L51]+10.1109/access.2020.3039423. **[MATRIZ-full/abs — full-text: L44, L45, L47; abstract-only: L43, L50, L51]** *[CORRIGIDO 2026-10-18: o rótulo único "[MATRIZ-full]" classificava errado 3 das 6 entradas — L43, L50 e L51 são metadados+abstract via OpenAlex (campo `verified` da matriz; lista das 27 full-text em `evidence-report.md` §1)].*
- Evidência externa de inviabilidade de célula grande: com ISD/células grandes, cobertura de borda com specs 4G/5G correntes é inviável sem enhancements — [L54]+10.1186/s13638-022-02133-3. Corrobora a causa do nosso sweep. **[MATRIZ-abs]**
- **[EQUIPE]** déficit de 25–30 dB do cenário publicado (antena 1×1 vs 8×8, NLOS, 30 vs 43 dBm); sweep controlado (só o raio muda): violação 0% @300 m → 9,7% @800 m → 58,7% @1500 m → 88% @2 km; RLF de borda como causa (~2,7 quedas/UE/600 s).

**Antecedente mais próximo (enfraquece?):** [L37]+10.1109/mecom67453.2025.11439664 estuda regime de congestionamento deliberado em stack real — aproxima-se da pergunta de regime (G6), mas não faz análise de viabilidade por link budget nem audita parâmetros de artigo terceiro.

**Perguntas de pesquisa testáveis:**
- **RQ1.1** — Qual é a fronteira de viabilidade física (raio × potência × configuração de antena × LOS) para SLAs per-UE de 0,5/1,0/1,5 Mbps em ns-3/5G-LENA calibrado, e onde o cenário publicado do SQR cai nessa fronteira? *(Testável: sweep paramétrico com o mesmo código; já executado em 1 dimensão — raio — e extensível a potência/antena/LOS.)*
- **RQ1.2** — Controladores quantílicos (SQR γ=0/γ=1) mantêm separação de comportamento apenas na região viável da fronteira? *(Testável: campanha nos 4 pontos do sweep × 10 seeds, medindo violação, consumo e diversidade de modos.)*

**Posicionamento do artigo novo:** primeira **auditoria de viabilidade física** de um cenário publicado de slicing+RL, com calibração ancorada em 3GPP ([L47] como método) e sweep empírico como discriminador implementação×parâmetro. Ponte entre duas literaturas que não conversam: calibração 5G-LENA [L45, L47, L43] × controladores RL [L55 e família].

---

## G2 — Irreprodutibilidade por omissão de parâmetros

**Definição** (`README.md` §4): 12+ parâmetros essenciais do SQR não publicados (C/banda/numerologia, tráfego, mobilidade, T_softmax, ν/ℓ do Matérn, regra de poda, adaptação τ/M, peso da atualização virtual, seeds/duração).

**Evidência na literatura:**
- **45/60** entradas sem artefato público citado (incluindo 2 que declaram explicitamente a ausência: [L37]+10.1109/mecom67453.2025.11439664 e [L59]+10.1109/icnp52444.2021.9651934); **12/60** com artefato público real citado (4 verificados nesta sessão: L12, L23, L49, L50); [L07]+10.1109/tmc.2022.3188013 promete "disponibilização pública futura" com URL candidata 404 na sessão (contagem literal em `evidence-report.md` §2.4). **[MATRIZ-full/abs]**
- Contraste interno do próprio par SQR×KBRL: KBRL publica simulador, algoritmos e scripts ([L23]+10.1109/twc.2022.3195570, rodapé Sec. VII) — **o baseline é reproduzível, o método proposto não** ([L55]: simulador não publicado). **[MATRIZ-full]**
- Suporte estatístico: **1/60** entradas publica seeds/runs + IC (KBRL [L23]: 30 runs, IC 90/95%); o SQR alega "múltiplas execuções" sem seeds/duração/IC [L55]; [L24]+10.1109/jsac.2023.3336191 reporta "melhores 64 runs" (seleção pós-hoc sem total de runs). **[MATRIZ-full]**
- Plataformas abertas da amostra (ColO-RAN [L07]+10.1109/tmc.2022.3188013, NORI [L12]+10.14209/sbrt.2025.1571157244, ORANSlice [L16]+10.1145/3636534.3701544, xDevSM [L20]+10.1109/tnsm.2026.3733344, SEM [L49]+10.1145/3345768.3355942) demonstram que o ecossistema **tem** infraestrutura para reprodutibilidade — a omissão é prática de relato, não impossibilidade técnica. **[MATRIZ-full]**

**Antecedente mais próximo (enfraquece?):** nenhum dos 60 incluídos é uma **auditoria de reprodutibilidade** de artigo de slicing+RL (pergunta 2 de posicionamento). **[BUSCA-LIM]** — se existir precedente fora da amostra (ex.: replication de papers de networking em venues como ReScience), o diferencial deve ser reformulado de "primeira auditoria" para "primeira auditoria em safe RAN slicing com reimplementação fiel e campanha pareada".

**Perguntas de pesquisa testáveis:**
- **RQ2.1** — Quanto do gap de desempenho entre o resultado publicado (violação <1%) e o reproduzido (84–90% em cenário calibrado) é explicável pelos parâmetros omissos? *(Testável: design fatorial sobre os 12+ parâmetros [ASSUMIDO] do registro de premissas v1.3, medindo sensibilidade da violação a cada um.)*
- **RQ2.2** — Um protocolo de rastreabilidade `[ARTIGO]/[ASSUMIDO]` (registro auditável de 31 linhas R01–R31) reduz o custo de replicação? *(Testável: replicação do nosso port por terceiro cronometrada vs esforço estimado sem registro.)*

**Posicionamento do artigo novo:** registro de premissas auditável + campanha pareada 10 seeds + IC 95% — igualando o melhor padrão da amostra ([L23]) e superando o do artigo auditado ([L55]); proposta de protocolo generalizável.

---

## G3 — Sem evidência de aplicação real da cota no MAC

**Definição** (`README.md` §4): o SQR não demonstra que `Σa ≤ C` chega ao agendamento (RBGs efetivamente ociosos); sem isso, o termo de custo `γ·a/C` é inobservável e a Fig. 4 não significa nada.

**Evidência na literatura:**
- A cadeia ação→efeito no MAC é **assumida**, não instrumentada, nos trabalhos com ação em PRBs por slice: ColO-RAN [L07]+10.1109/tmc.2022.3188013 (ação em nº de PRBs; campo `relation_to_our_gaps` registra "sem verificação de execução no MAC"), NORI [L12] (validação funcional de canal E2, não de execução), RSLAQ [L18]+10.1109/tmc.2026.3666787, [L31]+10.1109/icnc59896.2024.10556357. **[MATRIZ-full]**
- Trabalhos de scheduler 5G-LENA instrumentam o **agendador**, mas não um controlador RL externo: QoS scheduler por fluxo [L40]+10.1145/3592149.3592159; Lyapunov [L39]+10.1145/3659111.3659118; SARA sub-band [L41]+10.1145/3747204.3747214; extensão de slicing do 5G-LENA [L38]+10.1145/3837358.3837362. **[MATRIZ-full/abs]**
- **[EQUIPE]** gate R27 i–iv (smoke determinístico com fases Σa=0 / 0,5·C / C; utilização cota/agendado = 1,0000; 57% de RBGs ociosos na mediana em Σa=0,5·C; 0 RBG servido em Σa=0; telemetria `prb_requested/executed/scheduled/idle` por janela) + evidência de campanha: ≈75–77 PRBs ociosos/slot sob SQR vs ≈0 sob equal.

**Antecedente mais próximo (enfraquece?):** DORA [L37]+10.1109/mecom67453.2025.11439664 aplica cotas de PRB em stack OAI real (3GPP PHY/MAC) — a ação **é** executada por construção no scheduler real; porém não publica gate de verificação nem telemetria de RBGs ociosos como evidência de conformidade ação→execução. **[MATRIZ-full]**

**Perguntas de pesquisa testáveis:**
- **RQ3.1** — A projeção proporcional do SQR preserva `Σa ≤ C` **após** o mapeamento para RBGs pelo scheduler (inclusive com C não divisível — C=53 no nosso ambiente)? *(Testável: gate R27 estendido com varredura de C ∈ {primo, par, ímpar} e distribuições de ação extremas.)*
- **RQ3.2** — Quanto do ganho claimado de "economia de RBs" ([L55]: ">60%") sobrevive quando a cota é verificada no MAC? *(Testável: comparar consumo com telemetria R27 ativa vs estimado pela ação proposta, no mesmo run.)*

**Posicionamento do artigo novo:** primeiro protocolo de verificação ação→agendamento (telemetria por RBG + gates determinísticos) aplicado a um controlador quantílico, tornando o custo `γ·a/C` observável.

---

## G4 — Percentuais de economia internamente inconsistentes e sem denominador

**Definição** (`README.md` §4): SQR alega ">60% de economia de RBs" (Sec. V) vs "20%" (Conclusão), sem denominador/agregação definidos.

**Evidência na literatura:**
- O próprio par de claims está na linha [L55]+10.1109/ccnc65079.2026.11366589 (campo `key_results`, conferido contra o PDF via `tradução.md` — log local 132). **[MATRIZ-full + EQUIPE]**
- Prática da amostra: métricas sem denominador explícito são comuns — PRB ratio via CDF por slice [L07]+10.1109/tmc.2022.3188013; "consome 0,3×/0,6× menos recursos" sem agregação declarada [L31]+10.1109/icnc59896.2024.10556357; "9% mais recursos"/"19% menos consumo" [L26]+10.1109/jsac.2023.3336156 (abstract; agregação não reportada). **[MATRIZ-full/abs]**
- Só [L23]+10.1109/twc.2022.3195570 define a métrica de consumo com régua temporal explícita (RBs alocados por estágio de 50 ms, violações por estágio e cumulativas = regret). **[MATRIZ-full]**

**Pergunta de pesquisa testável:**
- **RQ4.1** — A conclusão "SQR economiza RBs" é sensível à escolha de denominador (UE-tempo × slice-tempo × célula-tempo)? *(Testável: recomputar a mesma campanha de 40 runs sob os três denominadores + IC 95%; reportar onde a ordem de grandeza muda — nossa infraestrutura já calcula ambos os níveis R19/R26/R23.)*

**Posicionamento do artigo novo:** métricas com denominadores pré-declarados, ambas as granularidades reportadas, IC 95% por seed — e a inconsistência 60%×20% documentada como caso de estudo de relato científico.

---

## G5 — O-RAN apenas conceitual no artigo; validação E2 sem paridade de caminho na literatura

**Definição** (`README.md` §4): o SQR afirma "compatível com near-RT RIC" sem demonstrar E2/RIC/xApp real.

**Evidência na literatura (a parte que ENFRAQUECE a novidade — citar obrigatoriamente):**
- **A validação E2 real existe e está madura**: ns-O-RAN (KPM+RC ASN.1 com RICs OSC/OpenRAN Gym) [L06]+10.1145/3592149.3592161; NORI (E2AP v2.02.03 + E2SM-KPM v2.0.3 + xApp de PRB) [L12]+10.14209/sbrt.2025.1571157244; FlexRIC (RTT/2, −83% CPU) [L11]+10.1145/3485983.3494870; xApp RL PPO sobre Colosseum [L31]+10.1109/icnc59896.2024.10556357; AI dentro do RIC com latência medida [L14]+10.1109/icccn69946.2026.11662852; REAL (PPO, loop E2 em SDR) [L15]+10.1109/iccworkshops67674.2025.11162144; xDevSM (portabilidade multi-stack, 4,5–7 ms) [L20]+10.1109/tnsm.2026.3733344; ORANSlice (plataforma física) [L16]+10.1145/3636534.3701544. **[MATRIZ-full]**
- **O que NENHUMA das 60 entradas reporta**: comparação de **paridade** entre caminho de controle local (IPC/shared-memory) e caminho E2 (decisões idênticas bit a bit + delta de latência p50/p95). [L14] mede latência dentro do RIC; [L15]/[L31] medem QoS via E2; nenhuma valida que o controlador se comporta identicamente nos dois caminhos. **[BUSCA-LIM]**
- **[EQUIPE]** paridade IPC×E2: 3585/3585 decisões idênticas, 1199/1199 janelas de telemetria, latência E2 p50 2,39 ms / p95 9,28 ms (budget 50 ms), custo E2 +1,23 ms (p50) — consistente em ordem de grandeza com [L14] (0,5–4 ms) e [L20] (4,5–7 ms). **Limitação declarada: modo DEGRADED** (e2sim local; Near-RT RIC real em k8s pendente de decisão do usuário).

**Perguntas de pesquisa testáveis:**
- **RQ5.1** — O caminho E2 introduz divergência de decisão (reordenamento, timeout, perda) vs IPC em régua de 100 ms sob carga? *(Testável: paridade bit a bit estendida para régua de carga com indicações concorrentes e perda injetada; métrica: taxa de divergência + p99 de latência.)*
- **RQ5.2** — O comportamento γ=0 vs γ=1 do SQR sobrevive à latência adicional do caminho E2 (estabilidade do loop com atraso)? *(Testável: mesma campanha via `--controlPath e2`, 10 seeds, comparada à campanha IPC.)*

**Posicionamento do artigo novo:** não reivindicamos o primeiro xApp E2 sobre ns-3 (a linhagem ns-O-RAN/NORI [L06, L12] e os xApps RL [L15, L31] nos precedem e devem ser citados como tais). Reivindicamos: **primeira validação de paridade de caminho de controle IPC×E2 para um controlador de slicing** + a ponte controlador quantílico↔E2SM-KPM/RC. A validação com RIC real (OSC Release I em k8s) completa o G5 e é pré-requisito para não repetir o padrão "O-RAN conceitual" que criticamos.

---

## G6 — Regime operacional ignorado (viável × marginal × inviável)

**Definição** (`README.md` §4): sob inviabilidade física, observamos `A_safe` permanentemente vazio (100% modo incerto), τ no piso, M no teto, γ=0 ≈ γ=1 — os efeitos claimados só existem num regime viável que o artigo nunca caracteriza.

**Evidência na literatura:**
- O SQR não caracteriza regime algum [L55]+10.1109/ccnc65079.2026.11366589 (cenário único, sem sweep de carga/raio). **[MATRIZ-full + EQUIPE: 100% modo incerto em ambas as variantes]**
- Trabalhos que **deliberadamente estudam** regime de sobrecarga: DORA (congestionamento induzido em RFsim) [L37]+10.1109/mecom67453.2025.11439664; RSLAQ (5 cenários low→stressed em MATLAB) [L18]+10.1109/tmc.2026.3666787; RAWS (restrição dura de inviabilidade como mecanismo de segurança) [L27]+10.1109/jsac.2020.3041405. **[MATRIZ-full]** — mas nenhum define **taxonomia de regime** para o controlador (viável/marginal/inviável) nem métrica de diagnóstico do **estado interno** do agente (ex.: diversidade de modos, esvaziamento de `A_safe`). **[BUSCA-LIM]**
- Métrica de risco de cauda como vizinhança: CVaR₀,05 [L28]+10.1109/ojcoms.2026.3699475 (abstract; full-text fechado) e taxa do 5-percentil no objetivo [L61]+10.1109/eucnc/6gsummit63408.2025.11037178 — instrumentos quantílicos que poderiam diagnosticar regime, mas nenhum dos dois é usado para esse fim. **[MATRIZ-abs/full]**
- Inviabilidade física como região de operação: [L54]+10.1186/s13638-022-02133-3 (cobertura de célula grande inviável com specs correntes) — existe na engenharia de cobertura, não entra na literatura de controladores RL. **[MATRIZ-abs]**

**Perguntas de pesquisa testáveis:**
- **RQ6.1** — Existe um preditor de regime acessível ao operador (ex.: fração de janelas com `A_safe` vazio, ou τ adaptado) que antecipe a perda de separação γ=0/γ=1? *(Testável: treinar/regredir o rótulo de regime — definido pelo sweep de raio — sobre telemetria interna do agente na campanha de 40 runs + sweep.)*
- **RQ6.2** — A degradação sob inviabilidade é *graceful* (violations crescem suavemente com o déficit) ou há ponto de transição abrupto? *(Testável: sweep fino de raio/potência, 10 seeds por ponto, com quebra estrutural testada na curva de violação.)*

**Posicionamento do artigo novo:** primeira **análise de regimes** para controladores quantílicos de slicing (taxonomia viável/marginal/inviável + diversidade de modos como diagnóstico interno), conectando a engenharia de cobertura [L54] à literatura de RL [L37, L18, L27].

---

## G7 — Granularidade de SLA (per-UE vs per-slice) não harmonizada

**Definição** (`README.md` §4): SQR opera per-UE (p5 do throughput, SLA 0,5/1,0/1,5 Mbps); KBRL per-slice; as Figs. 2–3 do artigo comparam as duas sem denominador comum.

**Evidência na literatura:**
- As duas granularidades coexistem sem conversão: per-slice em KBRL [L23]+10.1109/twc.2022.3195570, CMARS [L26]+10.1109/jsac.2023.3336156, RAWS [L27]+10.1109/jsac.2020.3041405, [L31]+10.1109/icnc59896.2024.10556357; per-UE em SQR [L55]+10.1109/ccnc65079.2026.11366589; hierarquia inter-slice × intra-slice (PF) explicitamente discutida em [L17]+10.1109/tmc.2024.3476338 (abstract) e [L05]+10.1109/tcomm.2024.3519550. **[MATRIZ-full/abs]**
- Nenhuma entrada reporta **ambas** as métricas com métrica primária pré-declarada para comparação entre métodos de granularidades distintas. **[BUSCA-LIM]**
- **[EQUIPE]** campanha reportando ambos os níveis: violação per-UE 84,27–89,94% vs per-slice 84,93–89,76% (IC 95%) — a ordenação dos métodos se mantém, mas os ICs de SQR γ=0/γ=1 se sobrepõem quase totalmente em ambos.

**Pergunta de pesquisa testável:**
- **RQ7.1** — A escolha de granularidade inverte a ordenação de métodos? *(Testável: no dataset existente das 40 runs, testar não-equivalência de ordenação per-UE × per-slice com IC 95%; estender com um método intrinsecamente per-slice (KBRL puro sem adaptação per-UE) vs per-UE em regime heterogêneo (G8).)*

**Posicionamento do artigo novo:** protocolo de reporte duplo (per-UE primário, per-slice secundário, pré-declarado) aplicado à comparação SQR×KBRL — resolvendo no nível de relato o conflito que [L55] carrega desde a escolha do baseline [L23].

---

## G8 — Sem heterogeneidade de serviços nem comparação com DRL moderno; coordenação conjunta consciente de risco (SQR-JC)

**Definição** (`README.md` §4): o SQR avalia só 3 slices eMBB; cita RSLAQ/DDQN/GAN-DDQN sem avaliá-los; a projeção proporcional global pode tornar ações localmente seguras em globalmente inseguras — a proposta SQR-JC (coordenação conjunta consciente de risco) substitui essa projeção.

**Evidência na literatura — antecedentes que limitam a novidade (citar obrigatoriamente):**
- **Safe/constrained RL para slicing é campo populoso**: policy-reuse TL seguro [L24]+10.1109/jsac.2023.3336191; SafeSlice (cost model aprendido + A2C; **projeção para a ação segura mais próxima** — análoga direta da nossa projeção Π) [L25]+10.1109/icmlcn64995.2025.11140546; CMARS (MARL restrito multi-agente — **coordenação entre slices já existe**) [L26]+10.1109/jsac.2023.3336156; RAWS [L27]+10.1109/jsac.2020.3041405; CVaR-PPO (Dirichlet + CVaR₀,05) [L28]+10.1109/ojcoms.2026.3699475; risk-aware SAC offline [L29]+10.1109/tmlcn.2025.3534139; Safe-NORA [L56]+10.1145/3583780.3615043; interior-point + safety layer [L57]+10.1109/icnp49622.2020.9259378; CLARA [L58]+10.1109/bigdata52589.2021.9671840; CaDRL (Lagrangiano, E2E multi-domínio) [L59]+10.1109/icnp52444.2021.9651934. **[MATRIZ-full/abs]**
- **Distributional/quantílico neural**: GAN-DDQN [L30]+10.1109/jsac.2019.2959185 (o baseline clássico citado pelo SQR); QR-DRL com CSI imperfeito [L60]+10.1109/twc.2021.3107520; offline + QR-DQN/CQR (taxa do 5-percentil no objetivo — primo exato do quantil-τ do SQR) [L61]+10.1109/eucnc/6gsummit63408.2025.11037178. **[MATRIZ-full]**
- **Heterogeneidade eMBB/URLLC/mMTC com SLA**: feita em stack real por [L15]+10.1109/iccworkshops67674.2025.11162144 e [L37]+10.1109/mecom67453.2025.11439664, e em MATLAB por RSLAQ [L18]+10.1109/tmc.2026.3666787 (baseline DRL citado e não avaliado pelo SQR; nosso ambiente MODA deriva do ambiente rslaq — `rslaq_env/rslaq_kpis` presentes no fork). **[MATRIZ-full]**
- **O que a amostra não tem**: análise publicada do **modo de falha** da projeção global (ação segura local → insegura global) nem coordenação conjunta com modelo de risco **quantílico-kernel sem rede neural**. CMARS coordena, mas com restrição no treino (não quantílica, não online); SafeSlice projeta, mas com cost model neural e sem cota verificável no MAC. **[BUSCA-LIM]**

**Perguntas de pesquisa testáveis:**
- **RQ8.1** — A projeção proporcional gera violações globais que uma coordenação conjunta consciente de risco (SQR-JC) evita, a que custo de consumo? *(Testável: implementar SQR-JC no `src/nsoran/sqr/` existente; campanha pareada SQR-Π × SQR-JC × KBRL × equal, 10 seeds, métricas R19/R26; Hipótese nula: violação per-UE igual entre Π e JC.)*
- **RQ8.2** — O resultado se mantém com heterogeneidade eMBB+URLLC+mMTC (onde URLLC amplifica o custo de projeção errada)? *(Testável: cenário multi-serviço do ambiente rslaq já presente no MODA; mesma campanha.)*
- **RQ8.3** — SQR-JC × SafeSlice-style (cost model neural) × CMARS-style (restrição no treino): onde cada mecanismo de segurança falha primeiro sob regime inviável (G6)? *(Testável: port dos três mecanismos para o mesmo cenário calibrado — comparabilidade restaurada por construção, ver `evidence-report.md` §6.)*

**Posicionamento do artigo novo:** SQR-JC não é o primeiro mecanismo de coordenação/constraint (CMARS [L26], CaDRL [L59] e SafeSlice [L25] nos precedem) nem o primeiro uso de quantis em RRM ([L61], [L60]). A combinação inédita (na amostra): modelo de risco **quantílico-kernel online sem rede neural** + **coordenação conjunta** + **cota verificável no MAC (G3)** + **regimes de viabilidade (G6)** + **validação E2 (G5)**. O experimento RQ8.3 posiciona esses mecanismos no mesmo ambiente — comparação que nenhum dos antecedentes faz entre si.

---

## Gaps transversais emergentes da matriz (além de G1–G8)

| # | Gap | Evidência | Pergunta testável |
|---|---|---|---|
| **N1** | Déficit estatístico é **do campo**, não só do artigo: 1/60 com seeds+IC | §5 de `evidence-report.md`; [L23] único exemplar; [L24] seleção pós-hoc "melhores 64 runs" | RQ-N1: adotar IC por seed como critério de revisão em slicing+RL muda conclusões publicadas? (Testável: recomputar 3 resultados claimados da amostra com campanha pareada — começando pelo SQR.) |
| **N2** | Paridade de caminho de controle (IPC×E2) ausente na validação O-RAN | §4.2 de `evidence-report.md`; [L14], [L15], [L31] medem latência/QoS, nenhuma mede divergência de decisão | RQ5.1 acima (já cobre N2). |
| **N3** | Ferramental de calibração 5G-LENA [L45, L47, L43, L50] não é usado por nenhum trabalho de controlador RL da amostra | Cruzamento das famílias de ambiente (§4.1) | RQ1.1 acima (já cobre N3). |

---

## Tabela-síntese: gap → evidência forte → antecedente a citar → RQ

| Gap | Evidência mais forte | Antecedente mais próximo (obrigatório citar) | RQ principal |
|---|---|---|---|
| G1 | [EQUIPE] sweep 0→88% + déficit 25–30 dB; [L47], [L54] | [L37] (regime em stack real) | RQ1.1 |
| G2 | 44/60 sem artefato; KBRL público vs SQR fechado [L23 vs L55] | [L23] (padrão-ouro de relato) | RQ2.1 |
| G3 | [EQUIPE] gate R27 + 75–77 RBGs ociosos/slot | [L37] (cota em OAI real, sem gate público) | RQ3.1–3.2 |
| G4 | Claim 60%×20% em [L55]; sem denom. em [L07, L26, L31] | [L23] (métrica por estágio) | RQ4.1 |
| G5 | [EQUIPE] paridade 3585/3585 + p95 9,28 ms | [L06], [L12], [L14], [L15], [L31], [L20] (E2 real — **nos precedem**) | RQ5.1–5.2 |
| G6 | [EQUIPE] 100% modo incerto; γ=0≈γ=1 | [L37], [L18], [L27] (regimes de carga, sem taxonomia de controlador) | RQ6.1–6.2 |
| G7 | [EQUIPE] ambos os níveis reportados | [L17], [L05] (hierarquia inter/intra-slice) | RQ7.1 |
| G8 | 10 antecedentes safe/constrained + 3 distributional | **[L25], [L26], [L59]** (projeção/coordenação/constraint — **limitam a novidade**) | RQ8.1–8.3 |

---

## Avisos de honestidade de busca

1. **[BUSCA-LIM] não é prova de inexistência**: as afirmações "não encontrado na amostra" (paridade IPC×E2; taxonomia de regime; auditoria de reprodutibilidade; falha da projeção global) repousam sobre 60 entradas com as limitações de `evidence-report.md` §7 (top-N do OpenAlex, inglês, 1 dia, 33 abstract-only — *[CORRIGIDO: era 29]*). Antes da submissão: reexecutar eixos D e B5, e buscar ativamente por **refutações** ("projection failure", "action shadowing", "safe action composition") nos venues ns-3/WNS3, ICMLCN, CoNEXT 2024–2026.
2. **Linhas abstract-only não sustentam gap sozinhas**: os gaps declarados acima combinam (a) full-texts lidos, (b) evidência experimental própria, e usam [MATRIZ-abs] apenas como corroboração contextual ([L17], [L26], [L28], [L29], [L54]).
3. **Riscos de sobreposição com publicações 2026 não indexadas**: [L28] (CVaR slicing, OJ-COMS 2026) e [L48] (SLA-driven autonomic, LATINCOM 2026) são os vizinhos temporais mais próximos; obter full-texts antes do claim de novidade do SQR-JC.
4. **Aprovação pendente**: conforme a skill, nenhuma redação de manuscrito começa antes de o pesquisador revisar este documento e o `evidence-report.md`.
