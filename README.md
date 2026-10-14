# Jussi_SBRC — Reprodução e extensão de "Safe RAN Slicing in O-RAN" (SQR, IEEE CCNC 2026)

Reimplementação do método **SQR** (Safe Quantile Regression — regressão quantílica online com kernel Matérn, *model-based RL* sem rede neural) do artigo *Tuerxun & Nakao, CCNC 2026* (DOI [10.1109/CCNC65079.2026.11366589](https://doi.org/10.1109/CCNC65079.2026.11366589)) sobre o ambiente **MODA** (ns-3.46 + 5G-LENA v5.2 + Python), com baseline **KBRL** e validação planejada com protocolo O-RAN **real** (E2/RIC via NORI).

> **Missão em evolução:** o que começou como reprodução tornou-se uma **auditoria empírica do estado da arte** — a reprodução rigorosa expôs gaps que sustentam um artigo novo (ver §4).

---

## 1. Arquitetura

```
ns-3.46 + 5G-LENA v5.2 (fork local em MODA/ns-3-dev)     Python (MODA/ns-o-ran-gym)
  scratch/sqr/sqr-sim.cc          ←IPC: semáforos POSIX + CSV→         src/nsoran/sqr/
  SqrMacScheduler (cota absoluta                                       kernel Matérn + regressor quantílico
  de RBGs por slice, RBGs idle,                                        + política hierárquica + projeção Π
  prb_requested/executed/scheduled/idle)                               + KBRL + equal (baselines)
                                                                  examples/sqr_run.py (loop fechado γ∈{0,1})
```

- **Cenário fiel à Tabela I**: 1 gNB, 2 GHz, μ=0, 10 MHz (C efetivo = 53 PRBs), 30 dBm, 15 dBi, UMa, Poisson 2 UE/min, sessão exp. 60 s, 3 slices eMBB, SLA/UE 0,5/1,0/1,5 Mbps, período de decisão 100 ms.
- **Documentação-fonte**: [`AGENTS.md`](AGENTS.md) (operação), [`PLANO_IMPLEMENTACAO_SQR_NORI.md`](PLANO_IMPLEMENTACAO_SQR_NORI.md) (plano), [`MODA/docs/sqr/`](MODA/docs/sqr/) (registro de premissas `[ARTIGO]/[ASSUMIDO]`, contrato de interface v1.2, rastreabilidade).
- **Gates determinísticos**: `scratch/sqr/smoke_test.sh` (network + IPC-dummy + R27 i–iv + soak 3 seeds), `ns-o-ran-gym/tests/run_sqr_tests.sh` (190 testes), `run_sqr_integration.sh` (loop fechado), `run_sqr_campaign.sh` + `run_sqr_analysis.sh` (campanha com IC 95%).

## 2. Estado (14/10/2026)

| Fase | Status |
|---|---|
| 0–2 — premissas, cenário ns-3 fiel, agente SQR | ✅ concluídas (revisão independente + gates) |
| 3 — baseline KBRL (port fiel) + equal | ✅ concluída (integração validada) |
| 4 — campanha 40 runs (4 métodos × 10 seeds × 600 s) | ✅ concluída — resultado central em §3 |
| — matriz de sensibilidade/calibração realista | ⏳ proposta (§3.2) |
| 5 — O-RAN real via NORI (E2/RIC, SQR como xApp) | ⬜ planejada |

## 3. Contextualização: por que o cenário do artigo é inviável — e o que isso revela

### 3.1 Resultado da campanha (raio 2 km, 30 dBm — parâmetros publicados)

Todos os métodos violam o SLA massivamente, **incluindo baselines sem agente**: equal 84,3% [80,4–88,0], KBRL 88,9%, SQR γ=0 89,6%, SQR γ=1 89,9% (IC 95%, 10 seeds). Causa raiz quantificada: **56,6% das janelas com demanda têm entrega nula** — UEs de borda em loop de RLF (~2,7 quedas/UE/600 s, buffer RLC zerado a cada reestabelecimento).

### 3.2 O discriminador (mesmo código, só o raio muda; network-only, 240 s, seed 1)

| Raio | RLF | Violação SLA | Throughput mediano/UE |
|---|---|---|---|
| 300 m | 0 | **0,0%** | 7,3 Mbps |
| 800 m | 0 | 9,7% | 2,8 Mbps |
| 1500 m | 10 | 58,7% | 0,86 Mbps |
| 2000 m (artigo) | 36 | **88,0%** | 0,0 Mbps |

A implementação está **autovalidada** (0% de violação quando a física permite; cadeia MAC→RLC→KPM→métrica correta). A inviabilidade é do **parâmetro**, não do código.

### 3.3 O que falta ao cenário para ser "real" (artigo omite; lemos conservador)

| # | Nossa leitura | 5G real | Déficit |
|---|---|---|---|
| 1 | Antena 1×1 omni ("15 dBi") | Array 8×8 dual-pol + beamforming | ~15–20 dB |
| 2 | UMa **NLOS** p/ todos (PL@2km ≈ 148 dB) | Artigo cita TS 36.942 (≈139,4 dB); fração LOS real | ~9 dB |
| 3 | 30 dBm totais | 43 dBm/10 MHz (TR 38.104, macro) | ~13 dB |
| 4 | TDD 3DL/7 (~43% DL) | "TTI 1 ms" sugere FDD/DL cheio | ~2,3× capacidade |

**1+2+3 ≈ 25–30 dB** — exatamente a ordem que encolhe o raio de viabilidade de ~2 km (macro real) para ~0,8–1,2 km (medido). Conclusão: a Tabela I do artigo descreve uma *small cell* transmitindo para uma macro-célula; nenhum simulador fiel à realidade entrega 99% de SLA nessa combinação.

## 4. Gaps identificados no artigo (evidência nossa) → artigo novo

**G1 — Inviabilidade física dos parâmetros publicados.** 30 dBm + 2 km + urban macro é inconsistente com link budget de qualquer padrão (§3). O artigo alega 99% de SLA sem publicar simulador, C, banda, MCS ou scheduler. *Contribuição nossa: análise de viabilidade por link budget + sweep empírico de raio (§3.2) + cenário calibrado realista.*

**G2 — Irreprodutibilidade por omissão.** 12+ parâmetros essenciais não publicados (C/banda/numerologia, tráfego, mobilidade, T_softmax, ν/ℓ do Matérn, regra de poda, adaptação τ/M, peso da atualização virtual, seeds/duração). *Contribuição: protocolo de rastreabilidade `[ARTIGO]/[ASSUMIDO]` + registro de premissas auditável (reutilizável pela comunidade).*

**G3 — Sem evidência de aplicação real da cota no MAC.** O artigo não demonstra que `Σa≤C` chega ao agendamento (RBGs efetivamente ociosos). *Contribuição: gate R27 i–iv com telemetria `prb_requested/executed/scheduled/idle` por janela — sem isso, γ·a/C é inobservável e a Fig. 4 não significa nada.*

**G4 — Percentuais internamente inconsistentes.** Seção V alega ">60% de economia de RBs"; Conclusão alega "20%" — sem denominador/agregação definidos. *Contribuição: métricas com denominadores explícitos (UE-tempo × slice-tempo) e IC 95% por seed.*

**G5 — O-RAN apenas conceitual.** Nenhuma interface E2/RIC/xApp real; "compatível com near-RT RIC" é afirmado, não demonstrado. *Contribuição: SQR como xApp sobre RIC O-RAN SC via NORI (E2SM-KPM + RIC Control), com comparação de paridade e latência IPC vs E2 — validação que o original não fez.*

**G6 — Regime operacional ignorado.** Sob inviabilidade (carga além da capacidade), observamos fenômeno não descrito: `A_safe` permanentemente vazio (100% modo incerto), τ saturando no piso e M no teto; o comportamento de γ=0 × γ=1 fica indistinguível — os efeitos do artigo só existem num regime viável que o artigo nunca caracteriza. *Contribuição: análise de regimes (viável × marginal × inviável) e diversidade de modos como métrica de diagnóstico.*

**G7 — Granularidade de SLA não harmonizada com o baseline.** SQR per-UE × KBRL per-slice sem denominador comum nas Figs. 2–3. *Contribuição: reporte nos dois níveis com métrica primária pré-declarada.*

**G8 — Sem heterogeneidade nem comparação com DRL moderno.** Só 3 slices eMBB; RSLAQ/DDQN/GAN-DDQN citados mas não avaliados. *Herda-se do GUIA: eMBB+URLLC+mMTC, transições de carga e a proposta SQR-JC (coordenação conjunta consciente de risco, substituindo a projeção proporcional que pode tornar ações localmente seguras em inseguras).*

**Artigo proposto (estado da arte):** *"Safe RAN slicing sob incerteza de viabilidade: auditoria empírica, calibração realista e validação O-RAN de controladores quantílicos"* — (i) auditoria de reprodutibilidade do SQR; (ii) análise de link budget/regimes com sweep empírico; (iii) controladores (SQR/KBRL/equal) sobre MAC com cota verificável em cenário calibrado 5G realista; (iv) validação E2 real via NORI.

## 5. Comandos essenciais

```bash
# Build/smoke do cenário (wrapper obrigatório: ns3 quebra no Python 3.14 do sistema)
bash MODA/ns-3-dev/scratch/sqr/ns3_gate.sh build
bash MODA/ns-3-dev/scratch/sqr/smoke_test.sh            # network + IPC-dummy + R27 + soak

# Testes Python (venv 3.13 em MODA/ns-o-ran-gym/.venv)
bash MODA/ns-o-ran-gym/tests/run_sqr_tests.sh           # 190 testes SQR/KBRL
bash MODA/ns-o-ran-gym/tests/run_full_tests.sh          # suíte completa

# Loop fechado e campanha
bash MODA/ns-o-ran-gym/examples/run_sqr_integration.sh       # SQR 120 s
bash MODA/ns-o-ran-gym/examples/run_sqr_integration_kbrl.sh  # KBRL 60 s
JOBS=6 bash MODA/ns-o-ran-gym/examples/run_sqr_campaign.sh sqr-g0,sqr-g1,kbrl,equal 1..10 600
bash MODA/ns-o-ran-gym/examples/run_sqr_analysis.sh          # figuras 2–4 + tabelas IC 95%
```

## 6. Referências rápidas

Artigo SQR (CCNC 2026) · KBRL ([jjalcaraz-upct/network-slicing](https://github.com/jjalcaraz-upct/network-slicing), TWC 2022) · NORI ([lasseufpa/nori](https://github.com/lasseufpa/nori)) · 5G-LENA ([docs](https://cttc-lena.gitlab.io/nr/html/)) · ns-O-RAN Gym ([openrangym.com](https://openrangym.com)) — detalhes e regras operacionais em [`AGENTS.md`](AGENTS.md); protocolo científico em [`GUIA_IMPLEMENTACAO_SQR_O_RAN_PYTHON_NS3.md`](GUIA_IMPLEMENTACAO_SQR_O_RAN_PYTHON_NS3.md).

---
*Resultados com IC 95% e dados brutos: `MODA/ns-o-ran-gym/results/sqr_campaign/` (figuras/tabelas versionadas; runs brutos locais). Registro de premissas e divergências: `MODA/docs/sqr/`.*
