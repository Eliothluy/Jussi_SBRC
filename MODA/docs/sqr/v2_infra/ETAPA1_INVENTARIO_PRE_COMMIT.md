# Inventário Pré-Commit e Auditoria de Integridade (`Etapa 1` — `sqr-v2`)

> **Data:** `2026-11-17` · **Branch inicial:** `main` · **HEAD pré-commit:** `3afe72cc80092039aca5cd22065db4812e9ee3cc` · **Remote `origin/main`:** `060d0581f950d041fb5c4f6482259528089cf019` (intocado)

## 1. Estado do Repositório Antes do Commit na `main`

Foram inspecionados todos os arquivos modificados (`1`) e não rastreados (`20`) presentes na árvore de trabalho antes de qualquer `git commit`, criação da tag local `v1.0-b0-frozen-audit-2026-11-17` ou abertura da branch `exp/sqr-v2-infra`:

| Status Git | Arquivo | Tamanho (bytes) | SHA-256 | Destino na Etapa 1 |
|---|---|---:|---|---|
| `M` | `README.md` | 25946 | `d326b75f1247cc9bbb99086f37eb7dd342554eb885482dbb12ded95947ffd27c` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/PLANO_CORRECAO_INFRAESTRUTURA_V2.md` | 48586 | `368a2bbdc7988f42e822496c76648a1faacb1c1f06ec2eb03236ef497d4d3ad5` | Incluído no commit documental `main` (`v1.1`) |
| `??` | `MODA/docs/sqr/auditoria_bugs_ns3_sqr_base_2026-11-17.md` | 32929 | `660c2dd5da8aed1c3aa4d9a76c0440646d084007b1c06515f4411c1010cd8d87` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/auditoria_especialista_oran_3gpp_2026-11-17.md` | 49378 | `6bc27891616ba813618c6ac96582d4579e0fc9dbe2e68238d043a845df8678d5` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/validacao_pesquisador_senior_2026-11-17.md` | 33827 | `f1e71e18f6250689caa3257ccce770864214202430ee3815c63d9d3f3b0bdad7` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/jc_decision/A5_GATE1_DIAGNOSTICO_RETROSPECTIVO.md` | 26262 | `139196c624b445f72450f482d22853e43ca9b5f0792718d7bd5ac0be8b49c0a9` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/jc_decision/A5_REVISAO_MATEMATICA_E_PLANO_TESTES.md` | 25442 | `f274c37a1255bc131999ce9614c505068bda61bf516dfc33faec3dbfe29b7c53` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_01_INVENTARIO_E_DADOS.md` | 21159 | `4ddd3fb32b66c42983fe7d43df45463d439dbf2884e9f18f4704b22cd51ab248` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_02_DISCRIMINACAO.md` | 26863 | `a89958c67aae4e105018921f2dd98c0c822fe1e18b3712aff6cf1bf40b113b9c` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_03_ORIGINALIDADE_E_ESCOLHA.md` | 14439 | `d0883f75191350f7fc9f7663c8e3038d3a855dacb07f190ab583068967bb728c` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_ESTADOS.csv` | 21931 | `f8b3ffed9adc30f2f2f2636b87b5cb270ee803dda5b8508904ea696f286066e3` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/jc_decision/GATE_CONTEXTUAL_PARECER_FINAL.md` | 8757 | `2f28f5667cb9a0f6b99f7dbc5990321f18abe522f5a9540c05af5e9ae96f57b7` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/jc_decision/figures/fig1_proposta_vs_baselines.png` | 860275 | `c8d1808f57115ea8889e94051bd9953bf117f34b2c1c8707243ca5af7a96e10d` | Incluído no commit documental `main` |
| `??` | `MODA/docs/sqr/jc_decision/figures/fig2_diagnostico_separabilidade_teto.png` | 514483 | `de133c7fd169eafbc0faf74c2da8b400ce10c98d5d04bfd3a0e5284f7e8a16c9` | Incluído no commit documental `main` |
| `??` | `MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_RETROSPECTIVE_ANALYSIS.json` | 504468 | `45fb78975c581f78e5da4416760076bb357c57afd9061e32321cd1f5ca0272f0` | Incluído no commit documental `main` |
| `??` | `MODA/ns-o-ran-gym/results/sqr_jc_decision/A5_GATE1_RETROSPECTIVE_TABLES.csv` | 46603 | `7ea811518c2c8dc9d6a9877b8231a08dba49a935ad23c9b283ce5bb79c947143` | Incluído no commit documental `main` |
| `??` | `MODA/ns-o-ran-gym/results/sqr_jc_decision/GATE_CONTEXTUAL_ANALYSIS.json` | 87902 | `8b9ca0ffc606ce6f96fe53a5d409d3eefa0836956c28456ed9c8158b17277bc3` | Incluído no commit documental `main` |
| `??` | `MODA/ns-o-ran-gym/results/sqr_jc_decision/gate1_retrospective_analysis.py` | 45151 | `e83d322ee7859c541acaf5dd3c837634bca38432ebe56b05e7439e848c44f606` | Incluído no commit documental `main` |
| `??` | `MODA/ns-o-ran-gym/results/sqr_jc_decision/gate_contextual_analyze.py` | 31481 | `e6d10fe2370f7159ae28df800c387cacd053cb4eebca1c5deb39cc49089a1870` | Incluído no commit documental `main` |
| `??` | `MODA/ns-o-ran-gym/results/sqr_jc_decision/generate_gate1_comparison_charts.py` | 18645 | `738d5641e934825c349d819771966f3a031dd6059e0709212a2695dd5e009abb` | Incluído no commit documental `main` |
| `??` | `MODA/docs/superpowers/specs/Auditoria.txt` | 20612 | `cfe517c6efc96691d2630d9b72526eb681b460b95e8772b90c9acd5ebe177924` | **Preservado intocado em disco** (não estagiado / não sobrescrito) |

## 2. Integridade Pré-Commit de `B0_FROZEN_MANIFEST.json` (`17/17` Código/Binário + `5/5` Checkpoints)

- Verificação executada em `2026-11-17` contra [`MODA/docs/sqr/jc_decision/B0_FROZEN_MANIFEST.json`](../jc_decision/B0_FROZEN_MANIFEST.json):
  - `git_provenance.code_sha256`: **`17/17` correspondências exatas** (incluindo `MODA/ns-3-dev/build/scratch/sqr/ns3.46-sqr-sim-default` = `6403e2ebdc7a7318adf18f37acb674134a9fc412421c3cbb0dd1d5528c9eb0b1`);
  - `verified_dev_checkpoints`: **`5/5` correspondências exatas**;
  - Suíte legada `bash MODA/ns-o-ran-gym/tests/run_sqr_tests.sh -x -q`: **`360 passed`**.
- **Regra remota:** nenhuma operação `git push`, `fetch` destrutivo ou modificação em `origin/*` será executada.
