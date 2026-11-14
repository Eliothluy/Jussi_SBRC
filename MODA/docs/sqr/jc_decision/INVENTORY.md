# INVENTORY.md — D0-B (mapa real do checkout, verificado 08/11/2026)

**Commit:** `b9f3c0e` (main) + artefatos out/nov não-commitados (hashes em `audit/D0A_input_inventory.csv`; working tree dirty — resultados e código novo da Etapa 2.5/D0 ainda sem commit, aguardando pedido explícito de push). **Ambiente:** venv `.venv` (Python 3.13) em `MODA/ns-o-ran-gym/`; ns-3 via `bash MODA/ns-3-dev/scratch/sqr/ns3_gate.sh` (wrapper Python 3.13 — o `./ns3` direto quebra com o 3.14 do sistema, AGENTS.md §3.1); testes `tests/run_sqr_tests.sh` / `run_full_tests.sh` (suíte atual: 342 passed).

## 1. Código do B0 (SQR-HET) e do learner

| Peça | Caminho (verificado) | Papel |
|---|---|---|
| Regressor quantílico kernelizado | `src/nsoran/sqr/quantile_regressor.py` | pinball online (subpred α=+ητ / overpred α=η(τ−1) / crítico ×P_penalty=2), τ adaptável, memória B |
| Controlador por slice | `src/nsoran/sqr/sqr_controller.py` (`SqrSliceController`) | safe-set f_τ(z,a) ≥ Λ, seleção (confident/ε-softmax/kNN-safety-first/fallback), margem a←a+round(σ·M), adaptação τ/M (R11, W=100) |
| **Projeção de capacidade** | `src/nsoran/sqr/coordinator.py` (`project_prbs` + `apply_virtual_updates`) | corte largest-remainder Σ≤C + **updates virtuais negativos R12** na proposta pré-projeção |
| **B0 heterogêneo** | `src/nsoran/sqr_jc/sqr_het.py` (`SqrHetController`) | 3×`SqrSliceController` audited (import, não reimplementado) + margem normalizada comum (Λ=0) + z_i 6 dims SEM features conjuntas + projeção original |
| Escalonamento de features | `src/nsoran/sqr_jc/feature_scaler.py` (+ manifest CAL congelado) | regras por feature (p1/p99 CAL p/ contagens/filas/ofertas; escalas naturais p/ KPIs/SNR) |
| SLA/margem | `src/nsoran/sqr_jc/sla_margin.py` + `sla_manifest.yaml` (v1.1-g4b) | R36–R38; primária URLLC = p95≤10 ms (miss≤5% secundária; G4-B: agregação rolling 1 s) |
| Driver de campanha | `examples/sqr_run.py` | `--controller {sqr,kbrl,equal,prop,replay}` — **`sqr_het` AINDA NÃO ESTÁ WIRED (gap D1)** |
| Replay | `src/nsoran/sqr/replay_controller.py` + `--controller replay --replay_base <arm> --replay_at_win t --replay_action ae,au,am` | determinístico (prefixo bit-idêntico comprovado 38× II-b + 10× D0-A + piloto 813/813) |

## 2. Manifests e contratos (hashes no inventário D0-A)

- `sla_manifest.yaml` v1.1-g4b (`src/nsoran/sqr_jc/`) — bloco g4b §19, status STOP_FINAL.
- `resource_unit_manifest.md` (`results/sqr_jc_final/01_resource_unit/`) — C=53=53RBGs=53PRBs, ALLOCATED_USAGE, 27.720 ações; **ERRATA aplicada no docstring `sqr_env.py:98-100`** (narrativa "13×4+1" refutada; número/telemetria corretos).
- Contrato IPC/ação/KPM: `docs/sqr/contrato_interface.md` (v1.4; `sqr-kpms.txt` 17 col., `rslaq_actions_for_ns3.csv`, `sqr_prb_windows.csv` com `prb_requested/executed/scheduled/idle` por slice — INV-05).
- Assumptions register: `docs/sqr/assumptions_register.md` v2.3 (R01 já correto na linha; errata no código aplicada).
- Protocolo R19: exclusão de janelas vacuous/parciais dos outcomes (adjudication `sla_ok` vacuous=OK congelado p/ julgamento de janela; FFR/FIR sobre completas — G4-A/B).

## 3. Dados existentes (para D1/D2 — evitar duplicação)

- **B0 SQR-HET: NUNCA rodou** no cenário heterogêneo (choices não inclui; nenhum log). D1 exigirá wired `--controller sqr_het` + log por decisão (schema §4.2 do protocolo: a_raw/a_after_margin/a_after_projection/a_sent_to_mac + predições antes/depois).
- Baselines estáticos: 30 runs `hetero_sweep{,_prop}` (equal×prop, λ{2,4,5,6,7}×3 seeds, 300 s) — fonte dos estados B0-substitutos até o SQR-HET rodar.
- Replays: 28 (II-b) + 31 re-exec D1–D5 + 18 D6 + 13 v2 (E1/E2x/E3x/E4x) + piloto — todos com A/A de prefixo; runner `oracle/run_replays.sh` (formato de args verificado; `LC_ALL=C`).
- CAL/G4B: 9+6+9 runs (`results/sqr_jc_final/03_calibration/`) — kernels calibrados ls=2.0/thr=90, G4-B STOP FINAL.

## 4. Fidelidade de replay (evidência acumulada)

1. Piloto II-a: 813/813 linhas UE bit-idênticas ao run congelado (exceto `resourceSharePct` da própria janela override).
2. II-b: 28/28 válidos (prefixo KPM bit-idêntico + prb + delays por entrega).
3. D0-A: recomputação offline 10/10 prefixos (KPM e delays) bit-idênticos.
4. **D0-B (novo): re-execução A/A fresca de um replay E1** — `results/sqr_jc_decision/d0b_aa/E1_lam5_s1_t1955_AA/` (resultado no `D0B_baseline_audit.md`).

## 5. Testes relevantes executados (D0-B)

`tests/test_sqr_coordinator.py` (projeção+virtuais), `tests/test_sqr_jc_controller.py` (incl. SQR-HET §27), `tests/test_sqr_replay.py`, `tests/test_sqr_static_prop.py` — executar subset nesta etapa; suíte integral 342 (validada 04/11).
