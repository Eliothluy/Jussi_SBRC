# Paridade IPC x E2 — SQR xApp (Fase 5.4)

- IPC run: `/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_e2_parity/run_20261014_184757_sim120s_seed1_g1/ipc` (sim out: `/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_e2_parity/run_20261014_184757_sim120s_seed1_g1/ipc/0b3cf76a-5188-4d8e-b046-a18aefa79559/results_sqr_network_only/seed=1_run=1`)
- E2 run:  `/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_e2_parity/run_20261014_184757_sim120s_seed1_g1/e2` (sim out: `/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_e2_parity/run_20261014_184757_sim120s_seed1_g1/e2/e2_seed=1/results_sqr_network_only/seed=1_run=1`)

| Verificação | IPC | E2 | Resultado |
|---|---|---|---|
| Janelas de decisão (step_metrics) | 3585 linhas | 3585 linhas | DIVERGENTES |
| &nbsp;&nbsp;divergências `timestamp_ms` | — | 3582 |
| &nbsp;&nbsp;divergências `mode` | — | 0 |
| &nbsp;&nbsp;divergências `raw_action` | — | 647 |
| &nbsp;&nbsp;divergências `action` | — | 848 |
| &nbsp;&nbsp;divergências `prb_requested` | — | 848 |
| &nbsp;&nbsp;divergências `prb_executed` | — | 893 |
| Janelas de telemetria prb_* (ns-3) | 1199 | 1199 | DIVERGENTES |
| &nbsp;&nbsp;divergências `prb_executed_rbg_0` | — | 76 |
| &nbsp;&nbsp;divergências `prb_executed_rbg_1` | — | 453 |
| &nbsp;&nbsp;divergências `prb_executed_rbg_2` | — | 368 |
| &nbsp;&nbsp;divergências `prb_scheduled_rbg_1` | — | 303 |
| &nbsp;&nbsp;divergências `prb_scheduled_rbg_2` | — | 524 |
| &nbsp;&nbsp;divergências `prb_budget_rbg_0` | — | 122 |
| &nbsp;&nbsp;divergências `prb_budget_rbg_1` | — | 520 |
| &nbsp;&nbsp;divergências `prb_budget_rbg_2` | — | 530 |
| &nbsp;&nbsp;divergências `prb_idle_rbg` | — | 538 |
| &nbsp;&nbsp;divergências `dl_slots` | — | 446 |
| Divergência esperada `prb_requested_pct_*` (inteiro RIC) | — | 1198 janelas, máx |Δ| = 82.0200 p.p. | divergência de transporte documentada (ASN.1 INTEGER, Fase 5.3) |
| Linhas KPM (sqr_kpm_history) | 1213 | 1213 | DIVERGENTES |
| Violações por-UE-janela (R19) | 994 | 1077 | DIFERENTES |

## Latência de decisão (orçamento contrato §5: 50 ms)

| Caminho | métrica | n | p50 (ms) | p95 (ms) | máx (ms) |
|---|---|---|---|---|---|
| IPC (semáforos+CSV; act only) | decision_time_ms | 1195 | 1.152 | 7.881 | 9.62 |
| E2 xApp — núcleo act() | decision_ms | 1199 | 1.324 | 8.354 | 10.73 |
| E2 xApp — indication→controle | handle_ms | 1199 | 2.369 | 9.584 | 12.009 |
| E2 ponte lado gNB (UDP JSON) | rtt_wall_ms | 1199 | 2.431 | 9.648 | 12.066 |

- Timeouts da ponte (degradação §5 contável): 0
- IPC p95 = 7.881 ms vs orçamento 50 ms → DENTRO
- E2 p95 (end-to-end) = 9.584 ms vs orçamento 50 ms → DENTRO
- Diferença de latência observada (p50 E2 − p50 IPC) = 1.217 ms; (p95: 1.703 ms)

## Exemplos de divergência de decisão
```
step=1 slice=0 timestamp_ms: ipc='0' e2='100'
step=1 slice=1 timestamp_ms: ipc='0' e2='100'
step=1 slice=2 timestamp_ms: ipc='0' e2='100'
step=2 slice=0 timestamp_ms: ipc='0' e2='200'
step=2 slice=1 timestamp_ms: ipc='0' e2='200'
step=2 slice=2 timestamp_ms: ipc='0' e2='200'
step=3 slice=0 timestamp_ms: ipc='0' e2='300'
step=3 slice=1 timestamp_ms: ipc='0' e2='300'
```
## Exemplos de divergência de telemetria aplicada
```
t=65900ms prb_executed_rbg_1: ipc=6 e2=7
t=65900ms prb_scheduled_rbg_1: ipc=282 e2=336
t=65900ms prb_budget_rbg_1: ipc=318 e2=371
t=65900ms prb_idle_rbg: ipc=1539 e2=1485
t=66000ms prb_scheduled_rbg_1: ipc=805 e2=1085
t=66000ms prb_scheduled_rbg_2: ipc=340 e2=442
t=66000ms prb_budget_rbg_0: ipc=23 e2=33
t=66000ms prb_budget_rbg_1: ipc=805 e2=1155
```

**PARITY DIVERGED**
