# Paridade IPC x E2 — SQR xApp (Fase 5.4)

- IPC run: `/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_e2_parity/run_20261014_192020_sim120s_seed1_g1/ipc` (sim out: `/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_e2_parity/run_20261014_192020_sim120s_seed1_g1/ipc/171f38f6-990b-4689-9c35-d2256df719cb/results_sqr_network_only/seed=1_run=1`)
- E2 run:  `/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_e2_parity/run_20261014_192020_sim120s_seed1_g1/e2` (sim out: `/home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-o-ran-gym/results/sqr_e2_parity/run_20261014_192020_sim120s_seed1_g1/e2/e2_seed=1/results_sqr_network_only/seed=1_run=1`)

| Verificação | IPC | E2 | Resultado |
|---|---|---|---|
| Janelas de decisão (step_metrics) | 3585 linhas | 3585 linhas | IDÊNTICAS |
| &nbsp;&nbsp;divergências `timestamp_ms` | — | 0 |
| &nbsp;&nbsp;divergências `mode` | — | 0 |
| &nbsp;&nbsp;divergências `raw_action` | — | 0 |
| &nbsp;&nbsp;divergências `action` | — | 0 |
| &nbsp;&nbsp;divergências `prb_requested` | — | 0 |
| &nbsp;&nbsp;divergências `prb_executed` | — | 0 |
| Janelas de telemetria prb_* (ns-3) | 1199 | 1199 | IDÊNTICAS |
| Divergência esperada `prb_requested_pct_*` (inteiro RIC) | — | 1198 janelas, máx |Δ| = 0.9600 p.p. | divergência de transporte documentada (ASN.1 INTEGER, Fase 5.3) |
| Linhas KPM (sqr_kpm_history) | 1213 | 1213 | IDÊNTICAS |
| Violações por-UE-janela (R19) | 994 | 994 | IGUAIS |

## Latência de decisão (orçamento contrato §5: 50 ms)

| Caminho | métrica | n | p50 (ms) | p95 (ms) | máx (ms) |
|---|---|---|---|---|---|
| IPC (semáforos+CSV; act only) | decision_time_ms | 1195 | 1.289 | 7.912 | 10.528 |
| E2 xApp — núcleo act() | decision_ms | 1195 | 1.282 | 8.058 | 10.094 |
| E2 xApp — indication→controle | handle_ms | 1195 | 2.342 | 9.217 | 11.317 |
| E2 ponte lado gNB (UDP JSON) | rtt_wall_ms | 1199 | 2.407 | 9.31 | 11.408 |

- Timeouts da ponte (degradação §5 contável): 0
- IPC p95 = 7.912 ms vs orçamento 50 ms → DENTRO
- E2 p95 (end-to-end) = 9.217 ms vs orçamento 50 ms → DENTRO
- Diferença de latência observada (p50 E2 − p50 IPC) = 1.053 ms; (p95: 1.305 ms)


**PARITY OK**
