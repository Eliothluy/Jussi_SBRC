# Fase 5.4 — SQR como xApp no caminho E2 + paridade IPC × E2

- **Data**: 2026-10-14. **Autor**: Agente B (RL/pesquisa).
- **Escopo**: `examples/xapp_sqr.py` (loop do controlador SQR como consumidor do caminho E2), `examples/run_sqr_e2_parity.sh` + `examples/sqr_parity_report.py` (paridade IPC × E2), extensão do lado C++ do loopback em `scratch/sqr/` (`--e2Endpoint=xapp`: ponte de transporte local UDP para o xApp), teste novo `tests/test_sqr_xapp_quant.py`.
- **Base**: PLANO §3-Fase 5.4 ("loop SQR movido para xApp no RIC (KPM via E2, RIC Control a cada 100 ms); ... comparando IPC vs E2: paridade de resultados + overhead de latência E2E (p50/p95)"); contrato v1.2 (§2 janela 100 ms, §3 ação, §4 KPM, §5 degradação); register R01/R09/R18/R19/R22/R24/R27; infraestrutura da Fase 5.3 (`sqr-e2-adapter.{h,cc}`, `--controlPath=e2`, `ApplyQuotaAction` única R27).

## 1. O que foi entregue

| Item | Arquivo | Papel |
|---|---|---|
| xApp SQR | `MODA/ns-o-ran-gym/examples/xapp_sqr.py` | Processo RIC-side: recebe a E2SM-KPM indication de cada janela de 100 ms, **materializa os registros UE no formato exato do contrato §4** (formatação `emitRow`: %.2f/%.4f/%.2f) e reusa o **MESMO** stack da Fase 2/4 (`SqrObservationBuilder`, `SqrSliceController`+Π, `SqrKpmParser` com imputação R09, `apply_virtual_updates` R12, seeds `build_controllers` do `sqr_run.py`) → decide → responde **RIC Control** (estilo RAN slicing, `SlicePRBQuota`) com as cotas. Aprendizado online com o par (z_t, a_final_t, y_t+1) na MESMA ordem do `run_episode` (learn antes do act seguinte, R22). Transporte abstrato: `UdpLoopbackTransport` (DEGRADED, concreto) e `SctpE2simTransport` (FULL, stub documentado — `NotImplementedError` enquanto o e2sim não existir; o núcleo de decisão é agnóstico a transporte) |
| Ponte C++ do loopback | `MODA/ns-3-dev/scratch/sqr/sqr-e2-adapter.{h,cc}` + `sqr-sim.cc` | Modo novo `XAPP_UDP` do `SqrE2LoopbackTerm` (`--e2Endpoint=xapp`, `--e2XappPort` 5560, `--e2XappTimeoutMs` 50, `--e2XappRetries` 1): serializa a indication em JSON (`%.17g` = round-trip exato), envia por UDP/localhost, espera a resposta com budget wall do contrato §5 (timeout = degradação contável, mantém cota anterior — mesma semântica do `sem_timedwait` do IPC), decodifica no **mesmo** `SqrE2RicControlMessage` → `OnRicControlMessage` → `ApplyQuotaAction` (aplicação única R27, compartilhada com o IPC). CSV de evidência `sqr_e2_xapp_bridge.csv` (seq, ts, rtt_wall_ms, applied, timeout) + telemetria `E2_CONTROL ... endpoint=xapp-udp xapp_timeouts=N`. O modo `loopback` (dummy da 5.3) permanece default e intocado (dispatcher) |
| Paridade | `MODA/ns-o-ran-gym/examples/run_sqr_e2_parity.sh` (+ `sqr_parity_report.py`) | Roda o MESMO seed/simTime/γ pelos dois caminhos (`sqr_run.py` IPC; `xapp_sqr.py` E2), compara e imprime tabela + `PARITY OK`/`PARITY DIVERGED`; escreve `parity_report.md` |
| Teste unitário | `MODA/ns-o-ran-gym/tests/test_sqr_xapp_quant.py` (9 testes) | Réplica de `ComputeRbgQuotas` (exemplos documentados da 5.3, caso de borda do `llround`, garantias R27, domínio Σpct≤100) + busca de % inteiros exatos + materialização KPM→parser |

## 2. Desenho do caminho E2 com xApp (DEGRADED)

```
sqr-sim (borda de janela t)                       xapp_sqr.py (processo "RIC")
 1. KPMs do contrato §4 → sqr-kpms.txt/sqr_kpm_history.csv
 2. dreno telemetria prb_* (alinhamento §3.5)
 3. SqrE2Adapter::SendKpmIndication ── UDP JSON (seq, ue_records, total_rbg) ──▶ seq dedup (retry = eco da
    (SqrE2LoopbackTerm modo XAPP_UDP)        decisão cacheada; nunca re-decide)
                                             4. materializa KPM formato §4 → SqrKpmParser (imputação R09)
                                                → obs (SqrObservationBuilder) → C efetivo (telemetria, R01)
                                             5. learn (z_{t−1}, a_final_{t−1}, y_t)  ← ordem idêntica ao IPC
                                             6. act (SqrSliceController, timer) → Π (R24)
                                             7. % inteiros que reproduzem EXATAMENTE as cotas RBG alvo
                                                (réplica de ComputeRbgQuotas + busca; ver §3.1)
                                             8. RIC Control JSON (RAN_SLICING, SlicePRBQuota int) ◀──
 9. OnRicControlMessage → clamp pct → ApplyQuotaAction(pct) → ComputeRbgQuotas → SetSliceQuotaRbg
    (idêntico ao caminho IPC; resposta em ~1 ms ≪ budget 50 ms)
```

- **Transporte local documentado (DEGRADED)**: UDP/localhost com JSON compacto (`separators=(",",":")`), `seq` monotônico por indicação, retry do lado gNB com o MESMO seq (dedup no xApp). Budget por tentativa = 50 ms (contrato §5, o mesmo do `sem_timedwait` do IPC). O xApp faz **bind ANTES** de spawnar o sqr-sim (binda em `--xappPort`, default 5560) e mata o processo ns-3 em qualquer exceção (nunca deixa órfão).
- **FULL**: troca do transporte (`UdpLoopbackTransport` → e2sim/SCTP) sem mudar a lógica — o stub `SctpE2simTransport` documenta o ponto de integração (Fase 5.1). **Não verificado nesta fase, por definição**: o e2sim/RIC não existe na máquina (mesma ressalva honesta da Fase 5.3 §4).
- **Alinhamento de janela do log**: o `step` do IPC bloqueia até a PRÓXIMA borda, então a linha k do `step_metrics.csv` carrega as derivadas de KPM/y_t/reward da janela k+1. O xApp adia o flush da linha k para a indicação k+1 e preenche as MESMAS colunas com a MESMA janela (e o `prb_scheduled` do dreno da mesma borda) — as colunas dos dois logs ficam comparáveis linha a linha.

## 3. Divergências e decisões registradas

1. **[ASSUMIDO Fase 5.4] Quantização inteira do RIC Control contornada com verificação exata**: `SlicePRBQuota.dedicatePRBRatio` é ASN.1 INTEGER (0–100) — não expressa `100·a_i/C` fracionário (restrição já registrada na 5.3). O xApp calcula a cota RBG alvo com uma **réplica exata** de `SqrMacScheduler::ComputeRbgQuotas` (piso + maiores restos, `llround` do somatório, R27), e busca o vetor de **% inteiras que a reproduz exatamente** (candidato construtivo `ceil(100·Q_i/totalRbg)`, depois busca local determinística). **Differential test contra oráculo C++ extraído verbatim**: 27.720 vetores de ação (todo o espaço Σa≤C, C=53) + 17.238 vetores inteiros de pct amostrados → **0 divergências**; a busca reproduz a cota alvo exatamente em **27.720/27.720** vetores. Resíduo documentado: `prb_requested_pct` difere do IPC por até **0,96 p.p.** (inteiros), SEM efeito nas cotas RBG aplicadas (`prb_executed_rbg` idênticas janela a janela — paridade real).
2. **[BUG CORRIGIDO — réplica × C++]**: a 1ª rodagem da paridade DIVERGIU na janela t=65900 (cota da slice 1: 6 vs 7 RBG). Causa raiz: `sum()` do Python é **compensado (Neumaier) desde o 3.12** e difere do fold-ingênuo do C++ no último ulp — `raw = (2,12,36)@53`: fold 26.5 → `llround` 27 vs `sum()` 26.499999999999996 → 26. A réplica foi corrigida para o fold-ingênuo (com comentário no código e teste de regressão dedicado, `test_replica_naive_fold_not_compensated_sum`). A divergência era da **ferramenta de paridade** (xApp), não do transporte E2.
3. **[CORRIGIDO] entrega da resposta da ponte**: o xApp respondia ao próprio endereço de bind em vez do par capturado no `recvfrom` (o gNB usa porta efêmera) — corrigido; e o parser JSON do C++ tolera espaço após `:` (json.dumps emite `"seq": 1`). Após as correções: 299/299 indicações respondidas e aplicadas em 30 s, 0 timeouts.
4. **[SEMÂNTICA] janelas além de `max_steps`**: o loop IPC termina em `max_steps = (simTime−0,5)·10` (1195 para 120 s) e o C++ mantém a última cota por timeout nas bordas restantes. O xApp faz o equivalente exato: para de decidir após `max_steps` e **echo congelado** da última decisão nas indicações seguintes (`frozen_echo=1` no log; sem re-decidir/re-aprender). Resultado: `controls_applied=1199` (todas as janelas), `xapp_timeouts=0`.
5. **[LOG] `kpm_timestamp=0` nas primeiras linhas do IPC** (e agora também do E2): janelas sem nenhum UE (chegada Poisson ainda não ocorreu) → parser sem linhas → watermark 0. Valor legítimo da semântica, não bug — idêntico nos dois caminhos (paridade da coluna: 0 divergências em 3585 linhas).

## 4. Verificação (honesto)

| Check | Comando | Resultado |
|---|---|---|
| Build com a ponte C++ | `bash MODA/ns-3-dev/scratch/sqr/ns3_gate.sh build` | **exit 0** (sqr-e2-adapter.cc + sqr-sim.cc recompilados) |
| Differential test da réplica | oráculo C++ extraído verbatim de `ComputeRbgQuotas` + `xapp_sqr.compute_rbg_quotas_py` | **0/27.720** (fluxo real 100·a/C) e **0/17.238** (grade inteira) divergências; busca exata **27.720/27.720** |
| Smoke 30 s do xApp E2 | `.venv/bin/python examples/xapp_sqr.py --simTime 30 ...` | **299 indications / 299 applied / 0 timeouts**, exit ns-3 = 0, 0 divergências de quantização, handle p95 ≈ 1,1 ms |
| **Paridade 120 s seed 1 γ=1** | `bash examples/run_sqr_e2_parity.sh 120 1 1` | **"PARITY OK"** — decisões: **3585/3585 linhas idênticas** (timestamp_ms, mode, raw_action, action, prb_requested, prb_executed — 0 divergências); telemetria ns-3 `sqr_prb_windows.csv`: **1199/1199 janelas idênticas** (prb_executed/scheduled/budget/idle, dl_slots); KPM `sqr_kpm_history.csv`: **1213 linhas idênticas**; violações por-UE-janela (R19): **994 = 994**; passos 1195 = 1195 = esperado; `IPC_DEGRADATION` 0/0 e `xapp_timeouts=0` |
| Regressão Fase 5.3 (loopback dummy) | `bash MODA/ns-3-dev/scratch/sqr/run_e2_smoke.sh 30 1` | **"E2 SMOKE OK"** (dispatcher não alterou o modo loopback) |
| Suíte Python | `.venv/bin/python -m pytest tests/ -q` | **335 passed, 1 skipped** (incl. 9 testes novos de `test_sqr_xapp_quant.py`) |
| FULL (e2sim/SCTP) | — | **NÃO verificado, por definição** (e2sim/RIC ausentes na máquina, MODA-PORT §4); pontos de integração documentados no stub `SctpE2simTransport` e no cabeçalho do adaptador |
| Soak/campanha ≥5 seeds (PLANO 5.4) | — | **Não executado neste pedido** (o pedido foi seed única de paridade: 120 s, seed 1, γ=1); o script de paridade é reutilizável para mais seeds |

**Diretório do run de paridade**: `MODA/ns-o-ran-gym/results/sqr_e2_parity/run_20261014_190646_sim120s_seed1_g1/` (`parity_report.md`, `ipc/`, `e2/`, logs dos dois drivers).

## 5. Latência de decisão — diferença observada (orçamento: 50 ms)

| Caminho | métrica | n | p50 (ms) | p95 (ms) | máx (ms) |
|---|---|---|---|---|---|
| IPC (semáforos+CSV; só `act()`) | `decision_time_ms` | 1195 | 1,16 | 8,18 | 10,77 |
| E2 xApp — núcleo `act()` | `decision_ms` | 1195 | 1,24 | 8,07 | 10,71 |
| E2 xApp — indication→controle (fim a fim) | `handle_ms` | 1195 | 2,39 | 9,28 | 12,05 |
| E2 ponte lado gNB (UDP JSON, ida e volta) | `rtt_wall_ms` | 1199 | 2,46 | 9,36 | 12,28 |

- **Diferença de latência observada (E2 − IPC): +1,23 ms no p50; +1,11 ms no p95** — o custo do caminho E2 sobre o IPC é a serialização JSON + transporte local UDP (~0,1 ms) + materialização/parse da indication; o núcleo de decisão `act()` é estatisticamente igual nos dois caminhos (1,24 vs 1,16 ms p50).
- **Ambos os caminhos ficam DENTRO do orçamento de 50 ms do contrato §5** com folga ≫ 5× (p95 E2 fim a fim 9,28 ms = 18,6% do orçamento; 0 timeouts de ponte em 1199 janelas).
- Observação: o p95 (~8–9 ms) é dominado por picos do `act()` (coletor/GC do NumPy nos dois caminhos), não pelo transporte.

## 6. Como rodar

```bash
# Paridade completa (IPC x E2) — MESMO seed/simTime/γ; imprime tabela + PARITY OK:
bash MODA/ns-o-ran-gym/examples/run_sqr_e2_parity.sh [simTime=120] [seed=1] [gamma=1]
#   saída: results/sqr_e2_parity/run_<ts>/  (parity_report.md, ipc/, e2/)

# xApp E2 isolado (spawn do sqr-sim --controlPath=e2 --e2Endpoint=xapp incluído):
cd MODA/ns-o-ran-gym && .venv/bin/python examples/xapp_sqr.py \
    --gamma 1 --seed 1 --simTime 120 --output results/sqr_e2_xapp --xappPort 5560

# Cenário E2 com o dummy da Fase 5.3 (regressão do modo loopback, default):
bash MODA/ns-3-dev/scratch/sqr/run_e2_smoke.sh [simTime] [seed]
```
