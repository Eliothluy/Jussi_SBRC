# Nota de divergência/diagnóstico — "vazamento de RSS" e degradação de ritmo nos runs heterogêneos (Etapa 2.3)

**Data:** 21/10/2026 (tarde, pós-investigação do dossiê `dossie_degradacao_2_3.md`)
**Autor:** Agente A (ns-3/5G-LENA), sob mandato do orquestrador
**Status:** causa-raiz encontrada com evidência; H1 (FlowMonitor) **refutada**; fix aplicado nos runners.

---

## 1. Veredito

**Não há vazamento de memória no binário do cenário (`scratch/sqr/sqr-sim.cc`, binário de 20/10 18:11).**
O fenômeno relatado no dossiê (RSS crescendo 60–150 MB/min de parela, ritmo caindo a ~1%, OOM 2,46 GB)
foi **efeito da concorrência excessiva entre braços do sweep + timeout do runner menor que a duração real
do run**, não de acúmulo de estado no simulador:

1. **Versões antigas dos runners** (pré-09:58, arquivo atual reescrito às 09:58) lançaram **5 runs do braço
   network-only + 5 do braço prop IPC no mesmo segundo** (evidência: `hetero_sweep_runner.log` 06:29:33 ×5;
   `hetero_sweep_prop_runner.log` 06:29:35 ×5) ≈ **10 sims + 5 pythons simultâneos**.
2. Sob essa concorrência, a tarde de hoje reproduziu o colapso nos logs já existentes: runs de 07:08 avançaram
   **t=23 s de sim em ~40 min de parela (~1%)** (`hetero_sweep/A_lam2_s2.log`, última linha PRB_WINDOW t=23100ms,
   mtime 07:48) e o **OOM global de 08:04:17** matou um `ns3.46-sqr-sim-` com **anon-rss 2.458.596 kB**
   (journalctl -k: `oom_kill_process`, cgroup `app-zcode-12566.scope`, `oom_score_adj 200`).
3. **O timeout do runner atual (420 s) é menor que a duração de parela de um run SOLO e SAUDÁVEL**
   (~480–650 s medidos hoje; ontem "1×" dava 5–6 min = ~330–360 s, raspando). Por isso as tentativas de
   09:59/10:06 falharam com `rc=124` (KPM completo, delays ainda não) e `rc=137` com **`checks: 3/3` —
   artefatos completos, processo morto pelo timeout** (`hetero_sweep_prop_runner.log` 07:05–07:06 e 10:06–10:13).
   O retrabalho em loop ("4 gerações de runners afetadas") era este aliasing entre timeout e duração legítima.

## 2. Evidência da não-existência de vazamento (todos os experimentos desta sessão, 17:00–18:00)

| Experimento | Receita | Resultado |
|---|---|---|
| Run solo network-only | exatamente a receita do dossiê (300 s, seed 1, λ=0,667×3, greedy,urllc,mmtc, cellRadius 300) via `ns3_gate.sh run` | **completa t=299900 ms**; RSS 82→**106 MB** (amostras 20 s, `/tmp/monitor_rep.csv`); ritmo 0,4–0,7×; ~511 s de parela; PACKET_DELAYS 395.817 |
| Run solo prop IPC | mesmo grade via `sqr_run.py --controller prop` | **completa, `ns3_exit_code: 0`**; python RSS 177→**178 MB flat**, ns3 68→**108 MB** (`/tmp/monitor_prop.csv`); ~480 s de parela |
| Lote de 5 network-only simultâneos | seeds 1–5, mesmos parâmetros | **5/5 completam t=299900 ms**; RSS por proc ≤ **120 MB**; PSI-CPU avg10 = 0,00; swap estável (1122 MB) (`/tmp/monitor_batch.csv`) |
| Código | `FlowMonitor::m_trackedPackets` (map por pacote em voo) | prunable: `CheckForLostPackets()` a cada 1 s de sim remove entradas >10 s (`flow-monitor.cc:314-336`, `PERIODIC_CHECK_INTERVAL` 1 s em :19); probes por nó (`sqr-sim.cc:967,2504`) não têm estado por pacote (`flow-probe.h:72` — stats agregadas por flow) |

Com 395.817 pacotes entregues por run e RSS final de 106 MB, um acumulador por pacote (H1) está excluído
no binário atual. Os contêineres do cenário/scheduler são drenados por janela (`g_packetDelayBuffer` →
`FlushPacketDelays`, sqr-sim.cc:486-507) ou locais por chamada (`sqr-mac-scheduler.cc:373-510`).

## 3. O que permanece não reproduzido (honestidade de registro)

- Os **~700 MB com t≈13 s de sim** citados no dossiê (medição da manhã sob 10+ processos) não foram
  reproduzidos com 1, 2 ou 5 processos — RSS máximo observado hoje: 120 MB/proc. Sob pressão de RAM com
  zram, page-faults e descompressão contam como CPU user (sem iowait), o que é consistente com
  "CPU ~100% user, iowait ~0, ritmo 1%" relatado; o valor exato do pico de cada processo naquela condição
  não é reconstruível retroativamente.
- O **governador powersave** e a carga de desktop (navegador aberto 16:56) explicam o ritmo solo de hoje
  (0,4–0,7×) ser menor que o de ontem à noite (~1×) — não há qualquer componente de software do cenário
  nessa diferença (o binário é o mesmo de ontem 18:11; nenhum fonte de `scratch/sqr` mudou depois — `stat`).

## 4. Fix aplicado (runners, não simulador)

`run_hetero_sweep.sh` e `run_hetero_sweep_prop.sh` (ambos `bash -n` OK):

1. **`TIMEOUT_S=1800`** (era 420): folga ~3× sobre o pior caso observado (~650 s). O antigo comentário que
   justificava 420 s pelo "vazamento de RSS" foi corrigido — o pressuposto era falso.
2. **`flock -x` em `/dev/shm/sqr_sweep_round2.lock` por run, compartilhado entre os dois braços**: os braços
   nunca mais rodam simultâneos mesmo que os dois runners sejam lançados juntos (causa-raiz da contenda de
   06:29→08:04).
3. Retry (3 tentativas) mantido apenas para falha genuína; a limpeza de semáforos IPC órfãos do braço prop
   permanece.

## 5. Recomendações operacionais para a campanha (Rodada 2)

- Lançar **um braço por vez** (o lock agora garante a serialização, mas a fila do 2º braço espera o 1º inteiro —
  preferir sequência explícita).
- Monitorar `grep -c PRB_WINDOW` + última linha do `sqr_kpm_history.csv` como sinal de PROGRESSO (substitui
  o critério de RSS do dossiê); ritmo saudável ≥ 0,3× tempo real com ≤ 2 procs de sim simultâneos.
- O run network-only solo de 300 s com 19 chegadas (seed 1) é o pior caso de duração observado (~8,5 min):
  qualquer timeout da campanha deve ser ≥ 3× isso.

## 5.1 Gates pós-fix (executados nesta sessão)

- **GATE 1 (receita do dossiê, network-only, 300 s, seed 1, λ=0,667×3, greedy,urllc,mmtc, cellRadius 300):**
  **completa t=299900 ms**, **pico de RSS 103 MB** (< 800 MB), PACKET_DELAYS 395.817 (idêntico ao run solo
  anterior — determinístico), parela total ≈ 470 s (o timeout antigo de 420 s teria morto este run saudável;
  o corrigido de 1800 s passa com folga). Evidência: `evidencia_degradacao_2026-10-21/monitor_gate1.csv`
  (amostras 20 s: RSS 65→103 MB, ritmo 0,27–1,34×, sem colapso).
- **GATE 2 (suíte):** `bash tests/run_sqr_tests.sh` → **207 passed** (1 warning legado de slice-id já
  conhecido), 6,25 s.

## 6. Observação acessória (não investigada — fora do escopo deste mandato)

Durante o lote de 5, uma leitura de RSS do monitor retornou 0 para um pid intermediário do wrapper
(pids do `ns3_gate.sh`/`./ns3` mudam ao longo do boot do run); TODOS os 5 runs completaram t=299900 ms.
Conclusão: monitorar o pid final do binário (`pgrep -f ns3.46-sqr-sim-default`) ou o CSV de KPM, não pids
intermediários da cadeia de wrappers.
