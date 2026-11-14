# Dossiê — degradação de ritmo e vazamento de RSS nos runs heterogêneos (Etapa 2.3, 21/10)

**Sintoma:** runs de 300 s do sweep 2.3 (equal network-only E prop IPC) não completam: o ritmo de simulação cai de ~1× tempo real para ~1% ao longo dos primeiros minutos de parede, com CPU 100% user constante e RSS crescendo continuamente. OOM killer confirmado às 08:04 (processo ns3 com 2,46 GB RSS).

## Fatos medidos (todos reproduzidos nesta sessão, 21/10, pós-reboot 06:23)

1. **Vazamento por tempo de PARELA, não de simulação**: monitor contínuo de um par equal+prop recém-lançado: RSS total 290→2265 MB em 13 min (~60–150 MB/min/proc) enquanto t de simulação avançava ~0,2 s/20 s. Um processo com t≈13 s de sim já tinha ~700 MB.
2. **Ritmo degrada com o RSS/tempo de vida**: testes foreground de ≤5 min (t de sim ≤230 s) rodaram a 0,85–2× tempo real (rápidos); os mesmos comandos sob os runners (execução longa) caem a ~1% e nunca completam 300 s (timeout 420 s atingido 2× com retry).
3. **CPU sempre ~100% user por processo** (3 procs = 297%), iowait ~0 — o processo gasta CPU em work útil-ao-ns3, não em IO bloqueante.
4. **Ocorre em TODOS os modos testados**: equal network-only (sem IPC/E2), prop IPC, λ 0,667–2,333/slice, seeds 1–3 — 100% das runs de runner afetadas (4 gerações).
5. **stdout e artefatos são pequenos** (log 45 KB crescendo 4 KB/20 s; unmapped_rntis ~0,5 MB em t=10 s; packet_delays normal): não é volume de escrita.
6. **Pré-reboot o mesmo código NÃO degradava**: ontem (20/10) lowload network-only 3×300 s completou em ~10–15 min; sanity v4 prop 3×300 s idem; Etapa 2.0 rodou 9×600 s (até 2h23 por run) sem degradação de ritmo (o RSS de ontem não foi registrado — o vazamento pode já existir sem degradar).
7. **Pós-reboot, testes curtos são rápidos** (6 testes, incl. setsid, /tmp, home, par 2-proc, greedy/heterogêneos, até t=230 s) — a degradação só se manifesta em processos de longa duração.
8. Sandbox bloqueia strace/ptrace attach (PTRACE_SEIZE EPERM) — perfil de syscall indisponível por aqui.
9. Sistema: THP=madvise; swap 4 GB com ~1,1 GB residual que não drena sem sudo; 16 GB RAM; governador powersave (não é a causa: benchmark rápido).

## Hipóteses (ordenadas; NÃO confirmadas)

- **(H1) Acúmulo de estado por evento/objeto que cresce com o nº acumulado de pacotes/UEs vistos** — candidatos: FlowMonitor (Ipv4FlowProbe com tracker por pacote: entradas de pacotes nunca entregues permanecem; ~1–2 M pacotes por run heterogêneo com URLLC 1000 pps), mapas de telemetria por RNTI (churn), timers não cancelados no teardown de UEs. O custo por evento cresceria com o tamanho dos contêineres → ritmo cai ~superlinearmente. Preveria degradação também pré-reboot (não observada — a menos que pré-reboot os runs terminassem antes do limiar: 300 s ontem completavam em ~10 min com RSS ~1,4 GB máx... compatível com limiar ~1,5–2 GB pós-reboot por causa do swap residual reduzir a memória efetiva).
- **(H2) Interação com swap residual/fragmentação pós-reboot**: o limiar de degradação desceu porque ~1,1 GB ficou em swap + fragmentação; page faults custam CPU-user. Preveria: com swap limpo/zerado, os runs completariam (teste barato: `sudo swapoff -a && swapon -a` — sem sudo no ambiente atual).
- **(H3) Thread/timer de wall-clock alocando** (menos provável: network-only sem IPC/E2 também sofre; e CPU 100% user contínuo é difícil para sleeper thread).

## Receita de reprodução (5 min)

```
cd /home/eliothluy/Documentos/Jussi_SBRC/MODA/ns-3-dev
setsid bash -c 'bash scratch/sqr/ns3_gate.sh run sqr-sim -- --simTime=300 --seed=1 --run=1 \
  --periodMs=100 --sliceLambdas=0.667,0.667,0.667 --sliceApps=greedy,urllc,mmtc \
  --cellRadius=300 --outputDir=/tmp/dossie_rep > /tmp/dossie_rep.log 2>&1' &
# monitorar: t de simulação (tail do sqr_kpm_history.csv) × RSS (ps) × tempo de parede
# degradação: t estagna < 300s com RSS > ~1 GB e CPU 100%
```

## Mandato para o fix

1. Root cause (H1 priorizada: instrumentar/serializar FlowMonitor? desabilitar probes por pacote? medir mapas internos; NS_LOG seletivo; heaptrack/valgrind se disponíveis; build ASan em build-dir separado se viável SEM quebrar o build do sweep).
2. Fix com gate: a receita acima deve completar t=299900 ms com ritmo monotônico ~1× e RSS final reportado; suíte `tests/run_sqr_tests.sh` passando; nenhum arquivo congelado tocado.
3. Se H1 confirmar em FlowMonitor: avaliar desligar `Ipv4FlowProbe` de pacotes individuais OU serializar/limpar periodicamente SEM perder as 13+4 colunas do KPM (dTx/dRx/dDelaySumSec vêm daí!) — pode exigir agregação própria alternativa; registrar decisão no contrato/register.

---

## DESFECHO (21/10, noite — workflow `dwfrun-9d75d014`, Agente A; gates PASS)

**H1 (vazamento no simulador) REFUTADA com evidência**: FlowMonitor faz pruning correto de pacotes rastreados (`src/flow-monitor/model/flow-monitor.cc:314-336`, CheckForLostPackets a cada 1 s de sim; receita do dossiê com 395.817 pacotes entregues → RSS final ~106 MB); lote de 5 network-only simultâneos: 5/5 completos, RSS ≤120 MB cada, PSI-CPU 0; run solo prop: ns3 68→108 MB flat. Gate independente: 300 s completos (t=299.900 ms) em ~470 s de parela.

**Causa real (tooling, 2 camadas)**: (1) runners antigos com 10 processos simultâneos → contenda/OOM global (08:04); (2) o timeout de 420 s adicionado na 4ª geração era MENOR que a duração legítima de um run saudável (~470–650 s medidos) — runs bons eram mortos (rc=124 com artefatos completos "checks 3/3" nos logs). Fix: TIMEOUT_S=1800 + flock por run; locks separados por braço (orquestrador) para 2 sims paralelos com base na evidência do lote de 5. Sweep relançado às 17:56.

**Nota honesta (aberta)**: as medições das gerações 1–4 feitas pelo orquestrador (RSS ~700 MB em processo com t=13 s de sim na 3ª geração com apenas 2 processos; crescimento linear ~60 MB/min de parela; ritmo caindo a 1% mesmo com memória disponível) não são integralmente explicadas pela tese "tooling/timeout" — o lote saudável do Agente A (mesmo binário, mesmas condições) não reproduz o crescimento medido então. Possíveis explicações residuais (não confirmadas): contaminação de estado do sistema pelo swap residual pós-OOM durante a manhã (normalizada ao longo do dia); efeito da contenda de 10 procs no comportamento do allocator/THP persistindo nos processos seguintes. Se a 5ª geração (relançada) apresentar qualquer degradação de ritmo ou RSS >300 MB/run, reabrir a investigação com instrumentação de allocator — critério de vigília registrado.
