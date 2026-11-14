# DECISÃO_URGENTE_censura_urllc.md — bug 2 é estrutural; 3 opções ao orientador

**Data:** 11/11/2026 (noite) · **Encerra:** a re-execução cega do D1 — a escolha abaixo muda o protocolo, não o código.

## O fato novo (medido, 4 runs quarentenados v5/v4)

O `HetRuntime` do B0 pergunta o outcome URLLC da janela W na decisão seguinte (início de W+1, ~100 ms após o fim de W). Mas **as entregas URLLC chegam ao arquivo de delays muito depois**:

| run | p50 | p90 | p99 | pacotes entregues >100 ms após o fim da janela | >200 ms | >300 ms |
|---|---|---|---|---|---|---|
| λ5s2102 | 61 ms | **354 ms** | 726 ms | **44,4%** | 29,8% | 15,9% |
| λ5s2103 | −32 ms | 416 ms | 2.100 ms | 14,2% | 12,9% | 11,7% |
| λ6s2103 | −30 ms | **708 ms** | 1.900 ms | 16,9% | 15,9% | 14,5% |
| λ6s2102 | −41 ms | 1 ms | 266 ms | 3,6% | 2,0% | 0,5% |

(p50 negativo = pacotes entregues ainda dentro da própria janela; a cauda é dominada por slots DL/TDD + backlog.) Consequência prática no v5: o runtime só enxergou como COMPLETE as janelas cujos pacotes chegaram a tempo (ex. 684/959) e **jamais haverá 3×COMPLETE confiável se a pergunta for feita 100 ms depois** — com folga de 1 janela extra, apenas 384/844 janelas estariam completas; com 2 janelas (~300 ms) ainda faltam ~15% no p99.

Isto **não é bug de código** (o stream lê o arquivo certo; o poll funciona): é uma restrição física do desenho de janela de decisão de 100 ms versus o atraso de entrega do próprio canal/TDD — o mesmo fenômeno da "cauda scheduler/TDD" da Rodada 2, agora medido do lado do observador.

## As 3 opções (mutuamente exclusivas; mudam o protocolo — decisão é do orientador)

**Opção A — Outcome com 2 janelas de folga (W perguntado em W+3):**
- B0 aprende/decide com estados levemente atrasados no eixo URLLC; avaliação ainda perde ~11–16% das janelas no p99 (ficam UNKNOWN — R19 separado).
- Custo: re-executar D1-v6 (~3 h) + D2-v2 (~12 h). Risco residual: censura não eliminada, apenas reduzida; janelas censuradas entram como UNKNOWN e reduzem o n de ALL3.

**Opção B — B0 runtime com estado parcial (como está) + avaliação e rótulos OFFLINE pelo arquivo completo pós-run:**
- O controlador B0 real opera com a informação disponível em tempo real (defasagem inerente — honesto e realista, como qualquer xApp); **toda a avaliação/rotulação (D1 denominadores, estratos D2, outcomes D2/D3) é computada offline do arquivo completo de delays, sem censura** (o gabarito 959/959 que já validei).
- Sem re-execução dos runs D1-v5? **Não** — o v5 foi morto no meio; mas o desenho v4/v5 dos runs é o MESMO; basta re-executar 1 vez com o runtime atual (estado parcial documentado) e fazer TODA a análise offline. Custo: 1 execução D1-v6 (~3 h) + D2-v2 (~12 h).
- Vantagem: nenhum híbrido de protocolo; o que o B0 "sabia" fica registrado (decisions.jsonl) e o que "aconteceu" vem do arquivo completo — a auditoria já é exatamente essa (window_kpis offline nos replays D2, alheia ao runtime).

**Opção C — Parar e repensar o desenho de janela (ex. período de decisão de 200–300 ms para a URLLC):**
- Muda o cenário congelado (período de decisão 100 ms é parâmetro do protocolo) — máxima consequência, reinício de cadeia; só justificável se a defasagem de 100 ms for considerada fatal para a tese.

**Recomendação do orquestrador: Opção B** — mantém o protocolo congelado intocado, separa com honestidade "o que o controlador via" (estado parcial, inerente a qualquer controlador real) de "o que aconteceu" (avaliação offline íntegra, já é o padrão de toda a auditoria D0/D2), não descarta janelas por censura e não inventa folgas ad-hoc. A Opção A ainda deixa ~15% de perda e introduz um parâmetro novo (folga=2) sem base física fechada.

## Estado atual (nenhuma execução em curso)

- Quarentenas: v1 (parser), v3/cold (contaminação), v4 (censura), v5 (censura, 5 runs parciais).
- Correções acumuladas no código: stream local (glob pós-reset) + poll extra (inócuo) — ambos mantidos em qualquer opção.
- Toda a cadeia D2/D3 aguarda essa decisão (a amostragem D2 depende dos estratos; os estratos dependem dos rótulos; os rótulos dependem da opção).
