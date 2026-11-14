# D1v8_ERRATA_e_reconciliacao.md — errata do relatório + reconciliação (pontos 1–5)

**Data:** 12/11/2026 · **Natureza:** errata técnica + reconciliação de denominadores do `D1v6_report.md` · **Fonte única:** logs canônicos D1-v8 (`d1_runs/`), sem dados de versões anteriores. Nada foi re-simulado.

## 1. Conciliação dos denominadores (ponto 1) — reconstruídos dos logs

Universo dos 9 runs canônicos (26.955 linhas):
- **Bootstrap-PROP:** 6.553 decisões (todas com outcome).
- **B0:** 15.133 decisões; **15.124 com outcome**; **9 sem outcome** = 1 por run (`end_of_run` — a última decisão governa janela além do fim do sim).
- **P_projection (B0, c/ outcome): 1.861/15.124 = 12,30%.**
- **Matriz frozen (B0, c/ outcome — 21.677 linhas incluindo bootstrap):** com proj 597 FAIL/1.264 OK; sem proj 3.251 FAIL/16.565 OK — total 21.677.

**A inconsistência do relatório anterior:** o texto dizia "15.124 decisões B0 com outcome" mas a matriz e os P por seed incluíam as **6.553 linhas de bootstrap** (21.677) — dois universos misturados no mesmo parágrafo. Além disso a matriz citada ("sem 3.161/12.824; com 561/1.400") era do **v6 contaminado por hold** (números que não batem com o v8). **Errata:** a partir desta errata, TODOS os indicadores D1 citam o universo explícito: [B0 c/ outcome = 15.124] ou [total c/ outcome = 21.677], nunca "decisões" a seco. P_projection = 12,30% (sobre B0 c/ outcome) ✓ reconciliado.

## 2. Contadores URLLC (ponto 2) — decomposição consistente e errata de fonte

**Errata de fonte:** a tabela Q2 do relatório anterior misturou fontes (RX de runs v4/v6 quarentenados + TX do v8). **Fonte única correta = D1-v8:**

| run | TX_kpm | RX_s1 | semRX | pendFim | perdidos | borda | miss%ent | p95 ms |
|---|---|---|---|---|---|---|---|---|
| λ2s2101 | 0 | 0 | 0 | 0 | 0 | 0 | — | — |
| λ2s2102 | 31.084 | 30.990 | 94 | 0 | 94 | 0 | 2,5% | 8,1 |
| λ2s2103 | 116.100 | 99.555 | 16.545 | 200 | 16.345 | 0 | 8,5% | 258,4 |
| λ5s2101 | 79.102 | 43.376 | 35.726 | 0 | 35.726 | 0 | 78,9% | 1.343,1 |
| λ5s2102 | 149.884 | 83.907 | 65.977 | 400 | 65.577 | 0 | 46,3% | 1.786,1 |
| λ5s2103 | 329.443 | 222.698 | 106.745 | 200 | 106.545 | 0 | 31,8% | 1.174,0 |
| λ6s2101 | 163.794 | 93.055 | 70.739 | 600 | 70.139 | 0 | 76,3% | 750,1 |
| λ6s2102 | 246.248 | 126.630 | 119.618 | 600 | 119.018 | 0 | 55,6% | 1.768,1 |
| λ6s2103 | 376.427 | 243.300 | 133.219 | 0 | 133.219 | 92 | 33,5% | 1.336,0 |

**Decomposição consistente:** `semRX = pendentes-no-fim (últimas 2 janelas) + perdidos (resto)` — identidade fechada em 8/9 runs; λ6s2103 tem **borda = 92 pacotes (0,024%)** com tx em [299,8; 300,0) s: o KPM fecha a janela 2998 no stamp 299900 e os 191 pacotes de cauda caem no deslocamento conhecido de ±1 janela (convenção `win = ts//100 − 1`, dataset.py) — ruído de borda documentado, não sobreposição de mecanismo. **A inconsistência "perdas > TX−RX" do relatório anterior era de fonte misturada** (RX do v6-colapso 239.438/244.985, cujas ações divergiram pelo bug 3 ⇒ entregas diferentes; TX do v8). Os números acima substituem o Q2 anterior.

**Sobreposição temporal descartada:** auditoria do ns-3 (`PACKET_DELAYS delivered_with_tag=499.764 rx_without_tag=0` no stdout λ6s2103) — nenhuma entrega sem tag; contadores originais preservados.

## 3. Atualizações quantílicas URLLC (ponto 3) — diagnóstico definitivo

**Tabela (fonte única v8, chave `policy` × margem não-null):**

| run | updates bootstrap | updates pós-bootstrap | descartes R19 URLLC | evolução margem segura URLLC pós-bootstrap |
|---|---|---|---|---|
| λ2s2101 | 0 | 0 | 1.800 | prior domina |
| λ2s2102 | 0 | 0 | 564 | prior domina |
| λ2s2103 | 0 | 0 | 843 | prior domina |
| λ5s2101 | 0 | 0 | 1.994 | prior domina |
| λ5s2102 | 0 | 0 | 1.994 | prior domina |
| λ5s2103 | 0 | 0 | 1.947 | prior domina |
| λ6s2101 | 0 | 0 | 1.994 | prior domina |
| λ6s2102 | 0 | 0 | 1.994 | prior domina |
| λ6s2103 | 0 | 0 | 1.994 | prior domina |

**Diagnóstico com mecanismo provado por instrumentação** (shim de log no `DelayStream`/`HetRuntime`, probe 25 s λ5s2102 — 231 decisões logadas):
1. Na decisão da janela W, o stream contém entregas **até W−1** (`wins_last = W−1` em 231/231 chamadas).
2. Cadeia no ns-3: o callback escreve o **KPM antes** e flusha os delays **depois** (`FlushPacketDelays` no fim do callback) ⇒ delays de W entram no arquivo só no callback de W+1.
3. O `outcome_for_window(W_g)` do pending é perguntado na iteração de W_g ⇒ `stats(W_g) = None` **estruturalmente, sempre** ⇒ R19 descarta ⇒ **0 updates URLLC é consequência legítima do observador parcial** (Opção B), não problema remanescente de instrumentação: o stream funciona (prova: polls n>0, `wins` crescentes no shim; offline 959/959).
4. **Impacto na validade B0×B1:** paridade total (mesmo checkpoint, mesmo fluxo de observação, mesmo déficit) — o que muda é só a seleção. Condição declarada: as escolhas URLLC de ambos os braços serão guiadas por prior/estado atual (n_u, fila), não por previsão calibrada; o valor do experimento D3 está na coordenação eMBB↔URLLC↔mMTC, não na previsão da cauda URLLC (que a Rodada 2 já mostrou ser insensível a a_u no interior).
5. Guard anti-repetição verificado no v8: nenhum `outcome_window_index` duplicado (λ5s2102: 0 dups); eMBB 1.582 / mMTC 1.944 updates por run — os outros dois modelos aprendem normalmente.

## 4. Métricas científicas (ponto 4) — pré-registro formal antes do D2-v2

- **Primária: STRICT_ALL3** (3×COMPLETE + all-SLA-OK), com cobertura e UNKNOWN reportados explicitamente.
- **Complementar (NÃO coprimária): eMBB∧URLLC co-avaliadas** — registada formalmente ANTES do D2-v2 por causa do achado de prevalência (3-slices simultâneas raras: 96/1.994 janelas em λ6s2103); mMTC mantém sua avaliação e aparece em todas as tabelas por slice.
- Recusa-se a coprimária dupla: decisões de gate usam STRICT_ALL3; e∧u é exploratória/complementar.

## 5. Orçamento da D2 (ponto 5) — esclarecimento

"390 replays × 3" significava **390 replays totais** (30 estados × 13 ramos) executados por **3 shards paralelos** (não 1.170). O orçamento aprovado (Emenda 1/2: 30 estados × 12 alternativas + paridade A/A por estado = 390 execuções) permanece **inalterado**. A D2-v2 está **PAUSADA** (22 replays parciais preservados em `d2v2_pre_audit_pause/`) até a aprovação desta errata; os shards serão relançados sobre a mesma amostra (seed 20261108) — a amostra vigente foi gerada sobre o universo D1-v8 e é válida.

## Recomendação fundamentada

**GO para D2-v2 sob a Opção B**, com: (i) indicadores reconciliados desta errata como únicos oficiais da D1; (ii) STRICT_ALL3 primário + e∧u complementar conforme §4; (iii) condição "URLLC online cego (prior domina), paridade B0×B1" declarada como limitação do método em todas as análises. **Sem repetição de simulações** — os 9 runs v8 são íntegros (hold 0× duplicado; checkpoints íntegros). Se o orientador considerar o déficit de aprendizado URLLC fatal para a tese, a alternativa é INCONCLUSIVE com repensar o desenho (Opção C da nota anterior) — avaliação é do orientador.

---

## ADENDO 12/11 (noite) — correção do universo "26.955" + investigação delayed feedback

**Correção do rótulo de universo (resposta ao pedido de classificação das "5.269 linhas restantes"):** o número 26.955 era o universo das versões QUARENTENADAS (v3/v4/v6/cold: 3.000 janelas-relógio × 9 runs, com holds logados como decisões — bug 3), herdado indevidamente do relatório anterior. O universo canônico v8 reconstruído dos logs: **27.000 janelas-relógio = 21.713 janelas com KPM + 5.287 janelas de célula vazia** (sem UE ativo: sem KPM, sem decisão — comportamento correto do v8); 21.713 janelas = **21.686 decisões** (6.553 bootstrap-PROP + 15.133 B0; 27 tail sem decisão). Os denominadores do §1 estavam corretos; as "5.269 linhas restantes" não existem no v8 — são a diferença entre o universo quarentenado e o canônico (5.287 vazias − 18 de arredondamento de versão).

**Borda λ6s2103 explícita (identidade exata):** `TX 376.427 = RX 243.300 − 92 + 132.605 (perdidos no meio) + 614 (em voo na cauda, janelas 2990–2996)` — fecha exatamente. Os 92 = pacotes com txSec ∈ [299,9; 300,0) **após o último callback de KPM** (janela 2999 existe só no stream de delays: RX sem denominador TX; janelas 2997/2998 têm RX completo 100/100). Não são perdas nem sobreposição de mecanismo.

**Investigação delayed feedback (ofício de 12/11):** VIÁVEL — ver [`DELAYED_FEEDBACK_inVESTIGACAO.md`](../../../../docs/sqr/jc_decision/DELAYED_FEEDBACK_investigacao.md). Síntese: 59,8% das 11.335 janelas com demanda URLLC fecham o pool ONLINE (`delivered(W)==tx_kpm(W)`, ambos disponíveis no instante; k mediano 1–3); teste isolado com flag `--urllc_delayed_update` (λ5s2102, 90 s): **237 updates URLLC** (vs 0), 0 duplicatas, 0 vazamento temporal, p95 do update == p95 offline completo em 237/237, aprendizado quantílico e estágio-1 eMBB/mMTC intactos. Proposta de Emenda 4 (update URLLC por fechamento observável do pool) aguarda decisão do orientador; D2-v2 não é bloqueada em nenhum dos dois caminhos.
