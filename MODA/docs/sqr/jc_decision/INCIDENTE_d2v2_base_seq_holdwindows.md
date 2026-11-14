# INCIDENTE_d2v2_base_seq_holdwindows.md — replay_base seq não cobria janelas além da última decisão

**Data:** 13/11/2026 · **Descoberta:** primeiros replays D2-v2 (GO autorizado) com `rc=1` em estados λ5/λ2 tardios · **Nenhum dado de campanha anterior é invalidado** (ver §4).

## 1. Sintoma

Replays com `--replay_at_win` além de `len(decisions.jsonl)` da base falhavam de duas formas:
1. `IndexError` em `replay_controller._base_split` (janelas finais do simTime estendido, **após** a janela alvo t);
2. `ValueError: seq_actions covers N windows; replay_at_win=t is beyond it` (validação de init) para 6 estados (3×λ2s2102, 3×λ2s2101) — runs com muitos holds (848–1.703 decisões em 3.000 janelas-relógio).

## 2. Causa

A base seq era construída com `n = len(intended)` (nº de linhas do `decisions.jsonl`). Nos runs v8/v9, janelas de célula vazia (HOLD) **não geram decisão logada** (correção do bug 3), então `len(intended) < nº de janelas-relógio` — a telemetria prb, porém, registra o **fluxo EXECUTADO** (quota por janela) para todas as janelas com slot DL (~2.972/2.999). Janelas além de `len(intended)` ficavam fora da seq mesmo com fluxo executado registrado. Na v8 o problema não foi atingido (22 replays parciais só λ2 com win ≤1.541 e bases v3 com 3.000 decisões/run).

## 3. Correção (2 pontos, sem tocar adjudicação/métricas)

1. `sqr_run.py`: `n = max janela-relógio da telemetria prb`; seq por janela-relógio = quota executada (`prb_budget_rbg_i/dl_slots`); furos (janelas sem slot DL, sem efeito físico no KPM) preenchidos com backward-fill da última quota executada (semântica de hold do ns-3) e, antes da 1ª quota, com as decisões pretendidas em ordem (comportamento pré-v9, bit-exact validado na D0-B).
2. `replay_controller.py`: clamp defensivo `_k = min(windows_seen-1, len(seq)-1)` para janelas além do fim da base (não avaliadas).

**Bit-exactness preservada:** janelas com slot DL têm quota idêntica nas construções antiga/nova (mesma fonte); furos (dl_slots=0) não geram agendamento ⇒ KPM inalterado. A adjudicação (prefixo bit-idêntico entre ramos + A/A) valida formalmente por estado — qualquer drift real é marcado INVALID/aa_ok=False no relatório, nunca silenciado.

## 4. Impacto sobre os dados

- **D1-v8/v9 e todos os indicadores anteriores: intocados** (o incidente é só do mecanismo de REPLAY D2).
- 26 replays concluídos antes da correção: mantidos (prefixos não contêm furo com efeito físico; re-verificação formal pela adjudicação).
- ~78 replays falhados pré-correção: dirs limpos e re-executados com a correção.
- 0 falhas pós-correção até 20:33 (verificação contínua no D2_report).

## 5b. ADENDO 13/11 — A/A do estado S1_lam5_s2101_t2064: causa isolada

O A/A formal falhou apenas neste estado (29/30 OK). Re-execuções idênticas reproduzem o desvio byte-a-byte (não é corrida de parede) e **com simTime=300 o desvio DESAPARECE (A/A bit-exact, 0 drift)**: a causa é a **borda de fim de simulação** — o simTime truncado (208 s vs 300 s do base) altera dinâmica interna determinística a partir da janela 1809 (ΔRX eMBB +0,25%, delays URLLC mediana −17 ms, TX idênticos). O desvio é comum aos 13 ramos do estado (prefixo entre ramos bit-exact ⇒ comparação interna contrafactual válida); rótulo do estado inalterado (NOT_RECOVERED). Recomendação registrada: estados com win > ~2.400 em futuras campanhas podem exigir simTime integral — nada foi re-executado na D2-v2 (item: sem repetir simulações desnecessariamente).

A construção corrigida permanece a regra "fluxo EXECUTADO" (PROTOCOL_FROZEN §4.2 / Emenda 2) — estende sua cobertura às janelas-relógio além da última decisão logada. Nenhuma métrica, SLA ou critério alterado. D3/PILOT/MAIN bloqueados.
