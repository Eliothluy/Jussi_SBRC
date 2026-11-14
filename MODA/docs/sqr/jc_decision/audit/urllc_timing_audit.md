# urllc_timing_audit.md — D0-A §5.6 (A07/G0A-7)

**Coorte vs rolling, separados e verificados:**
- Curva causal a_u (fig3 / `g4_urllc_curve.csv`): recomputada dos 6 replays G4 com o
  parser congelado de JANELA DE CHEGADA (win = floor(txSec·10) == t) = coorte causal
  pós-override (branch em t·100 ms). **Bit-idêntica ao reportado**: miss 5,0–5,5%
  plano em a_u∈{13,25,35} (p95 9,15–10,03 ms; n=200/janela).
- Rolling 1 s existe SOMENTE como agregação de observação dos rótulos G4-B (emenda
  §§3-5); nunca como prazo. Deadline por pacote = 10 ms em toda a cadeia.
- Colapso a_u=0: REPORTED (G3/II-c: D_urllc_obs 8,8–9,2; contraste executável:
  t1423 sob a_m=0/a_u=5 perdeu o pacote mMTC in-window — PDR100=0). Não recomputado.
- **Atribuição de mecanismo**: a evidência suporta (i) insensibilidade do miss a
  a_u∈[13,35] com a_e compensatório e (ii) colapso em a_u=0. A causa física fina
  (decisão do PF por slot vs oportunidade TDD) NÃO é identificável sem traces
  sub-slot ⇒ **MECHANISM_NOT_IDENTIFIED**; textos que diziam "scheduler/TDD-dominada"
  sem qualificação foram revisados para "insensível a alocação no interior do
  suporte; mecanismo fino não identificado (sem traces sub-slot)".
- Horizontes: ação 100 ms · julgamento da janela por coorte de chegada · rolling 1 s
  apenas para rótulo de aprendizado — os três nunca misturados nas 15 células.
