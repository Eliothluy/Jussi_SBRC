# PROTOCOL_FROZEN — EMENDA 1 (D2): redução exploratória 30 estados × 12 alternativas

**Data:** 09/11/2026 · **Autoridade:** decisão do orientador (09/11) aprovando a redução "mediante emenda documentada antes de sua execução" · **Estado:** CONGELADA antes de qualquer execução D2 (nenhum estado D2 selecionado ou executado — pré-registro legítimo per PROTOCOL_FROZEN regra de cabeçalho). Substitui os §§4.1–4.3 do PROTOCOL_FROZEN onde conflitantes; tudo o mais permanece.

## 1. Orçamento (o que muda)

| item | congelado original | emenda |
|---|---|---|
| estados | até 60 | **30** |
| alternativas por estado | 24 (+equal/prop+exaustiva em 5) | **12** (+ a_B0 de paridade + equal (18,17,18)) |
| A/A por estado | obrigatório | inalterado |

Custo estimado: 30 × (12+2) ≈ 420 replays (~t/10+2 s cada) ≈ 20–35 h de parede — executável.

## 2. Seleção de estados (regras; contagens realizadas na proposta de amostragem)

- Universo: decisões do **próprio B0** (policy=SQR_HET; fase bootstrap-PROP EXCLUÍDA) dos 9 runs D1-v3 (λ{2,5,6} × seeds 2101–2103), com outcome completo ou parcial (R19 preservado no rótulo).
- **Estratos OBRIGATÓRIOS** (preenchimento programático, sem cherrypicking; seed de amostragem congelada **20261108**; sem janelas sobrepostas >50% dentro de um estrato):
  - S1 projeção+violação (prioritário); S2 violação sem projeção; S3 projeção sem violação; S4 sem projeção sem violação (controles);
  - **diversidade de condições de SLA** (exigência do orientador): dentro de cada estrato, sub-estratificar por QUAL slice violou (e-only / u-only / múltiplas) e por demanda (n_ues ativos baixo/médio/alto por tercis);
  - **≥3 seeds e ≥2 λ representados por estrato** (S3/S4 podem ser menores se raros; reportar fração amostrada da população e razões de exclusão).
- Unidade de inferência: seed/run (relatórios agregam por seed; janelas nunca tratadas como independentes).

## 3. As 12 alternativas por estado (cobertura do espaço admissível — NÃO oracle-driven)

Composição FIXA e determinística por estado (todas Σ≤53, inteiras — A_common = simplex completo):

1. **4 redistribuições pareadas** ±Δ (Δ∈{2,4,8,16}) entre os PARES de slices ordenados pela assimetria de demanda do estado (e↔u, e↔m, u↔m, e o par de maior déficit previsto) — sempre nas duas direções sinalizadas pela assimetria;
2. **3 vizinhanças de a_B0**: a_B0 + (±1 em uma slice, ∓1 em outra) escolhidas deterministicamente pela menor distância L1 ao vértice de maior folga prevista;
3. **4 amostras aleatórias do simplex** (reprodutíveis: RNG seed = SHA1(run_id‖win) mod 2³¹; rejeição até Σ≤53; documentadas);
4. **1 referência**: equal (18,17,18).

**Papel dos modelos congelados (pré-screening):** NUNCA escolhem o conjunto (composição acima é fixa). Usam-se APENAS para (i) anotar a classe prevista de cada alternativa (diagnóstico OOD/suporte) e (ii) ORDENAR a fila de replays (confirmar primeiro as previstas recuperadoras — eficiência; sem efeito no conjunto testado nem nos rótulos).

## 4. Modelos do pré-screening — congelados e sem vazamento

- `embb_v2.npz` SHA-256 `2b40c712e7ceec6e…` e `urllc_frozen.npz` `98426dc663a3de92…` (oracle II-d-v2, `oracle_recalibrated_v2/models/`) — treinados EXCLUSIVAMENTE nos runs Rodada 2 (equal/prop 2023, seeds 1–3) e CAL (1001–1003): **disjuntos dos runs D1 (2101–2103) e de qualquer dado de avaliação D2/D3**.
- mMTC: placeholder PDR=1,0 in-suporte (R52) — anotação apenas.
- Nenhuma saída de replay/oracle entra nas features de qualquer seletor (INV-12 inalterado).

## 5. Rótulos e métricas (inalterados do §8.4–8.5)

RECOVERABLE_WITHIN_COMMON_SPACE / NOT_RECOVERED_IN_TESTED_SET / UNKNOWN; RecoverabilityFoundRate com denominadores e cobertura; nunca "impossível"/"ótimo". Gate D2 (≥2 seeds independentes com recuperação replicada em A_common) inalterado.

## 6. Vedações reafirmadas (decisão do orientador)

Não alterar o aprendizado quantílico, os SLAs ou o espaço de ações comum; não implementar novos mecanismos de proteção; D3/PILOT/MAIN bloqueados.
