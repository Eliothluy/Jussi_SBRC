"""SQR-JC — Safe Quantile Regression with Feasibility-Aware Joint Coordination.

Etapa 2.5 (protocolo FINAL congelado em 2026-11-03,
``docs/artigo/ETAPA_2_5_SQR_JC_PROTOCOLO_FINAL.md`` — autoridade máxima).

Este pacote é código NOVO isolado (AGENTS.md §5 regra 2; protocolo §74): NÃO
modifica o core ``rslaq_*`` nem o SQR existente em ``nsoran.sqr`` — o learner
SQR auditado (kernel Matérn, regressor quantílico online pinball, memória
delimitada, incerteza, tau dinâmico) é REUTILIZADO por importação/Composição.

Módulos (§74):
- ``sla_margin``           margem de SLA normalizada comum (§7) + SLAs congelados R36–R38
- ``kernel_models``        cabeças quantílicas α_safe/α_q25/α_q75 sobre as MESMAS
                           support locations (§9/§10)
- ``joint_state``          estado conjunto só com informação pré-decisão (§11/§12)
- ``feature_scaler``       normalização física + p1/p99 CAL congelados (§13)
- ``joint_action_space``   semântica ALLOCATED_USAGE auditada (§22/§35)
- ``action_search``        busca exaustiva vetorizada + coarse-to-fine + avaliador
                           oracle + benchmark de runtime (§36–§38)
- ``feasibility``          classificação 3-way slice/ação/estado (§46–§49)
- ``conditional_floors``   pisos condicionados ao estado (§40–§44)
- ``joint_controller``     bootstrap PROP 0–100 s + seleção lexicográfica/minimax,
                           SEM margem post-hoc (§29/§45/§50–§52)
- ``online_update``        atualização online com margens observadas (§54)
- ``logging``              colunas exatas de decisão e logs de kernel (§56/§57)
- ``sqr_het``              baseline SQR-HET (§27) — adapter, sem nada do SQR-JC
"""

from nsoran.sqr_jc.sla_margin import (
    SLA_EMBB_MBPS,
    SLA_MMTC_PDR,
    SLA_URLLC_MS,
    SLA_URLLC_MISS_PCT_SECONDARY,
    embb_margin,
    mmtc_margin,
    urllc_margin,
)

__all__ = [
    "SLA_EMBB_MBPS",
    "SLA_MMTC_PDR",
    "SLA_URLLC_MS",
    "SLA_URLLC_MISS_PCT_SECONDARY",
    "embb_margin",
    "mmtc_margin",
    "urllc_margin",
]

#: Versão do pacote (marca os checkpoints/logs desta Etapa 2.5).
SQR_JC_VERSION = "0.1.0-preflight"
