# claims_interpretation_review.md — D0-A §5.7

| claim | o que o dado mostra | o que NÃO mostra | frase permitida no manuscrito |
|---|---|---|---|
| INFEASIBLE | sem ação recuperadora confirmada no espaço discreto/suporte testado | inviabilidade universal/matemática | "sem recuperação confirmada no espaço testado" |
| 5/5 (48,5,0) | 5/5 eMBB e 4/5 URLLC in-window; mMTC rolling 1,0 portado por histórico (tx in-window=0) | recuperação causAL do mMTC sob a_m=0 | "recupera eMBB/URLLC na janela; mMTC não testável in-window nestes estados (e o único tx in-window sob a_m=0 foi perdido)" |
| 0/5 (37,13,3) | ERRADO: 3/5 frozen (2/5 strict) — bug de métrica do stage1 | — | "a ação in-suporte (37,13,3) recupera 3/5 no julgamento congelado" |
| FFR 11,08% G4-B | piso empírico sob gates pré-registrados | "limite irredutível de Bayes" | "FFR permaneceu 1,1pp acima do gate pré-registrado" |
| MARGINAL | banda operacional honesta na fronteira | garantia conformal/Chow | "classe de incerteza declarada" |
| RLF≈0/SNR | rádio não é causa principal nos runs auditados | rádio perfeito / causa única | "violações emergem com rádio saudável" |
| evitabilidade×espaço | (48,5,0) recupera 5/5 frozen onde (18,17,18) viola; (37,13,3) recupera 3/5 | "só fora dos pisos há recuperação" (caiu) | "a recuperação existe in-suporte (3/5) e amplia-se fora dos pisos (5/5); pisos estáticos restringem oportunidades" |
| TDD/scheduler | insensibilidade a a_u no interior; colapso em 0 | mecanismo fino identificado | "insensível à alocação no interior; mecanismo sub-slot não instrumentado" |
