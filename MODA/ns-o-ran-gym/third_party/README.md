# third_party/ — código de referência de terceiros

## network-slicing (baseline KBRL — Fase 3)

- **Origem:** <https://github.com/jjalcaraz-upct/network-slicing>
- **Commit clonado:** `65a3891ab16018830c912d6875550961545d637d`
  (2024-11-19, "Update README.md"; clone raso `--depth 1` em 12/10/2026 pelo
  Agente B)
- **Paper:** Alcaraz et al., "Model-Based Reinforcement Learning with Kernels
  for Resource Allocation in RAN Slices", IEEE TWC 2022,
  DOI [10.1109/TWC.2022.3195570](https://doi.org/10.1109/TWC.2022.3195570)

### Papel no projeto

**Somente referência de leitura** para o port do baseline KBRL
(`src/nsoran/sqr/kbrl/`, PLANO_IMPLEMENTACAO_SQR_NORI.md §3-Fase 3).
Nada deste diretório é importado em runtime pelo pacote `nsoran` — o port é
código novo, escrito aqui, com as divergências documentadas no docstring do
módulo portado.

### Arquivos de referência usados no port

| Arquivo | Papel |
|---|---|
| `kbrl_control.py` | `KBRL_Control` (scan de ação mínima segura + offset de segurança + atualização por monotonicidade) e `Learner` |
| `algorithms/kernel.py` | `GaussianKernel` (k = exp(−γ‖x−x'‖²), γ=1) e memória de SV com orçamento (NÃO usada pelo agente KBRL) |
| `algorithms/projectron.py` | `Projectron` (perceptron kernelizado com projeção; **o usado** por `create_kbrl_agent`) e `SVvariable` (memória sem orçamento, controlada pela projeção) |
| `scenario_creator.py` | Hiperparâmetros de criação do agente: `alfa = 0.05` (L187), `accuracy_range = [0.99, 0.999]` (default L197), `GaussianKernel(sv, 1)` + `Projectron(kernel)` (η=0.1, L217–219), fator de segurança inicial `rng.integers(2, 8)` (L191, L221) |
| `experiments_kbrl.py` | Protocolo: 30 runs/semente de rng por cenário, `accuracy_list = [[0.97,0.99],[0.99,0.999]]` |

### Dependências registradas (análise 12/10/2026)

- **Código do KBRL propriamente dito** (`kbrl_control.py`,
  `algorithms/kernel.py`, `algorithms/projectron.py`): **apenas NumPy** —
  verificado por inspeção dos imports (nenhum outro módulo).
- **Restante do repo** (`experiments_*.py`, `gym-ran_slice/setup.py`):
  `gym==0.15.3` (API antiga), `stable-baselines==2.10.1` (TensorFlow 1.9),
  `pandas`, `keras-rl` (NAF), `scipy`/`matplotlib` (plots) — versões de
  2019/2020 listadas no README.
- **Decisão:** NENHUMA dependência antiga precisa de ambiente isolado, pois o
  port usa só NumPy (já dependência do ns-o-ran-gym, `numpy>=2.1.1`) e o
  repo clonado não é instalado (`pip install -e .` NUNCA deve ser executado
  aqui — `setup.py` puxaria `gym`).
