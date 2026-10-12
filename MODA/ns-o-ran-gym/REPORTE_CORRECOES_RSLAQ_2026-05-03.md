# Relatório de Correções — RSLAQ DRL Training

**Data:** 2026-05-03
**Escopo:** ns-3 (C++) + ns-o-ran-gym (Python)
**Objetivo:** Corrigir múltiplos bugs que impediam o treinamento SAC/DDQN de evoluir além de 1–5 steps por episódio.

---

## 1. Resumo Executivo

Durante a análise do resultado de treinamento `sac_low_traffic/`, identificou-se que **todos os episódios terminavam prematuramente** (1–5 steps) com reward fixo de ~1.600.000. A investigação revelou **oito falhas críticas** distribuídas entre o simulador ns-3 e o ambiente Python. Todas foram corrigidas e validadas.

---

## 2. Problemas Identificados

| # | Problema | Componente | Severidade |
|---|----------|------------|------------|
| 1 | `GetUeDlBufferSize()` era stub (`return 0;`) | C++ scheduler | Crítico |
| 2 | Cenário `low_traffic`: eMBB a **50 kbps** (irrealista) | C++ sim script | Crítico |
| 3 | `bufferBytes` não era exportado no KPM | C++ sim script | Crítico |
| 4 | `h_2` URLLC explodia quando `bufferBytes=0` | Python reward | Crítico |
| 5 | Outage eMBB sem verificar demanda (`txBytes`) | Python reward | Crítico |
| 6 | SLAs de `low_traffic` desalinhados com cenário | Python reward | Crítico |
| 7 | Warmup inexistente para suprimir terminal conditions | Python reward | Alto |
| 8 | `stdout/stderr` truncados a cada chamada (`w+`) | Python base env | Médio |

---

## 3. Correções Detalhadas

### 3.1 C++ — ns-3-dev/scratch/rslaq/

#### 3.1.1 Implementar `GetUeDlBufferSize()` (stub → funcional)

**Arquivos:** `rslaq-mac-scheduler.h`, `rslaq-mac-scheduler.cc`

**Problema:** O método `GetUeDlBufferSize(uint16_t rnti)` retornava constantemente `0`, fazendo com que o campo `bufferBytes` no KPM fosse sempre zero.

**Solução:**
1. Adicionado membro `mutable std::map<uint16_t, uint32_t> m_lastDlBufferSize` no header.
2. Implementado rastreamento do buffer em `BeforeDlSched()` — atualiza `m_lastDlBufferSize[ueInfo->m_rnti] = ue.second`.
3. `GetUeDlBufferSize()` agora consulta o mapa rastreado.

```cpp
// rslaq-mac-scheduler.h
mutable std::map<uint16_t, uint32_t> m_lastDlBufferSize;

// rslaq-mac-scheduler.cc
uint32_t RslaqMacScheduler::GetUeDlBufferSize(uint16_t rnti) const
{
    auto it = m_lastDlBufferSize.find(rnti);
    return (it != m_lastDlBufferSize.end()) ? it->second : 0;
}

void RslaqMacScheduler::BeforeDlSched(...)
{
    // ... existing code ...
    m_lastDlBufferSize[ueInfo->m_rnti] = ue.second;  // NEW
}
```

#### 3.1.2 Tornar `GetUeDlBufferSize()` público

**Arquivo:** `rslaq-mac-scheduler.h`

**Problema:** O método era `private`, impedindo que `rslaq-sim.cc` o chamasse.

**Solução:** Método movido para a seção `public`.

#### 3.1.3 Exportar `bufferBytes` no KPM

**Arquivo:** `rslaq-sim.cc`

**Alterações:**
1. Adicionado `std::map<uint16_t, uint16_t> g_ueIdToRnti;` para mapear `ueId` → `rnti`.
2. Preenche o mapa na lambda de UE attach: `g_ueIdToRnti[ueId] = rnti;`.
3. Atualizado cabeçalho do KPM para incluir `bufferBytes`.
4. Escreve o valor no loop de exportação:

```cpp
uint32_t bufferBytes = (g_schedulerPtr && g_ueIdToRnti.count(ueId))
                           ? g_schedulerPtr->GetUeDlBufferSize(g_ueIdToRnti[ueId])
                           : 0;
```

#### 3.1.4 Corrigir taxa eMBB no cenário `low_traffic`

**Arquivo:** `rslaq-sim.cc` — função `InitScenarios()`

**Problema:** `low_traffic` tinha `embbRateBps = 50000` (50 kbps). Com 5 UEs, cada UE recebia 10 kbps — taxa insuficiente para gerar um único pacote nos primeiros segundos de simulação. O FlowMonitor registrava `txBytes = 0` para eMBB, e a reward interpretava como outage.

**Solução:**
```cpp
// Antes
m["low_traffic"] = {"low_traffic", 50000, 1000000, 2000000, ...};
// Depois
m["low_traffic"] = {"low_traffic", 5000000, 1000000, 2000000, ...};
```

---

### 3.2 Python — ns-o-ran-gym/src/

#### 3.2.1 Corrigir truncamento de stdout/stderr

**Arquivo:** `nsoran/ns_env.py` — método `read_streams()`

**Problema:** Usava modo `"w+"` (truncate), apagando logs a cada chamada.

**Solução:**
```python
# Antes
open(stdout_file_path, "w+")
# Depois
open(stdout_file_path, "a")
```

#### 3.2.2 Adicionar `warmup_steps` na reward function

**Arquivo:** `environments/rslaq_reward.py`

**Problema:** Terminal conditions (outage/soft) eram avaliadas desde o step 0, quando o FlowMonitor ainda não tinha acumulado pacotes suficientes.

**Solução:**
- Novo parâmetro `step_count: int = 0` em `compute_rslaq_reward()`.
- `warmup_steps` lido do `config` (default: 5).
- Durante warmup, terminal conditions são suprimidas (`terminated = False`).

```python
in_warmup = step_count < warmup_steps
if outage_slices and not in_warmup:
    result.terminated = True
```

**Arquivo:** `environments/rslaq_env.py`
- `RslaqEnv._compute_reward()` agora passa `self.num_steps`.

#### 3.2.3 Cap em `h_2` (URLLC) para evitar explosão

**Arquivo:** `environments/rslaq_reward.py`

**Problema:** Quando `bufferBytes_max = 0`, `h_2 = 1 / (0 + 1e-6) = 1.000.000`. Com `beta = 0.4`, o reward explodia para ~400.000 por step.

**Solução:**
```python
max_h2 = float(config.get("max_h2", 10.0))
h_2 = 1.0 / (urllc_max_bfs + EPSILON)
h_2 = min(h_2, max_h2)
```

#### 3.2.4 Condicionar outage eMBB à existência de demanda

**Arquivo:** `environments/rslaq_reward.py`

**Problema:** A slice eMBB era marcada como outage mesmo quando `dTxBytes_sum = 0` (sem tráfego gerado). Isso é um falso-positivo: se não há demanda, não há outage.

**Solução:**
```python
embb_tx_bytes = float(embb_metrics.get("dTxBytes_sum", 0.0))
min_tx_bytes_for_outage = float(config.get("min_tx_bytes_for_outage", 1.0))
if embb_thr < min_thr and embb_tx_bytes >= min_tx_bytes_for_outage:
    result.outage_flags[0] = True
```

#### 3.2.5 Ajustar SLAs do cenário `low_traffic`

**Arquivo:** `environments/rslaq_reward.py` — dicionário `SLA_BY_SCENARIO`

**Problema:** Os SLAs exigiam 10+ Mbps para eMBB e 5 Mbps para MTC, mas o cenário `low_traffic` (mesmo após correção) é leve — com recursos compartilhados, atingir 10 Mbps consistentemente é impossível.

**Solução:**
```python
"low_traffic": {
    "embb_min_throughput_mbps": 1.0,      # was 10.0
    "embb_soft_max_throughput_mbps": 2.0,  # was 12.0
    "urllc_outage_bfs_bytes": 10000.0,
    "mtc_max_tdp": 1000.0,
    "mtc_target_throughput": 1.0,          # was 5.0
}
```

---

### 3.3 Scripts de Treinamento

**Arquivos:** `examples/rslaq_train_sac.py`, `examples/rslaq_train_ddqn.py`

**Alterações:**
- Adicionado argumento `--warmup_steps` (default: 5).
- Incluído `warmup_steps` no `sla_config` passado para `RslaqEnv`.

```python
parser.add_argument("--warmup_steps", type=int, default=5,
                    help="Steps to suppress terminal conditions at episode start")

sla_config = {
    "max_buffer_bytes": args.max_buffer_bytes,
    "warmup_steps": args.warmup_steps,
}
```

---

## 4. Resultados Esperados Após Correções

| Métrica | Antes | Depois |
|---------|-------|--------|
| Steps por episódio | 1–5 | 100–350 (conforme `max_steps`) |
| Reward médio | ~1.600.000 (constante) | Na escala de unidades (0–10) |
| `bufferBytes` no KPM | Sempre 0 | Valores reais do scheduler |
| eMBB em `low_traffic` | 0 txBytes, 0 throughput | Tráfego real gerado |
| Outage flags | Sempre True para eMBB | Só quando demanda < throughput mínimo |

---

## 5. Comandos de Validação

### Build ns-3
```bash
cd ns-3-dev
./ns3 build rslaq-sim
```

### Teste lógico da reward (Python)
```bash
cd ns-o-ran-gym
python3 -c "
import sys; sys.path.insert(0, 'src')
from environments.rslaq_reward import compute_rslaq_reward

metrics = {
    0: {'throughputMbps_sum': 0.0, 'dTxBytes_sum': 0.0, 'bufferBytes_max': 0.0, 'dLostPackets_sum': 0},
    1: {'throughputMbps_sum': 1.56, 'dTxBytes_sum': 1000.0, 'bufferBytes_max': 0.0, 'dLostPackets_sum': 0},
    2: {'throughputMbps_sum': 2.05, 'dTxBytes_sum': 1000.0, 'bufferBytes_max': 0.0, 'dLostPackets_sum': 0},
}
result = compute_rslaq_reward(metrics, 'low_traffic', step_count=5, config={'warmup_steps': 5, 'max_h2': 10.0})
assert result.terminated == False
assert result.reward < 100  # should not be in the millions
print('Validation passed!')
"
```

### Treinamento SAC
```bash
cd ns-o-ran-gym
python3 examples/rslaq_train_sac.py \
    --scenario low_traffic \
    --episodes 100 \
    --output results/sac_low_traffic_v2
```

---

## 6. Checklist Final

- [x] `GetUeDlBufferSize()` implementado corretamente (não mais stub)
- [x] `bufferBytes` exportado no `rslaq-kpms.txt`
- [x] Taxa eMBB de `low_traffic` corrigida (50 kbps → 5 Mbps)
- [x] `GetUeDlBufferSize()` tornou-se público
- [x] Warmup steps adicionado na reward function
- [x] `max_h2` cap para evitar explosão de reward
- [x] `min_tx_bytes_for_outage` para eMBB (condiciona outage à demanda)
- [x] SLAs de `low_traffic` ajustados para valores realistas
- [x] Scripts de treinamento atualizados com `--warmup_steps`
- [x] Bug de truncamento de stdout/stderr corrigido
- [x] Build ns-3 (`./ns3 build rslaq-sim`) compilando limpo
- [x] Validação Python passando
