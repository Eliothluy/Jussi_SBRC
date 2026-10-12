import os
import sys

# [FIX 2026-10-11, Agente B — motivo registrado] Este arquivo usava o layout
# upstream 'nsoran.environments.*', que não existe neste fork (pacotes reais:
# 'nsoran' e 'environments' sob src/; ver pyproject [tool.hatch.build]).
# Corrigido para o caminho local + preamble sys.path padrao da suíte.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import json
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from environments.ts_env import TrafficSteeringEnv
import warnings

"""
Ignore Warnings format:

CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
    ]
]
"""
CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "This version of the mujoco environments depends on the mujoco-py bindings, which are no longer maintained and may stop working. Please upgrade to the v4 versions of the environments (which depend on the mujoco python bindings instead), unless you are trying to precisely replicate previous works).",
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        # NOTE: Why is this -infinity?
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "Casting input x to numpy array.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
    ]
]

# [FIX 2026-10-11, Agente B — motivo registrado] Este teste de integração
# exige o build upstream ns3-mmwave-oran (/workspace/...), que não existe no
# MODA (stack local: ns-3.46 + 5G-LENA). Sem o simulador o NsOranEnv não
# consegue nem resolver o executável (src/nsoran/ns_env.py:100-196) e o
# check_env chama reset(), que lança o processo ns-3. Guarda skipif mantém o
# teste ativo em setups que possuam o simulador e não polui o gate local.
# O caminho do JSON de configuração passou a ser resolvido a partir do repo
# (antes: relativo ao CWD, só funcionava com cwd=src/environments).
NS3_MMWAVE_PATH = "/workspace/ns3-mmwave-oran"
SCENARIO_CONFIGURATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "environments",
    "scenario_configurations", "ts_use_case.json",
)


@pytest.mark.skipif(
    not os.path.exists(NS3_MMWAVE_PATH),
    reason=f"requires the upstream ns3-mmwave-oran build at {NS3_MMWAVE_PATH} (absent in MODA)",
)
def test_all_env_api():

    configuration_path = SCENARIO_CONFIGURATION_PATH
    with open(configuration_path) as params_file:
        params = params_file.read()
    scenario_configuration = json.loads(params)
    assert scenario_configuration

    output_folder = '/workspace/ns-o-ran-gymnasium/output'

    env = TrafficSteeringEnv(ns3_path=NS3_MMWAVE_PATH, scenario_configuration=scenario_configuration, output_folder=output_folder, optimized=False)
    assert env

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env, skip_render_check=True)
        env.close()

    for warning in caught_warnings:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")
