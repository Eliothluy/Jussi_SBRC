import os
import sys

# [FIX 2026-10-11, Agente B — motivo registrado] Este arquivo usava o layout
# upstream 'nsoran.environments.*', que não existe neste fork (pacotes reais:
# 'nsoran' e 'environments' sob src/; ver pyproject [tool.hatch.build]).
# Corrigido para o caminho local + preamble sys.path padrao da suíte.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from environments.ts_env import TrafficSteeringEnv
import time
from pathlib import Path

if __name__ == '__main__':

    scenario_configuration = {
        "configuration": [0],
        "bufferSize": [10],
        "rlcAmEnabled": [1],
        "e2nrEnabled": [1],
        "simTime": [2.01],
        "indicationPeriodicity": [0.1],
        "hoSinrDifference": [5],
        "dataRate": [0],
        "useSemaphores": [1]
    }

    ues = [2, 4, 6, 8, 10]
    traffic_models = [0, 1, 2, 3]
    indication_periodicity = scenario_configuration["indicationPeriodicity"][0]

    output_folder = '/workspace/ns-o-ran-gymnasium/output'
    num_steps = 1000
    offset_step = 0.1 # used only for visualization purposes

    time_path = './time.csv'

    if not Path(time_path).is_file():
        with open(time_path, 'w') as time_file:
            time_file.write('UEs,TrafficModel,Time,Overhead\n')

    for ue in ues:
        for traffic_model in traffic_models:
            scenario_configuration["ues"] = [ue]
            scenario_configuration["trafficModel"] = [traffic_model]

            print(f'Running configuration {scenario_configuration}')

            env = TrafficSteeringEnv(ns3_path='/workspace/ns3-mmwave-oran',
                                     scenario_configuration=scenario_configuration, output_folder=output_folder, optimized=True)
            
            start_time_perf = time.perf_counter_ns()
            start_time = time.process_time_ns()
            obs, info = env.reset()

            for step in range(num_steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                print(f'Step t = {round(step * indication_periodicity + offset_step, 2)},', end=' ')
                print(f'terminated = {terminated}, truncated = {truncated}')

                if terminated or truncated:
                    break

            end_time_perf = time.perf_counter_ns()
            end_time = time.process_time_ns()

            with open(time_path, 'a') as time_file:
                time_file.write(
                    f'{ue},{traffic_model},{end_time_perf - start_time_perf},{end_time - start_time}\n')
