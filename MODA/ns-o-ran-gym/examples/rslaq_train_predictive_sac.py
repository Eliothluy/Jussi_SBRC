#!/usr/bin/env python3
"""Predictive SAC training for RSLAQ.

The policy observes the current RSLAQ 4x4 state plus a GRU forecast of
near-future SLA risk/KPIs. The forecaster is trained offline from previous
``step_metrics.csv`` logs and remains frozen during online SAC training.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import json
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environments.rslaq_env import RslaqEnv
from environments.rslaq_predictive import (
    FEATURE_DIM,
    FORECAST_DIM,
    RISK_DIM,
    TemporalKpiForecaster,
    TemporalKpiForecasterLSTM,
    StepFrame,
    forecast_from_history,
    frame_to_feature,
    load_forecaster,
)
from nsoran.compute_accounting import ComputeAccounting


DEFAULT_NS3_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "ns-3-dev")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "results_predictive")

SCENARIOS = [
    "low_traffic",
    "normal",
    "congestion",
    "stressed",
    "insufficient_resources",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_weights(weights: str) -> np.ndarray:
    values = np.array([float(x.strip()) for x in weights.split(",")], dtype=np.float64)
    if values.shape[0] != 3:
        raise ValueError("--p_sta_weights must contain exactly 3 comma-separated values")
    if values.sum() <= 0:
        raise ValueError("--p_sta_weights must sum to a positive value")
    return values / values.sum()


def choose_scenario(episode, mode, scenarios_list):
    if mode == "random":
        return random.choice(scenarios_list)
    return scenarios_list[0]


# load_forecaster is now imported from rslaq_predictive (factory-based)


class VectorSACActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        z = self.backbone(x)
        mean = self.fc_mean(z)
        log_std = torch.clamp(self.fc_log_std(z), -20, 2)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, x):
        mean, _ = self.forward(x)
        return torch.tanh(mean)


class VectorSACCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class PredictiveSACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.1,
        buffer_size: int = 50000,
        batch_size: int = 256,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)

        self.actor = VectorSACActor(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = VectorSACCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = VectorSACCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = VectorSACCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = VectorSACCritic(state_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

    def act(self, state, deterministic=False):
        s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                a = self.actor.deterministic(s)
            else:
                a, _ = self.actor.sample(s)
        return a.cpu().numpy().reshape(-1)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                np.asarray(action, dtype=np.float32).copy(),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                float(done),
            )
        )

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None, None
        batch = random.sample(self.memory, self.batch_size)
        states = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.array([b[1] for b in batch]), dtype=torch.float32, device=device)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)
        dones = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            q1_t = self.critic1_target(next_states, next_action)
            q2_t = self.critic2_target(next_states, next_action)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_log_prob
            q_backup = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * q_t

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        c1_loss = nn.MSELoss()(q1, q_backup)
        c2_loss = nn.MSELoss()(q2, q_backup)

        self.critic1_opt.zero_grad()
        c1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        c2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_opt.step()

        action_new, log_prob_new = self.actor.sample(states)
        q1_pi = self.critic1(states, action_new)
        q2_pi = self.critic2(states, action_new)
        actor_loss = (self.alpha * log_prob_new - torch.min(q1_pi, q2_pi)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return c1_loss.item(), actor_loss.item()

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
            },
            path,
        )


def feature_from_transition(obs, reward, info, scenario):
    action_info = info.get("action_info", {})
    prb_pct = action_info.get("prb_pct", [33.33, 33.33, 33.34])
    outage = info.get("outage_flags", {})
    soft = info.get("soft_flags", {})
    frame = StepFrame(
        sim_id=info.get("results", {}).get("meta", {}).get("id", ""),
        scenario=scenario,
        seed=0,
        episode=0,
        step=int(info.get("num_steps", 0)),
        obs=np.asarray(obs, dtype=np.float32),
        action=np.asarray(prb_pct, dtype=np.float32) / 100.0,
        reward=float(reward),
        outage_flags=np.array([bool(outage.get(sid, False)) for sid in range(3)], dtype=np.float32),
        soft_flags=np.array([bool(soft.get(sid, False)) for sid in range(3)], dtype=np.float32),
    )
    return frame_to_feature(frame)


def augmented_state(obs, forecast, last_action):
    return np.concatenate(
        [
            np.asarray(obs, dtype=np.float32).reshape(-1),
            np.asarray(forecast, dtype=np.float32).reshape(-1),
            np.asarray(last_action, dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32)


def evaluate_predictive_policy(
    env,
    agent,
    args,
    scenario_list,
    forecaster,
    sequence_len,
    compute_accounting,
):
    eval_episodes = int(args.eval_episodes)
    if eval_episodes <= 0:
        return {"enabled": False, "eval_episodes": 0}

    eval_seed = int(args.seed) + int(args.eval_seed_offset)
    env._base_seed = eval_seed
    env._seed_cycle = int(args.eval_seed_cycle)
    env._episode_count = 0

    log_path = os.path.join(args.output, "evaluation_log.csv")
    rows = []
    base_rewards = []
    shaped_rewards = []
    outage_counts = []
    soft_counts = []
    steps_completed = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode", "scenario", "seed", "reward_mode", "base_reward",
                "shaped_reward", "outage_count", "soft_count", "steps",
                "forecast_outage_risk", "forecast_soft_risk", "action_embb",
                "action_urllc", "action_mtc",
            ]
        )

    for ep in range(eval_episodes):
        chosen_scenario = scenario_list[ep % len(scenario_list)]
        env.scenario_configuration["scenario"] = [chosen_scenario]
        env.scenario_name = chosen_scenario
        obs, info = env.reset()

        history = deque(maxlen=sequence_len)
        last_action = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        forecast = forecast_from_history(
            forecaster, history, sequence_len, chosen_scenario, device=device
        )
        state = augmented_state(obs, forecast, last_action)

        ep_base_reward = 0.0
        ep_shaped_reward = 0.0
        ep_outages = 0
        ep_soft = 0
        ep_forecast_outage = []
        ep_forecast_soft = []
        step = 0
        prb_pct = [0.0, 0.0, 0.0]

        for step in range(env.max_steps):
            decision_start = time.perf_counter()
            action = agent.act(state, deterministic=True)
            policy_decision_ms = (time.perf_counter() - decision_start) * 1000.0
            next_obs, reward, terminated, truncated, info = env.step(action)

            action_info = info.get("action_info", {})
            prb_pct = action_info.get("prb_pct", prb_pct)
            last_action = np.asarray(prb_pct, dtype=np.float32) / 100.0
            history.append(feature_from_transition(next_obs, reward, info, chosen_scenario))
            forecast_start = time.perf_counter()
            next_forecast = forecast_from_history(
                forecaster, history, sequence_len, chosen_scenario, device=device
            )
            forecast_decision_ms = (time.perf_counter() - forecast_start) * 1000.0
            compute_accounting.decision_timer.record_ms(
                policy_decision_ms + forecast_decision_ms
            )

            outage_risk = float(np.max(forecast[:3]))
            soft_risk = float(np.max(forecast[3:RISK_DIM]))
            shaped_reward = (
                float(reward)
                - args.risk_penalty * outage_risk
                - args.soft_penalty * soft_risk
            )

            ep_base_reward += float(reward)
            ep_shaped_reward += shaped_reward
            ep_forecast_outage.append(outage_risk)
            ep_forecast_soft.append(soft_risk)
            if info.get("outage_flags") and any(info["outage_flags"].values()):
                ep_outages += 1
            if info.get("soft_flags") and any(info["soft_flags"].values()):
                ep_soft += 1

            state = augmented_state(next_obs, next_forecast, last_action)
            forecast = next_forecast
            if terminated or truncated:
                break

        current_seed = int(env.scenario_configuration.get("seed", [eval_seed])[0])
        rows.append(
            [
                ep + 1,
                chosen_scenario,
                current_seed,
                args.reward_mode,
                f"{ep_base_reward:.4f}",
                f"{ep_shaped_reward:.4f}",
                ep_outages,
                ep_soft,
                step + 1,
                f"{np.mean(ep_forecast_outage):.6f}" if ep_forecast_outage else "",
                f"{np.mean(ep_forecast_soft):.6f}" if ep_forecast_soft else "",
                f"{prb_pct[0]:.2f}",
                f"{prb_pct[1]:.2f}",
                f"{prb_pct[2]:.2f}",
            ]
        )
        base_rewards.append(ep_base_reward)
        shaped_rewards.append(ep_shaped_reward)
        outage_counts.append(ep_outages)
        soft_counts.append(ep_soft)
        steps_completed.append(step + 1)

    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerows(rows)

    return {
        "enabled": True,
        "eval_episodes": eval_episodes,
        "eval_seed": eval_seed,
        "eval_seed_cycle": int(args.eval_seed_cycle),
        "policy": "deterministic_tanh_mean",
        "mean_base_reward": float(np.mean(base_rewards)) if base_rewards else 0.0,
        "mean_shaped_reward": float(np.mean(shaped_rewards)) if shaped_rewards else 0.0,
        "mean_outage_count": float(np.mean(outage_counts)) if outage_counts else 0.0,
        "mean_soft_count": float(np.mean(soft_counts)) if soft_counts else 0.0,
        "mean_steps": float(np.mean(steps_completed)) if steps_completed else 0.0,
        "log_path": log_path,
    }


def train(args):
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    forecaster, ckpt_sequence_len, _ = load_forecaster(args.forecaster_checkpoint, device=device)
    sequence_len = args.sequence_len or ckpt_sequence_len

    scenario_list = args.scenarios.split(",") if args.scenarios else [args.scenario]
    if args.scenario == "random":
        mode = "random"
        if not args.scenarios:
            scenario_list = SCENARIOS
    else:
        mode = "fixed"
        scenario_list = [args.scenario]

    config = {
        "simTime": [args.simTime],
        "appStart": [args.appStart],
        "scenario": [scenario_list[0]],
        "seed": [args.seed],
        "seed_cycle": [args.seed_cycle],
        "periodMs": [args.periodMs],
    }
    sla_config = {
        "max_buffer_bytes": args.max_buffer_bytes,
        "warmup_steps": args.warmup_steps,
        "consecutive_outage_steps": args.consecutive_outage_steps,
        "terminate_on_sla_violation": args.terminate_on_sla_violation,
        "alpha": args.reward_alpha,
        "beta": args.reward_beta,
        "gamma": args.reward_gamma,
        "reward_mode": args.reward_mode,
        "resource_efficiency_weight": args.resource_efficiency_weight,
        "need_match_weight": args.need_match_weight,
        "waste_penalty_weight": args.waste_penalty_weight,
        "under_allocation_penalty_weight": args.under_allocation_penalty_weight,
        "embb_soft_guard_penalty_weight": args.embb_soft_guard_penalty_weight,
        "embb_soft_guard_ratio": args.embb_soft_guard_ratio,
        "action_smoothness_weight": args.action_smoothness_weight,
        "resource_dynamic_need_weight": args.resource_dynamic_need_weight,
        "resource_waste_deadband": args.resource_waste_deadband,
        "demand_aware_embb_outage": args.demand_aware_embb_outage,
        "period_ms": args.periodMs,
    }
    p_sta_weights = parse_weights(args.p_sta_weights)

    env = RslaqEnv(
        ns3_path=os.path.abspath(args.ns3_path),
        scenario_configuration=config,
        output_folder=args.output,
        optimized=False,
        action_mode="continuous",
        observation_mode=args.observation_mode,
        max_steps=args.max_steps if args.max_steps != "auto" else None,
        apply_p_sta=args.apply_p_sta,
        p_sta_weights=p_sta_weights,
        p_sta_static_fraction=args.p_sta_static_fraction,
        sla_config=sla_config,
        enable_step_logging=args.enable_step_logging,
        step_log_file=args.step_log_file,
    )
    interaction_budget = args.episodes * env.max_steps
    compute_accounting = ComputeAccounting(
        interaction_budget=interaction_budget,
        period_ms=args.periodMs,
        cost_per_hour_usd=args.compute_cost_per_hour_usd,
        avg_power_watts=args.compute_avg_power_watts,
        electricity_cost_usd_per_kwh=args.compute_electricity_cost_usd_per_kwh,
    )

    state_dim = int(np.prod(env.observation_space.shape)) + FORECAST_DIM + 3
    agent = PredictiveSACAgent(
        state_dim=state_dim,
        hidden_dim=args.policy_hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
    )

    log_path = os.path.join(args.output, "predictive_sac_training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode", "scenario", "base_reward", "shaped_reward",
                "avg_shaped_reward", "outage_count", "soft_count", "steps",
                "actor_loss", "critic_loss", "forecast_outage_risk",
                "forecast_soft_risk", "action_embb", "action_urllc",
                "action_mtc",
            ]
        )

    all_shaped_rewards = []
    best_avg = -float("inf")

    for ep in range(args.episodes):
        chosen_scenario = choose_scenario(ep, mode, scenario_list)
        env.scenario_configuration["scenario"] = [chosen_scenario]
        env.scenario_name = chosen_scenario
        obs, info = env.reset()

        history = deque(maxlen=sequence_len)
        last_action = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        forecast = forecast_from_history(
            forecaster, history, sequence_len, chosen_scenario, device=device
        )
        state = augmented_state(obs, forecast, last_action)

        ep_base_reward = 0.0
        ep_shaped_reward = 0.0
        ep_outages = 0
        ep_soft = 0
        ep_actor_losses = []
        ep_critic_losses = []
        ep_forecast_outage = []
        ep_forecast_soft = []
        step = 0
        prb_pct = [0.0, 0.0, 0.0]

        for step in range(env.max_steps):
            decision_start = time.perf_counter()
            action = agent.act(state)
            policy_decision_ms = (time.perf_counter() - decision_start) * 1000.0
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            action_info = info.get("action_info", {})
            prb_pct = action_info.get("prb_pct", [0.0, 0.0, 0.0])
            last_action = np.asarray(prb_pct, dtype=np.float32) / 100.0
            history.append(feature_from_transition(next_obs, reward, info, chosen_scenario))
            forecast_start = time.perf_counter()
            next_forecast = forecast_from_history(
                forecaster, history, sequence_len, chosen_scenario, device=device
            )
            forecast_decision_ms = (time.perf_counter() - forecast_start) * 1000.0
            compute_accounting.decision_timer.record_ms(
                policy_decision_ms + forecast_decision_ms
            )

            outage_risk = float(np.max(forecast[:3]))
            soft_risk = float(np.max(forecast[3:RISK_DIM]))
            shaped_reward = (
                float(reward)
                - args.risk_penalty * outage_risk
                - args.soft_penalty * soft_risk
            )

            next_state = augmented_state(next_obs, next_forecast, last_action)
            agent.remember(state, action, shaped_reward, next_state, done)
            c_loss, a_loss = agent.replay()
            if c_loss is not None:
                ep_critic_losses.append(c_loss)
            if a_loss is not None:
                ep_actor_losses.append(a_loss)

            ep_base_reward += float(reward)
            ep_shaped_reward += shaped_reward
            ep_forecast_outage.append(outage_risk)
            ep_forecast_soft.append(soft_risk)

            if info.get("outage_flags") and any(info["outage_flags"].values()):
                ep_outages += 1
            if info.get("soft_flags") and any(info["soft_flags"].values()):
                ep_soft += 1

            state = next_state
            forecast = next_forecast
            if done:
                break

        all_shaped_rewards.append(ep_shaped_reward)
        avg100 = (
            np.mean(all_shaped_rewards[-100:])
            if len(all_shaped_rewards) >= 100
            else np.mean(all_shaped_rewards)
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ep + 1,
                    chosen_scenario,
                    f"{ep_base_reward:.4f}",
                    f"{ep_shaped_reward:.4f}",
                    f"{avg100:.4f}",
                    ep_outages,
                    ep_soft,
                    step + 1,
                    f"{np.mean(ep_actor_losses):.6f}" if ep_actor_losses else "",
                    f"{np.mean(ep_critic_losses):.6f}" if ep_critic_losses else "",
                    f"{np.mean(ep_forecast_outage):.6f}" if ep_forecast_outage else "",
                    f"{np.mean(ep_forecast_soft):.6f}" if ep_forecast_soft else "",
                    f"{prb_pct[0]:.2f}",
                    f"{prb_pct[1]:.2f}",
                    f"{prb_pct[2]:.2f}",
                ]
            )

        if (ep + 1) % args.log_interval == 0:
            print(
                f"Ep {ep + 1}/{args.episodes} | Scenario: {chosen_scenario} | "
                f"Base: {ep_base_reward:.4f} | Shaped: {ep_shaped_reward:.4f} | "
                f"Avg100: {avg100:.4f} | Outages: {ep_outages} | Soft: {ep_soft}"
            )

        if avg100 > best_avg:
            best_avg = avg100
            agent.save(os.path.join(args.output, "predictive_sac_best.pt"))

    agent.save(os.path.join(args.output, "predictive_sac_final.pt"))
    evaluation_metrics = evaluate_predictive_policy(
        env, agent, args, scenario_list, forecaster, sequence_len, compute_accounting
    )
    summary = {
        "method": "predictive_sac",
        "scenario_mode": mode,
        "scenarios": scenario_list,
        "episodes": args.episodes,
        "interaction_budget": interaction_budget,
        "forecaster_checkpoint": os.path.abspath(args.forecaster_checkpoint),
        "sequence_len": sequence_len,
        "forecast_dim": FORECAST_DIM,
        "risk_penalty": args.risk_penalty,
        "soft_penalty": args.soft_penalty,
        "terminate_on_sla_violation": args.terminate_on_sla_violation,
        "demand_aware_embb_outage": args.demand_aware_embb_outage,
        "apply_p_sta": args.apply_p_sta,
        "p_sta_static_fraction": args.p_sta_static_fraction,
        "p_sta_weights": p_sta_weights.tolist(),
        "reward_weights": [args.reward_alpha, args.reward_beta, args.reward_gamma],
        "reward_mode": args.reward_mode,
        "resource_efficiency_reward": {
            "resource_efficiency_weight": args.resource_efficiency_weight,
            "need_match_weight": args.need_match_weight,
            "waste_penalty_weight": args.waste_penalty_weight,
            "under_allocation_penalty_weight": args.under_allocation_penalty_weight,
            "embb_soft_guard_penalty_weight": args.embb_soft_guard_penalty_weight,
            "embb_soft_guard_ratio": args.embb_soft_guard_ratio,
            "action_smoothness_weight": args.action_smoothness_weight,
            "resource_dynamic_need_weight": args.resource_dynamic_need_weight,
            "resource_waste_deadband": args.resource_waste_deadband,
        },
        "final_avg_100": float(avg100),
        "best_avg": float(best_avg),
        "evaluation_metrics": evaluation_metrics,
        "compute": compute_accounting.finish(),
    }
    with open(os.path.join(args.output, "predictive_sac_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    print(f"Training complete. Best avg shaped reward: {best_avg:.4f}")
    print(f"Logs saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Predictive SAC Training for RSLAQ")
    parser.add_argument("--scenario", type=str, default="normal")
    parser.add_argument("--scenarios", type=str, default="")
    parser.add_argument("--ns3_path", type=str, default=DEFAULT_NS3_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--forecaster_checkpoint", type=str, required=True)
    parser.add_argument("--sequence_len", type=int, default=0,
                        help="0 uses the value saved in the forecaster checkpoint")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed_cycle", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--simTime", type=float, default=4.0)
    parser.add_argument("--appStart", type=float, default=0.5)
    parser.add_argument("--periodMs", type=int, default=10)
    parser.add_argument("--max_steps", type=str, default="auto")
    parser.add_argument("--observation_mode", type=str, default="paper")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--policy_hidden_dim", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--risk_penalty", type=float, default=0.5)
    parser.add_argument("--soft_penalty", type=float, default=0.2)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.set_defaults(apply_p_sta=True)
    parser.add_argument("--no-apply-p-sta", action="store_false", dest="apply_p_sta")
    parser.add_argument("--max_buffer_bytes", type=float, default=100000.0)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--consecutive_outage_steps", type=int, default=5)
    parser.set_defaults(terminate_on_sla_violation=False)
    parser.add_argument("--terminate-on-sla-violation", action="store_true",
                        dest="terminate_on_sla_violation")
    parser.add_argument("--p_sta_static_fraction", type=float, default=0.25)
    parser.add_argument("--p_sta_weights", type=str, default="0.3333,0.4000,0.2667")
    parser.add_argument("--reward_alpha", type=float, default=0.3333)
    parser.add_argument("--reward_beta", type=float, default=0.4000)
    parser.add_argument("--reward_gamma", type=float, default=0.2667)
    parser.add_argument("--reward_mode", type=str, default="resource_efficient",
                        choices=["paper", "resource_efficient"],
                        help="Predictive SAC reward formulation")
    parser.add_argument("--resource_efficiency_weight", type=float, default=0.25)
    parser.add_argument("--need_match_weight", type=float, default=0.30)
    parser.add_argument("--waste_penalty_weight", type=float, default=0.35)
    parser.add_argument("--under_allocation_penalty_weight", type=float, default=0.15)
    parser.add_argument("--embb_soft_guard_penalty_weight", type=float, default=0.25)
    parser.add_argument("--embb_soft_guard_ratio", type=float, default=0.80)
    parser.add_argument("--action_smoothness_weight", type=float, default=0.05)
    parser.add_argument("--resource_dynamic_need_weight", type=float, default=0.75)
    parser.add_argument("--resource_waste_deadband", type=float, default=0.03)
    parser.set_defaults(demand_aware_embb_outage=True)
    parser.add_argument("--paper-faithful-outage", action="store_false",
                        dest="demand_aware_embb_outage")
    parser.set_defaults(enable_step_logging=True)
    parser.add_argument("--no-step-logging", action="store_false", dest="enable_step_logging")
    parser.add_argument("--step_log_file", type=str, default="step_metrics.csv")
    parser.add_argument("--eval_episodes", type=int, default=0,
                        help="Run this many deterministic evaluation episodes after training")
    parser.add_argument("--eval_seed_offset", type=int, default=100000,
                        help="Offset added to --seed for deterministic evaluation ns-3 seeds")
    parser.add_argument("--eval_seed_cycle", type=int, default=1,
                        help="Change evaluation ns-3 seed every N evaluation episodes")
    parser.add_argument("--compute_cost_per_hour_usd", type=float, default=0.0)
    parser.add_argument("--compute_avg_power_watts", type=float, default=0.0)
    parser.add_argument("--compute_electricity_cost_usd_per_kwh", type=float, default=0.0)
    args = parser.parse_args()

    if args.max_steps.lower() == "auto":
        args.max_steps = "auto"
    else:
        args.max_steps = int(args.max_steps)

    train(args)


if __name__ == "__main__":
    main()
