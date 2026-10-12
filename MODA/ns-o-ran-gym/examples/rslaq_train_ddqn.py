"""
DDQN Training for RSLAQ — Paper-faithful implementation.

Follows Algorithm 1 and hyperparameters from:
  Yungaicela-Naula et al., "RSLAQ - A Robust SLA-driven 6G O-RAN QoS xApp
  Using Deep Reinforcement Learning", IEEE TMC 2026.

Architecture: 4 Conv2D + BatchNorm + Tanh + FC (Section IV-C)
Hyperparameters: Hyp-set3 from Table VI (LR=0.001, gamma=0.80, decay=0.998)

Usage:
    python rslaq_train_ddqn.py \
        --scenario low_traffic \
        --episodes 35 \
        --simTime 2.0 \
        --ntsr 100 \
        --output results
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import csv
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from environments.rslaq_env import RslaqEnv
from environments.rslaq_action_spaces import build_discrete_action_table
from nsoran.compute_accounting import ComputeAccounting

# Default paths
DEFAULT_NS3_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "ns-3-dev"
)
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "results")

SCENARIOS = [
    "low_traffic",
    "normal",
    "congestion",
    "stressed",
    "insufficient_resources",
]

SCENARIO_UE_PROFILES = {
    "low_traffic": (2, 2, 6),
    "normal": (5, 5, 10),
    "congestion": (15, 10, 35),
    "stressed": (8, 12, 20),
    "insufficient_resources": (20, 10, 40),
}


def apply_ue_profile(config, scenario, args):
    cli_profile = (args.embbUes, args.urllcUes, args.mtcUes)
    if all(v > 0 for v in cli_profile):
        profile = cli_profile
    elif any(v > 0 for v in cli_profile):
        raise ValueError("Provide --embbUes, --urllcUes and --mtcUes together, or leave all as 0")
    else:
        profile = SCENARIO_UE_PROFILES.get(scenario, SCENARIO_UE_PROFILES["normal"])

    config["embbUes"] = [int(profile[0])]
    config["urllcUes"] = [int(profile[1])]
    config["mtcUes"] = [int(profile[2])]
    return profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Build discrete action table (will be updated in train_ddqn based on args)
ACTION_TABLE = build_discrete_action_table(step=0.1, include_scheduler=True)
NUM_ACTIONS = len(ACTION_TABLE)


class QNetwork(nn.Module):
    """
    DNN architecture from the RSLAQ paper (Section IV-C):
    "four 2D convolutional layers, each followed by a batch
    normalization layer, and a fully connected layer at the output.
    The layers are using the Tanh activation function."
    """

    def __init__(self, state_shape, action_size):
        super().__init__()
        h, w = state_shape

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        with torch.no_grad():
            x = torch.zeros(1, 1, h, w)
            x = torch.tanh(self.bn1(self.conv1(x)))
            x = torch.tanh(self.bn2(self.conv2(x)))
            x = torch.tanh(self.bn3(self.conv3(x)))
            x = torch.tanh(self.bn4(self.conv4(x)))
            flat = x.numel()

        self.fc = nn.Linear(flat, action_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = torch.tanh(self.bn3(self.conv3(x)))
        x = torch.tanh(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DDQNAgent:
    """
    Double DQN agent matching the RSLAQ paper's DDQL (Algorithm 1).

    Defaults follow paper Hyp-set3 (Table VI) with batch/buffer
    calibrated for ns-3 (shorter episodes due to outage termination):
        lr=0.001, gamma=0.80, epsilon_decay=0.998, epsilon_min=0.05,
        buffer_size=128, batch_size=32, target_update=200.
    """

    def __init__(
        self,
        state_shape,
        action_size,
        lr=1e-3,
        gamma=0.80,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        buffer_size=128,
        batch_size=32,
        target_update=200,
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.online_net = QNetwork(state_shape, action_size).to(device)
        self.target_net = QNetwork(state_shape, action_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.target_update = target_update
        self.step_count = 0

    def act(self, state, epsilon=None):
        policy_epsilon = self.epsilon if epsilon is None else epsilon
        if random.random() < policy_epsilon:
            return random.randrange(self.action_size)
        self.online_net.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            q = self.online_net(s)
        self.online_net.train()
        return q.argmax(dim=1).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state.copy(), action, reward, next_state.copy(), done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
        actions = torch.LongTensor([b[1] for b in batch]).to(device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(device)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q = (
                self.target_net(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    def save(self, path):
        torch.save(
            {
                "online": self.online_net.state_dict(),
                "target": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", 0.0)


def choose_scenario(episode, mode, scenarios_list):
    if mode == "random":
        return random.choice(scenarios_list)
    return scenarios_list[0]


def parse_weights(weights: str) -> np.ndarray:
    values = np.array([float(x.strip()) for x in weights.split(",")], dtype=np.float64)
    if values.shape[0] != 3:
        raise ValueError("--p_sta_weights must contain exactly 3 comma-separated values")
    if values.sum() <= 0:
        raise ValueError("--p_sta_weights must sum to a positive value")
    return values / values.sum()


def evaluate_ddqn_policy(env, agent, args, scenario_list, max_steps, compute_accounting):
    eval_episodes = int(args.eval_episodes)
    if eval_episodes <= 0:
        return {"enabled": False, "eval_episodes": 0}

    eval_seed = int(args.seed) + int(args.eval_seed_offset)
    env._base_seed = eval_seed
    env._seed_cycle = int(args.eval_seed_cycle)
    env._episode_count = 0

    log_path = os.path.join(args.output, "evaluation_log.csv")
    rows = []
    rewards = []
    outage_counts = []
    soft_counts = []
    steps_completed = []

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode", "scenario", "seed", "reward_mode", "total_reward",
                "outage_count", "soft_count", "steps", "action_embb",
                "action_urllc", "action_mtc",
            ]
        )

    for ep in range(eval_episodes):
        chosen_scenario = scenario_list[ep % len(scenario_list)]
        env.scenario_configuration["scenario"] = [chosen_scenario]
        current_ue_profile = apply_ue_profile(env.scenario_configuration, chosen_scenario, args)
        env.num_ues = sum(current_ue_profile)
        env.scenario_name = chosen_scenario

        obs, info = env.reset()
        state = obs.copy()
        ep_reward = 0.0
        ep_outages = 0
        ep_soft = 0
        step = 0
        prb_pct = [0.0, 0.0, 0.0]
        ep_max_steps = min(args.ntsr, max_steps)

        for step in range(ep_max_steps):
            with compute_accounting.decision_timer.measure():
                action_idx = agent.act(state, epsilon=0)
            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            ep_reward += reward
            if info.get("outage_flags") and any(info["outage_flags"].values()):
                ep_outages += 1
            if info.get("soft_flags") and any(info["soft_flags"].values()):
                ep_soft += 1
            action_info = info.get("action_info", {})
            prb_pct = action_info.get("prb_pct", prb_pct)
            state = next_obs.copy()
            if terminated or truncated:
                break

        current_seed = int(env.scenario_configuration.get("seed", [eval_seed])[0])
        row = [
            ep + 1,
            chosen_scenario,
            current_seed,
            args.reward_mode,
            f"{ep_reward:.4f}",
            ep_outages,
            ep_soft,
            step + 1,
            f"{prb_pct[0]:.2f}",
            f"{prb_pct[1]:.2f}",
            f"{prb_pct[2]:.2f}",
        ]
        rows.append(row)
        rewards.append(ep_reward)
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
        "policy": "deterministic",
        "epsilon": 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_outage_count": float(np.mean(outage_counts)) if outage_counts else 0.0,
        "mean_soft_count": float(np.mean(soft_counts)) if soft_counts else 0.0,
        "mean_steps": float(np.mean(steps_completed)) if steps_completed else 0.0,
        "log_path": log_path,
    }


def train_ddqn(args):
    set_seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    scenario_list = (
        args.scenarios.split(",")
        if args.scenarios
        else [args.scenario]
    )
    if args.scenario == "random":
        mode = "random"
        if not args.scenarios:
            scenario_list = SCENARIOS
    else:
        mode = "fixed"
        scenario_list = [args.scenario]

    # Update action table based on include_scheduler
    global ACTION_TABLE, NUM_ACTIONS
    ACTION_TABLE = build_discrete_action_table(
        step=0.1, include_scheduler=args.include_scheduler
    )
    NUM_ACTIONS = len(ACTION_TABLE)
    print(f"Action table size: {NUM_ACTIONS} (include_scheduler={args.include_scheduler})")
    print(f"ntsr={args.ntsr}, total steps={args.episodes * args.ntsr}")

    # Paper uses ntsr=100 steps per episode (Algorithm 1, Line 17-19)
    # Set max_steps to ntsr so episodes match the paper's periodic reset
    if args.max_steps == "auto":
        max_steps = args.ntsr
    else:
        max_steps = int(args.max_steps)

    config = {
        "simTime": [args.simTime],
        "appStart": [args.appStart],
        "scenario": [scenario_list[0]],
        "seed": [args.seed],
        "seed_cycle": [args.seed_cycle],
        "periodMs": [args.periodMs],
    }
    current_ue_profile = apply_ue_profile(config, scenario_list[0], args)

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
    }
    p_sta_weights = parse_weights(args.p_sta_weights)
    interaction_budget = args.episodes * min(args.ntsr, max_steps)
    compute_accounting = ComputeAccounting(
        interaction_budget=interaction_budget,
        period_ms=args.periodMs,
        cost_per_hour_usd=args.compute_cost_per_hour_usd,
        avg_power_watts=args.compute_avg_power_watts,
        electricity_cost_usd_per_kwh=args.compute_electricity_cost_usd_per_kwh,
    )

    env = RslaqEnv(
        ns3_path=os.path.abspath(args.ns3_path),
        scenario_configuration=config,
        output_folder=args.output,
        optimized=False,
        action_mode=args.action_mode,
        observation_mode=args.observation_mode,
        max_steps=max_steps,
        apply_p_sta=args.apply_p_sta,
        p_sta_weights=p_sta_weights,
        p_sta_static_fraction=args.p_sta_static_fraction,
        include_scheduler=args.include_scheduler,
        sla_config=sla_config,
        enable_step_logging=args.enable_step_logging,
        step_log_file=args.step_log_file,
    )

    state_shape = env.observation_space.shape
    agent = DDQNAgent(
        state_shape,
        NUM_ACTIONS,
        lr=args.lr,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
    )

    log_path = os.path.join(args.output, "ddqn_training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode",
                "scenario",
                "reward_mode",
                "total_reward",
                "avg_reward",
                "outage_count",
                "soft_count",
                "steps",
                "loss",
                "epsilon",
                "final_embb_thr",
                "final_urllc_plr",
                "final_mtc_lost",
                "action_embb",
                "action_urllc",
                "action_mtc",
                "embb_ues",
                "urllc_ues",
                "mtc_ues",
            ]
        )

    best_avg = -float("inf")
    all_rewards = []
    avg100 = 0.0

    for ep in range(args.episodes):
        chosen_scenario = choose_scenario(ep, mode, scenario_list)
        env.scenario_configuration["scenario"] = [chosen_scenario]
        current_ue_profile = apply_ue_profile(env.scenario_configuration, chosen_scenario, args)
        env.num_ues = sum(current_ue_profile)
        env.scenario_name = chosen_scenario

        obs, info = env.reset()
        state = obs.copy()
        ep_reward = 0.0
        ep_outages = 0
        ep_soft = 0
        ep_losses = []
        step = 0

        # ntsr: paper's periodic reset (Algorithm 1, Lines 17-19)
        ep_max_steps = min(args.ntsr, env.max_steps)

        for step in range(ep_max_steps):
            with compute_accounting.decision_timer.measure():
                action_idx = agent.act(state)
            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            next_state = next_obs.copy()
            done = terminated or truncated

            agent.remember(state, action_idx, reward, next_state, float(done))
            loss = agent.replay()
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += reward
            if info.get("outage_flags"):
                if any(info["outage_flags"].values()):
                    ep_outages += 1
            if info.get("soft_flags"):
                if any(info["soft_flags"].values()):
                    ep_soft += 1

            state = next_state
            if done:
                break

        all_rewards.append(ep_reward)
        avg100 = (
            np.mean(all_rewards[-100:])
            if len(all_rewards) >= 100
            else np.mean(all_rewards)
        )

        final_embb_thr = info.get("slice_0_eMBB", {}).get("throughput_mbps", 0.0)
        final_urllc_plr = info.get("slice_1_URLLC", {}).get("plr_mean", 0.0)
        final_mtc_lost = info.get("slice_2_MTC", {}).get("dLostPackets_sum", 0.0)
        action_info = info.get("action_info", {})
        prb_pct = action_info.get("prb_pct", [0.0, 0.0, 0.0])

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ep + 1,
                    chosen_scenario,
                    args.reward_mode,
                    f"{ep_reward:.4f}",
                    f"{avg100:.4f}",
                    ep_outages,
                    ep_soft,
                    step + 1,
                    f"{np.mean(ep_losses):.6f}" if ep_losses else "",
                    f"{agent.epsilon:.4f}",
                    f"{final_embb_thr:.4f}",
                    f"{final_urllc_plr:.4f}",
                    f"{final_mtc_lost:.4f}",
                    f"{prb_pct[0]:.2f}",
                    f"{prb_pct[1]:.2f}",
                    f"{prb_pct[2]:.2f}",
                    current_ue_profile[0],
                    current_ue_profile[1],
                    current_ue_profile[2],
                ]
            )

        if (ep + 1) % args.log_interval == 0:
            print(
                f"Ep {ep + 1}/{args.episodes} | Scenario: {chosen_scenario} | "
                f"Reward: {ep_reward:.4f} | Avg100: {avg100:.4f} | "
                f"Outages: {ep_outages} | Soft: {ep_soft} | Eps: {agent.epsilon:.4f}"
            )

        if avg100 > best_avg:
            best_avg = avg100
            agent.save(os.path.join(args.output, "ddqn_best.pt"))

    agent.save(os.path.join(args.output, "ddqn_final.pt"))
    evaluation_metrics = evaluate_ddqn_policy(
        env, agent, args, scenario_list, max_steps, compute_accounting
    )

    summary = {
        "method": "ddqn",
        "scenario_mode": mode,
        "scenarios": scenario_list,
        "episodes": args.episodes,
        "ntsr": args.ntsr,
        "interaction_budget": interaction_budget,
        "include_scheduler": args.include_scheduler,
        "apply_p_sta": args.apply_p_sta,
        "p_sta_static_fraction": args.p_sta_static_fraction,
        "p_sta_weights": p_sta_weights.tolist(),
        "reward_mode": args.reward_mode,
        "reward_weights": [args.reward_alpha, args.reward_beta, args.reward_gamma],
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
        "terminate_on_sla_violation": args.terminate_on_sla_violation,
        "scenario_ue_profiles": SCENARIO_UE_PROFILES,
        "ue_override": {
            "eMBB": args.embbUes,
            "URLLC": args.urllcUes,
            "MTC": args.mtcUes,
        },
        "final_avg_100": float(avg100),
        "best_avg": float(best_avg),
        "evaluation_metrics": evaluation_metrics,
        "compute": compute_accounting.finish(),
    }
    with open(os.path.join(args.output, "ddqn_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    print(f"\nTraining complete. Best avg reward: {best_avg:.4f}")
    print(f"Logs saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="DDQN Training for RSLAQ (paper-faithful)")
    parser.add_argument("--scenario", type=str, default="normal",
                        help="Scenario name or 'random'")
    parser.add_argument("--scenarios", type=str, default="",
                        help="Comma-separated list for random mode (default: all)")
    parser.add_argument("--embbUes", type=int, default=0,
                        help="Override eMBB UE count; 0 uses the scenario profile")
    parser.add_argument("--urllcUes", type=int, default=0,
                        help="Override URLLC UE count; 0 uses the scenario profile")
    parser.add_argument("--mtcUes", type=int, default=0,
                        help="Override MTC UE count; 0 uses the scenario profile")
    parser.add_argument("--ns3_path", type=str, default=DEFAULT_NS3_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seed_cycle", type=int, default=100,
                        help="Change seed every N episodes (default: 100)")
    # Paper: 35 episodes × 100 steps = 3500 total steps (Fig. 5)
    parser.add_argument("--episodes", type=int, default=35)
    parser.add_argument("--simTime", type=float, default=2.0,
                        help="Simulation time in seconds (ntsr=100 + appStart=0.5 needs >=1.5s)")
    parser.add_argument("--appStart", type=float, default=0.5)
    parser.add_argument("--periodMs", type=int, default=10)
    # Paper: ntsr=100 periodic reset (Algorithm 1, Line 17-19)
    parser.add_argument("--ntsr", type=int, default=100,
                        help="Steps per episode before periodic reset (paper: 100)")
    parser.add_argument("--max_steps", type=str, default="auto",
                        help="Max steps per episode or 'auto' (uses ntsr)")
    parser.add_argument("--observation_mode", type=str, default="paper")
    parser.add_argument("--action_mode", type=str, default="discrete")
    # Paper Hyp-set3: LR=0.001
    parser.add_argument("--lr", type=float, default=1e-3)
    # Paper Hyp-set3: gamma=0.80
    parser.add_argument("--gamma", type=float, default=0.80)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    # Paper Hyp-set3: lambda_epsilon=0.998
    parser.add_argument("--epsilon_decay", type=float, default=0.998)
    # Paper: buffer L=500, batch btsz=350 (calibrated to 128/32 for ns-3)
    parser.add_argument("--buffer_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    # Paper: nsut=200 (target update interval)
    parser.add_argument("--target_update", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=1)
    # Paper Eq. 3-5: P_STA decomposition ensures slice isolation
    # p_j = 0.5*omega_j + 0.5*p_opt  → minimum allocation guaranteed
    parser.set_defaults(apply_p_sta=True)
    parser.add_argument("--no-apply-p-sta", action="store_false", dest="apply_p_sta",
                        help="Disable P_STA decomposition in Python (ns-3 handles it)")
    # Paper: action always includes scheduler (198 actions)
    parser.set_defaults(include_scheduler=True)
    parser.add_argument("--no-include-scheduler", action="store_false", dest="include_scheduler",
                        help="Disable scheduler selection in action space (default: enabled)")
    parser.add_argument("--max_buffer_bytes", type=float, default=100000.0,
                        help="Max buffer bytes for URLLC normalization")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Steps to suppress terminal conditions at episode start")
    parser.add_argument("--consecutive_outage_steps", type=int, default=5,
                        help="Consecutive steps to declare outage (default 5 = 50ms)")
    parser.set_defaults(terminate_on_sla_violation=False)
    parser.add_argument("--terminate-on-sla-violation", action="store_true",
                        dest="terminate_on_sla_violation",
                        help="End the episode on outage/soft SLA violation. Default keeps DDQN episodes on the paper ntsr reset.")
    parser.add_argument("--p_sta_static_fraction", type=float, default=0.5,
                        help="Static fraction in P_STA decomposition")
    parser.add_argument("--p_sta_weights", type=str, default="0.3333,0.4000,0.2667",
                        help="Comma-separated P_STA static weights")
    parser.add_argument("--reward_alpha", type=float, default=0.3333,
                        help="Reward weight for eMBB")
    parser.add_argument("--reward_beta", type=float, default=0.4000,
                        help="Reward weight for URLLC")
    parser.add_argument("--reward_gamma", type=float, default=0.2667,
                        help="Reward weight for MTC")
    parser.add_argument("--reward_mode", type=str, default="paper",
                        help="Reward mode: paper or resource_efficient")
    parser.add_argument("--resource_efficiency_weight", type=float, default=0.20,
                        help="Weight for served-resource efficiency shaping")
    parser.add_argument("--need_match_weight", type=float, default=0.15,
                        help="Weight for PRB allocation-to-need matching")
    parser.add_argument("--waste_penalty_weight", type=float, default=0.25,
                        help="Penalty weight for PRB over-allocation")
    parser.add_argument("--under_allocation_penalty_weight", type=float, default=0.10,
                        help="Penalty weight for PRB under-allocation")
    parser.add_argument("--embb_soft_guard_penalty_weight", type=float, default=0.25,
                        help="Penalty for resource-efficient eMBB over-serving near the soft SLA limit")
    parser.add_argument("--embb_soft_guard_ratio", type=float, default=0.80,
                        help="Fraction of eMBB soft SLA where resource-efficient guard penalty starts")
    parser.add_argument("--action_smoothness_weight", type=float, default=0.05,
                        help="Penalty weight for action oscillation")
    parser.add_argument("--resource_dynamic_need_weight", type=float, default=0.75,
                        help="Blend between static slice weights and dynamic demand need")
    parser.add_argument("--resource_waste_deadband", type=float, default=0.03,
                        help="Deadband before over/under-allocation penalties apply")
    parser.set_defaults(enable_step_logging=True)
    parser.add_argument("--no-step-logging", action="store_false", dest="enable_step_logging",
                        help="Disable per-step step_metrics.csv logging")
    parser.add_argument("--step_log_file", type=str, default="step_metrics.csv",
                        help="Per-simulation step metrics filename")
    parser.add_argument("--eval_episodes", type=int, default=0,
                        help="Run this many deterministic evaluation episodes after training")
    parser.add_argument("--eval_seed_offset", type=int, default=100000,
                        help="Offset added to --seed for deterministic evaluation ns-3 seeds")
    parser.add_argument("--eval_seed_cycle", type=int, default=1,
                        help="Change evaluation ns-3 seed every N evaluation episodes")
    parser.add_argument("--compute_cost_per_hour_usd", type=float, default=0.0,
                        help="Hourly infrastructure cost used to estimate training spend")
    parser.add_argument("--compute_avg_power_watts", type=float, default=0.0,
                        help="Average host power draw used to estimate training energy")
    parser.add_argument("--compute_electricity_cost_usd_per_kwh", type=float, default=0.0,
                        help="Electricity price used to estimate local training energy cost")
    args = parser.parse_args()

    if args.max_steps.lower() == "auto":
        args.max_steps = "auto"  # train_ddqn will use args.ntsr

    train_ddqn(args)


if __name__ == "__main__":
    main()
