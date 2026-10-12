"""
SAC Training for RSLAQ (Resource-Only).

Usage:
    python rslaq_train_sac.py \
        --scenario normal \
        --episodes 10 \
        --simTime 4.0 \
        --periodMs 10 \
        --observation_mode paper \
        --action_mode continuous \
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
from nsoran.compute_accounting import ComputeAccounting

# Default paths (relative to repo root)
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


class SACActor(nn.Module):
    def __init__(self, state_shape, action_dim=3):
        super().__init__()
        h, w = state_shape
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Compute flattened size
        with torch.no_grad():
            x = torch.zeros(1, 1, h, w)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            flat = x.numel()
        self.fc_mean = nn.Linear(flat, action_dim)
        self.fc_log_std = nn.Linear(flat, action_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, x):
        mean, _ = self.forward(x)
        return torch.tanh(mean)


class SACCritic(nn.Module):
    def __init__(self, state_shape, action_dim=3):
        super().__init__()
        h, w = state_shape
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            x = torch.zeros(1, 1, h, w)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            flat = x.numel()
        self.fc1 = nn.Linear(flat + action_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, state, action):
        if state.dim() == 2:
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 3:
            state = state.unsqueeze(1)
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class SACAgent:
    def __init__(
        self,
        state_shape,
        action_dim=3,
        lr=1e-3,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        buffer_size=10000,
        batch_size=256,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim

        self.actor = SACActor(state_shape, action_dim).to(device)
        self.critic1 = SACCritic(state_shape, action_dim).to(device)
        self.critic2 = SACCritic(state_shape, action_dim).to(device)
        self.critic1_target = SACCritic(state_shape, action_dim).to(device)
        self.critic2_target = SACCritic(state_shape, action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.step_count = 0

    def act(self, state, deterministic=False):
        s = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            if deterministic:
                a = self.actor.deterministic(s)
            else:
                a, _ = self.actor.sample(s)
        # Return continuous action in [-1, 1] (shape (3,))
        return a.cpu().numpy().flatten()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state.copy(), action.copy(), reward, next_state.copy(), float(done))
        )

    def replay(self):
        if len(self.memory) < self.batch_size:
            return None, None
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
        dones = torch.FloatTensor([b[4] for b in batch]).to(device)

        with torch.no_grad():
            next_a, next_log_prob = self.actor.sample(next_states)
            q1_t = self.critic1_target(next_states, next_a)
            q2_t = self.critic2_target(next_states, next_a)
            q_t = torch.min(q1_t, q2_t) - self.alpha * next_log_prob
            q_backup = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * q_t

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        c1_loss = nn.MSELoss()(q1, q_backup)
        c2_loss = nn.MSELoss()(q2, q_backup)

        self.critic1_opt.zero_grad()
        c1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        c2_loss.backward()
        self.critic2_opt.step()

        a_new, log_prob_new = self.actor.sample(states)
        q1_pi = self.critic1(states, a_new)
        q2_pi = self.critic2(states, a_new)
        actor_loss = (self.alpha * log_prob_new - torch.min(q1_pi, q2_pi)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.step_count += 1
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

    def load(self, path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])


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


def evaluate_sac_policy(env, agent, args, scenario_list, compute_accounting):
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

        for step in range(env.max_steps):
            with compute_accounting.decision_timer.measure():
                action_cont = agent.act(state, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action_cont)
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
        rows.append(
            [
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
        )
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
        "policy": "deterministic_tanh_mean",
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_outage_count": float(np.mean(outage_counts)) if outage_counts else 0.0,
        "mean_soft_count": float(np.mean(soft_counts)) if soft_counts else 0.0,
        "mean_steps": float(np.mean(steps_completed)) if steps_completed else 0.0,
        "log_path": log_path,
    }


def train_sac(args):
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

    # Initialize env once to get spaces (scenario will be updated per episode)
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
        "period_ms": args.periodMs,
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

    env = RslaqEnv(
        ns3_path=os.path.abspath(args.ns3_path),
        scenario_configuration=config,
        output_folder=args.output,
        optimized=False,
        action_mode=args.action_mode,
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

    state_shape = env.observation_space.shape
    agent = SACAgent(
        state_shape,
        action_dim=3,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
    )

    log_path = os.path.join(args.output, "sac_training_log.csv")
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
                "actor_loss",
                "critic_loss",
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
        ep_actor_losses = []
        ep_critic_losses = []
        step = 0

        for step in range(env.max_steps):
            with compute_accounting.decision_timer.measure():
                action_cont = agent.act(state)
            next_obs, reward, terminated, truncated, info = env.step(action_cont)
            next_state = next_obs.copy()
            done = terminated or truncated

            agent.remember(state, action_cont, reward, next_state, float(done))
            c_loss, a_loss = agent.replay()
            if c_loss is not None:
                ep_critic_losses.append(c_loss)
            if a_loss is not None:
                ep_actor_losses.append(a_loss)

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
        avg100 = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)

        # Extract final KPIs from info
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
                    f"{np.mean(ep_actor_losses):.6f}" if ep_actor_losses else "",
                    f"{np.mean(ep_critic_losses):.6f}" if ep_critic_losses else "",
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
                f"Outages: {ep_outages} | Soft: {ep_soft}"
            )

        if avg100 > best_avg:
            best_avg = avg100
            agent.save(os.path.join(args.output, "sac_best.pt"))

    agent.save(os.path.join(args.output, "sac_final.pt"))
    evaluation_metrics = evaluate_sac_policy(
        env, agent, args, scenario_list, compute_accounting
    )

    summary = {
        "method": "sac",
        "scenario_mode": mode,
        "scenarios": scenario_list,
        "episodes": args.episodes,
        "interaction_budget": interaction_budget,
        "apply_p_sta": args.apply_p_sta,
        "p_sta_static_fraction": args.p_sta_static_fraction,
        "p_sta_weights": p_sta_weights.tolist(),
        "reward_mode": args.reward_mode,
        "reward_weights": [args.reward_alpha, args.reward_beta, args.reward_gamma],
        "terminate_on_sla_violation": args.terminate_on_sla_violation,
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
    with open(os.path.join(args.output, "sac_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    env.close()
    print(f"\nTraining complete. Best avg reward: {best_avg:.4f}")
    print(f"Logs saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="SAC Training for RSLAQ")
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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--simTime", type=float, default=4.0)
    parser.add_argument("--appStart", type=float, default=0.5)
    parser.add_argument("--periodMs", type=int, default=10)
    parser.add_argument("--max_steps", type=str, default="auto",
                        help="Max steps per episode or 'auto'")
    parser.add_argument("--observation_mode", type=str, default="paper")
    parser.add_argument("--action_mode", type=str, default="continuous")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=1)
    # Paper Eq. 3-5: P_STA decomposition ensures slice isolation
    # p_j = 0.5*omega_j + 0.5*p_opt  → minimum allocation guaranteed
    parser.set_defaults(apply_p_sta=True)
    parser.add_argument("--no-apply-p-sta", action="store_false", dest="apply_p_sta",
                        help="Disable P_STA decomposition in Python (ns-3 handles it)")
    parser.add_argument("--max_buffer_bytes", type=float, default=100000.0,
                        help="Max buffer bytes for URLLC normalization")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Steps to suppress terminal conditions at episode start")
    parser.add_argument("--consecutive_outage_steps", type=int, default=5,
                        help="Consecutive steps below threshold to declare outage (default 5 = 50ms)")
    parser.set_defaults(terminate_on_sla_violation=False)
    parser.add_argument("--terminate-on-sla-violation", action="store_true",
                        dest="terminate_on_sla_violation",
                        help="End the episode on outage/soft SLA violation. Default keeps SAC episodes on the configured max_steps horizon.")
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
                        choices=["paper", "resource_efficient"],
                        help="Reward formulation: paper RSLAQ or the resource-efficient contribution")
    parser.add_argument("--resource_efficiency_weight", type=float, default=0.20,
                        help="Positive weight for served demand per efficient slice allocation")
    parser.add_argument("--need_match_weight", type=float, default=0.15,
                        help="Positive weight for matching PRB share to dynamic slice need")
    parser.add_argument("--waste_penalty_weight", type=float, default=0.25,
                        help="Penalty for over-allocating PRBs beyond dynamic slice need")
    parser.add_argument("--under_allocation_penalty_weight", type=float, default=0.10,
                        help="Penalty for under-allocating PRBs to active slice need")
    parser.add_argument("--embb_soft_guard_penalty_weight", type=float, default=0.25,
                        help="Penalty for resource-efficient eMBB over-serving near the soft SLA limit")
    parser.add_argument("--embb_soft_guard_ratio", type=float, default=0.80,
                        help="Fraction of eMBB soft SLA where resource-efficient guard penalty starts")
    parser.add_argument("--action_smoothness_weight", type=float, default=0.05,
                        help="Penalty for abrupt PRB-share changes between control steps")
    parser.add_argument("--resource_dynamic_need_weight", type=float, default=0.75,
                        help="Blend factor for dynamic demand versus static RSLAQ weights")
    parser.add_argument("--resource_waste_deadband", type=float, default=0.03,
                        help="Allocation-share tolerance before over/under-allocation penalties apply")
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
        args.max_steps = None
    else:
        args.max_steps = int(args.max_steps)

    train_sac(args)


if __name__ == "__main__":
    main()
