import os
import torch

import gymnasium as gym
from env.minepump import MinePumpEnvV1

import masa
from masa.algorithms.on_policy import CPO, PPOLag, TRPOLag
from masa.common.wrappers import TimeLimit, ConstraintMonitor, RewardMonitor, ConstraintPersistentWrapper
from masa.common.labelled_env import LabelledEnv
from masa.common.label_fn import LabelFn
from masa.common import registry
from typing import Dict, Any
import numpy as np

import argparse

import jax

jax.config.update("jax_platform_name", "cpu")

class WinningRegionMonitor(ConstraintPersistentWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.in_winning_region = []

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.in_winning_region = []

        if "step" in info["constraint"]:
            self.in_winning_region.append(bool(info["in_winning_region"]))
        
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "step" in info["constraint"]:
            self.in_winning_region.append(bool(info["in_winning_region"]))

        if "episode" in info["constraint"]:
            info["constraint"]["episode"]["is_success_1"] = info["is_success_1"]
            info["constraint"]["episode"]["is_success_2"] = info["is_success_2"]
            info["constraint"]["episode"]["in_winning_region"] = np.mean(self.in_winning_region)

        return observation, reward, terminated, truncated, info

def label_fn(obs):
    labels = set()
    if obs["methane"]:
        labels.add("methane")
    if obs["highwater"]:
        labels.add("highwater")
    return labels

cost_fn = lambda labels: 1.0 if {"methane", "highwater"} <= labels else 0.0

def make_env(seed=0, max_steps=2000):
    env = MinePumpEnvV1(render_mode=None, unexpected_violation=False)
    env = TimeLimit(env, max_steps)
    constraint_ctor = registry.get_constraint("PCTL")
    env = LabelledEnv(env, label_fn)
    constraint_kwargs = {"cost_fn": cost_fn, "alpha": 0.01}
    env = constraint_ctor(env, **constraint_kwargs)
    env = ConstraintMonitor(env)
    env = RewardMonitor(env)
    env = WinningRegionMonitor(env)
    return env

def make_eval_env(seed=0, max_steps=2000):
    env = MinePumpEnvV1(render_mode=None, unexpected_violation=True)
    env = TimeLimit(env, max_steps)
    constraint_ctor = registry.get_constraint("PCTL")
    env = LabelledEnv(env, label_fn)
    constraint_kwargs = {"cost_fn": cost_fn, "alpha": 0.01}
    env = constraint_ctor(env, **constraint_kwargs)
    env = ConstraintMonitor(env)
    env = RewardMonitor(env)
    env = WinningRegionMonitor(env)
    return env


def one_run(run_seed: int, total_timesteps: int, n_eval_episodes: int, log_root: str, tb: bool = False, algo: str = "PPOLag"):
    run_dir = os.path.join(log_root, f"{algo}_seed_{run_seed}")
    os.makedirs(run_dir, exist_ok=True)

    train_env = make_env(seed=run_seed, max_steps=2000)

    algo_cls = {
        "CPO": CPO,
        "PPOLag": PPOLag,
        "TRPOLag": TRPOLag
    }[algo]

    base_kwargs = {
        "tensorboard_logdir": (str(run_dir) if tb else None),
        "seed": run_seed,
        "device": "cpu",
        "verbose": 0,
        "env_fn": make_eval_env,
    }

    if algo == "CPO":
        algo_kwargs = {
            "n_steps": 20_000,
            "learning_rate": 1e-4,
            "gae_lambda": 0.9,
            "n_critic_updates": 20,
            "fvp_sample_freq": 4,
            "cost_limit": 0.01,
        }
    
    if algo == "PPOLag":
        algo_kwargs = {
            "n_steps": 20_000,
            "learning_rate": 1e-4,
            "batch_size": 256,
            "n_epochs": 40,
            "lagrangian_multiplier_init": 10.0,
            "lambda_lr": 0.1,
            "cost_limit": 0.01,
        }

    if algo == "TRPOLag":
        algo_kwargs = {
            "n_steps": 20_000,
            "n_critic_updates": 20,
            "fvp_sample_freq:": 4,
            "lagrangian_multiplier_init": 10.0,
            "lambda_lr": 0.1,
            "cost_limit": 0.01,
        }

    model = algo_cls(train_env, **base_kwargs, **algo_kwargs)

    run_kwargs = {
        "num_eval_episodes": n_eval_episodes,
        "eval_freq": 20_000,
        "log_freq": 20_000,
        "prefill": 0,
        "save_freq": 0,
        "stats_window_size": 10,
        "stats_window_overrides": {"train/rollout/satisfied": 2000, "eval/rollout/satisfied": 200}
    }

    model.train(total_timesteps, **run_kwargs)

    return True
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runs", type=int, default=10)
    parser.add_argument("-t", "--timesteps", type=int, default=200_000)
    parser.add_argument("-e", "--eval", type=int, default=20)
    parser.add_argument("--logdir", type=str, default="./logdir/minepump")
    parser.add_argument("--tensorboard", action="store_true", default=False)
    parser.add_argument("--algo", type=str, choices=["CPO", "PPOLag", "TRPOLag"], default="PPOLag")
    args = parser.parse_args()

    runs = args.runs
    timesteps = args.timesteps
    eval_episodes = args.eval
    algo_id = args.algo

    log_root = os.path.join(args.logdir, f"{algo_id}")
    os.makedirs(log_root, exist_ok=True)

    seeds = [i for i in range(runs)]

    for i, seed in enumerate(seeds, 0):
        print(f"\n=== Run {i+1}/{runs} (seed={seed}) ===")
        res = one_run(seed, timesteps, eval_episodes, log_root, tb=args.tensorboard, algo=algo_id)

        assert res, "run failed"