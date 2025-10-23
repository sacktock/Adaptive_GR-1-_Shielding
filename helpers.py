import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

class EpisodeAccumulator(BaseCallback):
    """
    Collects per-episode training reward and is_success (if present) by reading 'infos'
    for 'episode' keys emitted by Monitor at episode ends.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        self.in_winning_region = []
        self.guarantee_1 = []
        self.guarantee_2 = []

        self.ep_rewards = []
        self.ep_success = []
        self.ep_overrides = []
        self.ep_success_1 = []
        self.ep_success_2 = []

        self.ep_in_winning_region = []
        self.ep_guarantee_1 = []
        self.ep_guarantee_2 = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is None:
            return True
        for info in infos:
            if "in_winning_region" in info:
                self.in_winning_region.append(float(info["in_winning_region"]))
            if "guarantee_1" in info:
                self.guarantee_1.append(float(info["guarantee_1"]))
            if "guarantee_2" in info:
                self.guarantee_2.append(float(info["guarantee_2"]))

            if "episode" in info:
                self.ep_rewards.append(float(info["episode"]["r"]))
                if "is_success" in info:
                    self.ep_success.append(float(info["is_success"]))
                if "override" in info:
                    self.ep_overrides.append(float(info["override"]))
                if "is_success_1" in info:
                    self.ep_success_1.append(float(info["is_success_1"]))
                if "is_success_2" in info:
                    self.ep_success_2.append(float(info["is_success_1"]))
                if "in_winning_region" in info:
                    self.ep_in_winning_region.append(float(np.mean(self.in_winning_region)))
                    self.in_winning_region = []
                if "guarantee_1" in info:
                    self.ep_guarantee_1.append(float(np.mean(self.guarantee_1)))
                    self.guarantee_1 = []
                if "guarantee_2" in info:
                    self.ep_guarantee_2.append(float(np.mean(self.guarantee_2)))
                    self.guarantee_2 = []

        return True

    def get_training_averages(self):
        rew_mean = np.mean(self.ep_rewards) if len(self.ep_rewards) > 0 else np.nan
        if len(self.ep_success) > 0:
            succ = np.array(self.ep_success, dtype=np.float32)
            succ_mean = np.nanmean(succ) if np.any(~np.isnan(succ)) else np.nan
        else:
            succ_mean = np.nan
        if len(self.ep_overrides) > 0:
            over = np.array(self.ep_success, dtype=np.float32)
            over_mean = np.nanmean(over) if np.any(~np.isnan(over)) else np.nan
        else:
            over_mean = np.nan
        if len(self.ep_success_1) > 0:
            succ_1 = np.array(self.ep_success_1, dtype=np.float32)
            succ_1_mean = np.nanmean(succ_1) if np.any(~np.isnan(succ_1)) else np.nan
        else:
            succ_1_mean = np.nan
        if len(self.ep_success_2) > 0:
            succ_2 = np.array(self.ep_success_2, dtype=np.float32)
            succ_2_mean = np.nanmean(succ_2) if np.any(~np.isnan(succ_2)) else np.nan
        else:
            succ_2_mean = np.nan
        if len(self.ep_in_winning_region) > 0:
            in_winning_region = np.array(self.ep_in_winning_region, dtype=np.float32)
            in_winning_region_mean = np.nanmean(in_winning_region) if np.any(~np.isnan(in_winning_region)) else np.nan
        else:
            in_winning_region_mean = np.nan
        if len(self.ep_guarantee_1) > 0:
            guarantee_1 = np.array(self.ep_guarantee_1, dtype=np.float32)
            guarantee_1_mean = np.nanmean(guarantee_1) if np.any(~np.isnan(guarantee_1)) else np.nan
        else:
            guarantee_1_mean = np.nan
        if len(self.ep_guarantee_2) > 0:
            guarantee_2 = np.array(self.ep_guarantee_2, dtype=np.float32)
            guarantee_2_mean = np.nanmean(guarantee_2) if np.any(~np.isnan(guarantee_2)) else np.nan
        else:
            guarantee_2_mean = np.nan

        output = {
            "train_reward_mean":rew_mean,
            "train_success_mean":succ_mean,
            "train_override_mean":over_mean,
            "train_success_1_mean":succ_1_mean,
            "train_success_2_mean":succ_2_mean,
            "train_winning_region_mean":in_winning_region_mean,
            "train_guarantee_1_mean":guarantee_1_mean,
            "train_guarantee_2_mean":guarantee_2_mean,
        }
        
        return output

def evaluate_controller(contoller, env, n_eval_episodes=10):

    episode_rewards = []
    episode_success = []
    episode_success_1 = []
    episode_success_2 = []
    episode_in_winning_region = []
    episode_guarantee_1 = []
    episode_guarantee_2 = []

    for episode in range(n_eval_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        in_winning_region = []
        guarantee_1 = []
        guarantee_2 = []

        while not (done or truncated):
            act = contoller(obs)
            obs, reward, done, truncated, info = env.step(act)
            total_reward += reward

            if "in_winning_region" in info:
                in_winning_region.append(float(info["in_winning_region"]))
            if "guarantee_1" in info:
                guarantee_1.append(float(info["guarantee_1"]))
            if "guarantee_2" in info:
                guarantee_2.append(float(info["guarantee_2"]))

        episode_rewards.append(total_reward)
        episode_success.append(int(info["is_success"]))
        if "is_success_1" in info:
            episode_success_1.append(int(info["is_success_1"]))
        if "is_success_2" in info:
            episode_success_2.append(int(info["is_success_2"]))
        if "in_winning_region" in info:
            episode_in_winning_region.append(np.mean(in_winning_region))
        if "guarantee_1" in info:
            episode_guarantee_1.append(np.mean(guarantee_1))
        if "guarantee_2" in info:
            episode_guarantee_2.append(np.mean(guarantee_2))

    output = {
        "eval_reward_mean": np.mean(episode_rewards),
        "eval_reward_std": np.std(episode_rewards),
        "eval_success_mean": np.mean(episode_success),
        "eval_success_1_mean": np.nanmean(episode_success_1),
        "eval_success_2_mean": np.nanmean(episode_success_2),
        "eval_in_winning_region_mean": np.nanmean(episode_in_winning_region),
        "eval_guarantee_1_mean": np.nanmean(episode_guarantee_1),
        "eval_guarantee_2_mean": np.nanmean(episode_guarantee_2),
    }
        
    return output

def evaluate_policy(model, env, n_eval_episodes=10):

    episode_rewards = []
    episode_success = []
    episode_overrides = []
    episode_success_1 = []
    episode_success_2 = []
    episode_in_winning_region = []
    episode_guarantee_1 = []
    episode_guarantee_2 = []

    is_monitor_wrapped = env.env_is_wrapped(Monitor)[0]

    for episode in range(n_eval_episodes):
        if isinstance(env, DummyVecEnv):
            obs = env.reset()
        else:
            obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        in_winning_region = []
        guarantee_1 = []
        guarantee_2 = []

        while not (done or truncated):
            act = model.predict(obs)
            if isinstance(env, DummyVecEnv):
                obs, reward, done, info = env.step(act[0])
            else:
                obs, reward, done, truncated, info = env.step(act)
            total_reward += reward

            if isinstance(env, DummyVecEnv):
                info = info[0]

            if 'override' in info:
                episode_overrides.append(info['override'])
            if "in_winning_region" in info:
                in_winning_region.append(float(info["in_winning_region"]))
            if "guarantee_1" in info:
                guarantee_1.append(float(info["guarantee_1"]))
            if "guarantee_2" in info:
                guarantee_2.append(float(info["guarantee_2"]))

        if is_monitor_wrapped:
            episode_rewards.append(float(info["episode"]["r"]))
        else:
            episode_rewards.append(total_reward)

        episode_success.append(int(info["is_success"]))
        if "is_success_1" in info:
            episode_success_1.append(int(info["is_success_1"]))
        if "is_success_2" in info:
            episode_success_2.append(int(info["is_success_2"]))
        if "in_winning_region" in info:
            episode_in_winning_region.append(np.mean(in_winning_region))
        if "guarantee_1" in info:
            episode_guarantee_1.append(np.mean(guarantee_1))
        if "guarantee_2" in info:
            episode_guarantee_2.append(np.mean(guarantee_2))

    output = {
        "eval_reward_mean": np.mean(episode_rewards),
        "eval_reward_std": np.std(episode_rewards),
        "eval_success_mean": np.mean(episode_success),
        "eval_override_mean": np.mean(episode_overrides),
        "eval_success_1_mean": np.nanmean(episode_success_1),
        "eval_success_2_mean": np.nanmean(episode_success_2),
        "eval_in_winning_region_mean": np.nanmean(episode_in_winning_region),
        "eval_guarantee_1_mean": np.nanmean(episode_guarantee_1),
        "eval_guarantee_2_mean": np.nanmean(episode_guarantee_2),
    }
        
    return output

def linear_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return initial_value * progress_remaining
    return schedule

def mean_se(vals):
    a = np.array(vals, dtype=np.float32)
    m = np.nanmean(a)
    n = int(np.sum(~np.isnan(a)))
    se = np.nanstd(a, ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, se