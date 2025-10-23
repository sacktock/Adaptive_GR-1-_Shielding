import gymnasium as gym

class ShieldWrapper(gym.Wrapper):

    def __init__(self, env, shield=None):
        super().__init__(env)
        self.shield = shield

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.shield is None:
            return self.env.reset(seed=seed, options=options)

        self.shield.reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self.shield.step(info)
        return obs, info

    def step(self, action):
        if self.shield is None:
            return self.env.step(action)
        # shield action before comitting it in the environment
        safe_action, override = self.shield(action)
        obs, reward, terminated, truncated, info = self.env.step(safe_action)
        self.shield.step(info)
        info.update({'override': override})
        return obs, reward, terminated, truncated, info