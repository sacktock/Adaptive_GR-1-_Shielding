import gymnasium as gym
from shield.adaptive_controller_shield import AdaptiveShield

class AdaptiveShieldWrapper(gym.Wrapper):

    def __init__(self, env, path_to_spec, env_keys, sys_abstr, act_abstr):
        super().__init__(env)

        self.path_to_spec = path_to_spec
        self.env_keys = env_keys
        self.sys_abstr = sys_abstr
        self.act_abstr = act_abstr
        # environment variables for GR(1)
        self._env_varibs = None

    def _init_shield(self, state=None):
        self.shield = AdaptiveShield(self.path_to_spec, initiate_spec_repair=True)
        self.shield.initiate_starting_state(state)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # reset env
        obs, info = self.env.reset(seed=seed, options=options)

        # extract the initial environment variables
        self._env_varibs = {k: info[k] for k in self.env_keys if k in info}

        # initialize shield from spec
        self._init_shield(state=self._env_varibs)

        return obs, info

    def step(self, action):
        # extract the system variables from the desired action
        sys_varibs = self.sys_abstr(action)

        #print("Env varibs:", self._env_varibs)
        #print("Sys varibs:", sys_varibs)

        #assert "operational" in self._env_varibs.keys()
        
        # shield before action is comitted
        safe_output = self.shield.get_safe_action(self._env_varibs, sys_varibs)  

        # extract the safe action from the shield output
        safe_action = self.act_abstr(safe_output, sys_varibs, action)

        obs, reward, done, truncated, info = self.env.step(safe_action)

        if safe_action == action:
            info.update({'override': False})
        else:
            info.update({'override': True})

        # extract the new environment variables
        self._env_varibs = {k: info[k] for k in self.env_keys if k in info}

        return obs, reward, done, truncated, info
        
        

        
        
