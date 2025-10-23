import numpy as np
import math
import gymnasium as gym
from gymnasium.utils import seeding

class Static_Shield_Star():

    """
    Static shield prioritizing both guarantee_1 and guarantee_2
        If neither can be satisified then let the agent do what they want 
    """
    def __init__(self):
        self.predicates = ["highwater", "methane", "pump"]
        self.highwater = None
        self.methane = None
        self.prev_highwater = None
        self.prev_methane = None

    def __call__(self, action):

        if self.prev_highwater and self.prev_methane:
            return action, False

        if self.prev_highwater:
            return 1, True

        if self.prev_methane:
            return 0, True

        return action, False

    def seed(self, seed=None):
        pass

    def reset(self, seed=seed):
        self.seed(seed)
        self.highwater = False
        self.methane = False

    def step(self, info):
        self.prev_highwater = self.highwater
        self.prev_methane = self.methane
        abstr = {k: info[k] for k in self.predicates}
        self.highwater = True if abstr['highwater'] == "true" else False
        self.methane = True if abstr['methane'] == "true" else False

    def repair(self):
        raise NotImplementedError


class Static_Shield_1():

    """
    Static shield prioritizing only guarantee_1
    """

    def __init__(self):
        self.predicates = ["highwater", "methane", "pump"]
        self.highwater = None
        self.methane = None
        self.prev_highwater = None
        self.prev_methane = None

    def __call__(self, action):
        if self.prev_highwater:
            return 1, True

        return action, False

    def seed(self, seed=None):
        pass

    def reset(self, seed=seed):
        self.seed(seed)
        self.highwater = False
        self.methane = False

    def step(self, info):
        self.prev_highwater = self.highwater
        self.prev_methane = self.methane
        abstr = {k: info[k] for k in self.predicates}
        self.highwater = True if abstr['highwater'] == "true" else False
        self.methane = True if abstr['methane'] == "true" else False

    def repair(self):
        raise NotImplementedError
        
class Static_Shield_2():

    """
    Static shield prioritizing only guarantee_2
    """

    def __init__(self):
        self.predicates = ["highwater", "methane", "pump"]
        self.highwater = None
        self.methane = None
        self.prev_highwater = None
        self.prev_methane = None

    def __call__(self, action):
        if self.prev_methane:
            return 0, True

        return action, False

    def seed(self, seed=None):
        pass

    def reset(self, seed=seed):
        self.seed(seed)
        self.highwater = False
        self.methane = False

    def step(self, info):
        self.prev_highwater = self.highwater
        self.prev_methane = self.methane
        abstr = {k: info[k] for k in self.predicates}
        self.highwater = True if abstr['highwater'] == "true" else False
        self.methane = True if abstr['methane'] == "true" else False

    def repair(self):
        raise NotImplementedError