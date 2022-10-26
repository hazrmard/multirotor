"""
This module defines OpenAI Gym compatible classes based on the Multirotor class.
"""

import numpy as np
import gym

from .simulation import Multirotor
from .helpers import find_nominal_speed


class BaseMultirotorEnv(gym.Env):


    def __init__(self, vehicle: Multirotor=None) -> None:
        # pos, vel, att, ang vel
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            shape=(12,)
        )
        self.vehicle = vehicle


    @property
    def state(self) -> np.ndarray:
        return self.vehicle.state


    def reset(self):
        if self.vehicle is not None:
            self.vehicle.reset()
        return self.state


    def reward(self, state, action, nstate):
        raise NotImplementedError


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError



class DynamicsMultirotorEnv(BaseMultirotorEnv):


    def __init__(self, vehicle: Multirotor=None, allocate: bool=False, max_rads=None) -> None:
        super().__init__(vehicle=vehicle)
        
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            shape=(6,)  # 3 forces, 3 torques
        )
        self.allocate = allocate
        self.max_rads = max_rads


    def step(self, action: np.ndarray):
        if self.allocate:
            speeds = self.vehicle.allocate_control(action[2], action[3:6])
            speeds = np.clip(speeds, a_min=0, a_max=self.max_rads)
            forces, torques = self.vehicle.get_forces_torques(speeds, self.vehicle.state)
            action = np.concatenate((forces, torques))
        self.vehicle.step_dynamics(u=action)
        return self.state, None, None, None



class SpeedsMultirotorEnv(BaseMultirotorEnv):


    def __init__(self, vehicle: Multirotor=None, max_rads=None) -> None:
        super().__init__(vehicle=vehicle)
        
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            shape=(len(vehicle.propellers),)  # action for each propeller
        )
        self.max_rads = max_rads


    def step(self, action: np.ndarray):
        action = np.clip(action, a_min=0, a_max=self.max_rads)
        self.vehicle.step_speeds(u=action)
        return self.state, None, None, None