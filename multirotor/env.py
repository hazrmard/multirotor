"""
This module defines OpenAI Gym compatible classes based on the Multirotor class.
"""

from typing import Tuple, List, Union
import numpy as np
import gym

from .simulation import Multirotor
from .helpers import find_nominal_speed


class BaseMultirotorEnv(gym.Env):
    """
    The base environment class, defining the episode, and reward function.
    """

    max_angle = np.pi/12
    """The max tilt angle in radians."""
    proximity = 0.5
    """Distance from the waypoint at which to consider it has been reached."""
    period = 10
    """Maximum duration of the episode (seconds)."""
    bounding_box = 20
    """Size of the cube in which the vehicle can fly, centered at origin."""
    motion_reward_scaling = bounding_box / 2
    bonus = bounding_box * 20


    def __init__(self, vehicle: Multirotor=None, seed: int=None) -> None:
        # pos, vel, att, ang vel
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            shape=(12,)
        )
        self.vehicle = vehicle
        self.seed(seed=seed, _seed_with_none=True)


    def seed(self, seed: int=None, _seed_with_none: bool=False) -> List[Union[int,tuple]]:
        if isinstance(seed, (int, float)):
            self.random = np.random.RandomState(seed)
            self._seeds = (seed,)
        elif seed is None and _seed_with_none:
            self.random = np.random.RandomState()
            self._seeds = (self.random.get_state())
        elif isinstance(seed, tuple):
            self.random = np.random.RandomState()
            self.random.set_state(seed)
            self._seeds = (self.random.get_state(),)
        return list(self._seeds)


    @property
    def state(self) -> np.ndarray:
        return self.vehicle.state
    @state.setter
    def state(self, x: np.ndarray):
        self.vehicle.state = np.asarray(x, self.vehicle.dtype)


    def reset(self, x: np.ndarray=None) -> np.ndarray:
        """
        Reset the vehicle to a random initial position.

        Parameters
        ----------
        x : np.ndarray, optional
            A state to set the vehicle to, by default None

        Returns
        -------
        np.ndarray
            The state vector of the vehicle.
        """
        if self.vehicle is not None:
            self.vehicle.reset()
            position = (self.random.rand(3) - 0.5) * self.bounding_box * 0.75
            position[(0<position) & (position<self.proximity)] = self.proximity
            position[(-self.proximity<position) & (position<0)] = -self.proximity
            self.vehicle.state[:3] = position
            if x is not None:
                self.vehicle.state = np.asarray(x, self.vehicle.dtype)
        # needed by reward() to calculate deviation from straight line
        self._des_unit_vec = - self.state[:3] / np.linalg.norm(self.state[:3])
        return self.state


    def reward(self, state: np.ndarray, action: np.ndarray, nstate: np.ndarray) -> float:
        dist = np.linalg.norm(nstate[:3])
        self._reached = dist <= self.proximity
        self._outofbounds = np.any(np.abs(state[:3]) > self.bounding_box / 2)
        self._outoftime = self.vehicle.t >= self.period
        self._tipped = np.any(np.abs(state[6:9]) > self.max_angle)
        self._done = self._outoftime or self._outofbounds or self._reached or self._tipped
        delta_pos = (nstate[:3] - state[:3])
        advance = np.linalg.norm(delta_pos)
        cross = np.linalg.norm(np.cross(delta_pos, self._des_unit_vec))
        delta_turn = np.abs(nstate[8]) - np.abs(state[8])
        reward = ((advance - cross - delta_turn) * self.motion_reward_scaling) - self.vehicle.simulation.dt
        if self._reached:
            reward += self.bonus
        elif self._tipped or self._outofbounds:
            reward -= self.bonus
        elif self._outoftime:
            reward -= (dist / self.bounding_box) * self.bonus
        return reward


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError



class DynamicsMultirotorEnv(BaseMultirotorEnv):


    def __init__(
        self, vehicle: Multirotor=None, allocate: bool=False, max_rads: float=np.inf
    ) -> None:
        """
        Parameters
        ----------
        vehicle : Multirotor
            The `Multirotor` vehicle to use as the environment
        max_rads: float, optional
            The maximum allocated speed of propellers, if `allocate==True`.
        allocate: bool, optional
            Whether the actions are the requested dynamics, in which case control
            allocation will be used to calculate the best possible propeller speeds.
            Otherwise, the forces and torques are directly applied to the system.
            By default False
        """
        super().__init__(vehicle=vehicle)
        
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            shape=(6,)  # 3 forces, 3 torques
        )
        self.allocate = allocate
        self.max_rads = max_rads


    def step(
        self, action: np.ndarray, disturb_forces: np.ndarray=0.,
        disturb_torques: np.ndarray=0.
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step environment by providing dynamics acting in local frame.

        Parameters
        ----------
        action : np.ndarray
            An array of x,y,z forces and x,y,z torques in local frame.
        disturb_forces : np.ndarray, optional
            Disturbinng x,y,z forces in the vehicle's local frame, by default 0.
        disturb_torques : np.ndarray, optional
            Disturbing x,y,z torques in the vehicle's local frame, by default 0.

        Returns
        -------
        Tuple[np.ndarray, float, bool, dict]
            The state and other environment variables.
        """
        if self.allocate:
            speeds = self.vehicle.allocate_control(action[2], action[3:6])
            speeds = np.clip(speeds, a_min=0, a_max=self.max_rads)
            forces, torques = self.vehicle.get_forces_torques(speeds, self.vehicle.state)
            action = np.concatenate((forces, torques))
        action[:3] += disturb_forces
        action[3:] += disturb_torques
        state = self.state
        nstate = self.vehicle.step_dynamics(u=action)
        reward = self.reward(state, action, nstate)
        return nstate, reward, self._done, {}



class SpeedsMultirotorEnv(BaseMultirotorEnv):
    """
    A multirotor environment that uses speed signals as action inputs. The speed
    signals can be one of two kinds:

      1. Actual speeds (rad/s).

        a. If the multirotor's propellers have `Motor` instances, then the 
        `MotorParams.speed_voltage_scaling` parameter should be provided. It 
        converts speed to voltage signal used for speed calculations. The 
        `helpers.learn_speed_voltage_scaling` function can be used for this.

        b. Else, if the propellers do not have a motor, then the parameter
        need not be provided.

      2. Voltage signals (V). In this case, propellers should have a `Motor` instance
      with the `speed_voltage_scaling` parameter equal to 1 (no need to learn it,
      since voltage is already being given).
    """


    def __init__(self, vehicle: Multirotor) -> None:
        """

        Parameters
        ----------
        vehicle : Multirotor
            The `Multirotor` vehicle to use as the environment
        """
        super().__init__(vehicle=vehicle)
        
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
            shape=(len(vehicle.propellers),)  # action for each propeller
        )


    def step(
        self, action: np.ndarray, disturb_forces: np.ndarray=0.,
        disturb_torques: np.ndarray=0.
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Step environment by providing speed signal.

        Parameters
        ----------
        action : np.ndarray
            An array of speed signals.
        disturb_forces : np.ndarray, optional
            Disturbinng x,y,z forces in the velicle's local frame, by default 0.
        disturb_torques : np.ndarray, optional
            Disturbing x,y,z torques in the vehicle's local frame, by default 0.

        Returns
        -------
        Tuple[np.ndarray, float, bool, dict]
            The state and other environment variables.
        """
        state = self.state
        nstate = self.vehicle.step_speeds(
            u=action,
            disturb_forces=disturb_forces,
            disturb_torques=disturb_torques
        )
        reward = self.reward(state, action, nstate)
        return nstate, reward, self._done, {}