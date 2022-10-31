"""
This module defines OpenAI Gym compatible classes based on the Multirotor class.
"""

from typing import Tuple
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
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
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
        Tuple[np.ndarray, None, None, None]
            The state and other environment variables.
        """
        if self.allocate:
            speeds = self.vehicle.allocate_control(action[2], action[3:6])
            speeds = np.clip(speeds, a_min=0, a_max=self.max_rads)
            forces, torques = self.vehicle.get_forces_torques(speeds, self.vehicle.state)
            action = np.concatenate((forces, torques))
        action[:3] += disturb_forces
        action[3:] += disturb_torques
        self.vehicle.step_dynamics(u=action)
        return self.state, 0., False, False, {}



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
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
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
        Tuple[np.ndarray, None, None, None]
            The state and other environment variables.
        """
        self.vehicle.step_speeds(
            u=action,
            disturb_forces=disturb_forces,
            disturb_torques=disturb_torques
        )
        return self.state, 0., False, False, {}