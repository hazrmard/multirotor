"""
This module defines `gymnasium <https://gymnasium.farama.org>`_ environments
based on the Multirotor class.
"""

from typing import Tuple
import numpy as np
import gymnasium as gym

from .env_kernels import reward_and_flags
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


    def __init__(self, vehicle: Multirotor=None) -> None:
        """
        Parameters
        ----------
        vehicle : Multirotor, optional
            The `Multirotor` vehicle to use as the environment.
        """
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
    @state.setter
    def state(self, x: np.ndarray):
        self.vehicle.state = np.asarray(x, self.vehicle.dtype)


    def reset(
        self, *, seed: int=None, options: dict=None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the vehicle to a random initial position.

        Parameters
        ----------
        seed : int, optional
            The seed of the random number generator `self.np_random`. The
            generator keeps its state across resets if no seed is given.
        options : dict, optional
            ``options['state']`` is a state to set the vehicle to, instead of
            the random one. If it is at the origin, the reward has no advance
            term and the first step reaches the origin.

        Returns
        -------
        Tuple[np.ndarray, dict]
            The state vector of the vehicle, and an empty info dictionary.
        """
        super().reset(seed=seed)
        if self.vehicle is not None:
            self.vehicle.reset()
            position = (self.np_random.random(3) - 0.5) * self.bounding_box * 0.75
            position[(0<position) & (position<self.proximity)] = self.proximity
            position[(-self.proximity<position) & (position<0)] = -self.proximity
            self.vehicle.state[:3] = position
            if options is not None and options.get('state') is not None:
                self.vehicle.state = np.asarray(options['state'], self.vehicle.dtype)
        # needed by reward() to calculate deviation from straight line
        distance = np.linalg.norm(self.state[:3])
        self._des_unit_vec = - self.state[:3] / distance if distance > 0 else np.zeros(3, self.state.dtype)
        return self.state, {}


    def reward(self, state: np.ndarray, action: np.ndarray, nstate: np.ndarray) -> float:
        """
        The reward of a step. Also sets the flags that end the episode:
        ``_reached``, ``_outofbounds``, ``_outoftime`` and ``_tipped``.

        Parameters
        ----------
        state : np.ndarray
            The vehicle's state before the step.
        action : np.ndarray
            The action of the step. It does not affect the reward.
        nstate : np.ndarray
            The vehicle's state after the step.

        Returns
        -------
        float
        """
        reward, self._reached, self._outofbounds, self._outoftime, self._tipped = \
            reward_and_flags(
                state, nstate, self._des_unit_vec, self.vehicle.t,
                self.vehicle.simulation.dt, self.bounding_box, self.proximity,
                self.max_angle, self.period, self.motion_reward_scaling, self.bonus)
        return reward


    def _end_of_step(self, nstate: np.ndarray, reward: float) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        The return value of `step()`, from the flags that `reward()` has set. The
        episode terminates when the vehicle reaches the origin, leaves the
        bounding box or tips over. It is truncated when it runs out of time.
        """
        terminated = self._reached or self._outofbounds or self._tipped
        info = dict(
            reached=self._reached, out_of_bounds=self._outofbounds,
            tipped=self._tipped, out_of_time=self._outoftime
        )
        return nstate, reward, terminated, self._outoftime, info


    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
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
            Disturbing x,y,z forces in the vehicle's local frame, by default 0.
        disturb_torques : np.ndarray, optional
            Disturbing x,y,z torques in the vehicle's local frame, by default 0.

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, dict]
            The state, reward, ``terminated``, ``truncated`` and ``info``. The
            ``info`` dictionary has a boolean for each way the episode can end:
            ``reached``, ``out_of_bounds``, ``tipped`` and ``out_of_time``.
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
        return self._end_of_step(nstate, reward)



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
            Disturbing x,y,z forces in the vehicle's local frame, by default 0.
        disturb_torques : np.ndarray, optional
            Disturbing x,y,z torques in the vehicle's local frame, by default 0.

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool, dict]
            The state, reward, ``terminated``, ``truncated`` and ``info``. The
            ``info`` dictionary has a boolean for each way the episode can end:
            ``reached``, ``out_of_bounds``, ``tipped`` and ``out_of_time``.
        """
        state = self.state
        nstate = self.vehicle.step_speeds(
            u=action,
            disturb_forces=disturb_forces,
            disturb_torques=disturb_torques
        )
        reward = self.reward(state, action, nstate)
        return self._end_of_step(nstate, reward)