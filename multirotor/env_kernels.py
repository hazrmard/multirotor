"""
Compiled kernels of the environments in `multirotor.env`. The environment classes
call them; they hold no state of their own, as the kernels of `multirotor.physics`
and `multirotor.controller.kernels` hold none.

``reward_and_flags`` is the only kernel. `BaseMultirotorEnv.reward` calls it
once per step.
"""
from typing import Tuple

import numpy as np
from numba import njit



@njit
def reward_and_flags(
    state: np.ndarray, nstate: np.ndarray, des_unit_vec: np.ndarray,
    t: float, dt: float, bounding_box: float, proximity: float, max_angle: float,
    period: float, motion_reward_scaling: float, bonus: float
) -> Tuple[float, bool, bool, bool, bool]:
    """
    The reward of one step, and the conditions that end the episode.

    The reward rewards advance along the straight line to the origin, and
    penalizes drift off that line, yaw, and the passing of time. It adds
    ``bonus`` on reaching the origin, subtracts ``bonus`` on tipping or leaving
    the bounding box, and subtracts a share of ``bonus`` that grows with the
    distance left when the time runs out.

    Parameters
    ----------
    state : np.ndarray
        The vehicle's 12 body states before the step. Position is ``[0:3]`` and
        orientation ``[6:9]``.
    nstate : np.ndarray
        The vehicle's 12 body states after the step.
    des_unit_vec : np.ndarray
        Unit vector from the start position to the origin, shape (3,).
    t : float
        The vehicle's time (s) after the step.
    dt : float
        The time step (s).
    bounding_box : float
        Side of the cube centered at the origin in which the vehicle must stay (m).
    proximity : float
        Distance from the origin at which the vehicle has reached it (m).
    max_angle : float
        Roll or pitch angle above which the vehicle has tipped (rad).
    period : float
        The maximum duration of the episode (s).
    motion_reward_scaling : float
        Scale of the advance, drift and yaw terms.
    bonus : float
        The size of the terminal reward or penalty.

    Returns
    -------
    Tuple[float, bool, bool, bool, bool]
        The reward, then whether the vehicle reached the origin, left the
        bounding box (judged on the state before the step), ran out of time, and
        tipped (judged on the state before the step).
    """
    dist = np.sqrt(nstate[0]**2 + nstate[1]**2 + nstate[2]**2)
    reached = dist <= proximity
    outofbounds = False
    tipped = False
    for i in range(3):
        outofbounds = outofbounds or abs(state[i]) > bounding_box / 2
        tipped = tipped or abs(state[6 + i]) > max_angle
    outoftime = t >= period
    dx, dy, dz = nstate[0] - state[0], nstate[1] - state[1], nstate[2] - state[2]
    advance = np.sqrt(dx**2 + dy**2 + dz**2)
    # Length of the cross product of the displacement and the desired direction
    cx = dy * des_unit_vec[2] - dz * des_unit_vec[1]
    cy = dz * des_unit_vec[0] - dx * des_unit_vec[2]
    cz = dx * des_unit_vec[1] - dy * des_unit_vec[0]
    drift = np.sqrt(cx**2 + cy**2 + cz**2)
    delta_turn = abs(nstate[8]) - abs(state[8])
    reward = (advance - drift - delta_turn) * motion_reward_scaling - dt
    if reached:
        reward += bonus
    elif tipped or outofbounds:
        reward -= bonus
    elif outoftime:
        reward -= (dist / bounding_box) * bonus
    return reward, reached, outofbounds, outoftime, tipped
