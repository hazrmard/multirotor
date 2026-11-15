"""
Transforms between the vehicle's coordinate frames and rates.

The functions are written once, against the array namespace ``xp``, and built for
numba here (see `multirotor.lazy`). `multirotor.controller.cascade` builds them again
for JAX, so that the controller can be differentiated. Orientations are the
roll, pitch and yaw angles (rad) of the vehicle, in that order.
"""
import numpy as np
from numba import njit

from . import xp_numba as xp
from .lazy import build



def body_to_inertial(vector: np.ndarray, dcm: np.ndarray) -> np.ndarray:
    """
    View body coordinates in terms of coordinates on inertial axes. Assumes
    both frames share same origin.

    Parameters
    ----------
    vector : np.ndarray
        The 3D vector of points, or a 3 x N matrix of N points
    dcm : np.ndarray
        The 3x3 direction cosine matrix

    Returns
    -------
    np.ndarray
        The transformed coordinates in the inertial frame
    """
    return dcm.T @ vector



def inertial_to_body(vector: np.ndarray, dcm: np.ndarray) -> np.ndarray:
    """
    The inverse of `body_to_inertial`.

    Parameters
    ----------
    vector : np.ndarray
        The 3D vector in the inertial frame, or a 3 x N matrix of N vectors.
    dcm : np.ndarray
        The 3x3 direction cosine matrix.

    Returns
    -------
    np.ndarray
        The vector in body coordinates.
    """
    return dcm @ vector



def direction_cosine_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    The matrix that rotates inertial axes into the body axes of an orientation.

    Parameters
    ----------
    roll, pitch, yaw : float
        The orientation (rad).

    Returns
    -------
    np.ndarray
        The direction cosine matrix, shape (3, 3).
    """
    cr, sr = xp.cos(roll), xp.sin(roll)     # phi
    cp, sp = xp.cos(pitch), xp.sin(pitch)   # theta
    cy, sy = xp.cos(yaw), xp.sin(yaw)       # psi
    return xp.array((
        (cp * cy,                  cp * sy,                  -sp),
        (-cr * sy + sr * sp * cy,  cr * cy + sr * sp * sy,   sr * cp),
        (sr * sy + cr * sp * cy,   -sr * cy + cr * sp * sy,  cr * cp)))



def angular_to_euler_rate(angular_velocity: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """
    Body angular rates (p, q, r) as rates of change of roll, pitch and yaw.

    Parameters
    ----------
    angular_velocity : np.ndarray
        The body angular rates (rad/s), shape (3,).
    orientation : np.ndarray
        The roll, pitch and yaw (rad), shape (3,).

    Returns
    -------
    np.ndarray
        The rates of change of roll, pitch and yaw (rad/s), shape (3,).
    """
    roll, pitch = orientation[0], orientation[1]
    p, q, r = angular_velocity[0], angular_velocity[1], angular_velocity[2]
    return xp.array((
        p + xp.tan(pitch) * (q * xp.sin(roll) + r * xp.cos(roll)),
        q * xp.cos(roll) - r * xp.sin(roll),
        (q * xp.sin(roll) + r * xp.cos(roll)) / xp.cos(pitch)))



def euler_to_angular_rate(euler_velocity: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """
    The inverse of `angular_to_euler_rate`.

    Parameters
    ----------
    euler_velocity : np.ndarray
        The rates of change of roll, pitch and yaw (rad/s), shape (3,).
    orientation : np.ndarray
        The roll, pitch and yaw (rad), shape (3,).

    Returns
    -------
    np.ndarray
        The body angular rates (rad/s), shape (3,).
    """
    roll, pitch = orientation[0], orientation[1]
    cr, cp = xp.cos(roll), xp.cos(pitch)
    sr, sp = xp.sin(roll), xp.sin(pitch)
    roll_rate, pitch_rate, yaw_rate = euler_velocity[0], euler_velocity[1], euler_velocity[2]
    return xp.array((
        roll_rate - yaw_rate * sp,
        pitch_rate * cr + yaw_rate * cp * sr,
        yaw_rate * cr * cp - pitch_rate * sr))



NAMES = (
    'body_to_inertial', 'inertial_to_body', 'direction_cosine_matrix',
    'angular_to_euler_rate', 'euler_to_angular_rate')
SOURCE = {name: globals()[name] for name in NAMES}
globals().update(vars(build(SOURCE, NAMES, 'numba', globals())))



@njit
def rotating_frame_derivative(
    value: np.ndarray, local_derivative: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    """
    The rate of change of a vector in a rotating frame, as seen from a fixed one. Not built
    for JAX: only the blade-element thrust model uses it.

    Parameters
    ----------
    value : np.ndarray
        The vector in the rotating frame, shape (3,).
    local_derivative : np.ndarray
        Its rate of change in the rotating frame, shape (3,).
    omega : np.ndarray
        The angular velocity of the rotating frame, shape (3,).

    Returns
    -------
    np.ndarray
        The rate of change in the fixed frame, shape (3,).
    """
    # d (value . vector) / dt
    # = vector . d value / dt + value . d vector / dt
    #  ( local derivative )  ( coriolis term )
    dv = np.copy(value)
    dv_l = local_derivative
    dv[0] = dv_l[0] - omega[2] * value[2]
    dv[1] = dv_l[1] + omega[2] * value[0]
    dv[2] = dv_l[2] - omega[1] * value[0] + omega[0] * value[2]
    return dv
