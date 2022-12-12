import numpy as np
from numpy import sin, cos, tan
from numba import njit



@njit
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
        The transformed coordinates in the intertial frame
    """
    return dcm.T @ vector



@njit
def inertial_to_body(vector: np.ndarray, dcm=np.ndarray) -> np.ndarray:
    return dcm @ vector



@njit
def direction_cosine_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = cos(roll)  # phi
    sr = sin(roll)
    cp = cos(pitch) # theta
    sp = sin(pitch)
    cy = cos(yaw)   # psi
    sy = sin(yaw)
    dcm = np.asarray([
        [cp * cy,                   cp * sy,                    -sp],
        [-cr * sy + sr * sp * cy,   cr * cy + sr * sp * sy,     sr * cp],
        [sr * sy + cr * sp * cy,    -sr * cy + cr * sp * sy,    cr * cp]
    ])
    return dcm



@njit
def rotating_frame_derivative(
    value: np.ndarray, local_derivative: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    # d (value . vector) / dt
    # = vector . d value / dt + value . d vector / dt
    #  ( local derivative )  ( coriolis term )
    dv = np.copy(value)
    dv_l = local_derivative
    dv[0] = dv_l[0] - omega[2] * value[2]
    dv[1] = dv_l[1] + omega[2] * value[0]
    dv[2] = dv_l[2] - omega[1] * value[0] + omega[0] * value[2]
    return dv



@njit
def angular_to_euler_rate(
    angular_velocity: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    roll, pitch, yaw = orientation
    p, q, r = angular_velocity
    roll_rate = p + tan(pitch) * (q * sin(roll) + r * cos(roll))
    pitch_rate = q * cos(roll) - r * sin(roll)
    yaw_rate = (q * sin(roll) + r * cos(roll)) / cos(pitch)
    return np.asarray([roll_rate, pitch_rate, yaw_rate])



@njit
def euler_to_angular_rate(
    euler_velocity: np.ndarray, orientation: np.ndarray
) -> np.ndarray:
    roll_rate, pitch_rate, yaw_rate = euler_velocity
    roll, pitch, yaw = orientation
    cr, cp = cos(roll), cos(pitch)
    sr, sp = sin(roll), sin(pitch)
    p = roll_rate - yaw_rate * sp
    q = pitch_rate * cr + yaw_rate * cp * sr
    r = yaw_rate * cr * cp - pitch_rate * sr
    return np.asarray([p, q, r])