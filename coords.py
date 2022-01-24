import numpy as np
from numpy import sin, cos, tan


def body_to_inertial(coords: np.ndarray, *, rotations=None, dcm=None, dcm_inverse=None) -> np.ndarray:
    if rotations is not None:
        dcm = direction_cosine_matrix(yaw=rotations[0], pitch=rotations[1], roll=rotations[2])
    if dcm is not None:
        dcm_inverse = dcm.T
    if dcm_inverse is not None:
        return dcm_inverse @ coords
    else:
        raise ValueError('Specify one of yaw, pitch, roll angles, direction cosine matrix, or its inverse.')



def inertial_to_body(coords: np.ndarray, *, rotations=None, dcm=None) -> np.ndarray:
    if rotations is not None:
        dcm = direction_cosine_matrix(yaw=rotations[0], pitch=rotations[1], roll=rotations[2])
    if dcm is not None:
        return dcm @ coords
    else:
        raise ValueError('Specify one of yaw, pitch, roll angles or direction cosine matrix.')



def direction_cosine_matrix(yaw: float, pitch: float, roll: float) -> np.matrix:
    cy = cos(yaw)
    sy = sin(yaw)
    cp = cos(pitch)
    sp = sin(pitch)
    cr = cos(roll)
    sr = sin(roll)
    dcm = np.asarray([
        [cp * cy,                   cp * sy,                    -sp],
        [-cr * sy + sr * sp * cy,   cr * cy + sr * sp * sy,     sr * cp],
        [sr * sy + cr * sp * cy,    -sr * cy + cr * sp * sy,    cr * cp]
    ])
    return np.asmatrix(dcm)



def rotating_frame_derivative(value: np.ndarray, local_derivative: np.ndarray, omega: np.ndarray) -> np.ndarray:
    # d (value . vector) / dt
    # = vector . d value / dt + value . d vector / dt
    #  ( local derivative )  ( coriolis term )
    dv = np.copy(value)
    dv_l = local_derivative
    dv[0] = dv_l[0] - omega[2] * value[2]
    dv[1] = dv_l[1] + omega[2] * value[0]
    dv[2] = dv_l[2] - omega[1] * value[0] + omega[0] * value[2]
    return dv



def angular_to_euler_rate(angular_velocity: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = orientation
    p, q, r = angular_velocity
    roll_rate = p + q * tan(pitch) * (q * sin(roll) + r * cos(roll))
    pitch_rate = q * cos(roll) - r * sin(roll)
    yaw_rate = (q * sin(roll) + r * cos(roll)) / cos(pitch)
    return np.asarray([roll_rate, pitch_rate, yaw_rate])



def euler_to_angular_rate(euler_velocity: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    roll_rate, pitch_rate, yaw_rate = euler_velocity
    roll, pitch, yaw = orientation
    p = roll_rate - yaw_rate * sin(pitch)
    q = pitch_rate * cos(roll) + yaw_rate * cos(pitch) * sin(roll)
    r = yaw_rate * cos(roll) * cos(pitch) - pitch_rate * sin(roll)
    return np.asarray([p, q, r])