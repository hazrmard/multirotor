import numpy as np
from numpy import sin, cos


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



def direction_cosine_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
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
    return dcm