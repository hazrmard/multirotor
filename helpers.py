from typing import Iterable

import numpy as np

from vehicle import PropellerParams, VehicleParams
from simulation import Propeller



def moment_of_inertia_tensor_from_cooords(point_masses: Iterable[float], coords: Iterable[np.ndarray]) -> np.matrix:
    coords = np.asarray(coords)
    masses = np.asarray(point_masses)
    x,y,z = coords[:,0], coords[:,1], coords[:2]
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = Iyx = -np.sum(x * y * masses)
    Iyz = Izy = -np.sum(y * z * masses)
    Ixz = Izx = -np.sum(x * z * masses)
    return np.asmatrix([
        [Ixx, Ixy, Ixz],
        [Iyx, Iyy, Iyz],
        [Izx, Izy, Izz]
    ])



def vehicle_params_factory(n: int, m_prop: float, d_prop: float, params: PropellerParams,
    m_body: float, body_shape: str='point', body_size: float=0.1):
    angles = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
    masses = [m_prop] * n
    x = np.cos(angles) * d_prop
    y = np.sin(angles) * d_prop
    z = np.zeros_like(y)
    coords = np.vstack((x, y, z)).T
    I_prop = moment_of_inertia_tensor_from_cooords(masses, coords)

    if body_shape == 'point':
        I_body = np.asmatrix(np.zeros((3,3)))
    elif body_shape == 'sphere_solid':
        I_body = np.asmatrix(np.eye(3) * 0.4 * m_body * body_size**2)
    elif body_shape == 'sphere_shell':
        I_body = np.asmatrix(np.eye(3) * 2 * m_body * body_size**2 / 3)
    elif body_shape == 'cube':
        I_body = np.asmatrix(np.eye(3) * 2 * m_body * body_size**2 / 12)

    I = I_body + I_prop

    return VehicleParams(
        propellers = [Propeller(params) for _ in range(n)],
        angles = angles,
        distances=np.ones(n) * d_prop,
        mass = n * m_prop + m_body,
        inertia_matrix = I
    )
