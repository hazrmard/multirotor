from typing import Callable, Iterable, Tuple

import numpy as np
from scipy.optimize import fsolve

from .vehicle import PropellerParams, VehicleParams



def moment_of_inertia_tensor_from_cooords(
    point_masses: Iterable[float], coords: Iterable[np.ndarray]
) -> np.ndarray:
    """
    Calculate the inertial matrix given a distribution of point masses.

    Parameters
    ----------
    point_masses : Iterable[float]
        A list of masses.
    coords : Iterable[np.ndarray]
        The corresponding coordinates of those masses about the center of rotation.
        Ideally, this would be the center of mass of the object.

    Returns
    -------
    np.ndarray
        The 3x3 inertial matrix.
    """
    # TODO: Conditionally calculate the center of mass and transform coordinates
    # about it if a boolean option is provided.
    coords = np.asarray(coords)
    masses = np.asarray(point_masses)
    x,y,z = coords[:,0], coords[:,1], coords[:,2]
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = Iyx = -np.sum(x * y * masses)
    Iyz = Izy = -np.sum(y * z * masses)
    Ixz = Izx = -np.sum(x * z * masses)
    return np.asarray([
        [Ixx, Ixy, Ixz],
        [Iyx, Iyy, Iyz],
        [Izx, Izy, Izz]
    ])



def vehicle_params_factory(
    n: int, m_prop: float, d_prop: float, params: PropellerParams,
    m_body: float, body_shape: str='sphere_solid', body_size: float=0.1
) -> VehicleParams:
    """
    Create a simple multirotor vehicle parameters object. The multirotor has
    evenly spaced propellers and a simple core shape (shell, cube etc.)

    Parameters
    ----------
    n : int
        The number of arms/propellers.
    m_prop : float
        The mass of each propeller.
    d_prop : float
        The distance of each propeller from the center of the multirotor.
    params : PropellerParams
        The parameters describing a propeller.
    m_body : float
        The mass of the central body.
    body_shape : str, optional
        The shape of the core, by default 'sphere_solid'
    body_size : float, optional
        The dimension of the core (m), by default 0.1

    Returns
    -------
    VehicleParams
        The parameters object.
    """
    angle_spacing = 2 * np.pi / n
    angles = np.arange(angle_spacing / 2, 2 * np.pi, angle_spacing)
    masses = [m_prop] * n
    x = np.cos(angles) * d_prop
    y = np.sin(angles) * d_prop
    z = np.zeros_like(y)
    coords = np.vstack((x, y, z)).T
    I_prop = moment_of_inertia_tensor_from_cooords(masses, coords)

    if body_shape == 'point':
        I_body = np.asarray(np.zeros((3,3)))
    elif body_shape == 'sphere_solid':
        I_body = np.asarray(np.eye(3) * 0.4 * m_body * body_size**2)
    elif body_shape == 'sphere_shell':
        I_body = np.asarray(np.eye(3) * 2 * m_body * body_size**2 / 3)
    elif body_shape == 'cube':
        I_body = np.asarray(np.eye(3) * 2 * m_body * body_size**2 / 12)

    I = I_body + I_prop

    return VehicleParams(
        propellers = [params for _ in range(n)],
        angles = angles,
        distances=np.ones(n) * d_prop,
        mass = n * m_prop + m_body,
        inertia_matrix = I,
    )



def find_nominal_speed(thrust_fn: Callable[[float], float], weight: float) -> float:
    """
    Calculate the speed a propeller must spin to balance the weight.

    Parameters
    ----------
    thrust_fn : Callable[[float], float]
        A function taking the speed as input and outputting thrust (N).
    weight : float
        The weight to balance.

    Returns
    -------
    float
        The speed to balance the weight.
    """
    def balance(speed: float) -> float:
        thrust = thrust_fn(speed)
        residual = thrust - weight
        return residual
    return fsolve(balance, 1e3)[0]



def learn_thrust_coefficient(
    thrust_fn: Callable[[float], float], domain: Tuple=(1, 10000)
) -> float:
    """
    Assuming a quadratic relationship between thrust and propeller speed,
    estimate the coefficient of proportionality k_thrust, where

        thrust = k_thrust . speed^2

    Parameters
    ----------
    thrust_fn : Callable[[float], float]
        The function accepting speed and returning thrust.
    domain : Tuple, optional
        The range of speeds to try, by default (1, 10000)

    Returns
    -------
    float
        The thrust coefficient.
    """
    speeds = np.linspace(domain[0], domain[1], num=250)
    thrust = np.zeros_like(speeds)
    for i, speed in enumerate(speeds):
        thrust[i] = thrust_fn(speed)
    return np.polyfit(speeds, thrust, deg=2)[0] # return coefficient of quadradic term



def learn_speed_voltage_scaling(
    speed_fn: Callable[[float], float], domain: Tuple=(0,20)
) -> float:
    """
    Assuming a linear relationship between voltage and motor speed, learn
    the scaling coefficient, k_scaling, where:

        

    Parameters
    ----------
    speed_fn : Callable[[float], float]
        A function accepting voltage and returning speed.
    domain : Tuple, optional
        The range of voltages to try to learn the coefficient, by default (0,20)

    Returns
    -------
    float
        The scaling coefficient.
    """
    signals = np.linspace(domain[0], domain[1], num=10)
    speeds = np.zeros_like(signals)
    for i, signal in enumerate(signals):
        speeds[i] = speed_fn(signal)
    return np.polyfit(signals, speeds, 1)[0]



def moment_of_inertia_disk(m: float, r: float) -> float:
    return 0.5 * m * r**2



def control_allocation_matrix(params: VehicleParams) -> Tuple[np.ndarray, np.ndarray]:
    alloc = np.zeros((4, len(params.propellers))) #[Fz, Mx, My, Mz] x n-Propellers
    x = params.distances * np.cos(params.angles)
    y = params.distances * np.sin(params.angles)
    for i, p in enumerate(params.propellers):
        alloc[0, i] = p.k_thrust                       # vertical force
        alloc[1, i] = p.k_thrust * y[i]                 # torque about x-axis
        alloc[2, i] = p.k_thrust * (-x[i])              # torque about y-axis
        alloc[3, i] = p.k_drag * params.clockwise[i]    # torque about z-axis
    alloc_inverse = np.linalg.pinv(alloc)
    return alloc, alloc_inverse