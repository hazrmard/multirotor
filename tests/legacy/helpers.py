"""
Frozen copies of ``control_allocation_matrix`` and ``get_vehicle_ability`` from
``multirotor/helpers.py`` at commit f08c6d3, used by the test oracle. Do not edit.
"""
from typing import Tuple

import numpy as np

from multirotor.vehicle import VehicleParams, SimulationParams

from .physics import torque



def control_allocation_matrix(params: VehicleParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the control allocation matrix such that:

        action = matrix @ [thrust, torque_x, torque_y, torque_z]

    Parameters
    ----------
    params : VehicleParams
        The vehicle parameters for which to compute the matrix

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The allocation matrix and its inverse. If no inverse exists, returns
        the Moore-Penrose Pseudo-inverse.
    """
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



def get_vehicle_ability(
    vp: VehicleParams, sp: SimulationParams,
    max_tilt: float=np.pi/12,
    max_rads: float=600
):
    n = len(vp.propellers)

    thrusts = [p.k_thrust * max_rads**2 for p in vp.propellers]
    max_f = sum(thrusts)
    max_acc_z = (max_f - (vp.mass * sp.g)) / vp.mass
    
    # Thrust produced at max_tilt to keep vehicle altitude constant.
    # Use that to calculate lateral component of thrust and lateral acceleration
    tilt_hover_thrust = (vp.mass * sp.g) / (n * np.cos(max_tilt))
    # Thrust is limited by what is possible by the propellers
    tilt_hover_thrust = min(max_f, tilt_hover_thrust)
    max_acc_xy = n * tilt_hover_thrust * np.sin(max_tilt) / vp.mass

    thrust_vec = np.zeros((3, n))
    # excessive thrust (above weight) that can contribute to torque
    thrust_vec[2] = np.asarray(thrusts) - (vp.mass * sp.g / n)
    k_drag_vec = np.asarray([p.k_drag for p in vp.propellers])
    inertia_vec = np.asarray([p.moment_of_inertia for p in vp.propellers])
    torques = torque(
        position_vector=vp.propeller_vectors,
        force=thrust_vec,
        clockwise=np.asarray(vp.clockwise).astype(float),
        drag_coefficient=k_drag_vec,
        moment_of_inertia=inertia_vec,
        prop_angular_acceleration=0,
        prop_angular_velocity=max_rads
    )
    torques = (torques * (torques > 0)).sum(axis=1)
    I = vp.inertia_matrix.diagonal()
    # t = i a
    ang_acc = torques / I

    # TODO: max angular velocity such that can accelerate to and decelerate from it
    # to 0

    res = dict(
        max_acc_xy=max_acc_xy,
        max_acc_z=max_acc_z,
        max_ang_acc=max(ang_acc)
    )
    return res
