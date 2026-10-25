"""
Frozen copy of ``control_allocation_matrix`` from ``multirotor/helpers.py`` at commit f08c6d3,
used by the test oracle. Do not edit.
"""
from typing import Tuple

import numpy as np

from multirotor.vehicle import VehicleParams



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
