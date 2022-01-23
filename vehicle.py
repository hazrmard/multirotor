from typing import List
from dataclasses import dataclass

import numpy as np

@dataclass
class PropellerParams:

    R: float = 0.0762   # propeller length/ disk radius (m) 
    A: float = np.pi * R ** 2
    rho: float = 1.225  #kg/m^3  at MSL
    a: float = 5.7      # Lift curve slope used in example in Stevens & Lewis
    b: float = 2        # number of blades
    c: float = 0.0274   # mean chord length (m)
    eta: float = 1      # propeller efficiency
    
    # Manufacturer propeller length x pitch specification:
    p_diameter: float = 6  #inches
    p_pitch: float = 3   #inches



@dataclass
class VehicleParams:

    propellers: List[PropellerParams]
    angles: np.ndarray
    distances: np.ndarray

    mass: float = 1.
    inertia_matrix: np.matrix = np.eye(3)



@dataclass
class SimulationParams:

    dt: float = 1e-2
    """Timestep of simulation"""
    g: float = 9.81
    """Gravitational acceleration"""
