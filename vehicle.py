from typing import Iterable, List, Union
from dataclasses import dataclass

import numpy as np

@dataclass
class PropellerParams:
    
    # Manufacturer propeller length x pitch specification:
    diameter: float = 6  #inches
    pitch: float = 3   #inches

    a: float = 5.7
    "Lift curve slope used in example in Stevens & Lewis"
    b: float = 2
    "Number of blades"
    c: float = 0.0274
    "Mean chord length (m)"
    eta: float = 1.
    "Propeller efficiency"

    def __post_init__(self):
        self.R = self.diameter * 0.0254
        self.A = np.pi * self.R**2
        self.theta0 = 2*np.arctan2(self.pitch, (2 * np.pi * 3/4 * self.diameter/2))
        self.theta1 = -4 / 3 * np.arctan2(self.pitch, 2 * np.pi * 3/4 * self.diameter/2)



@dataclass
class VehicleParams:

    propellers: List[PropellerParams]
    angles: np.ndarray
    distances: np.ndarray

    mass: float = 1.
    inertia_matrix: np.matrix = np.eye(3)


    def __post_init__(self):
        self.inertia_matrix_inverse = np.linalg.inv(self.inertia_matrix)



@dataclass
class SimulationParams:

    dt: float = 1e-2
    """Timestep of simulation"""
    g: float = 9.81
    """Gravitational acceleration"""
    rho: float = 1.225
    "Air density kg/m^3 at MSL"
