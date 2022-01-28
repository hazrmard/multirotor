from typing import Iterable, List, Union
from dataclasses import dataclass

import numpy as np

@dataclass
class PropellerParams:
    
    # Manufacturer propeller length x pitch specification:
    diameter: float = 6  #inches
    pitch: float = 3   #inches

    a: float = 5.7
    "Lift curve slope used in example in Stevens & Lewis" # TODO: read more
    b: float = 2
    "Number of blades"
    c: float = 0.0274
    "Mean chord length (m)"
    eta: float = 1.
    "Propeller efficiency"

    k_lift: float = None
    "Propeller's aerodynamic lift coefficient"
    k_drag: float = None
    "Propeller's aerodynamic drag coefficient"

    def __post_init__(self):
        self.R = self.diameter * 0.0254 # inches to metres
        "Radius in metres"
        self.A = np.pi * self.R**2
        "Area of propeller disc in metres squared"
        self.theta0 = 2*np.arctan2(self.pitch, (2 * np.pi * 3/4 * self.diameter/2))
        self.theta1 = -4 / 3 * np.arctan2(self.pitch, 2 * np.pi * 3/4 * self.diameter/2)



@dataclass
class MotorParams:

    kt: float
    "Torque constant"
    km: float
    "Motor constant"
    ke: float = None
    "Back-EMF constant=Torque constant"


    def __post_init__(self):
        self.ke = self.kt



@dataclass
class VehicleParams:

    propellers: List[PropellerParams]
    angles: np.ndarray
    distances: np.ndarray
    clockwise: np.ndarray = None

    mass: float = 1.
    inertia_matrix: np.matrix = np.eye(3)


    def __post_init__(self):
        self.inertia_matrix_inverse = np.linalg.inv(self.inertia_matrix)
        if self.clockwise is None:
            self.clockwise = np.ones(len(self.propellers), dtype=int)
            self.clockwise[::2] = 1
            self.clockwise[1::2] = -1



@dataclass
class SimulationParams:

    dt: float = 1e-2
    """Timestep of simulation"""
    g: float = 9.81
    """Gravitational acceleration"""
    rho: float = 1.225
    "Air density kg/m^3 at MSL"
