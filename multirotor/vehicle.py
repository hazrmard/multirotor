from typing import Iterable, List, Union
from dataclasses import dataclass

import numpy as np



@dataclass
class MotorParams:

    resistance: float
    "Motor resistance r. Current = (voltage - back_emf) / resistance"
     # See: http://learningrc.com/motor-kv/, http://web.mit.edu/first/scooter/motormath.pdf
    k_motor: float = None
    "Motor constant k_m, where speed = k_m * back_emf, and k_m = 1/k_e"
    k_emf: float = None
    "Back-EMF constant k_e, relates motor speed induced voltage, back_emf = k_e * omega"
    k_torque: float = None
    "Torque constant k_q, where torque Q = k_q * current. Equal to k_e"
    k_drag: float = None # overwritten by PropellerParams, if provided there
    "Aerodynamic drag coefficient, where torque = k_drag * omega^2"
    moment_of_inertia: float = 0.
    "Moment of inertia about rotational axis"
    k_df: float = 0.
    "Viscous damping coefficient. Torque = d_f * speed"
    static_friction: float = 0.
    speed_voltage_scaling: float = 1.
    """Scaling constant to convert speed signal (rad/s) into speed controller voltage (V).
    If 1, means input action is same as Voltage"""
    max_current: float = np.inf
    "Max allowed current"


    def __post_init__(self):
        # See: https://www.motioncontroltips.com/faq-difference-between-torque-back-emf-motor-constant/
        # For an ideal square-wave motor, torque and back-emf constants are same
        self.k_torque = self.k_torque or self.k_emf
        self.k_emf = self.k_emf or self.k_torque


@dataclass
class PropellerParams:

    moment_of_inertia: float
    
    # Manufacturer propeller length x pitch specification:
    diameter: float = 6  #inches
    pitch: float = 3   #inches

    a: float = 5.7
    "Lift curve slope used in example in Stevens & Lewis (eq 2.2-6a)"
    # d C_L / d alpha = 2 pi / (sqrt(1-M^2)), where M is mach number, alpha is 
    # aerodynamic angle of x-axis of body and x-axis of stability
    b: float = 2
    "Number of blades"
    c: float = 0.0274
    "Mean chord length (m) of the propeller blade"
    eta: float = 1.
    "Propeller efficiency"
    use_thrust_constant: bool = True
    "Use k_thrust instead of propeller geometry for thrust calculations."
    k_thrust: float = None
    "Propeller's aerodynamic thrust coefficient, where thrust =  k_thrust * angular velocity^2"
    k_drag: float = None
    "Torque constant or drag coefficient, where torque = k_drag * angular velocity^2"

    motor: MotorParams = None
    "The parameters of the motor to simulate speed, otherwise instantaneous."

    def __post_init__(self):
        # ensure torque function gets floats so adding float + int in the njit
        # torque function is not ignored
        self.moment_of_inertia = float(self.moment_of_inertia)
        self.R = self.diameter * 0.0254 # inches to metres
        "Radius in metres"
        self.A = np.pi * self.R**2
        "Area of propeller disc in metres squared"
        self.theta0 = 2*np.arctan2(self.pitch, (2 * np.pi * 3/4 * self.diameter/2))
        "Pitch angle at root of blade"
        self.theta1 = -4 / 3 * np.arctan2(self.pitch, 2 * np.pi * 3/4 * self.diameter/2)
        "Change in pitch angle towards tip of blade"
        if self.use_thrust_constant and self.k_thrust is None:
            raise ValueError('"k_thrust" is None, even though "use_thrust_constant=True"')
        if self.motor is not None:
            self.motor.k_drag = self.k_drag
        # Pitch angel is reduced to allow for even lift as blade velocity increases
        # with increasing radius. Assuming linear reduction from root to tip.



@dataclass
class BatteryParams:

    max_voltage: float = np.inf
    """Maximum voltage of the battery. Without a battery model (``capacity=None``)
    it limits the motors' voltage. With a battery model and no OCV table, it is
    the open-circuit voltage."""
    capacity: float = None
    """Capacity (Ah). None means no battery model: the battery is a fixed
    voltage limit, and its charge is not tracked. Once the whole capacity is
    drawn, the battery is empty and supplies no voltage or current."""
    internal_resistance: float = 0.
    "Series resistance (ohm). The terminal voltage sags by resistance x current."
    ocv_soc: np.ndarray = None
    "State of charge (0 to 1, increasing) at which ``ocv_voltage`` is given."
    ocv_voltage: np.ndarray = None
    "Open-circuit voltage (V) at each ``ocv_soc``. Interpolated linearly."
    min_voltage: float = 0.
    """Cutoff voltage, a protection on top of the battery model. When the
    terminal voltage falls below it, the battery cuts the motors' voltage to
    zero until it is reset. 0 disables the cutoff."""


    def __post_init__(self):
        if (self.ocv_soc is None) != (self.ocv_voltage is None):
            raise ValueError('Give both ocv_soc and ocv_voltage, or neither.')
        if self.ocv_soc is not None:
            self.ocv_soc = np.asarray(self.ocv_soc, dtype=float)
            self.ocv_voltage = np.asarray(self.ocv_voltage, dtype=float)
            if self.ocv_soc.shape != self.ocv_voltage.shape or self.ocv_soc.ndim != 1:
                raise ValueError('ocv_soc and ocv_voltage must be 1D and of the same length.')
            if np.any(np.diff(self.ocv_soc) <= 0):
                raise ValueError('ocv_soc must be increasing.')
        if self.capacity is not None and self.capacity <= 0:
            raise ValueError('capacity must be positive.')
        if self.internal_resistance < 0:
            raise ValueError('internal_resistance must not be negative.')
        if self.capacity is not None and self.ocv_soc is None and not np.isfinite(self.max_voltage):
            raise ValueError('A battery model needs a finite max_voltage or an OCV table.')



@dataclass
class VehicleParams:

    mass: float
    "The total mass of the vehicle"
    # TODO: Should moments of inertia of propeller motors be added to this matrix
    # manually? Currently we assume this represents the whole vehicle.
    inertia_matrix: np.ndarray
    "3x3 intertia matrix describing the rotational properties of the body."
    propellers: List[PropellerParams]
    angles: np.ndarray
    "Angle (radians) of propeller arm from the positive x-axis (forward) of the body frame."
    distances: np.ndarray
    "Distance (m) of each propeller from the centre of mass."
    clockwise: np.ndarray = None
    """1 if motor spins clockwise, -1 if anti-clockwise, looking from the top.
    Defaults to alternating clockwise/anti-clockwise."""
    battery: BatteryParams = None
    "The battery parameters"


    def __post_init__(self):
        self.nprops = len(self.propellers)
        "Number of propellers"
        self.distances = self.distances.astype(float)
        self.inertia_matrix_inverse = np.linalg.inv(self.inertia_matrix)
        x = np.cos(self.angles) * self.distances
        y = np.sin(self.angles) * self.distances
        z = np.zeros_like(y)
        self.propeller_vectors = np.vstack((x, y, z))
        if self.clockwise is None:
            self.clockwise = np.ones(len(self.propellers), dtype=int)
            self.clockwise[::2] = 1
            self.clockwise[1::2] = -1



@dataclass
class SimulationParams:

    dt: float = 1e-3
    """Timestep of simulation"""
    g: float = 9.81
    """Gravitational acceleration"""
    rho: float = 1.225
    "Air density kg/m^3 at MSL"
    dtype: type = None
    "Default data type for arrays. If None, inferred from VehicleParams inertia_matrix."
