"""
The parameters that describe a vehicle, as plain dataclasses.

`MotorParams`, `PropellerParams`, `BatteryParams` and `VehicleParams` are also JAX pytrees.
Their numeric fields are the leaves, so a function of a `VehicleParams` can be compiled,
mapped and differentiated: ``jax.grad(f)(params)`` returns the gradient as a `VehicleParams`.
`multirotor.builder.build_vehicle` turns them into the arrays the simulation takes, with
JAX operations, so the gradient reaches every field.

To keep that working, a dataclass here does no arithmetic when it is made: JAX makes
instances with placeholder values, and with tracers. The values derived from the fields
(``R``, ``A``, ``inertia_matrix_inverse`` and so on) are properties, computed when read.
Checks of the values are skipped for values that are not concrete numbers or arrays.
Fields that are not numbers to differentiate (``b``, ``use_thrust_constant``) are static.
"""
from typing import Any, List
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_dataclass



def _xp(*values: Any):
    "``jax.numpy`` if any of the values is a JAX array or a tracer, otherwise ``numpy``."
    return jnp if any(isinstance(v, jax.Array) for v in values) else np



def xp_of(tree: Any):
    "``jax.numpy`` if any leaf of the tree is a JAX array or a tracer, otherwise ``numpy``."
    return _xp(*jax.tree_util.tree_leaves(tree))



def _number(x: Any) -> Any:
    "A Python ``int`` as a ``float``, so that it can be differentiated. Anything else as is."
    return float(x) if isinstance(x, (int, np.integer)) and not isinstance(x, bool) else x



def _array(x: Any, keep_dtype: bool = False) -> Any:
    """
    A list, tuple or numpy array as a float array. Anything else, such as a tracer, as is.
    A numpy array of floats keeps its type if ``keep_dtype``.
    """
    if keep_dtype and isinstance(x, np.ndarray) and x.dtype.kind == 'f':
        return x
    return np.asarray(x, dtype=float) if isinstance(x, (list, tuple, np.ndarray)) else x



def _concrete(*values: Any) -> bool:
    "Whether all of the values are numbers, arrays or None, not tracers or placeholders."
    return all(
        v is None or isinstance(v, (int, float, np.number, np.ndarray, list, tuple))
        for v in values)



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


    def __post_init__(self) -> None:
        "Fill in whichever of the torque and back-EMF constants is missing from the other."
        # See: https://www.motioncontroltips.com/faq-difference-between-torque-back-emf-motor-constant/
        # For an ideal square-wave motor, torque and back-emf constants are same
        for name in _MOTOR_FIELDS:
            setattr(self, name, _number(getattr(self, name)))
        if self.k_torque is None:
            self.k_torque = self.k_emf
        if self.k_emf is None:
            self.k_emf = self.k_torque


_MOTOR_FIELDS = (
    'resistance', 'k_motor', 'k_emf', 'k_torque', 'k_drag', 'moment_of_inertia', 'k_df',
    'static_friction', 'speed_voltage_scaling', 'max_current')
register_dataclass(MotorParams, data_fields=list(_MOTOR_FIELDS), meta_fields=[])



@dataclass
class PropellerParams:

    moment_of_inertia: float

    # Manufacturer propeller length x pitch specification:
    diameter: float = 6.  #inches
    pitch: float = 3.   #inches

    a: float = 5.7
    "Lift curve slope used in example in Stevens & Lewis (eq 2.2-6a)"
    # d C_L / d alpha = 2 pi / (sqrt(1-M^2)), where M is mach number, alpha is
    # aerodynamic angle of x-axis of body and x-axis of stability
    b: int = 2
    "Number of blades. Static: it is not differentiated."
    c: float = 0.0274
    "Mean chord length (m) of the propeller blade"
    eta: float = 1.
    "Propeller efficiency"
    use_thrust_constant: bool = True
    "Use k_thrust instead of propeller geometry for thrust calculations. Static."
    k_thrust: float = None
    "Propeller's aerodynamic thrust coefficient, where thrust =  k_thrust * angular velocity^2"
    k_drag: float = None
    "Torque constant or drag coefficient, where torque = k_drag * angular velocity^2"

    motor: MotorParams = None
    "The parameters of the motor to simulate speed, otherwise instantaneous."

    def __post_init__(self) -> None:
        "Check the thrust model, and give the motor the propeller's drag coefficient."
        for name in _PROPELLER_FIELDS:
            setattr(self, name, _number(getattr(self, name)))
        if self.use_thrust_constant and self.k_thrust is None:
            raise ValueError('"k_thrust" is None, even though "use_thrust_constant=True"')
        if self.motor is not None and self.k_drag is not None:
            self.motor.k_drag = self.k_drag


    @property
    def R(self) -> float:
        "Radius in metres"
        return self.diameter * 0.0254 # inches to metres

    @property
    def A(self) -> float:
        "Area of propeller disc in metres squared"
        return np.pi * self.R**2

    @property
    def theta0(self) -> float:
        "Pitch angle at root of blade"
        # Pitch angle is reduced to allow for even lift as blade velocity increases
        # with increasing radius. Assuming linear reduction from root to tip.
        xp = _xp(self.pitch, self.diameter)
        return 2 * xp.arctan2(self.pitch, (2 * np.pi * 3/4 * self.diameter/2))

    @property
    def theta1(self) -> float:
        "Change in pitch angle towards tip of blade"
        xp = _xp(self.pitch, self.diameter)
        return -4 / 3 * xp.arctan2(self.pitch, 2 * np.pi * 3/4 * self.diameter/2)


_PROPELLER_FIELDS = (
    'moment_of_inertia', 'diameter', 'pitch', 'a', 'c', 'eta', 'k_thrust', 'k_drag')
register_dataclass(
    PropellerParams, data_fields=[*_PROPELLER_FIELDS, 'motor'],
    meta_fields=['b', 'use_thrust_constant'])



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


    def __post_init__(self) -> None:
        """
        Check the battery's parameters, when they are concrete values.

        Raises
        ------
        ValueError
            If only one of ``ocv_soc`` and ``ocv_voltage`` is given, if they are not 1D, of
            equal length and increasing, if the capacity is not positive, the internal
            resistance is negative, or a battery model has neither a finite ``max_voltage``
            nor a table.
        """
        for name in ('max_voltage', 'capacity', 'internal_resistance', 'min_voltage'):
            setattr(self, name, _number(getattr(self, name)))
        self.ocv_soc, self.ocv_voltage = _array(self.ocv_soc), _array(self.ocv_voltage)
        if not _concrete(*(getattr(self, f) for f in _BATTERY_FIELDS)):
            return
        if (self.ocv_soc is None) != (self.ocv_voltage is None):
            raise ValueError('Give both ocv_soc and ocv_voltage, or neither.')
        if self.ocv_soc is not None:
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


_BATTERY_FIELDS = (
    'max_voltage', 'capacity', 'internal_resistance', 'ocv_soc', 'ocv_voltage', 'min_voltage')
register_dataclass(BatteryParams, data_fields=list(_BATTERY_FIELDS), meta_fields=[])



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


    def __post_init__(self) -> None:
        "Make the arrays float arrays, and alternate the propellers' spin by default."
        self.mass = _number(self.mass)
        self.inertia_matrix = _array(self.inertia_matrix, keep_dtype=True)
        self.angles = _array(self.angles)
        self.distances = _array(self.distances)
        if self.clockwise is None:
            # The same ``n`` as ``propellers``, a list when JAX makes a placeholder
            if isinstance(self.propellers, (list, tuple)):
                self.clockwise = np.ones(len(self.propellers))
                self.clockwise[1::2] = -1.
        else:
            self.clockwise = _array(self.clockwise)


    @property
    def nprops(self) -> int:
        "Number of propellers"
        return len(self.propellers)

    @property
    def inertia_matrix_inverse(self) -> np.ndarray:
        "The inverse of the inertia matrix."
        return _xp(self.inertia_matrix).linalg.inv(self.inertia_matrix)

    @property
    def propeller_vectors(self) -> np.ndarray:
        "The propellers' positions relative to the center of mass, shape (3, n)."
        xp = _xp(self.angles, self.distances)
        x = xp.cos(self.angles) * self.distances
        y = xp.sin(self.angles) * self.distances
        return xp.stack((x, y, xp.zeros_like(y)))


register_dataclass(
    VehicleParams,
    data_fields=['mass', 'inertia_matrix', 'propellers', 'angles', 'distances', 'clockwise',
                 'battery'],
    meta_fields=[])



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
