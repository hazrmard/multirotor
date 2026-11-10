"""
Equations of the vehicle's subsystems: propeller thrust and drag, motor
electrical and mechanical behavior, battery voltage, and the rigid body's
response to forces and torques. Each function computes one quantity from its
inputs. `multirotor.dynamics` combines them into the right-hand sides that the
integrator advances.

Which function calls which, for following the order of execution. An arrow goes
from a caller to the function it calls. Everything is compiled with numba except
the blade-element thrust functions, which solve an equation with scipy's
``fsolve``::

    Motors
        motor_acc ----------------> motor_electrical
        motor_power --------------> motor_electrical

    Battery
        battery_operating_point --> motor_power --> motor_electrical
                                +-> battery_terminal
        battery_end_step ---------> battery_ocv
                                +-> battery_operating_point
        battery_ocv

    Body
        forces_torques -----------> body_torques
        body_derivative

    Blade-element thrust (not compiled)
        thrusts_blade_element_model --> thrust_blade_element_model
                                        --> thrustEqn (passed to fsolve)

Functions with no arrow after them, such as ``battery_ocv`` and
``body_derivative``, call no other function in this module.

`multirotor.simulation` also calls some of these directly, between solver steps:
``motor_electrical`` for each motor's voltage and current,
``battery_end_step`` and ``battery_ocv`` for the battery's state,
``body_torques`` and ``thrusts_blade_element_model`` in
`Multirotor.get_forces_torques`, and ``thrust_blade_element_model`` in
`Propeller.thrust`.
"""
from typing import Iterable, Tuple

import numpy as np
from numba import njit
from scipy.optimize import fsolve

from . import xp_numba as xp
from .lazy import build



# Row indices of the parameter matrices in `multirotor.data`. Each matrix has
# one column per propeller. Kernels index rows with these constants, which numba
# treats as compile-time constants.

# PropellerData.params
PROP_K_THRUST = 0
PROP_K_DRAG = 1
PROP_MOMENT_OF_INERTIA = 2
PROP_RADIUS = 3
PROP_AREA = 4
PROP_LIFT_SLOPE = 5
PROP_BLADES = 6
PROP_CHORD = 7
PROP_EFFICIENCY = 8
PROP_THETA0 = 9
PROP_THETA1 = 10
N_PROP_PARAMS = 11

# MotorData.params
MOTOR_RESISTANCE = 0
MOTOR_K_EMF = 1
MOTOR_K_TORQUE = 2
MOTOR_K_DF = 3
MOTOR_K_DRAG = 4
MOTOR_MOMENT_OF_INERTIA = 5
MOTOR_SPEED_VOLTAGE_SCALING = 6
MOTOR_MAX_CURRENT = 7
N_MOTOR_PARAMS = 8

# Battery parameter vector (see `simulation.Battery`)
BATTERY_CAPACITY = 0
"Capacity in coulombs."
BATTERY_INTERNAL_RESISTANCE = 1
BATTERY_MIN_VOLTAGE = 2
N_BATTERY_PARAMS = 3



@njit
def thrustEqn(vi, *prop_params):
    R,A,rho,a,b,c,eta,theta0,theta1,u,v,w,Omega = prop_params
    
    # Calculate local airflow velocity at propeller with vi, v'
    vprime = np.sqrt(u**2 + v**2 + (w - vi)**2)
    
    # Calculate Thrust averaged over one revolution of propeller using vi
    Thrust = 1/4 * rho * a * b * c * R * \
        ( (w - vi) * Omega * R + 2/3 * (Omega * R)**2 * (theta0 + 3/4 * theta1) + \
          (u**2 + v**2) * (theta0 + 1/2 * theta1) )
    
    # Calculate residual for equation: Thrust = mass flow rate * delta Velocity
    residual = eta * 2 * vi * rho * A * vprime - Thrust
    return residual



def thrust_blade_element_model(
    prop_speed, airstream_velocity, R, A, rho, a, b, c, eta, theta0, theta1
) -> float:
    u, v, w = airstream_velocity
    # Convert commanded RPM to rad/s
    # omega = 2 * np.pi / 60 * prop_speed # Commented out - already in rad/s
    omega = prop_speed
    vi_guess = airstream_velocity[2]
    
    #Collect propeller config, state, and input parameters
    prop_params = (R,A,rho,a,b,c,eta,theta0,theta1,u,v,w,omega)
    
    # Numerically solve for propeller induced velocity, vi
    # using nonlinear root finder, fsolve, and prop_params
    # TODO: numba jit gives error for fsolve ('Untyped global name fsolve')
    vi = fsolve(thrustEqn, vi_guess, args=prop_params, xtol=1e-4)[0]
    
    # Plug vi back into Thrust equation to solve for T
    vprime = np.sqrt(u**2 + v**2 + (w - vi)**2)
    thrust = eta * 2 * vi * rho * A * vprime
    return thrust



def thrusts_blade_element_model(
    speeds: np.ndarray, airstream_velocities: np.ndarray, prop_params: np.ndarray,
    rho: float
) -> np.ndarray:
    """
    Thrust of each propeller from the blade-element model, by solving
    `thrust_blade_element_model` for each propeller in turn.

    Parameters
    ----------
    speeds : np.ndarray
        Propeller speeds (rad/s), shape (n,).
    airstream_velocities : np.ndarray
        Airstream velocity at each propeller, shape (3, n).
    prop_params : np.ndarray
        The `PropellerData.params` matrix, shape (N_PROP_PARAMS, n).
    rho : float
        Air density (kg/m^3).

    Returns
    -------
    np.ndarray
        The thrust (N) of each propeller, shape (n,).
    """
    p = prop_params
    thrusts = np.zeros(len(speeds))
    for i in range(len(speeds)):
        thrusts[i] = thrust_blade_element_model(
            speeds[i], airstream_velocities[:, i],
            p[PROP_RADIUS, i], p[PROP_AREA, i], rho, p[PROP_LIFT_SLOPE, i],
            p[PROP_BLADES, i], p[PROP_CHORD, i], p[PROP_EFFICIENCY, i],
            p[PROP_THETA0, i], p[PROP_THETA1, i]
        )
    return thrusts



def _voltage(i, c):
    "Voltage of motor ``i``. ``c`` is the arguments of `motor_electrical`."
    speed, u, max_voltage, p = c
    return xp.minimum(xp.maximum(p[MOTOR_SPEED_VOLTAGE_SCALING, i] * u[i], 0.), max_voltage)



def _current(i, c):
    "Current of motor ``i``. ``c`` is the arguments of `motor_electrical`."
    speed, u, max_voltage, p = c
    back_emf = speed[i] * p[MOTOR_K_EMF, i]
    return xp.minimum(
        xp.maximum((_voltage(i, c) - back_emf) / p[MOTOR_RESISTANCE, i], 0.),
        p[MOTOR_MAX_CURRENT, i])



def motor_electrical(
    speed: np.ndarray, u: np.ndarray, max_voltage: float, motor_params: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voltage and current of each motor. The current responds instantly to the
    voltage and back-EMF; the motors' inductance is neglected.

    Parameters
    ----------
    speed : np.ndarray
        Motor speeds (rad/s), shape (n,).
    u : np.ndarray
        Speed signals, shape (n,). The voltage is
        ``speed_voltage_scaling * u``, clipped to ``[0, max_voltage]``.
    max_voltage : float
        The maximum voltage the power source can supply.
    motor_params : np.ndarray
        The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The voltage (V) and current (A) of each motor.
    """

    n = speed.shape[0]
    consts = (speed, u, max_voltage, motor_params)
    return xp.map_(_voltage, n, consts), xp.map_(_current, n, consts)



def _acc(i, c):
    "Angular acceleration of motor ``i``. ``c`` is the arguments of `motor_acc`."
    speed, u, max_voltage, p = c
    torque = p[MOTOR_K_TORQUE, i] * _current(i, c)
    friction = p[MOTOR_K_DF, i] * speed[i]
    drag = p[MOTOR_K_DRAG, i] * speed[i]**2
    return (torque - friction - drag) / p[MOTOR_MOMENT_OF_INERTIA, i]



def motor_acc(
    speed: np.ndarray, u: np.ndarray, max_voltage: float, motor_params: np.ndarray
) -> np.ndarray:
    """
    Angular acceleration of each motor: electrical torque minus viscous friction
    and propeller drag, divided by the moment of inertia.

    Parameters
    ----------
    speed : np.ndarray
        Motor speeds (rad/s), shape (n,).
    u : np.ndarray
        Speed signals, shape (n,).
    max_voltage : float
        The maximum voltage the power source can supply.
    motor_params : np.ndarray
        The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).

    Returns
    -------
    np.ndarray
        Angular acceleration (rad/s^2) of each motor, shape (n,).
    """

    n = speed.shape[0]
    return xp.map_(_acc, n, (speed, u, max_voltage, motor_params))



def _add_power(acc, i, c):
    "``acc`` plus the electrical power of motor ``i``."
    return acc + _voltage(i, c) * _current(i, c)



def motor_power(
    speed: np.ndarray, u: np.ndarray, max_voltage: float, motor_params: np.ndarray
) -> float:
    """
    Total electrical power (W) drawn by the motors.

    Parameters
    ----------
    speed, u, max_voltage, motor_params
        As in `motor_electrical`.

    Returns
    -------
    float
        The sum of each motor's voltage times its current.
    """

    return xp.fold(_add_power, 0., speed.shape[0], (speed, u, max_voltage, motor_params))



def _empty(op):
    return 0.



def _interpolate(op):
    return xp.interp(op[0], op[1], op[2])



def battery_ocv(soc: float, ocv_soc: np.ndarray, ocv_voltage: np.ndarray) -> float:
    """
    Open-circuit voltage of the battery at a state of charge, interpolated
    linearly from a table. Outside the table, the nearest end's voltage is used.
    An empty battery (``soc <= 0``) has no voltage.

    Parameters
    ----------
    soc : float
        State of charge, from 0 (empty) to 1 (full).
    ocv_soc : np.ndarray
        States of charge of the table, increasing.
    ocv_voltage : np.ndarray
        Open-circuit voltage (V) at each ``ocv_soc``.

    Returns
    -------
    float
        The open-circuit voltage (V).
    """

    return xp.cond(soc <= 0., _empty, _interpolate, (soc, ocv_soc, ocv_voltage))



def _ideal(op):
    return op[1], True



def _loaded(op):
    discriminant = op[1]**2 - 4. * op[2] * op[0]
    return xp.cond(discriminant < 0., _overloaded, _delivering, (op[1], discriminant))



def _overloaded(op):
    return 0.5 * op[0], False



def _delivering(op):
    return 0.5 * (op[0] + xp.sqrt_nonneg(op[1])), True



def battery_terminal(
    power: float, open_circuit_voltage: float, internal_resistance: float
) -> Tuple[float, bool]:
    """
    Terminal voltage of a battery delivering ``power``, modeling the motors'
    speed controllers as lossless.

    The terminal voltage is ``V = V_oc - R * I`` with current ``I = P / V``, so
    ``V = (V_oc + sqrt(V_oc**2 - 4 R P)) / 2``. If the battery can't deliver
    the power (``V_oc**2 < 4 R P``), the voltage is clamped at ``V_oc / 2``,
    where the battery delivers its maximum power.

    Parameters
    ----------
    power : float
        The power (W) drawn from the battery.
    open_circuit_voltage : float
        The open-circuit voltage (V).
    internal_resistance : float
        The battery's series resistance (ohm).

    Returns
    -------
    Tuple[float, bool]
        The terminal voltage (V), and whether the battery can deliver the power.
    """

    return xp.cond(
        internal_resistance == 0., _ideal, _loaded,
        (power, open_circuit_voltage, internal_resistance))



def _unlimited(op):
    "No motor is limited by the terminal voltage: it is as first computed."
    return op[0], op[1], op[2]



def _balance(v, op):
    "Zero where the motors' voltage limit ``v`` equals the terminal voltage."
    _, _, _, speed, u, v_oc, resistance, motor_params = op
    power = motor_power(speed, u, v, motor_params)
    terminal_voltage, _ = battery_terminal(power, v_oc, resistance)
    return v - terminal_voltage



def _limited(op):
    "Some motors are limited by the terminal voltage, found by bisection on [0, V_oc]."
    _, _, _, speed, u, v_oc, resistance, motor_params = op
    terminal_voltage = xp.root(_balance, 0., v_oc, 1e-12 * v_oc, op)
    power = motor_power(speed, u, terminal_voltage, motor_params)
    _, ok = battery_terminal(power, v_oc, resistance)
    return terminal_voltage, power, ok



def battery_operating_point(
    speed: np.ndarray, u: np.ndarray, open_circuit_voltage: float,
    internal_resistance: float, motor_params: np.ndarray
) -> Tuple[float, float, bool]:
    """
    Terminal voltage and power at which the battery and the motors agree: the
    terminal voltage limits the motors' voltage, and the power the motors draw
    sets the terminal voltage.

    If no motor's voltage reaches the terminal voltage, the power does
    not depend on it, and `battery_terminal` gives the answer directly. Otherwise
    the terminal voltage is found by bisection on ``[0, V_oc]``. The bisection
    always converges: a higher terminal voltage lets the motors draw more power,
    which lowers the terminal voltage, so there is exactly one solution.

    Parameters
    ----------
    speed : np.ndarray
        Motor speeds (rad/s), shape (n,).
    u : np.ndarray
        Speed signals, shape (n,).
    open_circuit_voltage : float
        The open-circuit voltage (V).
    internal_resistance : float
        The battery's series resistance (ohm).
    motor_params : np.ndarray
        The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).

    Returns
    -------
    Tuple[float, float, bool]
        The terminal voltage (V), the power (W) the motors draw, and whether the
        battery can deliver that power.
    """

    v_oc = open_circuit_voltage
    requested_voltage = motor_params[MOTOR_SPEED_VOLTAGE_SCALING] * u
    # Power with motor voltages limited only by the open-circuit voltage
    power = motor_power(speed, u, v_oc, motor_params)
    terminal_voltage, ok = battery_terminal(power, v_oc, internal_resistance)
    return xp.cond(
        requested_voltage.max() <= terminal_voltage, _unlimited, _limited,
        (terminal_voltage, power, ok, speed, u, v_oc, internal_resistance, motor_params))



def _dead(op):
    "A battery that is cut off or empty supplies nothing."
    charge_drawn, current, open_circuit_voltage, cutoff, overload = op[:5]
    return charge_drawn, current, open_circuit_voltage, cutoff, overload, 0.



def _live(op):
    charge_drawn, current, open_circuit_voltage, cutoff, overload = op[:5]
    motor_speeds, u, motor_params, battery_params = op[5:]
    voltage, _, ok = battery_operating_point(
        motor_speeds, u, open_circuit_voltage,
        battery_params[BATTERY_INTERNAL_RESISTANCE], motor_params)
    # The cutoff takes effect from the next step
    cutoff = cutoff | (voltage < battery_params[BATTERY_MIN_VOLTAGE])
    return charge_drawn, current, voltage, cutoff, xp.logical_not(ok), voltage



def battery_end_step(
    charge: float, charge_drawn: float, cutoff: bool, overload: bool,
    motor_speeds: np.ndarray, u: np.ndarray, motor_params: np.ndarray,
    battery_params: np.ndarray, ocv_soc: np.ndarray, ocv_voltage: np.ndarray,
    dt: float
) -> Tuple[float, float, float, bool, bool, float]:
    """
    The battery's state after one integrated time step. The charge drawn during
    the step is limited to the charge left in the battery. A battery that is cut
    off or empty supplies no current from then on. Otherwise the cutoff latches
    if the terminal voltage falls below the minimum.

    Parameters
    ----------
    charge : float
        Charge (C) drawn during the step, as integrated.
    charge_drawn : float
        Charge (C) drawn before the step.
    cutoff : bool
        Whether the battery had cut off the motors before the step.
    overload : bool
        Whether the previous step demanded more power than the battery can deliver.
    motor_speeds : np.ndarray
        Motor speeds (rad/s) at the end of the step, shape (n,).
    u : np.ndarray
        Speed signals during the step, shape (n,).
    motor_params : np.ndarray
        The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).
    battery_params, ocv_soc, ocv_voltage
        As in `multirotor.dynamics.battery_state`.
    dt : float
        The time step (s).

    Returns
    -------
    Tuple[float, float, float, bool, bool, float]
        The charge drawn, the battery current (A, averaged over the step), the
        terminal voltage (V), whether the battery has cut off, whether it was
        overloaded, and the motors' voltage limit (V) for the next step.
    """

    capacity = battery_params[BATTERY_CAPACITY]
    # The integrator can overshoot the capacity by its tolerance
    charge = xp.minimum(charge, capacity - charge_drawn)
    charge_drawn = charge_drawn + charge
    current = charge / dt
    soc = 1. - charge_drawn / capacity
    open_circuit_voltage = battery_ocv(soc, ocv_soc, ocv_voltage)
    return xp.cond(
        cutoff | (soc <= 0.), _dead, _live,
        (charge_drawn, current, open_circuit_voltage, cutoff, overload,
         motor_speeds, u, motor_params, battery_params))



def _add_torque(acc, i, c):
    "``acc`` plus the torques of propeller ``i``."
    thrusts, drag_torques, clockwise, positions = c
    # tau = r x F. Thrust acts only along the body z-axis, F = (0, 0, thrust),
    # so the cross product reduces to these two terms. Propellers tilted away
    # from the z-axis would need the full cross product.
    return (
        acc[0] + positions[1, i] * thrusts[i],
        acc[1] - positions[0, i] * thrusts[i],
        acc[2] + clockwise[i] * drag_torques[i])



def body_torques(
    thrusts: np.ndarray, drag_torques: np.ndarray, clockwise: np.ndarray,
    positions: np.ndarray
) -> np.ndarray:
    """
    Total torque on the three body axes (roll, pitch, yaw) due to the
    propellers' loads. Thrust acts along the body z-axis. Each propeller's drag
    torque acts about the z-axis, opposite to its direction of rotation.

    Parameters
    ----------
    thrusts : np.ndarray
        Thrust (N) of each propeller, shape (n,).
    drag_torques : np.ndarray
        Magnitude of the drag torque (N m) of each propeller, shape (n,). For a
        propeller, ``k_drag * speed**2``.
    clockwise : np.ndarray
        1 if a propeller spins clockwise, -1 if anti-clockwise, shape (n,).
    positions : np.ndarray
        Position of each propeller relative to the center of mass, shape (3, n).

    Returns
    -------
    np.ndarray
        The torque (N m) on the body, shape (3,).
    """

    # Angular acceleration of propellers is assumed to be negligible, so it
    # contributes no torque.
    n = thrusts.shape[0]
    roll, pitch, yaw = xp.fold(
        _add_torque, (0., 0., 0.), n, (thrusts, drag_torques, clockwise, positions))
    return xp.array((roll, pitch, yaw))



def forces_torques(
    speeds: np.ndarray, prop_params: np.ndarray, clockwise: np.ndarray,
    positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Total force and torque on the body from propellers whose thrust is
    ``k_thrust * speed**2`` and drag torque is ``k_drag * speed**2``.

    Parameters
    ----------
    speeds : np.ndarray
        Propeller speeds (rad/s), shape (n,).
    prop_params : np.ndarray
        The `PropellerData.params` matrix, shape (N_PROP_PARAMS, n).
    clockwise : np.ndarray
        1 if a propeller spins clockwise, -1 if anti-clockwise, shape (n,).
    positions : np.ndarray
        Position of each propeller relative to the center of mass, shape (3, n).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The force (N) and torque (N m) on the body, each shape (3,).
    """

    thrusts = prop_params[PROP_K_THRUST] * speeds**2
    drag_torques = prop_params[PROP_K_DRAG] * speeds**2
    torques = body_torques(thrusts, drag_torques, clockwise, positions)
    forces = xp.array((0., 0., thrusts.sum()))
    return forces, torques



def body_derivative(
    forces: np.ndarray, torques: np.ndarray, x: np.ndarray, g: float, mass: float,
    inertia_matrix: np.matrix, inertia_matrix_inverse: np.matrix
) -> np.ndarray:
    """
    Given forces and torqes, return the rate of change of state.

    Parameters
    ----------
    forces : np.ndarray
        Forces acting in the body frame.
    torques : np.ndarray
        Torques acting in the body frame.
    x : np.ndarray
        State of the vehicle.
    g : float
        Gravitational acceleration.
    mass : float
        Mass of the vehicle.
    inertia_matrix : np.matrix
        Inertial matrix.
    inertia_matrix_inverse : np.matrix
        Inverse of inertial matrix.

    Returns
    -------
    np.ndarray
        The rate of change of state d(state)/dt
    """
    # Store state variables in a readable format
    ub = x[3]       # linear velocity along body-frame-x-axis b1
    vb = x[4]       # linear velocity along body-frame-y-axis b2
    wb = x[5]       # linear velocity along body-frame-z-axis b3
    phi = x[6]      # Roll
    theta = x[7]    # Pitch
    psi = x[8]      # Yaw
    p = x[9]        # body-frame-x-axis rotation rate
    q = x[10]       # body-frame-y-axis rotation rate
    r = x[11]       # body-frame-z-axis rotation rate

    # Pre-calculate trig values
    cphi = xp.cos(phi);   sphi = xp.sin(phi)    # roll
    cthe = xp.cos(theta); sthe = xp.sin(theta)  # pitch
    cpsi = xp.cos(psi);   spsi = xp.sin(psi)    # yaw

    f1, f2, f3 = forces[0], forces[1], forces[2]    # in the body frame (b1, b2, b3)
    I = inertia_matrix
    I_inv = inertia_matrix_inverse

    pose_rate = xp.array([
        # velocity = dPosition (inertial) / dt: body velocity rotated to inertial
        cthe*cpsi*ub + (-cphi * spsi + sphi*sthe*cpsi) * vb + \
            (sphi*spsi+cphi*sthe*cpsi) * wb,
        cthe*spsi * ub + (cphi*cpsi+sphi*sthe*spsi) * vb + \
            (-sphi*cpsi+cphi*sthe*spsi) * wb,
        -sthe * ub + sphi*cthe * vb + cphi*cthe * wb,
        # Acceleration = dVelocity (body frame) / dt
        #           External forces     Gravity             Coriolis effect
        1/mass * (f1)     + g * sthe          + r * vb - q * wb,
        1/mass * (f2)     - g * sphi * cthe   - r * ub + p * wb,
        1/mass * (f3)     - g * cphi * cthe   + q * ub - p * vb,
        # Orientation
        p + (q*sphi + r*cphi) * sthe / cthe,
        q * cphi - r * sphi,
        (q * sphi + r * cphi) / cthe,
    ])
    # Angular rate
    gyro = xp.cross(x[9:12], I @ x[9:12])
    return xp.concatenate((pose_rate, I_inv @ (torques - gyro)))



# The equations above are plain Python against ``xp``. Compile them for numba
# here, under their own names; `multirotor.xp_jax` builds the same source for
# JAX with ``build(SOURCE, NAMES, 'jax', vars(physics))``.
NAMES = (
    '_voltage', '_current', 'motor_electrical', '_acc', 'motor_acc', '_add_power',
    'motor_power', '_empty', '_interpolate', 'battery_ocv', '_ideal', '_loaded',
    '_overloaded', '_delivering', 'battery_terminal', '_unlimited', '_balance',
    '_limited', 'battery_operating_point', '_dead', '_live', 'battery_end_step',
    '_add_torque', 'body_torques', 'forces_torques', 'body_derivative',
)
SOURCE = {name: globals()[name] for name in NAMES}
globals().update(vars(build(SOURCE, NAMES, 'numba', globals())))
