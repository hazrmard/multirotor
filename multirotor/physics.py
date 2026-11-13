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

Why some functions are small, and why there are named tuples in this module. The
equations are written once and built for numba or JAX (`multirotor.lazy`). JAX cannot
trace a Python ``if`` or ``for`` on traced values, so a branch is an ``xp.cond`` and a
loop is an ``xp.map_`` or ``xp.fori_loop`` (see `multirotor.xp_numba`), each taking a
module-level function that has one operand and cannot close over locals. That makes a
function per branch or loop body (``_empty`` and ``_interpolate`` are the two sides of
one ``if``; ``_voltage`` is the body of a loop over the motors), and a named tuple as
its input (``OcvCurve``, ``BatteryLoad``, ``MotorDrive``), built by the parent function
from its locals. Both backends compile them to what the plain ``if`` or ``for`` would
be. Such an input is named for what it holds and for what it is the input of, is local to
the functions that use it, and is not exported. A result that callers use
(``Terminal``, ``OperatingPoint``, ``BatteryEnd``) is named so that both sides of a
branch return the same type. The tuples that cross modules, such as the
vehicle and the controller's parameters, are in `multirotor.tuples`.

`multirotor.simulation` also calls some of these directly, between solver steps:
``motor_electrical`` for each motor's voltage and current,
``battery_end_step`` and ``battery_ocv`` for the battery's state,
``body_torques`` and ``thrusts_blade_element_model`` in
`Multirotor.get_forces_torques`, and ``thrust_blade_element_model`` in
`Propeller.thrust`.
"""
from typing import Iterable, NamedTuple, Tuple

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
def thrustEqn(vi: float, *prop_params: float) -> float:
    """
    Residual of the blade-element momentum equation, which ``fsolve`` drives to zero to find
    the induced velocity.

    ``vi``: The induced velocity (m/s).
    Returns: Efficiency-scaled mass flow times velocity change, minus the blade-element
      thrust (N).
    """
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
    prop_speed: float, airstream_velocity: np.ndarray, R: float, A: float, rho: float,
    a: float, b: float, c: float, eta: float, theta0: float, theta1: float
) -> float:
    """
    Thrust of one propeller from the blade-element model: the induced velocity is found
    numerically, then the thrust follows from momentum theory. Not compiled.

    ``prop_speed``: The propeller's speed (rad/s).
    ``airstream_velocity``: The airstream velocity at the propeller, shape (3,).
    ``R``, ``A``: The propeller's radius (m) and disc area (m^2).
    ``rho``: Air density (kg/m^3).
    ``c``: Chord of the blades (m).
    ``theta0``, ``theta1``: Pitch of the blades at the root and its twist (rad).
    Returns: The thrust (N).
    """
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

    ``speeds``: Propeller speeds (rad/s), shape (n,).
    ``airstream_velocities``: Airstream velocity at each propeller, shape (3, n).
    ``prop_params``: The `PropellerData.params` matrix, shape (N_PROP_PARAMS, n).
    ``rho``: Air density (kg/m^3).
    Returns: The thrust (N) of each propeller, shape (n,).
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



# Named tuples that the kernels below hand to each other, in place of tuples that the
# receiver would unpack by position. Both backends treat them as free (see
# `multirotor.tuples`).


class MotorDrive(NamedTuple):
    """
    The drive of the motors: their speeds, speed signals, the voltage limit and their
    parameters. The input to `motor_electrical` and to the per-motor loop bodies.
    """
    speed: np.ndarray
    u: np.ndarray
    max_voltage: float
    motor_params: np.ndarray


class PropellerForces(NamedTuple):
    """
    The forces of the propellers and their layout: thrusts, drag torques, spin directions and
    positions. The input to the per-propeller loop body `_add_torque`.
    """
    thrusts: np.ndarray
    drag_torques: np.ndarray
    clockwise: np.ndarray
    positions: np.ndarray


class Terminal(NamedTuple):
    """
    The output of `battery_terminal`: the terminal voltage, and whether the battery can
    deliver the power.
    """
    voltage: float
    ok: bool


class OperatingPoint(NamedTuple):
    """
    The output of `battery_operating_point`: the terminal voltage, the power the motors draw,
    and whether the battery can deliver it.
    """
    terminal_voltage: float
    power: float
    ok: bool


class BatteryEnd(NamedTuple):
    """
    The output of `battery_end_step`: the charge drawn, the battery current, the terminal
    voltage, whether the battery has cut off, whether it was overloaded, and the motors'
    voltage limit for the next step.
    """
    charge_drawn: float
    current: float
    voltage: float
    cutoff: bool
    overload: bool
    voltage_limit: float


class OcvCurve(NamedTuple):
    """
    The open-circuit-voltage curve of the battery and the state of charge to read it at. The
    input to `_empty` and `_interpolate`.
    """
    soc: float
    ocv_soc: np.ndarray
    ocv_voltage: np.ndarray


class BatteryLoad(NamedTuple):
    """
    The load on the battery: the power drawn, and the battery's source voltage and resistance.
    The input to `_ideal` and `_loaded`.
    """
    power: float
    open_circuit_voltage: float
    internal_resistance: float


class TerminalQuadratic(NamedTuple):
    """
    The quadratic that gives the terminal voltage: the source voltage and its discriminant.
    The input to `_overloaded` and `_delivering`.
    """
    open_circuit_voltage: float
    discriminant: float


class OperatingPointSearch(NamedTuple):
    """
    The search for the battery's operating point: the operating point if no motor is voltage
    limited, and the motors and the battery that a limited one is solved from. The input to
    `_unlimited`, `_limited` and `_balance`.
    """
    point: OperatingPoint
    speed: np.ndarray
    u: np.ndarray
    open_circuit_voltage: float
    internal_resistance: float
    motor_params: np.ndarray


class BatteryAfterStep(NamedTuple):
    """
    The battery after a step, before its cutoff and overload are updated: its `BatteryEnd` as
    it is for a battery that supplies nothing (``end.voltage`` is the open-circuit voltage and
    ``end.voltage_limit`` is 0), and the motors and battery that a live one is updated from.
    The input to `_dead` and `_live`.
    """
    end: BatteryEnd
    motor_speeds: np.ndarray
    u: np.ndarray
    motor_params: np.ndarray
    battery_params: np.ndarray


def _voltage(i: int, c: MotorDrive) -> float:
    """
    Voltage of motor ``i``.

    Returns: The voltage (V).
    """
    return xp.minimum(
        xp.maximum(c.motor_params[MOTOR_SPEED_VOLTAGE_SCALING, i] * c.u[i], 0.), c.max_voltage)



def _current(i: int, c: MotorDrive) -> float:
    """
    Current of motor ``i``. The arguments are those of `_voltage`.

    Returns: The current (A).
    """
    p = c.motor_params
    back_emf = c.speed[i] * p[MOTOR_K_EMF, i]
    return xp.minimum(
        xp.maximum((_voltage(i, c) - back_emf) / p[MOTOR_RESISTANCE, i], 0.),
        p[MOTOR_MAX_CURRENT, i])



def motor_electrical(
    speed: np.ndarray, u: np.ndarray, max_voltage: float, motor_params: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voltage and current of each motor. The current responds instantly to the
    voltage and back-EMF; the motors' inductance is neglected.

    ``speed``: Motor speeds (rad/s), shape (n,).
    ``u``: Speed signals, shape (n,). The voltage is ``speed_voltage_scaling * u``, clipped
      to ``[0, max_voltage]``.
    ``motor_params``: The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).
    Returns: The voltage (V) and current (A) of each motor.
    """

    n = speed.shape[0]
    consts = MotorDrive(speed, u, max_voltage, motor_params)
    return xp.map_(_voltage, n, consts), xp.map_(_current, n, consts)



def _acc(i: int, c: MotorDrive) -> float:
    """
    Angular acceleration of motor ``i``. The arguments are those of `_voltage`.

    Returns: The angular acceleration (rad/s^2).
    """
    p, speed = c.motor_params, c.speed
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

    ``speed``: Motor speeds (rad/s), shape (n,).
    ``u``: Speed signals, shape (n,).
    ``motor_params``: The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).
    Returns: Angular acceleration (rad/s^2) of each motor, shape (n,).
    """

    n = speed.shape[0]
    return xp.map_(_acc, n, MotorDrive(speed, u, max_voltage, motor_params))



def _add_power(acc: float, i: int, c: MotorDrive) -> float:
    """
    ``acc`` plus the electrical power of motor ``i``.

    Returns: The power (W) of the motors up to ``i``.
    """
    return acc + _voltage(i, c) * _current(i, c)



def motor_power(
    speed: np.ndarray, u: np.ndarray, max_voltage: float, motor_params: np.ndarray
) -> float:
    """Total electrical power (W) drawn by the motors."""

    return xp.fold(
        _add_power, 0., speed.shape[0], MotorDrive(speed, u, max_voltage, motor_params))



def _empty(op: OcvCurve) -> float:
    """The voltage of an empty battery: none."""
    return 0.



def _interpolate(op: OcvCurve) -> float:
    """
    The open-circuit voltage, interpolated from the table.

    Returns: The voltage (V).
    """
    return xp.interp(op.soc, op.ocv_soc, op.ocv_voltage)



def battery_ocv(soc: float, ocv_soc: np.ndarray, ocv_voltage: np.ndarray) -> float:
    """
    Open-circuit voltage of the battery at a state of charge, interpolated
    linearly from a table. Outside the table, the nearest end's voltage is used.
    An empty battery (``soc <= 0``) has no voltage.

    ``ocv_voltage``: Open-circuit voltage (V) at each ``ocv_soc``.
    Returns: The open-circuit voltage (V).
    """

    return xp.cond(soc <= 0., _empty, _interpolate, OcvCurve(soc, ocv_soc, ocv_voltage))



def _ideal(op: BatteryLoad) -> Terminal:
    """A battery without resistance: the terminal voltage is the source voltage."""
    return Terminal(op.open_circuit_voltage, True)



def _loaded(op: BatteryLoad) -> Terminal:
    """A battery with resistance: the larger root of the quadratic, if there is one."""
    discriminant = op.open_circuit_voltage**2 - 4. * op.internal_resistance * op.power
    return xp.cond(
        discriminant < 0., _overloaded, _delivering,
        TerminalQuadratic(op.open_circuit_voltage, discriminant))



def _overloaded(op: TerminalQuadratic) -> Terminal:
    """The battery can't deliver the power: the voltage at its maximum power."""
    return Terminal(0.5 * op.open_circuit_voltage, False)



def _delivering(op: TerminalQuadratic) -> Terminal:
    """The battery delivers the power, at the larger root of the quadratic."""
    return Terminal(0.5 * (op.open_circuit_voltage + xp.sqrt_nonneg(op.discriminant)), True)



def battery_terminal(
    power: float, open_circuit_voltage: float, internal_resistance: float
) -> Terminal:
    """
    Terminal voltage of a battery delivering ``power``, modeling the motors'
    speed controllers as lossless.

    The terminal voltage is ``V = V_oc - R * I`` with current ``I = P / V``, so
    ``V = (V_oc + sqrt(V_oc**2 - 4 R P)) / 2``. If the battery can't deliver
    the power (``V_oc**2 < 4 R P``), the voltage is clamped at ``V_oc / 2``,
    where the battery delivers its maximum power.

    ``power``: The power (W) drawn from the battery.
    ``open_circuit_voltage``: The open-circuit voltage (V).
    ``internal_resistance``: The battery's series resistance (ohm).
    Returns: The terminal voltage (V), and whether the battery can deliver the power.
    """

    return xp.cond(
        internal_resistance == 0., _ideal, _loaded,
        BatteryLoad(power, open_circuit_voltage, internal_resistance))



def _unlimited(op: OperatingPointSearch) -> OperatingPoint:
    """No motor is limited by the terminal voltage: it is as first computed."""
    return op.point



def _balance(v: float, op: OperatingPointSearch) -> float:
    """
    Zero where the motors' voltage limit equals the terminal voltage.

    ``v``: The motors' voltage limit (V).
    """
    power = motor_power(op.speed, op.u, v, op.motor_params)
    terminal = battery_terminal(power, op.open_circuit_voltage, op.internal_resistance)
    return v - terminal.voltage



def _limited(op: OperatingPointSearch) -> OperatingPoint:
    """Some motors are limited by the terminal voltage, found by bisection on [0, V_oc]."""
    v_oc = op.open_circuit_voltage
    terminal_voltage = xp.root(_balance, 0., v_oc, 1e-12 * v_oc, op)
    power = motor_power(op.speed, op.u, terminal_voltage, op.motor_params)
    terminal = battery_terminal(power, v_oc, op.internal_resistance)
    return OperatingPoint(terminal_voltage, power, terminal.ok)



def battery_operating_point(
    speed: np.ndarray, u: np.ndarray, open_circuit_voltage: float,
    internal_resistance: float, motor_params: np.ndarray
) -> OperatingPoint:
    """
    Terminal voltage and power at which the battery and the motors agree: the
    terminal voltage limits the motors' voltage, and the power the motors draw
    sets the terminal voltage.

    If no motor's voltage reaches the terminal voltage, the power does
    not depend on it, and `battery_terminal` gives the answer directly. Otherwise
    the terminal voltage is found by bisection on ``[0, V_oc]``. The bisection
    always converges: a higher terminal voltage lets the motors draw more power,
    which lowers the terminal voltage, so there is exactly one solution.

    ``speed``: Motor speeds (rad/s), shape (n,).
    ``u``: Speed signals, shape (n,).
    ``open_circuit_voltage``: The open-circuit voltage (V).
    ``internal_resistance``: The battery's series resistance (ohm).
    ``motor_params``: The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).
    Returns: The terminal voltage (V), the power (W) the motors draw, and whether the
      battery can deliver that power.
    """

    v_oc = open_circuit_voltage
    requested_voltage = motor_params[MOTOR_SPEED_VOLTAGE_SCALING] * u
    # Power with motor voltages limited only by the open-circuit voltage
    power = motor_power(speed, u, v_oc, motor_params)
    terminal = battery_terminal(power, v_oc, internal_resistance)
    return xp.cond(
        requested_voltage.max() <= terminal.voltage, _unlimited, _limited,
        OperatingPointSearch(
            OperatingPoint(terminal.voltage, power, terminal.ok), speed, u, v_oc,
            internal_resistance, motor_params))



def _dead(op: BatteryAfterStep) -> BatteryEnd:
    """A battery that is cut off or empty supplies nothing."""
    return op.end



def _live(op: BatteryAfterStep) -> BatteryEnd:
    """
    A battery that supplies the motors: its voltage, and whether it cuts off or is overloaded.
    """
    end = op.end
    point = battery_operating_point(
        op.motor_speeds, op.u, end.voltage,
        op.battery_params[BATTERY_INTERNAL_RESISTANCE], op.motor_params)
    # The cutoff takes effect from the next step
    cutoff = end.cutoff | (point.terminal_voltage < op.battery_params[BATTERY_MIN_VOLTAGE])
    return BatteryEnd(
        end.charge_drawn, end.current, point.terminal_voltage, cutoff,
        xp.logical_not(point.ok), point.terminal_voltage)



def battery_end_step(
    charge: float, charge_drawn: float, cutoff: bool, overload: bool,
    motor_speeds: np.ndarray, u: np.ndarray, motor_params: np.ndarray,
    battery_params: np.ndarray, ocv_soc: np.ndarray, ocv_voltage: np.ndarray,
    dt: float
) -> BatteryEnd:
    """
    The battery's state after one integrated time step. The charge drawn during
    the step is limited to the charge left in the battery. A battery that is cut
    off or empty supplies no current from then on. Otherwise the cutoff latches
    if the terminal voltage falls below the minimum.

    ``charge``: Charge (C) drawn during the step, as integrated.
    ``charge_drawn``: Charge (C) drawn before the step.
    ``motor_speeds``: Motor speeds (rad/s) at the end of the step, shape (n,).
    ``u``: Speed signals during the step, shape (n,).
    ``motor_params``: The `MotorData.params` matrix, shape (N_MOTOR_PARAMS, n).
    ``dt``: The time step (s).
    Returns: The charge drawn, the battery current (A, averaged over the step), the terminal
      voltage (V), whether the battery has cut off, whether it was overloaded, and the
      motors' voltage limit (V) for the next step.
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
        BatteryAfterStep(
            BatteryEnd(charge_drawn, current, open_circuit_voltage, cutoff, overload, 0.),
            motor_speeds, u, motor_params, battery_params))



def _add_torque(
    acc: Tuple[float, float, float], i: int, c: PropellerForces
) -> Tuple[float, float, float]:
    """
    ``acc`` plus the torques of propeller ``i``.

    Returns: The torques (N m) of the propellers up to ``i``.
    """
    thrusts, positions = c.thrusts, c.positions
    # tau = r x F. Thrust acts only along the body z-axis, F = (0, 0, thrust),
    # so the cross product reduces to these two terms. Propellers tilted away
    # from the z-axis would need the full cross product.
    return (
        acc[0] + positions[1, i] * thrusts[i],
        acc[1] - positions[0, i] * thrusts[i],
        acc[2] + c.clockwise[i] * c.drag_torques[i])



def body_torques(
    thrusts: np.ndarray, drag_torques: np.ndarray, clockwise: np.ndarray,
    positions: np.ndarray
) -> np.ndarray:
    """
    Total torque on the three body axes (roll, pitch, yaw) due to the
    propellers' loads. Thrust acts along the body z-axis. Each propeller's drag
    torque acts about the z-axis, opposite to its direction of rotation.

    ``thrusts``: Thrust (N) of each propeller, shape (n,).
    ``drag_torques``: Magnitude of the drag torque (N m) of each propeller, shape (n,). For
      a propeller, ``k_drag * speed**2``.
    ``clockwise``: 1 if a propeller spins clockwise, -1 if anti-clockwise, shape (n,).
    ``positions``: Position of each propeller relative to the center of mass, shape (3, n).
    Returns: The torque (N m) on the body, shape (3,).
    """

    # Angular acceleration of propellers is assumed to be negligible, so it
    # contributes no torque.
    n = thrusts.shape[0]
    roll, pitch, yaw = xp.fold(
        _add_torque, (0., 0., 0.), n, PropellerForces(thrusts, drag_torques, clockwise, positions))
    return xp.array((roll, pitch, yaw))



def forces_torques(
    speeds: np.ndarray, prop_params: np.ndarray, clockwise: np.ndarray,
    positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Total force and torque on the body from propellers whose thrust is
    ``k_thrust * speed**2`` and drag torque is ``k_drag * speed**2``.

    ``speeds``: Propeller speeds (rad/s), shape (n,).
    ``prop_params``: The `PropellerData.params` matrix, shape (N_PROP_PARAMS, n).
    ``clockwise``: 1 if a propeller spins clockwise, -1 if anti-clockwise, shape (n,).
    ``positions``: Position of each propeller relative to the center of mass, shape (3, n).
    Returns: The force (N) and torque (N m) on the body, each shape (3,).
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
    """Given forces and torqes, return the rate of change of state."""
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
