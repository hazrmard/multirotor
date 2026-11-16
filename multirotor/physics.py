"""
Equations of the vehicle's subsystems: propeller thrust and drag, motor
electrical and mechanical behavior, battery voltage, and the rigid body's
response to forces and torques. Each function computes one quantity from its
inputs. `multirotor.dynamics` combines them into the right-hand sides that the
integrator advances.

Which function calls which, for following the order of execution. An arrow goes
from a caller to the function it calls::

    Motors
        motor_acc ----------------> motor_electrical
        motor_power --------------> motor_electrical

    Battery
        battery_operating_point --> motor_power --> motor_electrical
                                +-> battery_terminal
                                +-> bisect
        battery_end_step ---------> battery_ocv
                                +-> battery_operating_point

    Body
        forces_torques -----------> body_torques
        body_derivative

    Blade-element thrust (numpy and scipy, not JAX)
        thrusts_blade_element_model --> thrust_blade_element_model
                                        --> thrustEqn (passed to fsolve)

Everything but the blade-element thrust is written for JAX: the functions take and
return JAX arrays, can be compiled with ``jax.jit``, mapped over vehicles with
``jax.vmap`` and differentiated. How they are written:

* A quantity of every motor or propeller is an array expression, shape (n,).
* A branch between two cheap values is a ``jnp.where``: both are computed, and one is
  kept.
* A branch that skips real work (the bisection in `battery_operating_point`, a dead
  battery in `battery_end_step`) is a ``lax.cond`` between two local functions, which
  read the enclosing function's variables. Under ``jax.vmap`` it also computes both.

The named tuples (`Terminal`, `OperatingPoint`, `BatteryEnd`) are results that callers
read by field.
"""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy.optimize import fsolve

Array = jax.Array



# Row indices of the parameter matrices in `multirotor.data`. Each matrix has
# one column per propeller. The equations index rows with these constants.

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



class Terminal(NamedTuple):
    """The output of `battery_terminal`: the terminal voltage, and whether the battery can
    deliver the power."""
    voltage: Array
    ok: Array


class OperatingPoint(NamedTuple):
    """The output of `battery_operating_point`: the terminal voltage, the power the motors
    draw, and whether the battery can deliver it."""
    terminal_voltage: Array
    power: Array
    ok: Array


class BatteryEnd(NamedTuple):
    """The output of `battery_end_step`: the charge drawn, the battery current, the terminal
    voltage, whether the battery has cut off, whether it was overloaded, and the motors'
    voltage limit for the next step."""
    charge_drawn: Array
    current: Array
    voltage: Array
    cutoff: Array
    overload: Array
    voltage_limit: Array


def sqrt_nonneg(x: Array) -> Array:
    """Square root that is zero for negative numbers, with a finite gradient."""
    positive = x > 0.
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.)), 0.)


def bisect(f: Callable[[Array], Array], low: Array, high: Array, tol: Array) -> Array:
    """
    Root of an increasing function on ``[low, high]``. The derivative comes from the
    implicit function theorem, not from differentiating the iterations.
    """
    def solve(fn: Callable[[Array], Array], x0: Array) -> Array:
        def halve(bounds: Tuple[Array, Array]) -> Tuple[Array, Array]:
            lo, hi = bounds
            middle = 0.5 * (lo + hi)
            below = fn(middle) < 0.
            return jnp.where(below, middle, lo), jnp.where(below, hi, middle)
        lo, hi = lax.while_loop(lambda b: b[1] - b[0] > tol, halve, (low, high))
        return 0.5 * (lo + hi)
    return lax.custom_root(f, 0.5 * (low + high), solve, lambda g, y: y / g(1.))


# ---------------------------------------------------------------------------
# Motors

def motor_electrical(
    speed: Array, u: Array, max_voltage: Array, motor_params: Array
) -> Tuple[Array, Array]:
    """
    Voltage (V) and current (A) of each motor, shape (n,). The voltage is
    ``speed_voltage_scaling * u``, clipped to ``[0, max_voltage]``. The current responds
    instantly to the voltage and back-EMF; the motors' inductance is neglected.
    """
    p = motor_params
    voltage = jnp.clip(p[MOTOR_SPEED_VOLTAGE_SCALING] * u, 0., max_voltage)
    back_emf = p[MOTOR_K_EMF] * speed
    current = jnp.clip((voltage - back_emf) / p[MOTOR_RESISTANCE], 0., p[MOTOR_MAX_CURRENT])
    return voltage, current


def motor_acc(speed: Array, u: Array, max_voltage: Array, motor_params: Array) -> Array:
    """Angular acceleration (rad/s^2) of each motor: electrical torque minus viscous
    friction and propeller drag, divided by the moment of inertia."""
    p = motor_params
    _, current = motor_electrical(speed, u, max_voltage, p)
    torque = p[MOTOR_K_TORQUE] * current
    friction = p[MOTOR_K_DF] * speed
    drag = p[MOTOR_K_DRAG] * speed**2
    return (torque - friction - drag) / p[MOTOR_MOMENT_OF_INERTIA]


def motor_power(speed: Array, u: Array, max_voltage: Array, motor_params: Array) -> Array:
    """Total electrical power (W) drawn by the motors."""
    voltage, current = motor_electrical(speed, u, max_voltage, motor_params)
    return jnp.sum(voltage * current)


# ---------------------------------------------------------------------------
# Battery

def battery_ocv(soc: Array, ocv_soc: Array, ocv_voltage: Array) -> Array:
    """Open-circuit voltage (V) at a state of charge, interpolated from a table. An empty
    battery (``soc <= 0``) has no voltage."""
    return jnp.where(soc <= 0., 0., jnp.interp(soc, ocv_soc, ocv_voltage))


def battery_terminal(
    power: Array, open_circuit_voltage: Array, internal_resistance: Array
) -> Terminal:
    """
    Terminal voltage of a battery delivering ``power``, with lossless speed controllers.

    ``V = V_oc - R * I`` and ``I = P / V`` give ``V = (V_oc + sqrt(V_oc**2 - 4 R P)) / 2``.
    If the battery can't deliver the power (``V_oc**2 < 4 R P``), the voltage is
    ``V_oc / 2``, where it delivers its maximum power.
    """
    v_oc = open_circuit_voltage
    discriminant = v_oc**2 - 4. * internal_resistance * power
    ideal = internal_resistance == 0.
    overloaded = discriminant < 0.
    delivering = 0.5 * (v_oc + sqrt_nonneg(discriminant))
    voltage = jnp.where(ideal, v_oc, jnp.where(overloaded, 0.5 * v_oc, delivering))
    return Terminal(voltage, jnp.logical_or(ideal, jnp.logical_not(overloaded)))


def battery_operating_point(
    speed: Array, u: Array, open_circuit_voltage: Array, internal_resistance: Array,
    motor_params: Array
) -> OperatingPoint:
    """
    Terminal voltage and power at which the battery and the motors agree: the terminal
    voltage limits the motors' voltage, and the power the motors draw sets the terminal
    voltage.

    If no motor's voltage reaches the terminal voltage, the power does not depend on it.
    Otherwise the terminal voltage is found by bisection on ``[0, V_oc]``: a higher
    terminal voltage lets the motors draw more power, which lowers it, so there is exactly
    one solution.
    """
    v_oc = open_circuit_voltage

    def at(voltage_limit: Array) -> Tuple[Array, Terminal]:
        """The motors' power under a voltage limit, and the battery's terminal at it."""
        power = motor_power(speed, u, voltage_limit, motor_params)
        return power, battery_terminal(power, v_oc, internal_resistance)

    power, terminal = at(v_oc)
    requested_voltage = motor_params[MOTOR_SPEED_VOLTAGE_SCALING] * u

    def unlimited() -> OperatingPoint:
        return OperatingPoint(terminal.voltage, power, terminal.ok)

    def limited() -> OperatingPoint:
        voltage = bisect(lambda v: v - at(v)[1].voltage, 0. * v_oc, v_oc, 1e-12 * v_oc)
        limited_power, limited_terminal = at(voltage)
        return OperatingPoint(voltage, limited_power, limited_terminal.ok)

    return lax.cond(requested_voltage.max() <= terminal.voltage, unlimited, limited)


def battery_end_step(
    charge: Array, charge_drawn: Array, cutoff: Array, overload: Array,
    motor_speeds: Array, u: Array, motor_params: Array, battery_params: Array,
    ocv_soc: Array, ocv_voltage: Array, dt: float
) -> BatteryEnd:
    """
    The battery's state after one integrated time step. The charge drawn during the step
    is limited to the charge left. A battery that is cut off or empty supplies nothing from
    then on. Otherwise the cutoff latches if the terminal voltage falls below the minimum.
    """
    capacity = battery_params[BATTERY_CAPACITY]
    # The integrator can overshoot the capacity by its tolerance
    charge = jnp.minimum(charge, capacity - charge_drawn)
    charge_drawn = charge_drawn + charge
    soc = 1. - charge_drawn / capacity
    v_oc = battery_ocv(soc, ocv_soc, ocv_voltage)
    cutoff, overload = jnp.asarray(cutoff), jnp.asarray(overload)

    def dead() -> Tuple[Array, Array, Array, Array]:
        return v_oc, cutoff, overload, jnp.zeros_like(v_oc)

    def live() -> Tuple[Array, Array, Array, Array]:
        point = battery_operating_point(
            motor_speeds, u, v_oc, battery_params[BATTERY_INTERNAL_RESISTANCE], motor_params)
        # The cutoff takes effect from the next step
        below_minimum = point.terminal_voltage < battery_params[BATTERY_MIN_VOLTAGE]
        return (
            point.terminal_voltage, cutoff | below_minimum, jnp.logical_not(point.ok),
            point.terminal_voltage)

    voltage, cutoff, overload, voltage_limit = lax.cond(cutoff | (soc <= 0.), dead, live)
    return BatteryEnd(charge_drawn, charge / dt, voltage, cutoff, overload, voltage_limit)


# ---------------------------------------------------------------------------
# Body

def body_torques(
    thrusts: Array, drag_torques: Array, clockwise: Array, positions: Array
) -> Array:
    """
    Total torque (N m) on the three body axes from the propellers, shape (3,). Thrust acts
    along the body z-axis, so ``r x F`` reduces to two terms. Each propeller's drag torque
    acts about the z-axis, opposite to its rotation.
    """
    return jnp.stack((
        jnp.sum(positions[1] * thrusts),
        -jnp.sum(positions[0] * thrusts),
        jnp.sum(clockwise * drag_torques)))


def forces_torques(
    speeds: Array, prop_params: Array, clockwise: Array, positions: Array
) -> Tuple[Array, Array]:
    """Total force (N) and torque (N m) on the body from propellers whose thrust is
    ``k_thrust * speed**2`` and drag torque is ``k_drag * speed**2``."""
    thrusts = prop_params[PROP_K_THRUST] * speeds**2
    drag_torques = prop_params[PROP_K_DRAG] * speeds**2
    forces = jnp.array([0., 0., 1.]) * jnp.sum(thrusts)
    return forces, body_torques(thrusts, drag_torques, clockwise, positions)


def body_derivative(
    forces: Array, torques: Array, x: Array, g: float, mass: float, inertia_matrix: Array,
    inertia_matrix_inverse: Array
) -> Array:
    """Rate of change of the 12 body states under forces and torques in the body frame."""
    ub, vb, wb = x[3], x[4], x[5]          # body-frame velocity
    phi, theta, psi = x[6], x[7], x[8]     # roll, pitch, yaw
    p, q, r = x[9], x[10], x[11]           # body-frame angular rate
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cthe, sthe = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)
    pose_rate = jnp.stack((
        # Inertial velocity: the body velocity rotated to the inertial frame
        cthe*cpsi*ub + (-cphi*spsi + sphi*sthe*cpsi)*vb + (sphi*spsi + cphi*sthe*cpsi)*wb,
        cthe*spsi*ub + (cphi*cpsi + sphi*sthe*spsi)*vb + (-sphi*cpsi + cphi*sthe*spsi)*wb,
        -sthe*ub + sphi*cthe*vb + cphi*cthe*wb,
        # Body acceleration: external force, gravity, Coriolis effect
        forces[0] / mass + g*sthe + r*vb - q*wb,
        forces[1] / mass - g*sphi*cthe - r*ub + p*wb,
        forces[2] / mass - g*cphi*cthe + q*ub - p*vb,
        # Euler rates
        p + (q*sphi + r*cphi) * sthe / cthe,
        q*cphi - r*sphi,
        (q*sphi + r*cphi) / cthe))
    rate = x[9:12]
    gyro = jnp.cross(rate, inertia_matrix @ rate)
    return jnp.concatenate((pose_rate, inertia_matrix_inverse @ (torques - gyro)))
