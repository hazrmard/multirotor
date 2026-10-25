from typing import Iterable, Tuple

import numpy as np
from numba import njit
from scipy.optimize import fsolve



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



def thrust(
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



def thrusts_physics(
    speeds: np.ndarray, airstream_velocities: np.ndarray, prop_params: np.ndarray,
    rho: float
) -> np.ndarray:
    """
    Thrust of each propeller from the blade-element model, by solving `thrust`
    for each propeller in turn.

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
        thrusts[i] = thrust(
            speeds[i], airstream_velocities[:, i],
            p[PROP_RADIUS, i], p[PROP_AREA, i], rho, p[PROP_LIFT_SLOPE, i],
            p[PROP_BLADES, i], p[PROP_CHORD, i], p[PROP_EFFICIENCY, i],
            p[PROP_THETA0, i], p[PROP_THETA1, i]
        )
    return thrusts



@njit
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
    p = motor_params
    n = speed.shape[0]
    voltage = np.empty(n)
    current = np.empty(n)
    for i in range(n):
        voltage[i] = min(max(p[MOTOR_SPEED_VOLTAGE_SCALING, i] * u[i], 0.), max_voltage)
        back_emf = speed[i] * p[MOTOR_K_EMF, i]
        current[i] = min(
            max((voltage[i] - back_emf) / p[MOTOR_RESISTANCE, i], 0.),
            p[MOTOR_MAX_CURRENT, i]
        )
    return voltage, current



@njit
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
    p = motor_params
    _, current = motor_electrical(speed, u, max_voltage, motor_params)
    acc = np.empty(speed.shape[0])
    for i in range(speed.shape[0]):
        torque = p[MOTOR_K_TORQUE, i] * current[i]
        friction = p[MOTOR_K_DF, i] * speed[i]
        drag = p[MOTOR_K_DRAG, i] * speed[i]**2
        acc[i] = (torque - friction - drag) / p[MOTOR_MOMENT_OF_INERTIA, i]
    return acc



@njit
def body_torques(
    thrusts: np.ndarray, drag_torques: np.ndarray, clockwise: np.ndarray,
    positions: np.ndarray
) -> np.ndarray:
    """
    Torque on the three body axes (roll, pitch, yaw) due to each propeller.
    Thrust acts along the body z-axis. The propeller's drag torque acts about
    the z-axis, opposite to its direction of rotation.

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
        Torque of each propeller, shape (3, n), in the data type of ``thrusts``.
    """
    # TODO: See here
    # https://andrew.gibiansky.com/downloads/pdf/Quadcopter%20Dynamics,%20Simulation,%20and%20Control.pdf
    # Angular acceleration of propellers is assumed to be negligible, so it
    # contributes no torque.
    n = thrusts.shape[0]
    tau = np.empty((3, n), dtype=thrusts.dtype)
    for i in range(n):
        # tau = r x F, with F = (0, 0, thrust)
        tau[0, i] = positions[1, i] * thrusts[i]
        tau[1, i] = -(positions[0, i] * thrusts[i])
        tau[2, i] = clockwise[i] * drag_torques[i]
    return tau



@njit
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
    forces = np.zeros(3)
    forces[2] = thrusts.sum()
    return forces, torques.sum(axis=1)



# @njit
def _body_derivative(
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
    xI = x[0]       # Inertial frame positions
    yI = x[1]
    zI = x[2]
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
    cphi = np.cos(phi);   sphi = np.sin(phi)    # roll
    cthe = np.cos(theta); sthe = np.sin(theta)  # pitch
    cpsi = np.cos(psi);   spsi = np.sin(psi)    # yaw

    f1, f2, f3 = forces # in the body frame (b1, b2, b3)
    t1, t2, t3 = torques
    I = inertia_matrix
    I_inv = inertia_matrix_inverse
    
    # Calculate the derivative of the state matrix using EOM
    xdot = np.zeros_like(x)

    # velocity = dPosition (inertial) / dt (convert body velocity to inertial)
    # Essentially = Rotation matrix (body to inertial) x body velocity
    # dcm = direction_cosine_matrix(roll=phi, pitch=theta, yaw=psi)
    # xdot[0:3] = body_to_inertial(x[3:6], dcm)
    xdot[0] = cthe*cpsi*ub + (-cphi * spsi + sphi*sthe*cpsi) * vb + \
        (sphi*spsi+cphi*sthe*cpsi) * wb  # = xIdot 
    xdot[1] = cthe*spsi * ub + (cphi*cpsi+sphi*sthe*spsi) * vb + \
        (-sphi*cpsi+cphi*sthe*spsi) * wb # = yIdot 
    xdot[2] = (-sthe * ub + sphi*cthe * vb + cphi*cthe * wb) # = zIdot

    #  Acceleration = dVelocity (body frame) / dt
    #           External forces     Gravity             Coriolis effect
    xdot[3] = 1/mass * (f1)     + g * sthe          + r * vb - q * wb  # = udot
    xdot[4] = 1/mass * (f2)     - g * sphi * cthe   - r * ub + p * wb # = vdot
    xdot[5] = 1/mass * (f3)     - g * cphi * cthe   + q * ub - p * vb # = wdot

    # Orientation
    xdot[6] = p + (q*sphi + r*cphi) * sthe / cthe  # = phidot
    xdot[7] = q * cphi - r * sphi  # = thetadot
    xdot[8] = (q * sphi + r * cphi) / cthe  # = psidot

    # Angular rate
    gyro = np.cross(x[9:12], I @ x[9:12])
    xdot[9:12] = I_inv @ (torques - gyro)
    
    return xdot


body_derivative = njit(_body_derivative)
