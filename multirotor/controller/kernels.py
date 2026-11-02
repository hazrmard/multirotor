"""
Compiled kernels of the cascaded controller. The classes in
`multirotor.controller.pid` call them; they hold no state of their own. The
arrays they take are the matrices of `multirotor.controller.data.ControllerData`:
a parameter matrix ``K`` and a state matrix ``S``, one column per control channel,
with rows named by the ``PID_*`` constants below. ``lo`` is the first column of
the stage that a kernel advances.

Every stage is the same PID law, `pid_channel`, plus a stage-specific step before
or after it:

    Stage     Kernel          Before the PID law       After the PID law
    --------  --------------  -----------------------  ------------------------
    position  `position_step` leash or square root     rotate to body frame, limit
    velocity  `velocity_step` none                     clip tilt, flip roll sign
    attitude  `attitude_step` optional square root     none
    rate      `rate_step`     Euler rates to body rates  clip, multiply by inertia
    altitude  `altitude_step` none                     clip climb rate
    climb     `climb_step`    none                     tilt-compensated thrust

Call chain::

    position_step ----> leash_distance
                  +---> sqrt_control or pid_channel
    velocity_step ----> pid_channel
    attitude_step ----> sqrt_control or pid_channel
    rate_step --------> pid_channel
    altitude_step ----> pid_channel
    climb_step -------> pid_channel
    pid_stage --------> pid_channel   (a bare `PIDController`)

Stages run in order, because the output of one is the reference of the next.

A call with ``persist=False`` is meant to leave the controller unchanged. A few
kernels change some of its state anyway; those lines are marked "bug". They keep
the behavior of the controller before it was compiled, so that the change could
be checked against the old implementation.
"""
import numpy as np
from numba import njit

from ..coords import (
    body_to_inertial, direction_cosine_matrix, euler_to_angular_rate)


# Rows of the parameter matrix
PID_K_P = 0
"Proportional gain"
PID_K_I = 1
"Integral gain"
PID_K_D = 2
"Derivative gain"
PID_MAX_ERR_I = 3
"Limit on the accumulated error"
PID_LIMIT = 4
"Limit on the output of the stage: max_velocity, max_tilt or max_acceleration"
PID_MAX_ACCELERATION = 5
"Acceleration limit of the square-root controller"
N_PID_PARAMS = 6

# Rows of the state matrix
PID_ERR = 0
PID_ERR_P = 1
PID_ERR_I = 2
PID_ERR_D = 3
PID_REFERENCE = 4
PID_ACTION = 5
N_PID_STATES = 6



@njit
def sqrt_control(err: float, k_p: float, derr_dt2_lim: float, dt: float):
    "Piece-wise P-controller, linear near zero and square root further away."
    # see: https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Math/control.cpp#L387
    correction_rate = 0.
    if derr_dt2_lim <= 0:
        correction_rate = err * k_p
    elif k_p == 0:
        if err > 0:
            correction_rate = np.sqrt(2. * derr_dt2_lim * err)
        if err < 0:
            correction_rate = -np.sqrt(2. * derr_dt2_lim * (-err))
        else:
            correction_rate = 0.
    else:
        linear_dist = derr_dt2_lim / k_p**2
        if err > linear_dist:
            correction_rate = np.sqrt(2. * derr_dt2_lim * (err - (linear_dist / 2.)))
        elif err < -linear_dist:
            correction_rate = -np.sqrt(2. * derr_dt2_lim * (-err - (linear_dist / 2.)))
        else:
            correction_rate = err * k_p
    if dt != 0.:
        abs_val = np.abs(err) / dt
        return min(max(-abs_val, correction_rate), abs_val)
    else:
        return correction_rate



@njit
def pid_channel(
    K: np.ndarray, S: np.ndarray, j: int, err: float, dt: float, persist: bool
) -> float:
    """
    The PID law for channel ``j``, given its error. The integral is the
    trapezoid rule over the last two errors.

    Parameters
    ----------
    K, S : np.ndarray
        Parameter and state matrices.
    j : int
        The channel (column).
    err : float
        The error, reference minus measurement.
    dt : float
        The time since the last step.
    persist : bool
        Whether to store the error terms and the action in ``S``.

    Returns
    -------
    float
        The action.
    """
    previous = S[PID_ERR, j]
    err_p = K[PID_K_P, j] * err
    integral = S[PID_ERR_I, j] + dt * (previous + err) / 2.
    limit = K[PID_MAX_ERR_I, j]
    # bug: the stored integral already includes k_i, so k_i compounds each step
    err_i = K[PID_K_I, j] * min(max(integral, -limit), limit)
    err_d = K[PID_K_D, j] * (err - previous) / dt
    action = err_p + err_i + err_d
    if persist:
        S[PID_ERR_P, j] = err_p
        S[PID_ERR_I, j] = err_i
        S[PID_ERR_D, j] = err_d
        S[PID_ACTION, j] = action
        S[PID_ERR, j] = err
    return action



@njit
def pid_stage(
    K: np.ndarray, S: np.ndarray, lo: int, hi: int, reference: np.ndarray,
    measurement: np.ndarray, dt: float, ref_is_error: bool, persist: bool,
    out: np.ndarray
) -> None:
    """
    Advance channels ``lo`` to ``hi`` by the plain PID law, writing the actions
    to ``out``. If ``ref_is_error`` the reference is the error.
    """
    for j in range(lo, hi):
        i = j - lo
        err = reference[i] if ref_is_error else reference[i] - measurement[i]
        out[i] = pid_channel(K, S, j, err, dt, persist)
        if persist:
            S[PID_REFERENCE, j] = reference[i]



@njit
def leash_distance(
    k_p: float, state: np.ndarray, acceleration: np.ndarray, sim_dt: float
) -> float:
    """
    The largest horizontal position error that the position controller acts on,
    given the vehicle's speed and acceleration. Infinite if the vehicle would
    cover it in one simulation step anyway.

    Parameters
    ----------
    k_p : float
        The position controller's proportional gain.
    state : np.ndarray
        The vehicle's state, shape (12,).
    acceleration : np.ndarray
        The vehicle's body-frame acceleration, shape (3,).
    sim_dt : float
        The simulation time step.
    """
    # https://nrotella.github.io/journal/arducopter-flight-controllers.html
    dcm = direction_cosine_matrix(state[6], state[7], state[8])
    acc = body_to_inertial(acceleration, dcm)
    vel = body_to_inertial(state[3:6], dcm)
    acc_mag = np.sqrt(acc[0]**2 + acc[1]**2)
    vel_mag = np.sqrt(vel[0]**2 + vel[1]**2)
    leash = np.abs(acc_mag / (2 * k_p**2) + vel_mag**2 / (2 * acc_mag + 1e-6))
    # if leash is very small such that vehicle will cover that distance under
    # current velocity, then a leash is not needed. It is more useful for
    # larger errors and bigger speeds.
    return np.inf if leash <= (vel_mag * sim_dt) else leash



@njit
def position_step(
    K: np.ndarray, S: np.ndarray, lo: int, reference: np.ndarray,
    measurement: np.ndarray, state: np.ndarray, acceleration: np.ndarray,
    sim_dt: float, dt: float, leashing: bool, square_root_scaling: bool,
    persist: bool, out: np.ndarray
) -> None:
    """
    Position stage (2 channels): inertial xy-position error to body-frame
    xy-velocity, written to ``out``.

    Parameters
    ----------
    reference, measurement : np.ndarray
        Reference and measured xy-position, shape (2,).
    state : np.ndarray
        The vehicle's state, shape (12,).
    acceleration : np.ndarray
        The vehicle's body-frame acceleration, shape (3,).
    sim_dt : float
        The simulation time step.
    dt : float
        The time since the last step of this stage.
    leashing : bool
        Whether to limit the position error with the leash.
    square_root_scaling : bool
        Whether to replace the P-term with the square-root controller.
    persist : bool
        Whether to store the state.
    """
    if persist:
        S[PID_REFERENCE, lo] = reference[0]
        S[PID_REFERENCE, lo + 1] = reference[1]
    err = reference - measurement
    err_len = np.sqrt(err[0]**2 + err[1]**2)
    velocity = np.zeros(2)
    if leashing:
        leash = leash_distance(K[PID_K_P, lo], state, acceleration, sim_dt)
        err_unit = err / (err_len + 1e-6)
        err_len = min(err_len, leash)
        err = err_unit * err_len
        for i in range(2):
            velocity[i] = K[PID_K_P, lo + i] * err[i]
            if persist:
                S[PID_ERR, lo + i] = err[i]
            # bug: stored even if not persist
            S[PID_ERR_P, lo + i] = velocity[i]
    if err_len > 0. and square_root_scaling:
        for i in range(2):
            velocity[i] = sqrt_control(
                err[i], K[PID_K_P, lo + i], K[PID_MAX_ACCELERATION, lo], dt)
            if persist:
                S[PID_ERR, lo + i] = err[i]
                S[PID_ERR_P, lo + i] = velocity[i]
    else:
        # The plain PID law acts on the error without the leash
        pid_stage(K, S, lo, lo + 2, reference, measurement, dt, False, persist,
                  velocity)
    # convert to body-frame velocity
    cos, sin = np.cos(state[8]), np.sin(state[8])
    body_x = cos * velocity[0] + sin * velocity[1]
    body_y = -sin * velocity[0] + cos * velocity[1]
    magnitude = np.sqrt(body_x**2 + body_y**2)
    speed = min(magnitude, K[PID_LIMIT, lo])
    out[0] = body_x / (magnitude + 1e-6) * speed
    out[1] = body_y / (magnitude + 1e-6) * speed
    if persist:
        S[PID_ACTION, lo] = out[0]
        S[PID_ACTION, lo + 1] = out[1]



@njit
def velocity_step(
    K: np.ndarray, S: np.ndarray, lo: int, reference: np.ndarray,
    measurement: np.ndarray, dt: float, persist: bool, out: np.ndarray
) -> None:
    """
    Velocity stage (2 channels): body-frame xy-velocity error to pitch and roll,
    written to ``out``. Tilt is limited to the stage's limit.
    """
    pid_stage(K, S, lo, lo + 2, reference, measurement, dt, False, persist, out)
    for i in range(2):
        limit = K[PID_LIMIT, lo + i]
        out[i] = min(max(out[i], -limit), limit)
    # +y motion requires negative roll
    out[1] = -out[1]
    if persist:
        S[PID_ACTION, lo] = out[0]
        S[PID_ACTION, lo + 1] = out[1]



@njit
def attitude_step(
    K: np.ndarray, S: np.ndarray, lo: int, reference: np.ndarray,
    measurement: np.ndarray, dt: float, square_root_scaling: bool, persist: bool,
    out: np.ndarray
) -> None:
    """
    Attitude stage (3 channels): orientation error to Euler rates, written to
    ``out``. Optionally replaces the P-term with the square-root controller.
    """
    err = reference - measurement
    if persist:
        for i in range(3):
            S[PID_REFERENCE, lo + i] = reference[i]
    err_len = np.sqrt(err[0]**2 + err[1]**2 + err[2]**2)
    if square_root_scaling and err_len > 0:
        for i in range(3):
            out[i] = sqrt_control(
                err[i], K[PID_K_P, lo + i], K[PID_MAX_ACCELERATION, lo], dt)
            if persist:
                S[PID_ERR, lo + i] = err[i]
                S[PID_ERR_P, lo + i] = out[i]
    else:
        pid_stage(K, S, lo, lo + 3, reference, measurement, dt, False, persist, out)
    if persist:
        for i in range(3):
            S[PID_ACTION, lo + i] = out[i]



@njit
def rate_step(
    K: np.ndarray, S: np.ndarray, lo: int, reference: np.ndarray,
    measurement: np.ndarray, orientation: np.ndarray, inertia: np.ndarray,
    dt: float, persist: bool, out: np.ndarray
) -> None:
    """
    Rate stage (3 channels): Euler rate error to body torques, written to ``out``.

    Parameters
    ----------
    reference, measurement : np.ndarray
        Reference and measured Euler rates, shape (3,).
    orientation : np.ndarray
        The vehicle's roll, pitch and yaw, shape (3,).
    inertia : np.ndarray
        The vehicle's inertia matrix, shape (3, 3).
    """
    ref = euler_to_angular_rate(reference, orientation)
    mea = euler_to_angular_rate(measurement, orientation)
    acc = np.empty(3)
    pid_stage(K, S, lo, lo + 3, ref, mea, dt, False, persist, acc)
    for i in range(3):
        limit = K[PID_LIMIT, lo + i]
        acc[i] = min(max(acc[i], -limit), limit)
        # bug: stored even if not persist, as the angular acceleration
        S[PID_ACTION, lo + i] = acc[i]
    torque = inertia @ acc
    for i in range(3):
        out[i] = torque[i]
        if persist:
            S[PID_ACTION, lo + i] = torque[i]



@njit
def altitude_step(
    K: np.ndarray, S: np.ndarray, lo: int, reference: np.ndarray,
    measurement: np.ndarray, dt: float, persist: bool, out: np.ndarray
) -> None:
    "Altitude stage (1 channel): z-position error to climb rate, written to ``out``."
    pid_stage(K, S, lo, lo + 1, reference, measurement, dt, False, persist, out)
    # bug: stored even if not persist, before the limit
    S[PID_ACTION, lo] = out[0]
    limit = K[PID_LIMIT, lo]
    out[0] = min(max(out[0], -limit), limit)
    if persist:
        S[PID_ACTION, lo] = out[0]



@njit
def climb_step(
    K: np.ndarray, S: np.ndarray, lo: int, reference: np.ndarray,
    measurement: np.ndarray, orientation: np.ndarray, mass: float, weight: float,
    dt: float, persist: bool, out: np.ndarray
) -> None:
    """
    Climb rate stage (1 channel): vertical velocity error to thrust force,
    written to ``out``. The thrust is tilted to compensate for roll and pitch.
    """
    pid_stage(K, S, lo, lo + 1, reference, measurement, dt, False, persist, out)
    out[0] = mass * (out[0] / (np.cos(orientation[0]) * np.cos(orientation[1]))) \
        + weight
    if persist:
        S[PID_ACTION, lo] = out[0]
