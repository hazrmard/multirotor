"""
Compiled kernels of the cascaded controller. The classes in
`multirotor.controller.pid` call them; they hold no state of their own.

Terms used here:

* A *channel* is one scalar control loop: one error in, one action out. It owns
  one column of the two matrices below.
* A *stage* is one controller of the cascade. It has one or more channels, in
  adjacent columns: position has 2 (x, y), attitude has 3 (roll, pitch, yaw),
  altitude has 1.
* ``start`` and ``stop`` are slice bounds: a stage owns columns ``start`` up to,
  but excluding, ``stop``. In a `Controller`, the matrices have one column per
  channel of all six stages, so each stage needs its own bounds.

The arrays that the kernels take are the matrices of
`multirotor.controller.data.ControllerData`:

* ``pid_params``, shape ``(N_PID_PARAMS, n_channels)``. One column per channel.
  The ``PID_K_P``, ... constants below name the rows, one per parameter.
* ``x``, shape ``(N_PID_STATES, n_channels)``. The state of the controller: the
  error terms, the reference and the action of each channel. The
  ``PID_ERR``, ... constants below name the rows. ``x`` is the controller's state; the vehicle's
  12 body state variables (position, velocity, orientation, angular rate) are
  called ``body_state``.

Every stage is the same PID law, `pid_channel`, plus a stage-specific step before
or after it::

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
    cascade_step -----> altitude_step, climb_step, position_step, velocity_step,
                        attitude_step, rate_step   (a whole `Controller`)

Stages run in order, because the output of one is the reference of the next.

A call with ``persist=False`` computes the action without changing the state.
"""
import numpy as np
from numba import njit

from ..coords import (
    angular_to_euler_rate, body_to_inertial, direction_cosine_matrix,
    euler_to_angular_rate)


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
PID_INTEGRAL = 6
"Accumulated error, before it is scaled by the integral gain"
N_PID_STATES = 7

# First column of each stage in a `Controller`'s matrices, in the order in which
# `Controller` adopts its stages
CASCADE_POSITION = 0
CASCADE_VELOCITY = 2
CASCADE_ATTITUDE = 4
CASCADE_RATE = 7
CASCADE_ALTITUDE = 10
CASCADE_CLIMB = 11
N_CASCADE_CHANNELS = 12



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
        elif err < 0:
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
    pid_params: np.ndarray, x: np.ndarray, j: int, err: float, dt: float,
    persist: bool
) -> float:
    """
    The PID law for channel ``j``, given its error. The integral is the
    trapezoid rule over the last two errors.

    Parameters
    ----------
    pid_params : np.ndarray
        Parameters, shape (N_PID_PARAMS, n_channels).
    x : np.ndarray
        Controller state, shape (N_PID_STATES, n_channels). Updated if ``persist``.
    j : int
        The channel (column).
    err : float
        The error, reference minus measurement.
    dt : float
        The time since the last step.
    persist : bool
        Whether to store the error terms and the action in ``x``.

    Returns
    -------
    float
        The action.
    """
    previous = x[PID_ERR, j]
    err_p = pid_params[PID_K_P, j] * err
    integral = x[PID_INTEGRAL, j] + dt * (previous + err) / 2.
    limit = pid_params[PID_MAX_ERR_I, j]
    integral = min(max(integral, -limit), limit)
    err_i = pid_params[PID_K_I, j] * integral
    err_d = pid_params[PID_K_D, j] * (err - previous) / dt
    action = err_p + err_i + err_d
    if persist:
        x[PID_ERR_P, j] = err_p
        x[PID_ERR_I, j] = err_i
        x[PID_INTEGRAL, j] = integral
        x[PID_ERR_D, j] = err_d
        x[PID_ACTION, j] = action
        x[PID_ERR, j] = err
    return action



@njit
def pid_stage(
    pid_params: np.ndarray, x: np.ndarray, start: int, stop: int, reference: np.ndarray,
    measurement: np.ndarray, dt: float, ref_is_error: bool, persist: bool,
    out: np.ndarray
) -> None:
    """
    Advance every channel of a stage by the plain PID law, writing the actions
    to ``out``. If ``ref_is_error`` the reference is the error.

    Parameters
    ----------
    pid_params, x : np.ndarray
        Parameters and controller state, as in `pid_channel`.
    start, stop : int
        The columns of the stage: ``start`` up to, not including, ``stop``.
    reference, measurement, out : np.ndarray
        One value per channel of the stage, shape (stop - start,).
    """
    for j in range(start, stop):
        i = j - start
        err = reference[i] if ref_is_error else reference[i] - measurement[i]
        out[i] = pid_channel(pid_params, x, j, err, dt, persist)
        if persist:
            x[PID_REFERENCE, j] = reference[i]



@njit
def leash_distance(
    k_p: float, body_state: np.ndarray, acceleration: np.ndarray, sim_dt: float
) -> float:
    """
    The largest horizontal position error that the position controller acts on,
    given the vehicle's speed and acceleration. Infinite if the vehicle would
    cover it in one simulation step anyway.

    Parameters
    ----------
    k_p : float
        The position controller's proportional gain.
    body_state : np.ndarray
        The vehicle's 12 body state variables.
    acceleration : np.ndarray
        The vehicle's body-frame acceleration, shape (3,).
    sim_dt : float
        The simulation time step.
    """
    # https://nrotella.github.io/journal/arducopter-flight-controllers.html
    dcm = direction_cosine_matrix(body_state[6], body_state[7], body_state[8])
    acc = body_to_inertial(acceleration, dcm)
    vel = body_to_inertial(body_state[3:6], dcm)
    acc_mag = np.sqrt(acc[0]**2 + acc[1]**2)
    vel_mag = np.sqrt(vel[0]**2 + vel[1]**2)
    leash = np.abs(acc_mag / (2 * k_p**2) + vel_mag**2 / (2 * acc_mag + 1e-6))
    # Skip the leash when the vehicle covers it within one time step at its
    # current speed. The leash matters for large errors and high speeds.
    return np.inf if leash <= (vel_mag * sim_dt) else leash



@njit
def position_step(
    pid_params: np.ndarray, x: np.ndarray, start: int, reference: np.ndarray,
    measurement: np.ndarray, body_state: np.ndarray, acceleration: np.ndarray,
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
    body_state : np.ndarray
        The vehicle's 12 body state variables.
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
        x[PID_REFERENCE, start] = reference[0]
        x[PID_REFERENCE, start + 1] = reference[1]
    err = reference - measurement
    err_len = np.sqrt(err[0]**2 + err[1]**2)
    velocity = np.zeros(2)
    if leashing:
        leash = leash_distance(
            pid_params[PID_K_P, start], body_state, acceleration, sim_dt)
        err_unit = err / (err_len + 1e-6)
        err_len = min(err_len, leash)
        err = err_unit * err_len
        for i in range(2):
            velocity[i] = pid_params[PID_K_P, start + i] * err[i]
            if persist:
                x[PID_ERR, start + i] = err[i]
                x[PID_ERR_P, start + i] = velocity[i]
    if err_len > 0. and square_root_scaling:
        for i in range(2):
            velocity[i] = sqrt_control(
                err[i], pid_params[PID_K_P, start + i],
                pid_params[PID_MAX_ACCELERATION, start], dt)
            if persist:
                x[PID_ERR, start + i] = err[i]
                x[PID_ERR_P, start + i] = velocity[i]
    else:
        # The plain PID law acts on the error without the leash
        pid_stage(pid_params, x, start, start + 2, reference, measurement, dt, False,
                  persist, velocity)
    # convert to body-frame velocity
    cos, sin = np.cos(body_state[8]), np.sin(body_state[8])
    body_x = cos * velocity[0] + sin * velocity[1]
    body_y = -sin * velocity[0] + cos * velocity[1]
    magnitude = np.sqrt(body_x**2 + body_y**2)
    speed = min(magnitude, pid_params[PID_LIMIT, start])
    out[0] = body_x / (magnitude + 1e-6) * speed
    out[1] = body_y / (magnitude + 1e-6) * speed
    if persist:
        x[PID_ACTION, start] = out[0]
        x[PID_ACTION, start + 1] = out[1]



@njit
def velocity_step(
    pid_params: np.ndarray, x: np.ndarray, start: int, reference: np.ndarray,
    measurement: np.ndarray, dt: float, persist: bool, out: np.ndarray
) -> None:
    """
    Velocity stage (2 channels): body-frame xy-velocity error to pitch and roll,
    written to ``out``. Tilt is limited to the stage's limit.
    """
    pid_stage(pid_params, x, start, start + 2, reference, measurement, dt, False,
              persist, out)
    for i in range(2):
        limit = pid_params[PID_LIMIT, start + i]
        out[i] = min(max(out[i], -limit), limit)
    # +y motion requires negative roll
    out[1] = -out[1]
    if persist:
        x[PID_ACTION, start] = out[0]
        x[PID_ACTION, start + 1] = out[1]



@njit
def attitude_step(
    pid_params: np.ndarray, x: np.ndarray, start: int, reference: np.ndarray,
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
            x[PID_REFERENCE, start + i] = reference[i]
    err_len = np.sqrt(err[0]**2 + err[1]**2 + err[2]**2)
    if square_root_scaling and err_len > 0:
        for i in range(3):
            out[i] = sqrt_control(
                err[i], pid_params[PID_K_P, start + i],
                pid_params[PID_MAX_ACCELERATION, start], dt)
            if persist:
                x[PID_ERR, start + i] = err[i]
                x[PID_ERR_P, start + i] = out[i]
    else:
        pid_stage(pid_params, x, start, start + 3, reference, measurement, dt, False,
                  persist, out)
    if persist:
        for i in range(3):
            x[PID_ACTION, start + i] = out[i]



@njit
def rate_step(
    pid_params: np.ndarray, x: np.ndarray, start: int, reference: np.ndarray,
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
    pid_stage(pid_params, x, start, start + 3, ref, mea, dt, False, persist, acc)
    for i in range(3):
        limit = pid_params[PID_LIMIT, start + i]
        acc[i] = min(max(acc[i], -limit), limit)
    torque = inertia @ acc
    for i in range(3):
        out[i] = torque[i]
        if persist:
            x[PID_ACTION, start + i] = torque[i]



@njit
def altitude_step(
    pid_params: np.ndarray, x: np.ndarray, start: int, reference: np.ndarray,
    measurement: np.ndarray, dt: float, persist: bool, out: np.ndarray
) -> None:
    "Altitude stage (1 channel): z-position error to climb rate, written to ``out``."
    pid_stage(pid_params, x, start, start + 1, reference, measurement, dt, False,
              persist, out)
    limit = pid_params[PID_LIMIT, start]
    out[0] = min(max(out[0], -limit), limit)
    if persist:
        x[PID_ACTION, start] = out[0]



@njit
def climb_step(
    pid_params: np.ndarray, x: np.ndarray, start: int, reference: np.ndarray,
    measurement: np.ndarray, orientation: np.ndarray, mass: float, weight: float,
    dt: float, persist: bool, out: np.ndarray
) -> None:
    """
    Climb rate stage (1 channel): vertical velocity error to thrust force,
    written to ``out``. The thrust is tilted to compensate for roll and pitch.
    """
    pid_stage(pid_params, x, start, start + 1, reference, measurement, dt, False,
              persist, out)
    out[0] = mass * (out[0] / (np.cos(orientation[0]) * np.cos(orientation[1]))) \
        + weight
    if persist:
        x[PID_ACTION, start] = out[0]



@njit
def cascade_step(
    pid_params: np.ndarray, x: np.ndarray, reference: np.ndarray,
    body_state: np.ndarray, acceleration: np.ndarray, inertia: np.ndarray,
    mass: float, weight: float, sim_dt: float, n: int, steps_p: int,
    steps_a: int, steps_z: int, feed_forward: np.ndarray, use_feed_forward: bool,
    feedforward_weight: float, leashing: bool, square_root_position: bool,
    square_root_attitude: bool, persist: bool, ref_vel: np.ndarray,
    pid_vel: np.ndarray, pitch_roll: np.ndarray, thrust: np.ndarray,
    torques: np.ndarray, action: np.ndarray
) -> None:
    """
    One step of the whole cascade of a `Controller`: the six stages in order,
    each at its own rate. Altitude and climb run every ``steps_z`` steps,
    position every ``steps_p`` and velocity, attitude and rate every
    ``steps_a``. In between, the last outputs are repeated.

    The loop variables ``ref_vel``, ``pid_vel``, ``pitch_roll``, ``thrust`` and
    ``torques`` carry the output of a stage to the stages that run after it, and
    to the steps in between. They are written even if ``persist`` is False.

    Parameters
    ----------
    pid_params, x : np.ndarray
        Parameters and state of the stages, shape (N_PID_PARAMS, 12) and
        (N_PID_STATES, 12). The ``CASCADE_*`` constants name the first column
        of each stage.
    reference : np.ndarray
        The target: x, y, z and yaw, shape (4,).
    body_state : np.ndarray
        The vehicle's 12 body state variables.
    acceleration : np.ndarray
        The vehicle's body-frame acceleration, shape (3,).
    inertia : np.ndarray
        The vehicle's inertia matrix, shape (3, 3).
    mass, weight : float
        The vehicle's mass and weight.
    sim_dt : float
        The simulation time step.
    n : int
        The number of steps taken before this one.
    steps_p, steps_a, steps_z : int
        Whole simulation steps per period of the position, attitude and
        altitude stages.
    feed_forward : np.ndarray
        Velocity reference that is blended into the position stage's output,
        shape (2,). Used if ``use_feed_forward``.
    use_feed_forward : bool
        Whether ``feed_forward`` is given.
    feedforward_weight : float
        The weight of ``feed_forward`` in the blend.
    leashing, square_root_position, square_root_attitude : bool
        Options of the position and attitude stages.
    persist : bool
        Whether the stages store their state.
    ref_vel, pid_vel, pitch_roll : np.ndarray
        Loop variables, shape (2,): the velocity reference of the velocity stage,
        the position stage's output, and the velocity stage's output.
    thrust, torques : np.ndarray
        Loop variables, shape (1,) and (3,): outputs of the climb and rate stages.
    action : np.ndarray
        Output: thrust and the three torques, shape (4,).
    """
    position = body_state[0:3]
    velocity = body_state[3:6]
    orientation = body_state[6:9]
    out1 = np.empty(1)
    out2 = np.empty(2)
    out3 = np.empty(3)

    if n % steps_z == 0:
        dt = steps_z * sim_dt
        altitude_step(
            pid_params, x, CASCADE_ALTITUDE, reference[2:3], position[2:3], dt,
            persist, out1)
        ref_vel_z = out1.copy()
        dcm = direction_cosine_matrix(orientation[0], orientation[1], orientation[2])
        inertial_z = body_to_inertial(velocity, dcm)[2:]
        climb_step(
            pid_params, x, CASCADE_CLIMB, ref_vel_z, inertial_z, orientation, mass,
            weight, dt, persist, thrust)

    if n % steps_p == 0:
        dt = steps_p * sim_dt
        position_step(
            pid_params, x, CASCADE_POSITION, reference[:2], position[:2], body_state,
            acceleration, sim_dt, dt, leashing, square_root_position, persist, pid_vel)
        for i in range(2):
            ref_vel[i] = pid_vel[i]
            if use_feed_forward:
                ref_vel[i] = feedforward_weight * feed_forward[i] \
                    + (1 - feedforward_weight) * pid_vel[i]
            limit = pid_params[PID_LIMIT, CASCADE_POSITION]
            ref_vel[i] = min(max(ref_vel[i], -limit), limit)

    if n % steps_a == 0:
        dt = steps_a * sim_dt
        velocity_step(
            pid_params, x, CASCADE_VELOCITY, ref_vel, velocity[:2], dt, persist,
            pitch_roll)
        ref_orientation = np.empty(3)
        ref_orientation[0] = pitch_roll[1]
        ref_orientation[1] = pitch_roll[0]
        ref_orientation[2] = reference[3]
        attitude_step(
            pid_params, x, CASCADE_ATTITUDE, ref_orientation, orientation, dt,
            square_root_attitude, persist, out3)
        euler_rate = angular_to_euler_rate(body_state[9:12], orientation)
        rate_step(
            pid_params, x, CASCADE_RATE, out3, euler_rate, orientation, inertia, dt,
            persist, torques)

    action[0] = thrust[0]
    for i in range(3):
        action[1 + i] = torques[i]
