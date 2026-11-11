"""
The cascaded controller, as pure functions of its parameters and state. A
`multirotor.controller.pid.Controller` calls `cascaded_step` once per simulation
step, with the matrices of its `multirotor.controller.data.ControllerData`.

Terms used here:

* A *channel* is one scalar control loop: one error in, one action out. It owns
  one column of the two matrices below.
* A *stage* is one controller of the cascade. It has one or more channels, in
  adjacent columns: position has 2 (x, y), attitude has 3 (roll, pitch, yaw),
  altitude has 1.
* ``params``, shape ``(N_PID_PARAMS, n_channels)``: one column per channel. The
  ``PID_K_P``, ... constants of `multirotor.controller.data` name the rows.
* ``x``, shape ``(N_PID_STATES, n_channels)``: the controller's state, which is
  the error terms, the reference and the action of each channel. The ``PID_ERR``,
  ... constants name the rows. In a whole `Controller` the matrices have one column
  per channel of all six stages, and the ``CASCADE_*`` constants name the first
  column of each stage. (The vehicle's own 12 state variables are called
  ``body_state``.)
* ``memo``, the outputs of three stages, kept between steps because a stage that does
  not run at a step repeats its last output, and because a stage uses the output of
  the stage before it. It is a `multirotor.tuples.Memo`: the velocity reference that
  the position stage gives the velocity stage (``ref_vel``, shape (2,)), the climb
  stage's ``thrust`` (1,), and the rate stage's ``torques`` (3,).

Every stage is the same PID law, `_pid_scalar`, plus a stage-specific step before or
after it::

    Stage     Function          Before the PID law       After the PID law
    --------  ----------------  -----------------------  ---------------------------
    position  `position_stage`  leash or square root     rotate to body frame, limit
    velocity  `velocity_stage`  none                     clip tilt, flip roll sign
    attitude  `attitude_stage`  optional square root     none
    rate      `rate_stage`      Euler rates to body      clip, multiply by inertia
                                rates
    altitude  `altitude_stage`  none                     clip climb rate
    climb     `climb_stage`     none                     tilt-compensated thrust

Call chain::

    cascaded_step ---> _run_z ---> altitude_stage, climb_stage
                  +--> _run_p ---> position_stage ---> leash_distance
                  |                                +-> sqrt_control or the PID law
                  +--> _run_a ---> velocity_stage, attitude_stage, rate_stage
    fleet_step -----> cascaded_step, for each vehicle of a fleet (numba only)
    pid_stage ------> the PID law (a bare `multirotor.controller.pid.PIDController`)

The stages run in this order, because the output of one is the reference of the
next. Altitude, position and attitude run at their own rates, a whole number of
simulation steps apart.

Nothing is written in place. A function takes the state and returns the new state;
the caller keeps it, or drops it to compute an action without changing the
controller (``persist=False`` in `multirotor.controller.pid.Controller.step`, which
still keeps the memo, so that ``thrust`` and ``torques`` report the last step).

The functions are written against the array namespace ``xp`` and built for numba
(`multirotor.xp_numba`) at the bottom of this module. `multirotor.rollout` builds
them for JAX (`multirotor.xp_jax`) as well, which differentiates a simulation with
respect to the gains. That is why stages return new matrices (a stage rebuilds ``x``
with `concatenate`), and why the branches are `xp.cond` and `xp.select`, not ``if``.

The values that functions pass to each other are named tuples (`Channel`, `Operand`, and so
on below), so a function reads ``op.memo`` where it would otherwise read ``op[1]``. Both
backends treat them as they do the tuples of `multirotor.tuples`: numba keeps them on the
stack, and JAX traces them as pytrees, so they cost nothing.
"""
from collections import namedtuple
from typing import Tuple

import numpy as np
from numba import njit

from .. import xp_numba as xp
from ..coords import (
    angular_to_euler_rate, body_to_inertial, direction_cosine_matrix,
    euler_to_angular_rate)
from ..lazy import build
from ..tuples import Memo, Steps
from .data import (
    CASCADE_ALTITUDE, CASCADE_ATTITUDE, CASCADE_CLIMB, CASCADE_POSITION,
    CASCADE_RATE, CASCADE_VELOCITY, PID_ACTION, PID_ERR, PID_ERR_D, PID_ERR_I,
    PID_ERR_P, PID_INTEGRAL, PID_K_D, PID_K_I, PID_K_P, PID_LIMIT,
    PID_MAX_ACCELERATION, PID_MAX_ERR_I, PID_REFERENCE)

INF = float('inf')


# ---------------------------------------------------------------------------
# Named tuples. Each is what one function hands to another, in place of a tuple that
# the receiver would unpack by position. They are cheap on both backends (see above).

PidLaw = namedtuple('PidLaw', 'action err err_p err_i err_d integral')
"What `_pid_scalar` returns: the action and the new error terms of a channel."

ChannelState = namedtuple(
    'ChannelState', 'err err_p err_i err_d reference action integral')
"""
The new state of one channel: the seven values of a column of ``x``, in the order of the
``PID_*`` rows. A channel function returns one, and `multirotor.xp_numba.map_rows` stacks
them into the matrix.
"""

Channel = namedtuple('Channel', 'params x reference measurement dt')
"""
What a channel function takes besides the channel's index: the stage's parameters and
state, its reference and measurement (one value per channel), and the time step.
"""

ClimbChannel = namedtuple('ClimbChannel', Channel._fields + ('orientation', 'mass', 'weight'))
"The `Channel` of the climb stage, with the vehicle's roll, pitch, yaw, mass and weight."

SqrtChannel = namedtuple('SqrtChannel', 'params x reference err dt')
"The `Channel` of the square-root controller: it takes the error, not the measurement."

Terms = namedtuple('Terms', 'params x reference measurement err dt')
"""
The operand of `_pid_terms` and `_sqrt_terms`, which `xp.cond` chooses between. Both take
every field, whether they use it or not.
"""

Leash = namedtuple('Leash', 'params err body_state acceleration sim_dt')
"The operand of `_leashed` and `_unleashed`: the position error and the vehicle's motion."

Inputs = namedtuple('Inputs', (
    'params reference body_state acceleration inertia mass weight sim_dt steps feed_forward '
    'use_feed_forward feedforward_weight leashing square_root_position square_root_attitude'))
"""
Everything besides the state and the memo that `cascaded_step` was given, with the same
names as its arguments, except that ``steps`` is a `multirotor.tuples.Steps`.
"""

Operand = namedtuple('Operand', 'x memo inputs')
"""
The operand of a branch of the cascade: the matrix of the stages' state ``x``, the
`multirotor.tuples.Memo`, and the `Inputs`. A branch returns the new ``(x, memo)``.
"""



def _clip(value: float, limit: float) -> float:
    """
    Limit a value to a symmetric range.

    Parameters
    ----------
    value : float
        The value.
    limit : float
        The bound.

    Returns
    -------
    float
        ``value`` limited to ``[-limit, limit]``.
    """
    return xp.minimum(xp.maximum(value, -limit), limit)



# ---------------------------------------------------------------------------
# Control laws. A channel function returns the new state of one channel, as a
# `ChannelState`.

def _pid_scalar(
    params: np.ndarray, x: np.ndarray, i: int, reference: float, measurement: float,
    dt: float
) -> PidLaw:
    """
    The PID law of channel ``i``, given its reference and measurement. The integral
    is the trapezoid rule over the last two errors, and is limited before it is
    scaled by the integral gain.

    Parameters
    ----------
    params, x : np.ndarray
        The stage's parameter and state matrices.
    i : int
        The channel, a column of ``params`` and ``x``.
    reference, measurement : float
        The channel's reference and measurement. The error is reference minus measurement.
    dt : float
        The time since the last step.

    Returns
    -------
    PidLaw
        The action, and the new error, P, I and D terms and integral.
    """
    err = reference - measurement
    previous = x[PID_ERR, i]
    err_p = params[PID_K_P, i] * err
    integral = _clip(
        x[PID_INTEGRAL, i] + dt * (previous + err) / 2., params[PID_MAX_ERR_I, i])
    err_i = params[PID_K_I, i] * integral
    err_d = params[PID_K_D, i] * (err - previous) / dt
    return PidLaw(err_p + err_i + err_d, err, err_p, err_i, err_d, integral)



def _pid_channel(i: int, c: Channel) -> ChannelState:
    """
    The PID law, unchanged.

    Parameters
    ----------
    i : int
        The channel.
    c : Channel
        The stage's parameters, state, reference and measurement, and the time step.

    Returns
    -------
    ChannelState
        The new state of the channel.
    """
    law = _pid_scalar(c.params, c.x, i, c.reference[i], c.measurement[i], c.dt)
    return ChannelState(
        law.err, law.err_p, law.err_i, law.err_d, c.reference[i], law.action, law.integral)



def _altitude_channel(i: int, c: Channel) -> ChannelState:
    """
    The PID law, with its action limited to the stage's limit (the climb rate).
    The arguments and the result are those of `_pid_channel`.
    """
    law = _pid_scalar(c.params, c.x, i, c.reference[i], c.measurement[i], c.dt)
    return ChannelState(
        law.err, law.err_p, law.err_i, law.err_d, c.reference[i],
        _clip(law.action, c.params[PID_LIMIT, i]), law.integral)



def _velocity_channel(i: int, c: Channel) -> ChannelState:
    """
    The PID law, with its action limited to the stage's limit (the tilt), and with
    the sign of the second channel flipped: +y motion requires negative roll.
    The arguments and the result are those of `_pid_channel`.
    """
    law = _pid_scalar(c.params, c.x, i, c.reference[i], c.measurement[i], c.dt)
    tilt = _clip(law.action, c.params[PID_LIMIT, i])
    return ChannelState(
        law.err, law.err_p, law.err_i, law.err_d, c.reference[i],
        xp.select(i == 1, -tilt, tilt), law.integral)



def _climb_channel(i: int, c: ClimbChannel) -> ChannelState:
    """
    The PID law as a thrust force, tilted to compensate for roll and pitch.

    Parameters
    ----------
    i : int
        The channel.
    c : ClimbChannel
        As for `_pid_channel`, with the vehicle's orientation, mass and weight.

    Returns
    -------
    ChannelState
        The new state of the channel. Its action is the thrust.
    """
    law = _pid_scalar(c.params, c.x, i, c.reference[i], c.measurement[i], c.dt)
    tilt_compensation = xp.cos(c.orientation[0]) * xp.cos(c.orientation[1])
    thrust = c.mass * (law.action / tilt_compensation) + c.weight
    return ChannelState(
        law.err, law.err_p, law.err_i, law.err_d, c.reference[i], thrust, law.integral)



def _sqrt_rate(derr_dt2_lim: float, distance: float) -> float:
    """
    The speed that stops within a distance at the acceleration limit.

    Parameters
    ----------
    derr_dt2_lim : float
        The acceleration limit.
    distance : float
        The distance to stop within.

    Returns
    -------
    float
        The speed, or 0 for a negative distance.
    """
    return xp.sqrt_nonneg(2. * derr_dt2_lim * distance)



def sqrt_control(err: float, k_p: float, derr_dt2_lim: float, dt: float) -> float:
    """
    Piece-wise P-controller, linear near zero and square root further away. It
    acts as the P-term of a channel, and gives the correction rate.

    Parameters
    ----------
    err : float
        The error.
    k_p : float
        The proportional gain, which sets the linear region: ``|err|`` below
        ``derr_dt2_lim / k_p**2``. With ``k_p=0`` there is no linear region.
    derr_dt2_lim : float
        The limit on the rate at which the correction rate may change, so on
        acceleration. With no limit (0 or less) the controller is linear.
    dt : float
        The time since the last step. The correction is limited to the error
        divided by ``dt``: the controller does not overshoot the target within
        one step. Not limited if ``dt`` is 0.

    Returns
    -------
    float
        The correction rate.

    Notes
    -----
    It is written with `xp.select`, not ``if``, so that JAX can differentiate it.
    See: https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Math/control.cpp#L387
    """
    # Without a gain the linear region has no width; divide by 1 instead of by 0, and
    # let the zero-gain branch be selected below.
    linear_distance = derr_dt2_lim / xp.select(k_p == 0., 1., k_p)**2
    with_gain = xp.select(
        err > linear_distance, _sqrt_rate(derr_dt2_lim, err - linear_distance / 2.),
        xp.select(
            err < -linear_distance, -_sqrt_rate(derr_dt2_lim, -err - linear_distance / 2.),
            err * k_p))
    without_gain = xp.select(
        err > 0., _sqrt_rate(derr_dt2_lim, err),
        xp.select(err < 0., -_sqrt_rate(derr_dt2_lim, -err), 0.))
    rate = xp.select(
        derr_dt2_lim <= 0., err * k_p, xp.select(k_p == 0., without_gain, with_gain))
    bound = xp.select(dt != 0., xp.abs(err) / xp.select(dt == 0., 1., dt), INF)
    return xp.minimum(xp.maximum(-bound, rate), bound)



def _sqrt_channel(i: int, c: SqrtChannel) -> ChannelState:
    """
    The square-root controller as the P-term of a channel. The integral and
    derivative terms are left as they were. The acceleration limit is the stage's, in its
    first column.

    Parameters
    ----------
    i : int
        The channel.
    c : SqrtChannel
        The stage's parameters and state, its reference, the error (already computed), and
        the time step.

    Returns
    -------
    ChannelState
        The new state of the channel.
    """
    action = sqrt_control(
        c.err[i], c.params[PID_K_P, i], c.params[PID_MAX_ACCELERATION, 0], c.dt)
    return ChannelState(
        c.err[i], action, c.x[PID_ERR_I, i], c.x[PID_ERR_D, i], c.reference[i], action,
        c.x[PID_INTEGRAL, i])



def _pid_terms(op: Terms) -> np.ndarray:
    """
    The PID law on every channel, when no square root. Same operand as `_sqrt_terms`.

    Parameters
    ----------
    op : Terms
        The stage's parameters, state, reference, measurement, error and time step.

    Returns
    -------
    np.ndarray
        The new state of the stage, shape (N_PID_STATES, n).
    """
    return xp.map_rows(
        _pid_channel, op.reference.shape[0],
        Channel(op.params, op.x, op.reference, op.measurement, op.dt))



def _sqrt_terms(op: Terms) -> np.ndarray:
    """
    The square-root controller on every channel. The operand and the result are those of
    `_pid_terms`.
    """
    return xp.map_rows(
        _sqrt_channel, op.reference.shape[0],
        SqrtChannel(op.params, op.x, op.reference, op.err, op.dt))



# ---------------------------------------------------------------------------
# Stages. A stage takes the columns of `params` and `x` that it owns (slices of
# the controller's matrices), and returns its output and its new columns of `x`.

def pid_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The plain PID law on every channel of a stage, as a bare
    `multirotor.controller.pid.PIDController` uses it.

    Parameters
    ----------
    params, x : np.ndarray
        The stage's columns of the parameter and state matrices, shape
        (N_PID_PARAMS, n) and (N_PID_STATES, n).
    reference, measurement : np.ndarray
        One value per channel, shape (n,). The error is reference minus measurement.
    dt : float
        The time since the last step.

    Returns
    -------
    action : np.ndarray
        Shape (n,).
    x : np.ndarray
        The stage's new state, shape (N_PID_STATES, n).
    """
    rows = xp.map_rows(
        _pid_channel, reference.shape[0], Channel(params, x, reference, measurement, dt))
    return rows[PID_ACTION], rows



def altitude_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Altitude stage (1 channel): z-position error to climb rate, limited. The arguments
    and results are those of `pid_stage`.
    """
    rows = xp.map_rows(
        _altitude_channel, 1, Channel(params, x, reference, measurement, dt))
    return rows[PID_ACTION], rows



def climb_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    orientation: np.ndarray, mass: float, weight: float, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Climb rate stage (1 channel): vertical velocity error to thrust force. The thrust is
    tilted to compensate for roll and pitch.

    Parameters
    ----------
    params, x, reference, measurement, dt
        As in `pid_stage`. The reference and measurement are inertial vertical velocities.
    orientation : np.ndarray
        The vehicle's roll, pitch and yaw, shape (3,).
    mass, weight : float
        The vehicle's mass and weight. The thrust that holds it up at a zero error is its weight.

    Returns
    -------
    action : np.ndarray
        The thrust, shape (1,).
    x : np.ndarray
        The stage's new state, shape (N_PID_STATES, 1).
    """
    rows = xp.map_rows(
        _climb_channel, 1,
        ClimbChannel(params, x, reference, measurement, dt, orientation, mass, weight))
    return rows[PID_ACTION], rows



def velocity_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Velocity stage (2 channels): body-frame xy-velocity error to pitch and roll,
    limited to the stage's tilt limit. The arguments and results are those of
    `pid_stage`; the action is ``(pitch, roll)``.
    """
    rows = xp.map_rows(_velocity_channel, 2, Channel(params, x, reference, measurement, dt))
    return rows[PID_ACTION], rows



def attitude_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    dt: float, square_root_scaling: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attitude stage (3 channels): orientation error to Euler rates. Optionally replaces
    the P-term with the square-root controller, where there is an error.

    Parameters
    ----------
    params, x, dt
        As in `pid_stage`.
    reference, measurement : np.ndarray
        The vehicle's target and measured roll, pitch and yaw, shape (3,).
    square_root_scaling : bool
        Whether to replace the P-term with `sqrt_control`.

    Returns
    -------
    action : np.ndarray
        The Euler rate references, shape (3,).
    x : np.ndarray
        The stage's new state, shape (N_PID_STATES, 3).
    """
    err = reference - measurement
    err_len = xp.sqrt_nonneg(err[0]**2 + err[1]**2 + err[2]**2)
    rows = xp.cond(
        square_root_scaling & (err_len > 0.), _sqrt_terms, _pid_terms,
        Terms(params, x, reference, measurement, err, dt))
    return rows[PID_ACTION], rows



def rate_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    orientation: np.ndarray, inertia: np.ndarray, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rate stage (3 channels): Euler rate error to body torques.

    Parameters
    ----------
    params, x, dt
        As in `pid_stage`.
    reference, measurement : np.ndarray
        Reference and measured Euler rates, shape (3,).
    orientation : np.ndarray
        The vehicle's roll, pitch and yaw, shape (3,).
    inertia : np.ndarray
        The vehicle's inertia matrix, shape (3, 3). The PID law gives an angular
        acceleration, limited, and the inertia turns it into a torque.

    Returns
    -------
    action : np.ndarray
        The body torques, shape (3,).
    x : np.ndarray
        The stage's new state, shape (N_PID_STATES, 3), with the torques as its action.
    """
    body_reference = euler_to_angular_rate(reference, orientation)
    body_measurement = euler_to_angular_rate(measurement, orientation)
    rows = xp.map_rows(
        _pid_channel, 3, Channel(params, x, body_reference, body_measurement, dt))
    torques = inertia @ _clip(rows[PID_ACTION], params[PID_LIMIT])
    return torques, xp.with_row(rows, PID_ACTION, torques)



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

    Returns
    -------
    float
        The distance (m).
    """
    # https://nrotella.github.io/journal/arducopter-flight-controllers.html
    dcm = direction_cosine_matrix(body_state[6], body_state[7], body_state[8])
    acc = body_to_inertial(acceleration, dcm)
    vel = body_to_inertial(body_state[3:6], dcm)
    acc_mag = xp.sqrt_nonneg(acc[0]**2 + acc[1]**2)
    vel_mag = xp.sqrt_nonneg(vel[0]**2 + vel[1]**2)
    leash = xp.abs(acc_mag / (2 * k_p**2) + vel_mag**2 / (2 * acc_mag + 1e-6))
    # Skip the leash when the vehicle covers it within one time step at its
    # current speed. The leash matters for large errors and high speeds.
    return xp.select(leash <= vel_mag * sim_dt, INF, leash)



def _leashed(op: Leash) -> Tuple[np.ndarray, float]:
    """
    The position error shortened to the leash.

    Parameters
    ----------
    op : Leash
        The position stage's parameters, the error, and the vehicle's motion.

    Returns
    -------
    err : np.ndarray
        The shortened error, shape (2,).
    length : float
        Its length.
    """
    err_len = xp.sqrt_nonneg(op.err[0]**2 + op.err[1]**2)
    leash = leash_distance(op.params[PID_K_P, 0], op.body_state, op.acceleration, op.sim_dt)
    shortened_len = xp.minimum(err_len, leash)
    return op.err / (err_len + 1e-6) * shortened_len, shortened_len



def _unleashed(op: Leash) -> Tuple[np.ndarray, float]:
    "The position error as it is, and its length. The operand and results are those of `_leashed`."
    return op.err, xp.sqrt_nonneg(op.err[0]**2 + op.err[1]**2)



def position_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    body_state: np.ndarray, acceleration: np.ndarray, sim_dt: float, dt: float,
    leashing: bool, square_root_scaling: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Position stage (2 channels): inertial xy-position error to body-frame xy-velocity.

    Parameters
    ----------
    params, x, dt
        As in `pid_stage`.
    reference, measurement : np.ndarray
        Reference and measured xy-position, shape (2,).
    body_state : np.ndarray
        The vehicle's 12 body state variables.
    acceleration : np.ndarray
        The vehicle's body-frame acceleration, shape (3,).
    sim_dt : float
        The simulation time step. ``dt`` is the time since the last step of this
        stage.
    leashing : bool
        Whether to limit the position error with the leash (`leash_distance`).
    square_root_scaling : bool
        Whether to replace the P-term with the square-root controller (`sqrt_control`),
        which acts on the leashed error. Otherwise the plain PID law acts on the
        error without the leash.

    Returns
    -------
    action : np.ndarray
        The body-frame xy-velocity reference, limited in speed, shape (2,).
    x : np.ndarray
        The stage's new state, shape (N_PID_STATES, 2).
    """
    err, err_len = xp.cond(
        leashing, _leashed, _unleashed,
        Leash(params, reference - measurement, body_state, acceleration, sim_dt))
    rows = xp.cond(
        square_root_scaling & (err_len > 0.), _sqrt_terms, _pid_terms,
        Terms(params, x, reference, measurement, err, dt))
    # Rotate the inertial velocity into the body frame (by yaw), and limit its speed
    velocity = rows[PID_ACTION]
    cos, sin = xp.cos(body_state[8]), xp.sin(body_state[8])
    body_x = cos * velocity[0] + sin * velocity[1]
    body_y = -sin * velocity[0] + cos * velocity[1]
    magnitude = xp.sqrt_nonneg(body_x**2 + body_y**2)
    speed = xp.minimum(magnitude, params[PID_LIMIT, 0])
    body_velocity = xp.array((
        body_x / (magnitude + 1e-6) * speed, body_y / (magnitude + 1e-6) * speed))
    return body_velocity, xp.with_row(rows, PID_ACTION, body_velocity)



# ---------------------------------------------------------------------------
# The cascade. A branch takes an `Operand` and returns the new `(x, memo)`.

def _skip(op: Operand) -> Tuple[np.ndarray, Memo]:
    """
    A stage that does not run at this step keeps its state and repeats its output.

    Parameters
    ----------
    op : Operand
        The state, the memo and the inputs of the step.

    Returns
    -------
    x : np.ndarray
        The state, unchanged, shape (N_PID_STATES, 12).
    memo : Memo
        The memo, unchanged.
    """
    return op.x, op.memo



def _run_z(op: Operand) -> Tuple[np.ndarray, Memo]:
    """
    Altitude and climb stages: target z -> thrust. The operand and the results are those
    of `_skip`, with the new state of the two stages and the new thrust.
    """
    i = op.inputs
    dt = i.steps.altitude * i.sim_dt
    orientation = i.body_state[6:9]
    z, climb = CASCADE_ALTITUDE, CASCADE_CLIMB
    climb_rate_reference, x_altitude = altitude_stage(
        i.params[:, z:z + 1], op.x[:, z:z + 1], i.reference[2:3], i.body_state[2:3], dt)
    dcm = direction_cosine_matrix(orientation[0], orientation[1], orientation[2])
    inertial_climb_rate = body_to_inertial(i.body_state[3:6], dcm)[2:]
    thrust, x_climb = climb_stage(
        i.params[:, climb:climb + 1], op.x[:, climb:climb + 1], climb_rate_reference,
        inertial_climb_rate, orientation, i.mass, i.weight, dt)
    x_new = xp.concatenate((op.x[:, :z], x_altitude, x_climb), axis=1)
    return x_new, Memo(op.memo.ref_vel, thrust, op.memo.torques)



def _run_p(op: Operand) -> Tuple[np.ndarray, Memo]:
    """
    Position stage: target x, y -> velocity reference. The operand and the results are
    those of `_skip`, with the new state of the stage and the new velocity reference.
    """
    i = op.inputs
    dt = i.steps.position * i.sim_dt
    start, stop = CASCADE_POSITION, CASCADE_VELOCITY
    pid_velocity, x_position = position_stage(
        i.params[:, start:stop], op.x[:, start:stop], i.reference[:2], i.body_state[:2],
        i.body_state, i.acceleration, i.sim_dt, dt, i.leashing, i.square_root_position)
    # Blend in the feed-forward velocity, if given, and limit the speed
    blend = xp.select(i.use_feed_forward, i.feedforward_weight, 0.)
    ref_vel = _clip(
        blend * i.feed_forward + (1. - blend) * pid_velocity, i.params[PID_LIMIT, start])
    x_new = xp.concatenate((x_position, op.x[:, stop:]), axis=1)
    return x_new, Memo(ref_vel, op.memo.thrust, op.memo.torques)



def _run_a(op: Operand) -> Tuple[np.ndarray, Memo]:
    """
    Velocity, attitude and rate stages: velocity and yaw targets -> torques. The operand
    and the results are those of `_skip`, with the new state of the three stages and the
    new torques.
    """
    i = op.inputs
    dt = i.steps.attitude * i.sim_dt
    orientation = i.body_state[6:9]

    pitch_roll, x_velocity = velocity_stage(
        i.params[:, CASCADE_VELOCITY:CASCADE_ATTITUDE],
        op.x[:, CASCADE_VELOCITY:CASCADE_ATTITUDE], op.memo.ref_vel, i.body_state[3:5], dt)
    # The roll angle moves the vehicle along y, and the pitch angle along x
    reference_orientation = xp.array((pitch_roll[1], pitch_roll[0], i.reference[3]))
    euler_rate_reference, x_attitude = attitude_stage(
        i.params[:, CASCADE_ATTITUDE:CASCADE_RATE], op.x[:, CASCADE_ATTITUDE:CASCADE_RATE],
        reference_orientation, orientation, dt, i.square_root_attitude)
    euler_rate = angular_to_euler_rate(i.body_state[9:12], orientation)
    torques, x_rate = rate_stage(
        i.params[:, CASCADE_RATE:CASCADE_ALTITUDE], op.x[:, CASCADE_RATE:CASCADE_ALTITUDE],
        euler_rate_reference, euler_rate, orientation, i.inertia, dt)
    x_new = xp.concatenate(
        (op.x[:, :CASCADE_VELOCITY], x_velocity, x_attitude, x_rate,
         op.x[:, CASCADE_ALTITUDE:]), axis=1)
    return x_new, Memo(op.memo.ref_vel, op.memo.thrust, torques)



def cascaded_step(
    params: np.ndarray, x: np.ndarray, memo: Memo, reference: np.ndarray,
    body_state: np.ndarray, acceleration: np.ndarray, inertia: np.ndarray, mass: float,
    weight: float, sim_dt: float, n: int, steps_p: int, steps_a: int, steps_z: int,
    feed_forward: np.ndarray, use_feed_forward: bool, feedforward_weight: float,
    leashing: bool, square_root_position: bool, square_root_attitude: bool
) -> Tuple[np.ndarray, Memo, np.ndarray]:
    """
    One step of the whole cascade of a `Controller`: the six stages in order, each at
    its own rate. Altitude and climb run every ``steps_z`` steps, position every
    ``steps_p``, and velocity, attitude and rate every ``steps_a``. In between, the
    last outputs are repeated.

    Parameters
    ----------
    params, x : np.ndarray
        Parameters and state of the stages, shape (N_PID_PARAMS, 12) and
        (N_PID_STATES, 12). The ``CASCADE_*`` constants name the first column of
        each stage.
    memo : Memo
        The outputs of three stages, as returned by the last step, or zeros for the first.
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
        Whole simulation steps per period of the position, attitude and altitude
        stages.
    feed_forward : np.ndarray
        Velocity reference that is blended into the position stage's output, shape
        (2,). Used if ``use_feed_forward``.
    use_feed_forward : bool
        Whether ``feed_forward`` is given.
    feedforward_weight : float
        The weight of ``feed_forward`` in the blend.
    leashing, square_root_position, square_root_attitude : bool
        Options of the position and attitude stages.

    Returns
    -------
    x : np.ndarray
        The state after the step, shape (N_PID_STATES, 12). Keep it for the next step,
        or drop it to leave the controller as it was.
    memo : Memo
        The memo after the step.
    action : np.ndarray
        Thrust and the three torques, shape (4,).
    """
    inputs = Inputs(
        params, reference, body_state, acceleration, inertia, mass, weight, sim_dt,
        Steps(steps_p, steps_a, steps_z), feed_forward, use_feed_forward, feedforward_weight,
        leashing, square_root_position, square_root_attitude)
    x, memo = xp.cond(n % steps_z == 0, _run_z, _skip, Operand(x, memo, inputs))
    x, memo = xp.cond(n % steps_p == 0, _run_p, _skip, Operand(x, memo, inputs))
    x, memo = xp.cond(n % steps_a == 0, _run_a, _skip, Operand(x, memo, inputs))
    return x, memo, xp.concatenate((memo.thrust, memo.torques))



NAMES = (
    '_clip', '_pid_scalar', '_pid_channel', '_altitude_channel', '_velocity_channel',
    '_climb_channel', '_sqrt_rate', 'sqrt_control', '_sqrt_channel', '_pid_terms', '_sqrt_terms',
    'pid_stage', 'altitude_stage', 'climb_stage', 'velocity_stage', 'attitude_stage',
    'rate_stage', 'leash_distance', '_leashed', '_unleashed', 'position_stage', '_skip',
    '_run_z', '_run_p', '_run_a', 'cascaded_step')
SOURCE = {name: globals()[name] for name in NAMES}
NOT_INLINED = ('cascaded_step',)   # large, and called once per step
globals().update(vars(build(SOURCE, NAMES, 'numba', globals(), NOT_INLINED)))



@njit(cache=xp.CACHE)
def fleet_step(
    params: np.ndarray, x: np.ndarray, memo: Memo, reference: np.ndarray,
    body_state: np.ndarray, acceleration: np.ndarray, inertia: np.ndarray, mass: np.ndarray,
    weight: np.ndarray, sim_dt: float, n: int, steps_p: int, steps_a: int, steps_z: int,
    feed_forward: np.ndarray, use_feed_forward: bool, feedforward_weight: np.ndarray,
    leashing: bool, square_root_position: bool, square_root_attitude: bool
) -> Tuple[np.ndarray, Memo, np.ndarray]:
    """
    One step of the controllers of a fleet: `cascaded_step` for each vehicle, in a
    loop. For numba only; with JAX, map `cascaded_step` over the vehicles with
    ``jax.vmap``.

    Every array has the vehicle as its first axis, ``K`` vehicles in all. Apart from
    that, the arguments and the results are those of `cascaded_step`: the memo holds arrays
    of shape (K, 2), (K, 1) and (K, 3), ``mass``, ``weight`` and ``feedforward_weight``
    have shape (K,), and the action has shape (K, 4). The step counter, the periods and
    the options are the same for all vehicles.

    Returns
    -------
    x : np.ndarray
        The state of every vehicle after the step, shape (K, N_PID_STATES, 12).
    memo : Memo
        The memo of every vehicle after the step.
    action : np.ndarray
        Thrust and torques of every vehicle, shape (K, 4).
    """
    x_new = np.empty_like(x)
    ref_vel, thrust, torques = (
        np.empty_like(memo.ref_vel), np.empty_like(memo.thrust), np.empty_like(memo.torques))
    action = np.empty((params.shape[0], 4))
    for k in range(params.shape[0]):
        x_new[k], memo_k, action[k] = cascaded_step(
            params[k], x[k], Memo(memo.ref_vel[k], memo.thrust[k], memo.torques[k]),
            reference[k], body_state[k], acceleration[k], inertia[k], mass[k], weight[k],
            sim_dt, n, steps_p, steps_a, steps_z, feed_forward[k], use_feed_forward,
            feedforward_weight[k], leashing, square_root_position, square_root_attitude)
        ref_vel[k], thrust[k], torques[k] = memo_k.ref_vel, memo_k.thrust, memo_k.torques
    return x_new, Memo(ref_vel, thrust, torques), action
