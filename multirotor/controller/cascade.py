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
* ``x``, shape ``(N_PID_STATES, n_channels)``: the controller's state. Each column is the
  state of one channel, a `ChannelState`, whose 7 fields are the 7 rows, in the order of
  the ``PID_ERR``, ... constants: the error terms, the reference, the action and the
  integral. So the number of rows is the number of fields, and the number of columns is
  the number of channels. In a whole `Controller` the matrices have one column per
  channel of all six stages, 12 in all, and the ``CASCADE_*`` constants name the first
  column of each stage. For example the attitude stage's ``x`` is ``x[:, 4:7]``, shape
  (7, 3), and ``x[PID_ACTION, 4:7]`` is the action of its roll, pitch and yaw channels.
  (The vehicle's own 12 state variables are called ``body_state``.)
* ``memo``, the outputs of three stages, kept between steps because a stage that does
  not run at a step repeats its last output, and because a stage uses the output of
  the stage before it. It is a `multirotor.tuples.Memo`: the velocity reference that
  the position stage gives the velocity stage (``ref_vel``, shape (2,)), the climb
  stage's ``thrust`` (1,), and the rate stage's ``torques`` (3,).

The layers, from the bottom:

* The scalar PID law, `_pid_scalar`, takes one channel's column of ``pid_params`` and
  ``x``, its reference and its measurement, and gives that channel's new `ChannelState`.
* A channel function (`_pid_channel`, `_altitude_channel`, ...) wraps it for
  `xp.map_rows`: it takes the channel's index ``i`` and a `ChannelInput` that holds the
  whole stage, calls `_pid_scalar` for column ``i``, and may limit or transform the action.
* A stage (`position_stage`, ...) calls `xp.map_rows` to run a channel function on each of
  its channels. `xp.map_rows` is a loop with numba and a ``jax.vmap`` with JAX, and stacks
  the channels' `ChannelState` columns into the stage's new ``x``.
* The cascade, `cascaded_step`, runs the stages in order, each at its own rate.

Every stage is the same PID law plus a stage-specific step before or after it::

    Stage     Function          Before the PID law       After the PID law
    --------  ----------------  -----------------------  ---------------------------
    position  `position_stage`  leash or square root     rotate to body frame, limit
    velocity  `velocity_stage`  none                     clip tilt, flip roll sign
    attitude  `attitude_stage`  optional square root     none
    rate      `rate_stage`      Euler rates to body      clip, multiply by inertia
                                rates
    altitude  `altitude_stage`  none                     clip climb rate
    climb     `climb_stage`     none                     tilt-compensated thrust

Call chain, from the cascade down to the PID law of a channel. A stage calls
`xp.map_rows` on its channel function, and the arrows after a channel function are
what that function calls::

    cascaded_step
      +-> _run_altitude --> altitude_stage --> _altitude_channel --> _pid_scalar
      |                 +-> climb_stage -----> _climb_channel -----> _pid_scalar
      +-> _run_position --> position_stage --> _leashed or _unleashed --> leash_distance
      |                                    +-> _pid_terms ---> _pid_channel --> _pid_scalar
      |                                    +-> _sqrt_terms --> _sqrt_channel --> sqrt_control
      +-> _run_attitude --> velocity_stage --> _velocity_channel --> _pid_scalar
                        +-> attitude_stage --> _pid_terms or _sqrt_terms (as for position)
                        +-> rate_stage ------> _pid_channel --> _pid_scalar
    fleet_step -----> cascaded_step, for each vehicle of a fleet (numba only)
    pid_stage ------> _pid_channel (a bare `multirotor.controller.pid.PIDController`)

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

The values that functions pass to each other are named tuples (`ChannelInput`,
`StageBranchInput`, and so on below), so a function reads ``op.memo`` where it would
otherwise read ``op[1]``. Both backends treat them as they do the tuples of
`multirotor.tuples`: numba keeps them on the stack, and JAX traces them as pytrees, so
they cost nothing.
"""
from typing import NamedTuple, Tuple

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
EPS = 1e-6
"Added to a length that is divided by, so that a zero length gives zero."


# ---------------------------------------------------------------------------
# Named tuples. Each is what one function hands to another, in place of a tuple that
# the receiver would unpack by position. They are cheap on both backends (see above).
# They serve the layers of the controller, from the bottom: the scalar PID law of one
# channel (`_pid_scalar`), a channel function that adapts it to `xp.map_rows`, a stage
# that maps a channel function over its channels, and the cascade that runs the stages.


class ChannelState(NamedTuple):
    """
    The state of one PID channel after a step: one column of a stage's state matrix ``x``.
    The fields are the rows of ``x``, in the order of the ``PID_*`` constants, so a stage's
    ``x`` has ``N_PID_STATES`` (7) rows, one per field, and one column per channel. It is the
    output of the scalar PID law `_pid_scalar` and of every channel function, and
    `xp.map_rows` stacks the columns of a stage's channels into its ``x``.
    """
    err: float
    err_p: float
    err_i: float
    err_d: float
    reference: float
    action: float
    integral: float


class ChannelInput(NamedTuple):
    """
    The input to a channel function, besides the channel's index: the PID parameters and the
    state of the stage, one column per channel, the stage's reference and measurement, one value
    per channel, and the time step.
    """
    pid_params: np.ndarray
    x: np.ndarray
    reference: np.ndarray
    measurement: np.ndarray
    dt: float


class ClimbChannelInput(NamedTuple):
    """
    The input to the climb stage's channel function: a `ChannelInput`, with the vehicle's roll,
    pitch, yaw, mass and weight.
    """
    channel: ChannelInput
    orientation: np.ndarray
    mass: float
    weight: float


class ChannelWithError(NamedTuple):
    """
    The input to `_pid_terms` and `_sqrt_terms`, between which ``xp.cond`` chooses: a
    `ChannelInput` and the error of the stage, already computed. The square-root controller
    takes the error, and the PID law takes the measurement.
    """
    channel: ChannelInput
    err: np.ndarray


class LeashInput(NamedTuple):
    """
    The input to `_leashed` and `_unleashed`, which shorten the position error: the position
    stage's PID parameters, its error, and the vehicle's body state and acceleration.
    """
    pid_params: np.ndarray
    err: np.ndarray
    body_state: np.ndarray
    acceleration: np.ndarray
    sim_dt: float


class CascadeStepInputs(NamedTuple):
    """
    The input to a controller step, besides the controller's state, memo and step count.

    ``pid_params`` are the PID parameters of all the stages, and ``steps`` the periods of the
    stages in whole simulation steps (`multirotor.tuples.Steps`). ``reference`` is the target: x,
    y, z and yaw. ``body_state``, ``acceleration``, ``inertia``, ``mass`` and ``weight`` are the
    vehicle's. ``feed_forward`` is a velocity reference, blended into the position stage by
    ``feedforward_weight`` if ``use_feed_forward``. The last three fields are options.
    """
    pid_params: np.ndarray
    reference: np.ndarray
    body_state: np.ndarray
    acceleration: np.ndarray
    inertia: np.ndarray
    mass: float
    weight: float
    sim_dt: float
    steps: Steps
    feed_forward: np.ndarray
    use_feed_forward: bool
    feedforward_weight: float
    leashing: bool
    square_root_position: bool
    square_root_attitude: bool


class StageBranchInput(NamedTuple):
    """
    The input to a branch of the controller step, which runs a group of stages or skips them:
    the controller's state matrix ``x``, its `multirotor.tuples.Memo`, and the
    `CascadeStepInputs`. A branch returns the new ``(x, memo)``.
    """
    x: np.ndarray
    memo: Memo
    inputs: CascadeStepInputs


def _clip(value: float, limit: float) -> float:
    """Limit a value to a symmetric range."""
    return xp.minimum(xp.maximum(value, -limit), limit)



def _length2(v: np.ndarray) -> float:
    """
    Length of the first two elements of a vector.

    Returns: ``sqrt(v[0]**2 + v[1]**2)``, with a finite gradient at zero.
    """
    return xp.sqrt_nonneg(v[0]**2 + v[1]**2)



# ---------------------------------------------------------------------------
# Control laws. A channel function returns the new state of one channel, as a
# `ChannelState`.

def _pid_scalar(
    pid_params: np.ndarray, x: np.ndarray, i: int, reference: float, measurement: float,
    dt: float
) -> ChannelState:
    """
    The scalar PID law of channel ``i``: it reads the channel's column of the stage's
    ``pid_params`` and ``x``, and gives the new `ChannelState`, with the action not yet
    limited or transformed. The integral is the trapezoid rule over the last two errors,
    and is limited before it is scaled by the integral gain.

    The channel functions below call it with their ``reference[i]`` and ``measurement[i]``
    and adapt the action; they differ from it by taking the whole `ChannelInput`, as
    `xp.map_rows` calls them.
    """
    err = reference - measurement
    previous = x[PID_ERR, i]
    err_p = pid_params[PID_K_P, i] * err
    integral = _clip(
        x[PID_INTEGRAL, i] + dt * (previous + err) / 2., pid_params[PID_MAX_ERR_I, i])
    err_i = pid_params[PID_K_I, i] * integral
    err_d = pid_params[PID_K_D, i] * (err - previous) / dt
    return ChannelState(err, err_p, err_i, err_d, reference, err_p + err_i + err_d, integral)



def _pid_channel(i: int, c: ChannelInput) -> ChannelState:
    """The channel function of the plain PID law: `_pid_scalar` on channel ``i`` of ``c``."""
    return _pid_scalar(c.pid_params, c.x, i, c.reference[i], c.measurement[i], c.dt)



def _altitude_channel(i: int, c: ChannelInput) -> ChannelState:
    """
    The channel function of the altitude stage: the PID law, with its action limited to the
    stage's limit (the climb rate).
    """
    law = _pid_channel(i, c)
    return ChannelState(
        law.err, law.err_p, law.err_i, law.err_d, law.reference,
        _clip(law.action, c.pid_params[PID_LIMIT, i]), law.integral)



def _velocity_channel(i: int, c: ChannelInput) -> ChannelState:
    """
    The channel function of the velocity stage: the PID law, with its action limited to the
    stage's limit (the tilt), and with the sign of the second channel flipped: +y motion
    requires negative roll.
    """
    law = _pid_channel(i, c)
    tilt = _clip(law.action, c.pid_params[PID_LIMIT, i])
    return ChannelState(
        law.err, law.err_p, law.err_i, law.err_d, law.reference,
        xp.select(i == 1, -tilt, tilt), law.integral)



def _climb_channel(i: int, c: ClimbChannelInput) -> ChannelState:
    """
    The channel function of the climb stage: the PID law as a thrust force, tilted to
    compensate for roll and pitch.
    """
    law = _pid_channel(i, c.channel)
    tilt_compensation = xp.cos(c.orientation[0]) * xp.cos(c.orientation[1])
    thrust = c.mass * (law.action / tilt_compensation) + c.weight
    return ChannelState(
        law.err, law.err_p, law.err_i, law.err_d, law.reference, thrust, law.integral)



def _sqrt_rate(derr_dt2_lim: float, distance: float) -> float:
    """The speed that stops within a distance at the acceleration limit."""
    return xp.sqrt_nonneg(2. * derr_dt2_lim * distance)



def sqrt_control(err: float, k_p: float, derr_dt2_lim: float, dt: float) -> float:
    """
    Piece-wise P-controller, linear near zero and square root further away. It
    acts as the P-term of a channel, and gives the correction rate.
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



def _sqrt_channel(i: int, c: ChannelWithError) -> ChannelState:
    """
    The square-root controller as the P-term of a channel. The integral and
    derivative terms are left as they were. The acceleration limit is the stage's, in its
    first column.
    """
    ch = c.channel
    action = sqrt_control(
        c.err[i], ch.pid_params[PID_K_P, i], ch.pid_params[PID_MAX_ACCELERATION, 0], ch.dt)
    return ChannelState(
        c.err[i], action, ch.x[PID_ERR_I, i], ch.x[PID_ERR_D, i], ch.reference[i], action,
        ch.x[PID_INTEGRAL, i])



def _pid_terms(op: ChannelWithError) -> np.ndarray:
    """
    The PID law on every channel of a stage, when there is no square root. It takes the same
    input as `_sqrt_terms`, between which ``xp.cond`` chooses.

    Returns: The new state of the stage, shape (N_PID_STATES, n).
    """
    return xp.map_rows(_pid_channel, op.channel.reference.shape[0], op.channel)



def _sqrt_terms(op: ChannelWithError) -> np.ndarray:
    """
    The square-root controller on every channel of a stage. Its input and result are those
    of `_pid_terms`.
    """
    return xp.map_rows(_sqrt_channel, op.channel.reference.shape[0], op)



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

    ``params``, ``x``: The stage's columns of the parameter and state matrices, shape
      (N_PID_PARAMS, n) and (N_PID_STATES, n).
    ``reference``, ``measurement``: One value per channel, shape (n,). The error is
      reference minus measurement.
    Returns: Shape (n,).
    Returns: The stage's new state, shape (N_PID_STATES, n).
    """
    rows = xp.map_rows(
        _pid_channel, reference.shape[0], ChannelInput(params, x, reference, measurement, dt))
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
        _altitude_channel, 1, ChannelInput(params, x, reference, measurement, dt))
    return rows[PID_ACTION], rows



def climb_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    orientation: np.ndarray, mass: float, weight: float, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Climb rate stage (1 channel): vertical velocity error to thrust force. The thrust is
    tilted to compensate for roll and pitch.

    ``orientation``: The vehicle's roll, pitch and yaw, shape (3,).
    Returns: The thrust, shape (1,).
    Returns: The stage's new state, shape (N_PID_STATES, 1).
    """
    rows = xp.map_rows(
        _climb_channel, 1,
        ClimbChannelInput(ChannelInput(params, x, reference, measurement, dt), orientation, mass, weight))
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
    rows = xp.map_rows(_velocity_channel, 2, ChannelInput(params, x, reference, measurement, dt))
    return rows[PID_ACTION], rows



def attitude_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    dt: float, square_root_scaling: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attitude stage (3 channels): orientation error to Euler rates. Optionally replaces
    the P-term with the square-root controller, where there is an error.

    ``reference``, ``measurement``: The vehicle's target and measured roll, pitch and yaw,
      shape (3,).
    Returns: The Euler rate references, shape (3,).
    Returns: The stage's new state, shape (N_PID_STATES, 3).
    """
    err = reference - measurement
    err_len = xp.sqrt_nonneg(err[0]**2 + err[1]**2 + err[2]**2)
    rows = xp.cond(
        square_root_scaling & (err_len > 0.), _sqrt_terms, _pid_terms,
        ChannelWithError(ChannelInput(params, x, reference, measurement, dt), err))
    return rows[PID_ACTION], rows



def rate_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    orientation: np.ndarray, inertia: np.ndarray, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rate stage (3 channels): Euler rate error to body torques.

    ``reference``, ``measurement``: Reference and measured Euler rates, shape (3,).
    ``orientation``: The vehicle's roll, pitch and yaw, shape (3,).
    ``inertia``: The vehicle's inertia matrix, shape (3, 3). The PID law gives an angular
      acceleration, limited, and the inertia turns it into a torque.
    Returns: The body torques, shape (3,).
    Returns: The stage's new state, shape (N_PID_STATES, 3), with the torques as its action.
    """
    body_reference = euler_to_angular_rate(reference, orientation)
    body_measurement = euler_to_angular_rate(measurement, orientation)
    rows = xp.map_rows(
        _pid_channel, 3, ChannelInput(params, x, body_reference, body_measurement, dt))
    torques = inertia @ _clip(rows[PID_ACTION], params[PID_LIMIT])
    return torques, xp.with_row(rows, PID_ACTION, torques)



def leash_distance(
    k_p: float, body_state: np.ndarray, acceleration: np.ndarray, sim_dt: float
) -> float:
    """
    The largest horizontal position error that the position controller acts on,
    given the vehicle's speed and acceleration. Infinite if the vehicle would
    cover it in one simulation step anyway.

    ``acceleration``: The vehicle's body-frame acceleration, shape (3,).
    Returns: The distance (m).
    """
    # https://nrotella.github.io/journal/arducopter-flight-controllers.html
    dcm = direction_cosine_matrix(body_state[6], body_state[7], body_state[8])
    acc = body_to_inertial(acceleration, dcm)
    vel = body_to_inertial(body_state[3:6], dcm)
    acc_mag = _length2(acc)
    vel_mag = _length2(vel)
    leash = xp.abs(acc_mag / (2 * k_p**2) + vel_mag**2 / (2 * acc_mag + EPS))
    # Skip the leash when the vehicle covers it within one time step at its
    # current speed. The leash matters for large errors and high speeds.
    return xp.select(leash <= vel_mag * sim_dt, INF, leash)



def _leashed(op: LeashInput) -> Tuple[np.ndarray, float]:
    """
    The position error shortened to the leash.

    Returns: The shortened error, shape (2,).
    """
    err_len = _length2(op.err)
    leash = leash_distance(op.pid_params[PID_K_P, 0], op.body_state, op.acceleration, op.sim_dt)
    shortened_len = xp.minimum(err_len, leash)
    return op.err / (err_len + EPS) * shortened_len, shortened_len



def _unleashed(op: LeashInput) -> Tuple[np.ndarray, float]:
    "The position error as it is, and its length. Its input and results are those of `_leashed`."
    return op.err, _length2(op.err)



def position_stage(
    params: np.ndarray, x: np.ndarray, reference: np.ndarray, measurement: np.ndarray,
    body_state: np.ndarray, acceleration: np.ndarray, sim_dt: float, dt: float,
    leashing: bool, square_root_scaling: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Position stage (2 channels): inertial xy-position error to body-frame xy-velocity.

    ``reference``, ``measurement``: Reference and measured xy-position, shape (2,).
    ``acceleration``: The vehicle's body-frame acceleration, shape (3,).
    Returns: The body-frame xy-velocity reference, limited in speed, shape (2,).
    Returns: The stage's new state, shape (N_PID_STATES, 2).
    """
    err, err_len = xp.cond(
        leashing, _leashed, _unleashed,
        LeashInput(params, reference - measurement, body_state, acceleration, sim_dt))
    rows = xp.cond(
        square_root_scaling & (err_len > 0.), _sqrt_terms, _pid_terms,
        ChannelWithError(ChannelInput(params, x, reference, measurement, dt), err))
    # Rotate the inertial velocity into the body frame (by yaw), and limit its speed
    velocity = rows[PID_ACTION]
    cos, sin = xp.cos(body_state[8]), xp.sin(body_state[8])
    body_x = cos * velocity[0] + sin * velocity[1]
    body_y = -sin * velocity[0] + cos * velocity[1]
    magnitude = xp.sqrt_nonneg(body_x**2 + body_y**2)
    speed = xp.minimum(magnitude, params[PID_LIMIT, 0])
    denominator = magnitude + EPS
    body_velocity = xp.array((body_x / denominator * speed, body_y / denominator * speed))
    return body_velocity, xp.with_row(rows, PID_ACTION, body_velocity)



# ---------------------------------------------------------------------------
# The cascade. A branch takes a `StageBranchInput` and returns the new `(x, memo)`.

def _skip(op: StageBranchInput) -> Tuple[np.ndarray, Memo]:
    """
    A stage that does not run at this step keeps its state and repeats its output.

    Returns: The state, unchanged, shape (N_PID_STATES, 12).
    """
    return op.x, op.memo



def _run_altitude(op: StageBranchInput) -> Tuple[np.ndarray, Memo]:
    """
    Altitude and climb stages: target z -> thrust. The input and the results are those of
    `_skip`, with the new state of the two stages and the new thrust.
    """
    i = op.inputs
    dt = i.steps.altitude * i.sim_dt
    orientation = i.body_state[6:9]
    z, climb = CASCADE_ALTITUDE, CASCADE_CLIMB
    climb_rate_reference, x_altitude = altitude_stage(
        i.pid_params[:, z:z + 1], op.x[:, z:z + 1], i.reference[2:3], i.body_state[2:3], dt)
    dcm = direction_cosine_matrix(orientation[0], orientation[1], orientation[2])
    inertial_climb_rate = body_to_inertial(i.body_state[3:6], dcm)[2:]
    thrust, x_climb = climb_stage(
        i.pid_params[:, climb:climb + 1], op.x[:, climb:climb + 1], climb_rate_reference,
        inertial_climb_rate, orientation, i.mass, i.weight, dt)
    x_new = xp.concatenate((op.x[:, :z], x_altitude, x_climb), axis=1)
    return x_new, Memo(op.memo.ref_vel, thrust, op.memo.torques)



def _run_position(op: StageBranchInput) -> Tuple[np.ndarray, Memo]:
    """
    Position stage: target x, y -> velocity reference. The input and the results are those
    of `_skip`, with the new state of the stage and the new velocity reference.
    """
    i = op.inputs
    dt = i.steps.position * i.sim_dt
    start, stop = CASCADE_POSITION, CASCADE_VELOCITY
    pid_velocity, x_position = position_stage(
        i.pid_params[:, start:stop], op.x[:, start:stop], i.reference[:2], i.body_state[:2],
        i.body_state, i.acceleration, i.sim_dt, dt, i.leashing, i.square_root_position)
    # Blend in the feed-forward velocity, if given, and limit the speed
    blend = xp.select(i.use_feed_forward, i.feedforward_weight, 0.)
    ref_vel = _clip(
        blend * i.feed_forward + (1. - blend) * pid_velocity, i.pid_params[PID_LIMIT, start])
    x_new = xp.concatenate((x_position, op.x[:, stop:]), axis=1)
    return x_new, Memo(ref_vel, op.memo.thrust, op.memo.torques)



def _run_attitude(op: StageBranchInput) -> Tuple[np.ndarray, Memo]:
    """
    Velocity, attitude and rate stages: velocity and yaw targets -> torques. The input and
    the results are those of `_skip`, with the new state of the three stages and the
    new torques.
    """
    i = op.inputs
    dt = i.steps.attitude * i.sim_dt
    orientation = i.body_state[6:9]

    pitch_roll, x_velocity = velocity_stage(
        i.pid_params[:, CASCADE_VELOCITY:CASCADE_ATTITUDE],
        op.x[:, CASCADE_VELOCITY:CASCADE_ATTITUDE], op.memo.ref_vel, i.body_state[3:5], dt)
    # The roll angle moves the vehicle along y, and the pitch angle along x
    reference_orientation = xp.array((pitch_roll[1], pitch_roll[0], i.reference[3]))
    euler_rate_reference, x_attitude = attitude_stage(
        i.pid_params[:, CASCADE_ATTITUDE:CASCADE_RATE], op.x[:, CASCADE_ATTITUDE:CASCADE_RATE],
        reference_orientation, orientation, dt, i.square_root_attitude)
    euler_rate = angular_to_euler_rate(i.body_state[9:12], orientation)
    torques, x_rate = rate_stage(
        i.pid_params[:, CASCADE_RATE:CASCADE_ALTITUDE], op.x[:, CASCADE_RATE:CASCADE_ALTITUDE],
        euler_rate_reference, euler_rate, orientation, i.inertia, dt)
    x_new = xp.concatenate(
        (op.x[:, :CASCADE_VELOCITY], x_velocity, x_attitude, x_rate,
         op.x[:, CASCADE_ALTITUDE:]), axis=1)
    return x_new, Memo(op.memo.ref_vel, op.memo.thrust, torques)



def cascaded_step(
    x: np.ndarray, memo: Memo, inputs: CascadeStepInputs, n: int
) -> Tuple[np.ndarray, Memo, np.ndarray]:
    """
    One step of the whole cascade of a `Controller`: the six stages in order, each at
    its own rate. Altitude and climb run every ``inputs.steps.altitude`` steps, position
    every ``inputs.steps.position``, and velocity, attitude and rate every
    ``inputs.steps.attitude``. In between, the last outputs are repeated.

    ``x``: State of the stages, shape (N_PID_STATES, 12). The ``CASCADE_*`` constants name
      the first column of each stage.
    Returns: The state after the step, shape (N_PID_STATES, 12). Keep it for the next step,
      or drop it to leave the controller as it was.
    Returns: Thrust and the three torques, shape (4,).
    """
    steps = inputs.steps
    x, memo = xp.cond(n % steps.altitude == 0, _run_altitude, _skip, StageBranchInput(x, memo, inputs))
    x, memo = xp.cond(n % steps.position == 0, _run_position, _skip, StageBranchInput(x, memo, inputs))
    x, memo = xp.cond(n % steps.attitude == 0, _run_attitude, _skip, StageBranchInput(x, memo, inputs))
    return x, memo, xp.concatenate((memo.thrust, memo.torques))



NAMES = (
    '_clip', '_pid_scalar', '_pid_channel', '_altitude_channel', '_velocity_channel',
    '_climb_channel', '_length2', '_sqrt_rate', 'sqrt_control', '_sqrt_channel', '_pid_terms', '_sqrt_terms',
    'pid_stage', 'altitude_stage', 'climb_stage', 'velocity_stage', 'attitude_stage',
    'rate_stage', 'leash_distance', '_leashed', '_unleashed', 'position_stage', '_skip',
    '_run_altitude', '_run_position', '_run_attitude', 'cascaded_step')
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

    Returns: The state of every vehicle after the step, shape (K, N_PID_STATES, 12).
    Returns: Thrust and torques of every vehicle, shape (K, 4).
    """
    x_new = np.empty_like(x)
    ref_vel, thrust, torques = (
        np.empty_like(memo.ref_vel), np.empty_like(memo.thrust), np.empty_like(memo.torques))
    action = np.empty((params.shape[0], 4))
    for k in range(params.shape[0]):
        inputs = CascadeStepInputs(
            params[k], reference[k], body_state[k], acceleration[k], inertia[k], mass[k],
            weight[k], sim_dt, Steps(steps_p, steps_a, steps_z), feed_forward[k],
            use_feed_forward, feedforward_weight[k], leashing, square_root_position,
            square_root_attitude)
        x_new[k], memo_k, action[k] = cascaded_step(
            x[k], Memo(memo.ref_vel[k], memo.thrust[k], memo.torques[k]), inputs, n)
        ref_vel[k], thrust[k], torques[k] = memo_k.ref_vel, memo_k.thrust, memo_k.torques
    return x_new, Memo(ref_vel, thrust, torques), action
