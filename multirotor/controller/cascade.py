"""
The cascaded controller, as pure functions of its parameters and state, written for
JAX. A `multirotor.controller.pid.Controller` calls `cascaded_step` once per simulation
step, with the matrices of `multirotor.controller.data`.
`multirotor.rollout` calls it inside a compiled loop, and can differentiate a
simulation with respect to the gains.

Terms used here:

* A *channel* is one scalar control loop: one error in, one action out. It owns
  one column of the two matrices below.
* A *stage* is one controller of the cascade. It has one or more channels, in
  adjacent columns: position has 2 (x, y), attitude has 3 (roll, pitch, yaw),
  altitude has 1.
* ``pid_params``, shape ``(N_PID_PARAMS, n_channels)``: one column per channel. The
  ``PID_K_P``, ... constants of `multirotor.controller.data` name the rows.
* ``x``, shape ``(N_PID_STATES, n_channels)``: the controller's state. Each row is one
  field of `ChannelState`, in the order of the ``PID_ERR``, ... constants: the error
  terms, the reference, the action and the integral. Each column is one channel. In a
  whole `Controller` the matrices have one column per channel of all six stages, 12 in
  all, and the ``CASCADE_*`` constants name the first column of each stage. For example
  the attitude stage's ``x`` is ``x[:, 4:7]``, shape (7, 3), and ``x[PID_ACTION, 4:7]``
  is the action of its roll, pitch and yaw channels. (The vehicle's own 12 state
  variables are called ``body_state``.)
* ``memo``, the outputs of three stages, kept between steps because a stage that does
  not run at a step repeats its last output, and because a stage uses the output of
  the stage before it. It is a `multirotor.tuples.Memo`: the velocity reference that
  the position stage gives the velocity stage (``ref_vel``, shape (2,)), the climb
  stage's ``thrust`` (1,), and the rate stage's ``torques`` (3,).

The layers, from the bottom:

* A control law (`pid_law`, `sqrt_law`) takes a stage's columns of ``pid_params`` and
  ``x``, and gives the stage's new state as a `ChannelState` of arrays, one value per
  channel. `stack_state` turns it into the stage's new ``x``.
* A stage (`altitude_stage`, `position_stage`, ...) applies a law to its channels and
  adapts the action: a limit, a change of frame, a conversion to thrust or torque. It
  returns its output and its new ``x``.
* The cascade (`cascaded_step`) runs the six stages in order, each at its own rate, and
  writes each stage's columns of ``x``.

    cascaded_step --> altitude_stage, climb_stage         (every steps.altitude steps)
                  +-> position_stage --> leash_distance   (every steps.position steps)
                  +-> velocity_stage, attitude_stage, rate_stage
                                                          (every steps.attitude steps)
    pid_stage: the plain law, as a bare `multirotor.controller.pid.PIDController` uses it

The stages run in this order because the output of one is the reference of the next.

Nothing is written in place. A function takes the state and returns the new state;
the caller keeps it, or drops it to compute an action without changing the
controller (``persist=False`` in `multirotor.controller.pid.Controller.step`).

Branches. An option (the leash, the square-root controller) chooses between two cheap
values with ``jnp.where``: both are computed. A group of stages that does not run at
a step is skipped with ``lax.cond``.
"""
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from ..coords import (
    angular_to_euler_rate, body_to_inertial, direction_cosine_matrix,
    euler_to_angular_rate)
from ..physics import sqrt_nonneg
from ..tuples import Memo, Steps
from .data import (
    CASCADE_ALTITUDE, CASCADE_ATTITUDE, CASCADE_CLIMB, CASCADE_POSITION,
    CASCADE_RATE, CASCADE_VELOCITY, PID_ACTION, PID_ERR, PID_ERR_D, PID_ERR_I,
    PID_INTEGRAL, PID_K_D, PID_K_I, PID_K_P, PID_LIMIT, PID_MAX_ACCELERATION,
    PID_MAX_ERR_I)

Array = jax.Array

INF = float('inf')
EPS = 1e-6
"Added to a length that is divided by, so that a zero length gives zero."
LEASH_FLOOR = 1e-6
"""
Below this distance (m) a leash is too short to mean anything, and is ignored. A vehicle that
is level and at rest has a speed and acceleration of rounding-error size (1e-14 or less), and
a leash of that size is decided by the noise, not by the vehicle.
"""


class ChannelState(NamedTuple):
    """
    The state of the channels of a stage after a step. Each field holds one value per
    channel, shape (n,), and is one row of the stage's state matrix ``x``, in the order of
    the ``PID_*`` constants. It is the output of the control laws `pid_law` and `sqrt_law`.
    """
    err: Array
    err_p: Array
    err_i: Array
    err_d: Array
    reference: Array
    action: Array
    integral: Array


class CascadeStepInputs(NamedTuple):
    """
    The input to a controller step, besides the controller's state, memo and step count.

    ``pid_params`` are the PID parameters of all the stages, and ``steps`` the periods of the
    stages in whole simulation steps (`multirotor.tuples.Steps`). ``reference`` is the target: x,
    y, z and yaw. ``body_state``, ``acceleration``, ``inertia``, ``mass`` and ``weight`` are the
    vehicle's. ``feed_forward`` is a velocity reference, blended into the position stage by
    ``feedforward_weight`` if ``use_feed_forward``. The last three fields are options.
    """
    pid_params: Array
    reference: Array
    body_state: Array
    acceleration: Array
    inertia: Array
    mass: Array
    weight: Array
    sim_dt: float
    steps: Steps
    feed_forward: Array
    use_feed_forward: bool
    feedforward_weight: Array
    leashing: bool
    square_root_position: bool
    square_root_attitude: bool



def stack_state(state: ChannelState) -> Array:
    """The state matrix of a stage, shape (N_PID_STATES, n), from its `ChannelState`."""
    return jnp.stack(state)



def _clip(value: Array, limit: Array) -> Array:
    """Limit a value to a symmetric range."""
    return jnp.clip(value, -limit, limit)



def _length2(v: Array) -> Array:
    """``sqrt(v[0]**2 + v[1]**2)``, with a finite gradient at zero."""
    return sqrt_nonneg(v[0]**2 + v[1]**2)



# ---------------------------------------------------------------------------
# Control laws

def pid_law(
    pid_params: Array, x: Array, reference: Array, measurement: Array, dt: float
) -> ChannelState:
    """
    The PID law on the channels of a stage, with the action not yet limited or transformed.
    The integral is the trapezoid rule over the last two errors, and is limited before it
    is scaled by the integral gain.

    ``pid_params``, ``x``: The stage's columns, shape (N_PID_PARAMS, n) and (N_PID_STATES, n).
    ``reference``, ``measurement``: One value per channel, shape (n,).
    """
    err = reference - measurement
    previous = x[PID_ERR]
    err_p = pid_params[PID_K_P] * err
    integral = _clip(x[PID_INTEGRAL] + dt * (previous + err) / 2., pid_params[PID_MAX_ERR_I])
    err_i = pid_params[PID_K_I] * integral
    err_d = pid_params[PID_K_D] * (err - previous) / dt
    return ChannelState(err, err_p, err_i, err_d, reference, err_p + err_i + err_d, integral)



def _sqrt_rate(derr_dt2_lim: Array, distance: Array) -> Array:
    """The speed that stops within a distance at the acceleration limit."""
    return sqrt_nonneg(2. * derr_dt2_lim * distance)



def sqrt_control(err: Array, k_p: Array, derr_dt2_lim: Array, dt: float) -> Array:
    """
    Piece-wise P-controller, linear near zero and square root further away. It
    acts as the P-term of a channel, and gives the correction rate.
    """
    # Without a gain the linear region has no width; divide by 1 instead of by 0, and
    # let the zero-gain branch be selected below.
    linear_distance = derr_dt2_lim / jnp.where(k_p == 0., 1., k_p)**2
    with_gain = jnp.where(
        err > linear_distance, _sqrt_rate(derr_dt2_lim, err - linear_distance / 2.),
        jnp.where(
            err < -linear_distance, -_sqrt_rate(derr_dt2_lim, -err - linear_distance / 2.),
            err * k_p))
    without_gain = jnp.where(
        err > 0., _sqrt_rate(derr_dt2_lim, err),
        jnp.where(err < 0., -_sqrt_rate(derr_dt2_lim, -err), 0.))
    rate = jnp.where(
        derr_dt2_lim <= 0., err * k_p, jnp.where(k_p == 0., without_gain, with_gain))
    bound = jnp.where(dt != 0., jnp.abs(err) / jnp.where(dt == 0., 1., dt), INF)
    return jnp.clip(rate, -bound, bound)



def sqrt_law(
    pid_params: Array, x: Array, reference: Array, err: Array, dt: float
) -> ChannelState:
    """
    The square-root controller as the P-term of the channels of a stage. The integral and
    derivative terms are left as they were. The acceleration limit is the stage's, in its
    first column.

    ``err``: The error of each channel, shape (n,). It need not be reference minus
      measurement: the position stage shortens it to the leash.
    """
    action = sqrt_control(err, pid_params[PID_K_P], pid_params[PID_MAX_ACCELERATION, 0], dt)
    return ChannelState(
        err, action, x[PID_ERR_I], x[PID_ERR_D], reference, action, x[PID_INTEGRAL])



def _pid_or_sqrt(
    use_sqrt: Array, pid_params: Array, x: Array, reference: Array, measurement: Array,
    err: Array, dt: float
) -> ChannelState:
    """The state from `sqrt_law` where ``use_sqrt``, else from `pid_law`."""
    pid = pid_law(pid_params, x, reference, measurement, dt)
    sqrt = sqrt_law(pid_params, x, reference, err, dt)
    return ChannelState(*(jnp.where(use_sqrt, s, p) for s, p in zip(sqrt, pid)))



# ---------------------------------------------------------------------------
# Stages. A stage takes the columns of ``pid_params`` and ``x`` that it owns (slices of
# the controller's matrices), and returns its output and its new columns of ``x``.

def pid_stage(
    pid_params: Array, x: Array, reference: Array, measurement: Array, dt: float
) -> Tuple[Array, Array]:
    """
    The plain PID law on every channel of a stage, as a bare
    `multirotor.controller.pid.PIDController` uses it.

    Returns: The action, shape (n,), and the stage's new state, shape (N_PID_STATES, n).
    """
    state = pid_law(pid_params, x, reference, measurement, dt)
    return state.action, stack_state(state)



def altitude_stage(
    pid_params: Array, x: Array, reference: Array, measurement: Array, dt: float
) -> Tuple[Array, Array]:
    """Altitude stage (1 channel): z-position error to climb rate, limited."""
    state = pid_law(pid_params, x, reference, measurement, dt)
    climb_rate = _clip(state.action, pid_params[PID_LIMIT])
    return climb_rate, stack_state(state._replace(action=climb_rate))



def climb_stage(
    pid_params: Array, x: Array, reference: Array, measurement: Array, orientation: Array,
    mass: Array, weight: Array, dt: float
) -> Tuple[Array, Array]:
    """
    Climb rate stage (1 channel): vertical velocity error to thrust force (N). The thrust
    is tilted to compensate for roll and pitch.

    ``orientation``: The vehicle's roll, pitch and yaw, shape (3,).
    """
    state = pid_law(pid_params, x, reference, measurement, dt)
    tilt_compensation = jnp.cos(orientation[0]) * jnp.cos(orientation[1])
    thrust = mass * (state.action / tilt_compensation) + weight
    return thrust, stack_state(state._replace(action=thrust))



def velocity_stage(
    pid_params: Array, x: Array, reference: Array, measurement: Array, dt: float
) -> Tuple[Array, Array]:
    """
    Velocity stage (2 channels): body-frame xy-velocity error to ``(pitch, roll)``, limited
    to the stage's tilt limit. The sign of roll is flipped: +y motion requires negative
    roll.
    """
    state = pid_law(pid_params, x, reference, measurement, dt)
    pitch_roll = _clip(state.action, pid_params[PID_LIMIT]) * jnp.array([1., -1.])
    return pitch_roll, stack_state(state._replace(action=pitch_roll))



def attitude_stage(
    pid_params: Array, x: Array, reference: Array, measurement: Array, dt: float,
    square_root_scaling: bool
) -> Tuple[Array, Array]:
    """
    Attitude stage (3 channels): orientation error to Euler rates. Optionally replaces
    the P-term with the square-root controller, where there is an error.

    ``reference``, ``measurement``: Target and measured roll, pitch and yaw, shape (3,).
    """
    err = reference - measurement
    has_error = sqrt_nonneg(jnp.sum(err**2)) > 0.
    state = _pid_or_sqrt(
        square_root_scaling & has_error, pid_params, x, reference, measurement, err, dt)
    return state.action, stack_state(state)



def rate_stage(
    pid_params: Array, x: Array, reference: Array, measurement: Array, orientation: Array,
    inertia: Array, dt: float
) -> Tuple[Array, Array]:
    """
    Rate stage (3 channels): Euler rate error to body torques (N m). The PID law gives an
    angular acceleration, limited, and the inertia matrix turns it into a torque.

    ``reference``, ``measurement``: Reference and measured Euler rates, shape (3,).
    """
    state = pid_law(
        pid_params, x, euler_to_angular_rate(reference, orientation),
        euler_to_angular_rate(measurement, orientation), dt)
    torques = inertia @ _clip(state.action, pid_params[PID_LIMIT])
    return torques, stack_state(state._replace(action=torques))



def leash_distance(k_p: Array, body_state: Array, acceleration: Array, sim_dt: float) -> Array:
    """
    The largest horizontal position error (m) that the position controller acts on,
    given the vehicle's speed and acceleration. Infinite if the vehicle would
    cover it in one simulation step anyway, or if it is shorter than `LEASH_FLOOR`
    (a vehicle at rest, whose speed and acceleration are only noise).

    ``acceleration``: The vehicle's body-frame acceleration, shape (3,).
    """
    # https://nrotella.github.io/journal/arducopter-flight-controllers.html
    dcm = direction_cosine_matrix(body_state[6], body_state[7], body_state[8])
    acc_mag = _length2(body_to_inertial(acceleration, dcm))
    vel_mag = _length2(body_to_inertial(body_state[3:6], dcm))
    leash = jnp.abs(acc_mag / (2 * k_p**2) + vel_mag**2 / (2 * acc_mag + EPS))
    # Skip the leash when the vehicle covers it within one time step at its
    # current speed, or when it is too short to be more than noise. The leash matters
    # for large errors and high speeds.
    return jnp.where(leash <= vel_mag * sim_dt + LEASH_FLOOR, INF, leash)



def position_stage(
    pid_params: Array, x: Array, reference: Array, measurement: Array, body_state: Array,
    acceleration: Array, sim_dt: float, dt: float, leashing: bool, square_root_scaling: bool
) -> Tuple[Array, Array]:
    """
    Position stage (2 channels): inertial xy-position error to body-frame xy-velocity,
    limited in speed.

    ``reference``, ``measurement``: Reference and measured xy-position, shape (2,).
    ``acceleration``: The vehicle's body-frame acceleration, shape (3,).
    """
    err = reference - measurement
    err_len = _length2(err)
    # The leash shortens the error that the square-root controller acts on
    leash = leash_distance(pid_params[PID_K_P, 0], body_state, acceleration, sim_dt)
    leashed_len = jnp.where(leashing, jnp.minimum(err_len, leash), err_len)
    leashed_err = jnp.where(leashing, err / (err_len + EPS) * leashed_len, err)
    state = _pid_or_sqrt(
        square_root_scaling & (leashed_len > 0.), pid_params, x, reference, measurement,
        leashed_err, dt)
    # Rotate the inertial velocity into the body frame (by yaw), and limit its speed
    velocity = state.action
    cos, sin = jnp.cos(body_state[8]), jnp.sin(body_state[8])
    body_velocity = jnp.stack((
        cos * velocity[0] + sin * velocity[1], -sin * velocity[0] + cos * velocity[1]))
    magnitude = _length2(body_velocity)
    speed = jnp.minimum(magnitude, pid_params[PID_LIMIT, 0])
    body_velocity = body_velocity / (magnitude + EPS) * speed
    return body_velocity, stack_state(state._replace(action=body_velocity))



# ---------------------------------------------------------------------------
# The cascade

def cascaded_step(
    x: Array, memo: Memo, inputs: CascadeStepInputs, n: Array
) -> Tuple[Array, Memo, Array]:
    """
    One step of the whole cascade of a `Controller`: the six stages in order, each at
    its own rate. Altitude and climb run every ``inputs.steps.altitude`` steps, position
    every ``inputs.steps.position``, and velocity, attitude and rate every
    ``inputs.steps.attitude``. In between, the last outputs are repeated.

    ``x``: State of the stages, shape (N_PID_STATES, 12). The ``CASCADE_*`` constants name
      the first column of each stage.
    ``n``: The number of steps taken before this one.
    Returns: The state after the step (keep it for the next step, or drop it to leave the
      controller as it was), the new memo, and thrust and the three torques, shape (4,).
    """
    i, steps = inputs, inputs.steps
    p, body_state = i.pid_params, i.body_state
    orientation = body_state[6:9]
    position, velocity, attitude, rate, altitude, climb = (
        slice(CASCADE_POSITION, CASCADE_VELOCITY), slice(CASCADE_VELOCITY, CASCADE_ATTITUDE),
        slice(CASCADE_ATTITUDE, CASCADE_RATE), slice(CASCADE_RATE, CASCADE_ALTITUDE),
        slice(CASCADE_ALTITUDE, CASCADE_CLIMB), slice(CASCADE_CLIMB, CASCADE_CLIMB + 1))

    def run_altitude(x: Array, memo: Memo) -> Tuple[Array, Memo]:
        """Altitude and climb stages: target z -> thrust."""
        dt = steps.altitude * i.sim_dt
        climb_rate_reference, x_altitude = altitude_stage(
            p[:, altitude], x[:, altitude], i.reference[2:3], body_state[2:3], dt)
        dcm = direction_cosine_matrix(orientation[0], orientation[1], orientation[2])
        inertial_climb_rate = body_to_inertial(body_state[3:6], dcm)[2:]
        thrust, x_climb = climb_stage(
            p[:, climb], x[:, climb], climb_rate_reference, inertial_climb_rate,
            orientation, i.mass, i.weight, dt)
        x = x.at[:, altitude].set(x_altitude).at[:, climb].set(x_climb)
        return x, memo._replace(thrust=thrust)

    def run_position(x: Array, memo: Memo) -> Tuple[Array, Memo]:
        """Position stage: target x, y -> velocity reference."""
        dt = steps.position * i.sim_dt
        pid_velocity, x_position = position_stage(
            p[:, position], x[:, position], i.reference[:2], body_state[:2], body_state,
            i.acceleration, i.sim_dt, dt, i.leashing, i.square_root_position)
        # Blend in the feed-forward velocity, if given, and limit the speed
        blend = jnp.where(i.use_feed_forward, i.feedforward_weight, 0.)
        ref_vel = _clip(
            blend * i.feed_forward + (1. - blend) * pid_velocity,
            p[PID_LIMIT, CASCADE_POSITION])
        return x.at[:, position].set(x_position), memo._replace(ref_vel=ref_vel)

    def run_attitude(x: Array, memo: Memo) -> Tuple[Array, Memo]:
        """Velocity, attitude and rate stages: velocity and yaw targets -> torques."""
        dt = steps.attitude * i.sim_dt
        pitch_roll, x_velocity = velocity_stage(
            p[:, velocity], x[:, velocity], memo.ref_vel, body_state[3:5], dt)
        # The roll angle moves the vehicle along y, and the pitch angle along x
        reference_orientation = jnp.stack((pitch_roll[1], pitch_roll[0], i.reference[3]))
        euler_rate_reference, x_attitude = attitude_stage(
            p[:, attitude], x[:, attitude], reference_orientation, orientation, dt,
            i.square_root_attitude)
        euler_rate = angular_to_euler_rate(body_state[9:12], orientation)
        torques, x_rate = rate_stage(
            p[:, rate], x[:, rate], euler_rate_reference, euler_rate, orientation,
            i.inertia, dt)
        x = x.at[:, velocity].set(x_velocity).at[:, attitude].set(x_attitude)
        return x.at[:, rate].set(x_rate), memo._replace(torques=torques)

    def skip(x: Array, memo: Memo) -> Tuple[Array, Memo]:
        """A group that does not run at this step keeps its state and repeats its output."""
        return x, memo

    x, memo = lax.cond(n % steps.altitude == 0, run_altitude, skip, x, memo)
    x, memo = lax.cond(n % steps.position == 0, run_position, skip, x, memo)
    x, memo = lax.cond(n % steps.attitude == 0, run_attitude, skip, x, memo)
    return x, memo, jnp.concatenate((memo.thrust, memo.torques))
