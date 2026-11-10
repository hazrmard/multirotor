"""
The cascaded controller as pure functions of its state: ``(params, state, memo)``
in, ``(state, memo, action)`` out. Nothing is written in place, so the same
source builds for numba (`multirotor.xp_numba`) and for JAX (`multirotor.xp_jax`),
which differentiates it with respect to the gains.

It computes what `multirotor.controller.kernels.cascaded_pid_step` computes. The
differences are in the interface:

* ``persist`` does not exist. The caller keeps the returned ``state`` or drops it.
* The loop variables are the ``memo``: the outputs of the position, climb and rate
  stages, which later steps and stages reuse. It is a tuple
  ``(ref_vel (2,), thrust (1,), torques (3,))``.
* A stage runs on its own columns of the matrices and returns its new columns.
  ``x`` is rebuilt with `concatenate`, which both backends have.
* A stage that does not run this step (``n % steps != 0``) is a `cond` that
  returns its inputs.
"""
from .. import xp_numba as xp
from ..lazy import build
from .kernels import (
    CASCADE_ALTITUDE, CASCADE_ATTITUDE, CASCADE_CLIMB, CASCADE_POSITION,
    CASCADE_RATE, CASCADE_VELOCITY, PID_ACTION, PID_ERR, PID_ERR_D, PID_ERR_I,
    PID_ERR_P, PID_INTEGRAL, PID_K_D, PID_K_I, PID_K_P, PID_LIMIT,
    PID_MAX_ACCELERATION, PID_MAX_ERR_I, PID_REFERENCE)

INF = float('inf')



def _dcm(roll, pitch, yaw):
    cr, sr = xp.cos(roll), xp.sin(roll)
    cp, sp = xp.cos(pitch), xp.sin(pitch)
    cy, sy = xp.cos(yaw), xp.sin(yaw)
    return xp.array((
        (cp * cy,                 cp * sy,                  -sp),
        (-cr * sy + sr * sp * cy, cr * cy + sr * sp * sy,   sr * cp),
        (sr * sy + cr * sp * cy,  -sr * cy + cr * sp * sy,  cr * cp)))



def _angular_to_euler_rate(w, orientation):
    roll, pitch = orientation[0], orientation[1]
    p, q, r = w[0], w[1], w[2]
    return xp.array((
        p + xp.tan(pitch) * (q * xp.sin(roll) + r * xp.cos(roll)),
        q * xp.cos(roll) - r * xp.sin(roll),
        (q * xp.sin(roll) + r * xp.cos(roll)) / xp.cos(pitch)))



def _euler_to_angular_rate(e, orientation):
    roll, pitch = orientation[0], orientation[1]
    cr, cp = xp.cos(roll), xp.cos(pitch)
    sr, sp = xp.sin(roll), xp.sin(pitch)
    return xp.array((
        e[0] - e[2] * sp,
        e[1] * cr + e[2] * cp * sr,
        e[2] * cr * cp - e[1] * sr))



def _clip(value, limit):
    return xp.minimum(xp.maximum(value, -limit), limit)



# A stage works on its own columns of the matrices, one channel at a time, and
# returns its state block, shape (N_PID_STATES, n), in the order of the PID_*
# rows. `xp.map_rows` runs the per-channel function: a loop on numba, a vmap on JAX.

def _pid_scalar(p, x, i, reference, measurement, dt):
    "The PID law of channel ``i``: the action and the new error terms."
    err = reference - measurement
    previous = x[PID_ERR, i]
    err_p = p[PID_K_P, i] * err
    integral = _clip(x[PID_INTEGRAL, i] + dt * (previous + err) / 2., p[PID_MAX_ERR_I, i])
    err_i = p[PID_K_I, i] * integral
    err_d = p[PID_K_D, i] * (err - previous) / dt
    return err_p + err_i + err_d, err, err_p, err_i, err_d, integral



def _pid_channel(i, c):
    p, x, reference, measurement, dt = c
    action, err, err_p, err_i, err_d, integral = _pid_scalar(
        p, x, i, reference[i], measurement[i], dt)
    return err, err_p, err_i, err_d, reference[i], action, integral



def _limited_channel(i, c):
    "The PID law, limited to the stage's limit (altitude stage)."
    p, x, reference, measurement, dt = c
    action, err, err_p, err_i, err_d, integral = _pid_scalar(
        p, x, i, reference[i], measurement[i], dt)
    return err, err_p, err_i, err_d, reference[i], _clip(action, p[PID_LIMIT, i]), integral



def _tilt_channel(i, c):
    "The PID law, limited, with the sign of the second channel flipped (velocity stage)."
    p, x, reference, measurement, dt = c
    action, err, err_p, err_i, err_d, integral = _pid_scalar(
        p, x, i, reference[i], measurement[i], dt)
    out = _clip(action, p[PID_LIMIT, i])
    return err, err_p, err_i, err_d, reference[i], xp.select(i == 1, -out, out), integral



def _climb_channel(i, c):
    "The PID law as a thrust force, tilted to compensate for roll and pitch."
    p, x, reference, measurement, dt, orientation, mass, weight = c
    action, err, err_p, err_i, err_d, integral = _pid_scalar(
        p, x, i, reference[i], measurement[i], dt)
    out = mass * (action / (xp.cos(orientation[0]) * xp.cos(orientation[1]))) + weight
    return err, err_p, err_i, err_d, reference[i], out, integral



def _sqrt_scalar(e, k, lim, dt):
    """
    Piece-wise P-controller, linear near zero and square root further away,
    without branches that a derivative would have to pass through.
    """
    k_safe = xp.select(k == 0., 1., k)
    linear = lim / k_safe**2
    general = xp.select(
        e > linear, xp.sqrt_nonneg(2. * lim * (e - linear / 2.)),
        xp.select(
            e < -linear, -xp.sqrt_nonneg(2. * lim * (-e - linear / 2.)), e * k))
    zero_gain = xp.select(
        e > 0., xp.sqrt_nonneg(2. * lim * e),
        xp.select(e < 0., -xp.sqrt_nonneg(-2. * lim * e), 0.))
    rate = xp.select(lim <= 0., e * k, xp.select(k == 0., zero_gain, general))
    bound = xp.abs(e) / dt
    return xp.minimum(xp.maximum(-bound, rate), bound)



def _sqrt_channel(i, c):
    "The square-root P-term replaces the PID law; the other terms are left as they were."
    p, x, reference, err, dt = c
    out = _sqrt_scalar(err[i], p[PID_K_P, i], p[PID_MAX_ACCELERATION, 0], dt)
    return err[i], out, x[PID_ERR_I, i], x[PID_ERR_D, i], reference[i], out, x[PID_INTEGRAL, i]



def _pid_terms(op):
    p, x, reference, measurement, err, dt = op
    return xp.map_rows(_pid_channel, reference.shape[0], (p, x, reference, measurement, dt))



def _sqrt_terms(op):
    p, x, reference, measurement, err, dt = op
    return xp.map_rows(_sqrt_channel, reference.shape[0], (p, x, reference, err, dt))



def _altitude(p, x, reference, measurement, dt):
    rows = xp.map_rows(_limited_channel, 1, (p, x, reference, measurement, dt))
    return rows[PID_ACTION], rows



def _climb(p, x, reference, measurement, orientation, mass, weight, dt):
    rows = xp.map_rows(
        _climb_channel, 1, (p, x, reference, measurement, dt, orientation, mass, weight))
    return rows[PID_ACTION], rows



def _velocity(p, x, reference, measurement, dt):
    rows = xp.map_rows(_tilt_channel, 2, (p, x, reference, measurement, dt))
    return rows[PID_ACTION], rows



def _rate(p, x, reference, measurement, orientation, inertia, dt):
    ref = _euler_to_angular_rate(reference, orientation)
    mea = _euler_to_angular_rate(measurement, orientation)
    rows = xp.map_rows(_pid_channel, 3, (p, x, ref, mea, dt))
    torque = inertia @ _clip(rows[PID_ACTION], p[PID_LIMIT])
    return torque, xp.with_row(rows, PID_ACTION, torque)



def _attitude(p, x, reference, measurement, dt, square_root_scaling):
    err = reference - measurement
    err_len = xp.sqrt_nonneg(err[0]**2 + err[1]**2 + err[2]**2)
    rows = xp.cond(
        square_root_scaling & (err_len > 0.), _sqrt_terms, _pid_terms,
        (p, x, reference, measurement, err, dt))
    return rows[PID_ACTION], rows



def _leash(k_p, body_state, acceleration, sim_dt):
    "Largest horizontal position error to act on; infinite if covered within one step."
    dcm = _dcm(body_state[6], body_state[7], body_state[8])
    acc = (dcm.T @ acceleration)
    vel = (dcm.T @ body_state[3:6])
    acc_mag = xp.sqrt_nonneg(acc[0]**2 + acc[1]**2)
    vel_mag = xp.sqrt_nonneg(vel[0]**2 + vel[1]**2)
    leash = xp.abs(acc_mag / (2 * k_p**2) + vel_mag**2 / (2 * acc_mag + 1e-6))
    return xp.select(leash <= vel_mag * sim_dt, INF, leash)



def _leashed(op):
    p, err, body_state, acceleration, sim_dt = op
    err_len = xp.sqrt_nonneg(err[0]**2 + err[1]**2)
    leash = _leash(p[PID_K_P][0], body_state, acceleration, sim_dt)
    shortened = err / (err_len + 1e-6) * xp.minimum(err_len, leash)
    return shortened, xp.minimum(err_len, leash)


def _unleashed(op):
    p, err, body_state, acceleration, sim_dt = op
    return err, xp.sqrt_nonneg(err[0]**2 + err[1]**2)



def _position(
    p, x, reference, measurement, body_state, acceleration, sim_dt, dt, leashing,
    square_root_scaling
):
    err, err_len = xp.cond(
        leashing, _leashed, _unleashed, (p, reference - measurement, body_state, acceleration, sim_dt))
    # square root: acts on the (leashed) error. Otherwise the plain PID law acts
    # on the error without the leash.
    rows = xp.cond(
        square_root_scaling & (err_len > 0.), _sqrt_terms, _pid_terms,
        (p, x, reference, measurement, err, dt))
    velocity = rows[PID_ACTION]
    cos, sin = xp.cos(body_state[8]), xp.sin(body_state[8])
    body_x = cos * velocity[0] + sin * velocity[1]
    body_y = -sin * velocity[0] + cos * velocity[1]
    magnitude = xp.sqrt_nonneg(body_x**2 + body_y**2)
    speed = xp.minimum(magnitude, p[PID_LIMIT][0])
    out = xp.array((body_x / (magnitude + 1e-6) * speed, body_y / (magnitude + 1e-6) * speed))
    return out, xp.with_row(rows, PID_ACTION, out)



def _skip(op):
    return op[0], op[1]



def _run_z(op):
    "Altitude and climb stages: target z -> thrust."
    x, memo, c = op
    params, reference, body_state, mass, weight, sim_dt, steps = (
        c[0], c[1], c[2], c[5], c[6], c[7], c[8])
    dt = steps[2] * sim_dt
    orientation, velocity = body_state[6:9], body_state[3:6]
    z, z_cols = CASCADE_ALTITUDE, CASCADE_CLIMB
    climb_rate_ref, x_alt = _altitude(
        params[:, z:z + 1], x[:, z:z + 1], reference[2:3], body_state[2:3], dt)
    dcm = _dcm(orientation[0], orientation[1], orientation[2])
    inertial_z = (dcm.T @ velocity)[2:]
    thrust, x_climb = _climb(
        params[:, z_cols:z_cols + 1], x[:, z_cols:z_cols + 1], climb_rate_ref,
        inertial_z, orientation, mass, weight, dt)
    x_new = xp.concatenate((x[:, :z], x_alt, x_climb), axis=1)
    return x_new, (memo[0], thrust, memo[2])



def _run_p(op):
    "Position stage: target x, y -> velocity reference."
    x, memo, c = op
    (params, reference, body_state, acceleration, sim_dt, steps, feed_forward,
     use_feed_forward, feedforward_weight, leashing, square_root_position) = (
        c[0], c[1], c[2], c[3], c[7], c[8], c[9], c[10], c[11], c[12], c[13])
    dt = steps[0] * sim_dt
    s, e = CASCADE_POSITION, CASCADE_POSITION + 2
    pid_vel, x_pos = _position(
        params[:, s:e], x[:, s:e], reference[:2], body_state[:2], body_state,
        acceleration, sim_dt, dt, leashing, square_root_position)
    weight = xp.select(use_feed_forward, feedforward_weight, 0.)
    ref_vel = _clip(weight * feed_forward + (1. - weight) * pid_vel, params[PID_LIMIT, s])
    x_new = xp.concatenate((x_pos, x[:, e:]), axis=1)
    return x_new, (ref_vel, memo[1], memo[2])



def _run_a(op):
    "Velocity, attitude and rate stages: velocity, yaw target -> torques."
    x, memo, c = op
    (params, reference, body_state, inertia, sim_dt, steps, square_root_attitude) = (
        c[0], c[1], c[2], c[4], c[7], c[8], c[14])
    dt = steps[1] * sim_dt
    ref_vel = memo[0]
    orientation = body_state[6:9]
    v, a, r = CASCADE_VELOCITY, CASCADE_ATTITUDE, CASCADE_RATE
    pitch_roll, x_vel = _velocity(
        params[:, v:a], x[:, v:a], ref_vel, body_state[3:5], dt)
    ref_orientation = xp.array((pitch_roll[1], pitch_roll[0], reference[3]))
    euler_ref, x_att = _attitude(
        params[:, a:r], x[:, a:r], ref_orientation, orientation, dt, square_root_attitude)
    euler_rate = _angular_to_euler_rate(body_state[9:12], orientation)
    torques, x_rate = _rate(
        params[:, r:CASCADE_ALTITUDE], x[:, r:CASCADE_ALTITUDE], euler_ref, euler_rate,
        orientation, inertia, dt)
    x_new = xp.concatenate((x[:, :v], x_vel, x_att, x_rate, x[:, CASCADE_ALTITUDE:]), axis=1)
    return x_new, (memo[0], memo[1], torques)



def cascaded_step(
    params, x, memo, reference, body_state, acceleration, inertia, mass, weight,
    sim_dt, n, steps_p, steps_a, steps_z, feed_forward, use_feed_forward,
    feedforward_weight, leashing, square_root_position, square_root_attitude
):
    """
    One step of the whole cascade: the six stages in order, each at its own
    rate. Altitude and climb run every ``steps_z`` steps, position every
    ``steps_p``, velocity, attitude and rate every ``steps_a``. In between, the
    last outputs are repeated.

    Returns
    -------
    x : state matrix after the step, shape (N_PID_STATES, 12)
    memo : ``(ref_vel (2,), thrust (1,), torques (3,))``
    action : thrust and the three torques, shape (4,)
    """
    steps = (steps_p, steps_a, steps_z)
    c = (params, reference, body_state, acceleration, inertia, mass, weight, sim_dt,
         steps, feed_forward, use_feed_forward, feedforward_weight, leashing,
         square_root_position, square_root_attitude)
    x, memo = xp.cond(n % steps_z == 0, _run_z, _skip, (x, memo, c))
    x, memo = xp.cond(n % steps_p == 0, _run_p, _skip, (x, memo, c))
    x, memo = xp.cond(n % steps_a == 0, _run_a, _skip, (x, memo, c))
    action = xp.concatenate((memo[1], memo[2]))
    return x, memo, action



NAMES = (
    '_dcm', '_angular_to_euler_rate', '_euler_to_angular_rate', '_clip',
    '_pid_scalar', '_pid_channel', '_limited_channel', '_tilt_channel',
    '_climb_channel', '_sqrt_scalar', '_sqrt_channel', '_pid_terms', '_sqrt_terms',
    '_altitude', '_climb', '_velocity', '_rate', '_attitude', '_leash', '_leashed', '_unleashed',
    '_position', '_skip', '_run_z', '_run_p', '_run_a', 'cascaded_step')
SOURCE = {name: globals()[name] for name in NAMES}
NOT_INLINED = ('cascaded_step',)   # large, and called once per step
globals().update(vars(build(SOURCE, NAMES, 'numba', globals(), NOT_INLINED)))
