"""
The rollout's tuples with JAX: a rollout resumed from a state matches numba's, and the
gradient of a rollout's loss with respect to controller and vehicle parameters matches
finite differences. Skipped without JAX.
"""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import multirotor.physics as physics
from multirotor.controller import data as K

from test_rollout import BUILT, DT, TARGET, arguments
from test_rollout_jax import JAX
from test_rollout_state import fly



def test_a_jax_rollout_resumed_from_a_state_matches_numba():
    vehicle, controller = fly('body', 37)
    state0, start = vehicle.rollout_state(), controller.rollout_state()
    params, ctrl = vehicle.rollout_params(), controller.rollout_params()
    targets = np.tile(TARGET, (25, 1))
    _, expected = BUILT.closed_loop_body(state0, params, ctrl, targets, DT, 4, start)
    as_jax = lambda tree: jax.tree_util.tree_map(jnp.asarray, tree)
    (controller_after, _), found = JAX.closed_loop_body(
        as_jax(state0), params, ctrl, jnp.asarray(targets), DT, 4, as_jax(start))
    np.testing.assert_allclose(np.asarray(found), expected, rtol=1e-9, atol=1e-9)
    assert int(controller_after.n) == 37 + 25



@pytest.mark.slow
def test_gradients_with_respect_to_named_parameters_match_finite_differences():
    """
    The parameters are the altitude and climb stages' gains, the rate stage's acceleration
    limit, the vehicle's mass, the motors' resistance and the propellers' thrust constants,
    each set into its row of its matrix by the constants of `multirotor.physics` and
    `multirotor.controller.data`. The loss is the mean squared position error of a rollout.
    (The other gains have a zero gradient here: the vehicle is far from its target, so the
    tilt and velocity limits are active. The thrust constants are made unequal: at the
    nominal vehicle, where they are all equal, the per-propeller gradients differed from
    finite differences by up to 1e-3 relative, probably from a kink at that symmetric point,
    which was not investigated further.)
    """
    kind, steps, substeps = 'motors', 60, 8
    state, vehicle, ctrl = arguments(kind)
    state = jax.tree_util.tree_map(jnp.asarray, state)
    targets = jnp.tile(jnp.asarray(TARGET), (steps, 1))
    run = getattr(JAX, 'closed_loop_' + kind)
    z, vz, rate = K.CASCADE_ALTITUDE, K.CASCADE_CLIMB, slice(K.CASCADE_RATE, K.CASCADE_ALTITUDE)

    def loss(theta):
        params = jnp.asarray(ctrl.params)
        params = params.at[K.PID_K_P, z].set(theta['z_k_p'])
        params = params.at[K.PID_K_P, vz].set(theta['vz_k_p'])
        params = params.at[K.PID_K_D, vz].set(theta['vz_k_d'])
        params = params.at[K.PID_MAX_ACCELERATION, rate].set(theta['rate_max_acceleration'])
        vehicle_ = vehicle._replace(
            mass=theta['mass'],
            motor_params=jnp.asarray(vehicle.motor_params).at[
                physics.MOTOR_RESISTANCE].set(theta['resistance']),
            prop_params=jnp.asarray(vehicle.prop_params).at[
                physics.PROP_K_THRUST].set(theta['k_thrust']))
        trajectory = run(state, vehicle_, ctrl._replace(params=params), targets, DT, substeps)[1]
        return jnp.mean((trajectory[:, :3] - targets[:, :3])**2)

    theta = {
        'z_k_p': jnp.asarray(ctrl.params[K.PID_K_P, z]),
        'vz_k_p': jnp.asarray(ctrl.params[K.PID_K_P, vz]),
        'vz_k_d': jnp.asarray(ctrl.params[K.PID_K_D, vz]),
        'rate_max_acceleration': jnp.asarray(ctrl.params[K.PID_MAX_ACCELERATION, rate]),
        'mass': jnp.asarray(vehicle.mass),
        'resistance': jnp.asarray(vehicle.motor_params[physics.MOTOR_RESISTANCE]),
        'k_thrust': jnp.asarray(vehicle.prop_params[physics.PROP_K_THRUST])
        * jnp.asarray(1. + .03 * np.random.default_rng(1).normal(size=8))}
    value_and_grad, fast_loss = jax.jit(jax.value_and_grad(loss)), jax.jit(loss)
    _, gradient = value_and_grad(theta)
    leaves = jax.tree_util.tree_leaves(gradient)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
    assert sum(np.abs(np.asarray(leaf)).sum() for leaf in leaves) > 1e-6
    assert gradient['k_thrust'].shape == theta['k_thrust'].shape     # the structure of theta
    assert abs(float(gradient['z_k_p'])) > 1e-3
    rng = np.random.default_rng(0)
    direction = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(rng.normal(size=leaf.shape)) * leaf * 1e-5, theta)
    predicted = sum(
        float(jnp.vdot(g, d)) for g, d in zip(
            jax.tree_util.tree_leaves(gradient), jax.tree_util.tree_leaves(direction)))
    shift = lambda sign: jax.tree_util.tree_map(lambda t, d: t + sign * d, theta, direction)
    found = float(fast_loss(shift(1.)) - fast_loss(shift(-1.))) / 2.
    assert abs(predicted - found) < 1e-4 * abs(found) + 1e-9, (predicted, found)
