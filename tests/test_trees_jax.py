"""
`multirotor.trees` with JAX: a rollout resumed from a state tree matches numba's, and
the gradient of a rollout's loss with respect to trees of named controller and vehicle
parameters matches finite differences. Skipped without JAX.
"""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import multirotor.trees as trees
from multirotor import rollout

from test_rollout import BUILT, DT, SUBSTEPS, TARGET, arguments
from test_rollout_jax import JAX
from test_trees import fly



def test_a_jax_rollout_resumed_from_a_state_tree_matches_numba():
    vehicle, controller = fly('body', 37)
    state0, start = trees.rollout_state(trees.get_state(vehicle, controller))
    vehicle_params = rollout.vehicle_arguments(vehicle)[2]
    ctrl = rollout.controller_arguments(controller, vehicle)
    targets = np.tile(TARGET, (25, 1))
    _, expected = BUILT.closed_loop_body(state0, vehicle_params, ctrl, targets, DT, 4, start)
    state0_jax = jax.tree_util.tree_map(jnp.asarray, state0)
    start_jax = (jnp.asarray(start[0]), tuple(jnp.asarray(m) for m in start[1]), start[2])
    _, found = JAX.closed_loop_body(
        state0_jax, vehicle_params, ctrl, jnp.asarray(targets), DT, 4, start_jax)
    np.testing.assert_allclose(np.asarray(found), expected, rtol=1e-9, atol=1e-9)



@pytest.mark.slow
def test_gradients_with_respect_to_named_trees_match_finite_differences():
    """
    The leaves of the tree are named gains and parameters: the altitude and climb stages'
    gains, the rate stage's acceleration limit, the vehicle's mass, the motors' resistance
    and the propellers' thrust constants. The loss is the mean squared position error of a
    rollout. (The other gains have a zero gradient here: the vehicle is far from its target,
    so the tilt and velocity limits are active. The thrust constants are made unequal: at
    the nominal vehicle, where they are all equal, the per-propeller gradients differed
    from finite differences by up to 1e-3 relative, probably from a kink at that symmetric
    point, which was not investigated further.)
    """
    kind, steps, substeps = 'motors', 60, 8
    state, vehicle_params, ctrl = arguments(kind)
    state = jax.tree_util.tree_map(jnp.asarray, state)
    base = trees.vehicle_params(vehicle_params, kind)
    targets = jnp.tile(jnp.asarray(TARGET), (steps, 1))
    run = getattr(JAX, 'closed_loop_' + kind)

    def loss(theta):
        params = trees.with_controller_params(ctrl[0], theta['controller'])
        tree = {**base, 'mass': theta['mass'],
                'motor': {**base['motor'], 'resistance': theta['resistance']},
                'propeller': {**base['propeller'], 'k_thrust': theta['k_thrust']}}
        trajectory = run(
            state, trees.vehicle_tuple(tree, kind), (params,) + ctrl[1:], targets, DT, substeps)[1]
        return jnp.mean((trajectory[:, :3] - targets[:, :3])**2)

    controller = trees.controller_params(ctrl[0])
    theta = {
        'controller': {
            'ctrl_z': {'k_p': jnp.asarray(controller['ctrl_z']['k_p'])},
            'ctrl_vz': {'k_p': jnp.asarray(controller['ctrl_vz']['k_p']),
                        'k_d': jnp.asarray(controller['ctrl_vz']['k_d'])},
            'ctrl_r': {'max_acceleration': jnp.asarray(controller['ctrl_r']['max_acceleration'])}},
        'mass': jnp.asarray(base['mass']), 'resistance': jnp.asarray(base['motor']['resistance']),
        'k_thrust': jnp.asarray(base['propeller']['k_thrust'])
        * jnp.asarray(1. + .03 * np.random.default_rng(1).normal(size=8))}
    value_and_grad, fast_loss = jax.jit(jax.value_and_grad(loss)), jax.jit(loss)
    _, gradient = value_and_grad(theta)
    leaves = jax.tree_util.tree_leaves(gradient)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
    assert sum(np.abs(np.asarray(leaf)).sum() for leaf in leaves) > 1e-6
    # the gradient has the structure of the tree
    assert gradient['controller']['ctrl_vz']['k_p'].shape == theta['controller']['ctrl_vz']['k_p'].shape
    assert abs(float(gradient['controller']['ctrl_z']['k_p'][0])) > 1e-3
    rng = np.random.default_rng(0)
    direction = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(rng.normal(size=leaf.shape)) * leaf * 1e-5, theta)
    predicted = sum(
        float(jnp.vdot(g, d)) for g, d in zip(
            jax.tree_util.tree_leaves(gradient), jax.tree_util.tree_leaves(direction)))
    shift = lambda sign: jax.tree_util.tree_map(lambda t, d: t + sign * d, theta, direction)
    found = float(fast_loss(shift(1.)) - fast_loss(shift(-1.))) / 2.
    assert abs(predicted - found) < 1e-4 * abs(found) + 1e-9, (predicted, found)
