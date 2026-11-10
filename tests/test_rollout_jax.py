"""
`multirotor.rollout` built for JAX: parity with the numba build and with the loop of
`Controller` and `Multirotor`, convergence of RK4 with the number of substeps, and
gradients of a rollout's loss with respect to gains and vehicle parameters.
Skipped without JAX.
"""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import multirotor.physics as physics
import multirotor.rollout as rollout
from multirotor.controller import kernels as K

from test_rollout import (
    BUILT, DT, POSITION_TOLERANCE, SUBSTEPS, TARGET, arguments, dop853, reference)

JAX = rollout.build_for('jax')



def run_jax(kind, steps, substeps, ctrl=None, plant=None):
    state, plant_, ctrl_ = arguments(kind)
    state = jax.tree_util.tree_map(jnp.asarray, state)
    return getattr(JAX, 'closed_loop_' + kind)(
        state, plant or plant_, ctrl or ctrl_,
        jnp.tile(jnp.asarray(TARGET), (steps, 1)), DT, substeps)[1]



@pytest.mark.parametrize('kind', ('body', 'motors', 'motors_battery'))
def test_jax_rollout_matches_the_loop(kind):
    steps = 60
    trajectory = np.asarray(run_jax(kind, steps, SUBSTEPS[kind]))
    expected = reference(kind, steps)
    assert np.abs(trajectory - expected)[:, :3].max() < POSITION_TOLERANCE[kind]



def test_jax_rollout_matches_numba():
    steps = 30
    state, plant, ctrl = arguments('motors_battery')
    _, numba_trajectory = BUILT.closed_loop_motors_battery(
        state, plant, ctrl, np.tile(TARGET, (steps, 1)), DT, 8)
    trajectory = np.asarray(run_jax('motors_battery', steps, 8))
    np.testing.assert_allclose(trajectory, numba_trajectory, rtol=1e-9, atol=1e-9)



def test_more_substeps_converge_for_motors():
    "RK4 error against a tight integrator falls as the step is divided."
    steps = 60
    expected = reference('motors', steps, dop853)
    errors = [
        np.abs(np.asarray(run_jax('motors', steps, substeps)) - expected)[:, :3].max()
        for substeps in (2, 8, 16)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 2e-4



@pytest.mark.slow
@pytest.mark.parametrize('kind', ('motors', 'motors_battery'))
def test_gradients_match_finite_differences(kind):
    """
    Loss = mean squared position error over a short rollout. Parameters: a scale on
    the position gains, on the mass, on the motors' resistance and, with a battery,
    on its internal resistance. With a battery, the gradient passes through the
    power-limited bisection (``xp.root``) inside the scan.
    """
    steps, substeps = 60, 8
    state, plant, ctrl = arguments(kind)
    state = jax.tree_util.tree_map(jnp.asarray, state)
    targets = jnp.tile(jnp.asarray(TARGET), (steps, 1))

    def loss(theta):
        params = jnp.asarray(ctrl[0]).at[K.PID_K_P].multiply(theta[0])
        plant_ = list(plant)
        plant_[1] = plant[1] * theta[1]
        plant_[7] = jnp.asarray(plant[7]).at[physics.MOTOR_RESISTANCE].multiply(theta[2])
        if kind == 'motors_battery':
            plant_[10] = jnp.asarray(plant[10]).at[
                physics.BATTERY_INTERNAL_RESISTANCE].multiply(theta[3])
        trajectory = getattr(JAX, 'closed_loop_' + kind)(
            state, tuple(plant_), (params,) + ctrl[1:], targets, DT, substeps)[1]
        return jnp.mean((trajectory[:, :3] - targets[:, :3])**2)

    value, gradient = jax.jit(jax.value_and_grad(loss))(jnp.ones(4))
    gradient = np.asarray(gradient)
    assert np.isfinite(gradient).all()
    # Far from the target the square-root controller's output does not depend on the
    # position gain, so its gradient is zero (and must match the finite difference)
    # in the short rollout with a battery, whose vehicle climbs more slowly.
    must_matter = (1, 2) + ((3,) if kind == 'motors_battery' else (0,))
    assert all(abs(gradient[i]) > 1e-6 for i in must_matter), gradient
    loss = jax.jit(loss)
    for i in range(4 if kind == 'motors_battery' else 3):
        h = jnp.zeros(4).at[i].set(1e-6)
        fd = float((loss(jnp.ones(4) + h) - loss(jnp.ones(4) - h)) / 2e-6)
        assert abs(gradient[i] - fd) < 1e-4 * abs(fd) + 1e-9, (i, gradient[i], fd)
