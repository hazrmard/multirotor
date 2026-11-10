"""
`multirotor.controller.cascade` built for JAX: parity with the numba build, and
gradients of a closed rollout with respect to the gains. Skipped without JAX.
"""
import numpy as np
import pytest

jax = pytest.importorskip('jax')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import multirotor.controller.cascade as cascade
from multirotor.controller import kernels as K
from multirotor.lazy import build
from test_cascade import INERTIA, MASS, OPTIONS, SIM_DT, WEIGHT, inputs, make_params

JAX = build(cascade.SOURCE, cascade.NAMES, 'jax', vars(cascade))



def test_jax_matches_numba():
    rng = np.random.default_rng(1)
    for leashing, sqrt_pos, sqrt_att, feed_forward, steps in OPTIONS[::3]:
        params = make_params(rng)
        step = jax.jit(JAX.cascaded_step, static_argnums=(11, 12, 13, 15, 17, 18, 19))
        x_n = x_j = np.zeros((K.N_PID_STATES, 12))
        memo_n = memo_j = (np.zeros(2), np.zeros(1), np.zeros(3))
        for n in range(20):
            reference, body_state, acceleration, velocity = inputs(rng)
            args = (reference, body_state, acceleration, INERTIA, MASS, WEIGHT, SIM_DT, n,
                    *steps, velocity, feed_forward, .4, leashing, sqrt_pos, sqrt_att)
            x_n, memo_n, a_n = cascade.cascaded_step(params, x_n, memo_n, *args)
            x_j, memo_j, a_j = step(params, x_j, memo_j, *args)
            np.testing.assert_allclose(x_j, x_n, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(a_j, a_n, rtol=1e-10, atol=1e-10)



@pytest.mark.parametrize('steps,flags', (
    ((1, 1, 1), (False, False, False)), ((3, 2, 5), (True, True, True)),
    ((3, 2, 5), (True, False, True))))
def test_gradient_through_a_rollout_matches_finite_differences(steps, flags):
    "Multi-rate steps, the leash and square-root control, carried through lax.scan."
    rng = np.random.default_rng(2)
    leashing, sqrt_pos, sqrt_att = flags
    params = make_params(rng)
    params[K.PID_MAX_ERR_I] = 2.
    targets = jnp.tile(jnp.array([.5, -.3, .4, .2]), (40, 1))

    def rollout(p):
        def body(carry, n_ref):
            x, memo, state = carry
            n, reference = n_ref
            x, memo, action = JAX.cascaded_step(
                p, x, memo, reference, state, jnp.zeros(3), INERTIA, MASS, WEIGHT,
                SIM_DT, n, *steps, jnp.zeros(2), False, 0., leashing, sqrt_pos, sqrt_att)
            # a toy plant that closes the loop
            state = state.at[2].add(1e-3 * (action[0] - WEIGHT) / MASS)
            return (x, memo, state.at[6:9].add(1e-3 * action[1:])), action
        carry = (jnp.zeros((K.N_PID_STATES, 12)),
                 (jnp.zeros(2), jnp.zeros(1), jnp.zeros(3)), jnp.zeros(12))
        return jax.lax.scan(body, carry, (jnp.arange(40), targets))[1]

    scale = float(jnp.mean(rollout(params)**2))
    loss = jax.jit(lambda p: jnp.mean(rollout(p)**2) / scale)
    grad = np.asarray(jax.grad(loss)(jnp.asarray(params)))
    assert np.isfinite(grad).all()
    entries = [(r, c) for r in (K.PID_K_P, K.PID_K_I, K.PID_K_D, K.PID_LIMIT)
               for c in (0, 2, 4, 7, 10, 11)]
    largest = max(abs(grad[e]) for e in entries)
    for entry in entries:
        h = 1e-5 * max(abs(params[entry]), 1.)
        up, down = params.copy(), params.copy()
        up[entry] += h
        down[entry] -= h
        fd = (float(loss(jnp.asarray(up))) - float(loss(jnp.asarray(down)))) / (2 * h)
        assert abs(grad[entry] - fd) < 1e-3 * largest, entry
