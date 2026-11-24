"""build_controller agrees with the controller's own arrays, and is differentiable."""
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from multirotor.builder import controller_model
from multirotor.controller.gains import build_controller
from multirotor.helpers import vehicle_params_factory
from multirotor.optimize import DEFAULTS, default_gains, get_controller
from multirotor.simulation import Multirotor
from multirotor.vehicle import SimulationParams
from argparse import Namespace

from common import make_params


def _vehicle():
    params, sim = make_params(8, dtype=np.float64)
    return params, sim, Multirotor(params, sim)


def _same(a, b):
    assert a.steps == b.steps
    assert (a.leashing, a.square_root_position, a.square_root_attitude) == (
        b.leashing, b.square_root_position, b.square_root_attitude)
    assert a.feedforward_weight == b.feedforward_weight
    np.testing.assert_allclose(np.asarray(a.params), np.asarray(b.params), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(a.alloc_inverse), np.asarray(b.alloc_inverse), atol=1e-9)


@pytest.mark.parametrize('leashing', [True, False])
@pytest.mark.parametrize('sqrt', [True, False])
def test_build_matches_controller(leashing, sqrt):
    params, sim, vehicle = _vehicle()
    args = Namespace(**{**vars(DEFAULTS), 'leashing': leashing, 'sqrt_scaling': sqrt})
    ctrl = get_controller(vehicle, args=args)
    _same(build_controller(ctrl.gains(), params, sim), ctrl.rollout_params())
    _same(build_controller(default_gains(args), params, sim), ctrl.rollout_params())


def test_attitude_sqrt_and_per_axis_limit():
    params, sim, vehicle = _vehicle()
    gains = default_gains()
    gains.attitude = dataclasses.replace(gains.attitude, square_root_scaling=True)
    gains.rate = dataclasses.replace(gains.rate, max_acceleration=np.array([20., 10., 5.]))
    from multirotor.controller import Controller
    ctrl = Controller.from_gains(gains, vehicle)
    _same(build_controller(gains, params, sim), ctrl.rollout_params())
    _same(build_controller(ctrl.gains(), params, sim), ctrl.rollout_params())


def test_gradient_wrt_gain():
    params, sim, _ = _vehicle()
    gains = default_gains()
    def f(k):
        g = dataclasses.replace(gains, velocity=dataclasses.replace(gains.velocity, k_p=k))
        return jnp.sum(build_controller(g, params, sim).params ** 2)
    k = 0.5
    grad = jax.grad(f)(k)
    h = 1e-6
    fd = (f(k + h) - f(k - h)) / (2 * h)
    assert abs(grad - fd) < 1e-6 * max(1., abs(fd))
    assert grad != 0


def test_gradient_reaches_arm_length():
    base, sim = make_params(8, dtype=np.float64)
    propeller = base.propellers[0]
    def f(arm):
        p = vehicle_params_factory(8, 0.05, arm, propeller, 2.)
        return jnp.sum(build_controller(default_gains(), p, sim).alloc_inverse)
    assert abs(jax.grad(f)(0.4)) > 0


def test_controller_model_matches():
    params, sim, vehicle = _vehicle()
    ctrl = get_controller(vehicle)
    m, ref = controller_model(params, sim), ctrl.rollout_model()
    np.testing.assert_allclose([m.mass, m.weight], [ref.mass, ref.weight])
    np.testing.assert_allclose(m.inertia, ref.inertia)
