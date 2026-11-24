"""
Design gradients: the parameter dataclasses are JAX pytrees, `multirotor.builder` builds the
simulation's arrays from them with JAX operations, and a rollout's loss can be differentiated
all the way back to the geometry. Also the controller's model of the vehicle, which a rollout
can take apart from the plant's parameters.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import multirotor.rollout as rollout
from multirotor.builder import build_vehicle
from multirotor.helpers import control_allocation_matrix, vehicle_params_factory
from multirotor.simulation import Multirotor
from multirotor.tuples import ControllerModel
from multirotor.vehicle import PropellerParams, SimulationParams

from common import make_params
from test_controller import make_controller
import multirotor.controller.pid as pid
from test_rollout import DT, SUBSTEPS, make_vehicle



def test_dataclasses_are_pytrees():
    vp, sp = make_params(8, 'quadratic', True, np.float64)
    leaves, tree = jax.tree_util.tree_flatten(vp)
    assert all(isinstance(leaf, (float, np.ndarray)) for leaf in leaves)
    again = jax.tree_util.tree_unflatten(tree, leaves)
    assert again.mass == vp.mass and len(again.propellers) == 8
    # The motors' drag coefficient is the propeller's
    assert again.propellers[0].motor.k_drag == again.propellers[0].k_drag
    # Static fields stay out of the leaves: a gradient needs float leaves
    assert vp.propellers[0].b == 2
    assert not any(isinstance(leaf, (int, bool)) for leaf in leaves)



@pytest.mark.parametrize('kind', ('body', 'motors', 'motors_battery'))
def test_build_vehicle_agrees_for_numpy_and_jax_values(kind):
    vehicle = make_vehicle(kind)
    params = vehicle.params
    numpy_tuple = build_vehicle(params, vehicle.simulation)
    as_jax = jax.tree_util.tree_map(jnp.asarray, params)
    jax_tuple = jax.jit(lambda p: build_vehicle(p, vehicle.simulation))(as_jax)
    assert type(numpy_tuple) is type(jax_tuple)
    for a, b in zip(numpy_tuple, jax_tuple):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)



def test_gradient_of_the_built_arrays_matches_finite_differences():
    # Every field of the arrays that depends on the arm length and the body's mass
    vp, sp = make_params(8, 'quadratic', True, np.float64)
    propeller = vp.propellers[0]

    def summary(theta):
        params = vehicle_params_factory(8, 0.05, theta[0], propeller, theta[1])
        v = build_vehicle(params, sp)
        return v.mass + v.inertia.sum() + v.inertia_inverse.sum() + v.positions.sum() ** 2 \
            + (v.positions ** 2).sum()

    theta = jnp.array([0.4, 2.0])
    gradient = np.asarray(jax.grad(summary)(theta))
    for i in range(2):
        h = jnp.zeros(2).at[i].set(1e-6)
        fd = float((summary(theta + h) - summary(theta - h)) / 2e-6)
        assert gradient[i] == pytest.approx(fd, rel=1e-6)
    assert np.all(np.abs(gradient) > 1e-3)



def test_gradient_reaches_a_propeller_and_a_motor_field():
    vp, sp = make_params(8, 'quadratic', True, np.float64)
    gradient = jax.grad(lambda p: build_vehicle(p, sp).prop_params.sum()
                        + build_vehicle(p, sp).motor_params[0].sum())(vp)
    assert type(gradient) is type(vp)
    assert gradient.propellers[0].k_thrust == pytest.approx(1.)
    assert gradient.propellers[0].motor.resistance == pytest.approx(1.)



def test_allocation_is_differentiable():
    vp, sp = make_params(8, 'quadratic', False, np.float64)
    propeller = vp.propellers[0]

    def total(d):
        params = vehicle_params_factory(8, 0.05, d, propeller, 2.)
        return (control_allocation_matrix(params)[1] ** 2).sum()

    d = 0.4
    fd = (total(d + 1e-6) - total(d - 1e-6)) / 2e-6
    assert float(jax.grad(total)(d)) == pytest.approx(float(fd), rel=1e-6)



@pytest.mark.slow
def test_rollout_gradient_with_respect_to_the_arm_length():
    # The loss of a rollout is a function of the geometry, through the factory and the builder.
    # The controller was tuned for the nominal vehicle (its allocation and its model of the
    # vehicle), so a different arm length changes the flight. If the controller knew the
    # true geometry, the allocation would cancel the arm length and the gradient would be zero.
    vehicle = make_vehicle('body')
    controller = make_controller(pid, vehicle)
    ctrl, nominal = controller.rollout_params(), controller.rollout_model()
    state0 = jax.tree_util.tree_map(jnp.asarray, vehicle.rollout_state())
    propeller = vehicle.params.propellers[0]
    targets = jnp.tile(jnp.array([1., -0.5, 1., 0.3]), (80, 1))
    n = vehicle.params.nprops
    m_prop, m_body, d0 = 0.1, 2., float(vehicle.params.distances[0])

    def loss(d):
        plant = build_vehicle(
            vehicle_params_factory(n, m_prop, d, propeller, m_body), vehicle.simulation)
        trajectory = rollout.closed_loop_body(
            state0, plant, ctrl, targets, DT, SUBSTEPS['body'], model=nominal)[1]
        return jnp.mean((trajectory[:, :3] - targets[:, :3]) ** 2) \
            + jnp.mean(trajectory[:, 6:9] ** 2)

    value, gradient = jax.jit(jax.value_and_grad(loss))(d0)
    assert np.isfinite(float(gradient)) and abs(float(gradient)) > 1e-4
    fd = float((jax.jit(loss)(d0 + 1e-3) - jax.jit(loss)(d0 - 1e-3)) / 2e-3)
    assert float(gradient) == pytest.approx(fd, rel=1e-3)



def test_the_controller_model_defaults_to_the_plant():
    vehicle = make_vehicle('motors')
    controller = make_controller(pid, vehicle)
    plant = rollout.plant_model(vehicle.rollout_params())
    nominal = controller.rollout_model()
    assert nominal.mass == pytest.approx(float(plant.mass))
    assert nominal.weight == pytest.approx(float(plant.weight))
    np.testing.assert_allclose(nominal.inertia, plant.inertia)



def test_a_controller_model_separates_the_controller_from_the_plant():
    vehicle = make_vehicle('motors')
    controller = make_controller(pid, vehicle)
    state0 = jax.tree_util.tree_map(jnp.asarray, vehicle.rollout_state())
    plant, ctrl = vehicle.rollout_params(), controller.rollout_params()
    nominal = controller.rollout_model()
    targets = jnp.tile(jnp.array([0., 0., 2., 0.]), (60, 1))

    def height(mass, model):
        trajectory = rollout.closed_loop_motors(
            state0, plant._replace(mass=mass), ctrl, targets, DT, SUBSTEPS['motors'],
            model=model)[1]
        return trajectory[-1, 2]

    heavier = plant.mass * 1.2
    adaptive = float(height(heavier, None))
    # A model equal to the plant's own values is the default
    same = ControllerModel(heavier, heavier * plant.g, plant.inertia)
    assert float(height(heavier, same)) == pytest.approx(adaptive, rel=1e-12)
    # A controller that keeps its nominal mass does not climb as high with a heavier plant
    assert float(height(heavier, nominal)) < adaptive - 1e-3
    # Both gradients with respect to the plant's mass are right, and different
    for model in (None, nominal):
        g = float(jax.grad(height)(jnp.asarray(plant.mass), model))
        fd = float((height(plant.mass + 1e-3, model) - height(plant.mass - 1e-3, model)) / 2e-3)
        assert g == pytest.approx(fd, rel=1e-3)
    assert float(jax.grad(height)(jnp.asarray(plant.mass), None)) != pytest.approx(
        float(jax.grad(height)(jnp.asarray(plant.mass), nominal)), rel=0.1)
