"""
The tuples that `multirotor.rollout` takes and returns, and the methods of `Multirotor` and
`Controller` that make them and take them (``rollout_params``, ``rollout_state``,
``set_rollout_state``). The tuples must hold what the imperative classes hold, and the
state must go back and forth between the classes and the rollout without loss. The JAX
gradients with respect to the tuples are in ``test_rollout_state_jax.py``.
"""
import numpy as np
import pytest

import multirotor.controller.pid as pid
from multirotor import tuples

from test_controller import make_controller
from test_rollout import BUILT, DT, SUBSTEPS, TARGET, arguments, make_vehicle

slow = pytest.mark.slow
KINDS = ('body', pytest.param('motors_battery', marks=slow))



def fly(kind: str, steps: int):
    "A vehicle and its controller after ``steps`` steps of the imperative classes."
    vehicle = make_vehicle(kind)
    controller = make_controller(pid, vehicle)
    for _ in range(steps):
        action = controller.step(TARGET)
        vehicle.step_speeds(vehicle.allocate_control(action[0], action[1:]))
    return vehicle, controller



def assert_same(a, b, path='', **tolerance):
    "Equal nested tuples of arrays and numbers."
    if isinstance(a, tuple):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f'{path}[{i}]', **tolerance)
    else:
        np.testing.assert_allclose(a, b, **(tolerance or dict(rtol=0, atol=0)), err_msg=path)



def test_controller_params_are_what_the_controller_holds():
    controller = make_controller(pid, make_vehicle('body'))
    ctrl = controller.rollout_params()
    assert isinstance(ctrl, tuples.ControllerParams)
    np.testing.assert_array_equal(ctrl.params, controller._data.params)
    np.testing.assert_array_equal(ctrl.alloc_inverse, controller.vehicle.alloc_inverse)
    assert ctrl.steps == (controller.steps_p, controller.steps_a, controller.steps_z)
    ctrl.params[:] = 0.                      # a copy
    assert controller._data.params.any()



@pytest.mark.parametrize('kind, expected', [
    ('body', tuples.BodyVehicle), ('motors', tuples.MotorsVehicle),
    ('motors_battery', tuples.BatteryVehicle)])
def test_vehicle_params_have_the_type_of_the_vehicle(kind, expected):
    vehicle = make_vehicle(kind)
    params = vehicle.rollout_params()
    assert type(params) is expected and tuples.kind_of(params) == kind
    np.testing.assert_array_equal(params.mass, vehicle.params.mass)
    np.testing.assert_array_equal(params.prop_params, vehicle._data.params)
    np.testing.assert_allclose(params.inertia @ params.inertia_inverse, np.eye(3), atol=1e-12)
    if kind != 'body':
        np.testing.assert_array_equal(params.motor_params, vehicle._data.motors.params)
    if kind == 'motors_battery':
        np.testing.assert_array_equal(params.battery_params, vehicle.battery._params_vector)



@pytest.mark.parametrize('kind', KINDS)
def test_state_goes_through_the_rollout_and_back_without_loss(kind):
    vehicle, controller = fly(kind, 37)       # off a multiple of every period
    vehicle_state, controller_state = vehicle.rollout_state(), controller.rollout_state()
    assert controller_state.n == 37
    steps, t0 = 25, vehicle.t
    (controller_after, vehicle_after), _ = getattr(BUILT, 'closed_loop_' + kind)(
        vehicle_state, vehicle.rollout_params(), controller.rollout_params(),
        np.tile(TARGET, (steps, 1)), DT, SUBSTEPS[kind], controller_state)
    assert controller_after.n == 37 + steps
    np.testing.assert_allclose(vehicle_after.t, t0 + steps * DT)
    vehicle.set_rollout_state(vehicle_after)
    controller.set_rollout_state(controller_after)
    assert_same(vehicle.rollout_state(), vehicle_after, rtol=1e-14, atol=1e-14)
    assert_same(controller.rollout_state(), controller_after, rtol=1e-14, atol=1e-14)
    assert controller.n == 37 + steps and controller.t == vehicle.t



@pytest.mark.parametrize('kind', KINDS)
def test_a_resumed_rollout_equals_one_rollout(kind):
    "Phases of the stages that run every few steps are kept: the split is at step 35."
    vehicle, controller = fly(kind, 0)
    params, ctrl = vehicle.rollout_params(), controller.rollout_params()
    run = getattr(BUILT, 'closed_loop_' + kind)
    targets = np.tile(TARGET, (60, 1))
    _, whole = run(vehicle.rollout_state(), params, ctrl, targets, DT, SUBSTEPS[kind],
                   controller.rollout_state())
    (controller_after, vehicle_after), first = run(
        vehicle.rollout_state(), params, ctrl, targets[:35], DT, SUBSTEPS[kind],
        controller.rollout_state())
    vehicle.set_rollout_state(vehicle_after)           # through the objects and back
    controller.set_rollout_state(controller_after)
    _, second = run(vehicle.rollout_state(), params, ctrl, targets[35:], DT, SUBSTEPS[kind],
                    controller.rollout_state())
    np.testing.assert_allclose(np.concatenate((first, second)), whole, rtol=1e-12, atol=1e-12)



@pytest.mark.parametrize('kind', KINDS)
def test_the_imperative_classes_continue_from_a_rollout(kind):
    "After `set_rollout_state`, `Controller` and `Multirotor` fly on as the rollout would."
    vehicle, controller = fly(kind, 20)
    params, ctrl = vehicle.rollout_params(), controller.rollout_params()
    run = getattr(BUILT, 'closed_loop_' + kind)
    (controller_mid, vehicle_mid), _ = run(
        vehicle.rollout_state(), params, ctrl, np.tile(TARGET, (30, 1)), DT, SUBSTEPS[kind],
        controller.rollout_state())
    _, expected = run(vehicle_mid, params, ctrl, np.tile(TARGET, (10, 1)), DT, SUBSTEPS[kind],
                      controller_mid)
    vehicle.set_rollout_state(vehicle_mid)
    controller.set_rollout_state(controller_mid)
    positions = []
    for _ in range(10):
        action = controller.step(TARGET)
        vehicle.step_speeds(vehicle.allocate_control(action[0], action[1:]))
        positions.append(vehicle.position.copy())
    assert np.abs(np.array(positions) - expected[:, :3]).max() < 3e-3



def test_set_rollout_state_rejects_the_state_of_another_kind_of_vehicle():
    body_vehicle, _ = fly('body', 1)
    motors_vehicle, _ = fly('motors', 1)
    with pytest.raises(ValueError):
        motors_vehicle.set_rollout_state(body_vehicle.rollout_state())
