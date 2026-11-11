"""
`multirotor.trees`: named trees of the controller's parameters, the vehicle's parameters and
the state of both. The trees must read what the imperative classes hold,
and the state must be handed back and forth between the classes and the rollout without
loss. The JAX gradients with respect to a tree are in ``test_trees_jax.py``.
"""
import numpy as np
import pytest

import multirotor.controller.pid as pid
import multirotor.trees as trees
from multirotor import rollout
from multirotor.controller import data as controller_data

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



def assert_same_tree(a, b, path='', **tolerance):
    "Equal nested dicts of arrays and numbers."
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            assert_same_tree(a[key], b[key], f'{path}/{key}', **tolerance)
    elif isinstance(a, tuple):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same_tree(x, y, f'{path}[{i}]', **tolerance)
    else:
        np.testing.assert_allclose(a, b, **(tolerance or dict(rtol=0, atol=0)), err_msg=path)



def test_gains_are_the_parameters_that_the_stages_hold():
    controller = make_controller(pid, make_vehicle('body'))
    tree = trees.controller_params(controller._data.params)
    assert tree.keys() == {'ctrl_p', 'ctrl_v', 'ctrl_a', 'ctrl_r', 'ctrl_z', 'ctrl_vz'}
    for stage, values in controller.get_params().items():
        if stage not in tree:
            continue
        for name, held in values.items():
            if name in tree[stage]:
                np.testing.assert_array_equal(
                    tree[stage][name], np.broadcast_to(held, tree[stage][name].shape),
                    err_msg=f'{stage}.{name}')
    # every parameter that is stored in the matrix is named: 4 shared, and the limits
    assert set(tree['ctrl_p']) == {
        'k_p', 'k_i', 'k_d', 'max_err_i', 'max_velocity', 'max_acceleration'}
    assert set(tree['ctrl_v']) == {'k_p', 'k_i', 'k_d', 'max_err_i', 'max_tilt'}
    assert set(tree['ctrl_r']) == {'k_p', 'k_i', 'k_d', 'max_err_i', 'max_acceleration'}



def test_with_gains_replaces_only_what_it_is_given():
    controller = make_controller(pid, make_vehicle('body'))
    params = controller._data.params
    kept = params.copy()
    assert np.array_equal(trees.with_controller_params(params, trees.controller_params(params)), params)
    new = trees.with_controller_params(params, {'ctrl_a': {'k_p': 7.}, 'ctrl_r': {'k_d': [1., 2., 3.]}})
    np.testing.assert_array_equal(params, kept)    # not changed
    expected = kept.copy()
    expected[controller_data.PID_K_P, 4:7] = 7.
    expected[controller_data.PID_K_D, 7:10] = [1., 2., 3.]
    np.testing.assert_array_equal(new, expected)
    with pytest.raises(KeyError):
        trees.with_controller_params(params, {'ctrl_a': {'max_velocity': 1.}})
    with pytest.raises(KeyError):
        trees.with_controller_params(params, {'ctrl_x': {'k_p': 1.}})



def test_set_gains_writes_what_set_params_would():
    vehicle = make_vehicle('body')
    by_tree, by_name = make_controller(pid, vehicle), make_controller(pid, vehicle)
    tree = {'ctrl_p': {'max_velocity': 3.5}, 'ctrl_r': {'k_p': [4., 5., 6.], 'max_acceleration': 2.}}
    trees.set_controller_params(by_tree, tree)
    by_name.set_params(
        ctrl_p=dict(max_velocity=3.5), ctrl_r=dict(k_p=[4., 5., 6.], max_acceleration=2.))
    np.testing.assert_array_equal(by_tree._data.params, by_name._data.params)
    np.testing.assert_array_equal(
        by_tree._data.params, trees.with_controller_params(make_controller(pid, vehicle)._data.params, tree))



@pytest.mark.parametrize('kind', ('body', 'motors', 'motors_battery'))
def test_vehicle_params_round_trip(kind):
    _, vehicle_params, _ = arguments(kind)
    tree = trees.vehicle_params(vehicle_params, kind)
    assert 'motor' in tree or kind == 'body'
    assert ('battery' in tree) == (kind == 'motors_battery')
    rebuilt = trees.vehicle_tuple(tree, kind)
    assert len(rebuilt) == len(vehicle_params)
    for name, a, b in zip(rollout.VEHICLE_FIELDS, rebuilt, vehicle_params):
        if name == 'inertia_inverse':      # recomputed, so equal to rounding
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15, err_msg=name)
        else:
            np.testing.assert_array_equal(a, b, err_msg=name)



def test_the_inverse_inertia_follows_the_inertia():
    _, vehicle_params, _ = arguments('body')
    tree = trees.vehicle_params(vehicle_params, 'body')
    assert 'inertia_inverse' not in tree
    tree['inertia'] = np.asarray(tree['inertia']) * np.array([1., 2., 3.])
    np.testing.assert_allclose(
        trees.vehicle_tuple(tree, 'body')[3], np.linalg.inv(tree['inertia']), rtol=1e-12)



def test_vehicle_params_names_the_parameters_that_the_vehicle_holds():
    vehicle = make_vehicle('motors_battery')
    _, vehicle_params, _ = arguments('motors_battery')
    tree = trees.vehicle_params(vehicle_params, 'motors_battery')
    np.testing.assert_array_equal(tree['mass'], vehicle.params.mass)
    np.testing.assert_array_equal(tree['propeller']['k_thrust'], vehicle._data.params[0])
    np.testing.assert_array_equal(tree['motor']['resistance'], vehicle._data.motors.params[0])
    np.testing.assert_array_equal(tree['battery']['internal_resistance'], .08)



@pytest.mark.parametrize('kind', KINDS)
def test_state_tree_hands_the_state_back_and_forth_without_loss(kind):
    vehicle, controller = fly(kind, 37)       # off a multiple of every period
    state = trees.get_state(vehicle, controller)
    assert state['controller']['n'] == 37
    _, vehicle_state, vehicle_params = rollout.vehicle_arguments(vehicle)
    state0, start = trees.rollout_state(state)
    steps = 25
    carry, _ = getattr(BUILT, 'closed_loop_' + kind)(
        state0, vehicle_params, rollout.controller_arguments(controller, vehicle),
        np.tile(TARGET, (steps, 1)), DT, SUBSTEPS[kind], start)
    after = trees.advance_state(state, carry, steps, DT)
    assert after['controller']['n'] == 37 + steps
    np.testing.assert_allclose(after['vehicle']['t'], vehicle.t + steps * DT)
    trees.set_state(vehicle, controller, after)
    assert_same_tree(trees.get_state(vehicle, controller), after, rtol=1e-14, atol=1e-14)
    assert controller.n == 37 + steps



@pytest.mark.parametrize('kind', KINDS)
def test_a_rollout_resumed_from_a_state_tree_equals_one_rollout(kind):
    "Phases of the stages that run every few steps are kept: the split is at step 35."
    vehicle, controller = fly(kind, 0)
    state = trees.get_state(vehicle, controller)
    _, _, vehicle_params = rollout.vehicle_arguments(vehicle)
    ctrl = rollout.controller_arguments(controller, vehicle)
    run = getattr(BUILT, 'closed_loop_' + kind)
    targets = np.tile(TARGET, (60, 1))
    state0, start = trees.rollout_state(state)
    _, whole = run(state0, vehicle_params, ctrl, targets, DT, SUBSTEPS[kind], start)
    carry, first = run(state0, vehicle_params, ctrl, targets[:35], DT, SUBSTEPS[kind], start)
    middle = trees.advance_state(state, carry, 35, DT)
    trees.set_state(vehicle, controller, middle)       # through the objects and back
    state0, start = trees.rollout_state(trees.get_state(vehicle, controller))
    _, second = run(state0, vehicle_params, ctrl, targets[35:], DT, SUBSTEPS[kind], start)
    np.testing.assert_allclose(np.concatenate((first, second)), whole, rtol=1e-12, atol=1e-12)



@pytest.mark.parametrize('kind', KINDS)
def test_the_imperative_classes_continue_from_a_rollout(kind):
    "After `set_state`, `Controller` and `Multirotor` fly on as the rollout would."
    vehicle, controller = fly(kind, 20)
    state = trees.get_state(vehicle, controller)
    _, _, vehicle_params = rollout.vehicle_arguments(vehicle)
    ctrl = rollout.controller_arguments(controller, vehicle)
    run = getattr(BUILT, 'closed_loop_' + kind)
    state0, start = trees.rollout_state(state)
    carry, _ = run(state0, vehicle_params, ctrl, np.tile(TARGET, (30, 1)), DT, SUBSTEPS[kind], start)
    middle = trees.advance_state(state, carry, 30, DT)
    state0, start = trees.rollout_state(middle)
    _, expected = run(state0, vehicle_params, ctrl, np.tile(TARGET, (10, 1)), DT, SUBSTEPS[kind], start)
    trees.set_state(vehicle, controller, middle)
    positions = []
    for _ in range(10):
        action = controller.step(TARGET)
        vehicle.step_speeds(vehicle.allocate_control(action[0], action[1:]))
        positions.append(vehicle.position.copy())
    assert np.abs(np.array(positions) - expected[:, :3]).max() < 3e-3



def test_set_state_rejects_the_state_of_another_kind_of_vehicle():
    body_vehicle, body_controller = fly('body', 1)
    motors_vehicle, motors_controller = fly('motors', 1)
    with pytest.raises(ValueError):
        trees.set_state(motors_vehicle, motors_controller, trees.get_state(body_vehicle, body_controller))
