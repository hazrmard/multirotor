"""
Tests for `FleetController`: a fleet in closed loop must fly like separate
vehicles, each with its own `Controller`, and the fleet's controllers must accept
only what the fleet's arrays can hold.

As in ``test_fleet.py``, a tight-tolerance integrator takes the solver's error out
of the comparison.
"""
from copy import deepcopy

import numpy as np
import pytest

from multirotor.controller import FleetController
from multirotor.controller.data import PID_K_P
from multirotor.fleet import Fleet
from multirotor.optimize import get_controller, get_fleet_controller
from multirotor.simulation import Multirotor
from multirotor.vehicle import BatteryParams

from test_fleet import BATTERY, K, make_fleet_params, stacked, tight

STEPS = 400
config = pytest.mark.parametrize('kwargs', (
    dict(motor=False, battery=None),
    dict(motor=True, battery=None),
    dict(motor=True, battery=BATTERY),
), ids=('no motors', 'motors', 'motors and battery'))



def make_controllers(singles, spread=True):
    "A controller for each vehicle. With ``spread``, their gains differ."
    controllers = []
    for k, vehicle in enumerate(singles):
        controller = get_controller(vehicle)
        if spread:
            controller.set_params(
                ctrl_a=dict(k_p=np.asarray([8. + k, 8. + k, 0.])),
                ctrl_r=dict(k_p=np.asarray([32. - 2 * k, 32. - 2 * k, 0.])),
                ctrl_vz=dict(k_p=np.asarray([10. + k])))
        controllers.append(controller)
    return controllers



def start_state() -> np.ndarray:
    """
    A different state for each vehicle, moving horizontally. A vehicle at rest has
    a speed and acceleration of rounding-error size, and the position stage's
    leash compares those two, so vehicles at rest are not a fair comparison.
    """
    state = np.zeros((K, 12))
    state[:, 0] = 0.3 * np.arange(K)
    state[:, 1] = -0.2 * np.arange(K)
    state[:, 2] = 0.5
    state[:, 3] = 0.4 + 0.1 * np.arange(K)
    state[:, 4] = -0.3 + 0.05 * np.arange(K)
    return state



def make_closed_loop(kwargs, spread=True, **overrides):
    "Separate vehicles with controllers, and the same as a fleet."
    plist, sp = make_fleet_params(**kwargs)
    singles = [Multirotor(p, sp, tight) for p in plist]
    for vehicle, state in zip(singles, start_state()):
        vehicle.state[:] = state
    controllers = make_controllers(singles, spread)
    for controller in controllers:
        for name, value in overrides.items():
            setattr(controller, name, value)
    fleet = Fleet(plist, sp, tight)
    fleet.reset(start_state())
    return singles, controllers, fleet, FleetController(fleet, controllers)



def waypoint(i: int) -> np.ndarray:
    "A target for each vehicle, which changes every 100 steps."
    corners = np.asarray([[4., 0., 3.], [4., 4., 3.], [0., 4., 2.], [0., 0., 3.]])
    return np.stack([[*(corners[(i // 100 + k) % 4]), 0.1 * k] for k in range(K)])



def fly(singles, controllers, fleet, fleet_controller, steps=STEPS, kwargs_at=None,
        reference_at=waypoint, atol=1e-5):
    """
    Fly both in closed loop, each with its own action, and compare the actions and
    states at every step. ``kwargs_at(i)`` gives the arguments of ``step`` at step
    ``i``, and ``reference_at(i)`` the reference, shape (K, 4).
    """
    for i in range(steps):
        kwargs = kwargs_at(i) if kwargs_at is not None else {}
        reference = reference_at(i)
        actions = np.stack([
            c.step(r, **{k: (v[j] if k == 'feed_forward_velocity' else v)
                         for k, v in kwargs.items()})
            for j, (c, r) in enumerate(zip(controllers, reference))])
        fleet_actions = fleet_controller.step(reference, **kwargs)
        np.testing.assert_allclose(fleet_actions, actions, atol=atol, rtol=1e-4,
                                   err_msg=f'step {i}')
        if kwargs.get('persist', True):
            for vehicle, a in zip(singles, actions):
                vehicle.step_speeds(vehicle.allocate_control(a[0], a[1:4]))
            fleet.step_speeds(fleet.allocate_control(fleet_actions[:, 0], fleet_actions[:, 1:4]))
            np.testing.assert_allclose(
                fleet.state, stacked(singles, 'state'), atol=1e-4, err_msg=f'step {i}')



@config
def test_fleet_flies_like_separate_vehicles(kwargs):
    fly(*make_closed_loop(kwargs))



def test_multi_rate_periods_and_feed_forward_and_persist():
    # Periods that are not multiples of each other, feed-forward velocity, steps
    # that do not persist, and an error as the reference
    system = make_closed_loop(
        dict(motor=False, battery=None), feedforward_weight=0.4, steps_p=7,
        steps_a=2, steps_z=5)
    ff = np.stack([np.asarray([0.5, -0.3]) * (k + 1) for k in range(K)])

    def kwargs_at(i):
        if i % 11 == 3:
            return dict(persist=False)
        if i % 3 == 0:
            return dict(feed_forward_velocity=ff)
        return {}

    fly(*system, kwargs_at=kwargs_at)


def test_reference_as_error():
    def error_at(i):
        return np.stack([[0.5, -0.2, 0.3, 0.]] * K) * (1 + 0.1 * i % 5)

    fly(*make_closed_loop(dict(motor=False, battery=None)), steps=100,
        reference_at=error_at, kwargs_at=lambda i: dict(ref_is_error=True))


def test_one_controller_for_all_vehicles_and_defaults():
    plist, sp = make_fleet_params()
    singles = [Multirotor(p, sp, tight) for p in plist]
    template = get_controller(singles[0])
    fleet = Fleet(plist, sp, tight)
    fleet_controller = FleetController(fleet, template)
    # The vehicles differ in mass, but the gains are those of the template
    assert np.all(fleet_controller.params == template._data.params)
    defaults = get_fleet_controller(fleet)
    assert np.all(defaults.params == fleet_controller.params)
    # The template is a copy: stepping the fleet leaves it unchanged
    state = template._data.state.copy()
    for i in range(20):
        fleet_controller.step(waypoint(i))
        fleet.step_speeds(np.tile(fleet.allocate_control(
            fleet.weight, np.zeros((K, 3)))[0], (K, 1)))
    assert np.all(template._data.state == state)
    assert not np.all(fleet_controller.state == 0)


def test_reset_restores_the_start():
    singles, controllers, fleet, fleet_controller = make_closed_loop(
        dict(motor=False, battery=None))
    first = []
    for round_ in range(2):
        fleet.reset(start_state())
        fleet_controller.reset()
        actions = []
        for i in range(30):
            a = fleet_controller.step(waypoint(i))
            fleet.step_speeds(fleet.allocate_control(a[:, 0], a[:, 1:4]))
            actions.append(a)
        first.append(np.stack(actions))
    np.testing.assert_array_equal(first[0], first[1])


def test_gains_are_per_vehicle():
    singles, controllers, fleet, fleet_controller = make_closed_loop(
        dict(motor=False, battery=None))
    k_p = fleet_controller.params[:, PID_K_P, 4:7]    # attitude stage
    assert k_p.shape == (K, 3)
    assert np.allclose(k_p[:, 0], [8. + k for k in range(K)])
    # set_params and get_params use the nested format of `Controller`
    fleet_controller.set_params(2, ctrl_a=dict(k_p=np.asarray([5., 5., 0.])))
    assert np.allclose(fleet_controller.params[2, PID_K_P, 4:7], [5., 5., 0.])
    assert np.allclose(fleet_controller.params[1, PID_K_P, 4:7], [9., 9., 0.])
    assert np.allclose(fleet_controller.get_params(2)['ctrl_a']['k_p'], [5., 5., 0.])
    # what get_params returns belongs to the caller: changing it changes nothing else
    first = fleet_controller.get_params(2)
    first['ctrl_a']['k_p'][:] = 0.
    assert np.allclose(fleet_controller.params[2, PID_K_P, 4:7], [5., 5., 0.])
    # and it acts: the vehicle with changed gains no longer flies like its controller
    controllers[2].set_params(ctrl_a=dict(k_p=np.asarray([5., 5., 0.])))
    fly(singles, controllers, fleet, fleet_controller, steps=100)


def test_a_setting_outside_the_arrays_does_not_carry_over_to_other_vehicles():
    # max_jerk is only used to make the position gain. It is not in the arrays.
    singles, controllers, fleet, fleet_controller = make_closed_loop(
        dict(motor=False, battery=None))
    before = fleet_controller.get_params(1)['ctrl_p']['max_jerk']
    fleet_controller.set_params(0, ctrl_p=dict(max_jerk=12345.))
    assert fleet_controller.get_params(1)['ctrl_p']['max_jerk'] == before
    assert before != 12345.



def test_structure_must_be_the_same_for_all_vehicles():
    plist, sp = make_fleet_params()
    singles = [Multirotor(p, sp) for p in plist]
    controllers = make_controllers(singles)
    fleet = Fleet(plist, sp)
    with pytest.raises(ValueError, match='one per vehicle'):
        FleetController(fleet, controllers[:-1])
    other_period = deepcopy(controllers)
    other_period[1].steps_p = 3
    with pytest.raises(ValueError, match='same periods'):
        FleetController(fleet, other_period)
    other_option = make_controllers(singles)
    other_option[0].ctrl_a.square_root_scaling = True
    with pytest.raises(ValueError, match='same periods'):
        FleetController(fleet, other_option)
    other_dt = make_fleet_params()[0]
    _, sp_other = make_fleet_params()
    sp_other = type(sp)(dt=0.02, g=sp.g, dtype=sp.dtype)
    slow = [get_controller(Multirotor(p, sp_other)) for p in other_dt]
    with pytest.raises(ValueError, match='time step'):
        FleetController(fleet, slow)
    fleet_controller = FleetController(fleet, controllers)
    with pytest.raises(ValueError, match='same for all'):
        fleet_controller.set_params(0, ctrl_p=dict(leashing=not fleet_controller.leashing))
