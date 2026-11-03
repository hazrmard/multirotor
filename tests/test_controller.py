"""
Tests of the controller against the frozen oracle in ``tests/legacy/controller.py``.

Each scenario builds the same controller twice, from the current classes and from
the oracle's, and steps both on the same vehicle each step. Only the oracle's
action is applied to the vehicle, so the two controllers always see identical
inputs and any difference is the controllers' own.

The oracle's known bugs are part of the behaviour that the current controller
must reproduce, so that these tests prove that a refactor changes nothing else.
"""
from argparse import Namespace

import numpy as np
import pytest

import legacy.controller as legacy_pid
import multirotor.controller.kernels as kernels
import multirotor.controller.pid as pid
from multirotor.optimize import DEFAULTS
from multirotor.simulation import Multirotor

from common import make_params

STEPS = 400
TOLERANCE = {
    np.float64: dict(rtol=1e-9, atol=1e-10),
    # The oracle computes the altitude controller in float32, and divides by dt
    np.float32: dict(rtol=2e-3, atol=2e-3),
}
CONTROLLERS = ('ctrl_p', 'ctrl_v', 'ctrl_a', 'ctrl_r', 'ctrl_z', 'ctrl_vz')



def make_vehicle(dtype=np.float64) -> Multirotor:
    vp, sp = make_params(8, 'quadratic', False, dtype)
    return Multirotor(vp, sp)



def make_stages(module, vehicle: Multirotor, args: Namespace=DEFAULTS, **kwargs):
    """
    The six stages of the controller of ``optimize.get_controller``, built from
    the classes of ``module`` (the current implementation or the oracle).

    Parameters
    ----------
    module
        Either `multirotor.controller.pid` or `legacy.controller`.
    vehicle : Multirotor
        The vehicle to control.
    args : Namespace
        The defaults to build from.
    kwargs
        ``att_sqrt`` sets the attitude controller's ``square_root_scaling``.
    """
    pos = module.PosController(
        0.8, 0., 3.75, max_err_i=args.max_velocity, vehicle=vehicle,
        max_velocity=args.max_velocity, max_acceleration=args.max_acceleration,
        square_root_scaling=args.sqrt_scaling, leashing=args.leashing)
    vel = module.VelController(
        1, 0., 0, max_err_i=args.max_acceleration, vehicle=vehicle,
        max_tilt=args.max_tilt)
    att = module.AttController(
        [1., 1., 0.], 0, 0., max_err_i=args.max_err_i_attitude, vehicle=vehicle,
        square_root_scaling=kwargs.get('att_sqrt', False))
    rat = module.RateController(
        [4, 4, 0], 0, [40, 40, 0], max_err_i=args.max_err_i_rate, vehicle=vehicle)
    alt = module.AltController(
        1, 0, 0, max_err_i=args.max_velocity, vehicle=vehicle,
        max_velocity=args.max_velocity)
    alt_rate = module.AltRateController(
        10, 0, 0, max_err_i=args.max_acceleration, vehicle=vehicle)
    return pos, vel, att, rat, alt, alt_rate



def make_controller(module, vehicle: Multirotor, args: Namespace=DEFAULTS, **kwargs):
    "A `Controller` of the stages of `make_stages`, with the same arguments."
    return module.Controller(
        *make_stages(module, vehicle, args, **kwargs),
        period_p=0.1, period_a=0.01, period_z=0.1)



def waypoints(i: int) -> np.ndarray:
    "Far waypoints, so that the leash and the square-root controller are active."
    if i < STEPS // 2:
        return np.asarray([6., -4., 10., 0.5])
    return np.asarray([-3., 5., 8., -0.3])



def assert_same(new, old, tol: dict, context: str, check_state: bool=True) -> None:
    "Compare what `DataLog` and the optimizer read from two controllers."
    np.testing.assert_allclose(new.action, old.action, **tol, err_msg=context + ' action')
    if not check_state:
        return
    np.testing.assert_allclose(new.state, old.state, **tol, err_msg=context + ' state')
    np.testing.assert_allclose(new.reference, old.reference, **tol, err_msg=context + ' reference')
    for name in CONTROLLERS:
        n, o = getattr(new, name), getattr(old, name)
        for attr in ('reference', 'action', 'err', 'err_p', 'err_i', 'err_d'):
            np.testing.assert_allclose(
                getattr(n, attr), getattr(o, attr), **tol, err_msg=f'{context} {name}.{attr}')



def run(new, old, vehicle, tol, steps=STEPS, refs=waypoints, step_kwargs=None,
        check_state=True, after_step=None) -> None:
    """
    Step both controllers on the vehicle and compare them, driving the vehicle
    with the oracle's action.

    Parameters
    ----------
    refs : Callable
        Reference at step ``i``.
    step_kwargs : Callable, optional
        Extra arguments of ``step`` at step ``i``. May include ``persist=False``,
        which makes the loop take an extra step with ``persist=True`` after it.
    """
    for i in range(steps):
        kwargs = step_kwargs(i) if step_kwargs is not None else {}
        ref = refs(i).astype(vehicle.dtype)
        a_old = old.step(reference=ref, **kwargs)
        a_new = new.step(reference=ref, **kwargs)
        np.testing.assert_allclose(a_new, a_old, **tol, err_msg=f'step {i} return')
        assert_same(new, old, tol, f'step {i}', check_state)
        if kwargs.get('persist', True):
            vehicle.step_speeds(vehicle.allocate_control(a_old[0], a_old[1:4]))
        if after_step is not None:
            after_step(i)



@pytest.fixture
def coverage(monkeypatch):
    "Counts the oracle's executions of the branches that the scenarios must reach."
    counts = {'sqrt': 0, 'leash': 0}
    sqrt = legacy_pid.sqrt_control
    leash = legacy_pid.PosController.leash.fget

    def counted_sqrt(*args):
        counts['sqrt'] += 1
        return sqrt(*args)

    def counted_leash(self):
        value = leash(self)
        counts['leash'] += int(np.isfinite(value))
        return value

    monkeypatch.setattr(legacy_pid, 'sqrt_control', counted_sqrt)
    monkeypatch.setattr(legacy_pid.PosController, 'leash', property(counted_leash))
    return counts



@pytest.mark.parametrize('dtype', (np.float64, np.float32))
def test_waypoint_flight_matches_oracle(dtype, coverage):
    vehicle = make_vehicle(dtype)
    new, old = make_controller(pid, vehicle), make_controller(legacy_pid, vehicle)
    # float32 state differs by round-off in dt-divided terms, so compare returns only
    run(new, old, vehicle, TOLERANCE[dtype], check_state=dtype is np.float64)
    assert coverage['sqrt'] > 0 and coverage['leash'] > 0, 'branches were not reached'



@pytest.mark.parametrize('args', (
    dict(leashing=False), dict(sqrt_scaling=False),
    dict(leashing=False, sqrt_scaling=False),
), ids=('no_leash', 'no_sqrt', 'plain_pid'))
def test_position_options_match_oracle(args):
    vehicle = make_vehicle()
    options = Namespace(**{**vars(DEFAULTS), **args})
    new = make_controller(pid, vehicle, options)
    old = make_controller(legacy_pid, vehicle, options)
    run(new, old, vehicle, TOLERANCE[np.float64])



def test_attitude_square_root_matches_oracle():
    vehicle = make_vehicle()
    new = make_controller(pid, vehicle, att_sqrt=True)
    old = make_controller(legacy_pid, vehicle, att_sqrt=True)
    run(new, old, vehicle, TOLERANCE[np.float64])



def test_set_params_matches_oracle():
    """
    Integral and derivative terms, and array-valued limits as the optimizer sets
    them. The oracle accumulated the integral wrongly unless k_i was 1 (or 0), so
    the integral gains are 1 here.
    """
    vehicle = make_vehicle()
    new, old = make_controller(pid, vehicle), make_controller(legacy_pid, vehicle)
    params = dict(
        ctrl_p=dict(k_i=1., k_d=2.),
        ctrl_v=dict(k_p=2., k_i=1., k_d=0.1),
        ctrl_a=dict(k_p=np.asarray([3., 3., 1.]), k_i=np.asarray([1., 1., 1.]),
                    k_d=np.asarray([.5, .5, .1])),
        ctrl_r=dict(k_p=np.asarray([5., 5., 2.]), k_i=np.asarray([1., 1., 1.]),
                    k_d=np.asarray([30., 30., 5.]),
                    max_acceleration=np.asarray([2., 2., 1.]),
                    max_err_i=np.asarray([2., 2., 1.])),
        ctrl_vz=dict(k_p=12., k_i=1., k_d=0.2),
    )
    for ctrl in (new, old):
        ctrl.set_params(**params)
    np.testing.assert_allclose(new.ctrl_r.max_acceleration, old.ctrl_r.max_acceleration)
    run(new, old, vehicle, TOLERANCE[np.float64])



def test_get_params_matches_oracle():
    vehicle = make_vehicle()
    new, old = make_controller(pid, vehicle), make_controller(legacy_pid, vehicle)
    p_new, p_old = new.get_params(), old.get_params()
    assert p_new.keys() == p_old.keys()
    for name in p_old:
        if name == 'feedforward_weight':
            assert p_new[name] == p_old[name]
            continue
        assert p_new[name].keys() == p_old[name].keys(), name
        for param in p_old[name]:
            if (name, param) == ('ctrl_vz', 'max_err_i'):
                # The oracle truncated it to an integer, because the controller
                # was built with integer gains. Deliberately not reproduced.
                assert p_old[name][param] == 2 and p_new[name][param] == 2.5
                continue
            # Values are compared after broadcasting: shapes differ, for example
            # a limit that was a number is now an array with one value per channel
            np.testing.assert_allclose(
                p_new[name][param], p_old[name][param], err_msg=f'{name}.{param}')



def test_reference_is_error_matches_oracle():
    vehicle = make_vehicle()
    new, old = make_controller(pid, vehicle), make_controller(legacy_pid, vehicle)
    run(new, old, vehicle, TOLERANCE[np.float64],
        refs=lambda i: np.asarray([2., -1., 1., 0.2]),
        step_kwargs=lambda i: dict(ref_is_error=True))



def test_feed_forward_matches_oracle():
    vehicle = make_vehicle()
    new, old = make_controller(pid, vehicle), make_controller(legacy_pid, vehicle)
    for ctrl in (new, old):
        ctrl.feedforward_weight = 0.6
    run(new, old, vehicle, TOLERANCE[np.float64],
        step_kwargs=lambda i: dict(feed_forward_velocity=np.asarray([1., -2., 0.])))



def test_persist_false_leaves_the_controller_unchanged():
    """
    A step with ``persist=False`` returns what the step would return, and changes
    nothing: the controller then stays in step with an oracle that never saw it.
    """
    vehicle = make_vehicle()
    new, old = make_controller(pid, vehicle), make_controller(legacy_pid, vehicle)
    tol = TOLERANCE[np.float64]
    for i in range(STEPS):
        ref = waypoints(i).astype(vehicle.dtype)
        if i % 7 == 3:
            data, n, action = new._data.state.copy(), new.n, new.action.copy()
            probe = new.step(reference=ref, persist=False)
            np.testing.assert_array_equal(new._data.state, data, err_msg=f'step {i}')
            assert new.n == n
            np.testing.assert_array_equal(new.action, action)
        a_old, a_new = old.step(reference=ref), new.step(reference=ref)
        if i % 7 == 3:
            np.testing.assert_allclose(probe, a_new, **tol, err_msg=f'step {i} probe')
        np.testing.assert_allclose(a_new, a_old, **tol, err_msg=f'step {i}')
        vehicle.step_speeds(vehicle.allocate_control(a_old[0], a_old[1:4]))



def test_reset_matches_oracle():
    vehicle = make_vehicle()
    new, old = make_controller(pid, vehicle), make_controller(legacy_pid, vehicle)
    run(new, old, vehicle, TOLERANCE[np.float64], steps=120)
    for ctrl in (new, old):
        ctrl.reset()
    vehicle.reset()
    run(new, old, vehicle, TOLERANCE[np.float64], steps=120)



def test_vehicle_swap_matches_oracle():
    first, second = make_vehicle(), make_vehicle()
    new, old = make_controller(pid, first), make_controller(legacy_pid, first)
    run(new, old, first, TOLERANCE[np.float64], steps=60)
    for ctrl in (new, old):
        ctrl.vehicle = second
        ctrl.reset()
    run(new, old, second, TOLERANCE[np.float64], steps=60)



def test_standalone_pid_matches_oracle():
    "A `PIDController` without a vehicle."
    rng = np.random.default_rng(0)
    kwargs = dict(k_p=np.asarray([2., 1., 3.]), k_i=np.asarray([1., 1., 0.]),
                  k_d=np.asarray([.1, 0., .3]), max_err_i=np.asarray([.5, 1., 1.]))
    new, old = pid.PIDController(**kwargs), legacy_pid.PIDController(**kwargs)
    for i in range(50):
        ref, mea = rng.normal(size=3), rng.normal(size=3)
        persist = i % 5 != 0
        np.testing.assert_allclose(
            new.step(ref, mea, dt=0.02, persist=persist),
            old.step(ref, mea, dt=0.02, persist=persist), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(new.state, old.state, rtol=1e-9, atol=1e-12)




def test_stages_are_views_of_one_data_object():
    vehicle = make_vehicle()
    ctrl = make_controller(pid, vehicle)
    stages = [getattr(ctrl, name) for name in CONTROLLERS]
    assert all(stage._data is ctrl._data for stage in stages)
    assert ctrl._data.params.shape[1] == sum(stage.n for stage in stages) == 12
    # Writing through a stage changes the shared matrix, and the reverse
    ctrl.ctrl_a.k_p[:] = 7.
    assert np.all(ctrl._data.params[pid.PID_K_P, 4:7] == 7.)
    ctrl._data.params[pid.PID_K_D, 7:10] = 3.
    assert np.all(ctrl.ctrl_r.k_d == 3.)
    assert np.shares_memory(ctrl.ctrl_r.k_d, ctrl._data.params)



def test_controller_keeps_the_parameters_of_the_stages_it_takes_over():
    vehicle = make_vehicle()
    stages = make_stages(pid, vehicle)
    stages[3].set_params(k_p=9., max_acceleration=np.asarray([1., 2., 3.]))
    controller = pid.Controller(*stages)
    assert all(stage._data is controller._data for stage in stages)
    np.testing.assert_array_equal(controller.ctrl_r.k_p, 9.)
    np.testing.assert_array_equal(controller.ctrl_r.max_acceleration, [1., 2., 3.])



def test_a_stage_belongs_to_one_controller():
    vehicle = make_vehicle()
    stages = make_stages(pid, vehicle)
    first = pid.Controller(*stages)
    with pytest.raises(ValueError, match='already belongs'):
        pid.Controller(*stages)
    assert all(stage._data is first._data for stage in stages), 'the first keeps its stages'



def test_a_controller_rejects_one_stage_in_two_roles():
    vehicle = make_vehicle()
    stages = list(make_stages(pid, vehicle))
    stages[1] = stages[0]
    with pytest.raises(ValueError, match='different object'):
        pid.Controller(*stages)
    assert not any(stage._adopted for stage in stages), 'no stage moved'



def test_standalone_stage_owns_its_data():
    vehicle = make_vehicle()
    stage = pid.AttController(1., 0., 0., 0.1, vehicle=vehicle)
    assert stage._data.params.shape[1] == stage.n == 3
    assert stage.k_p.shape == (3,)



@pytest.mark.parametrize('err, expected', ((1., 2.), (-1., -2.), (0., 0.)))
def test_sqrt_control_without_p_gain_follows_the_acceleration_limit(err, expected):
    "With k_p=0 the correction is sqrt(2 * limit * |err|), signed. It was 0 for err > 0."
    assert kernels.sqrt_control(err, 0., 2., 0.) == pytest.approx(expected)



def test_integral_term_is_the_scaled_trapezoid_of_the_error():
    "Three steps of error 1 at dt=1 accumulate 0.5 + 1 + 1. It used to compound k_i."
    stage = pid.PIDController(np.zeros(1), 2 * np.ones(1), np.zeros(1), np.asarray([1e9]))
    for _ in range(3):
        action = stage.step(np.ones(1), np.zeros(1), dt=1.)
    np.testing.assert_allclose(stage.integral, 2.5)
    np.testing.assert_allclose(action, 2 * 2.5)



def test_accumulated_error_is_limited_before_it_is_scaled():
    stage = pid.PIDController(np.zeros(1), 3 * np.ones(1), np.zeros(1), np.asarray([1.]))
    for _ in range(5):
        action = stage.step(np.ones(1), np.zeros(1), dt=1.)
    np.testing.assert_allclose(stage.integral, 1.)
    np.testing.assert_allclose(action, 3.)



def test_accumulated_error_is_not_limited_without_a_maximum():
    stage = pid.PIDController(np.zeros(1), np.ones(1), np.zeros(1), None)
    for _ in range(3):
        stage.step(np.full(1, 1e6), np.zeros(1), dt=1.)
    assert np.isinf(stage.max_err_i).all()
    np.testing.assert_allclose(stage.integral, 2.5e6)



def test_set_params_reaches_every_stage():
    "The altitude controller was skipped, and the attitude controller set twice."
    ctrl = make_controller(pid, make_vehicle())
    values = {name: 3. + i for i, name in enumerate(CONTROLLERS)}
    ctrl.set_params(**{name: dict(k_p=value) for name, value in values.items()})
    for name, value in values.items():
        np.testing.assert_array_equal(getattr(ctrl, name).k_p, value, err_msg=name)
    np.testing.assert_array_equal(ctrl.get_params()['ctrl_z']['k_p'], values['ctrl_z'])
