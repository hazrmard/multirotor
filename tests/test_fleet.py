"""
Tests for `Fleet`: it must reproduce separate `Multirotor` vehicles, and reject
fleets that it cannot hold in arrays.

The kernels are compared with a tight-tolerance integrator on both sides, so that
the comparison does not depend on the solver's error. With the default
integrator the fleet's vehicles share a step size, so the results can differ from
those of separate vehicles by the solver's tolerance. The main test runs with both.
"""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import odeint

from multirotor.fleet import Fleet
from multirotor.simulation import Multirotor, odeint_integrator
from multirotor.vehicle import BatteryParams

from common import hover_signals, make_params, schedule

K = 4
STEPS = 40
BATTERY = BatteryParams(max_voltage=22.2, capacity=16., internal_resistance=0.05)

CONFIGS = {
    'no motors': dict(motor=False, battery=None),
    'motors': dict(motor=True, battery=None),
    'motors and battery': dict(motor=True, battery=BATTERY),
}
config = pytest.mark.parametrize('kwargs', CONFIGS.values(), ids=CONFIGS.keys())



def tight(fun, x0, dt, args=()):
    "An integrator whose error is far below the differences that the tests look for."
    return odeint(fun, x0, (0, dt), args=args, rtol=1e-9, atol=1e-9, tfirst=True,
                  mxstep=20000)[-1]



def make_fleet_params(n=6, count=K, dtype=np.float64, motor=False, battery=None):
    """
    The parameters of ``count`` vehicles that differ in mass and inertia, and the
    simulation parameters.
    """
    plist = []
    for k in range(count):
        vp, sp = make_params(n, 'quadratic', motor, dtype, battery=battery)
        vp = replace(vp, mass=vp.mass * (1 + 0.1 * k))
        vp.inertia_matrix = vp.inertia_matrix * (1 + 0.1 * k)
        vp.inertia_matrix_inverse = np.linalg.inv(vp.inertia_matrix)
        plist.append(vp)
    return plist, sp



def signals(plist, sp, steps=STEPS) -> np.ndarray:
    "Speed signals around hover, shape (steps, K, n), different for each vehicle."
    base = hover_signals(plist[0], sp)
    return np.stack([schedule(base, 0.05, steps, seed=k) for k in range(len(plist))], axis=1)



def step_both(singles, fleet, u):
    "Step the separate vehicles and the fleet with signals ``u`` of shape (K, n)."
    for vehicle, signal in zip(singles, u):
        vehicle.step_speeds(signal)
    fleet.step_speeds(u)



def stacked(singles, attribute):
    return np.stack([getattr(v, attribute) for v in singles])



@config
@pytest.mark.parametrize('integrator, atol', (
    (tight, 1e-7),
    # The fleet's vehicles share a step size, so with the default integrator
    # (rtol = atol = 1e-4) they match separate vehicles to its tolerance
    (odeint_integrator, 5e-4),
), ids=('tight', 'default'))
def test_fleet_matches_separate_vehicles(kwargs, integrator, atol):
    plist, sp = make_fleet_params(**kwargs)
    singles = [Multirotor(p, sp, integrator) for p in plist]
    fleet = Fleet(plist, sp, integrator)
    for u in signals(plist, sp):
        step_both(singles, fleet, u)
        np.testing.assert_allclose(fleet.state, stacked(singles, 'state'), atol=atol)
    # The other outputs, after the last step
    np.testing.assert_allclose(
        fleet.acceleration, stacked(singles, 'acceleration'), atol=10 * atol)
    np.testing.assert_allclose(fleet.speeds, stacked(singles, 'speeds'), atol=10 * atol)
    np.testing.assert_allclose(fleet.euler_rate, stacked(singles, 'euler_rate'), atol=atol)
    np.testing.assert_allclose(
        fleet.inertial_velocity, stacked(singles, 'inertial_velocity'), atol=atol)
    np.testing.assert_allclose(fleet.weight, [v.weight for v in singles])
    assert fleet.t == pytest.approx(singles[0].t)
    if kwargs['motor']:
        np.testing.assert_allclose(
            fleet.motor_currents, stacked(singles, 'motor_currents'), atol=10 * atol)
    np.testing.assert_allclose(
        fleet.battery_current, [v.battery_current for v in singles], atol=10 * atol)
    if kwargs['battery'] is not None:
        np.testing.assert_allclose(fleet.soc, [v.battery.soc for v in singles], atol=atol)
        np.testing.assert_allclose(
            fleet.battery_voltage, [v.battery.voltage for v in singles], atol=10 * atol)



def test_step_dynamics_matches_separate_vehicles():
    plist, sp = make_fleet_params(motor=True)
    singles = [Multirotor(p, sp, tight) for p in plist]
    fleet = Fleet(plist, sp, tight)
    rng = np.random.default_rng(0)
    for _ in range(STEPS):
        u = rng.normal(0, 1, (K, 6)) + np.asarray([0, 0, 30, 0, 0, 0])
        for vehicle, forces_torques in zip(singles, u):
            vehicle.step_dynamics(forces_torques)
        fleet.step_dynamics(u)
        np.testing.assert_allclose(fleet.state, stacked(singles, 'state'), atol=1e-7)



def test_disturbances_reach_each_vehicle():
    plist, sp = make_fleet_params()
    singles = [Multirotor(p, sp, tight) for p in plist]
    fleet = Fleet(plist, sp, tight)
    force = np.arange(K * 3, dtype=float).reshape(K, 3) / 10
    torque = np.full((K, 3), 0.01)
    for vehicle, f, t in zip(singles, force, torque):
        vehicle.step_speeds(hover_signals(plist[0], sp), f, t)
    fleet.step_speeds(np.tile(hover_signals(plist[0], sp), (K, 1)), force, torque)
    np.testing.assert_allclose(fleet.state, stacked(singles, 'state'), atol=1e-7)



def test_allocate_control_matches_separate_vehicles():
    plist, sp = make_fleet_params()
    singles = [Multirotor(p, sp) for p in plist]
    fleet = Fleet(plist, sp)
    thrust = np.asarray([v.weight for v in singles]) * 1.1
    torques = np.random.default_rng(1).normal(0, 0.05, (K, 3))
    expected = np.stack([v.allocate_control(f, t) for v, f, t in zip(singles, thrust, torques)])
    np.testing.assert_allclose(fleet.allocate_control(thrust, torques), expected, rtol=1e-6)



@config
def test_float32_fleet_keeps_float32_precision_and_tracks_float64(kwargs):
    plist64, sp64 = make_fleet_params(dtype=np.float64, **kwargs)
    plist32, sp32 = make_fleet_params(dtype=np.float32, **kwargs)
    fleet64, fleet32 = Fleet(plist64, sp64), Fleet(plist32, sp32)
    for u in signals(plist64, sp64, 20):
        fleet64.step_speeds(u)
        fleet32.step_speeds(u)
        state = fleet32.state
        assert np.array_equal(state, state.astype(np.float32).astype(np.float64))
    np.testing.assert_allclose(fleet32.state, fleet64.state, atol=1e-3)



@config
def test_state_is_one_array_that_the_fleet_updates_in_place(kwargs):
    plist, sp = make_fleet_params(**kwargs)
    fleet = Fleet(plist, sp)
    state = fleet.state
    assert state.shape == (K, 12)
    assert np.shares_memory(state, fleet.position)
    assert np.shares_memory(state, fleet.orientation)
    fleet.position[:] = [1, 2, 3]
    assert np.all(state[:, :3] == [1, 2, 3])
    fleet.step_speeds(np.tile(hover_signals(plist[0], sp), (K, 1)))
    assert fleet.state is not None and np.shares_memory(state, fleet.state)
    assert np.all(state[:, :3] != 0)
    fleet.reset()
    assert np.all(state == 0) and np.shares_memory(state, fleet.state)
    assert np.all(fleet.speeds == 0)
    fleet.reset(np.ones(12))
    assert np.all(state == 1)



def test_battery_cutoff_and_empty_batteries_match_separate_vehicles():
    # The vehicles' batteries differ: one runs empty, one cuts off at its
    # minimum voltage, one is large enough to last.
    plist, sp = make_fleet_params(motor=True, count=3)
    small = BatteryParams(max_voltage=22.2, capacity=0.001, internal_resistance=0.05)
    sagging = BatteryParams(
        max_voltage=22.2, capacity=16., internal_resistance=0.8, min_voltage=21.)
    large = BatteryParams(max_voltage=22.2, capacity=16., internal_resistance=0.05)
    for params, battery in zip(plist, (small, sagging, large)):
        params.battery = battery
    singles = [Multirotor(p, sp, tight) for p in plist]
    fleet = Fleet(plist, sp, tight)
    for u in signals(plist, sp, 60):
        step_both(singles, fleet, u)
        np.testing.assert_allclose(fleet.state, stacked(singles, 'state'), atol=1e-6)
        assert list(fleet.battery_cutoff) == [v.battery.cutoff for v in singles]
        np.testing.assert_allclose(fleet.soc, [v.battery.soc for v in singles], atol=1e-9)
    assert fleet.soc[0] <= 0., 'the small battery ran empty'
    assert fleet.battery_cutoff[1], 'the sagging battery cut off'
    assert not fleet.battery_cutoff[2]



def test_current_without_a_battery_model():
    plist, sp = make_fleet_params(motor=True, count=2)
    plist[0].battery = BatteryParams(max_voltage=22.2)
    plist[1].battery = BatteryParams(max_voltage=22.2)
    singles = [Multirotor(p, sp) for p in plist]
    fleet = Fleet(plist, sp)
    for u in signals(plist, sp, 5):
        step_both(singles, fleet, u[:2])
    np.testing.assert_allclose(
        fleet.battery_current, [v.battery_current for v in singles], rtol=1e-4)
    plain, sp = make_fleet_params(motor=True, count=2)
    fleet = Fleet(plain, sp)
    fleet.step_speeds(np.tile(hover_signals(plain[0], sp), (2, 1)))
    assert np.all(np.isnan(fleet.battery_current)), 'an infinite maximum voltage'



def test_fleets_that_cannot_be_held_in_arrays_are_rejected():
    plist, sp = make_fleet_params(motor=True, battery=BATTERY)
    with pytest.raises(ValueError, match='at least one'):
        Fleet([], sp)
    other_n, _ = make_fleet_params(n=4, count=1, motor=True, battery=BATTERY)
    with pytest.raises(ValueError, match='number of propellers'):
        Fleet(plist + other_n, sp)
    no_motor, _ = make_fleet_params(count=1, motor=False)
    with pytest.raises(ValueError, match='motors'):
        Fleet(plist + no_motor, sp)
    no_model, _ = make_fleet_params(count=1, motor=True, battery=None)
    with pytest.raises(ValueError, match='battery model'):
        Fleet(plist + no_model, sp)
    table = BatteryParams(
        capacity=16., internal_resistance=0.05, ocv_soc=[0., 0.5, 1.],
        ocv_voltage=[18., 21., 22.2], max_voltage=22.2)
    other_table = replace(table, ocv_soc=[0., 1.], ocv_voltage=[18., 22.2])
    first, second = make_fleet_params(count=2, motor=True, battery=table)[0], \
        make_fleet_params(count=1, motor=True, battery=other_table)[0]
    with pytest.raises(ValueError, match='table length'):
        Fleet(first + second, sp)
    blade_element, bp = make_params(4, 'fsolve', False)
    with pytest.raises(NotImplementedError, match='blade-element'):
        Fleet([blade_element], bp)



def test_wrong_signal_shape_is_an_error():
    plist, sp = make_fleet_params()
    fleet = Fleet(plist, sp)
    with pytest.raises(ValueError, match='shape'):
        fleet.step_speeds(np.zeros((K, 3)))
    with pytest.raises(ValueError, match='shape'):
        fleet.step_speeds(np.zeros(6))
