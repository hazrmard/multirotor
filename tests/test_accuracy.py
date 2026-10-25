"""
Accuracy tests for vehicles with motors, which integrate the motors and the body
together.

The reference is the same right-hand side integrated with tight tolerances. The
first tests check that this right-hand side matches the oracle's physics; the
rest check that `Multirotor.step_speeds` follows the reference closely.
"""
from copy import deepcopy

import numpy as np
import pytest
from scipy.integrate import odeint

from multirotor.data import MotorData
from multirotor.dynamics import N_BODY_STATES
from multirotor.models.tarott18 import MP, PP
from multirotor.physics import motor_acc
from multirotor.simulation import Motor, Multirotor
from multirotor.vehicle import BatteryParams, SimulationParams

from common import (
    N_SCHEDULE, hover_signals, make_pair, make_params, motor_signal,
    nominal_speed, schedule, settle_at_hover
)
from legacy.simulation import Motor as LegacyMotor, Multirotor as LegacyMultirotor

REFERENCE_TOL = 1e-10



def reference_trajectory(vehicle: Multirotor, us: np.ndarray) -> np.ndarray:
    """
    Integrate the vehicle's right-hand side with tight tolerances, holding each
    speed signal for one time step. Starts from the vehicle's current state.

    Returns
    -------
    np.ndarray
        The body states and motor speeds after each step, shape (steps, 12 + n).
    """
    y = np.concatenate((vehicle.state.astype(float), vehicle._data.motors.speed))
    out = []
    for u in us:
        args = (np.asarray(u, float), float(vehicle.battery.voltage), np.zeros(6)) \
               + vehicle._kernel_params
        y = odeint(vehicle._rhs, y, (0, vehicle.simulation.dt), args=args,
                   rtol=REFERENCE_TOL, atol=REFERENCE_TOL, tfirst=True)[-1]
        out.append(y)
    return np.asarray(out)



def simulated_trajectory(vehicle, us: np.ndarray) -> np.ndarray:
    """
    Step a current or legacy vehicle with ``step_speeds``.

    Returns
    -------
    np.ndarray
        The body states and motor speeds after each step, shape (steps, 12 + n).
    """
    out = []
    for u in us:
        vehicle.step_speeds(u)
        speeds = [p.motor.speed for p in vehicle.propellers]
        out.append(np.concatenate((np.asarray(vehicle.state, float), speeds)))
    return np.asarray(out)



def errors(trajectory: np.ndarray, reference: np.ndarray) -> dict:
    """Maximum error in position (mm), attitude (mrad) and motor speed (rad/s)."""
    return dict(
        position=1e3 * np.abs(trajectory[:, 0:3] - reference[:, 0:3]).max(),
        attitude=1e3 * np.abs(trajectory[:, 6:9] - reference[:, 6:9]).max(),
        motor_speed=np.abs(trajectory[:, N_BODY_STATES:] - reference[:, N_BODY_STATES:]).max(),
    )



def test_body_equations_match_oracle():
    # The body part of the combined right-hand side, for given motor speeds,
    # equals the oracle's derivative for propellers spinning at those speeds.
    vp, sp = make_params(8, 'quadratic', True, dtype=np.float64)
    vehicle = Multirotor(vp, sp)
    vp_no_motor, _ = make_params(8, 'quadratic', False, dtype=np.float64)
    oracle = LegacyMultirotor(vp_no_motor, sp)
    oracle.dxdt_decimals = 12
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.uniform(-0.5, 0.5, 12)
        speeds = nominal_speed(vp, sp) * rng.uniform(0.8, 1.2, 8)
        y = np.concatenate((x, speeds))
        args = (speeds, np.inf, np.zeros(6)) + vehicle._kernel_params
        dydt = vehicle._rhs(0., y, *args)
        np.testing.assert_allclose(
            dydt[:N_BODY_STATES], oracle.dxdt_speeds(0., x, speeds), rtol=0, atol=1e-12)



@pytest.mark.parametrize('max_voltage', (np.inf, 8.))
def test_motor_equations_match_oracle(max_voltage):
    # The oracle's motor stores the acceleration at the start of each step.
    params = deepcopy(MP)
    params.k_drag = PP.k_drag
    data = MotorData([params])
    sp = SimulationParams(dt=0.01)
    for speed in (0., 150., 300., 450.):
        for u in (0., 100., 300., 600., 2000.):
            oracle = LegacyMotor(params, sp)
            oracle.speed = speed
            oracle.step(u, max_voltage=max_voltage)
            acc = motor_acc(np.asarray([speed]), np.asarray([u]), max_voltage, data.params)
            np.testing.assert_allclose(acc[0], oracle._last_angular_acc, rtol=1e-12, atol=1e-9)



@pytest.mark.parametrize('dtype', (np.float32, np.float64), ids=lambda d: d.__name__)
@pytest.mark.parametrize('frac, bounds', [
    (0.05, dict(position=0.05, attitude=0.05, motor_speed=0.1)),
    (0.20, dict(position=0.1, attitude=0.25, motor_speed=0.5)),
], ids=('5%', '20%'))
def test_step_speeds_follows_reference(frac, bounds, dtype):
    vp, sp = make_params(8, 'quadratic', True, dtype=dtype)
    vehicle = Multirotor(vp, sp)
    settle_at_hover(vehicle, nominal_speed(vp, sp))
    us = schedule(hover_signals(vp, sp), frac, N_SCHEDULE)
    reference = reference_trajectory(vehicle, us)
    error = errors(simulated_trajectory(vehicle, us), reference)
    for key, bound in bounds.items():
        assert error[key] < bound, (key, error[key])



def test_step_speeds_is_more_accurate_than_oracle():
    vp, sp = make_params(8, 'quadratic', True, dtype=np.float64)
    new, old = make_pair(8, 'quadratic', True, dtype=np.float64)
    for v in (new, old):
        settle_at_hover(v, nominal_speed(vp, sp))
    us = schedule(hover_signals(vp, sp), 0.05, N_SCHEDULE)
    reference = reference_trajectory(new, us)
    new_error = errors(simulated_trajectory(new, us), reference)
    old_error = errors(simulated_trajectory(old, us), reference)
    for key in new_error:
        assert new_error[key] * 100 < old_error[key], (key, new_error[key], old_error[key])



@pytest.mark.parametrize('limit', ('max_current', 'max_voltage'))
def test_step_speeds_follows_reference_when_clipped(limit):
    # Clipped current or voltage puts corners in the right-hand side.
    vp, sp = make_params(8, 'quadratic', True, dtype=np.float64)
    hover = hover_signals(vp, sp)
    motor = vp.propellers[0].motor
    if limit == 'max_current':
        speed = nominal_speed(vp, sp)
        hover_current = (motor.k_drag * speed**2) / motor.k_torque
        motor.max_current = 1.05 * hover_current
    else:
        vp.battery = BatteryParams(max_voltage=1.02 * motor.speed_voltage_scaling * hover[0])
    vehicle = Multirotor(vp, sp)
    settle_at_hover(vehicle, nominal_speed(vp, sp))
    us = schedule(hover, 0.20, N_SCHEDULE)
    reference = reference_trajectory(vehicle, us)
    error = errors(simulated_trajectory(vehicle, us), reference)
    assert error['position'] < 0.1, error
    assert error['attitude'] < 0.25, error
    assert error['motor_speed'] < 0.5, error



def test_motor_step_response_does_not_overshoot():
    # The motor is close to first-order, so its speed rises to the target
    # without overshooting, even when dt is longer than its time constant.
    params = deepcopy(MP)
    params.k_drag = PP.k_drag
    sp = SimulationParams(dt=0.01)
    target = 315.7
    u = motor_signal(params, target)
    motor = Motor(params, sp)
    speeds = np.asarray([motor.step(u) for _ in range(50)])
    # Monotonic up to the integrator's tolerance (atol=1e-4 per step)
    assert np.all(np.diff(speeds) >= -1e-3), 'speed must rise monotonically'
    assert speeds.max() < target * 1.001
    assert speeds[-1] == pytest.approx(target, rel=1e-3)
    # The oracle's explicit update overshoots by about 30% at this time step.
    oracle = LegacyMotor(params, sp)
    oracle_speeds = np.asarray([oracle.step(u) for _ in range(50)])
    assert oracle_speeds.max() > target * 1.2
