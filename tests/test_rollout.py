"""
`multirotor.rollout`: the closed loop of the cascaded controller and a vehicle in one
compiled loop, against the loop of `Controller` and `Multirotor`; convergence of RK4 with
the number of substeps; and gradients of a rollout's loss with respect to controller and
vehicle parameters.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import multirotor.controller.pid as pid
import multirotor.physics as physics
import multirotor.rollout as rollout
from multirotor.controller import data as K
from multirotor.simulation import Multirotor
from multirotor.tuples import kind_of
from multirotor.vehicle import BatteryParams

from common import make_params
from test_controller import make_controller


DT = .01
TARGET = np.array([3., -2., 5., .4])
# RK4 substeps per step: those of the classes, which the rollout must then reproduce. The
# motors' dynamics are fast, and need more substeps than the body alone.
SUBSTEPS = rollout.DEFAULT_SUBSTEPS
POSITION_TOLERANCE = 1e-9



def make_vehicle(kind: str, substeps: int = None) -> Multirotor:
    battery = BatteryParams(
        max_voltage=25.2, capacity=20., internal_resistance=.08,
        ocv_soc=np.linspace(0, 1, 6), ocv_voltage=np.linspace(20., 25.2, 6)
    ) if kind == 'motors_battery' else None
    vp, sp = make_params(8, 'quadratic', kind != 'body', np.float64, DT, battery)
    return Multirotor(vp, sp, substeps=substeps)



def reference(kind: str, steps: int, substeps: int = None) -> np.ndarray:
    "State and controller output per step, from `Controller` and `Multirotor`."
    vehicle = make_vehicle(kind, substeps)
    controller = make_controller(pid, vehicle)
    out = []
    for _ in range(steps):
        action = controller.step(TARGET)
        vehicle.step_speeds(vehicle.allocate_control(action[0], action[1:]))
        out.append(np.concatenate((vehicle.state, action)))
    return np.array(out)



def arguments(kind: str):
    "The rollout's arguments from a fresh vehicle and controller."
    vehicle = make_vehicle(kind)
    vehicle_params = vehicle.rollout_params()
    assert kind_of(vehicle_params) == kind
    return vehicle.rollout_state(), vehicle_params, make_controller(pid, vehicle).rollout_params()



def run(kind, steps, substeps, ctrl=None, vehicle_params=None):
    "The trajectory of a rollout from rest."
    state, vehicle_params_, ctrl_ = arguments(kind)
    return getattr(rollout, 'closed_loop_' + kind)(
        state, vehicle_params or vehicle_params_, ctrl or ctrl_,
        jnp.tile(jnp.asarray(TARGET), (steps, 1)), DT, substeps)[1]



@pytest.mark.parametrize('kind', ('body', 'motors', 'motors_battery'))
def test_rollout_matches_the_loop(kind):
    steps = 60
    trajectory = np.asarray(run(kind, steps, SUBSTEPS[kind]))
    expected = reference(kind, steps)
    assert np.abs(trajectory - expected)[:, :3].max() < POSITION_TOLERANCE
    np.testing.assert_allclose(trajectory[:, 9:12], expected[:, 9:12], atol=1e-8)



def test_more_substeps_converge_for_motors():
    "RK4 error falls as the step is divided. The reference has many more substeps."
    steps = 60
    expected = np.asarray(run('motors', steps, 256))
    errors = [
        np.abs(np.asarray(run('motors', steps, substeps)) - expected)[:, :3].max()
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
    power-limited bisection (`multirotor.physics.bisect`) inside the scan.
    """
    steps, substeps = 60, 8
    state, vehicle_params, ctrl = arguments(kind)
    state = jax.tree_util.tree_map(jnp.asarray, state)
    targets = jnp.tile(jnp.asarray(TARGET), (steps, 1))

    def loss(theta):
        ctrl_ = ctrl._replace(params=jnp.asarray(ctrl.params).at[K.PID_K_P].multiply(theta[0]))
        vehicle_ = vehicle_params._replace(
            mass=vehicle_params.mass * theta[1],
            motor_params=jnp.asarray(vehicle_params.motor_params).at[
                physics.MOTOR_RESISTANCE].multiply(theta[2]))
        if kind == 'motors_battery':
            vehicle_ = vehicle_._replace(battery_params=jnp.asarray(
                vehicle_params.battery_params).at[
                    physics.BATTERY_INTERNAL_RESISTANCE].multiply(theta[3]))
        trajectory = getattr(rollout, 'closed_loop_' + kind)(
            state, vehicle_, ctrl_, targets, DT, substeps)[1]
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
