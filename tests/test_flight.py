"""
Closed-loop flights with the default controller of `multirotor.optimize`.

The other tests compare against the frozen oracle. These check that the default
controller flies: it reaches a waypoint far from the start and stays there, with a
steady torque rather than one that flips sign every step. The oracle's simulation
rounded the state, which kept the vehicle at the origin and hid this.
"""
import numpy as np
import pytest

from multirotor.optimize import get_controller
from multirotor.simulation import Multirotor

from common import make_params

STEPS = 5000
WAYPOINTS = ((6., -4., 10., 0.), (-10., 8., 5., 0.))



def fly(n: int, waypoint, steps: int=STEPS):
    "Fly an ``n``-rotor vehicle to the waypoint; return the distances and pitch torques."
    vp, sp = make_params(n, 'quadratic', False, np.float64)
    vehicle = Multirotor(vp, sp)
    ctrl = get_controller(vehicle)
    ref = np.asarray(waypoint)
    distance, pitch_torque = [], []
    for _ in range(steps):
        action = ctrl.step(reference=ref)
        vehicle.step_speeds(vehicle.allocate_control(action[0], action[1:4]))
        distance.append(np.linalg.norm(vehicle.position - ref[:3]))
        pitch_torque.append(action[2])
    return np.asarray(distance), np.asarray(pitch_torque)



@pytest.mark.parametrize('n', (4, 8))
@pytest.mark.parametrize('waypoint', WAYPOINTS)
def test_default_controller_reaches_a_far_waypoint(n, waypoint):
    distance, _ = fly(n, waypoint)
    assert distance[-500:].max() < 0.1



def test_default_controller_torque_does_not_chatter():
    "The rate controller's torque used to alternate in sign at every step."
    _, torque = fly(8, WAYPOINTS[0], steps=500)
    flips = np.mean(np.diff(np.sign(np.diff(torque))) != 0)
    assert flips < 0.2
