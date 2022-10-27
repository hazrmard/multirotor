from collections import namedtuple
from typing import Iterable

import numpy as np
from pyscurve import ScurvePlanner
from pyscurve.trajectory import PlanningError

from multirotor.simulation import Multirotor
from multirotor.vehicle import VehicleParams, SimulationParams, PropellerParams
from multirotor.helpers import control_allocation_matrix
from multirotor.physics import torque


def get_vehicle_ability(
    vp: VehicleParams, sp: SimulationParams,
    max_tilt: float=np.pi/12,
    max_angular_acc: float=5,
    max_rads: float=600
):
    alloc, alloc_inverse = control_allocation_matrix(vp)
    I_roll, I_pitch, I_yaw = vp.inertia_matrix.diagonal()
    n = len(vp.propellers)

    thrusts = [p.k_thrust * max_rads**2 for p in vp.propellers]
    max_f = sum(thrusts)
    max_acc_z = (max_f - (vp.mass * sp.g)) / vp.mass
    
    # Thrust produced at max_tilt to keep vehicle altitude constant.
    # Use that to calculate lateral component of thrust and lateral acceleration
    tilt_hover_thrust = (vp.mass * sp.g) / (n * np.cos(max_tilt))
    # Thrust is limited by what is possible by the propellers
    tilt_hover_thrust = min(max_f, tilt_hover_thrust)
    max_acc_xy = n * tilt_hover_thrust * np.sin(max_tilt) / vp.mass

    thrust_vec = np.zeros((3, n))
    thrust_vec[2] = np.asarray(thrusts)
    k_drag_vec = np.asarray([p.k_drag for p in vp.propellers])
    inertia_vec = np.asarray([p.moment_of_inertia for p in vp.propellers])
    torques = torque(
        position_vector=vp.propeller_vectors,
        force=thrust_vec,
        clockwise=np.asarray(vp.clockwise).astype(float),
        drag_coefficient=k_drag_vec,
        moment_of_inertia=inertia_vec,
        prop_angular_acceleration=0,
        prop_angular_velocity=max_rads
    )
    I = vp.inertia_matrix.diagonal()
    ang_acc = None

    res = dict(
        max_acc_xy=max_acc_xy,
        max_acc_z=max_acc_z,
    )
    return res



class Trajectory:
    """
    Iterate over waypoints for a multirotor. The trajectory class can segment a
    list of waypoints into smaller sections and feed them to the controller when
    the vehicle is within a radius of its current waypoint.

    For example:

        m = Multirotor(...)
        traj = Trajectory(
            points=[(0,0,0), (0,0,2), (10,0,2)],
            vehicle=m, proximity=0.1, resolution=0.5)
        for point in traj:
            Each point is spaced `resolution` units apart in euclidean distance.
            When m.position is within `proximity` of current point, the next
            point in the trajectory is yielded.
    """


    def __init__(
        self, vehicle: Multirotor, points: np.ndarray, proximity: float=None,
        resolution: float=None    
    ):
        """
        Parameters
        ----------
        vehicle : Multirotor
            The vehicle to track.
        points : np.ndarray
            A list of 3D coordinates.
        proximity : float, optional
            The distance from current waypoint at which to send the next waypoint,
            by default None. If None, simply iterates over provided points.
        resolution : float, optional
            The segmentation of the trajectory, by default None. If provided, it
            is the distance between intermediate points generated from the waypoints
            provided. For e.g. resolution=2 and points=(0,0,0), (10,0,0) will
            create intermediate points a distance 2 apart.
        """
        self.points = points
        self.vehicle = vehicle
        self.proximity = proximity
        self.resolution = resolution
        

    def __len__(self):
        return len(self._points)


    def __getitem__(self, i: int):
        return self._points[i]


    def __iter__(self):
        self._points, self._durations = self.generate_trajectory(self.vehicle.position)
        if self.proximity is not None:
            for i in range(len(self)):
                while np.linalg.norm((self.vehicle.position - self[i])) >= self.proximity:
                        yield self[i], None
                for _ in range(self._durations[i] - 1):
                        yield self[i], None
        else:
            for i in range(len(self)):
                for _ in range(self._durations[i]):
                        yield self[i], None


    def generate_trajectory(self, curr_pos=None):
        if curr_pos is not None:
            points = [curr_pos, *self.points]
        durations = [1 if len(p)==3 else p[-1] for p in points]
        points = np.asarray([p[:3] for p in points])
        if self.resolution is not None:
            _points = []
            _durations = []
            for i, (p1, p2) in enumerate(zip(points[:-1], points[1:])):
                dist = np.linalg.norm(p2 - p1)
                num = int(dist / self.resolution) + 1
                _points.extend(np.linspace(p1, p2, num=num, endpoint=True))
                dur = np.ones(num, dtype=int)
                dur[0] = durations[i]
                _durations.extend(dur)
            _durations[-1] = durations[-1]
        else:
            _points = points
            _durations = durations
        return _points, _durations


    def add_waypoint(self, point: np.ndarray):
        pass


    def next_waypoint(self):
        pass



class GuidedTrajectory:
    # Guided mode call stack overview
    # https://ardupilot.org/dev/docs/apmcopter-code-overview.html
    # Square-root controller for pos control, sets P value
    # https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Math/control.cpp#L286

    def __init__(self, vehicle: Multirotor, waypoints: Iterable[np.ndarray],
        interval: int=10, proximity: float=2., max_velocity: float=5., max_acceleration: float=3.,
        max_jerk: float=100, turn_factor: float=(1/np.sqrt(2))) -> None:
        self.vehicle = vehicle
        self.waypoints = np.asarray(waypoints)
        self.interval = int(interval)
        self.proximity = float(proximity)
        self.max_velocity = float(max_velocity)
        self.max_acceleration = float(max_acceleration)
        self.max_jerk = float(max_jerk)
        self.turn_factor = turn_factor
        self.trajectory = None
        self.trajs = []


    def _setup(self):
        if self.interval is None:
            dt = self.vehicle.simulation.dt
            # default run at 100Hz
            self.interval = int(max(1, 0.01 / dt))


    def __iter__(self):
        # https://www.youtube.com/watch?v=MQiNDJGJDWs
        self._setup()
        i = 0 # steps since beginning of trajectory
        j = 0 # steps since last trajectory replan
        k = 0 # waypoint index
        planner = ScurvePlanner(debug=False)
        self._ref = None
        for k, wp in enumerate(self.waypoints):
            print('WP @ i=%d' % i, wp)
            replan = True # new waypoint, so replan immediately
            while not self.reached(wp):
                # Refresh trajectory plan
                if i % self.interval == 0 or replan:
                    r0 = self.vehicle.position[:2]
                    v0 = self.vehicle.velocity[:2]
                    r1 = wp[:2]
                    # Assume that destination velocity is 0
                    v1 = np.zeros(2)
                    # Unless there is another leg of flight, in which case
                    # target velocity should depend on the sharpness of turn
                    if k < len(self.waypoints)-1:
                        r2 = self.waypoints[k+1][:2]
                        # TODO: should this be the shortest path to the trajectory
                        # instead of the next waypoint? I.e prioritize returning
                        # to planned path, instead of taking the shortest diagonal path
                        # to the destination?
                        r01 = r1 - r0 # current desired heading vector
                        # r01 = v0
                        r12 = r2 - r1 # next desired heading vector
                        r01_ = r01 / np.linalg.norm(r01)
                        r12_ = r12 / np.linalg.norm(r12)
                        # Projection of vectors along current leg. If projection is <0,
                        # means vehicle needs to reverse. So clip to 0 so it comes
                        # to a stop at the waypoint.
                        projection = max(0, np.dot(r12_, r01_))
                        v1 = self.max_velocity * self.turn_factor * projection * r01_
                    try:
                        self.trajectory = planner.plan_trajectory(
                            q0=r0, q1=r1, v0=v0, v1=v1,
                            v_max=max(self.max_velocity, np.linalg.norm(v0)),
                            a_max=self.max_acceleration,
                            j_max=self.max_jerk
                        )
                        self.trajs.append(self.trajectory)
                        replan = False # once replanned, wait for next waypoint or interval
                        j = 0 # reset steps since replan
                    except PlanningError as e:
                        # If planning error was on a new point/or replanning was
                        # requested. When not choosing to raise exception, trust
                        # that the vehicle can catch up and later on the trajectory
                        # can become feasible again.
                        if replan:
                            # print('r0', r0, 'r1', r1)
                            # print('Err: %s' % e)
                            # raise e
                            pass
                        replan = True # after error, replan at next interval
                # Simulate trajectory plan over time
                for _ in range(self.interval):
                    target = self.trajectory(self.vehicle.simulation.dt * j)
                    point = target[:, 2]
                    velocity = target[:, 1]
                    self._ref = np.concatenate((point, wp[2:], [0])) # position, yaw
                    i += 1 # steps since beginning of trajectory
                    j += 1 # steps since last trajectory replan
                    yield self._ref, velocity


    def reached(self, wp: np.ndarray) -> bool:
        return np.linalg.norm(self.vehicle.position - wp) <= self.proximity