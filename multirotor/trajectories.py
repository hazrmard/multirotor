import numpy as np
from multirotor.simulation import Multirotor

from multirotor.vehicle import VehicleParams, SimulationParams, PropellerParams
from multirotor.helpers import control_allocation_matrix



def get_vehicle_ability(
    vp: VehicleParams, sp: SimulationParams, max_tilt: float=np.pi/12
):
    alloc, alloc_inverse = control_allocation_matrix(vp)
    I_roll, I_pitch, I_yaw = vp.inertia_matrix.diagonal()
    # T = I . d_omega
    # s = 0.5 d_omega t^2
    # time to reach max tilt (s):
    # t_m = sqrt(2 s / d_omega) = t = sqrt(2 s I / T)
    # max_torque = from propellers x geometry
    # t_m = np.sqrt(2 * max_tilt * vp.inertia_matrix.diagonal() / max_torque)
    # max lateral thrust
    # T_l = Thrust sin(max_tilt)
    # max lateral acc
    # a = T_l / m
    # avg velocity = dist / time
    # x / (t_m + sqrt(2 (x/2) m / T_l) + 2 t_m + sqrt(2 (x/2) m / T_l))
    #      tilt   acc half way          tilt reverse   dec half way to stop



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
        self, points: np.ndarray, vehicle: Multirotor=None, proximity: float=None,
        resolution: float=None    
    ):
        """
        Parameters
        ----------
        points : np.ndarray
            A list of 3D coordinates.
        vehicle : Multirotor, optional
            The vehicle to track. Required if proximity is provided, by default None
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
        if self.proximity is not None and self.vehicle is not None:
            for i in range(len(self)):
                while np.linalg.norm((self.vehicle.position - self[i])) >= self.proximity:
                        yield self[i]
                for _ in range(self._durations[i] - 1):
                        yield self[i]
        elif self.proximity is None:
            for i in range(len(self)):
                for _ in range(self._durations[i]):
                        yield self[i]
        else:
            raise ValueError('Vehicle must be provided if a proximity value is given.')


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