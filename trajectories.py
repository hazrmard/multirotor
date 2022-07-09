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
        # TODO: loiter in trajectory
        self.proximity = proximity
        self.vehicle = vehicle
        self.points = np.asarray(points)
        if resolution is not None:
            points = []
            for p1, p2 in zip(self.points[:-1], self.points[1:]):
                dist = np.linalg.norm(p2 - p1)
                num = int(dist / resolution) + 1
                points.extend(np.linspace(p1, p2, num=num, endpoint=True))
            self._points = points
        else:
            self._points = points


    def __len__(self):
        return len(self._points)


    def __getitem__(self, i: int):
        return self._points[i]


    def __iter__(self):
        if self.proximity is not None and self.vehicle is not None:
            for i in range(len(self)):
                while np.linalg.norm((self.vehicle.position - self[i])) >= self.proximity:
                    yield self[i]
        elif self.proximity is None:
            for i in range(len(self)):
                yield self[i]
        else:
            raise ValueError('Vehicle must be provided if a proximity value is given.')
