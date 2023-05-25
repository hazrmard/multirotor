from typing import Callable, Iterable, Tuple
from types import SimpleNamespace

import numpy as np
from scipy.optimize import fsolve

from .vehicle import PropellerParams, VehicleParams, SimulationParams
from .physics import torque



def moment_of_inertia_tensor_from_cooords(
    point_masses: Iterable[float], coords: Iterable[np.ndarray]
) -> np.ndarray:
    """
    Calculate the inertial matrix given a distribution of point masses.

    Parameters
    ----------
    point_masses : Iterable[float]
        A list of masses.
    coords : Iterable[np.ndarray]
        The corresponding coordinates of those masses about the center of rotation.
        Ideally, this would be the center of mass of the object.

    Returns
    -------
    np.ndarray
        The 3x3 inertial matrix.
    """
    # TODO: Conditionally calculate the center of mass and transform coordinates
    # about it if a boolean option is provided.
    coords = np.asarray(coords)
    masses = np.asarray(point_masses)
    x,y,z = coords[:,0], coords[:,1], coords[:,2]
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = Iyx = -np.sum(x * y * masses)
    Iyz = Izy = -np.sum(y * z * masses)
    Ixz = Izx = -np.sum(x * z * masses)
    return np.asarray([
        [Ixx, Ixy, Ixz],
        [Iyx, Iyy, Iyz],
        [Izx, Izy, Izz]
    ])



def vehicle_params_factory(
    n: int, m_prop: float, d_prop: float, params: PropellerParams,
    m_body: float, body_shape: str='sphere_solid', body_size: float=0.1
) -> VehicleParams:
    """
    Create a simple multirotor vehicle parameters object. The multirotor has
    evenly spaced propellers and a simple core shape (shell, cube etc.)

    Parameters
    ----------
    n : int
        The number of arms/propellers.
    m_prop : float
        The mass of each propeller.
    d_prop : float
        The distance of each propeller from the center of the multirotor.
    params : PropellerParams
        The parameters describing a propeller.
    m_body : float
        The mass of the central body.
    body_shape : str, optional
        The shape of the core, by default 'sphere_solid'
    body_size : float, optional
        The dimension of the core (m), by default 0.1

    Returns
    -------
    VehicleParams
        The parameters object.
    """
    angle_spacing = 2 * np.pi / n
    angles = np.arange(angle_spacing / 2, 2 * np.pi, angle_spacing)
    masses = [m_prop] * n
    x = np.cos(angles) * d_prop
    y = np.sin(angles) * d_prop
    z = np.zeros_like(y)
    coords = np.vstack((x, y, z)).T
    I_prop = moment_of_inertia_tensor_from_cooords(masses, coords)

    if body_shape == 'point':
        I_body = np.asarray(np.zeros((3,3)))
    elif body_shape == 'sphere_solid':
        I_body = np.asarray(np.eye(3) * 0.4 * m_body * body_size**2)
    elif body_shape == 'sphere_shell':
        I_body = np.asarray(np.eye(3) * 2 * m_body * body_size**2 / 3)
    elif body_shape == 'cube':
        I_body = np.asarray(np.eye(3) * 2 * m_body * body_size**2 / 12)

    I = I_body + I_prop

    return VehicleParams(
        propellers = [params for _ in range(n)],
        angles = angles,
        distances=np.ones(n) * d_prop,
        mass = n * m_prop + m_body,
        inertia_matrix = I,
    )



def find_nominal_speed(thrust_fn: Callable[[float], float], weight: float) -> float:
    """
    Calculate the speed a propeller must spin to balance the weight.

    Parameters
    ----------
    thrust_fn : Callable[[float], float]
        A function taking the speed as input and outputting thrust (N).
    weight : float
        The weight to balance.

    Returns
    -------
    float
        The speed to balance the weight.
    """
    def balance(speed: float) -> float:
        thrust = thrust_fn(speed)
        residual = thrust - weight
        return residual
    return fsolve(balance, 1e3)[0]



def learn_thrust_coefficient(
    thrust_fn: Callable[[float], float], domain: Tuple=(1, 10000)
) -> float:
    """
    Assuming a quadratic relationship between thrust and propeller speed,
    estimate the coefficient of proportionality k_thrust, where

        thrust = k_thrust . speed^2

    Parameters
    ----------
    thrust_fn : Callable[[float], float]
        The function accepting speed and returning thrust.
    domain : Tuple, optional
        The range of speeds to try, by default (1, 10000)

    Returns
    -------
    float
        The thrust coefficient.
    """
    speeds = np.linspace(domain[0], domain[1], num=250)
    thrust = np.zeros_like(speeds)
    for i, speed in enumerate(speeds):
        thrust[i] = thrust_fn(speed)
    return np.polyfit(speeds, thrust, deg=2)[0] # return coefficient of quadradic term



def learn_speed_voltage_scaling(
    speed_fn: Callable[[float], float], domain: Tuple=(0,20)
) -> float:
    """
    Assuming a linear relationship between voltage and motor speed, learn
    the scaling coefficient, k_scaling, where:

        voltage = k_scaling * speed

    This can be put in `MotorParams.speed_voltage_scaling` so the speeds
    generated by the control allocation matrix are converted to corresponding
    voltages.

    Parameters
    ----------
    speed_fn : Callable[[float], float]
        A function accepting voltage and returning speed. This should be the
        nominal case for a `Motor` class which has `speed_voltage_scaling==1`.
    domain : Tuple, optional
        The range of voltages to try to learn the coefficient, by default (0,20)

    Returns
    -------
    float
        The scaling coefficient.
    """
    signals = np.linspace(domain[0], domain[1], num=10)
    speeds = np.zeros_like(signals)
    for i, signal in enumerate(signals):
        speeds[i] = speed_fn(signal)
    return np.polyfit(speeds, signals, 1)[0]



def moment_of_inertia_disk(m: float, r: float) -> float:
    return 0.5 * m * r**2



def control_allocation_matrix(params: VehicleParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the control allocation matrix such that:

        action = matrix @ [thrust, torque_x, torque_y, torque_z]

    Parameters
    ----------
    params : VehicleParams
        The vehicle parameters for which to compute the matrix

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The allocation matrix and its inverse. If no inverse exists, returns
        the Moore-Penrose Pseudo-inverse.
    """
    alloc = np.zeros((4, len(params.propellers))) #[Fz, Mx, My, Mz] x n-Propellers
    x = params.distances * np.cos(params.angles)
    y = params.distances * np.sin(params.angles)
    for i, p in enumerate(params.propellers):
        alloc[0, i] = p.k_thrust                       # vertical force
        alloc[1, i] = p.k_thrust * y[i]                 # torque about x-axis
        alloc[2, i] = p.k_thrust * (-x[i])              # torque about y-axis
        alloc[3, i] = p.k_drag * params.clockwise[i]    # torque about z-axis
    alloc_inverse = np.linalg.pinv(alloc)
    return alloc, alloc_inverse



def get_vehicle_ability(
    vp: VehicleParams, sp: SimulationParams,
    max_tilt: float=np.pi/12,
    max_rads: float=600
):
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
    # excessive thrust (above weight) that can contribute to torque
    thrust_vec[2] = np.asarray(thrusts) - (vp.mass * sp.g / n)
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
    torques = (torques * (torques > 0)).sum(axis=1)
    I = vp.inertia_matrix.diagonal()
    # t = i a
    ang_acc = torques / I

    # TODO: max angular velocity such that can accelerate to and decelerate from it
    # to 0

    res = dict(
        max_acc_xy=max_acc_xy,
        max_acc_z=max_acc_z,
        max_ang_acc=max(ang_acc)
    )
    return res



class DataLog:
    """
    Records state and action variables for a multirotor and controller for each
    simulation step.
    """
    def __init__(
        self, vehicle: 'Multirotor'=None, controller: 'Controller'=None,
        other_vars=None
    ):
        """
        Parameters
        ----------
        vehicle : Multirotor, optional
            The Multirotor to track, by default None
        controller : Controller, optional
            The controller instance to track, by default None
        other_vars : Iterable[str]
            A list of names representing other variables to log. The names
            become attributes of this class and can be called as DataLog.NAME.
            By default None.
        """
        return self.track(vehicle, controller, other_vars)


    def track(self, vehicle, controller, other_vars=None):
        """
        Register Multirotor and Controller instances to track, along with names
        of any other variables to be manually added.

        >>> DataLog.track(Multirotor(), Controller(), other_vars=('error',))

        Parameters
        ----------
        vehicle : Multirotor
            The vehicle to track.
        controller : Controller
            The controller to track.
        """
        self._arrayed = False
        self._states = []
        self.states = None
        self._actions = []
        self.actions = None
        self.times = None
        self._times = []
        self._args = () if other_vars is None else other_vars
        own_vars = ('arrayed', 'states', 'actions', 'times', 'args', 'target')
        not_allowed = [v for v in own_vars if v in self._args]
        if len(not_allowed) > 0:
            raise AttributeError(
                ('The following `other_vars` are not allowed since they are attributes '
                  ', '.join(not_allowed))
            )
        for arg in self._args:
            setattr(self, arg, None)
            setattr(self, '_' + str(arg), [])
        self._target = dict(
            position=[], velocity=[], orientation=[], rate=[]
        )
        self.vehicle = vehicle
        self.controller = controller


    def __getitem__(self, item):
        if isinstance(item, slice):
            d = DataLog(other_vars=self._args)
            pass
        # TODO: Finish. Return a DataLog with the requested slice

        
    def log(self, **kwargs):
        """
        Add the state and action variables from the Multirotor and Controller.
        Any keyword arguments should already have been registered in `track()`
        and their values are now appended to the list.

        >>> DataLog.log(error=5)
        """
        self._arrayed = False
        if self.vehicle is not None:
            self._states.append(self.vehicle.state)
            self._times.append(self.vehicle.t)
        if self.controller is not None:
            self._actions.append(self.controller.action)
            self._target['position'].append(
                self.controller.reference[:3]
            )
            self._target['velocity'].append(
                np.concatenate((self.controller.ctrl_v.reference, self.controller.ctrl_z.action))
            )
            self._target['orientation'].append(
                np.concatenate((self.controller.ctrl_a.reference, self.controller.reference[3:4]))
            )
            self._target['rate'].append(
                self.controller.ctrl_r.reference
            )
        for key, value in kwargs.items():
            getattr(self, '_' + key).append(value)

            
    def done_logging(self):
        """
        Indicate that no more logs are going to be put so the python lists are converted
        to numpy arrays and discarded.
        """
        self._make_arrays()
        self._states = []
        self._actions = []
        self._times = []
        for arg in self._args:
            setattr(self, '_' + arg, [])
        self._target = dict(
            position=[], velocity=[], orientation=[], rate=[]
        )


    def append(self, log: 'DataLog', relative=True):
        assert set(self._args) == set(log._args), 'Inconsistent logged variables'
        old_len = len(self)
        self._states.extend(log._states)
        self._actions.extend(log._actions)
        self._times.extend(log._times)
        if self._arrayed and log._arrayed:
            self.states = np.concatenate((self.states, log.states), dtype=self.vehicle.dtype)
            self.actions = np.concatenate((self.actions, log.actions), dtype=self.vehicle.dtype)
            self.times = np.concatenate((self.times, log.times), dtype=self.vehicle.dtype)

        for arg in self._args:
            lst = getattr(self, '_' + arg, [])
            otherlst = getattr(log, '_' + arg, [])
            lst.extend(otherlst)
            setattr(self, '_' + arg, lst)
            if self._arrayed and log._arrayed:
                arr = getattr(self, arg)
                otherarr = getattr(log, arg)
                arr = np.concatenate((arr, otherarr), dtype=self.vehicle.dtype)
                setattr(self, arg, arr)

        self._target = {name: lst + log._target[name] for name, lst in self._target.items()}
        if self._arrayed and log._arrayed:
            d = {}
            for k in self._target.keys():
                a1 = getattr(self.target, k)
                a2 = getattr(log.target, k)
                a = np.concatenate((a1, a2), dtype=self.vehicle.dtype)
                d[k] = a
            self.target = SimpleNamespace(**d)

        if relative:
            # last_pos = self.position[old_len - 1]
            if not self._arrayed:
                last_time = self._times[old_len - 1]
                for i in range(old_len, len(self)):
                        # self._states[i][:3] += last_pos
                        self._times[i] += last_time
            elif self._arrayed:
                last_time = self.times[old_len - 1]
                # self.states[old_len:,:3] += last_pos
                self.times[old_len:] += last_time

            
    def _make_arrays(self, relative_to=None):
        """
        Convert python list to array and put up a flag that all arrays are up
        to date.
        """
        if not self._arrayed:
            self.states = np.asarray(self._states, self.vehicle.dtype)
            self.actions = np.asarray(self._actions, self.vehicle.dtype)
            self.times = np.asarray(self._times, self.vehicle.dtype)
            for arg in self._args:
                setattr(self, arg, np.asarray(getattr(self, '_' + arg), self.vehicle.dtype))
            self.target = SimpleNamespace(**{k: np.asarray(v, self.vehicle.dtype) for k, v in self._target.items()})
        self._arrayed = True


    def __len__(self):
        if len(self._states)==0:
            if self.states is not None:
                return len(self.states)
        return len(self._states)

        
    @property
    def t(self):
        self._make_arrays()
        return self.times
    @property
    def position(self):
        self._make_arrays()
        return self.states[:, 0:3]
    @property
    def x(self):
        return self.position[:, 0].reshape(-1)
    @property
    def y(self):
        return self.position[:, 1].reshape(-1)
    @property
    def z(self):
        return self.position[:, 2].reshape(-1)
    @property
    def velocity(self):
        self._make_arrays()
        return self.states[:, 3:6]
    @property
    def orientation(self):
        self._make_arrays()
        return self.states[:, 6:9]
    @property
    def roll(self):
        return self.orientation[:, 0].reshape(-1)
    @property
    def pitch(self):
        return self.orientation[:, 1].reshape(-1)
    @property
    def yaw(self):
        return self.orientation[:, 2].reshape(-1)
    @property
    def angular_rate(self):
        self._make_arrays()
        return self.states[:, 9:12]
    # TODO: add properties for controller state
