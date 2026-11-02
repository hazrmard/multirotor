"""
The cascaded PID controller. `Controller` tracks a position and yaw and outputs
thrust and moments, through six stages that each wrap the same PID law::

    (x,y) --> PosController --> VelController --> AttController --> RateController --> (moments)
    (z)   --> AltController --> AltRateController --> (thrust)

The stages are views of one `ControllerData`, which holds their parameters and
state as matrices, one column per control channel. Each stage's ``step`` passes
the matrices and the vehicle's state to a compiled kernel in
`multirotor.controller.kernels`, which does the arithmetic.

Call chain of `Controller.step`, between the vehicle's steps::

    Controller.step
        AltController.step --------> kernels.altitude_step
        AltRateController.step ----> kernels.climb_step
        PosController.step --------> kernels.position_step
        VelController.step --------> kernels.velocity_step
        AttController.step --------> kernels.attitude_step
        RateController.step -------> kernels.rate_step

Altitude and position stages run every ``period_z`` and ``period_p`` seconds. The
others run every ``period_a`` seconds.
"""
from __future__ import annotations
from typing import Union, Dict

import numpy as np

from ..simulation import Multirotor
from .data import ControllerData
from .kernels import (
    PID_K_P, PID_K_I, PID_K_D, PID_MAX_ERR_I, PID_LIMIT, PID_MAX_ACCELERATION,
    PID_ERR, PID_ERR_P, PID_ERR_I, PID_ERR_D, PID_REFERENCE, PID_ACTION, PID_INTEGRAL,
    pid_stage, leash_distance, position_step, velocity_step, attitude_step, rate_step,
    altitude_step, climb_step,
    sqrt_control # noqa: F401, kept importable from here
)

# See more:
# https://archive.ph/3Aco3
# https://ardupilot.org/dev/docs/apmcopter-code-overview.html
# https://ardupilot.org/copter/docs/traditional-helicopter-control-system.html
# https://ardupilot.org/dev/docs/apmcopter-programming-attitude-control-2.html
# https://www.youtube.com/watch?v=-PC69jcMizA


def _vector(x) -> np.ndarray:
    "A number or array as a float64 vector, without a copy if it already is one."
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _state64(vehicle: Multirotor) -> np.ndarray:
    "The state of a vehicle as float64, without a copy if it already is."
    return np.asarray(vehicle.state, dtype=np.float64)


def _param(row: int, doc: str) -> property:
    "A property that views one row of the parameter matrix, over a stage's columns."
    def get(self):
        return self._data.params[row, self._start:self._stop]
    def set(self, value):
        self._data.params[row, self._start:self._stop] = value
    return property(get, set, doc=doc)


def _state(row: int, doc: str) -> property:
    "A property that views one row of the state matrix, over a stage's columns."
    def get(self):
        return self._data.state[row, self._start:self._stop]
    def set(self, value):
        self._data.state[row, self._start:self._stop] = value
    return property(get, set, doc=doc)


def _scalar_param(row: int, doc: str) -> property:
    "A property for a parameter of a stage that has one value for all its channels."
    def get(self):
        return float(self._data.params[row, self._start])
    def set(self, value):
        self._data.params[row, self._start:self._stop] = value
    return property(get, set, doc=doc)


class PIDController:
    """
    Proportional Integral Derivative controller. Tracks a reference signal and outputs
    a control signal to minimize output. The equation is:

        err = reference - measurement

        u = k_p * err + k_d * d(err)/dt + k_i * int(err . dt)
    
    Can control a single or an array of signals, given float or array PID constants.

    The constants and the state are views of columns of a `ControllerData`. A
    controller owns one until a `Controller` takes it over.
    """

    k_p = _param(PID_K_P, "Proportional constant")
    k_i = _param(PID_K_I, "Integral constant")
    k_d = _param(PID_K_D, "Derivative constant")
    max_err_i = _param(PID_MAX_ERR_I, "Maximum accumulated error")
    err = _state(PID_ERR, "Error at the last step")
    err_p = _state(PID_ERR_P, "Proportional term at the last step")
    err_i = _state(PID_ERR_I, "Integral term at the last step")
    err_d = _state(PID_ERR_D, "Derivative term at the last step")
    reference = _state(PID_REFERENCE, "Reference at the last step")
    action = _state(PID_ACTION, "Action at the last step")
    integral = _state(PID_INTEGRAL, "Accumulated error, before it is scaled by k_i")

    def __init__(
        self, k_p: np.ndarray, k_i: np.ndarray, k_d: np.ndarray,
        max_err_i: np.ndarray, n: int=None
    ):
        """
        Parameters
        ----------
        k_p, k_i, k_d : np.ndarray
            Proportional, integral and derivative constants. A number applies to
            all channels.
        max_err_i : np.ndarray
            Maximum accumulated error. None for no limit.
        n : int, optional
            The number of channels, by default the size of the largest constant.
        """
        if n is None:
            n = max(np.size(k_p), np.size(k_i), np.size(k_d), 1)
        self.dtype = np.dtype(np.float64)
        self._bind(ControllerData(n), 0, n)
        self.k_p, self.k_i, self.k_d = k_p, k_i, k_d
        self.max_err_i = np.inf if max_err_i is None else max_err_i
        self._params = ('k_p', 'k_i', 'k_d', 'max_err_i')


    def _bind(self, data: ControllerData, start: int, n: int) -> None:
        "Make this controller a view of ``n`` columns of ``data``, from ``start``."
        self._data: ControllerData = data
        self._start: int = start
        self._stop: int = start + n
        self.n: int = n


    def _attach(self, data: ControllerData, start: int) -> None:
        "Move this controller, with its values, to columns of ``data`` from ``start``."
        stop = start + self.n
        data.params[:, start:stop] = self._data.params[:, self._start:self._stop]
        data.state[:, start:stop] = self._data.state[:, self._start:self._stop]
        self._bind(data, start, self.n)


    def reset(self):
        self._data.state[:, self._start:self._stop] = 0.


    def set_params(self, **params: Dict[str, Union[np.ndarray, bool, float, int]]):
        for name, param in params.items():
            if hasattr(self, name):
                attr = getattr(self, name)
                if isinstance(attr, np.ndarray):
                    param = np.asarray(param, dtype=attr.dtype)
                    # cast to an axis & assign in-place
                    # for cases where float param is assigned to array attr
                    if attr.ndim > 0:
                        attr[:] = param
                    else:
                        attr = param
                else:
                    attr = param
                setattr(self, name, attr)
            else:
                raise AttributeError('Attribute %s not part of class.' % name)


    def get_params(self) -> Dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self._params}


    @property
    def state(self) -> np.ndarray:
        return np.concatenate((self.err_p, self.err_i, self.err_d))


    def step(
        self, reference: np.ndarray, measurement: np.ndarray, dt: float=1.,
        ref_is_error: bool=False, persist: bool=True
    ) -> np.ndarray:
        """
        Calculate the output, based on the current measurement and the reference
        signal.

        Parameters
        ----------
        reference : np.ndarray
            The reference signal(s) to track. Can be a number or an array.
        measurement : np.ndarray
            The actual measurement(s).
        ref_is_error: bool
            Whether to interpret the reference input as the error.
        persist: bool
            Whether to store the current state for the next step.

        Returns
        -------
        np.ndarray
            The action signal.
        """
        shape = (self.n,)
        ref = np.array(np.broadcast_to(_vector(reference), shape))
        mea = np.zeros(shape) if ref_is_error \
            else np.array(np.broadcast_to(_vector(measurement), shape))
        out = np.empty(shape)
        pid_stage(self._data.params, self._data.state, self._start, self._stop,
                  ref, mea, float(dt), ref_is_error, persist, out)
        return out



class PosController(PIDController):
    """
    Position controller. Convert xy-waypoint into required xy-velocity in the body-frame.
    """

    max_velocity = _scalar_param(PID_LIMIT, "Maximum velocity in m/s")
    max_acceleration = _scalar_param(PID_MAX_ACCELERATION, "Maximum acceleration in m/s/s")

    def __init__(
        self, k_p, k_i, k_d, max_err_i, vehicle: Multirotor,
        max_velocity: float=7.0, max_acceleration: float=3., max_jerk: float=100.0,
        square_root_scaling: bool=True, leashing: bool=True
    ):
        """
        Parameters
        ----------
        vehicle : Multirotor
            The vehicle to control.
        max_velocity : float
            Maximum velocity in m/s
        max_acceleration : float
            Maximum acceleration in m/s/s
        max_jerk : float
            Maximum jerk in m/s/s/s
        square_root_scaling : bool
            Whether to scale P-gain with the square root of the error
        leashing : bool
            Whether to limit proportional position error
        """
        self.vehicle: Multirotor = vehicle
        self.max_jerk: float = max_jerk
        self.square_root_scaling: bool = square_root_scaling
        self.leashing: bool = leashing
        super().__init__(k_p, k_i, k_d, max_err_i, n=2)
        self.dtype = self.vehicle.dtype
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        if self.leashing or self.square_root_scaling:
            self.k_p = 0.5 * self.max_jerk / self.max_acceleration
        self._params = tuple(list(self._params) + \
            ['max_velocity', 'max_acceleration', 'max_jerk', 'square_root_scaling', 'leashing'])


    @property
    def leash(self) -> float:
        "The largest horizontal position error that is acted on, given the vehicle's motion."
        if not self.leashing:
            return np.inf
        return leash_distance(
            float(self.k_p[0]), _state64(self.vehicle),
            _vector(self.vehicle.acceleration), self.vehicle.simulation.dt)


    def step(self, reference, measurement, dt, persist: bool=True):
        out = np.empty(2)
        position_step(
            self._data.params, self._data.state, self._start,
            _vector(reference), _vector(measurement), _state64(self.vehicle),
            _vector(self.vehicle.acceleration), self.vehicle.simulation.dt,
            float(dt), self.leashing, self.square_root_scaling, persist, out)
        return out



class VelController(PIDController):
    """
    Velocity controller. Takes reference x/y body-frame velocity and outputs reference 
    pitch and roll angles for x and y motion, respectively.

    Compares against measured velocity. The deficit is used
    to change pitch and roll angles to increase and decrease velocity.

    Able to limit tilt angles when tracking reference waypoints.
    """

    max_tilt = _scalar_param(PID_LIMIT, "Maximum tilt angle in radians")

    def __init__(
        self, k_p, k_i, k_d, max_err_i, vehicle: Multirotor,
        max_tilt: float=np.pi / 18
    ):
        """
        Parameters
        ----------
        vehicle : Multirotor
            The vehicle to control.
        max_tilt : float
            Maximum tilt angle in radians
        """
        self.vehicle: Multirotor = vehicle
        super().__init__(k_p, k_i, k_d, max_err_i, n=2)
        self.dtype = self.vehicle.dtype
        self.max_tilt = max_tilt
        self._params = tuple(list(self._params) + ['max_tilt'])


    def step(self, reference, measurement, dt, persist: bool=True):
        # desired pitch, roll
        # ctrl[0] -> x dir -> pitch -> forward
        # ctrl[1] -> y dir -> roll -> lateral
        out = np.empty(2)
        velocity_step(
            self._data.params, self._data.state, self._start,
            _vector(reference), _vector(measurement), float(dt), persist, out)
        return out # desired pitch, roll



class AttController(PIDController):
    """
    Attitude controller. Convert velocity reference into angular rate control
    signal.
    """

    max_acceleration = _scalar_param(PID_MAX_ACCELERATION, "Maximum acceleration in rad/s/s")

    def __init__(
        self, k_p, k_i, k_d, max_err_i, vehicle: Multirotor,
        max_acceleration: float=0.2, max_jerk: float=100.0,
        square_root_scaling: bool=False
    ):
        """
        Parameters
        ----------
        vehicle : Multirotor
            The vehicle to control.
        max_acceleration : float
            Maximum acceleration in rad/s/s
        max_jerk : float
            Maximum jerk in rad/s/s/s
        square_root_scaling : bool
            Whether to scale P-gain with the square root of the error
        """
        self.vehicle: Multirotor = vehicle
        self.max_jerk: float = max_jerk
        self.square_root_scaling: bool = square_root_scaling
        super().__init__(k_p, k_i, k_d, max_err_i, n=3)
        self.dtype = self.vehicle.dtype
        self.max_acceleration = max_acceleration
        if self.square_root_scaling:
            self.k_p = self.max_jerk / self.max_acceleration
        self._params = tuple(list(self._params) + \
            ['max_acceleration', 'max_jerk', 'square_root_scaling'])


    def step(self, reference, measurement, dt, persist: bool=True):
        out = np.empty(3)
        attitude_step(
            self._data.params, self._data.state, self._start,
            _vector(reference), _vector(measurement), float(dt),
            self.square_root_scaling, persist, out)
        return out # Euler rate



class RateController(PIDController):
    """
    Attitude rate controller. Tracks reference roll, pitch, yaw rates and outputs
    the necessary moments about each x,y,z axes to achieve them.

    Uses change in orientation from measured to reference as approximate reference
    angular rate. Compares against measured angular rate. Outputs required change
    in angular rate (angular acceleration) as moments.
    """

    def __init__(
        self, k_p, k_i, k_d, max_err_i, vehicle: Multirotor,
        max_acceleration: float=0.2
    ):
        """
        Parameters
        ----------
        vehicle : Multirotor
            The vehicle to control.
        max_acceleration : float or np.ndarray
            Maximum acceleration in rad/s/s, for all or for each axis.
        """
        self.vehicle: Multirotor = vehicle
        # TODO: check max acceleration limits from ardupilot
        super().__init__(k_p, k_i, k_d, max_err_i, n=3)
        self.dtype = self.vehicle.dtype
        self.max_acceleration = max_acceleration
        self._params = tuple(list(self._params) + ['max_acceleration'])


    @property
    def max_acceleration(self) -> Union[float, np.ndarray]:
        "Maximum acceleration in rad/s/s. A number if it is the same for all axes."
        limit = self._data.params[PID_LIMIT, self._start:self._stop]
        return float(limit[0]) if np.all(limit == limit[0]) else limit.copy()
    @max_acceleration.setter
    def max_acceleration(self, value):
        self._data.params[PID_LIMIT, self._start:self._stop] = value


    def step(self, reference, measurement, dt, persist: bool=True):
        out = np.empty(3)
        rate_step(
            self._data.params, self._data.state, self._start,
            _vector(reference), _vector(measurement), _vector(self.vehicle.orientation),
            np.asarray(self.vehicle.params.inertia_matrix, dtype=np.float64),
            float(dt), persist, out)
        return out # torque



class AltController(PIDController):
    """
    Altitude Controller. Tracks z-position and outputs velocity.

    Uses change in z-position as approximate vertical velocity. Compares against
    measured velocity. Outputs the change in velocity (acceleration) as thrust force,
    given orientation of vehicle.
    """

    max_velocity = _scalar_param(PID_LIMIT, "Maximum velocity in m/s")

    def __init__(
        self, k_p, k_i, k_d, max_err_i, vehicle: Multirotor, max_velocity: float=5
    ):
        """
        Parameters
        ----------
        vehicle : Multirotor
            The vehicle to control.
        max_velocity : float
            Maximum velocity in m/s
        """
        self.vehicle: Multirotor = vehicle
        super().__init__(k_p, 0., 0., max_err_i, n=1) # strictly a P controller
        self.dtype = self.vehicle.dtype
        self.max_velocity = max_velocity
        self._params = tuple(list(self._params) + ['max_velocity'])


    def step(
        self, reference: np.ndarray, measurement: np.ndarray, dt: float = 1, persist: bool=True
    ) -> np.ndarray:
        out = np.empty(1)
        altitude_step(
            self._data.params, self._data.state, self._start,
            _vector(reference), _vector(measurement), float(dt), persist, out)
        return out



class AltRateController(PIDController):
    """
    Climb rate controller. Tracks z-velocity and outputs thrust force needed.
    """

    def __init__(self, k_p, k_i, k_d, max_err_i, vehicle: Multirotor):
        """
        Parameters
        ----------
        vehicle : Multirotor
            The vehicle to control.
        """
        self.vehicle: Multirotor = vehicle
        super().__init__(k_p, k_i, k_d, max_err_i, n=1)
        self.dtype = self.vehicle.dtype


    def step(self, reference, measurement, dt, persist: bool=True):
        out = np.empty(1)
        climb_step(
            self._data.params, self._data.state, self._start,
            _vector(reference), _vector(measurement), _vector(self.vehicle.orientation),
            float(self.vehicle.params.mass), float(self.vehicle.weight), float(dt),
            persist, out)
        return out # thrust force



class Controller:
    """
    The cascaded PID controller. Tracks position and yaw, and outputs thrust and
    moments needed.

        (x,y) --> Position --> Velocity --> Attitude --> Rate --> (Moments)
        (z)   --> Altitude --> Velocity --> (Forces)
    """

    def __init__(
        self,
        ctrl_p: PosController, ctrl_v: VelController,
        ctrl_a: AttController, ctrl_r: RateController,
        ctrl_z: AltController, ctrl_vz: AltRateController,
        period_p: float=1e-1,
        period_a: float=1e-2,
        period_z: float=1e-1,
        feedforward_weight: float=0.
    ):
        """
        Parameters
        ----------
        ctrl_p : PosController
            The position controller.
        ctrl_v : VelController
            The velocity controller.
        ctrl_a : AttController
            The attitude controller.
        ctrl_r : AttRateController
            The attitude rate controller.
        ctrl_z : AltController
            The altitude controller
        ctrl_vz : AltRateController
            The altitude rate controller
        period_[p | a | z] : float, optional
            The time resolution of the [position | attitude | altitude] controller.
            The controller will only renew an action after this interval has passed.
            Otherwise it will apply the last action. For e.g. if interval=1, and vehicle dt=0.1, a
            new action will only be applied every interval/dt=10 steps. by default 1
        feedforward_weight: float, optional
            The weight to assign to a separately specified velocity reference,
            which overrides the velocity signal given by the position controller.
        
        Attributes
        ----------
        vehicle: Multirotor
            The vehicle being controlled.
        """
        self.ctrl_p = ctrl_p
        self.ctrl_v = ctrl_v
        self.ctrl_a = ctrl_a
        self.ctrl_r = ctrl_r
        self.ctrl_z = ctrl_z
        self.ctrl_vz = ctrl_vz
        self.vehicle = self.ctrl_a.vehicle
        self.dtype = self.vehicle.dtype
        self.period_p = period_p
        self.period_a = period_a
        self.period_z = period_z
        self.steps_p = int(self.period_p // self.vehicle.simulation.dt)
        self.steps_a = int(self.period_a // self.vehicle.simulation.dt)
        self.steps_z = int(self.period_z // self.vehicle.simulation.dt)
        self.feedforward_weight = feedforward_weight
        assert self.ctrl_a.vehicle is self.ctrl_p.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_v.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_r.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_z.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_vz.vehicle, "Vehicle instances different."
        # The stages become views of the columns of one data object
        stages = (self.ctrl_p, self.ctrl_v, self.ctrl_a, self.ctrl_r, self.ctrl_z,
                  self.ctrl_vz)
        self._data = ControllerData(sum(stage.n for stage in stages))
        start = 0
        for stage in stages:
            stage._attach(self._data, start)
            start += stage.n
        self.reset()


    def reset(self):
        self.action = np.zeros(4, self.vehicle.dtype)
        self.reference = np.zeros_like(self.action)
        self.thrust = None
        self.torques = None
        self._ref_vel = np.zeros(2, self.vehicle.dtype)
        self._pid_vel = np.zeros_like(self._ref_vel)
        self._scurve_vel = np.zeros_like(self._ref_vel)
        self._pitch_roll = np.zeros(2, self.vehicle.dtype)
        self.n = 0
        self.t = self.vehicle.t
        self.ctrl_p.reset()
        self.ctrl_v.reset()
        self.ctrl_a.reset()
        self.ctrl_r.reset()
        self.ctrl_z.reset()
        self.ctrl_vz.reset()
        return self.state


    def set_params(self, **params):
        self.ctrl_p.set_params(**params.get('ctrl_p', {}))
        self.ctrl_v.set_params(**params.get('ctrl_v', {}))
        self.ctrl_a.set_params(**params.get('ctrl_a', {}))
        self.ctrl_r.set_params(**params.get('ctrl_r', {}))
        self.ctrl_z.set_params(**params.get('ctrl_z', {}))
        self.ctrl_vz.set_params(**params.get('ctrl_vz', {}))
        local_params = {k:v for k,v in params.items() if not isinstance(v, dict)}
        for name, param in local_params.items():
            if hasattr(self, name):
                # Controller class has no np.array attributes, so no casting/
                # in-place assignment needed like tith PIDController class's
                # set_params() method
                setattr(self, name, param)


    def get_params(self) -> Dict[str, Dict[str, np.ndarray]]:
        p = dict(
            ctrl_p=self.ctrl_p.get_params(),
            ctrl_v=self.ctrl_v.get_params(),
            ctrl_a=self.ctrl_a.get_params(),
            ctrl_r=self.ctrl_r.get_params(),
            ctrl_z=self.ctrl_z.get_params(),
            ctrl_vz=self.ctrl_vz.get_params(),
            feedforward_weight=self.feedforward_weight
        )
        return p


    @property
    def state(self) -> np.ndarray:
        return np.concatenate(
            (self.ctrl_p.state, self.ctrl_v.state, self.ctrl_a.state,
            self.ctrl_r.state, self.ctrl_z.state, self.ctrl_vz.state)
        )
    @property
    def vehicle(self) -> Multirotor:
        return self.ctrl_p.vehicle
    @vehicle.setter
    def vehicle(self, v: Multirotor):
        self.ctrl_p.vehicle = v
        self.ctrl_v.vehicle = v
        self.ctrl_a.vehicle = v
        self.ctrl_r.vehicle = v
        self.ctrl_z.vehicle = v
        self.ctrl_vz.vehicle = v


    def step(
        self, reference: np.ndarray, measurement=None, ref_is_error: bool=False,
        feed_forward_velocity: np.ndarray=None, persist: bool=True
    ):
        if ref_is_error:
            error = reference
            ref_xy = self.vehicle.position[:2] + error[:2]
            ref_z = self.vehicle.position[2] + error[2]
            ref_yaw = self.vehicle.orientation[2] + error[3]
        else:
            self.reference = reference
            ref_xy = reference[:2]
            ref_z = reference[2]
            ref_yaw = reference[3]

        if self.n % self.steps_z == 0:
            dt = self.steps_z * self.vehicle.simulation.dt
            ref_vel_z = self.ctrl_z.step(ref_z, self.vehicle.position[2:], dt=dt, persist=persist)
            self.thrust = self.ctrl_vz.step(ref_vel_z, self.vehicle.inertial_velocity[2:], dt=dt, persist=persist)

        if self.n % self.steps_p == 0:
            dt = self.steps_p * self.vehicle.simulation.dt
            self._pid_vel = self._ref_vel = self.ctrl_p.step(ref_xy, self.vehicle.position[:2], dt=dt, persist=persist)
            if feed_forward_velocity is not None:
                self._ref_vel = (self.feedforward_weight * feed_forward_velocity[:2]) + (1 - self.feedforward_weight) * self._pid_vel
            self._ref_vel = np.clip(self._ref_vel, -self.ctrl_p.max_velocity, self.ctrl_p.max_velocity)
        
        if self.n % self.steps_a == 0:
            dt = self.steps_a * self.vehicle.simulation.dt
            self._pitch_roll = self.ctrl_v.step(self._ref_vel, self.vehicle.velocity[:2], dt=dt, persist=persist)
            ref_orientation = np.asarray([self._pitch_roll[1], self._pitch_roll[0], ref_yaw])
            ref_rate = self.ctrl_a.step(
                ref_orientation, self.vehicle.orientation, dt=dt, persist=persist)
            self.torques = self.ctrl_r.step(ref_rate, self.vehicle.euler_rate, dt=dt, persist=persist)

        action = np.asarray([*self.thrust, *self.torques], self.dtype)
        if persist:
            self.n += 1
            self.t = self.vehicle.t
            self.action = action
        return action


    def predict(self, ref, deterministic=True):
        return self.step(reference=ref)


    @classmethod
    def make_for(cls, vehicle: Multirotor) -> Controller:
        """
        Make a `Controller` with p-only parameters for a vehicle. The parameters
        can be learned using the `optimize` function.

        Parameters
        ----------
        vehicle : Multirotor
            The vehicle to control.

        Returns
        -------
        Controller
            The controller with p-only parameters.
        """
        pid = dict(k_p=1., k_i=0, k_d=0, max_err_i=1., vehicle=vehicle)
        ctrl = cls(
            ctrl_p=PosController(**pid, square_root_scaling=True, leashing=True),
            ctrl_v=VelController(**pid),
            ctrl_a=AttController(**pid),
            ctrl_r=RateController(**pid),
            ctrl_z=AltController(**pid),
            ctrl_vz=AltRateController(**pid),
            period_p=1e-1,
            period_a=1e-2,
            period_z=1e-1,
            feedforward_weight=0.
        )
        return ctrl
