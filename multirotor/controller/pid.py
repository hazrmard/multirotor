from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from numba import njit

from ..simulation import Multirotor
from ..coords import euler_to_angular_rate

# See more:
# https://archive.ph/3Aco3
# https://ardupilot.org/dev/docs/apmcopter-code-overview.html
# https://ardupilot.org/copter/docs/traditional-helicopter-control-system.html
# https://ardupilot.org/dev/docs/apmcopter-programming-attitude-control-2.html
# https://www.youtube.com/watch?v=-PC69jcMizA


@njit
def sqrt_control(err: float, k_p: float, derr_dt2_lim: float, dt: float):
    # piece-wise P-controller
    # see: https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_Math/control.cpp#L387
    correction_rate = 0.
    if derr_dt2_lim <= 0:
        correction_rate = err * k_p
    elif k_p == 0:
        if err > 0:
            correction_rate = np.sqrt(2. * derr_dt2_lim * err)
        if err < 0:
            correction_rate = -np.sqrt(2. * derr_dt2_lim * (-err))
        else:
            correction_rate = 0.
    else:
        linear_dist = derr_dt2_lim / k_p**2
        if err > linear_dist:
            correction_rate = np.sqrt(2. * derr_dt2_lim * (err - (linear_dist / 2.)))
        elif err < -linear_dist:
            correction_rate = -np.sqrt(2. * derr_dt2_lim * (-err - (linear_dist / 2.)))
        else:
            correction_rate = err * k_p
    if dt != 0.:
        abs_val = np.abs(err) / dt
        return min(max(-abs_val, correction_rate), abs_val)
    else:
        return correction_rate


@dataclass
class PIDController:
    """
    Proportional Integral Derivative controller. Tracks a reference signal and outputs
    a control signal to minimize output. The equation is:

        err = reference - measurement

        u = k_p * err + k_d * d(err)/dt + k_i * int(err . dt)
    
    Can control a single or an array of signals, given float or array PID constants.
    """

    k_p: np.ndarray
    "Proportional constant"
    k_i: np.ndarray
    "Integral constant"
    k_d: np.ndarray
    "Derivative constant"
    max_err_i: np.ndarray
    "Maximum accumulated error"

    def __post_init__(self):
        self.err_p = np.atleast_1d(np.zeros_like(self.k_p))
        self.err_i = np.atleast_1d(np.zeros_like(self.k_i))
        self.err_d = np.atleast_1d(np.zeros_like(self.k_d))
        self.err = np.atleast_1d(np.zeros_like(self.k_p))
        if self.max_err_i is None:
            self.max_err_i = np.inf
        else:
            self.max_err_i = np.asarray(self.max_err_i, dtype=self.err.dtype)
        self.action = np.zeros_like(self.err)


    def reset(self):
        self.action = np.zeros_like(self.err)
        self.err *= 0
        self.err_p *= 0
        self.err_i *= 0
        self.err_d *= 0


    @property
    def state(self) -> np.ndarray:
        return np.concatenate((
            np.atleast_1d(self.err_p),
            np.atleast_1d(self.err_i),
            np.atleast_1d(self.err_d)
        ))


    def step(
        self, reference: np.ndarray, measurement: np.ndarray, dt: float=1.,
        ref_is_error: bool=False
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

        Returns
        -------
        np.ndarray
            The action signal.
        """
        if ref_is_error:
            err = reference
        else:
            err = reference - measurement
        self.err_p = self.k_p * err
        self.err_i = self.k_i * np.clip(
            self.err_i + trapezoid((self.err, err), dx=dt, axis=0),
            a_min=-self.max_err_i, a_max=self.max_err_i
        )
        self.err_d = self.k_d * (err - self.err) / dt
        self.err = err
        self.action = self.err_p + self.err_i + self.err_d
        return self.action



@dataclass
class PosController(PIDController):
    """
    Position controller. Convert xy-waypoint into required xy-velocity in the body-frame.
    """

    vehicle: Multirotor
    max_velocity: float = 7.0
    "Maximum velocity in m/s"
    max_acceleration: float = 3.
    "Maximum acceleration in m/s/s"
    max_jerk: float = 100.0
    "Maximum jerk in m/s/s/s"
    square_root_scaling: bool = False
    "Whether to scale P-gain with the square root of the error"
    leashing: bool = False
    "Whether to limit proportional position error"


    def __post_init__(self):
        self.k_p = np.ones(2) * np.asarray(self.k_p)
        self.k_i = np.ones(2) * np.asarray(self.k_i)
        self.k_d = np.ones(2) * np.asarray(self.k_d)
        if self.leashing or self.square_root_scaling:
            self.k_p[:] = 0.5 * self.max_jerk / self.max_acceleration
        super().__post_init__()


    @property
    def leash(self) -> float:
        if not self.leashing:
            return np.inf
        # https://nrotella.github.io/journal/arducopter-flight-controllers.html
        acc = self.vehicle.inertial_acceleration[:2]
        acc_mag = np.linalg.norm(acc)
        vel = self.vehicle.inertial_velocity[:2]
        vel_mag = np.linalg.norm(vel)
        leash = np.abs(acc_mag / (2 * self.k_p[0]**2) + vel_mag**2 / (2 * acc_mag + 1e-6))
        # if leash is very small such that vehicle will cover that distance under
        # current velocity, then a leash is not needed. It is more useful for
        # larger errors and bigger speeds.
        leash = np.inf if leash <= (vel_mag * self.vehicle.simulation.dt) else leash
        return leash


    def step(self, reference, measurement, dt):
        # inertial frame velocity
        err = reference - measurement
        err_len = np.linalg.norm(err)
        # TODO check conditional logic
        if self.leashing:
            err_unit = err / (err_len + 1e-6)
            err_len = min(err_len, self.leash)
            err = err_unit * err_len
            self.err = err
            velocity = self.err_p = self.k_p * err
        if err_len > 0. and self.square_root_scaling:
            velocity = np.zeros_like(self.k_p)
            velocity[0] = sqrt_control(err[0], self.k_p[0], self.max_acceleration, dt)
            velocity[1] = sqrt_control(err[1], self.k_p[1], self.max_acceleration, dt)
            self.err = err
            self.err_p = velocity
        else:
            velocity = super().step(reference, measurement, dt=dt)
        # convert to body-frame velocity
        roll, pitch, yaw = self.vehicle.orientation
        cos, sin = np.cos(yaw), np.sin(yaw)
        rot = np.asarray([
            [cos,   sin],
            [-sin, cos],
        ])
        ref_velocity = rot @ velocity
        ref_velocity_mag = np.linalg.norm(ref_velocity)
        ref_velocity_unit = ref_velocity / (ref_velocity_mag + 1e-6)
        self.action = ref_velocity_unit * min(ref_velocity_mag, self.max_velocity)
        return self.action


@dataclass
class VelController(PIDController):
    """
    Velocity controller. Takes reference x/y body-frame velocity and outputs reference 
    pitch and roll angles for x and y motion, respectively.

    Compares against measured velocity. The deficit is used
    to change pitch and roll angles to increase and decrease velocity.

    Able to limit tilt angles when tracking reference waypoints.
    """

    vehicle: Multirotor
    max_tilt: float = np.pi / 18
    "Maximum tilt angle in radians"



    def __post_init__(self):
        self.k_p = np.ones(2) * np.asarray(self.k_p)
        self.k_i = np.ones(2) * np.asarray(self.k_i)
        self.k_d = np.ones(2) * np.asarray(self.k_d)
        super().__post_init__()


    def step(self, reference, measurement, dt):
        # desired pitch, roll
        pitch_roll = super().step(reference, measurement, dt=dt)
        # ctrl[0] -> x dir -> pitch -> forward
        # ctrl[1] -> y dir -> roll -> lateral
        pitch_roll[0:2] = np.clip(pitch_roll[0:2], a_min=-self.max_tilt, a_max=self.max_tilt)
        pitch_roll[1] *= -1 # +y motion requires negative roll
        self.action = pitch_roll
        return self.action # desired pitch, roll



@dataclass
class AttController(PIDController):
    """
    Attitude controller. Convert velocity reference into angular rate control
    signal.
    """

    vehicle: Multirotor
    max_acceleration: float = 0.2
    "Maximum acceleration in rad/s/s"
    max_jerk: float = 100.0
    "Maximum jerk in rad/s/s/s"
    square_root_scaling: bool = False
    "Whether to scale P-gain with the square root of the error"


    def __post_init__(self):
        self.k_p = np.ones(3) * np.asarray(self.k_p)
        self.k_i = np.ones(3) * np.asarray(self.k_i)
        self.k_d = np.ones(3) * np.asarray(self.k_d)
        if self.square_root_scaling:
            self.k_p[:] = self.max_jerk / self.max_acceleration
        super().__post_init__()


    def step(self, reference, measurement, dt):
        err = reference - measurement
        err_len = np.linalg.norm(err)
        if self.square_root_scaling and err_len > 0:
            velocity = np.zeros_like(self.k_p)
            velocity[0] = sqrt_control(err[0], self.k_p[0], self.max_acceleration, dt)
            velocity[1] = sqrt_control(err[1], self.k_p[1], self.max_acceleration, dt)
            velocity[2] = sqrt_control(err[2], self.k_p[2], self.max_acceleration, dt)
            # velocity = (np.abs(err) / err_len) * velocity
            self.err = err
            self.err_p = velocity
        else:
            velocity = super().step(reference=reference, measurement=measurement, dt=dt)
        self.action = velocity # Euler rate
        return self.action



@dataclass
class RateController(PIDController):
    """
    Attitude rate controller. Tracks reference roll, pitch, yaw rates and outputs
    the necessary moments about each x,y,z axes to achieve them.

    Uses change in orientation from measured to reference as approximate reference
    angular rate. Compares against measured angular rate. Outputs required change
    in angular rate (angular acceleration) as moments.
    """

    vehicle: Multirotor
    # TODO: check max acceleration limits from ardupilot
    max_acceleration: float = 0.2
    "Maximum acceleration in rad/s/s"


    def __post_init__(self):
        self.k_p = np.ones(3) * np.asarray(self.k_p)
        self.k_i = np.ones(3) * np.asarray(self.k_i)
        self.k_d = np.ones(3) * np.asarray(self.k_d)
        super().__post_init__()


    def step(self, reference, measurement, dt):
        # desired angular velocity
        ref = euler_to_angular_rate(reference, self.vehicle.orientation)
        # ref = reference
        # actual change in angular velocity
        mea = euler_to_angular_rate(measurement, self.vehicle.orientation)
        # mea = measurement
        # prescribed change in velocity i.e. angular acc
        self.action = np.clip(
            super().step(reference=ref, measurement=mea, dt=dt),
            -self.max_acceleration, self.max_acceleration
        )
        # torque = moment of inertia . angular_acceleration
        self.action = self.vehicle.params.inertia_matrix.dot(self.action)
        return self.action



@dataclass
class AltController(PIDController):
    """
    Altitude Controller. Tracks z-position and outputs velocity.

    Uses change in z-position as approximate vertical velocity. Compares against
    measured velocity. Outputs the change in velocity (acceleration) as thrust force,
    given orientation of vehicle.
    """

    vehicle: Multirotor
    max_velocity: float = 5


    def __post_init__(self):
        self.k_p = np.ones(1) * np.asarray(self.k_p)
        # Alt controller is strictly a P controller
        self.k_i = np.zeros(1) * np.asarray(self.k_i)
        self.k_d = np.zeros(1) * np.asarray(self.k_d)
        super().__post_init__()


    def step(
        self, reference: np.ndarray, measurement: np.ndarray, dt: float = 1
    ) -> np.ndarray:
        self.action = super().step(reference, measurement, dt)
        self.action = np.clip(self.action, a_min=-self.max_velocity, a_max=self.max_velocity)
        return self.action


            
@dataclass
class AltRateController(PIDController):
    """
    Climb rate controller. Tracks z-velocity and outputs thrust force needed.
    """

    vehicle: Multirotor


    def step(self, reference, measurement, dt):
            roll, pitch, yaw = self.vehicle.orientation
            # change in velocity i.e. acceleration
            ctrl = super().step(reference=reference, measurement=measurement, dt=dt)
            # convert acceleration to required z-force, given orientation
            ctrl = self.vehicle.params.mass * (
                    ctrl / (np.cos(roll) * np.cos(pitch))
                ) + \
                self.vehicle.weight
            self.action = ctrl
            return ctrl # thrust force



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
        interval_p: float=1.,
        interval_a: float=1.,
        interval_z: float=1.
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
        interval_[p | a | z] : float, optional
            The time resolution of the [position | attitude | altitude] controller.
            The controller will only renew an action after this interval has passed.
            Otherwise it will apply the last action. For e.g. if interval=1, and vehicle dt=0.1, a
            new action will only be applied every interval/dt=10 steps. by default 1
        """
        self.ctrl_p = ctrl_p
        self.ctrl_v = ctrl_v
        self.ctrl_a = ctrl_a
        self.ctrl_r = ctrl_r
        self.ctrl_z = ctrl_z
        self.ctrl_vz = ctrl_vz
        self.vehicle = self.ctrl_a.vehicle
        self.interval_p = interval_p
        self.interval_a = interval_a
        self.interval_z = interval_z
        self.steps_p = int(self.interval_p // self.vehicle.simulation.dt)
        self.steps_a = int(self.interval_a // self.vehicle.simulation.dt)
        self.steps_z = int(self.interval_z // self.vehicle.simulation.dt)
        assert self.ctrl_a.vehicle is self.ctrl_p.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_v.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_r.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_z.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_vz.vehicle, "Vehicle instances different."
        self.reset()


    def reset(self):
        self.action = np.zeros(4, self.vehicle.dtype)
        self.thrust = None
        self.torques = None
        self._ref_vel = np.zeros(2, self.vehicle.dtype)
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


    @property
    def state(self) -> np.ndarray:
        return np.concatenate(
            (self.ctrl_p.state, self.ctrl_a.state, self.ctrl_z.state)
        )


    def step(
        self, reference: np.ndarray, measurement=None, ref_is_error: bool=False,
        feed_forward_velocity: np.ndarray=None
    ):
        if ref_is_error:
            error = reference
            ref_xy = self.vehicle.position[:2] + error[:2]
            ref_z = self.vehicle.position[2] + error[2]
            ref_yaw = self.vehicle.orientation[2] + error[3]
        else:
            ref_xy = reference[:2]
            ref_z = reference[2]
            ref_yaw = reference[3]

        if self.n % self.steps_z == 0:
            dt = self.steps_z * self.vehicle.simulation.dt
            ref_vel_z = self.ctrl_z.step(ref_z, self.vehicle.position[2:], dt=dt)
            self.thrust = self.ctrl_vz.step(ref_vel_z, self.vehicle.inertial_velocity[2:], dt=dt)

        if self.n % self.steps_p == 0:
            dt = self.steps_p * self.vehicle.simulation.dt
            self._ref_vel = self.ctrl_p.step(ref_xy, self.vehicle.position[:2], dt=dt)
            if feed_forward_velocity is not None:
                self._ref_vel += feed_forward_velocity
                self._ref_vel = np.clip(self._ref_vel, -self.ctrl_p.max_velocity, self.ctrl_p.max_velocity)
        
        if self.n % self.steps_a == 0:
            dt = self.steps_a * self.vehicle.simulation.dt
            self._pitch_roll = self.ctrl_v.step(self._ref_vel, self.vehicle.velocity[:2], dt=dt)
            ref_orientation = np.asarray([self._pitch_roll[1], self._pitch_roll[0], ref_yaw])
            ref_rate = self.ctrl_a.step(ref_orientation, self.vehicle.orientation, dt=dt)
            self.torques = self.ctrl_r.step(ref_rate, self.vehicle.euler_rate, dt=dt)

        self.action = np.asarray([*self.thrust, *self.torques])
        self.n += 1
        self.t = self.vehicle.t
        return self.action


    def predict(self, ref, deterministic=True):
        return self.step(reference=ref)
