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
    dt: float
    "Simulation parameters to set timestep"

    def __post_init__(self):
        self.action = None
        self.err_p = np.atleast_1d(np.zeros_like(self.k_p))
        self.err_i = np.atleast_1d(np.zeros_like(self.k_i))
        self.err_d = np.atleast_1d(np.zeros_like(self.k_d))
        self.err = np.atleast_1d(np.zeros_like(self.k_p))
        if self.max_err_i is None:
            self.max_err_i = np.inf
        else:
            self.max_err_i = np.asarray(self.max_err_i, dtype=self.err.dtype)


    def reset(self):
        self.action = None
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
        self, reference: np.ndarray, measurement: np.ndarray, ref_is_error: bool=False
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
            self.err_i + trapezoid((self.err, err), dx=self.dt, axis=0),
            a_min=-self.max_err_i, a_max=self.max_err_i
        )
        self.err_d = self.k_d * (err - self.err) / self.dt
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
    square_root_scaling: bool = True
    "Whether to scale P-gain with the square root of the error"
    leashing: bool = True
    "Whether to limit proportional position error"


    def __post_init__(self):
        self.k_p = np.ones(2) * np.asarray(self.k_p)
        # Att angle controller is strictly a P controller
        self.k_i = np.zeros(2) * np.asarray(self.k_i)
        self.k_d = np.zeros(2) * np.asarray(self.k_d)
        self.leash = 0 if self.leashing else np.inf
        super().__post_init__()


    def step(self, reference, measurement):
        # inertial frame velocity
        err = reference - measurement
        err_len = np.linalg.norm(err)
        k_p = 0.5 * self.max_jerk / self.max_acceleration
        if self.leashing:
            # https://nrotella.github.io/journal/arducopter-flight-controllers.html
            acc = self.vehicle.inertial_acceleration[:2]
            acc_mag = np.linalg.norm(acc)
            vel = self.vehicle.inertial_velocity[:2]
            vel_mag = np.linalg.norm(vel)
            self.leash = np.abs(acc_mag / (2 * k_p**2) + vel_mag**2 / (2 * acc_mag + 1e-6))
            self.leash = np.inf if self.leash==0 else self.leash
            err_unit = err / (err_len + 1e-6)
            err_len = min(err_len, self.leash)
            err = err_unit * err_len
            self.err = err
            velocity = k_p * err
        if err_len > 0. and self.square_root_scaling:
            velocity = np.zeros_like(self.k_p)
            velocity[0] = sqrt_control(err[0], k_p, self.max_acceleration, self.dt)
            velocity[1] = sqrt_control(err[1], k_p, self.max_acceleration, self.dt)
            # scale velocity correction by error size
            # velocity = (np.abs(err) / err_len) * velocity
            self.err = err
        else:
            velocity = super().step(reference, measurement)
        # convert to body-frame velocity
        roll, pitch, yaw = self.vehicle.orientation
        rot = np.asarray([
            [np.cos(yaw),   np.sin(yaw)],
            [-np.sin(yaw),  np.cos(yaw)],
        ])
        ref_velocity = rot @ velocity
        abs_max_vel = np.abs((ref_velocity / (np.linalg.norm(ref_velocity) + 1e-6)) * self.max_velocity)
        ref_velocity = np.clip(ref_velocity, a_min=-abs_max_vel, a_max=abs_max_vel)
        self.action = ref_velocity
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


    def step(self, reference, measurement):
        # desired pitch, roll
        pitch_roll = super().step(reference, measurement)
        # ctrl[0] -> x dir -> pitch -> forward
        # ctrl[1] -> y dir -> roll -> lateral
        pitch_roll[0:2] = np.clip(pitch_roll[0:2], a_min=-self.max_tilt, a_max=self.max_tilt)
        pitch_roll[1] *= -1 # +y motion requires negative roll
        self.action = pitch_roll
        return self.action # desired pitch, roll



@dataclass
class AttController(PIDController):

    vehicle: Multirotor
    max_acceleration: float = 0.2
    "Maximum acceleration in rad/s/s"
    max_jerk: float = 10.0
    "Maximum jerk in rad/s/s/s"
    square_root_scaling: bool = False
    "Whether to scale P-gain with the square root of the error"


    def __post_init__(self):
        self.k_p = np.ones(3) * np.asarray(self.k_p)
        # Att angle controller is strictly a P controller
        self.k_i = np.zeros(3) * np.asarray(self.k_i)
        self.k_d = np.zeros(3) * np.asarray(self.k_d)
        super().__post_init__()


    def step(self, reference, measurement):
        err = reference - measurement
        err_len = np.linalg.norm(err)
        if self.square_root_scaling and err_len > 0:
            k_p = self.max_jerk / self.max_acceleration
            velocity = np.zeros_like(self.k_p)
            velocity[0] = sqrt_control(err[0], k_p, self.max_acceleration, self.dt)
            velocity[1] = sqrt_control(err[1], k_p, self.max_acceleration, self.dt)
            velocity[2] = sqrt_control(err[2], k_p, self.max_acceleration, self.dt)
            # velocity = (np.abs(err) / err_len) * velocity
            self.err = err
            self.err_p = velocity
        else:
            velocity = super().step(reference=reference, measurement=measurement)
        self.action = velocity
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


    def __post_init__(self):
        self.k_p = np.ones(3) * np.asarray(self.k_p)
        self.k_i = np.ones(3) * np.asarray(self.k_i)
        self.k_d = np.ones(3) * np.asarray(self.k_d)
        super().__post_init__()


    def step(self, reference, measurement):
        # desired change in angular velocity
        ref = euler_to_angular_rate(reference, self.vehicle.orientation)
        # actual change in angular velocity
        mea = euler_to_angular_rate(measurement, self.vehicle.orientation)
        # prescribed change in velocity i.e. angular acc
        acceleration = super().step(reference=ref, measurement=mea)
        # torque = moment of inertia . angular_acceleration
        self.action = self.vehicle.params.inertia_matrix.dot(acceleration)
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


    def __post_init__(self):
        self.k_p = np.ones(1) * np.asarray(self.k_p)
        # Alt controller is strictly a P controller
        self.k_i = np.zeros(1) * np.asarray(self.k_i)
        self.k_d = np.zeros(1) * np.asarray(self.k_d)
        super().__post_init__()


            
@dataclass
class AltRateController(PIDController):
    """
    Climb rate controller. Tracks z-velocity and outputs thrust force needed.
    """

    vehicle: Multirotor


    def step(self, reference, measurement):
            roll, pitch, yaw = self.vehicle.orientation
            # change in velocity i.e. acceleration
            ctrl = super().step(reference=reference, measurement=measurement)
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
        interval: float=None
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
        interval : float, optional
            The time resolution of the controller. The controller will only renew
            an action after this interval has passed. Otherwise it will apply
            the last action. For e.g. if interval=1, and vehicle dt=0.1, a
            new action will only be applied every 10 steps. by default None
        """
        self.ctrl_p = ctrl_p
        self.ctrl_v = ctrl_v
        self.ctrl_a = ctrl_a
        self.ctrl_r = ctrl_r
        self.ctrl_z = ctrl_z
        self.ctrl_vz = ctrl_vz
        self.vehicle = self.ctrl_a.vehicle
        self.interval = self.ctrl_a.vehicle.simulation.dt if interval is None else interval
        self.interval_n = int(self.interval // self.ctrl_a.vehicle.simulation.dt)
        self.action = None
        self.n = 0
        assert self.ctrl_a.vehicle is self.ctrl_p.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_v.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_r.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_z.vehicle, "Vehicle instances different."
        assert self.ctrl_a.vehicle is self.ctrl_vz.vehicle, "Vehicle instances different."


    def reset(self):
        self.n = 0
        self.action = None
        self.ctrl_a.reset()
        self.ctrl_p.reset()
        self.ctrl_z.reset()
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
        if self.n % self.interval_n != 0:
            self.n += 1
            return self.action
        # If the reference argument is the relative error, and not the absolute
        # value. Using the relationship error = reference - measurement to
        # get absolute reference values
        if ref_is_error:
            error = reference
            ref_xy = self.vehicle.position[:2] + error[:2]
            ref_z = self.vehicle.position[2] + error[2]
            ref_yaw = self.vehicle.orientation[2] + error[3]
        else:
            ref_xy = reference[:2]
            ref_z = reference[2]
            ref_yaw = reference[3]

        ref_vel_z = self.ctrl_z.step(ref_z, self.vehicle.position[2:])
        thrust = self.ctrl_vz.step(ref_vel_z, self.vehicle.inertial_velocity[2:])

        ref_vel = self.ctrl_p.step(ref_xy, self.vehicle.position[:2])
        if feed_forward_velocity is not None:
            ref_vel += feed_forward_velocity
            ref_vel = np.clip(ref_vel, -self.ctrl_p.max_velocity, self.ctrl_p.max_velocity)
        pitch_roll = self.ctrl_v.step(ref_vel, self.vehicle.velocity[:2])
        ref_orientation = np.asarray([pitch_roll[1], pitch_roll[0], ref_yaw])
        ref_rate = self.ctrl_a.step(ref_orientation, self.vehicle.orientation)
        torques = self.ctrl_a.step(ref_rate, self.vehicle.euler_rate)

        self.action = np.asarray([*thrust, *torques])
        self.n += 1
        return self.action
