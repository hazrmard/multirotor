"""
The cascaded PID controller. `Controller` tracks a position and yaw and outputs
thrust and moments, through six stages that each wrap the same PID law::

    (x,y) --> PosController --> VelController --> AttController --> RateController --> (moments)
    (z)   --> AltController --> AltRateController --> (thrust)

The stages (`PosController`, ...) are passive containers of the PID parameters and
state of their control channels. They compute nothing and know no vehicle. A
`Controller` is the executor, as `Multirotor` is for `VehicleParams`: it takes
over the stages' values into one `ControllerData`, which holds them as matrices
with one column per control channel, and passes the matrices and its vehicle's
state to the compiled functions in `multirotor.controller.cascade`.

`PIDController` is the one stage that executes: a generic PID controller whose
``step`` runs the PID law (`multirotor.controller.cascade.pid_stage`). It can be
used on its own, or to build a cascade with other stages.

Call chain of `Controller.step`, between the vehicle's steps. The stages do not
call each other: ``cascaded_step`` calls the stage functions in order::

    Controller.step ---> cascade.cascaded_step
        --> cascade.altitude_stage
        --> cascade.climb_stage
        --> cascade.position_stage
        --> cascade.velocity_stage
        --> cascade.attitude_stage
        --> cascade.rate_stage

The functions take the controller's state and return its new state, so a step with
``persist=False`` is one whose new state is dropped. `multirotor.controller.cascade`
describes the stages and the arrays.

Altitude and position stages run every ``period_z`` and ``period_p`` seconds. The
others run every ``period_a`` seconds.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from .._compile import step_function
from ..simulation import Multirotor
from .cascade import CascadeStepInputs, cascaded_step, leash_distance, pid_stage
from .cascade import sqrt_control # noqa: F401, kept importable from here
from ..tuples import ControllerParams, ControllerState, Memo, Steps
from .data import (
    ControllerData, PID_K_P, PID_K_I, PID_K_D, PID_MAX_ERR_I, PID_LIMIT,
    PID_MAX_ACCELERATION, PID_ERR, PID_ERR_P, PID_ERR_I, PID_ERR_D, PID_REFERENCE,
    PID_ACTION, PID_INTEGRAL, N_PID_STATES, N_CASCADE_CHANNELS
)

# See more:
# https://archive.ph/3Aco3
# https://ardupilot.org/dev/docs/apmcopter-code-overview.html
# https://ardupilot.org/copter/docs/traditional-helicopter-control-system.html
# https://ardupilot.org/dev/docs/apmcopter-programming-attitude-control-2.html
# https://www.youtube.com/watch?v=-PC69jcMizA


Gain = Union[float, np.ndarray]
"A parameter of a stage: one number for all its channels, or one per channel."


# The compiled steps. Each takes the state it advances first (see
# `multirotor._compile.step_function`) and returns one array, so that a step converts one
# array to numpy.

@step_function
def _pid_stage(
    x: jax.Array, pid_params: jax.Array, reference: jax.Array, measurement: jax.Array,
    dt: float
) -> jax.Array:
    """`pid_stage`. Returns the stage's new state, whose ``PID_ACTION`` row is the action."""
    return pid_stage(pid_params, x, reference, measurement, dt)[1]


N_CONTROLLER_STATE = N_PID_STATES * N_CASCADE_CHANNELS + 6
"The length of a controller's state as one array: the state matrix, raveled, and the memo."


def _cascaded_step(inputs: jax.Array, fixed: CascadeStepInputs) -> jax.Array:
    """
    `cascaded_step`, with what changes at every step as one array.

    ``inputs``: The controller's state matrix (raveled), then the memo's velocity reference
      (2), thrust (1) and torques (3), which together are the first `N_CONTROLLER_STATE`
      elements; then the reference (4), the vehicle's body state (12) and acceleration (3),
      the feed-forward velocity (2), whether to use it (1, as a float), and the number of
      steps taken (1, as a float).
    ``fixed``: The other inputs, which change only when the controller's parameters do. Its
      fields for the inputs in ``inputs`` are ignored.
    Returns: The first `N_CONTROLLER_STATE` elements of ``inputs`` after the step, then the
      action (4).
    """
    size = N_PID_STATES * N_CASCADE_CHANNELS
    x = inputs[:size].reshape(N_PID_STATES, N_CASCADE_CHANNELS)
    memo = Memo(inputs[size:size + 2], inputs[size + 2:size + 3], inputs[size + 3:size + 6])
    moving = inputs[N_CONTROLLER_STATE:]
    step_inputs = fixed._replace(
        reference=moving[:4], body_state=moving[4:16], acceleration=moving[16:19],
        feed_forward=moving[19:21], use_feed_forward=moving[21] > .5)
    x, memo, action = cascaded_step(x, memo, step_inputs, moving[22].astype(int))
    return jnp.concatenate((x.ravel(), memo.ref_vel, memo.thrust, memo.torques, action))


_compiled_cascaded_step = step_function(_cascaded_step)
_leash_distance = step_function(leash_distance)


def _vector(x: Union[float, np.ndarray]) -> np.ndarray:
    """A number or array as a float64 vector, without a copy if it already is one."""
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _state64(vehicle: Multirotor) -> np.ndarray:
    """The state of a vehicle as float64, without a copy if it already is."""
    return np.asarray(vehicle.state, dtype=np.float64)


def _param(row: int, doc: str) -> property:
    """A property that views one row of the parameter matrix, over a stage's columns."""
    def get(self: PIDStage) -> np.ndarray:
        "The row, over the stage's columns."
        return self._data.params[row, self._start:self._stop]
    def set(self: PIDStage, value: Gain) -> None:
        "Set the row, over the stage's columns."
        self._data.params[row, self._start:self._stop] = value
    return property(get, set, doc=doc)


def _state(row: int, doc: str) -> property:
    """A property that views one row of the state matrix, over a stage's columns."""
    def get(self: PIDStage) -> np.ndarray:
        "The row, over the stage's columns."
        return self._data.state[row, self._start:self._stop]
    def set(self: PIDStage, value: Gain) -> None:
        "Set the row, over the stage's columns."
        self._data.state[row, self._start:self._stop] = value
    return property(get, set, doc=doc)


def _scalar_param(row: int, doc: str) -> property:
    """A property for a parameter of a stage that has one value for all its channels."""
    def get(self: PIDStage) -> float:
        "The value of the first channel."
        return float(self._data.params[row, self._start])
    def set(self: PIDStage, value: float) -> None:
        "Set the value of every channel."
        self._data.params[row, self._start:self._stop] = value
    return property(get, set, doc=doc)


class PIDStage:
    """
    The parameters and state of one or more PID control channels. A passive
    container: it only holds values, and computes nothing. The PID equation that
    the values are for is:

        err = reference - measurement

        u = k_p * err + k_d * d(err)/dt + k_i * int(err . dt)

    The constants and the state are views of columns of a `ControllerData`. A
    stage owns one until a `Controller` takes it over.
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
        self, k_p: Gain, k_i: Gain, k_d: Gain, max_err_i: Optional[Gain], n: Optional[int] = None
    ) -> None:
        """
        Parameters
        ----------
        k_p, k_i, k_d : float or np.ndarray
            Proportional, integral and derivative constants. A number applies to
            all channels.
        max_err_i : float or np.ndarray, optional
            Maximum accumulated error. None for no limit.
        n : int, optional
            The number of channels, by default the size of the largest constant.
        """
        if n is None:
            n = max(np.size(k_p), np.size(k_i), np.size(k_d), 1)
        self._bind(ControllerData(n), 0, n)
        self._adopted: bool = False
        self.k_p, self.k_i, self.k_d = k_p, k_i, k_d
        self.max_err_i = np.inf if max_err_i is None else max_err_i
        self._params = ('k_p', 'k_i', 'k_d', 'max_err_i')


    def _bind(self, data: ControllerData, start: int, n: int) -> None:
        """Make this controller a view of ``n`` columns of ``data``, from ``start``."""
        self._data: ControllerData = data
        self._start: int = start
        self._stop: int = start + n
        self.n: int = n


    def _adopt(self, data: ControllerData, start: int) -> None:
        """Move this controller, with its values, to columns of ``data`` from ``start``."""
        self._adopted = True
        stop = start + self.n
        data.params[:, start:stop] = self._data.params[:, self._start:self._stop]
        data.state[:, start:stop] = self._data.state[:, self._start:self._stop]
        self._bind(data, start, self.n)


    def reset(self) -> None:
        "Set the state of every channel to zero."
        self._data.state[:, self._start:self._stop] = 0.


    def set_params(self, **params: Union[np.ndarray, bool, float, int]) -> None:
        """
        Set parameters by name. A parameter that is an array is assigned in place.

        Parameters
        ----------
        **params : np.ndarray, bool, float or int
            The new values, by the names in `get_params`.

        Raises
        ------
        AttributeError
            For a name that is not an attribute.
        """
        for name, param in params.items():
            if not hasattr(self, name):
                raise AttributeError('Attribute %s not part of class.' % name)
            attr = getattr(self, name)
            if isinstance(attr, np.ndarray) and attr.ndim > 0:
                # Assign in place, which broadcasts a number to every channel. The
                # attribute may be a copy of the matrix's values, so set it back.
                attr[:] = np.asarray(param, dtype=attr.dtype)
                param = attr
            setattr(self, name, param)


    def get_params(self) -> Dict[str, Any]:
        """
        The stage's parameters.

        Returns
        -------
        dict
            The value of each parameter, by name.
        """
        return {name: getattr(self, name) for name in self._params}


    @property
    def state(self) -> np.ndarray:
        "The proportional, integral and derivative terms at the last step, concatenated."
        return np.concatenate((self.err_p, self.err_i, self.err_d))



class PIDController(PIDStage):
    """
    Proportional Integral Derivative controller. Tracks a reference signal and
    outputs a control signal to minimize the error, using the PID equation of
    `PIDStage`. Can control a single signal or an array of signals, given float or
    array PID constants.
    """

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
        # A reference that is the error is a measurement of zero
        mea = np.zeros(shape) if ref_is_error \
            else np.array(np.broadcast_to(_vector(measurement), shape))
        columns = slice(self._start, self._stop)
        state = self._data.state[:, columns]
        new_state = np.asarray(_pid_stage(
            state, self._data.params[:, columns], ref, mea, float(dt)))
        if persist:
            state[:] = new_state
        return new_state[PID_ACTION].copy()



class PosController(PIDStage):
    """
    Position controller. Convert xy-waypoint into required xy-velocity in the body-frame.
    """

    max_velocity = _scalar_param(PID_LIMIT, "Maximum velocity in m/s")
    max_acceleration = _scalar_param(PID_MAX_ACCELERATION, "Maximum acceleration in m/s/s")

    def __init__(
        self, k_p: Gain, k_i: Gain, k_d: Gain, max_err_i: Optional[Gain],
        max_velocity: float=7.0, max_acceleration: float=3., max_jerk: float=100.0,
        square_root_scaling: bool=True, leashing: bool=True
    ) -> None:
        """
        Parameters
        ----------
        k_p, k_i, k_d : float or np.ndarray
            Proportional, integral and derivative constants.
        max_err_i : float or np.ndarray, optional
            Maximum accumulated error. None for no limit.
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
        self.max_jerk: float = max_jerk
        self.square_root_scaling: bool = square_root_scaling
        self.leashing: bool = leashing
        super().__init__(k_p, k_i, k_d, max_err_i, n=2)
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        if self.leashing or self.square_root_scaling:
            self.k_p = 0.5 * self.max_jerk / self.max_acceleration
        self._params += (
            'max_velocity', 'max_acceleration', 'max_jerk', 'square_root_scaling', 'leashing')


class VelController(PIDStage):
    """
    Velocity controller. Takes a reference x/y body-frame velocity (m/s) and
    outputs reference pitch and roll angles (rad) for x and y motion.

    The error between reference and measured velocity sets the pitch and roll
    angles that speed the vehicle up or slow it down.

    Limits the tilt angles with ``max_tilt``.
    """

    max_tilt = _scalar_param(PID_LIMIT, "Maximum tilt angle in radians")

    def __init__(
        self, k_p: Gain, k_i: Gain, k_d: Gain, max_err_i: Optional[Gain],
        max_tilt: float=np.pi / 18
    ) -> None:
        """
        Parameters
        ----------
        k_p, k_i, k_d : float or np.ndarray
            Proportional, integral and derivative constants.
        max_err_i : float or np.ndarray, optional
            Maximum accumulated error. None for no limit.
        max_tilt : float
            Maximum tilt angle in radians
        """
        super().__init__(k_p, k_i, k_d, max_err_i, n=2)
        self.max_tilt = max_tilt
        self._params += ('max_tilt',)


class AttController(PIDStage):
    """
    Attitude controller. Convert velocity reference into angular rate control
    signal.
    """

    max_acceleration = _scalar_param(PID_MAX_ACCELERATION, "Maximum acceleration in rad/s/s")

    def __init__(
        self, k_p: Gain, k_i: Gain, k_d: Gain, max_err_i: Optional[Gain],
        max_acceleration: float=0.2, max_jerk: float=100.0,
        square_root_scaling: bool=False
    ) -> None:
        """
        Parameters
        ----------
        k_p, k_i, k_d : float or np.ndarray
            Proportional, integral and derivative constants.
        max_err_i : float or np.ndarray, optional
            Maximum accumulated error. None for no limit.
        max_acceleration : float
            Maximum acceleration in rad/s/s
        max_jerk : float
            Maximum jerk in rad/s/s/s
        square_root_scaling : bool
            Whether to scale P-gain with the square root of the error
        """
        self.max_jerk: float = max_jerk
        self.square_root_scaling: bool = square_root_scaling
        super().__init__(k_p, k_i, k_d, max_err_i, n=3)
        self.max_acceleration = max_acceleration
        if self.square_root_scaling:
            self.k_p = self.max_jerk / self.max_acceleration
        self._params += ('max_acceleration', 'max_jerk', 'square_root_scaling',)


class RateController(PIDStage):
    """
    Attitude rate controller. Tracks reference roll, pitch, yaw rates and outputs
    the necessary moments about each x,y,z axes to achieve them.

    Uses change in orientation from measured to reference as approximate reference
    angular rate. Compares against measured angular rate. Outputs required change
    in angular rate (angular acceleration) as moments.
    """

    def __init__(
        self, k_p: Gain, k_i: Gain, k_d: Gain, max_err_i: Optional[Gain],
        max_acceleration: float=0.2
    ) -> None:
        """
        Parameters
        ----------
        k_p, k_i, k_d : float or np.ndarray
            Proportional, integral and derivative constants.
        max_err_i : float or np.ndarray, optional
            Maximum accumulated error. None for no limit.
        max_acceleration : float or np.ndarray
            Maximum acceleration in rad/s/s, for all or for each axis.
        """
        # TODO: check max acceleration limits from ardupilot
        super().__init__(k_p, k_i, k_d, max_err_i, n=3)
        self.max_acceleration = max_acceleration
        self._params += ('max_acceleration',)


    def _get_max_acceleration(self) -> Union[float, np.ndarray]:
        "The acceleration limit: a number if it is the same for all axes, else one per axis."
        limit = self._data.params[PID_LIMIT, self._start:self._stop]
        return float(limit[0]) if np.all(limit == limit[0]) else limit.copy()

    def _set_max_acceleration(self, value: Gain) -> None:
        """Set the acceleration limit."""
        self._data.params[PID_LIMIT, self._start:self._stop] = value

    max_acceleration = property(
        _get_max_acceleration, _set_max_acceleration,
        doc="Maximum acceleration in rad/s/s. A number if it is the same for all axes.")


class AltController(PIDStage):
    """
    Altitude Controller. Tracks z-position and outputs velocity.

    Uses change in z-position as approximate vertical velocity. Compares against
    measured velocity. Outputs the change in velocity (acceleration) as thrust force,
    given orientation of vehicle.
    """

    max_velocity = _scalar_param(PID_LIMIT, "Maximum velocity in m/s")

    def __init__(
        self, k_p: Gain, k_i: Gain, k_d: Gain, max_err_i: Optional[Gain],
        max_velocity: float = 5
    ) -> None:
        """
        Parameters
        ----------
        k_p, k_i, k_d : float or np.ndarray
            Proportional, integral and derivative constants.
        max_err_i : float or np.ndarray, optional
            Maximum accumulated error. None for no limit.
        max_velocity : float
            Maximum velocity in m/s
        """
        super().__init__(k_p, 0., 0., max_err_i, n=1) # strictly a P controller
        self.max_velocity = max_velocity
        self._params += ('max_velocity',)


class AltRateController(PIDStage):
    """
    Climb rate controller. Tracks z-velocity and outputs thrust force needed.
    """

    def __init__(self, k_p: Gain, k_i: Gain, k_d: Gain, max_err_i: Optional[Gain]) -> None:
        """
        Parameters
        ----------
        k_p, k_i, k_d : float or np.ndarray
            Proportional, integral and derivative constants.
        max_err_i : float or np.ndarray, optional
            Maximum accumulated error. None for no limit.
        """
        super().__init__(k_p, k_i, k_d, max_err_i, n=1)


class Controller:
    """
    The cascaded PID controller. Tracks position and yaw, and outputs thrust and
    moments needed.

        (x,y) --> Position --> Velocity --> Attitude --> Rate --> (Moments)
        (z)   --> Altitude --> Velocity --> (Forces)
    """

    _STAGE_NAMES = ('ctrl_p', 'ctrl_v', 'ctrl_a', 'ctrl_r', 'ctrl_z', 'ctrl_vz')
    "The attributes that hold the stages, in the order of their columns."

    @property
    def _stages(self) -> tuple:
        "The six stages, in the order of their columns."
        return tuple(getattr(self, name) for name in self._STAGE_NAMES)


    def __init__(
        self,
        ctrl_p: PosController, ctrl_v: VelController,
        ctrl_a: AttController, ctrl_r: RateController,
        ctrl_z: AltController, ctrl_vz: AltRateController,
        vehicle: Multirotor,
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
        vehicle : Multirotor
            The vehicle to control.
        period_[p | a | z] : float, optional
            The period (s) of the [position | attitude | altitude] controller, by
            default 1. The controller computes a new action once per period and
            repeats it in between. The period is rounded to a whole number of
            steps, at least one. For example, a period of 1 s and ``dt=0.1`` give a
            new action every 10 steps.
        feedforward_weight: float, optional
            The weight to assign to a separately specified velocity reference,
            which overrides the velocity signal given by the position controller.
        
        """
        self.ctrl_p = ctrl_p
        self.ctrl_v = ctrl_v
        self.ctrl_a = ctrl_a
        self.ctrl_r = ctrl_r
        self.ctrl_z = ctrl_z
        self.ctrl_vz = ctrl_vz
        self.vehicle = vehicle
        self.dtype = vehicle.dtype
        self.period_p = period_p
        self.period_a = period_a
        self.period_z = period_z
        dt = self.vehicle.simulation.dt
        # Whole simulation steps per period, and at least one: a stage with a
        # period shorter than the simulation step runs at every step.
        self.steps_p = max(1, round(self.period_p / dt))
        self.steps_a = max(1, round(self.period_a / dt))
        self.steps_z = max(1, round(self.period_z / dt))
        self.feedforward_weight = feedforward_weight
        # The stages become views of the columns of one data object
        stages = self._stages
        # A stage moves into this controller's data, so it can belong to only one
        # controller. Check all stages before moving any.
        if len({id(stage) for stage in stages}) < len(stages):
            raise ValueError('Each stage must be a different object.')
        if any(stage._adopted for stage in stages):
            raise ValueError(
                'A stage already belongs to a Controller. Build new stages.')
        self._data = ControllerData(sum(stage.n for stage in stages))
        start = 0
        for stage in stages:
            stage._adopt(self._data, start)
            start += stage.n
        self._fixed_inputs: Optional[CascadeStepInputs] = None
        self._fixed_key: tuple = ()
        self.reset()


    def _inputs(self) -> CascadeStepInputs:
        """
        The inputs of a step that change only when a parameter does, as JAX arrays. They are
        made again when a parameter, a period or an option has changed since the last step,
        so that a step does not convert them every time.
        """
        vehicle = self.vehicle
        key = (
            self.steps_p, self.steps_a, self.steps_z, float(self.feedforward_weight),
            bool(self.ctrl_p.leashing), bool(self.ctrl_p.square_root_scaling),
            bool(self.ctrl_a.square_root_scaling), float(vehicle.params.mass),
            float(vehicle.weight), float(vehicle.simulation.dt))
        if key != self._fixed_key or not np.array_equal(self._data.params, self._fixed_params):
            steps_p, steps_a, steps_z, weight, leashing, sqrt_p, sqrt_a, mass, weight_n, dt = key
            self._fixed_params = self._data.params.copy()
            self._fixed_key = key
            self._fixed_inputs = jax.tree_util.tree_map(jnp.asarray, CascadeStepInputs(
                self._fixed_params, np.zeros(4), np.zeros(12), np.zeros(3),
                np.asarray(vehicle.params.inertia_matrix, dtype=np.float64), mass, weight_n,
                dt, Steps(steps_p, steps_a, steps_z), np.zeros(2), False, weight, leashing,
                sqrt_p, sqrt_a))
        return self._fixed_inputs


    def reset(self) -> np.ndarray:
        """
        Reset the controller's state and its step counter, and those of its stages.

        Returns
        -------
        np.ndarray
            The state of the stages.
        """
        self.action = np.zeros(4, self.vehicle.dtype)
        self.reference = np.zeros_like(self.action)
        # Outputs of the stages that later stages, and the steps in between, reuse
        # (the memo of `cascaded_step`), in float64 like the integrator's state. The
        # public `thrust` and `torques` are in the vehicle's dtype.
        self._ref_vel = np.zeros(2)
        self._thrust = np.zeros(1)
        self._torques = np.zeros(3)
        self.n = 0
        self.t = self.vehicle.t
        for stage in self._stages:
            stage.reset()
        return self.state


    def rollout_params(self) -> ControllerParams:
        """
        The parameters of this controller as the `multirotor.tuples.ControllerParams` that
        `multirotor.rollout` takes, with float64 copies of the arrays. The control allocation
        is the vehicle's, as of this call: it does not follow later changes to the vehicle.

        Returns
        -------
        ControllerParams
            The parameters.
        """
        return ControllerParams(
            np.array(self._data.params, float), np.array(self.vehicle.alloc_inverse, float),
            Steps(self.steps_p, self.steps_a, self.steps_z), float(self.feedforward_weight),
            bool(self.ctrl_p.leashing), bool(self.ctrl_p.square_root_scaling),
            bool(self.ctrl_a.square_root_scaling))


    def rollout_state(self) -> ControllerState:
        """
        The state of this controller as the `multirotor.tuples.ControllerState` that a rollout
        starts from, with float64 copies of the arrays.

        Returns
        -------
        ControllerState
            The state.
        """
        return ControllerState(
            np.array(self._data.state, float),
            Memo(np.array(self._ref_vel, float), np.array(self._thrust, float),
                 np.array(self._torques, float)), int(self.n))


    def set_rollout_state(self, state: ControllerState) -> None:
        """
        Continue from a `multirotor.tuples.ControllerState`, such as the one a rollout returns.
        Set the vehicle's state first: the controller's time is the vehicle's.

        Parameters
        ----------
        state : ControllerState
            The state to continue from.
        """
        self._data.state[:] = np.asarray(state.x, float)
        self._ref_vel, self._thrust, self._torques = (np.array(m, float) for m in state.memo)
        self.n = int(state.n)
        self.t = self.vehicle.t
        self.action = np.concatenate((self._thrust, self._torques)).astype(self.dtype)


    def set_params(self, **params: Union[Dict[str, Any], float]) -> None:
        """
        Set the parameters of the stages, and of the controller.

        Parameters
        ----------
        **params : dict or float
            For a stage (``ctrl_p``, ``ctrl_v``, ``ctrl_a``, ``ctrl_r``, ``ctrl_z`` or
            ``ctrl_vz``), a dict of its parameters, as in `PIDStage.set_params`. For the
            controller, the values of attributes such as ``feedforward_weight``.
        """
        for name in self._STAGE_NAMES:
            getattr(self, name).set_params(**params.get(name, {}))
        local_params = {k:v for k,v in params.items() if not isinstance(v, dict)}
        for name, param in local_params.items():
            if hasattr(self, name):
                # Controller class has no np.array attributes, so no casting/
                # in-place assignment needed like tith PIDController class's
                # set_params() method
                setattr(self, name, param)


    def get_params(self) -> Dict[str, Any]:
        """
        The parameters of the stages and of the controller.

        Returns
        -------
        dict
            The parameters of each stage by its name (``ctrl_p``, ...), as in
            `PIDStage.get_params`, and the ``feedforward_weight``.
        """
        p = {name: getattr(self, name).get_params() for name in self._STAGE_NAMES}
        p['feedforward_weight'] = self.feedforward_weight
        return p


    @property
    def state(self) -> np.ndarray:
        "The state of the six stages, concatenated."
        return np.concatenate([stage.state for stage in self._stages])


    @property
    def thrust(self) -> np.ndarray:
        "The output of the climb stage at the last step."
        return self._thrust.astype(self.dtype)


    @property
    def torques(self) -> np.ndarray:
        "The output of the rate stage at the last step."
        return self._torques.astype(self.dtype)


    @property
    def leash(self) -> float:
        "The largest horizontal position error that is acted on, given the vehicle's motion."
        if not self.ctrl_p.leashing:
            return np.inf
        return float(_leash_distance(
            float(self.ctrl_p.k_p[0]), _state64(self.vehicle),
            _vector(self.vehicle.acceleration), self.vehicle.simulation.dt))


    def step(
        self, reference: np.ndarray, measurement: Optional[np.ndarray] = None,
        ref_is_error: bool = False, feed_forward_velocity: Optional[np.ndarray] = None,
        persist: bool = True
    ) -> np.ndarray:
        """
        Calculate the controller's action for the vehicle's current state.

        Parameters
        ----------
        reference : np.ndarray
            The target: x, y, z and yaw, shape (4,).
        measurement : np.ndarray, optional
            Ignored. The controller reads the state of its vehicle.
        ref_is_error : bool
            Whether ``reference`` is the error (target minus the vehicle's position and yaw)
            instead of the target.
        feed_forward_velocity : np.ndarray, optional
            An inertial xy-velocity reference, blended into the position stage's output
            with ``feedforward_weight``.
        persist : bool
            Whether to keep the new state for the next step. If False the controller is
            left as it was, apart from the outputs reported by `thrust` and `torques`.

        Returns
        -------
        np.ndarray
            The thrust and the three torques, shape (4,). Pass them to
            `Multirotor.allocate_control`.
        """
        vehicle = self.vehicle
        if ref_is_error:
            error = _vector(reference)
            target = np.empty(4)
            target[:2] = vehicle.position[:2] + error[:2]
            target[2] = vehicle.position[2] + error[2]
            target[3] = vehicle.orientation[2] + error[3]
        else:
            self.reference = reference
            target = _vector(reference)
        use_feed_forward = feed_forward_velocity is not None
        feed_forward = _vector(feed_forward_velocity)[:2] if use_feed_forward \
            else np.zeros(2)
        inputs = np.concatenate((
            self._data.state.ravel(), self._ref_vel, self._thrust, self._torques, target,
            vehicle.state, vehicle.acceleration, feed_forward, (use_feed_forward, self.n)),
            dtype=float)
        out = np.asarray(_compiled_cascaded_step(inputs, self._inputs()))
        size = self._data.state.size
        action = out[size + 6:].astype(self.dtype)
        # The outputs of the stages are kept even if the state is not, as `thrust` and
        # `torques` report the last step that ran, probe or not.
        self._ref_vel, self._thrust, self._torques = (
            out[size:size + 2], out[size + 2:size + 3], out[size + 3:size + 6])
        if persist:
            self._data.state[:] = out[:size].reshape(self._data.state.shape)
            self.n += 1
            self.t = vehicle.t
            self.action = action
        return action


    def predict(self, ref: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        The action for a target, as a policy of a reinforcement learning environment.

        Parameters
        ----------
        ref : np.ndarray
            The target: x, y, z and yaw, shape (4,).
        deterministic : bool
            Ignored: the controller is deterministic.

        Returns
        -------
        np.ndarray
            The thrust and the three torques, shape (4,).
        """
        return self.step(reference=ref)
