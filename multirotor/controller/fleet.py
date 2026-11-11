"""
The controllers of a `multirotor.fleet.Fleet`. `FleetController` holds the
parameters and the state of the cascaded controller of each of ``K`` vehicles as
arrays with the vehicle as the first axis, and steps all of them with one compiled
loop over the vehicles (`multirotor.controller.cascade.fleet_step`).

It is the `Controller` of every vehicle at once. The controllers of the vehicles
have the same structure: the same periods and options. Their gains and limits can
differ from vehicle to vehicle.

The arrays follow the layout of `multirotor.controller.data.ControllerData`, with
the vehicle in front. A vehicle's slice, such as ``params[k]``, is the matrix of
one `Controller`: one column per control channel of its six stages. The ``PID_*``
and ``CASCADE_*`` constants in `multirotor.controller.data` name the rows and
the first column of each stage.
"""
from copy import deepcopy
from typing import Any, Dict, Sequence, Union

import numpy as np

from ..fleet import Fleet
from ..tuples import Memo
from .cascade import fleet_step
from .data import N_CASCADE_CHANNELS, N_PID_STATES
from .pid import Controller



class FleetController:
    """
    A batch of cascaded PID controllers, one for each vehicle in a `Fleet`. The
    controllers have their own gains (i.e. PID parameters), but must have an
    identical structure and properties (periods, ``leashing``, ``square_root_scaling``).

    Call `step` between the fleet's steps, as one calls `Controller.step` between
    the steps of a vehicle.
    """

    def __init__(
        self, fleet: Fleet, controllers: Union[Controller, Sequence[Controller]]
    ) -> None:
        """
        Parameters
        ----------
        fleet : Fleet
            The vehicles to control.
        controllers : Controller or Sequence[Controller]
            The controller of each vehicle, or one controller for all of them.
            `FleetController` copies their parameters at construction: changing
            a controller afterwards does not affect the fleet's controllers, and
            a controller is not changed by the fleet. Each must have been built
            on a vehicle with the simulation time step of the fleet. They must
            have the same periods and options (``leashing``, ``square_root_scaling``),
            but can have different gains.

        Raises
        ------
        ValueError
            If the controllers are not as many as the vehicles, or differ in
            structure.
        """
        self.fleet: Fleet = fleet
        k_vehicles = len(fleet)
        if isinstance(controllers, Controller):
            controllers = [controllers] * k_vehicles
        controllers = list(controllers)
        if len(controllers) != k_vehicles:
            raise ValueError(
                f'Expected {k_vehicles} controllers, one per vehicle, got {len(controllers)}.')
        first = controllers[0]
        for controller in controllers:
            if controller.vehicle.simulation.dt != fleet.simulation.dt:
                raise ValueError(
                    'A controller was built on a vehicle with another time step than the fleet.')
            if self._structure(controller) != self._structure(first):
                raise ValueError(
                    'All controllers must have the same periods and options '
                    '(leashing and square_root_scaling).')
        (self.steps_p, self.steps_a, self.steps_z,
         self.leashing, self.square_root_position, self.square_root_attitude) = \
            self._structure(first)
        # A pristine copy. The parameter getter and setter work on a fresh copy of
        # it for each call, so that they accept and return what `Controller` does,
        # and a setting that is not in the arrays does not carry over to other vehicles.
        self._template: Controller = self._copy(first)

        self.params: np.ndarray = np.stack(
            [c._data.params for c in controllers]).astype(float)
        "Parameters, shape (K, N_PID_PARAMS, 12)."
        self.state: np.ndarray = np.zeros((k_vehicles, N_PID_STATES, N_CASCADE_CHANNELS))
        "State of the stages, shape (K, N_PID_STATES, 12)."
        self.feedforward_weight: np.ndarray = np.asarray(
            [c.feedforward_weight for c in controllers], float)
        "The weight of the feed-forward velocity in the position stage's output, shape (K,)."
        self.reset()


    @staticmethod
    def _structure(controller: Controller) -> tuple:
        "What must be the same for all controllers of a fleet."
        return (
            controller.steps_p, controller.steps_a, controller.steps_z,
            bool(controller.ctrl_p.leashing),
            bool(controller.ctrl_p.square_root_scaling),
            bool(controller.ctrl_a.square_root_scaling),
        )


    @staticmethod
    def _copy(controller: Controller) -> Controller:
        """
        A copy of a controller that shares its vehicle, which is not part of its gains.

        Parameters
        ----------
        controller : Controller
            The controller to copy.

        Returns
        -------
        Controller
            The copy.
        """
        return deepcopy(controller, {id(controller.vehicle): controller.vehicle})


    def _controller_of(self, index: int) -> Controller:
        """
        A new `Controller` with the parameters of one vehicle's controller.

        Parameters
        ----------
        index : int
            The index of the vehicle in the fleet.

        Returns
        -------
        Controller
            The new controller.
        """
        controller = self._copy(self._template)
        controller._data.params[:] = self.params[index]
        controller.feedforward_weight = self.feedforward_weight[index]
        return controller


    def reset(self) -> None:
        "Reset the state of every controller, and the step counter."
        k_vehicles = len(self.fleet)
        self.state[:] = 0.
        self.action = np.zeros((k_vehicles, 4), self.fleet.dtype)
        self.reference = np.zeros((k_vehicles, 4), self.fleet.dtype)
        # The memo of `fleet_step`, in float64 like the integrator's state. The
        # public `thrust` and `torques` are in the fleet's dtype.
        self._ref_vel = np.zeros((k_vehicles, 2))
        self._thrust = np.zeros((k_vehicles, 1))
        self._torques = np.zeros((k_vehicles, 3))
        self.n = 0
        self.t = self.fleet.t


    @property
    def thrust(self) -> np.ndarray:
        "The thrust of each vehicle's climb stage at the last step, shape (K, 1)."
        return self._thrust.astype(self.fleet.dtype)


    @property
    def torques(self) -> np.ndarray:
        "The torques of each vehicle's rate stage at the last step, shape (K, 3)."
        return self._torques.astype(self.fleet.dtype)


    def set_params(self, index: int, **params: Any) -> None:
        """
        Set the gains (i.e. the PID parameters) of one vehicle's controller, as
        `Controller.set_params` does for a controller.

        Parameters
        ----------
        index : int
            The index of the vehicle, and so of its controller, in the fleet.
        **params
            The gains, for example ``ctrl_r=dict(k_p=[20, 20, 0])`` for the
            rate stage.

        Raises
        ------
        ValueError
            If the parameters change the periods or options, which are shared by
            all vehicles.
        """
        scratch = self._controller_of(index)
        scratch.set_params(**params)
        if self._structure(scratch) != (
            self.steps_p, self.steps_a, self.steps_z, self.leashing,
            self.square_root_position, self.square_root_attitude
        ):
            raise ValueError(
                'The periods and options (leashing, square_root_scaling) are the '
                'same for all vehicles of a fleet.')
        self.params[index] = scratch._data.params
        self.feedforward_weight[index] = scratch.feedforward_weight


    def get_params(self, index: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        The gains (i.e. the PID parameters) of one vehicle's controller, as
        `Controller.get_params` returns them.

        Parameters
        ----------
        index : int
            The index of the vehicle, and so of its controller, in the fleet.

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            Copies of the parameters.
        """
        return self._controller_of(index).get_params()


    def step(
        self, reference: np.ndarray, ref_is_error: bool=False,
        feed_forward_velocity: np.ndarray=None, persist: bool=True
    ) -> np.ndarray:
        """
        Calculate the thrust and torques that each vehicle needs, as
        `Controller.step` does for one vehicle.

        Parameters
        ----------
        reference : np.ndarray
            The target of each vehicle: x, y, z and yaw, shape (K, 4). A target
            of shape (4,) is the target of every vehicle.
        ref_is_error : bool, optional
            Whether the reference is the error, relative to each vehicle's
            position and yaw.
        feed_forward_velocity : np.ndarray, optional
            A velocity reference for each vehicle, shape (K, 2) or (2,), blended into the
            position stage's output with ``feedforward_weight``.
        persist : bool, optional
            Whether to store the controllers' state for the next step.

        Returns
        -------
        np.ndarray
            The thrust and the three torques of each vehicle, shape (K, 4). Pass
            them to `Fleet.allocate_control`.
        """
        fleet = self.fleet
        k_vehicles = len(fleet)
        reference = np.broadcast_to(np.asarray(reference, dtype=float), (k_vehicles, 4))
        if ref_is_error:
            target = np.empty((k_vehicles, 4))
            target[:, :3] = fleet._state[:, :3] + reference[:, :3]
            target[:, 3] = fleet._state[:, 8] + reference[:, 3]
        else:
            target = np.ascontiguousarray(reference)
            self.reference = target.astype(fleet.dtype)
        use_feed_forward = feed_forward_velocity is not None
        feed_forward = np.ascontiguousarray(np.broadcast_to(
            np.asarray(feed_forward_velocity, dtype=float)[..., :2], (k_vehicles, 2))) \
            if use_feed_forward else np.zeros((k_vehicles, 2))
        state, memo, action = fleet_step(
            self.params, self.state, Memo(self._ref_vel, self._thrust, self._torques), target,
            np.ascontiguousarray(fleet._state), np.ascontiguousarray(fleet._dxdt[:, 3:6]),
            fleet._inertia, fleet._mass, fleet.weight, float(fleet.simulation.dt), self.n,
            self.steps_p, self.steps_a, self.steps_z, feed_forward, use_feed_forward,
            self.feedforward_weight, self.leashing, self.square_root_position,
            self.square_root_attitude)
        action = action.astype(fleet.dtype)
        # The outputs of the stages are kept even if the state is not, as `thrust` and
        # `torques` report the last step that ran, probe or not.
        self._ref_vel, self._thrust, self._torques = memo
        if persist:
            self.state[:] = state
            self.n += 1
            self.t = fleet.t
            self.action = action
        return action
