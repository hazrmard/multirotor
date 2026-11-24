"""
A fleet of vehicles simulated together. `Fleet` holds the parameters and the state
of ``K`` vehicles as arrays with the vehicle as the first axis, and steps through all
vehicles together. It is a peer of `multirotor.simulation.Multirotor`, not a
collection of `Multirotor` instances. It exposes the same core API as `Multirotor`:
`Fleet.step_speeds()`, `Fleet.step_dynamics()` and `Fleet.allocate_control()`, with
one row per vehicle in every input and output. It has no objects for single
propellers, motors or batteries; their state is in arrays.

Compared with ``K`` separate `Multirotor` vehicles, a `Fleet`:

* advances all vehicles with one compiled call, which maps the step of a single
  vehicle over the fleet (``jax.vmap``), so the cost of the call is shared.
* needs vehicles of a homogeneous configuration: the same number of propellers,
  motors on all or on none, and a battery model on all or on none. Masses,
  inertia, propeller, motor and battery parameter values can differ.
* matches separate vehicles to rounding: both use the same fixed-step integrator.
* supports propellers with ``k_thrust * speed**2`` thrust. Blade-element thrust
  needs a numerical solution at every evaluation of the derivative and is not
  supported.
"""
from functools import partial
from typing import List, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from ._compile import step_function
from .coords import angular_to_euler_rate, direction_cosine_matrix
from .dynamics import N_BODY_STATES
from .simulation import (
    Multirotor, _step_body, _step_forces, _step_motors, _step_motors_battery)
from .tuples import BatteryVehicle, BodyVehicle, MotorsVehicle
from .vehicle import SimulationParams, VehicleParams


# The steps of `multirotor.simulation`, mapped over the vehicles. The inputs and the
# vehicle's parameters have the vehicle as their first axis, and so has the result.

Vehicle = Union[BodyVehicle, MotorsVehicle, BatteryVehicle]


@partial(step_function, static_argnames=('substeps',))
def _fleet_step_forces(inputs: jax.Array, vehicle: Vehicle, dt: float, substeps: int) -> jax.Array:
    """`multirotor.simulation._step_forces` for every vehicle."""
    return jax.vmap(lambda i, v: _step_forces.plain(i, v, dt, substeps))(inputs, vehicle)


@partial(step_function, static_argnames=('substeps',))
def _fleet_step_body(inputs: jax.Array, vehicle: Vehicle, dt: float, substeps: int) -> jax.Array:
    """`multirotor.simulation._step_body` for every vehicle."""
    return jax.vmap(lambda i, v: _step_body.plain(i, v, dt, substeps))(inputs, vehicle)


@partial(step_function, static_argnames=('substeps',))
def _fleet_step_motors(inputs: jax.Array, vehicle: Vehicle, dt: float, substeps: int) -> jax.Array:
    """`multirotor.simulation._step_motors` for every vehicle."""
    return jax.vmap(lambda i, v: _step_motors.plain(i, v, dt, substeps))(inputs, vehicle)


@partial(step_function, static_argnames=('substeps',))
def _fleet_step_motors_battery(inputs: jax.Array, vehicle: Vehicle, dt: float, substeps: int) -> jax.Array:
    """`multirotor.simulation._step_motors_battery` for every vehicle."""
    return jax.vmap(lambda i, v: _step_motors_battery.plain(i, v, dt, substeps))(inputs, vehicle)



class Fleet:
    """
    ``K`` multirotors simulated together. It exposes the same core API as
    `Multirotor`: `step_speeds()`, `step_dynamics()` and `allocate_control()`.
    Every state quantity and every input has one row per vehicle.

    The results match those of separate `Multirotor` vehicles to rounding. The vehicles
    must have a homogeneous configuration
    (the same propeller, motor and battery setup), but parameter values can vary.
    """

    def __init__(
        self, params: Sequence[VehicleParams], simulation: SimulationParams,
        substeps: Optional[int] = None
    ) -> None:
        """
        Parameters
        ----------
        params : Sequence[VehicleParams]
            The parameters of each vehicle. `Fleet` copies them at construction.
            Changing them afterwards does not affect the fleet.
        simulation : SimulationParams
            The simulation parameters, shared by all vehicles.
        substeps : int, optional
            RK4 steps per simulation time step, as for `Multirotor`.

        Raises
        ------
        ValueError
            If there are no vehicles, or their configurations are not homogeneous.
        NotImplementedError
            If the propellers use the blade-element thrust model.
        """
        params = list(params)
        if len(params) == 0:
            raise ValueError('A fleet needs at least one vehicle.')
        self.simulation: SimulationParams = simulation
        # The derived values (kernel parameters, control allocation, battery
        # tables) come from `Multirotor`, which defines them. Only their arrays
        # are kept.
        vehicles = [Multirotor(p, simulation, substeps) for p in params]
        first = vehicles[0]
        self.substeps: int = first.substeps
        self.params: List[VehicleParams] = [v.params for v in vehicles]
        self.dtype = first.dtype
        self._check_alike(vehicles)

        n = len(first.propellers)
        self.n_propellers: int = n
        self._motors: bool = first._has_motors
        self._battery: bool = first.battery.has_model
        kernel = [v._vehicle for v in vehicles]
        # The parameters of all vehicles as one tuple, each field with the vehicle as its
        # first axis: what the compiled steps map over
        self._vehicle = type(kernel[0])(*(jnp.asarray(np.stack(field), float) for field in zip(*kernel)))
        self._dt = jnp.asarray(float(simulation.dt))
        self._g = float(simulation.g)
        self._mass = np.asarray([k.mass for k in kernel])
        self._inertia = np.stack([k.inertia for k in kernel])
        self._inertia_inverse = np.stack([k.inertia_inverse for k in kernel])
        self._prop_params = np.stack([k.prop_params for k in kernel])
        self._motor_params = np.stack([k.motor_params for k in kernel]) \
            if self._motors else None
        self._clockwise = np.stack([k.clockwise for k in kernel])
        self._positions = np.stack([k.positions for k in kernel])
        self._alloc_inverse = np.stack(
            [v.alloc_inverse.astype(float) for v in vehicles])
        if self._battery:
            self._battery_params = np.stack([v.battery._params_vector for v in vehicles])
            self._ocv_soc = np.stack([v.battery._ocv_soc for v in vehicles]).astype(float)
            self._ocv_voltage = np.stack(
                [v.battery._ocv_voltage for v in vehicles]).astype(float)
        # The battery's open-circuit voltage when full: the motors' voltage limit
        # at the start, and the limit without a battery model
        self._voltage_full = np.asarray([v.battery.voltage for v in vehicles], float)
        self._max_voltage = np.asarray(
            [v.battery.params.max_voltage for v in vehicles], float)

        # The integrated state, one row per vehicle: the 12 body states, then
        # with motors their speeds. The state is float64 whatever the dtype, which only
        # sets the precision that is kept: see `step_speeds`.
        self._width = N_BODY_STATES + (n if self._motors else 0)
        self._x = np.zeros((len(vehicles), self._width))
        # The body states, float64 whatever the dtype. The public properties
        # return copies of them in the dtype of the simulation.
        self._state = self._x[:, :N_BODY_STATES]
        if self._motors:
            self._speed = self._x[:, N_BODY_STATES:N_BODY_STATES + n]
            self._voltage = np.zeros((len(vehicles), n))
            self._current = np.zeros((len(vehicles), n))
        else:
            self._speed = np.zeros((len(vehicles), n))
        self._dxdt = np.zeros((len(vehicles), N_BODY_STATES))
        self._battery_charge_drawn = np.zeros(len(vehicles))
        self._battery_current = np.zeros(len(vehicles))
        self._battery_voltage = np.zeros(len(vehicles))
        self._battery_cutoff = np.zeros(len(vehicles), dtype=np.bool_)
        self._battery_overload = np.zeros(len(vehicles), dtype=np.bool_)
        self._motor_voltage_limit = np.zeros(len(vehicles))
        self.t: float = 0.
        self.reset()


    def _check_alike(self, vehicles: List[Multirotor]) -> None:
        """Raise if the vehicles are not of one kind."""
        first = vehicles[0]
        kinds = {
            'number of propellers': lambda v: len(v.propellers),
            'motors': lambda v: v._has_motors,
            'thrust model': lambda v: v._use_thrust_constant,
            'battery model': lambda v: v.battery.has_model,
            'dtype': lambda v: np.dtype(v.dtype),
        }
        if first.battery.has_model:
            kinds['battery table length'] = lambda v: len(v.battery._ocv_soc)
        for name, kind in kinds.items():
            if len({kind(v) for v in vehicles}) > 1:
                raise ValueError(f'All vehicles in a fleet must have the same {name}.')
        if not first._use_thrust_constant:
            raise NotImplementedError(
                'A Fleet needs quadratic thrust (use_thrust_constant=True). '
                'The blade-element model is not supported.')


    def __len__(self) -> int:
        "The number of vehicles."
        return self._x.shape[0]


    def reset(self, state: np.ndarray=None) -> np.ndarray:
        """
        Reset the state of the fleet: the vehicles' states, propellers, motors
        and batteries, and the time. All in place.

        Parameters
        ----------
        state : np.ndarray, optional
            The vehicles' body states, shape (K, 12), or (12,) for all vehicles.
            By default every vehicle starts at rest at the origin.

        Returns
        -------
        np.ndarray
            The state of the fleet, shape (K, 12).
        """
        self.t = 0.
        self._x[:] = 0.
        self._speed[:] = 0.
        if self._motors:
            self._voltage[:] = 0.
            self._current[:] = 0.
        self._dxdt[:] = 0.
        self._battery_charge_drawn[:] = 0.
        self._battery_current[:] = 0.
        self._battery_voltage[:] = self._voltage_full
        self._battery_cutoff[:] = False
        self._battery_overload[:] = False
        self._motor_voltage_limit[:] = self._voltage_full
        if state is not None:
            self.state = state
        return self.state


    @property
    def state(self) -> np.ndarray:
        """
        The body states, shape (K, 12): position ``[0:3]``, body velocity
        ``[3:6]``, orientation ``[6:9]``, body angular rate ``[9:12]``. A copy,
        in the dtype of the simulation: set the state with ``fleet.state = ...``.
        """
        return self._state.astype(self.dtype)
    @state.setter
    def state(self, state: np.ndarray):
        self._state[:] = np.asarray(state, dtype=self.dtype)


    @property
    def position(self) -> np.ndarray:
        """Positions in the inertial frame, shape (K, 3). A copy."""
        return self._state[:, 0:3].astype(self.dtype)


    @property
    def velocity(self) -> np.ndarray:
        """Body-frame velocities, shape (K, 3). A copy."""
        return self._state[:, 3:6].astype(self.dtype)


    @property
    def inertial_velocity(self) -> np.ndarray:
        """Velocities in the inertial frame, shape (K, 3)."""
        # One direction cosine matrix per vehicle, on the last axis, shape (3, 3, K)
        dcm = direction_cosine_matrix(*self._state[:, 6:9].T)
        return np.einsum('jik,kj->ki', dcm, self._state[:, 3:6]).astype(self.dtype)


    @property
    def acceleration(self) -> np.ndarray:
        """Body-frame accelerations at the start of the last step, shape (K, 3). A copy."""
        return self._dxdt[:, 3:6].astype(self.dtype)


    @property
    def orientation(self) -> np.ndarray:
        """Euler rotations (roll, pitch, yaw), shape (K, 3). A copy."""
        return self._state[:, 6:9].astype(self.dtype)


    @property
    def angular_rate(self) -> np.ndarray:
        """Angular rate of the body frame axes, shape (K, 3). A copy."""
        return self._state[:, 9:12].astype(self.dtype)


    @property
    def euler_rate(self) -> np.ndarray:
        """Rate of change of roll, pitch and yaw, shape (K, 3)."""
        return angular_to_euler_rate(
            self._state[:, 9:12].T, self._state[:, 6:9].T).T.astype(self.dtype)


    @property
    def weight(self) -> np.ndarray:
        """Weight of each vehicle, shape (K,)."""
        return self._g * self._mass


    @property
    def speeds(self) -> np.ndarray:
        """Propeller speeds (rad/s), shape (K, n). A copy."""
        return np.array(self._speed, self.dtype)


    @property
    def motor_currents(self) -> np.ndarray:
        """
        Current (A) drawn by each motor, shape (K, n), on the motor side of its
        speed controller. None if the vehicles have no motors. A copy.
        """
        return self._current.copy() if self._motors else None


    @property
    def soc(self) -> np.ndarray:
        """State of charge of each battery, from 0 (empty) to 1 (full), shape (K,)."""
        if not self._battery:
            return np.ones(len(self))
        return 1. - self._battery_charge_drawn / self._battery_params[:, 0]


    @property
    def battery_voltage(self) -> np.ndarray:
        """Terminal voltage (V) of each battery, shape (K,)."""
        return self._battery_voltage.copy()


    @property
    def battery_cutoff(self) -> np.ndarray:
        """Whether each battery has cut off the motors, shape (K,)."""
        return self._battery_cutoff.copy()


    @property
    def battery_current(self) -> np.ndarray:
        """
        Current (A) drawn from each battery, shape (K,). As for
        `Multirotor.battery_current`: with a battery model, the average over the
        last step. Without one, the motors' power divided by the maximum voltage,
        or NaN if the vehicles have no motors or the voltage is infinite.
        """
        if self._battery:
            return self._battery_current.copy()
        if not self._motors:
            return np.full(len(self), np.nan)
        power = np.sum(self._current * self._voltage, axis=1)
        return np.divide(
            power, self._max_voltage, out=np.full(len(self), np.nan),
            where=np.isfinite(self._max_voltage))


    def _signals(self, u: np.ndarray) -> np.ndarray:
        """
        Speed signals as a float64 array of shape (K, n).

        Returns: The signals, contiguous, of shape (K, n).
        """
        u = np.asarray(u, dtype=float)
        if u.shape != (len(self), self.n_propellers):
            raise ValueError(
                f'Expected speed signals of shape {(len(self), self.n_propellers)}, '
                f'got {u.shape}.')
        return np.ascontiguousarray(u)


    def _disturbance(
        self, forces: Union[float, np.ndarray], torques: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Disturbing forces and torques as an array of shape (K, 6).

        ``forces``, ``torques``: The disturbing forces and torques, shape (3,) or (K, 3), or a
          number for all.
        Returns: The disturbances of every vehicle, shape (K, 6).
        """
        disturbance = np.zeros((len(self), 6))
        disturbance[:, :3] += forces
        disturbance[:, 3:] += torques
        return disturbance





    def _round_state(self) -> None:
        "Keep the body states at the precision of the simulation's dtype."
        if self.dtype != np.float64:
            self._state[:] = self._state.astype(self.dtype)


    def step_dynamics(self, u: np.ndarray) -> np.ndarray:
        """
        Given the 6-vector of x,y,z-forces and roll,pitch,yaw-torques of each
        vehicle, calculate the next state of the fleet.

        Parameters
        ----------
        u : np.ndarray
            The 6-vectors, shape (K, 6): forces (N) and torques (N m).

        Returns
        -------
        np.ndarray
            The new states, shape (K, 12).
        """
        self.t += self.simulation.dt
        inputs = np.concatenate((self._state, u), axis=1, dtype=float)
        out = np.asarray(_fleet_step_forces(
            inputs, self._vehicle, self._dt, substeps=self.substeps))
        self._state[:] = out[:, :N_BODY_STATES]
        self._dxdt[:] = out[:, N_BODY_STATES:]
        self._round_state()
        return self.state


    def step_speeds(
        self, u: np.ndarray, disturb_forces: np.ndarray=0.,
        disturb_torques: np.ndarray=0.
    ) -> np.ndarray:
        """
        Calculate the next state of the fleet from the speed signals of the
        propellers of each vehicle.

        With motors, the body and the motors are integrated together, as in
        `Multirotor.step_speeds`.

        With a dtype other than float64, the body states are rounded to that
        dtype after each step. The forces are not.

        Parameters
        ----------
        u : np.ndarray
            The speed signals, shape (K, n). A signal is a speed (rad/s), or a
            voltage (V) when the propellers have a motor and
            ``MotorParams.speed_voltage_scaling`` is set.
        disturb_forces : np.ndarray, optional
            Disturbing x,y,z forces in each vehicle's local frame, by default 0.
            A number, or an array that broadcasts to shape (K, 3).
        disturb_torques : np.ndarray, optional
            Disturbing x,y,z torques in each vehicle's local frame, by default 0.

        Returns
        -------
        np.ndarray
            The new states, shape (K, 12).
        """
        u = self._signals(u)
        disturbance = self._disturbance(disturb_forces, disturb_torques)
        self.t += self.simulation.dt
        if self._motors:
            self._step_with_motors(u, disturbance)
        else:
            self._step_with_fixed_forces(u, disturbance)
        self._round_state()
        return self.state


    def _step_with_fixed_forces(self, u: np.ndarray, disturbance: np.ndarray) -> None:
        """
        Advance vehicles without motors by one time step. Forces and torques
        depend only on the speed signals, which are fixed over the step, so
        they are calculated once.

        ``u``: The speed signals, shape (K, n).
        ``disturbance``: The disturbing forces and torques, shape (K, 6).
        """
        inputs = np.concatenate((self._state, u, disturbance), axis=1)
        out = np.asarray(_fleet_step_body(
            inputs, self._vehicle, self._dt, substeps=self.substeps))
        self._state[:] = out[:, :N_BODY_STATES]
        self._dxdt[:] = out[:, N_BODY_STATES:]
        self._speed[:] = u


    def _step_with_motors(self, u: np.ndarray, disturbance: np.ndarray) -> None:
        """
        Advance the bodies, the motors and the batteries by one time step,
        integrating them together.

        ``u``: The speed signals, shape (K, n).
        ``disturbance``: The disturbing forces and torques, shape (K, 6).
        """
        n, width = self.n_propellers, self._width
        if self._battery:
            inputs = np.concatenate((
                self._x, u, disturbance, self._battery_charge_drawn[:, None],
                self._battery_cutoff[:, None], self._battery_overload[:, None]), axis=1,
                dtype=float)
            out = np.asarray(_fleet_step_motors_battery(
                inputs, self._vehicle, self._dt, substeps=self.substeps))
            electrical = width + N_BODY_STATES + 6
            (self._battery_charge_drawn[:], self._battery_current[:],
             self._battery_voltage[:], cutoff, overload, self._motor_voltage_limit[:]) = \
                out[:, width + N_BODY_STATES:electrical].T
            self._battery_cutoff[:], self._battery_overload[:] = cutoff > .5, overload > .5
        else:
            inputs = np.concatenate(
                (self._x, u, disturbance, self._motor_voltage_limit[:, None]), axis=1)
            out = np.asarray(_fleet_step_motors(
                inputs, self._vehicle, self._dt, substeps=self.substeps))
            electrical = width + N_BODY_STATES
        self._x[:] = out[:, :width]
        self._dxdt[:] = out[:, width:width + N_BODY_STATES]
        self._voltage[:] = out[:, electrical:electrical + n]
        self._current[:] = out[:, electrical + n:]


    def allocate_control(self, thrust: np.ndarray, torques: np.ndarray) -> np.ndarray:
        """
        Allocate control to the propellers of each vehicle by converting the
        prescribed forces and torques into propeller speeds.

        Parameters
        ----------
        thrust : np.ndarray
            The thrust in each body's z-direction, shape (K,).
        torques : np.ndarray
            The roll, pitch, yaw torques required about (x, y, z) axes, shape (K, 3).

        Returns
        -------
        np.ndarray
            The prescribed propeller speeds (rad/s), shape (K, n).
        """
        vec = np.concatenate(
            (np.reshape(thrust, (-1, 1)), np.reshape(torques, (-1, 3))), axis=1
        ).astype(self.dtype).astype(float)
        speeds_squared = np.einsum('knj,kj->kn', self._alloc_inverse, vec)
        return np.sqrt(np.clip(speeds_squared, a_min=0., a_max=None))
