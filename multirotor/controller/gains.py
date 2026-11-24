"""
The gains and limits of the cascaded controller as named, differentiable parameters, and the
function that turns them into the arrays a rollout takes.

`CascadeGains` holds the gains of the six stages (`PosGains`, `VelGains`, `AttGains`,
`RateGains`, `AltGains`, `ClimbGains`), one dataclass for each of the stage classes of
`multirotor.controller.pid`, with the same names. Like the vehicle's parameters they are
JAX pytrees: the numbers are the leaves, and the options (``leashing``, the square-root
scaling and the stages' periods) are static. `build_controller` turns them into
`multirotor.tuples.ControllerParams` with JAX operations, so a loss can be differentiated
with respect to a gain, a limit or the arm length that sets the control allocation.

`multirotor.controller.Controller.gains` reads the gains of a controller, and
`Controller.from_gains` makes a controller from them.
"""
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
from jax.tree_util import register_dataclass

from ..helpers import control_allocation_matrix
from ..tuples import ControllerParams, Steps
from ..vehicle import SimulationParams, VehicleParams, xp_of
from . import data as K

Gain = Union[float, np.ndarray]
"A number for all of a stage's channels, or one number for each."



@dataclass
class PosGains:
    "The position stage: two channels, x and y. See `multirotor.controller.PosController`."
    k_p: Gain = 0.8
    k_i: Gain = 0.
    k_d: Gain = 3.75
    max_err_i: Optional[Gain] = None
    max_velocity: float = 7.
    max_acceleration: float = 3.
    max_jerk: float = 100.
    square_root_scaling: bool = True
    leashing: bool = True
    "With either option, ``k_p`` is ``0.5 * max_jerk / max_acceleration``."

register_dataclass(
    PosGains, data_fields=['k_p', 'k_i', 'k_d', 'max_err_i', 'max_velocity', 'max_acceleration',
                           'max_jerk'],
    meta_fields=['square_root_scaling', 'leashing'])



@dataclass
class VelGains:
    "The velocity stage: two channels. See `multirotor.controller.VelController`."
    k_p: Gain = 0.5
    k_i: Gain = 0.
    k_d: Gain = 0.
    max_err_i: Optional[Gain] = None
    max_tilt: float = np.pi / 18

register_dataclass(
    VelGains, data_fields=['k_p', 'k_i', 'k_d', 'max_err_i', 'max_tilt'], meta_fields=[])



@dataclass
class AttGains:
    "The attitude stage: three channels, roll, pitch and yaw. See `AttController`."
    k_p: Gain = 8.
    k_i: Gain = 0.
    k_d: Gain = 0.
    max_err_i: Optional[Gain] = None
    max_acceleration: float = 0.2
    max_jerk: float = 100.
    square_root_scaling: bool = False
    "With it, ``k_p`` is ``max_jerk / max_acceleration``, for all three channels."

register_dataclass(
    AttGains, data_fields=['k_p', 'k_i', 'k_d', 'max_err_i', 'max_acceleration', 'max_jerk'],
    meta_fields=['square_root_scaling'])



@dataclass
class RateGains:
    "The rate stage: three channels. See `multirotor.controller.RateController`."
    k_p: Gain = 32.
    k_i: Gain = 0.
    k_d: Gain = 0.
    max_err_i: Optional[Gain] = None
    max_acceleration: Gain = 0.2

register_dataclass(
    RateGains, data_fields=['k_p', 'k_i', 'k_d', 'max_err_i', 'max_acceleration'],
    meta_fields=[])



@dataclass
class AltGains:
    "The altitude stage: one channel, proportional only. See `AltController`."
    k_p: Gain = 1.
    max_err_i: Optional[Gain] = None
    max_velocity: float = 5.

register_dataclass(AltGains, data_fields=['k_p', 'max_err_i', 'max_velocity'], meta_fields=[])



@dataclass
class ClimbGains:
    "The climb-rate stage: one channel. See `multirotor.controller.AltRateController`."
    k_p: Gain = 10.
    k_i: Gain = 0.
    k_d: Gain = 0.
    max_err_i: Optional[Gain] = None

register_dataclass(ClimbGains, data_fields=['k_p', 'k_i', 'k_d', 'max_err_i'], meta_fields=[])



@dataclass
class CascadeGains:
    """
    The gains of the whole cascade, and when its stages run.

    ``period_p``, ``period_a`` and ``period_z`` (s) are those of the position stage, the
    attitude group (velocity, attitude and rate stages) and the altitude group. They
    are static. A stage computes a new action once per period, which is rounded to a whole
    number of simulation steps, at least one. ``feedforward_weight`` weights a separately
    given velocity reference.
    """
    position: PosGains
    velocity: VelGains
    attitude: AttGains
    rate: RateGains
    altitude: AltGains
    climb: ClimbGains
    feedforward_weight: float = 0.
    period_p: float = 0.1
    period_a: float = 0.01
    period_z: float = 0.1

register_dataclass(
    CascadeGains,
    data_fields=['position', 'velocity', 'attitude', 'rate', 'altitude', 'climb',
                 'feedforward_weight'],
    meta_fields=['period_p', 'period_a', 'period_z'])



def _columns(
    xp, n: int, k_p: Gain, k_i: Gain, k_d: Gain, max_err_i: Optional[Gain],
    limit: Any = 0., max_acceleration: Any = 0.
):
    """The parameter matrix of one stage, shape (N_PID_PARAMS, n): one column per channel."""
    rows = {
        K.PID_K_P: k_p, K.PID_K_I: k_i, K.PID_K_D: k_d,
        K.PID_MAX_ERR_I: np.inf if max_err_i is None else max_err_i,
        K.PID_LIMIT: limit, K.PID_MAX_ACCELERATION: max_acceleration}
    return xp.stack([
        xp.broadcast_to(xp.asarray(rows[i], dtype=float), (n,)) for i in range(K.N_PID_PARAMS)])



def build_controller(
    gains: CascadeGains, params: VehicleParams, simulation: SimulationParams
) -> ControllerParams:
    """
    The `multirotor.tuples.ControllerParams` that a rollout takes, from the controller's gains
    and the vehicle it is for.

    It is a function of its arguments alone, with numpy or JAX operations, so it can be
    differentiated with respect to the gains and to the vehicle's parameters. The vehicle
    sets the control allocation. A controller tuned for one vehicle and flown on another has
    the allocation and model of the first (`multirotor.builder.controller_model`).

    Parameters
    ----------
    gains : CascadeGains
        The gains and limits of the stages, and their periods.
    params : VehicleParams
        The vehicle that the controller allocates thrust and torques to.
    simulation : SimulationParams
        The simulation parameters. ``dt`` sets the number of steps in each period.

    Returns
    -------
    ControllerParams
        The matrix of the stages' parameters, the inverse of the allocation matrix, the
        stages' periods in steps and the options.
    """
    xp = xp_of((gains, params))
    pos, vel, att, rat = gains.position, gains.velocity, gains.attitude, gains.rate
    p_k_p = 0.5 * pos.max_jerk / pos.max_acceleration \
        if pos.leashing or pos.square_root_scaling else pos.k_p
    a_k_p = att.max_jerk / att.max_acceleration if att.square_root_scaling else att.k_p
    matrix = xp.concatenate([
        _columns(xp, 2, p_k_p, pos.k_i, pos.k_d, pos.max_err_i, pos.max_velocity,
                 pos.max_acceleration),
        _columns(xp, 2, vel.k_p, vel.k_i, vel.k_d, vel.max_err_i, vel.max_tilt),
        _columns(xp, 3, a_k_p, att.k_i, att.k_d, att.max_err_i, 0., att.max_acceleration),
        _columns(xp, 3, rat.k_p, rat.k_i, rat.k_d, rat.max_err_i, rat.max_acceleration),
        _columns(xp, 1, gains.altitude.k_p, 0., 0., gains.altitude.max_err_i,
                 gains.altitude.max_velocity),
        _columns(xp, 1, gains.climb.k_p, gains.climb.k_i, gains.climb.k_d,
                 gains.climb.max_err_i),
    ], axis=1)
    dt = simulation.dt
    steps = Steps(*(max(1, round(period / dt))
                    for period in (gains.period_p, gains.period_a, gains.period_z)))
    return ControllerParams(
        matrix, control_allocation_matrix(params)[1], steps, gains.feedforward_weight,
        bool(pos.leashing), bool(pos.square_root_scaling), bool(att.square_root_scaling))
