"""
The named tuples that `multirotor.rollout` takes and returns.

Both backends read them as they are: numba types a named tuple as a value on the
stack, and JAX treats it as a pytree that it can differentiate, and returns the
gradient in the same type, so ``grad(loss)(vehicle).mass`` is the gradient with
respect to the mass. A kernel reads ``vehicle.mass`` where it once read ``vehicle[1]``.

The imperative classes convert to and from these tuples, and nowhere else: see
`multirotor.simulation.Multirotor.rollout_params` and
`multirotor.controller.Controller.rollout_params`. The matrices inside them keep the
layout of the imperative classes. A row of ``prop_params`` is named by the ``PROP_*``
constants of `multirotor.physics`, and a row of ``ControllerParams.params`` by the ``PID_*``
constants of `multirotor.controller.data`.

The values can be numpy or JAX arrays. A named tuple is immutable. To change one value of
a tuple ``v``, make a new one with ``v._replace(mass=2.)``. That works with numpy and
JAX, but not inside numba-compiled code.

``inertia_inverse`` is a field of its own, so that a step does not invert the matrix. To
differentiate with respect to the inertia, replace both fields:
``v._replace(inertia=m, inertia_inverse=jnp.linalg.inv(m))``. Replacing only ``inertia``
leaves the dynamics with the old inverse, and the gradient is silently wrong.
"""
from collections import namedtuple


BodyVehicle = namedtuple('BodyVehicle', (
    'g mass inertia inertia_inverse prop_params clockwise positions'))
"""
The parameters of a vehicle without motors. ``prop_params`` is the propeller matrix,
``clockwise`` the sign of each propeller's spin and ``positions`` the propellers' positions
relative to the center of mass, shape (3, n).
"""

MotorsVehicle = namedtuple('MotorsVehicle', BodyVehicle._fields + (
    'motor_params', 'max_voltage', 'disturbance'))
"The parameters of a vehicle with motors: those of `BodyVehicle` and the motor matrix."

BatteryVehicle = namedtuple('BatteryVehicle', MotorsVehicle._fields + (
    'battery_params', 'ocv_soc', 'ocv_voltage'))
"The parameters of a vehicle with motors and a battery model."

KINDS = {BodyVehicle: 'body', MotorsVehicle: 'motors', BatteryVehicle: 'motors_battery'}
"The ``closed_loop_<kind>`` function that takes each type of vehicle."


VehicleState = namedtuple('VehicleState', 'x aux acceleration t')
"""
The state of a vehicle:

* ``x``: the 12 body states, then the motor speeds if the vehicle has motors.
* ``aux``: ``(charge drawn, cut off, overloaded)`` of the battery, as floats; a
  placeholder otherwise.
* ``acceleration``: the body-frame acceleration at the start of the last step, which
  the controller's leash reads.
* ``t``: the time.
"""


ControllerParams = namedtuple('ControllerParams', (
    'params alloc_inverse steps feedforward_weight leashing square_root_position '
    'square_root_attitude'))
"""
The parameters of a `multirotor.controller.Controller`: the matrix of the stages'
parameters, the inverse of the vehicle's control allocation matrix, the periods of the
position, attitude and altitude stages (``steps``), and the options.
"""

ControllerState = namedtuple('ControllerState', 'x memo n')
"""
The state of a controller: the matrix of the stages' state ``x``, the ``memo`` of the
outputs that the stages running at their own period hold between their runs
(``(ref_vel, thrust, torques)``), and the number of steps ``n`` taken. ``n`` sets the phase
of those stages.
"""


def kind_of(vehicle) -> str:
    "``'body'``, ``'motors'`` or ``'motors_battery'``: the type of a vehicle's parameters."
    return KINDS[type(vehicle)]
