"""
A passive container for the parameters and state of the controller's PID stages,
as arrays. It computes nothing: the classes in `multirotor.controller.pid` read and
write it, and the functions in `multirotor.controller.cascade` take its arrays as
inputs. It plays the role for the controller that `multirotor.data` plays for the
vehicle's propellers and motors.

* In a `Controller`, one `ControllerData` holds all six stages, side by side. Each
  stage (`PosController`, `VelController`, ...) is a view of its own columns: it
  has no arrays of its own, and its properties read and write the shared ones.
* A standalone stage, or a bare `PIDController`, owns a `ControllerData` of its
  own, as a standalone `Propeller` owns a `PropellerData` of size 1.

Matrices have shape ``(n_rows, n)``: one row per parameter or state variable, one
column per control channel, as in `multirotor.data`. The
``PID_*`` constants below name the rows. A stage with 3 channels
(attitude) owns 3 adjacent columns, so a row such as ``params[PID_K_P, start:stop]`` is
a contiguous array with one value per channel.
"""
import numpy as np

# Rows of the parameter matrix
PID_K_P = 0
"Proportional gain"
PID_K_I = 1
"Integral gain"
PID_K_D = 2
"Derivative gain"
PID_MAX_ERR_I = 3
"Limit on the accumulated error"
PID_LIMIT = 4
"Limit on the output of the stage: max_velocity, max_tilt or max_acceleration"
PID_MAX_ACCELERATION = 5
"Acceleration limit of the square-root controller"
N_PID_PARAMS = 6

# Rows of the state matrix. A stage returns its state in this order.
PID_ERR = 0
PID_ERR_P = 1
PID_ERR_I = 2
PID_ERR_D = 3
PID_REFERENCE = 4
PID_ACTION = 5
PID_INTEGRAL = 6
"Accumulated error, before it is scaled by the integral gain"
N_PID_STATES = 7

# First column of each stage in a `Controller`'s matrices, in the order in which
# `Controller` adopts its stages
CASCADE_POSITION = 0
CASCADE_VELOCITY = 2
CASCADE_ATTITUDE = 4
CASCADE_RATE = 7
CASCADE_ALTITUDE = 10
CASCADE_CLIMB = 11
N_CASCADE_CHANNELS = 12



class ControllerData:
    """
    The parameters and state of ``n`` control channels.

    Attributes
    ----------
    params : np.ndarray
        Parameter matrix, shape (N_PID_PARAMS, n). The ``PID_*`` constants name the
        rows. The accumulated error is not limited by default.
    state : np.ndarray
        State matrix, shape (N_PID_STATES, n). The ``PID_*``
        constants name the rows: the error terms, the reference and the action.
    """

    def __init__(self, n: int) -> None:
        """
        Parameters
        ----------
        n : int
            The number of control channels.
        """
        self.n: int = n
        self.params: np.ndarray = np.zeros((N_PID_PARAMS, n))
        self.params[PID_MAX_ERR_I] = np.inf
        self.state: np.ndarray = np.zeros((N_PID_STATES, n))
