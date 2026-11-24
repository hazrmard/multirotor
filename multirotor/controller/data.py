"""
The layout of the controller's parameters and state: matrices with one row per parameter or
state variable and one column per control channel. The ``PID_*`` constants name the rows
and the ``CASCADE_*`` constants the first column of each stage. A stage with 3 channels
(attitude) owns 3 adjacent columns, so a row such as ``params[PID_K_P, start:stop]`` is a
contiguous array with one value per channel.

* In a `multirotor.controller.Controller`, the matrices hold all six stages, side by side.
  Each stage (`PosController`, `VelController`, ...) indexes its own columns: it has no
  arrays of its own, and its properties read and write the shared ones.
* A standalone stage, or a bare `PIDController`, owns matrices of its own.

The functions in `multirotor.controller.cascade` take the matrices as inputs.
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



def new_matrices(n: int) -> tuple:
    """
    The parameter and state matrices of ``n`` control channels.

    Parameters
    ----------
    n : int
        The number of control channels.

    Returns
    -------
    tuple
        The parameter matrix, shape (N_PID_PARAMS, n), whose accumulated error is not limited
        by default, and the state matrix, shape (N_PID_STATES, n), of zeros.
    """
    params = np.zeros((N_PID_PARAMS, n))
    params[PID_MAX_ERR_I] = np.inf
    return params, np.zeros((N_PID_STATES, n))
