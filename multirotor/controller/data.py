"""
A passive container for the parameters and state of the controller's PID stages,
as arrays. It computes nothing: the classes in `multirotor.controller.pid` read and
write it, and the kernels in `multirotor.controller.kernels` take its arrays as
inputs. It plays the role for the controller that `multirotor.data` plays for the
vehicle's propellers and motors.

* In a `Controller`, one `ControllerData` holds all six stages, side by side. Each
  stage (`PosController`, `VelController`, ...) is a view of its own columns: it
  has no arrays of its own, and its properties read and write the shared ones.
* A standalone stage, or a bare `PIDController`, owns a `ControllerData` of its
  own, as a standalone `Propeller` owns a `PropellerData` of size 1.

Matrices have shape ``(n_rows, n)``: one row per parameter or state variable, one
column per control channel, as in `multirotor.data`. The
``PID_*`` constants in `multirotor.controller.kernels` name the rows. A stage with 3 channels
(attitude) owns 3 adjacent columns, so a row such as ``params[PID_K_P, start:stop]`` is
a contiguous array with one value per channel.
"""
import numpy as np

from .kernels import N_PID_PARAMS, N_PID_STATES, PID_MAX_ERR_I



class ControllerData:
    """
    The parameters and state of ``n`` control channels.

    Attributes
    ----------
    params : np.ndarray
        Parameter matrix, shape (N_PID_PARAMS, n). The
        ``PID_*`` constants in `multirotor.controller.kernels` name the rows. The accumulated
        error is not limited by default.
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
