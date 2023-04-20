from .pid import (
    PIDController,
    PosController,
    VelController,
    AttController,
    RateController,
    AltController,
    AltRateController,
    Controller
)
try:
    import pyscurve
    from .scurves import SCurveController
except ImportError:
    pass
