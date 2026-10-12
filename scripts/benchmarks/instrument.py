"""Split step time into model time and integrator time.

``Multirotor`` accepts an ``integrator`` callable. ``TimedIntegrator`` wraps one
and times the derivative function it is handed:

* model time: spent inside ``fun`` (the vehicle's ``dxdt_*``: forces, torques,
  state derivative);
* integrator time: the rest of the integrator call (step-size control, solver
  bookkeeping, and the C/Fortran-to-Python call overhead);
* everything outside the integrator call (rounding, propeller/battery stepping,
  the ``_dxdt`` evaluation that precedes it, and in closed-loop cases the
  controller and logging) is "other".
"""
import time
from typing import Callable, Dict, Optional

pc = time.perf_counter_ns


class TimedIntegrator:
    def __init__(self, base: Callable):
        self.base = base
        self.reset()

    def reset(self) -> None:
        self.call_ns = 0    # wall time inside integrator calls (model included)
        self.model_ns = 0   # wall time inside fun
        self.evals = 0      # evaluations of fun requested by the integrator
        self.steps = 0      # integrator calls

    def __call__(self, fun, y0, dt, args=()):
        def timed(*a, **k):
            t = pc()
            y = fun(*a, **k)
            self.model_ns += pc() - t
            self.evals += 1
            return y

        t0 = pc()
        y = self.base(timed, y0, dt, args)
        self.call_ns += pc() - t0
        self.steps += 1
        return y


def timer_overhead_ns(n: int = 100_000) -> float:
    """Approximate per-evaluation cost of the timing wrapper itself."""
    ti = TimedIntegrator(lambda fun, y0, dt, args: fun(0., y0))
    t0 = pc()
    for _ in range(n):
        ti(lambda t, y: y, 0., 0., ())
    return (pc() - t0) / n


def breakdown(runner, steps: int) -> Optional[Dict[str, float]]:
    """Run ``steps`` steps with a ``TimedIntegrator`` and report per-step costs.

    Returns None when the runner has no vehicle exposing ``integrator`` (e.g. an
    older commit without the injection point) or never calls the integrator.
    Runs separately from the clean timing pass because wrapping adds overhead.
    """
    v = getattr(runner, 'v', None)
    if v is None or not hasattr(v, 'integrator'):
        return None
    timed = TimedIntegrator(v.integrator)
    v.integrator = timed
    try:
        runner.reset()
        t0 = pc()
        runner.run(steps)
        total_ns = pc() - t0
    finally:
        v.integrator = timed.base
    if timed.steps == 0 or timed.evals == 0:
        return None
    n = timed.steps
    return dict(
        evals_per_step=timed.evals / n,
        model_us=timed.model_ns / n / 1e3,
        integrator_us=(timed.call_ns - timed.model_ns) / n / 1e3,
        other_us=(total_ns - timed.call_ns) / n / 1e3,
        total_us=total_ns / n / 1e3,
        model_us_per_eval=timed.model_ns / timed.evals / 1e3,
        timer_overhead_ns=timer_overhead_ns(),
    )
