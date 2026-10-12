# Benchmarks

Speed and memory benchmarks for the simulator. They live outside the
`multirotor/` package on purpose: the wheel only ships `multirotor/`, and the
scripts import the installed package (same convention as numpy/scipy `benchmarks/`).

```
uv sync
uv run python scripts/benchmarks/run.py --label baseline     # full run, a few minutes
uv run python scripts/benchmarks/run.py --quick              # smoke test, not a measurement
uv run python scripts/benchmarks/run.py --list               # case ids
# ... change code ...
uv run python scripts/benchmarks/run.py --label candidate
uv run python scripts/benchmarks/compare.py scripts/benchmarks/results/baseline.json scripts/benchmarks/results/candidate.json
uv run python scripts/benchmarks/plot.py scripts/benchmarks/results/baseline.json scripts/benchmarks/results/candidate.json --out charts
```

Results go to `scripts/benchmarks/results/` (git-ignored). `compare.py` exits 1
if any case is more than `--threshold` percent slower (default 15) or its
`tracemalloc` peak grows more than `--mem-threshold` percent (default 10, cases
under 100 KiB ignored).

## Cases

| case | what it times |
|---|---|
| `step_speeds` | `Multirotor.step_speeds` (one `odeint` call). Inputs cycle through a seeded +-5% perturbation of hover speeds (`excite=True`); one `excite=False` case at exact hover is the integrator's best case. Rotors 4/6/8; constant-thrust, with motor model, and blade-element (`fsolve`) thrust. |
| `step_dynamics` | `Multirotor.step_dynamics`, excited and at-hover variants. |
| `controller_step` | `Controller.step` alone, vehicle held fixed. |
| `closed_loop` | controller -> `allocate_control` -> `step_speeds`, with and without `DataLog`. |
| `run_sim` | `optimize.run_sim` through `DynamicsMultirotorEnv` (uses `step_dynamics`). |
| `log_scaling` | Memory only: `DataLog` over 1k/5k/20k steps. |
| `cold_start` | Fresh-process import, construct, first step (numba JIT) and second step. |

All use the `tarott18` propeller/motor constants, `dt=0.01`, `float32`, and hold
5 m altitude where a reference is needed. Add a case by writing a `Runner`
(`reset()`, `run(n)`, `dt`) in `cases.py` and listing its parameter grid in `CASES`.

## Metrics

* `us_per_step`: wall time per step, one sample per repeat (default 7). Compare on
  the **median**; `min` is the noise floor. `real_time_factor` = simulated seconds
  per wall second.
* `scopes`: a scope tree (`scopes.py`) from separate instrumented passes, one per
  level, because wrapping adds overhead. Each scope has `calls_per_step`,
  inclusive and exclusive us/step, keyed by path (`loop/step_speeds/integrator/model`).
  * `coarse`: controller, allocation, env step, reward, the `step_*` calls, logging.
  * `fine`: adds the integrator, the model (`dxdt_*`), `forces_torques` (the
    per-rotor loop inside the model) and `Propeller.step`.
  * Invariants, exact by construction: `inclusive = exclusive + sum(children's
    inclusive)` and `sum(exclusive over all scopes) = inclusive(loop)`. Time no
    child scope covers stays in the parent's exclusive time, so there is no
    unexplained "other" bucket.
  * The same function can appear under two paths: `step_speeds/model` is the one
    `_dxdt` evaluation before the solve; `step_speeds/integrator/model` are the
    evaluations the solver requests (`evals_per_step` = their count per integrator call).
  * `integrator` exclusive time is solver overhead (LSODA bookkeeping plus the
    call into Python); it excludes the model time below it.
  * The wrappers cost roughly 0.8 us per call. `scopes.calibrate()` measures that
    and `report()` removes it (`a_ns`, `b_ns` in the file). `vs_clean` is the
    instrumented total divided by the clean median; far from 1 (the run prints a
    warning above 1.3) means the instrumented pass hit a noisy machine, so do not
    trust its shares.
  * Jitted functions (`apply_forces_torques`, `torque`) cannot be wrapped, so they
    stay in the exclusive time of the scope that calls them.
  * Which callables become scopes is declared per case in `cases.py`
    (`Runner.scope_points`). Missing attributes are skipped, so older commits
    degrade instead of failing. The integrator scope needs the `integrator`
    argument of `Multirotor`.
  * `run.py --tree` prints the tree per case; `--scopes off|coarse|fine|both`
    selects the passes; `compare.py --detail` compares fine scopes.
* `peak_rss_mb`: process peak RSS of the timing run (includes the interpreter,
  numpy, scipy, numba; compare deltas, not absolutes).
* `tracemalloc`: Python/numpy allocations during the run, measured in a separate
  process because tracemalloc slows execution. `peak_kb` is the high-water mark,
  `net_kb` what is still held at the end. `net_bytes_per_step` is only meaningful
  for logging cases; elsewhere it is a small constant divided by the step count.

## Method notes

* Each case runs in its own subprocess, once for timing and once for memory.
* Warm-up steps run first so numba compilation is not in the samples
  (it is reported separately as `warmup_s` and in `cold_start`).
* GC is disabled inside timed regions.
* Run on an idle machine, optionally pinned (`taskset -c 0`). Only compare files
  from the same host; `compare.py` warns when the CPU differs. Each file records
  git SHA/dirty flag and library versions.
* Timing noise: two full runs of the same commit on the dev VM differed by up to
  11% per case (short cases and the blade model are noisiest). Treat deltas under
  ~15% as noise unless repeated; `evals_per_step` changes are always real.
