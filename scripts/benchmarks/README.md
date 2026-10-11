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
if any case is more than `--threshold` percent slower (default 10).

## Cases

| case | what it times |
|---|---|
| `step_speeds` | `Multirotor.step_speeds` (one `odeint` call) with fixed hover speeds. Rotors 4/6/8; constant-thrust, with motor model, and blade-element (`fsolve`) thrust. |
| `step_dynamics` | `Multirotor.step_dynamics` with fixed force/torque input. |
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
* Timing noise: run the same commit twice and compare before trusting a small delta.
