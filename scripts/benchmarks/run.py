"""Run the benchmark suite and write a JSON result file.

    uv run python scripts/benchmarks/run.py --label baseline
    uv run python scripts/benchmarks/run.py --quick --cases step_speeds closed_loop

Every case runs in a fresh subprocess, once for timing and once for memory
(``tracemalloc``), so the two never perturb each other and peak RSS is per case.
"""
import argparse
import gc
import json
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from instrument import breakdown  # noqa: E402
from common import (RESULTS_DIR, SCHEMA_VERSION, environment_metadata,  # noqa: E402
                    save_results, summarize)

WARMUP_STEPS = 50


def _peak_rss_mb() -> float:
    # ru_maxrss is KiB on Linux, bytes on macOS
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024 if sys.platform == 'darwin' else 1024)


def worker_time(name: str, params: dict, repeats: int) -> dict:
    import cases
    t0 = time.perf_counter()
    runner = cases.build(name, params)
    build_s = time.perf_counter() - t0
    steps = params['steps']

    # Warmup: numba JIT, caches, first-call allocations. Timed separately.
    runner.reset()
    t0 = time.perf_counter()
    runner.run(min(WARMUP_STEPS, steps))
    warmup_s = time.perf_counter() - t0

    samples = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            runner.reset()
            t0 = time.perf_counter_ns()
            runner.run(steps)
            samples.append((time.perf_counter_ns() - t0) / 1e3 / steps)  # us/step
    finally:
        gc.enable()
    stats = summarize(samples)
    split = breakdown(runner, min(300, steps))
    return dict(
        breakdown=split,
        us_per_step=samples, stats=stats,
        steps_per_s=1e6 / stats['median'],
        real_time_factor=runner.dt * 1e6 / stats['median'],
        build_s=build_s, warmup_s=warmup_s, peak_rss_mb=_peak_rss_mb(),
    )


def worker_mem(name: str, params: dict) -> dict:
    import cases
    runner = cases.build(name, params)
    steps = params['steps']
    runner.reset()
    runner.run(min(WARMUP_STEPS, steps))  # JIT/caches outside the traced region
    runner.reset()
    gc.collect()
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    runner.run(steps)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(
        peak_kb=(peak - base) / 1024, net_kb=(cur - base) / 1024,
        net_bytes_per_step=(cur - base) / steps,
        peak_bytes_per_step=(peak - base) / steps,
    )


def worker_cold() -> dict:
    t0 = time.perf_counter()
    import numpy as np  # noqa: F401
    from multirotor.controller import Controller  # noqa: F401
    from multirotor.simulation import Multirotor
    t_import = time.perf_counter() - t0
    import cases
    t0 = time.perf_counter()
    v = cases.make_vehicle(8)
    t_construct = time.perf_counter() - t0
    u = cases.hover_speeds(v)
    t0 = time.perf_counter()
    v.step_speeds(u)
    t_first = time.perf_counter() - t0
    t0 = time.perf_counter()
    v.step_speeds(u)
    t_second = time.perf_counter() - t0
    return dict(import_s=t_import, construct_s=t_construct,
                first_step_s=t_first, second_step_s=t_second,
                peak_rss_mb=_peak_rss_mb())


def run_worker(args) -> None:
    params = json.loads(args.params) if args.params else {}
    if args.worker == 'time':
        out = worker_time(args.case, params, args.repeats)
    elif args.worker == 'mem':
        out = worker_mem(args.case, params)
    else:
        out = worker_cold()
    print('@@RESULT@@' + json.dumps(out))


def spawn(mode: str, case: str = '', params: dict = None, repeats: int = 0) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', mode]
    if case:
        cmd += ['--case', case, '--params', json.dumps(params),
                '--repeats', str(repeats)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith('@@RESULT@@'):
            return json.loads(line[len('@@RESULT@@'):])
    raise RuntimeError(
        f'worker {mode} {case} {params} failed (exit {proc.returncode}):\n'
        f'{proc.stderr[-2000:]}')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--label', default=None, help='name of the result file')
    ap.add_argument('--out', default=None, help='explicit output path')
    ap.add_argument('--cases', nargs='*', help='case names to run (default: all)')
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--quick', action='store_true', help='smoke test: few steps')
    ap.add_argument('--no-memory', action='store_true', help='skip tracemalloc runs')
    ap.add_argument('--list', action='store_true', help='list case ids and exit')
    ap.add_argument('--worker', choices=['time', 'mem', 'cold'], help=argparse.SUPPRESS)
    ap.add_argument('--case', help=argparse.SUPPRESS)
    ap.add_argument('--params', help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.worker:
        return run_worker(args)

    import cases
    selected = cases.expand(args.cases, quick=args.quick)
    if args.list:
        for cid, _, _ in selected:
            print(cid)
        return
    if not selected:
        sys.exit('no cases selected')

    repeats = 3 if args.quick else args.repeats
    results = []
    for cid, name, params in selected:
        print(f'{cid:<62}', end=' ', flush=True)
        # log_scaling is a memory-only case: skip the timing repeats
        rec = dict(id=cid, case=name, params=params)
        if name != 'log_scaling':
            rec.update(spawn('time', name, params, repeats))
        if not args.no_memory:
            rec['tracemalloc'] = spawn('mem', name, params)
        results.append(rec)
        msg = []
        if 'stats' in rec:
            bd = rec.get('breakdown')
            msg.append(f'{rec["stats"]["median"]:9.1f} us/step '
                       f'(RTF {rec["real_time_factor"]:5.1f}x)')
            if bd:
                msg.append(f'model {bd["model_us"]:7.1f} integ {bd["integrator_us"]:6.1f}'
                           f' other {bd["other_us"]:6.1f}  {bd["evals_per_step"]:4.1f} evals')
        if 'tracemalloc' in rec:
            msg.append(f'{rec["tracemalloc"]["net_bytes_per_step"]:8.1f} B/step net')
        print('  '.join(msg))

    print(f'{"cold_start":<62}', end=' ', flush=True)
    cold = [spawn('cold') for _ in range(3 if args.quick else 5)]
    cold_rec = dict(
        id='cold_start', case='cold_start', params={},
        first_step_s=summarize([c['first_step_s'] for c in cold]),
        import_s=summarize([c['import_s'] for c in cold]),
        construct_s=summarize([c['construct_s'] for c in cold]),
        second_step_s=summarize([c['second_step_s'] for c in cold]),
        peak_rss_mb=summarize([c['peak_rss_mb'] for c in cold])['median'],
    )
    results.append(cold_rec)
    print(f'first step {cold_rec["first_step_s"]["median"]:.2f} s')

    label = args.label or time.strftime('%Y%m%d-%H%M%S')
    payload = dict(schema=SCHEMA_VERSION, label=label, quick=args.quick,
                   meta=environment_metadata(), results=results)
    out = Path(args.out) if args.out else RESULTS_DIR / f'{label}.json'
    save_results(out, payload)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
