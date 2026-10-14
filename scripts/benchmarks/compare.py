"""Compare two result files.

    uv run python scripts/benchmarks/compare.py baseline.json candidate.json
    uv run python scripts/benchmarks/compare.py A.json B.json --threshold 10

Prints per-case medians and the relative change (negative = faster / smaller).
Exits with status 1 if any case is slower than ``--threshold`` percent, so the
script can gate CI.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import index_by_id, load_results  # noqa: E402
from scopes import Level  # noqa: E402


def pct(new: float, old: float) -> float:
    return (new / old - 1.) * 100. if old else float('nan')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('baseline')
    ap.add_argument('candidate')
    ap.add_argument('--threshold', type=float, default=15.,
                    help='percent slowdown that counts as a regression (default 15; '
                         'two runs of one commit on the dev VM differed by up to 11%%)')
    ap.add_argument('--mem-threshold', type=float, default=10.,
                    help='percent growth of tracemalloc peak that counts as a '
                         'regression; cases under 100 KiB are ignored (default 10)')
    ap.add_argument('--detail', action='store_true',
                    help='show fine scopes (integrator, model, propellers) instead of coarse')
    args = ap.parse_args(argv)

    a, b = load_results(args.baseline), load_results(args.candidate)
    for tag, d in (('baseline', a), ('candidate', b)):
        m = d['meta']
        print(f'{tag:<10} {d["label"]}  {m["git_sha"][:8]}{"+dirty" if m["git_dirty"] else ""}'
              f'  {m["cpu"]}  py{m["python"]} numpy{m["numpy"]} numba{m["numba"]}')
    if a['meta']['cpu'] != b['meta']['cpu']:
        print('WARNING: different CPUs; timing deltas are not meaningful.')
    if a.get('quick') or b.get('quick'):
        print('WARNING: --quick results are smoke tests, not benchmarks.')

    ia, ib = index_by_id(a), index_by_id(b)
    regressions = []
    print(f'\n{"case":<60} {"base us":>10} {"new us":>10} {"time %":>8}'
          f' {"RSS %":>8} {"peak %":>8}')
    for cid in ia:
        if cid not in ib:
            print(f'{cid:<60} (missing in candidate)')
            continue
        ra, rb = ia[cid], ib[cid]
        if 'error' in ra or 'error' in rb:
            side = 'candidate' if 'error' in rb else 'baseline'
            print(f'{cid:<60} (worker failed in {side})')
            if 'error' in rb and 'error' not in ra:
                regressions.append((cid, 'worker failed'))
            continue
        if cid == 'cold_start':
            ta, tb = ra['first_step_s']['median'], rb['first_step_s']['median']
            print(f'{"cold_start first step (s)":<60} {ta:>10.2f} {tb:>10.2f}'
                  f' {pct(tb, ta):>+8.1f}')
            continue
        row = f'{cid:<60}'
        if 'stats' in ra and 'stats' in rb:
            ta, tb = ra['stats']['median'], rb['stats']['median']
            d = pct(tb, ta)
            flag = ' !' if d > args.threshold else ''
            row += f' {ta:>10.1f} {tb:>10.1f} {d:>+8.1f}{flag}'
            if d > args.threshold:
                regressions.append((cid, f'{d:+.1f}%'))
            row += f' {pct(rb["peak_rss_mb"], ra["peak_rss_mb"]):>+8.1f}'
        else:
            row += f' {"-":>10} {"-":>10} {"-":>8} {"-":>8}'
        ma, mb = ra.get('tracemalloc'), rb.get('tracemalloc')
        if ma and mb:
            d = pct(mb['peak_kb'], ma['peak_kb'])
            big = ma['peak_kb'] >= 100
            flag = '!' if big and d > args.mem_threshold else ' '
            row += f' {d:>+7.1f}{flag}'
            if flag == '!':
                regressions.append((cid + ' (memory)', f'{d:+.1f}%'))
        else:
            row += f' {"-":>8}'
        print(row)
    for cid in ib:
        if cid not in ia:
            print(f'{cid:<60} (new in candidate)')

    # Diagnostic only: never affects the exit status.
    level = (Level.FINE if args.detail else Level.COARSE).key
    shown = False
    for cid in ia:
        sa = (ia[cid].get('scopes') or {}).get(level) if cid in ib else None
        sb = (ib[cid].get('scopes') or {}).get(level) if cid in ib else None
        if not sa or not sb:
            continue
        if not shown:
            print(f'\nwhere the time went ({level} scopes; exclusive us/step, '
                  f'base>new; scopes under 1% of the step are hidden):')
            shown = True
        print(f'\n{cid}')
        xa = {i['path']: i for i in sa['items']}
        xb = {i['path']: i for i in sb['items']}
        for path in sorted(set(xa) | set(xb)):
            ea = xa.get(path, {}).get('excl_us', 0.)
            eb = xb.get(path, {}).get('excl_us', 0.)
            if max(ea / sa['total_us'], eb / sb['total_us']) < 0.01:
                continue
            ca = xa.get(path, {}).get('calls_per_step', 0.)
            cb = xb.get(path, {}).get('calls_per_step', 0.)
            parts = path.split('/')
            label = '  ' * (len(parts) - 1) + parts[-1]
            print(f'  {label:<34} {ea:>8.1f} > {eb:<8.1f} {pct(eb, ea):>+7.1f}%'
                  f'   calls {ca:.2f} > {cb:.2f}')
        print(f'  {"total (instrumented)":<34} {sa["total_us"]:>8.1f} > {sb["total_us"]:<8.1f}'
              f'   instrumented/clean {sa["vs_clean"]:.2f} > {sb["vs_clean"]:.2f}')

    if regressions:
        print(f'\n{len(regressions)} regression(s) (time > {args.threshold:.0f}%, '
              f'memory > {args.mem_threshold:.0f}%):')
        for cid, what in regressions:
            print(f'  {cid}: {what}')
        return 1
    print('\nno regressions over threshold.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
