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
                regressions.append((cid, d))
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
                regressions.append((cid + ' (memory)', d))
        else:
            row += f' {"-":>8}'
        print(row)
    for cid in ib:
        if cid not in ia:
            print(f'{cid:<60} (new in candidate)')

    # Diagnostic only: never affects the exit status.
    rows = [(cid, ia[cid]['breakdown'], ib[cid]['breakdown']) for cid in ia
            if cid in ib and ia[cid].get('breakdown') and ib[cid].get('breakdown')]
    if rows:
        print(f'\nwhere the time went (us/step; model = derivative evaluation, '
              f'integrator = solver overhead):')
        print(f'{"case":<60} {"evals/step":>11} {"model/eval":>11} {"model":>11}'
              f' {"integrator":>11} {"other":>11}')
        for cid, x, y in rows:
            def pair(k, f='{:.1f}'):
                return (f.format(x[k]) + '>' + f.format(y[k])).rjust(11)
            print(f'{cid:<60} {pair("evals_per_step")} {pair("model_us_per_eval")}'
                  f' {pair("model_us")} {pair("integrator_us")} {pair("other_us")}')

    if regressions:
        print(f'\n{len(regressions)} regression(s) (time > {args.threshold:.0f}%, '
              f'memory > {args.mem_threshold:.0f}%):')
        for cid, d in regressions:
            print(f'  {cid}: {d:+.1f}%')
        return 1
    print('\nno regressions over threshold.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
