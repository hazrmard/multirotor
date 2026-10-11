"""Chart one or more result files (one color per file, in the order given).

    uv run python scripts/benchmarks/plot.py results/baseline.json results/opt.json
    uv run python scripts/benchmarks/plot.py A.json B.json --out charts --theme dark

Writes PNGs: ``time_per_step.png`` (median with min-max whiskers),
``speedup.png`` (first file / each later file), ``memory.png`` (tracemalloc
peak per case) and ``log_scaling.png`` (DataLog growth vs steps).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import index_by_id, load_results  # noqa: E402

# Categorical slots 1-4 from the reference palette, fixed order (light, dark).
SERIES = {'light': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'],
          'dark': ['#3987e5', '#d95926', '#199e70', '#c98500']}
THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', grid='#e1e0d9'),
    'dark': dict(surface='#1a1a19', ink='#f0efec', ink2='#c3c2b7', grid='#2c2c2a'),
}


def style(ax, t):
    ax.set_facecolor(t['surface'])
    ax.grid(axis='x', color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(t['grid'])
    ax.tick_params(colors=t['ink2'], length=0)
    ax.xaxis.label.set_color(t['ink2'])
    ax.title.set_color(t['ink'])


def new_fig(t, height):
    fig, ax = plt.subplots(figsize=(9, height), facecolor=t['surface'])
    style(ax, t)
    return fig, ax


def legend(ax, t, labels, colors):
    if len(labels) < 2:
        return
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    leg = ax.legend(handles, labels, frameon=False, loc='lower right')
    for txt in leg.get_texts():
        txt.set_color(t['ink2'])


def timed(files):
    """Case ids that have timing in the first file, in file order."""
    return [cid for cid, r in index_by_id(files[0]).items() if 'stats' in r]


def hbar_groups(ax, ids, values, colors, errs=None, gap=0.2):
    n = len(values)
    h = (1 - gap) / n
    y0 = np.arange(len(ids))
    for k, (v, c) in enumerate(zip(values, colors)):
        y = y0 - (1 - gap) / 2 + h * (k + 0.5)
        xerr = None if errs is None else errs[k]
        ax.barh(y, v, height=h * 0.9, color=c, xerr=xerr,
                error_kw=dict(ecolor=c, alpha=.6, capsize=2, lw=1))
    ax.set_yticks(y0, ids)
    ax.invert_yaxis()


def plot_time(files, labels, colors, t, out):
    ids = timed(files)
    idx = [index_by_id(f) for f in files]
    med = [[i.get(c, {}).get('stats', {}).get('median', np.nan) for c in ids] for i in idx]
    lo = [[m - i.get(c, {}).get('stats', {}).get('min', m) if m == m else 0
           for m, c in zip(row, ids)] for row, i in zip(med, idx)]
    hi = [[i.get(c, {}).get('stats', {}).get('max', m) - m if m == m else 0
           for m, c in zip(row, ids)] for row, i in zip(med, idx)]
    fig, ax = new_fig(t, 0.55 * len(ids) * len(files) ** .5 + 1.5)
    hbar_groups(ax, ids, med, colors, errs=[[l, h] for l, h in zip(lo, hi)])
    ax.set_xscale('log')
    ax.set_xlabel('median time per simulation step (us, log scale; whiskers = min-max)')
    ax.set_title('Simulation step cost', loc='left')
    legend(ax, t, labels, colors)
    fig.tight_layout()
    fig.savefig(out / 'time_per_step.png', dpi=150, facecolor=t['surface'])
    plt.close(fig)


def plot_speedup(files, labels, colors, t, out):
    if len(files) < 2:
        return
    ids = timed(files)
    idx = [index_by_id(f) for f in files]
    base = [idx[0][c]['stats']['median'] for c in ids]
    vals = [[b / i[c]['stats']['median'] if c in i and 'stats' in i[c] else np.nan
             for b, c in zip(base, ids)] for i in idx[1:]]
    fig, ax = new_fig(t, 0.5 * len(ids) * (len(files) - 1) ** .5 + 1.5)
    hbar_groups(ax, ids, vals, colors[1:])
    ax.axvline(1, color=t['ink2'], lw=1)
    ax.set_xlabel(f'speedup vs {labels[0]} (>1 is faster)')
    ax.set_title('Speedup', loc='left')
    legend(ax, t, labels[1:], colors[1:])
    fig.tight_layout()
    fig.savefig(out / 'speedup.png', dpi=150, facecolor=t['surface'])
    plt.close(fig)


def plot_memory(files, labels, colors, t, out):
    ids = [c for c, r in index_by_id(files[0]).items()
           if 'tracemalloc' in r and r['case'] != 'log_scaling']
    if not ids:
        return
    idx = [index_by_id(f) for f in files]
    vals = [[i.get(c, {}).get('tracemalloc', {}).get('peak_kb', np.nan) for c in ids]
            for i in idx]
    fig, ax = new_fig(t, 0.5 * len(ids) * len(files) ** .5 + 1.5)
    hbar_groups(ax, ids, vals, colors)
    ax.set_xlabel('tracemalloc peak during run (KiB)')
    ax.set_title('Python-level memory per run', loc='left')
    legend(ax, t, labels, colors)
    fig.tight_layout()
    fig.savefig(out / 'memory.png', dpi=150, facecolor=t['surface'])
    plt.close(fig)


def plot_log_scaling(files, labels, colors, t, out):
    fig, ax = new_fig(t, 4.5)
    ax.grid(axis='y', color=t['grid'], linewidth=0.8)
    drawn = False
    for f, label, c in zip(files, labels, colors):
        pts = sorted((r['params']['steps'], r['tracemalloc']['peak_kb'])
                     for r in f['results']
                     if r['case'] == 'log_scaling' and 'tracemalloc' in r)
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, [v / 1024 for v in y], color=c, lw=2, marker='o', ms=8,
                markeredgecolor=t['surface'], markeredgewidth=2, label=label)
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    ax.set_xlabel('logged steps')
    ax.set_ylabel('peak memory while logging (MiB)')
    ax.yaxis.label.set_color(t['ink2'])
    ax.set_title('Log memory growth', loc='left')
    if len(files) > 1:
        leg = ax.legend(frameon=False, loc='upper left')
        for txt in leg.get_texts():
            txt.set_color(t['ink2'])
    fig.tight_layout()
    fig.savefig(out / 'log_scaling.png', dpi=150, facecolor=t['surface'])
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--out', default='charts')
    ap.add_argument('--theme', choices=['light', 'dark'], default='light')
    args = ap.parse_args(argv)
    if len(args.files) > 4:
        sys.exit('at most 4 result files per chart')

    files = [load_results(p) for p in args.files]
    labels = [f['label'] for f in files]
    t = THEMES[args.theme]
    colors = SERIES[args.theme][:len(files)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_time(files, labels, colors, t, out)
    plot_speedup(files, labels, colors, t, out)
    plot_memory(files, labels, colors, t, out)
    plot_log_scaling(files, labels, colors, t, out)
    print('wrote', *sorted(p.name for p in out.glob('*.png')), 'to', out)


if __name__ == '__main__':
    main()
