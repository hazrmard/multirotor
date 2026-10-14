"""Hierarchical scope timing.

``ScopeTimer`` keeps a stack of open scopes and records, per scope *path*
(e.g. ``loop/step_speeds/integrator/model``), the call count, inclusive time and
exclusive time. ``wrap`` turns a callable into a scope. ``restore`` removes the
wrappers.

Usage::

    timer = ScopeTimer()
    # Each wrap needs an owner object that has the named attribute.
    timer.wrap(vehicle, 'step_speeds', 'step_speeds')   # vehicle.step_speeds is now timed
    timer.wrap(vehicle, 'dxdt_speeds', 'model')

    timer.enter('loop')                                 # root scope
    for _ in range(1000):
        vehicle.step_speeds(speeds)
    timer.exit()
    timer.restore()                                     # remove the wrappers

    a_ns, b_ns = calibrate()                            # wrapper overhead
    report = timer.report(1000, a_ns, b_ns)             # per-step costs, overhead removed
    print(format_tree(report))

``Level``, ``ScopePoint`` and ``Runner.scope_points()`` (in ``cases.py``) describe
which callables a benchmark case wraps.

Invariants (exact, by construction):

* ``inclusive(p) = exclusive(p) + sum(inclusive(c) for c in children(p))``
* ``sum(exclusive over all scopes) = inclusive(root)``

Nothing is lost to an "other" bucket: time not covered by a child scope stays in
the parent's exclusive time, which shows where a new scope would help.

Wrapping costs time. ``calibrate()`` measures two constants and ``report()``
removes them:

* ``a``: wrapper time inside a scope's own window (charged to that scope);
* ``b``: wrapper time outside it (charged to the parent scope).

Run this module to self-test the invariants: ``python scopes.py``.
"""
import enum
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

pc = time.perf_counter_ns
Path = Tuple[str, ...]


class Level(enum.IntEnum):
    """Detail of a scope tree. Levels are cumulative: ``FINE`` includes ``COARSE``."""
    COARSE = 0   # controller, allocation, env, reward, step_*, logging
    FINE = 1     # adds integrator, model, forces_torques, propeller

    @property
    def key(self) -> str:
        """Key of this level in result files: ``'coarse'`` or ``'fine'``."""
        return self.name.lower()


class ScopePoint(NamedTuple):
    """A callable to time as a scope when a pass runs at ``level`` or finer.

    ``owner.attr`` must exist: ``attr`` is the name of a method or callable
    attribute of ``owner`` (an instance or a class). ``ScopeTimer.wrap`` replaces
    it with a timed version. If it does not exist, the scope is skipped and listed
    under ``skipped`` in the report.

    Example: ``ScopePoint(Level.COARSE, controller, 'step', 'controller')`` times
    ``controller.step`` as the scope ``controller``.
    """
    level: Level
    owner: Any
    attr: str
    name: str


class ScopeTimer:
    def __init__(self):
        self.stack: List[list] = []             # frames: [path, t_start, child_ns]
        self.stats: Dict[Path, List[int]] = {}  # path -> [calls, incl_ns, excl_ns]
        self._restores: List[Callable] = []
        self.skipped: List[str] = []            # 'Owner.attr' of scopes that could not be wrapped

    # -- recording ---------------------------------------------------------
    def enter(self, name: str) -> None:
        path = (self.stack[-1][0] if self.stack else ()) + (name,)
        frame = [path, 0, 0]
        self.stack.append(frame)
        frame[1] = pc()                          # clock read last

    def exit(self) -> None:
        t = pc()                                 # clock read first
        path, t0, child = self.stack.pop()
        dur = t - t0
        s = self.stats.get(path)
        if s is None:
            s = self.stats[path] = [0, 0, 0]
        s[0] += 1
        s[1] += dur
        s[2] += dur - child
        if self.stack:
            self.stack[-1][2] += dur

    def wrap(self, owner, attr: str, name: str) -> bool:
        """Replace ``owner.attr`` with a version that records the scope ``name``.

        ``owner`` is an instance or a class. ``attr`` must be the name of a callable
        attribute of ``owner``, for example ``wrap(vehicle, 'step_speeds', 'step')``.
        ``wrap`` returns False and leaves ``owner`` untouched when ``attr`` does not
        exist (for example on an older commit). It records ``'Owner.attr'`` in
        ``self.skipped``, so a misspelled attribute shows up in the report.
        """
        try:
            orig = getattr(owner, attr)
        except AttributeError:
            owner_name = owner.__name__ if isinstance(owner, type) else type(owner).__name__
            self.skipped.append(f'{owner_name}.{attr}')
            return False
        had_own = attr in vars(owner)
        enter, exit_ = self.enter, self.exit

        def timed(*a, **k):
            enter(name)
            try:
                return orig(*a, **k)
            finally:
                exit_()

        setattr(owner, attr, timed)

        def restore():
            if had_own:
                setattr(owner, attr, orig)
            else:
                delattr(owner, attr)
        self._restores.append(restore)
        return True

    def restore(self) -> None:
        for r in reversed(self._restores):
            r()
        self._restores.clear()

    # -- reporting ---------------------------------------------------------
    def report(self, steps: int, a_ns: float = 0., b_ns: float = 0.) -> Dict:
        """Per-step costs with the wrapper overhead (``a``, ``b``) removed."""
        children = defaultdict(list)
        for p in self.stats:
            if len(p) > 1:
                children[p[:-1]].append(p)
        calls = {p: s[0] for p, s in self.stats.items()}
        excl = {}
        for p, (n, _, ex) in self.stats.items():
            child_calls = sum(calls[c] for c in children[p])
            excl[p] = max(ex - a_ns * n - b_ns * child_calls, 0.)
        incl = {}
        for p in sorted(self.stats, key=len, reverse=True):
            incl[p] = excl[p] + sum(incl[c] for c in children[p])
        roots = [p for p in self.stats if len(p) == 1]
        root_incl = sum(incl[r] for r in roots)
        raw_root = sum(self.stats[r][1] for r in roots)
        us = 1. / steps / 1e3
        items = [
            dict(path='/'.join(p), calls_per_step=calls[p] / steps,
                 incl_us=incl[p] * us, excl_us=excl[p] * us)
            for p in sorted(self.stats)
        ]
        return dict(total_us=root_incl * us, raw_total_us=raw_root * us,
                    a_ns=a_ns, b_ns=b_ns, skipped=list(self.skipped), items=items)


def calibrate(n: int = 200_000) -> Tuple[float, float]:
    """Measure wrapper overhead ``(a_ns, b_ns)`` using an empty wrapped call."""
    class Holder:
        def f(self):
            return None

    h = Holder()
    t0 = pc()
    for _ in range(n):
        h.f()
    bare = (pc() - t0) / n

    timer = ScopeTimer()
    timer.wrap(h, 'f', 'child')
    timer.enter('parent')
    t0 = pc()
    for _ in range(n):
        h.f()
    loop = (pc() - t0) / n
    timer.exit()
    timer.restore()
    child_incl = timer.stats[('parent', 'child')][1] / n
    parent_excl = timer.stats[('parent',)][2] / n
    a = max(child_incl - bare, 0.)
    b = max(parent_excl - bare, 0.)
    return a, b


def format_tree(rep: Dict, min_share: float = 0.0) -> str:
    """Indented text tree: inclusive, exclusive, calls per step and share."""
    total = rep['total_us'] or 1.
    lines = [f'{"scope":<44} {"incl us":>9} {"excl us":>9} {"calls":>7} {"excl %":>7}']
    for it in rep['items']:
        share = it['excl_us'] / total * 100
        if share < min_share:
            continue
        parts = it['path'].split('/')
        label = '  ' * (len(parts) - 1) + parts[-1]
        lines.append(f'{label:<44} {it["incl_us"]:>9.1f} {it["excl_us"]:>9.1f}'
                     f' {it["calls_per_step"]:>7.2f} {share:>6.1f}%')
    return '\n'.join(lines)


def evals_per_step(rep: Dict) -> Optional[float]:
    """Model evaluations requested by the integrator, per integrator call."""
    integ = [i for i in rep['items'] if i['path'].endswith('/integrator')]
    if not integ:
        return None
    n_integ = sum(i['calls_per_step'] for i in integ)
    evals = sum(i['calls_per_step'] for i in rep['items']
                if i['path'].endswith('/integrator/model'))
    return evals / n_integ if n_integ else None


def selftest() -> None:
    """Check ``ScopeTimer`` on a synthetic workload built from ``time.sleep``.

    Wraps ``Obj.a`` (sleeps 2 ms, calls ``Obj.b`` twice) and ``Obj.b`` (sleeps 1 ms),
    runs ``a`` five times under a root scope, then asserts that:

    * ``wrap`` skips a missing attribute, records it in ``skipped``, and
      ``restore`` removes the instance patches;
    * ``inclusive = exclusive + sum(children's inclusive)`` for every scope;
    * the exclusive times sum to the root's inclusive time;
    * call counts and timings match the workload (B is called twice per A);
    * ``calibrate()`` returns plausible overhead constants;
    * the same scope name under different parents gets different paths.

    Raises ``AssertionError`` on failure and prints a summary on success.
    """
    sleep = lambda s: time.sleep(s)

    class Obj:
        def a(self):
            sleep(0.002)
            self.b()
            self.b()

        def b(self):
            sleep(0.001)

    o = Obj()
    t = ScopeTimer()
    assert t.wrap(o, 'a', 'A') and t.wrap(o, 'b', 'B')
    assert not t.wrap(o, 'missing', 'X')
    assert t.skipped == ['Obj.missing'], t.skipped
    t.enter('root')
    for _ in range(5):
        o.a()
    t.exit()
    t.restore()
    assert 'a' not in vars(o) and 'b' not in vars(o), 'instance patches not removed'
    rep = t.report(steps=5)
    by = {i['path']: i for i in rep['items']}
    # inclusive = exclusive + children, exactly
    for p, i in by.items():
        kids = [j for q, j in by.items() if q.count('/') == p.count('/') + 1
                and q.startswith(p + '/')]
        assert abs(i['incl_us'] - (i['excl_us'] + sum(k['incl_us'] for k in kids))) < 1e-6, p
    # sum of exclusive = root inclusive
    assert abs(sum(i['excl_us'] for i in rep['items']) - rep['total_us']) < 1e-6
    # B is called twice per A, sleeps ~1 ms each
    assert abs(by['root/A/B']['calls_per_step'] - 2) < 1e-9
    assert 1800 < by['root/A/B']['incl_us'] < 3500, by['root/A/B']
    assert 1500 < by['root/A']['excl_us'] < 3500, by['root/A']
    a, b = calibrate(20_000)
    assert 0 <= a < 5e3 and 0 <= b < 5e3, (a, b)
    # the same code twice at different depths gives separate paths
    t2 = ScopeTimer()
    t2.enter('r'); t2.enter('m'); t2.exit(); t2.enter('i'); t2.enter('m'); t2.exit(); t2.exit(); t2.exit()
    assert {'r/m', 'r/i/m'} <= {'/'.join(p) for p in t2.stats}
    print(f'scopes selftest ok (wrapper overhead: a={a:.0f} ns, b={b:.0f} ns)')


if __name__ == '__main__':
    selftest()
