"""Shared helpers: environment metadata, statistics, and result-file IO."""
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS_DIR = HERE / 'results'
SCHEMA_VERSION = 1


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ['git', *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return ''


def _cpu_model() -> str:
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment_metadata() -> Dict:
    """Facts needed to judge whether two result files are comparable."""
    import numba
    import numpy
    import scipy
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        git_sha=_git('rev-parse', 'HEAD'),
        git_branch=_git('rev-parse', '--abbrev-ref', 'HEAD'),
        git_dirty=bool(_git('status', '--porcelain', '--untracked-files=no')),
        python=platform.python_version(),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
        platform=platform.platform(),
        cpu=_cpu_model(),
        cpu_count=os.cpu_count(),
        executable=sys.executable,
    )


def summarize(samples: List[float]) -> Dict[str, float]:
    """Robust summary of repeated measurements (any unit)."""
    s = sorted(samples)
    n = len(s)
    if n >= 4:
        q = statistics.quantiles(s, n=4, method='inclusive')
        iqr = q[2] - q[0]
    else:
        iqr = s[-1] - s[0]
    return dict(
        min=s[0], median=statistics.median(s), mean=statistics.fmean(s),
        max=s[-1], iqr=iqr, n=n,
    )


def save_results(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_results(path) -> Dict:
    with open(path) as f:
        data = json.load(f)
    if data.get('schema') != SCHEMA_VERSION:
        raise ValueError(f'{path}: unsupported schema {data.get("schema")!r}')
    return data


def index_by_id(data: Dict) -> Dict[str, Dict]:
    return {r['id']: r for r in data['results']}
