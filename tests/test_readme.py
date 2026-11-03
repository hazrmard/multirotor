"""
The example in the README must run and fly the path it describes.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

README = Path(__file__).parent.parent / 'README.md'
SEARCH_TRIALS = 'ntrials=100'


def test_readme_example_flies_its_waypoints():
    source = '\n'.join(re.findall(r'```python\n(.*?)```', README.read_text(), flags=re.DOTALL))
    assert SEARCH_TRIALS in source, 'The README example should call optimize().'
    # A short search keeps the test fast. The test checks the flight, not the
    # quality of the search.
    source = source.replace(SEARCH_TRIALS, 'ntrials=10')
    namespace = {}
    exec(compile(source, str(README), 'exec'), namespace)
    # The loop breaks at 6000 steps if a waypoint is not reached.
    assert namespace['step'] < 6000
    np.testing.assert_allclose(namespace['vehicle'].position, [0, 0, 2], atol=0.6)
