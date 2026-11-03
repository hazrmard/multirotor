"""
The example in the README must run and fly the path it describes.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np

README = Path(__file__).parent.parent / 'README.md'


def test_readme_example_flies_its_waypoints():
    blocks = re.findall(r'```python\n(.*?)```', README.read_text(), flags=re.DOTALL)
    assert len(blocks) == 1, 'The README should have one python example.'
    namespace = {}
    exec(compile(blocks[0], str(README), 'exec'), namespace)
    # The loop breaks at 6000 steps if a waypoint is not reached.
    assert namespace['step'] < 6000
    np.testing.assert_allclose(namespace['vehicle'].position, [0, 0, 2], atol=0.6)
