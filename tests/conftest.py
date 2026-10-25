"""
Test configuration.

The frozen oracle in ``tests/legacy`` rounds the vehicle state to 4 decimals
after each step, and the state derivative to a dt-dependent number of decimals.
The current implementation does neither, so every test compares against the
oracle with rounding disabled.
"""
import numpy as np
import pytest

import legacy.simulation



class NumpyWithoutRounding:
    """numpy, except that ``around`` returns its input unchanged."""

    def __getattr__(self, name):
        return getattr(np, name)


    @staticmethod
    def around(a, decimals=0):
        return a



@pytest.fixture(autouse=True)
def oracle_without_rounding(monkeypatch):
    monkeypatch.setattr(legacy.simulation, 'np', NumpyWithoutRounding())
