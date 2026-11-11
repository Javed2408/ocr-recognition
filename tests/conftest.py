import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def needs(path):
    """Skip marker for tests that need trained weights or downloaded data."""
    return pytest.mark.skipif(not os.path.exists(path), reason=f"missing {os.path.relpath(path, ROOT)}")


@pytest.fixture(scope="session")
def genuine_sample():
    from cheque_forensics.synth.tampering import make_sample
    return make_sample(3, "test", tamper_prob=0.0, seed=501, with_original=True)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
