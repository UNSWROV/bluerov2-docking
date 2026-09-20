"""Navigation-error model: the shared implementation lives in the perception package
(perception.utils.nav_error) so the injector node and the offline replays use one
generator. This module re-exports it for the analysis package."""
from __future__ import annotations

import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "perception")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
from perception.utils.nav_error import (  # noqa: E402,F401
    LEVELS,
    GaussMarkovBias,
    NavErrorParams,
    corrupt,
    gauss_markov_bias,
)
