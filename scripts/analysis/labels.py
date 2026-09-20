"""Trial label parsing shared by the sweep runner and the analysis.

A bag directory is named
    <tag>_<arm>_<dock>_p<period>_ph<phase>[_nav<level>]_<YYYYMMDD>_<HHMMSS>
where arm is the estimator/controller arm (the 2026-07 sweeps used the filter
regime names "static" and "sway", newer sweeps use A, B, C, D), dock is "sway"
or "static", period is the dock sway period in seconds, phase the start phase
in radians, and the optional nav level names the injected navigation-error level.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

_RE = re.compile(
    r"^(?P<tag>.+?)_(?P<arm>[A-Za-z]+)_(?P<dock>sway|static)_p(?P<period>[0-9.]+)"
    r"_ph(?P<phase>[0-9.]+)(?:_nav(?P<nav>[A-Za-z0-9.]+))?_(?P<stamp>\d{8}_\d{6})$"
)

# Display names for the arms, old regime names included.
ARM_NAMES = {
    "static": "A reactive",
    "sway": "B CV + feedforward",
    "A": "A reactive",
    "B": "B CV + feedforward",
    "C": "C fix",
    "D": "D oracle",
}


@dataclass(frozen=True)
class TrialLabel:
    tag: str
    arm: str
    dock: str
    period: float
    phase: float
    nav: str | None
    stamp: str

    @property
    def cell(self) -> str:
        """Label without the timestamp, as the sweep CSV records it."""
        nav = f"_nav{self.nav}" if self.nav else ""
        return f"{self.tag}_{self.arm}_{self.dock}_p{self.period:g}_ph{self.phase}{nav}"

    @property
    def arm_name(self) -> str:
        return ARM_NAMES.get(self.arm, self.arm)


def parse_label(name: str) -> TrialLabel | None:
    """Parse a bag directory name or path; None if it is not a trial bag."""
    m = _RE.match(os.path.basename(os.path.normpath(name)))
    if not m:
        return None
    return TrialLabel(
        tag=m["tag"], arm=m["arm"], dock=m["dock"], period=float(m["period"]),
        phase=float(m["phase"]), nav=m["nav"], stamp=m["stamp"],
    )
