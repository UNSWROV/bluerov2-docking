from analysis.labels import parse_label


def test_parses_legacy_regime_label():
    lab = parse_label("clamped_sway_sway_p8_ph2.094_20260725_064421")
    assert lab is not None
    assert (lab.tag, lab.arm, lab.dock) == ("clamped", "sway", "sway")
    assert lab.period == 8.0 and lab.phase == 2.094 and lab.nav is None
    assert lab.arm_name == "B CV + feedforward"
    assert lab.cell == "clamped_sway_sway_p8_ph2.094"


def test_parses_static_dock_and_nav_level():
    lab = parse_label("/x/y/ut27_C_static_p0_ph0.0_navhigh_20261025_010203/")
    assert lab is not None
    assert lab.arm == "C" and lab.dock == "static" and lab.period == 0.0
    assert lab.nav == "high" and lab.arm_name == "C fix"


def test_rejects_non_trial_names():
    assert parse_label("clamped_results.csv") is None
    assert parse_label("auto_sway_p18_ph0.0_20260723_102057") is None
