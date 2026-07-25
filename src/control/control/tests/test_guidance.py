"""Black-box tests for the pure coarse-approach guidance math."""

import math

import numpy as np

from control.guidance import (
    compute_guidance,
    standoff_point_in_target,
    standoff_pose_in_target,
)


def test_world_to_body_at_dock_approach_yaw():
    """At the ocean.world approach the ROV faces yaw ~270 deg (world -Y, toward the
    dock front). A dock velocity along world -Y is then straight AHEAD of the
    vehicle: body (forward, left, up) = (+v, 0, 0)."""
    from scipy.spatial.transform import Rotation as R

    from control.guidance import world_to_body

    rov_quat = R.from_euler("z", 3.0 * math.pi / 2.0).as_quat()  # (x, y, z, w)
    v_world = np.array([0.0, -0.126, 0.0])
    v_body = world_to_body(v_world, rov_quat)
    np.testing.assert_allclose(v_body, [0.126, 0.0, 0.0], atol=1e-9)


def test_world_to_body_identity_at_zero_yaw():
    from scipy.spatial.transform import Rotation as R

    from control.guidance import world_to_body

    quat = R.from_euler("z", 0.0).as_quat()
    v = np.array([0.1, -0.2, 0.05])
    np.testing.assert_allclose(world_to_body(v, quat), v, atol=1e-12)

AIM_OFFSET = (0.0, 0.310, 0.042)  # 401/402 midpoint in dock frame


def test_standoff_point_identity_dock():
    # Dock at origin, identity orientation. Aim point = aim offset; standoff
    # sits standoff_distance in front along dock -Y.
    pt = standoff_point_in_target(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    assert np.allclose(pt, [0.0, -0.690, 0.042], atol=1e-6)


def test_standoff_point_yawed_dock():
    # Dock yawed +90deg about Z: dock +Y now points to world -X, dock -Y -> world +X.
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    pt = standoff_point_in_target(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=q,
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    # aim = R*(0,0.310,0.042) = (-0.310, 0, 0.042); standoff = aim + R*(0,-1,0) = aim + (1,0,0)
    assert np.allclose(pt, [0.690, 0.0, 0.042], atol=1e-6)


def test_rov_at_origin_facing_x():
    # Dock+ROV at origin, both identity. Standoff is at world (0,-0.690,0.042).
    # ROV faces world +X, so the standoff is to its right (-Y=right) and slightly up.
    g = compute_guidance(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        rov_pos=(0.0, 0.0, 0.0),
        rov_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    # rel_pos_body = R_rov^-1 (standoff - rov) = standoff (identity) = [fwd, left, up]
    assert np.allclose(g.rel_pos_body, [0.0, -0.690, 0.042], atol=1e-6)
    # boresight = dock +Y = world +Y; in body that is +left -> yaw_err = +pi/2
    assert math.isclose(g.yaw_err, math.pi / 2, abs_tol=1e-6)


def test_on_axis_at_standoff_is_zeroed():
    # ROV sitting AT the standoff point, facing dock +Y (turned +90deg about Z).
    standoff = (0.0, -0.690, 0.042)
    rov_q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # +X -> +Y
    g = compute_guidance(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        rov_pos=standoff,
        rov_quat_xyzw=rov_q,
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    assert np.allclose(g.rel_pos_body, [0.0, 0.0, 0.0], atol=1e-6)
    assert math.isclose(g.yaw_err, 0.0, abs_tol=1e-6)
    assert math.isclose(g.range_to_standoff_m, 0.0, abs_tol=1e-6)
    assert math.isclose(g.axis_offset_m, 0.0, abs_tol=1e-6)


def test_axis_offset_detects_off_boresight():
    # ROV offset +0.5 in world X from the boresight line (which runs along world Y
    # through x=0, z=0.042 for an identity dock). Perp distance should be 0.5.
    g = compute_guidance(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        rov_pos=(0.5, -0.690, 0.042),
        rov_quat_xyzw=(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    assert math.isclose(g.axis_offset_m, 0.5, abs_tol=1e-6)


def test_standoff_pose_identity_dock_faces_boresight():
    pos, quat = standoff_pose_in_target(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    assert np.allclose(pos, [0.0, -0.690, 0.042], atol=1e-6)
    # desired heading = +90deg about Z (body +X -> dock +Y) -> (0,0,sin45,cos45)
    assert np.allclose(
        quat, [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)], atol=1e-6
    )


def test_range_to_dock():
    g = compute_guidance(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        rov_pos=(0.0, 0.0, 0.0),
        rov_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    # aim = (0, 0.310, 0.042); ROV at origin -> distance = hypot(0.310, 0.042)
    assert math.isclose(g.range_to_dock_m, (0.310**2 + 0.042**2) ** 0.5, abs_tol=1e-6)


def test_heading_faces_the_dock():
    # Heading points the nose at the dock (aim point), not the standoff point.
    # ROV at the standoff facing the dock -> yaw_err 0; ROV rotated away -> nonzero
    # toward the dock. Here ROV is left of the dock so it must yaw to face it.
    g = compute_guidance(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        rov_pos=(0.5, -0.690, 0.042),  # off to the +x side, near standoff depth
        rov_quat_xyzw=(0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)),
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=1.0,
    )
    # aim=(0,0.310,0.042); aim-rov=(-0.5,1.0,0); body(+90 about z): fwd=+1.0, left=+0.5
    # -> yaw_err = atan2(0.5, 1.0) > 0 (turn left to face the dock)
    assert g.yaw_err > 0.0
    assert math.isclose(g.yaw_err, math.atan2(0.5, 1.0), abs_tol=1e-6)


def test_yaw_boresight_well_conditioned_near_aim():
    # Dock identity at origin; entry axis (boresight) = world +Y. ROV faces +Y
    # (aligned with the boresight) sitting ~3.6 cm from the aim with a small side
    # offset. "Face aim" yaw is singular here (blows up); boresight yaw stays ~0.
    dock = dict(
        dock_pos=(0.0, 0.0, 0.0),
        dock_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        aim_offset_in_dock=AIM_OFFSET,
        standoff_distance_m=0.0,
    )
    qz, qw = math.sin(math.pi / 4), math.cos(math.pi / 4)  # face world +Y
    rov = dict(rov_pos=(0.02, 0.28, 0.042), rov_quat_xyzw=(0.0, 0.0, qz, qw))

    face_aim = compute_guidance(**dock, **rov, yaw_to_boresight=False)
    boresight = compute_guidance(**dock, **rov, yaw_to_boresight=True)

    assert face_aim.range_to_dock_m < 0.05  # genuinely on top of the aim point
    assert abs(face_aim.yaw_err) > math.radians(20)  # singular: large/bogus angle
    assert abs(boresight.yaw_err) < math.radians(1)  # stable: ~0 (truly aligned)
