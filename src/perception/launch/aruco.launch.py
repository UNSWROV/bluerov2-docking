from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument, TimerAction


REFERENCE_MARKER_SIZE = 0.1  # m

# Physical sizes are total print dimensions. aruco_ros measures the black square
# only, so each size is multiplied by the black/total pixel ratio (1470/1889 =
# 0.7782).
MARKER_SIZES = [
    "201:0.15564",  # 200mm total x 0.7782
    "202:0.15564",  # 200mm total x 0.7782
    "301:0.07782",  # 100mm total x 0.7782
    "302:0.04669",  # 60mm total x 0.7782
    "303:0.04669",  # 60mm total x 0.7782
    "304:0.04669",  # 60mm total x 0.7782
    "305:0.04669",  # 60mm total x 0.7782
    "401:0.03696",  # 47.5mm total x 0.7782
    "402:0.03696",  # 47.5mm total x 0.7782
]

# Noise kernel scale (Candidate B: sigma_pos = alpha * r^2 / s, sigma_rot = alpha * r / s).
# Same alpha applies to both position and rotation (same underlying pixel-noise
# physics) but they can be tuned independently if needed.
# alpha=0.001 -> ~1cm pos std and ~0.57deg rot std at 1m for a 100mm marker.
NOISE_SCALE_ALPHA = 0.001
NOISE_SCALE_ALPHA_ROT = 0.001

# Health thresholds. See lib/health.py for trade-off notes.
HEALTHY_MAX_AGE_S = 0.5
# Moving dock: the sway process-noise regime floors pos_std near 0.033 m (measured
# vs ground truth, tracking error ~4.5 cm), so the static-calibrated 0.02 blocked the
# coarse->fine handoff. 0.04 matches the actual estimation accuracy (handoff window ~5 cm).
HEALTHY_MAX_POSITION_STD_M = 0.04
STALE_MAX_AGE_S = 3.0

DOCK_POSE_FILTER_LAUNCH_DELAY_S = 5.0


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            DeclareLaunchArgument("target_frame", default_value="odom"),
            # Filter process-noise regime: "sway" (9-state CV, estimates dock velocity,
            # feeds the feedforward) or "static" (velocity pinned to 0 -> ff off, the
            # pre-#36 baseline). An argument so a batch sweep can ablate the method.
            DeclareLaunchArgument("process_noise_regime", default_value="sway"),
            # Sway-regime WNA density sigma_a (m/s^2); velocity-state gain knob (#36).
            DeclareLaunchArgument("sway_sigma_a", default_value="0.16"),
            # bounded velocity state during blackout (fix arm); hold 0 disables
            DeclareLaunchArgument("stale_velocity_hold_s", default_value="0.0"),
            DeclareLaunchArgument("stale_velocity_decay_s", default_value="1.0"),
            # Default True: aruco.launch.py is almost always invoked via
            # sim.launch.py during development. Override to False for real-
            # hardware runs where /clock isn't published.
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(
                package="aruco_ros",
                executable="marker_publisher",
                name="arucos",
                remappings=[
                    ("image", "/camera/image_raw"),
                    ("camera_info", "/camera/camera_info"),
                ],
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "image_is_rectified": True,
                        "reference_frame": "camera_link",
                        "camera_frame": "camera_link",
                        "marker_size": REFERENCE_MARKER_SIZE,
                    }
                ],
                output="screen",
            ),
            Node(
                package="perception",
                executable="aruco_relay",
                name="aruco_relay",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "reference_marker_size_m": REFERENCE_MARKER_SIZE,
                        "marker_sizes_m": MARKER_SIZES,
                    }
                ],
                output="screen",
            ),
            Node(
                package="perception",
                executable="aruco_fusion",
                name="aruco_fusion",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "noise_scale_alpha_pos": NOISE_SCALE_ALPHA,
                        "noise_scale_alpha_rot": NOISE_SCALE_ALPHA_ROT,
                        "consensus_threshold_deg": 8.0,
                        "min_markers_for_consensus": 3,
                    }
                ],
                output="screen",
            ),
            TimerAction(
                period=DOCK_POSE_FILTER_LAUNCH_DELAY_S,
                actions=[
                    Node(
                        package="perception",
                        executable="dock_pose_filter",
                        name="dock_pose_filter",
                        parameters=[
                            {
                                "use_sim_time": use_sim_time,
                                "predict_rate_hz": 30.0,
                                "process_noise_regime": LaunchConfiguration(
                                    "process_noise_regime"
                                ),
                                "sway_sigma_a": ParameterValue(
                                    LaunchConfiguration("sway_sigma_a"),
                                    value_type=float,
                                ),
                                "stale_velocity_hold_s": ParameterValue(
                                    LaunchConfiguration("stale_velocity_hold_s"),
                                    value_type=float,
                                ),
                                "stale_velocity_decay_s": ParameterValue(
                                    LaunchConfiguration("stale_velocity_decay_s"),
                                    value_type=float,
                                ),
                                "min_markers_for_init": 2,
                                "healthy_max_age_s": HEALTHY_MAX_AGE_S,
                                "healthy_max_position_std_m": HEALTHY_MAX_POSITION_STD_M,
                                "stale_max_age_s": STALE_MAX_AGE_S,
                                "target_frame": LaunchConfiguration("target_frame"),
                            }
                        ],
                        output="screen",
                    ),
                    Node(
                        package="perception",
                        executable="dock_visualizer",
                        name="dock_visualizer",
                        parameters=[
                            {
                                "use_sim_time": use_sim_time,
                            }
                        ],
                        output="screen",
                    ),
                ],
            ),
        ]
    )
