from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # With the joystick deadman (use_deadman:=true), autonomy publishes to
    # /cmd_vel_auto and is relayed to /cmd_vel only while the deadman button is
    # held. Without it, /cmd_vel_auto is unused and the controllers write /cmd_vel
    # directly. Note: autonomy requires use_deadman:=true to publish
    # /docking/engaged; without that the FSM stays in IDLE and docking never
    # starts regardless of use_control.
    cmd_vel_topic = PythonExpression(
        [
            "'/cmd_vel_auto' if '",
            LaunchConfiguration("use_deadman"),
            "' == 'true' else '/cmd_vel'",
        ]
    )
    # Feedforward source (#68): the oracle node runs and the controllers read its
    # twist only when feedforward_source:=oracle (diagnostic arm D).
    use_oracle = PythonExpression(
        ["'", LaunchConfiguration("feedforward_source"), "' == 'oracle'"]
    )
    dock_velocity_topic = PythonExpression(
        [
            "'/dock/oracle_velocity' if '",
            LaunchConfiguration("feedforward_source"),
            "' == 'oracle' else '/perception/dock_pose_filtered/velocity'",
        ]
    )
    # Navigation error (#69): when a level is set the injector runs and the TF relay
    # reads its output instead of the ground-truth odometry.
    nav_error_on = PythonExpression(
        ["'", LaunchConfiguration("nav_error_level"), "' != 'none'"]
    )
    robot_odom_topic = PythonExpression(
        [
            "'/nav/odometry' if '",
            LaunchConfiguration("nav_error_level"),
            "' != 'none' else '/model/bluerov2_heavy/odometry'",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_docking_rviz", default_value="false"),
            # Web FSM visualizer at http://localhost:<fsm_viewer_port>. Container
            # uses --network=host, so no port forwarding is needed. Only shows
            # data when use_control is on (the FSM publishes /fsm_viewer).
            DeclareLaunchArgument("use_fsm_viewer", default_value="false"),
            DeclareLaunchArgument("fsm_viewer_port", default_value="5000"),
            DeclareLaunchArgument("use_deadman", default_value="false"),
            DeclareLaunchArgument("use_joy", default_value="false"),
            # Optional joystick pinning; see blue_sim (joy_use_dev/joy_dev).
            DeclareLaunchArgument("joy_use_dev", default_value="false"),
            DeclareLaunchArgument("joy_dev", default_value="/dev/input/js0"),
            DeclareLaunchArgument("use_key", default_value="false"),
            DeclareLaunchArgument("use_ardusub", default_value="true"),
            # POSHOLD at idle: holds the armed heading, so the vehicle stays put
            # at startup. ALT_HOLD leaves heading free and the vehicle settles
            # onto the autopilot heading reference (~90 deg yaw drift). The docking
            # FSM commands ALT_HOLD on COARSE entry, so the controllers still get
            # the horizontal authority they need once docking actually starts.
            DeclareLaunchArgument("flight_mode", default_value="POSHOLD"),
            DeclareLaunchArgument("use_mock_led", default_value="true"),
            DeclareLaunchArgument("use_aruco", default_value="true"),
            DeclareLaunchArgument("use_foxglove", default_value="false"),
            DeclareLaunchArgument("use_control", default_value="false"),
            # Filter regime, forwarded to aruco.launch.py. "sway" = CV filter + ff
            # (method); "static" = velocity pinned, ff off (baseline). For ablation.
            DeclareLaunchArgument("process_noise_regime", default_value="sway"),
            # the fix arm (C): velocity-closed feedforward in the controllers and a
            # bounded velocity state in the filter; defaults are the original method
            DeclareLaunchArgument("feedforward_mode", default_value="open_loop"),
            DeclareLaunchArgument("stale_velocity_hold_s", default_value="0.0"),
            DeclareLaunchArgument("stale_velocity_decay_s", default_value="1.0"),
            # Sway-regime WNA density sigma_a; forwarded to aruco.launch.py (#36 trade study).
            DeclareLaunchArgument("sway_sigma_a", default_value="0.16"),
            # Feedforward source: "filter" (the CV filter's velocity state) or "oracle"
            # (ground-truth dock velocity, diagnostic arm D, #68).
            DeclareLaunchArgument("feedforward_source", default_value="filter"),
            # Navigation-error level injected into the odometry every node sees
            # (none | low | medium | high, #69). Ground truth stays on the original topic.
            DeclareLaunchArgument("nav_error_level", default_value="none"),
            DeclareLaunchArgument("nav_error_seed", default_value="0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("blue_sim"), "launch/sim.launch.py"]
                    )
                ),
                launch_arguments={
                    "use_sim": "true",
                    "use_rviz": "false",
                    "use_joy": LaunchConfiguration("use_joy"),
                    "joy_use_dev": LaunchConfiguration("joy_use_dev"),
                    "joy_dev": LaunchConfiguration("joy_dev"),
                    "use_key": LaunchConfiguration("use_key"),
                    "model": "bluerov2_heavy",
                    "use_ardusub": LaunchConfiguration("use_ardusub"),
                    "flight_mode": LaunchConfiguration("flight_mode"),
                    "gazebo_world_file": [
                        PathJoinSubstitution(
                            [FindPackageShare("description"), "worlds"]
                        ),
                        "/ocean.world",
                    ],
                }.items(),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=[
                    "-d",
                    PathJoinSubstitution(
                        [FindPackageShare("description"), "rviz/docking_sim.rviz"]
                    ),
                ],
                condition=IfCondition(LaunchConfiguration("use_docking_rviz")),
                output="screen",
            ),
            # Needs to add this camera info bridge since 3rd party ardusub_driver only bridges /camera/image_raw
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=[
                    "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"
                ],
                output="screen",
            ),
            # Ground-truth dock pose: the dock's PosePublisher emits its true world
            # pose on the gz topic /model/docking_station/pose. Bridge it to ROS as
            # /dock/ground_truth/pose so docking trials can score the filter estimate
            # against ground truth (issue #36). gz -> ROS only (the sim owns the truth).
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=[
                    "/model/docking_station/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose"
                ],
                remappings=[
                    ("/model/docking_station/pose", "/dock/ground_truth/pose"),
                ],
                output="screen",
            ),
            Node(
                package="perception",
                executable="nav_error_injector",
                name="nav_error_injector",
                parameters=[{
                    "use_sim_time": True,
                    "level": LaunchConfiguration("nav_error_level"),
                    "seed": LaunchConfiguration("nav_error_seed"),
                }],
                condition=IfCondition(nav_error_on),
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("perception"), "launch/led_mock.launch.py"]
                    )
                ),
                launch_arguments={"robot_odom_topic": robot_odom_topic}.items(),
                condition=IfCondition(LaunchConfiguration("use_mock_led")),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("perception"), "launch/aruco.launch.py"]
                    )
                ),
                launch_arguments={
                    "target_frame": "map",
                    "process_noise_regime": LaunchConfiguration("process_noise_regime"),
                    "sway_sigma_a": LaunchConfiguration("sway_sigma_a"),
                    "stale_velocity_hold_s": LaunchConfiguration("stale_velocity_hold_s"),
                    "stale_velocity_decay_s": LaunchConfiguration("stale_velocity_decay_s"),
                }.items(),
                condition=IfCondition(LaunchConfiguration("use_aruco")),
            ),
            # Foxglove bridge: WebSocket server (ws://localhost:8765) for the
            # Foxglove/Lichtblick viewer. Open description/foxglove/docking.json.
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                parameters=[
                    {
                        "use_sim_time": True,
                        "port": 8765,
                        "asset_uri_allowlist": ["^package://.*", "^file://.*"],
                    }
                ],
                condition=IfCondition(LaunchConfiguration("use_foxglove")),
                output="screen",
            ),
            # Oracle dock velocity (arm D): differentiates the ground-truth dock pose
            # and feeds the controllers through the same twist interface.
            Node(
                package="perception",
                executable="oracle_dock_velocity",
                name="oracle_dock_velocity",
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(use_oracle),
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("control"),
                            "launch/coarse_approach.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "target_frame": "map",
                    "cmd_vel_topic": cmd_vel_topic,
                    "dock_velocity_topic": dock_velocity_topic,
                    "feedforward_mode": LaunchConfiguration("feedforward_mode"),
                    "robot_odom_topic": robot_odom_topic,
                }.items(),
                condition=IfCondition(LaunchConfiguration("use_control")),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("control"), "launch/fine_align.launch.py"]
                    )
                ),
                launch_arguments={
                    "target_frame": "map",
                    "cmd_vel_topic": cmd_vel_topic,
                    "dock_velocity_topic": dock_velocity_topic,
                    "feedforward_mode": LaunchConfiguration("feedforward_mode"),
                    "robot_odom_topic": robot_odom_topic,
                }.items(),
                condition=IfCondition(LaunchConfiguration("use_control")),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("orchestrator"), "launch/docking_fsm.launch.py"]
                    )
                ),
                condition=IfCondition(LaunchConfiguration("use_control")),
            ),
            # Joystick deadman: relays /cmd_vel_auto -> /cmd_vel only while the
            # deadman button (RB) is held. Requires use_joy for a /joy source.
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("orchestrator"),
                            "launch/autonomy_deadman.launch.py",
                        ]
                    )
                ),
                condition=IfCondition(LaunchConfiguration("use_deadman")),
            ),
            # YASMIN web FSM viewer (serves http://localhost:5000). The FSM node
            # publishes /fsm_viewer; this node renders it.
            Node(
                package="yasmin_viewer",
                executable="yasmin_viewer_node",
                name="yasmin_viewer",
                parameters=[
                    {
                        "port": ParameterValue(
                            LaunchConfiguration("fsm_viewer_port"), value_type=int
                        )
                    }
                ],
                condition=IfCondition(LaunchConfiguration("use_fsm_viewer")),
                output="screen",
            ),
        ]
    )
