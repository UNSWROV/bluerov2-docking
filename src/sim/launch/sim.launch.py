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
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("perception"), "launch/led_mock.launch.py"]
                    )
                ),
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
