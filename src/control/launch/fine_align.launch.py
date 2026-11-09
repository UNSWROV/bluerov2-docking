from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("target_frame", default_value="map"),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
            # Dock-velocity source for the feedforward: the filter's velocity state by
            # default, or the ground-truth oracle (/dock/oracle_velocity) for arm D (#68).
            DeclareLaunchArgument(
                "dock_velocity_topic",
                default_value="/perception/dock_pose_filtered/velocity",
            ),
            # open_loop (arms A, B, D) or velocity_loop (the fix arm, C)
            DeclareLaunchArgument("feedforward_mode", default_value="open_loop"),
            # navigation odometry the velocity loop closes on (the injector's
            # /nav/odometry under navigation error)
            DeclareLaunchArgument(
                "robot_odom_topic", default_value="/model/bluerov2_heavy/odometry"
            ),
            # open-loop feedforward gain override; negative keeps the controller's
            # own gain (the rescaled open-loop arm E passes 1 / plant gain)
            DeclareLaunchArgument("ff_gain_override", default_value="-1.0"),
            Node(
                package="control",
                executable="fine_align_node",
                name="fine_align",
                parameters=[
                    PathJoinSubstitution(
                        [FindPackageShare("control"), "config", "fine_pbvs.yaml"]
                    ),
                    {
                        "target_frame": LaunchConfiguration("target_frame"),
                        "feedforward_mode": LaunchConfiguration("feedforward_mode"),
                        "robot_odom_topic": LaunchConfiguration("robot_odom_topic"),
                        "ff_gain_override": LaunchConfiguration("ff_gain_override"),
                    },
                ],
                remappings=[
                    ("/cmd_vel", LaunchConfiguration("cmd_vel_topic")),
                    (
                        "/perception/dock_pose_filtered/velocity",
                        LaunchConfiguration("dock_velocity_topic"),
                    ),
                ],
                output="screen",
            ),
        ]
    )
