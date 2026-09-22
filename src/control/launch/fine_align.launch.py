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
            # open_loop (arms A, B, D) or velocity_loop (the fix arm, C)
            DeclareLaunchArgument("feedforward_mode", default_value="open_loop"),
            # navigation odometry the velocity loop closes on (the injector's
            # /nav/odometry under navigation error)
            DeclareLaunchArgument(
                "robot_odom_topic", default_value="/model/bluerov2_heavy/odometry"
            ),
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
                    },
                ],
                remappings=[("/cmd_vel", LaunchConfiguration("cmd_vel_topic"))],
                output="screen",
            ),
        ]
    )
